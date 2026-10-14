"""A1 molecular cross-validation and conditional A2 fixed-mask correction.

Frozen constants live in core_scores. A1 is never registered as a mask endpoint.
All outputs and checkpoints belong to this run; historical files are read-only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import resource
import time
from pathlib import Path
import cv2
import h5py
import numpy as np
import pandas as pd
import tifffile
import yaml
from scipy import sparse
from scipy.spatial import cKDTree
from . import common
from .contracts import A2_ENDPOINTS
from .core_scores import (PARAMETERS,SHUFFLE_SEEDS,reference_arrays,legal_types,
                          molecule_fold,matched_donors,profile,cross_scores,paired_bootstrap,decide)
from .core_refine import (NuclearSupport,atomic_npy,atomic_npz,interface_band,fixed_domain,
                          score_proposals,apply_proposals)

LIMITATIONS=[
    'Rep1 is the development specimen; molecular splitting is not independent tissue validation.',
    'Historical nuclear types may reflect all nuclear RNA: this is conditional information given those labels.',
    'Bootstrap intervals are conditional on the fixed split and 20 permutations, not randomization p values.',
    'Nuclear held-out candidate diagnostics do not measure true extracellular owner accuracy.',
    'Complete-nucleus support differs from decoder view candidate visibility.',
    'Core/cytoplasmic distribution shift, sparse Core coverage and single specimen/seed remain.',
    'A2 only transfers existing foreground cell to cell; it cannot recover unassigned RNA.'
]


class SafeStop(RuntimeError):
    """At a committed safe boundary; resume retains all exact completed work."""


def _notify(cfg,node,stage,**values):
    common.progress(cfg,node,stage,**values)
    if common.stop_requested(cfg):raise SafeStop('stop requested after committed '+stage)


def _json_safe(value):
    if isinstance(value,dict):return {str(k):_json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_json_safe(v) for v in value]
    if isinstance(value,np.ndarray):return _json_safe(value.tolist())
    if isinstance(value,np.generic):return _json_safe(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def _legacy_identity(obj):
    if common.sha256(obj['path'])!=obj['sha256']:raise ValueError('historical input drift '+obj['path'])
    if 'bytes' in obj and Path(obj['path']).stat().st_size!=obj['bytes']:raise ValueError('historical size drift')


def _inputs(cfg):
    mini=cfg.get('a_core_mini_inputs')
    if mini:
        if 'mini' not in Path(cfg['output_root']).parts:raise ValueError('mini input override outside mini root')
        manifest=common.read(mini);paths=manifest['paths'];machine=common.read(paths['machine'])
        for obj in manifest['outputs'].values():common.check_identity(obj)
        ref=common.read(cfg['inputs']['reference_region']);meta=np.load(paths['metadata'])
        records=manifest['coordinate_records'];source=dict(mini_input=common.identity(mini),mini_provenance=manifest['provenance'])
        expected=manifest['molecules']
    else:
        machine=common.read(cfg['endpoints']['BASE_MATCH']['machine']);ref=common.read(cfg['inputs']['reference_region'])
        meta=np.load(cfg['inputs']['nucleus_metadata']);matrix=common.read(Path(cfg['endpoints']['BASE_MATCH']['matrix'])/'complete.json')
        if matrix.get('validation_kind')!='independent_full_native_resize_lookup_and_conservation':raise ValueError('coordinates lack independent native 2D oracle')
        _legacy_identity(matrix['validation'])
        for key in ('transcripts','genes'):
            _legacy_identity(matrix['dependencies'][key])
            if common.sha256(machine[key])!=matrix['dependencies'][key]['sha256']:raise ValueError('machine/matrix input mismatch '+key)
        cache=common.read(Path(matrix['coordinate_cache'])/'complete.json')
        if cache.get('duplicate_ids')!=0:raise ValueError('historical coordinate IDs not unique')
        cdeps=cache['dependencies']
        if cdeps['scale_x']!=machine['scale_x'] or cdeps['scale_y']!=machine['scale_y'] or cdeps['transcripts']!=matrix['dependencies']['transcripts']['sha256'] or cdeps['genes']!=matrix['dependencies']['genes']['sha256']:raise ValueError('coordinate dependency mismatch')
        records=matrix['receipts'];expected=34442716
        if matrix['totals']['eligible']!=expected:raise ValueError('full eligible molecule denominator drift')
        source=dict(machine=common.identity(cfg['endpoints']['BASE_MATCH']['machine']),
            metadata=common.identity(cfg['inputs']['nucleus_metadata']),reference=common.identity(cfg['inputs']['reference_region']),
            matrix_receipt=common.identity(Path(cfg['endpoints']['BASE_MATCH']['matrix'])/'complete.json'),
            coordinate_validation=matrix['validation'],coordinate_cache=common.identity(Path(matrix['coordinate_cache'])/'complete.json'),
            transcripts=matrix['dependencies']['transcripts'],genes=matrix['dependencies']['genes'])
    nuclei=tifffile.imread(machine['nuclei_mask']);baseline=tifffile.imread(machine['final_mask'])
    if nuclei.shape!=baseline.shape or machine['pixel_um']!=1.0:raise ValueError('native shape/pixel scale drift')
    if not np.issubdtype(nuclei.dtype,np.integer) or np.any(nuclei<0):raise ValueError('nuclear labels invalid')
    ids=meta['ids'].astype(np.int64);types=meta['type_index'].astype(int);genes=Path(machine['genes']).read_text().splitlines()
    gene_indices,theta,valid=reference_arrays(ref,genes)
    if len(genes)!=313:raise ValueError('313 spatial genes required')
    if not mini:
        if len(ids)!=107955:raise ValueError('global nuclear population drift')
        with h5py.File(machine['preannotation']) as f:
            aid=f['ids'][:].astype(np.int64);at=f['data'][:]
        order=np.argsort(aid)
        if not np.array_equal(aid[order],ids) or not np.array_equal(at[order],types):raise ValueError('nuclear annotation ID/type mismatch')
    settings=yaml.safe_load(Path(machine['baseline_config']).read_text())
    support=NuclearSupport(nuclei,ids,types,ref['type_names'],settings['model_params']['elongated'])
    if not np.array_equal(support.area,meta['area']) or not np.allclose(support.centroids,meta['centroid_xy'],rtol=0,atol=1e-8):raise ValueError('metadata/complete-nucleus geometry mismatch')
    source.update(nuclei=common.identity(machine['nuclei_mask']),baseline=common.identity(machine['final_mask']),
                  baseline_config=common.identity(machine['baseline_config']),preannotation=common.identity(machine['preannotation']))
    return dict(machine=machine,reference=ref,ids=ids,types=types,genes=genes,gene_indices=gene_indices,
        theta=theta,valid=valid,nuclei=nuclei,baseline=baseline,support=support,records=records,expected=expected,
        source=source,mini=bool(mini))


def _coordinates(data):
    rows=0
    for i,rec in enumerate(data['records']):
        path=rec['coordinate_file']
        if common.sha256(path)!=rec['coordinate_sha256']:raise ValueError('coordinate record drift')
        arr=np.load(path,mmap_mode='r')
        if len(arr)!=rec['rows'] or rec['row_start']!=rows or rec['row_stop']!=rows+len(arr):raise ValueError('coordinate chunk range gap/overlap')
        if arr.dtype.names!=('transcript_id','x','y','gene'):raise ValueError('coordinate record schema')
        good=(arr['x']>=0)&(arr['y']>=0)
        if np.any(arr['x'][good]>=data['nuclei'].shape[1]) or np.any(arr['y'][good]>=data['nuclei'].shape[0]):raise ValueError('coordinate OOB')
        if np.any(arr['gene']<0) or np.any(arr['gene']>=len(data['genes'])):raise ValueError('coordinate gene index')
        rows+=len(arr);yield i,rec,arr
    if rows!=data['expected']:raise ValueError('coordinate total mismatch')


def _count_core(cfg,data,out,stop_after=None):
    """Per-input-chunk sparse increments; recovery sums each committed range once."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True);core,area=data['support'].core_image()
    n=len(data['ids']);ng=len(data['gene_indices']);lookup=np.full(len(data['genes']),-1,int);lookup[data['gene_indices']]=np.arange(ng)
    deps=dict(inputs=common.digest(data['source']),parameters=common.digest(PARAMETERS),source=common.sha256(__file__),
              score_source=common.sha256(Path(__file__).with_name('core_scores.py')),refine_source=common.sha256(Path(__file__).with_name('core_refine.py')))
    counts=np.zeros((2,n,ng),np.int64);totals=dict(eligible=0,out_of_bounds=0,nuclear=0,core=0,core_G=0,outside_core=0)
    allids=[];chunks=out/'count_chunks';chunks.mkdir(exist_ok=True);new_chunks=0
    for i,rec,a in _coordinates(data):
        receipt=chunks/f'{i:04d}.json';npz=chunks/f'{i:04d}.npz';chunkdeps=dict(**deps,coordinate=rec['coordinate_sha256'],row_start=rec['row_start'],row_stop=rec['row_stop'])
        if receipt.exists():
            saved=common.read(receipt)
            if saved['dependencies']!=chunkdeps:raise ValueError('Core chunk recovery dependencies drift')
            common.check_identity(saved['output']);part=np.load(npz);c=saved['totals']
        else:
            good=(a['x']>=0)&(a['y']>=0);owner=np.zeros(len(a),np.int64);nucowner=owner.copy()
            owner[good]=core[a['y'][good],a['x'][good]];nucowner[good]=data['nuclei'][a['y'][good],a['x'][good]]
            selected=(owner>0)&(lookup[a['gene']]>=0);rows=np.searchsorted(data['ids'],owner[selected])
            if not np.array_equal(data['ids'][rows],owner[selected]):raise ValueError('Core ID lookup mismatch')
            fold=molecule_fold(a['transcript_id'][selected]);g=lookup[a['gene'][selected]]
            key=(fold.astype(np.int64)*n+rows)*ng+g;keys,values=np.unique(key,return_counts=True)
            c=dict(eligible=len(a),out_of_bounds=int((~good).sum()),nuclear=int((nucowner>0).sum()),core=int((owner>0).sum()),core_G=int(selected.sum()),outside_core=int((owner==0).sum()))
            atomic_npz(npz,keys=keys,counts=values,transcript_id=a['transcript_id'][selected],fold=fold,
                row=rows.astype(np.int32),gene=g.astype(np.int16),pixel=a['y'][selected].astype(np.int64)*core.shape[1]+a['x'][selected])
            common.atomic_json(receipt,dict(dependencies=chunkdeps,output=common.identity(npz),totals=c));part=np.load(npz);new_chunks+=1
        counts.ravel()[part['keys']]+=part['counts']
        for k,v in c.items():totals[k]+=v
        allids.append(a['transcript_id'].copy())
        _notify(cfg,'A1_CORE','core_count_chunk',chunk=i+1,total_chunks=len(data['records']),molecules=totals['eligible'])
        if stop_after is not None and new_chunks>=stop_after and i+1<len(data['records']):return None,dict(complete=False,processed_chunks=i+1)
    tids=np.concatenate(allids);tids.sort();duplicates=int(np.count_nonzero(tids[1:]==tids[:-1]));del tids,allids
    if duplicates:raise ValueError('duplicate transcript IDs')
    if totals['eligible']!=totals['core']+totals['outside_core'] or int(counts.sum())!=totals['core_G'] or totals['core']>totals['nuclear']:raise ValueError('Core count conservation')
    atomic_npz(out/'core_counts.npz',counts=counts,ids=data['ids'],gene_indices=data['gene_indices'],core_area=area)
    common.atomic_json(out/'split_manifest.json',dict(algorithm='SHA256(20261015|decimal_numeric_transcript_id), first-byte least-significant bit',
        cross_directions=['F0->F1','F1->F0'],split_disjoint=True,unique_transcript_ids=True,duplicate_ids=duplicates,
        fold_counts=counts.sum((1,2)).tolist(),counts_conserved=True,totals=totals,
        chunk_receipts=[common.identity(chunks/f'{i:04d}.json') for i in range(len(data['records']))],upstream_labels_may_use_full_nuclear_expression=True))
    return counts,dict(complete=True,totals=totals,core_area=area)


def _pixel_expression(data,pixels,core=None):
    """Sparse 313/intersection expression in exact native pixel order."""
    ng=len(data['gene_indices']);lookup=np.full(len(data['genes']),-1,int);lookup[data['gene_indices']]=np.arange(ng)
    mats=[sparse.csr_matrix((len(pixels),ng),dtype=np.int64) for _ in range(3)]
    if not len(pixels):return mats
    w=data['nuclei'].shape[1]
    for _,_,a in _coordinates(data):
        good=(a['x']>=0)&(a['y']>=0)&(lookup[a['gene']]>=0);ix=np.flatnonzero(good)
        flat=a['y'][ix].astype(np.int64)*w+a['x'][ix];p=np.searchsorted(pixels,flat);keep=p<len(pixels)
        ix=ix[keep];flat=flat[keep];p=p[keep];keep=pixels[p]==flat;ix=ix[keep];p=p[keep]
        gene=lookup[a['gene'][ix]]
        mats[0]+=sparse.coo_matrix((np.ones(len(ix),np.int64),(p,gene)),shape=mats[0].shape).tocsr()
        if core is not None and len(ix):
            inside=core[a['y'][ix],a['x'][ix]]>0;ix=ix[inside];p=p[inside];gene=gene[inside];fold=molecule_fold(a['transcript_id'][ix])
            for f in (0,1):
                use=fold==f;mats[1+f]+=sparse.coo_matrix((np.ones(int(use.sum()),np.int64),(p[use],gene[use])),shape=mats[0].shape).tocsr()
    return mats


def _query_pixels(data,core=None,band=None):
    occupied=np.zeros(data['nuclei'].shape,bool)
    for _,_,a in _coordinates(data):
        good=(a['x']>=0)&(a['y']>=0);occupied[a['y'][good],a['x'][good]]=True
    if band is not None:occupied&=band&(data['baseline']>0)&(data['nuclei']==0)
    elif core is not None:occupied&=(core>0)|(data['nuclei']==0)
    return np.flatnonzero(occupied.ravel())


def _candidate_expression(counts,pixels,offsets,rows,log_profiles):
    edge_pixel=np.repeat(np.arange(len(pixels)),np.diff(offsets));out=np.zeros(len(rows),float)
    totals=np.asarray(counts.sum(1)).ravel();order=np.argsort(rows,kind='stable');sorted_rows=rows[order]
    if not len(rows):return out
    starts=np.r_[0,1+np.flatnonzero(sorted_rows[1:]!=sorted_rows[:-1]),len(rows)]
    for a,b in zip(starts[:-1],starts[1:]):
        ix=order[a:b];p=edge_pixel[ix];out[ix]=np.asarray(counts[p]@log_profiles[int(sorted_rows[a])]).ravel()/np.maximum(totals[p],1)
    return out


def _winner(scores,offsets,ids,rows):
    result=np.zeros(len(offsets)-1,np.int64);gap=np.full(len(result),np.nan)
    lengths=np.diff(offsets);nonempty=np.flatnonzero(lengths)
    if not len(nonempty):return result,gap
    starts=offsets[nonempty];best=np.maximum.reduceat(scores,starts)
    group=np.repeat(np.arange(len(nonempty)),lengths[nonempty])
    winners=np.minimum.reduceat(np.where(scores==best[group],ids[rows],np.iinfo(np.int64).max),starts)
    second=scores.copy();second[ids[rows]==winners[group]]=-np.inf
    runner_up=np.maximum.reduceat(second,starts)
    result[nonempty]=winners;gap[nonempty]=best-runner_up
    return result,gap


def _support_diagnostics(cfg,data,counts,stats,out):
    """Population-wide fixed support diagnostics; no invented outer owner labels."""
    support=data['support'];core,_=support.core_image();pixels=_query_pixels(data,core=core)
    offsets,rows,coverage=support.candidate_csr(pixels,lambda i,n:_notify(cfg,'A1_CORE','complete_support',nuclei=i,total=n))
    geom=support.candidate_geometry(pixels,offsets,rows,lambda i,n:_notify(cfg,'A1_CORE','complete_geometry',nuclei=i,total=n))
    # Keep every candidate and geometry score, including unavailable profiles.
    atomic_npz(out/'candidate_geometry.npz',pixels=pixels,offsets=offsets,candidate_rows=rows,candidate_ids=data['ids'][rows],geometry=geom)
    expression,f0,f1=_pixel_expression(data,pixels,core);nuclear=data['nuclei'].ravel()[pixels]>0
    owner=data['nuclei'].ravel()[pixels].astype(np.int64);owner_rows=np.searchsorted(data['ids'],owner)
    valid_owner=nuclear&(owner_rows<len(data['ids']));k=np.diff(offsets);edge_pixel=np.repeat(np.arange(len(pixels)),k)
    owner_safe=np.minimum(owner_rows,len(data['ids'])-1)
    prior=data['theta'][np.where(legal_types(data['types'],data['valid']),data['types'],0)]
    type_expr=_candidate_expression(expression,pixels,offsets,rows,np.log(prior))
    full=counts.sum(0);q=profile(full,prior);donor,bins=matched_donors(full,data['types'],stats['eligible'],SHUFFLE_SEEDS[0])
    shuf=prior.copy();available=donor>=0;shuf[available]=profile(full[donor[available]],prior[available])
    core_expr=_candidate_expression(expression,pixels,offsets,rows,np.log(q));shuf_expr=_candidate_expression(expression,pixels,offsets,rows,np.log(shuf))
    available_edges=stats['eligible'][rows]&(donor[rows]>=0)
    unavailable=np.bincount(edge_pixel,weights=~available_edges,minlength=len(pixels))>0
    outer=(~nuclear)&(k>=2);outer_common=outer&~unavailable
    wg,gg=_winner(geom,offsets,data['ids'],rows);wt,_=_winner(geom+type_expr,offsets,data['ids'],rows)
    wc,_=_winner(geom+core_expr,offsets,data['ids'],rows);ws,_=_winner(geom+shuf_expr,offsets,data['ids'],rows)
    molecule_n=np.asarray(expression.sum(1)).ravel()
    outer_rows=[]
    for kval in np.unique(k[~nuclear]):
        use=(~nuclear)&(k==kval)
        outer_rows.append(dict(candidate_count=int(kval),pixels=int(use.sum()),G_molecules=int(molecule_n[use].sum()),
            all_candidates_core_and_donor_available_pixels=int((use&~unavailable).sum()),
            geometry_unique_best_pixels=int((use&(gg>0)).sum())))
    pd.DataFrame(outer_rows).to_csv(out/'outer_support_coverage.csv',index=False)
    # Nuclear difficulty: same-type alternatives that are themselves in the paired population.
    same=(data['types'][rows]==data['types'][owner_safe[edge_pixel]])&stats['paired'][rows]&stats['paired'][owner_safe[edge_pixel]]&nuclear[edge_pixel]
    alternatives=same&(data['ids'][rows]!=owner[edge_pixel]);same_n=np.bincount(edge_pixel,weights=alternatives,minlength=len(pixels))
    nuclear_scores=np.zeros((len(data['ids']),3));nuclear_molecules=np.zeros(len(data['ids']),np.int64)
    diag_geometry_decisive=0;diag_nuclear_n=0
    for fold,held in ((0,f1),(1,f0)):
        qp=profile(counts[fold],prior);logshuffle=np.zeros_like(prior);rr=np.flatnonzero(stats['paired'])
        for si in range(20):
            d=stats['donors'][si,fold]
            for start in range(0,len(rr),4096):
                ix=rr[start:start+4096];logshuffle[ix]+=np.log(profile(counts[fold,d[ix]],prior[ix]))/20
        ce=_candidate_expression(held,pixels,offsets,rows,np.log(qp));te=_candidate_expression(held,pixels,offsets,rows,np.log(prior));se=_candidate_expression(held,pixels,offsets,rows,logshuffle)
        hn=np.asarray(held.sum(1)).ravel();use=nuclear&(same_n>0)&(hn>0)&stats['paired'][owner_safe]
        for p in np.flatnonzero(use):
            a,b=offsets[p:p+2];ix=np.arange(a,b);own=ix[data['ids'][rows[ix]]==owner[p]];other=ix[alternatives[ix]]
            if len(own)!=1 or not len(other):continue
            for j,e in enumerate((ce,te,se)):nuclear_scores[owner_rows[p],j]+=(e[own[0]]-e[other].max())*hn[p]
            nuclear_molecules[owner_rows[p]]+=hn[p];diag_nuclear_n+=int(hn[p]);diag_geometry_decisive+=int(hn[p])*(geom[own[0]]>geom[other].max())
    values=np.divide(nuclear_scores,nuclear_molecules[:,None],out=np.full_like(nuclear_scores,np.nan),where=nuclear_molecules[:,None]>0)
    pd.DataFrame(dict(cell_id=data['ids'],heldout_molecules=nuclear_molecules,core_expression_margin=values[:,0],
        type_expression_margin=values[:,1],shuffle_expression_margin=values[:,2])).to_csv(out/'nuclear_candidate_diagnostic.csv',index=False)
    for name,scores in [('type',type_expr),('core',core_expr),('shuffled',shuf_expr)]:atomic_npy(out/f'outer_{name}_expression_scores.npy',scores)
    diag=dict(**coverage,query_pixels=len(pixels),outer_overlap_pixels=int(outer.sum()),outer_overlap_G_molecules=int(molecule_n[outer].sum()),
        outer_common_pixels=int(outer_common.sum()),outer_common_G_molecules=int(molecule_n[outer_common].sum()),
        outer_missing_core_or_donor_pixels=int((outer&unavailable).sum()),
        outer_G_vs_GT_disagreement_pixels=int((outer_common&(wg!=wt)).sum()),
        outer_GT_vs_GTC_disagreement_pixels=int((outer_common&(wt!=wc)).sum()),
        outer_GTC_vs_SHUFFLED_disagreement_pixels=int((outer_common&(wc!=ws)).sum()),
        outer_GT_vs_GTC_disagreement_G_molecules=int(molecule_n[outer_common&(wt!=wc)].sum()),
        outer_geometry_decisive_pixels=int((outer_common&(gg>0)).sum()),
        nuclear_same_type_heldout_molecules=diag_nuclear_n,nuclear_geometry_already_decisive_molecules=diag_geometry_decisive,
        nuclear_candidate_cells=int((nuclear_molecules>0).sum()),nuclear_candidate_diagnostic_only=True,
        outer_accuracy=None,outer_accuracy_reason='No independent true owner labels',
        expression_aggregation='mean per molecule within native pixel; nuclear diagnostics use held-out Core folds',
        support_scope='all original complete nuclei; not decoder view visibility')
    common.atomic_json(out/'support_diagnostics.json',_json_safe(diag));return diag


def _write_statistics(cfg,data,counts,count_info,out):
    stats=cross_scores(counts,data['types'],data['theta'],data['valid']);eligible=stats['eligible'];paired=stats['paired']
    types=data['types'];ids=data['ids'];support=data['support'];known=legal_types(types,data['valid'])
    reasons=np.select([~known,count_info['core_area']==0,stats['fold_n'][0]<10,stats['fold_n'][1]<10,~paired],
        ['invalid_unknown_or_reference','empty_eroded_core','F0_below_10','F1_below_10','no_matched_donor_both_folds'],default='paired')
    density=cKDTree(support.centroids).query_ball_point(support.centroids,30,return_length=True,workers=1)-1
    edges=np.unique(np.quantile(density,[.25,.5,.75]));density_group=np.searchsorted(edges,density,side='left')
    names=[data['reference']['type_names'][t] if 0<=t<len(data['reference']['type_names']) else 'Unknown' for t in types]
    coverage=pd.DataFrame(dict(cell_id=ids,type_index=types,type=names,nuclear_area=support.area,core_area=count_info['core_area'],
        F0=stats['fold_n'][0],F1=stats['fold_n'][1],total_core_G=stats['total_n'],
        type_valid=known,evaluable=eligible,paired=paired,reason=reasons,x_um=support.centroids[:,0],y_um=support.centroids[:,1],
        density_30um=density,density_group=density_group))
    coverage.to_csv(out/'cell_coverage.csv',index=False)
    shuffle_mean=np.full(len(ids),np.nan);shuffle_mean[paired]=stats['shuffle'][:,paired].mean(0)
    scores=pd.DataFrame(dict(cell_id=ids,evaluable=eligible,paired=paired,core=stats['core'],type=stats['type'],
        shuffle_mean=shuffle_mean,core_minus_type=stats['core']-stats['type'],core_minus_shuffle=stats['core']-shuffle_mean,
        heldout_molecules=stats['total_n'],split_cosine=stats['split_cosine'],type_residual_cosine=stats['type_residual_cosine'],
        within_type_centered_cosine=stats['within_type_centered_cosine']))
    scores.to_csv(out/'per_cell_scores.csv',index=False)
    atomic_npz(out/'donor_tables.npz',donors=stats['donors'],fold_bins=stats['bins'],ids=ids,eligible=eligible,paired=paired)
    shuffle_rows=[]
    for i,seed in enumerate(SHUFFLE_SEEDS):
        for row in np.flatnonzero(paired):
            shuffle_rows.append((int(ids[row]),seed,float(stats['shuffle'][i,row]),int(ids[stats['donors'][i,0,row]]),
                int(ids[stats['donors'][i,1,row]]),int(stats['fold_n'][0,row]),int(stats['fold_n'][1,row]),
                int(stats['fold_n'][0,stats['donors'][i,0,row]]),int(stats['fold_n'][1,stats['donors'][i,1,row]])))
    pd.DataFrame(shuffle_rows,columns=['cell_id','seed','heldout_score','F0_donor_id','F1_donor_id','F0_recipient_count','F1_recipient_count','F0_donor_count','F1_donor_count']).to_csv(out/'shuffle_results.csv',index=False)
    del shuffle_rows
    vals=np.c_[stats['core'][paired],stats['type'][paired],shuffle_mean[paired]];uncertainty={}
    for side in (500,1000,2000):
        r,samples=paired_bootstrap(vals,support.centroids[paired],side,lambda i,n:_notify(cfg,'A1_CORE','bootstrap_'+str(side),replicate=i,total=n));uncertainty[str(side)]=r
        atomic_npy(out/f'bootstrap_{side}.npy',samples)
    common.atomic_json(out/'spatial_uncertainty.json',uncertainty)
    decision=decide(uncertainty['1000']);decision.update(total_original_cells=len(ids),evaluable_cells=int(eligible.sum()),paired_cells=int(paired.sum()),
        paired_fraction_all=float(paired.mean()),paired_fraction_evaluable=float(paired.sum()/eligible.sum()) if eligible.sum() else None,
        all_evaluable_core_minus_type=float((stats['core'][eligible]-stats['type'][eligible]).mean()) if eligible.any() else None,
        paired_molecule_weighted_core_minus_type=float(np.average((stats['core']-stats['type'])[paired],weights=stats['total_n'][paired])) if paired.any() else None,
        scope='entire available nuclear population' if not data['mini'] else 'engineering mini only; not formal A1 decision',
        limitations=LIMITATIONS,dependencies=common.receipt_dependencies(cfg,'A1_CORE'),
        parameters=PARAMETERS,density_edges=edges.tolist())
    # Preformal scope clarification from integration: reuse the SAME original
    # gate for type scope, without weakening coverage or significance thresholds.
    # This never changes global A1 science_status. Unestablished types abstain
    # in every A2 arm and remain visible in coverage/strata outputs.
    scope={};applicable=[]
    for t in np.unique(types[paired]):
        use=paired&(types==t)
        tv=np.c_[stats['core'][use],stats['type'][use],shuffle_mean[use]]
        tr,ts=paired_bootstrap(tv,support.centroids[use],1000,lambda i,n:_notify(cfg,'A1_CORE','type_scope_bootstrap',type_index=int(t),replicate=i,total=n))
        td=decide(tr);name=data['reference']['type_names'][int(t)]
        scope[name]=dict(type_index=int(t),**td)
        if td['science_status']=='GO_A2':applicable.append(int(t))
        atomic_npy(out/f'bootstrap_type_{int(t):02d}_1000.npy',ts)
    decision.update(A2_applicable_type_indices=applicable,
        A2_scope_status='ESTABLISHED_TYPES' if applicable else 'INCONCLUSIVE_SCOPE',
        A2_scope_rule='same original A1 gate applied per type; unestablished types abstain in all arms; global gate unchanged',
        scope_clarification='Conservative preformal integration clarification; independent audit must review against plan section 3.5.')
    common.atomic_json(out/'type_scope.json',scope)
    common.atomic_json(out/'decision.json',decision)
    return stats,coverage,scores,decision


def _strata(data,stats,coverage,scores,out):
    # Spatial density and support-defined same/heterotypic neighbourhoods are frozen from input.
    candidate=np.load(out/'candidate_geometry.npz');pixels=candidate['pixels'];off=candidate['offsets'];rows=candidate['candidate_rows']
    edge_pixel=np.repeat(np.arange(len(pixels)),np.diff(off));owner=data['nuclei'].ravel()[pixels];orow=np.searchsorted(data['ids'],owner)
    active=(owner[edge_pixel]>0)&(orow[edge_pixel]<len(data['ids']))
    er=np.flatnonzero(active);own=orow[edge_pixel[er]];cand=rows[er]
    other=data['ids'][cand]!=data['ids'][own];er=er[other];own=own[other];cand=cand[other]
    same=np.zeros(len(data['ids']),bool);hetero=same.copy();unknown=same.copy()
    np.logical_or.at(same,own,(data['types'][cand]>=0)&(data['types'][cand]==data['types'][own]))
    np.logical_or.at(hetero,own,(data['types'][cand]>=0)&(data['types'][own]>=0)&(data['types'][cand]!=data['types'][own]))
    np.logical_or.at(unknown,own,data['types'][cand]<0)
    groups=dict(type=coverage['type'].to_numpy(),core_count=np.select([stats['total_n']<20,stats['total_n']<50,stats['total_n']<100],['below20','[20,50)','[50,100)'],default='>=100'),
        density=coverage.density_group.to_numpy(),same_type_neighbour=same,heterotypic_neighbour=hetero,unknown_neighbour=unknown)
    output=[]
    for population,mask in [('all_evaluable',stats['eligible']),('paired',stats['paired'])]:
        for family,values in groups.items():
            for level in np.unique(values):
                selected=mask&(values==level);n=int(selected.sum());rr=scores.loc[selected]
                item=dict(population=population,family=family,group=str(level),n_cells=n,
                    n_all_group=int((values==level).sum()),heldout_molecules=int(stats['total_n'][selected].sum()),
                    core_minus_type=float(rr.core_minus_type.mean()) if n else None,
                    core_minus_shuffle=float(rr.core_minus_shuffle.mean()) if population=='paired' and n else None,
                    split_cosine=float(rr.split_cosine.mean()) if n else None,
                    type_residual_cosine=float(rr.type_residual_cosine.mean()) if n else None,
                    within_type_centered_cosine=float(rr.within_type_centered_cosine.mean()) if n else None)
                output.append(item)
    pd.DataFrame(output).to_csv(out/'strata_scores.csv',index=False)
    atomic_npz(out/'support_neighbour_flags.npz',ids=data['ids'],same=same,heterotypic=hetero,unknown=unknown)
    return output


def _run_a1(cfg):
    out=common.output_path(cfg,'A1_CORE');out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return validate_node(cfg,'A1_CORE')
    start=time.monotonic();data=_inputs(cfg)
    manifest=dict(inputs=data['source'],parameters=PARAMETERS,genes=[data['genes'][i] for i in data['gene_indices']],
        reference_used_fields=['theta','gene_indices','gene_names','type_names','valid'],DAPI_loaded=False,
        nuclear_population=len(data['ids']),eligible_molecules=data['expected'],coordinate='verified historical original native INTER_NEAREST cache in original row order',
        dependencies=common.receipt_dependencies(cfg,'A1_CORE'))
    if (out/'input_manifest.json').exists() and common.read(out/'input_manifest.json')!=manifest:raise ValueError('A1 input manifest drift')
    common.atomic_json(out/'input_manifest.json',manifest)
    counts,count_info=_count_core(cfg,data,out)
    stats,coverage,scores,decision=_write_statistics(cfg,data,counts,count_info,out)
    _notify(cfg,'A1_CORE','primary_statistics',paired_cells=int(stats['paired'].sum()))
    diagnostics=_support_diagnostics(cfg,data,counts,stats,out);strata=_strata(data,stats,coverage,scores,out)
    # Type restrictions are descriptive, never an adaptive gate or donor population change.
    type_values=[r for r in strata if r['population']=='paired' and r['family']=='type' and r['n_cells']]
    positive_types=[r['group'] for r in type_values if r['core_minus_type']>=.01 and r['core_minus_shuffle']>0]
    decision.update(descriptively_positive_types=positive_types,scope_caveat='Descriptive means do not decide applicability; only the unchanged A1 gate applied per type establishes scope.')
    common.atomic_json(out/'decision.json',decision)
    elapsed=time.monotonic()-start
    common.atomic_json(out/'runtime.json',dict(wall_seconds=elapsed,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,maximum_K=diagnostics['maximum_K']))
    (out/'report.md').write_text('# A1 Core information test\n\nExecution: COMPLETED. Science: '+decision['science_status']+'.\n\n'
        f"Original nuclei: {len(data['ids'])}; evaluable: {int(stats['eligible'].sum())}; paired: {int(stats['paired'].sum())}.\n\n"
        'The primary statistic is cell-equal held-out log predictive gain, with no geometry term. '
        'All paired Type/Core/shuffle comparisons use identical cells and block resamples. '
        'A1 is an information test, not a mask/F1 endpoint.\n\n'+'\n'.join('- '+v for v in LIMITATIONS)+'\n')
    common.write_complete(cfg,'A1_CORE',decision['science_status'],maximum_K=diagnostics['maximum_K'],wall_seconds=elapsed,
        scope=decision['scope'],eligible_molecules=data['expected'],nuclear_population=len(data['ids']))
    return validate_node(cfg,'A1_CORE')


def _a2_masks(cfg,data,counts,eligible,out,oracle=False,interrupt_after=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);baseline=data['baseline'];nuclei=data['nuclei'];sup=data['support']
    full=counts.sum(0);prior=data['theta'][np.where(legal_types(data['types'],data['valid']),data['types'],0)]
    donor,bins=matched_donors(full,data['types'],eligible,SHUFFLE_SEEDS[0]);q=profile(full,prior);shuf=prior.copy();good=donor>=0
    shuf[good]=profile(full[donor[good]],prior[good])
    pixels=_query_pixels(data,band=interface_band(baseline));expression=_pixel_expression(data,pixels)[0]
    offsets,rows,support_stats=sup.candidate_csr(pixels,lambda i,n:_notify(cfg,'A2_CORE','fixed_support',nuclei=i,total=n))
    pos,offsets,rows,reasons=fixed_domain(baseline,nuclei,pixels,offsets,rows,data['ids'],eligible,donor,expression)
    pixels=pixels[pos];expression=expression[pos];geometry=sup.candidate_geometry(pixels,offsets,rows)
    atomic_npz(out/'fixed_domain.npz',pixels=pixels,offsets=offsets,candidate_rows=rows,candidate_ids=data['ids'][rows],geometry=geometry,
        full_count_donors=donor,full_count_bins=bins,eligible=eligible,ids=data['ids'])
    sparse.save_npz(out/'fixed_domain_expression.npz',expression)
    common.atomic_json(out/'domain.json',dict(support_stats=support_stats,abstentions=reasons,eligible_pixels=len(pixels),
        eligible_G_molecules=int(expression.sum()),same_domain_for=['A2_GT','A2_GTC','A2_SHUFFLED'],
        seed=SHUFFLE_SEEDS[0],full_count_donor_groups=True,core_and_edited_molecules_disjoint=bool(np.all(nuclei.ravel()[pixels]==0)),
        scope='2 um band from frozen positive-positive 8-neighbour interface, nuclear exterior, expression, complete support and baseline adjacent candidates'))
    results={};summary=[]
    for name,prof in [('A2_GT',prior),('A2_GTC',q),('A2_SHUFFLED',shuf)]:
        arm=out/'endpoints'/name;arm.mkdir(parents=True,exist_ok=True)
        proposals,scores,below=score_proposals(baseline,pixels,offsets,rows,geometry,expression,data['ids'],prof)
        atomic_npy(arm/'proposals.npy',proposals);atomic_npy(arm/'candidate_scores.npy',scores)
        deps=dict(config=common.digest(cfg),inputs=common.digest(data['source']),domain=common.sha256(out/'fixed_domain.npz'),
            proposals=common.sha256(arm/'proposals.npy'),source=common.sha256(Path(__file__).with_name('core_refine.py')),profile=name)
        mask,info=apply_proposals(baseline,nuclei,proposals,arm/'recovery',deps,
            lambda i,n:_notify(cfg,'A2_CORE','apply_'+name,proposal_cursor=i,total_proposals=n),stop_after=interrupt_after,oracle=oracle)
        if mask is None:return None,dict(interrupted=name,**info)
        final=arm/'predictions'/name/'final.tif';final.parent.mkdir(parents=True,exist_ok=True)
        tmp=final.with_name('final.tmp.tif');tifffile.imwrite(tmp,mask.astype(np.uint32));os.replace(tmp,final)
        decisions=np.load(common.read(arm/'recovery/state.json')['decisions']['path'])
        table=pd.DataFrame.from_records(proposals);table['accepted']=decisions==1
        if len(table):
            flat=table.y.to_numpy(np.int64)*mask.shape[1]+table.x.to_numpy(np.int64);ix=np.searchsorted(pixels,flat)
            table['G_molecules']=np.asarray(expression.sum(1)).ravel()[ix]
        else:table['G_molecules']=pd.Series(dtype=np.int64)
        table.to_csv(arm/'transfers.csv',index=False)
        accepted=table.accepted.to_numpy(bool)
        record=dict(endpoint=name,eligible_pixels=len(pixels),eligible_G_molecules=int(expression.sum()),raw_proposals=len(proposals),
            below_margin_or_current_best=below,accepted_pixels=info['accepted'],rejected_structure=info['rejected'],
            accepted_G_molecules=int(table.loc[accepted,'G_molecules'].sum()),**info['reasons'])
        common.atomic_json(arm/'transfer_summary.json',record);summary.append(record);results[name]=dict(final=str(final),summary=record)
    pd.DataFrame(summary).fillna(0).to_csv(out/'proposal_and_rejection_counts.csv',index=False)
    return results,dict(support=support_stats,domain_pixels=len(pixels),same_domain=True)


def _run_a2(cfg):
    out=common.output_path(cfg,'A2_CORE');out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return validate_node(cfg,'A2_CORE')
    validate_node(cfg,'A1_CORE');decision=common.read(common.output_path(cfg,'A1_CORE','decision.json'))
    if decision['science_status']!='GO_A2':raise ValueError('A2 requires complete A1 GO_A2; runtime must record scientific skip')
    if cfg.get('a_core_mini_inputs'):raise ValueError('formal conditional A2 must not run on mini override')
    data=_inputs(cfg);saved=np.load(common.output_path(cfg,'A1_CORE','core_counts.npz'));counts=saved['counts']
    coverage=pd.read_csv(common.output_path(cfg,'A1_CORE','cell_coverage.csv'));eligible=coverage.evaluable.to_numpy(bool)
    eligible &= np.isin(data['types'],decision['A2_applicable_type_indices'])
    if not np.array_equal(coverage.cell_id,data['ids']) or not np.array_equal(saved['ids'],data['ids']):raise ValueError('A1/A2 population mismatch')
    common.atomic_json(out/'input_manifest.json',dict(inputs=data['source'],A1=common.identity(common.output_path(cfg,'A1_CORE','complete.json')),parameters=PARAMETERS))
    start=time.monotonic();results,info=_a2_masks(cfg,data,counts,eligible,out)
    from .evaluation import build_and_evaluate,validated_reuse,validate_endpoint
    validated_reuse(cfg,'A2_NATIVE');native=validate_endpoint(cfg,'BASE_MATCH')
    evaluations={}
    for name in A2_ENDPOINTS[1:]:
        e=common.endpoint_spec(cfg,name);machine=dict(data['machine']);machine.update(final_mask=e['final'],pre_mask=e['final'],final_matrix=str(Path(e['matrix'])/'expr_mat.csv'))
        common.atomic_json(e['machine'],machine);evaluations[name]=build_and_evaluate(cfg,name)
        if evaluations[name]['eligible']!=native['eligible'] or evaluations[name]['assigned']!=native['assigned']:raise ValueError('A2 final matrix capture not exactly constant')
        _notify(cfg,'A2_CORE','evaluated_'+name,assigned=evaluations[name]['assigned'])
    from .reports import compare_endpoints
    comparisons={}
    for candidate,control in [('A2_GTC','A2_GT'),('A2_GTC','A2_NATIVE'),('A2_GTC','A2_SHUFFLED')]:
        comparisons[candidate+'_vs_'+control]=compare_endpoints(cfg,candidate,control)
    science='EXPLORATORY_FIXED_REFINEMENT_EVALUATED'
    common.atomic_json(out/'comparison.json',dict(execution_status='COMPLETED',science_status=science,
        comparisons=comparisons,capture_exactly_equal=True,assigned=native['assigned'],eligible=native['eligible'],
        adoption_requires_both=['A2_GTC_vs_A2_GT','A2_GTC_vs_A2_NATIVE'],limitations=LIMITATIONS))
    common.write_complete(cfg,'A2_CORE',science,wall_seconds=time.monotonic()-start,maximum_K=info['support']['maximum_K'])
    return validate_node(cfg,'A2_CORE')


def run_node(config,node):
    cfg=common.load_run(config)
    if node not in ('A1_CORE','A2_CORE'):raise ValueError('not an A_CORE node')
    try:return _run_a1(cfg) if node=='A1_CORE' else _run_a2(cfg)
    except SafeStop:raise
    except Exception as exc:
        common.atomic_json(common.output_path(cfg,node,'failure.json'),dict(execution_status='FAILED_INTEGRITY' if isinstance(exc,ValueError) else 'FAILED_RUNTIME',
            science_status='INVALID',exception_type=type(exc).__name__,reason=str(exc),dependencies=common.receipt_dependencies(cfg,node)))
        raise


def validate_node(config,node):
    cfg=common.load_run(config);rec=common.validate_complete(cfg,node);out=common.output_path(cfg,node)
    if node=='A1_CORE':
        counts=np.load(out/'core_counts.npz');coverage=pd.read_csv(out/'cell_coverage.csv');scores=pd.read_csv(out/'per_cell_scores.csv')
        split=common.read(out/'split_manifest.json');decision=common.read(out/'decision.json');uncertainty=common.read(out/'spatial_uncertainty.json')
        if not np.array_equal(coverage.cell_id,counts['ids']) or not np.array_equal(scores.cell_id,counts['ids']):raise ValueError('A1 output cell population mismatch')
        if not np.array_equal(counts['counts'].sum(2),coverage[['F0','F1']].to_numpy().T) or int(counts['counts'].sum())!=split['totals']['core_G']:raise ValueError('A1 fold/count conservation')
        if not split['split_disjoint'] or split['duplicate_ids']!=0:raise ValueError('A1 split leakage')
        donors=np.load(out/'donor_tables.npz');d=donors['donors'];paired=scores.paired.to_numpy(bool)
        if not np.array_equal(paired,donors['paired']) or not np.array_equal(paired,coverage.paired.to_numpy(bool)):raise ValueError('paired population disagreement')
        for f in (0,1):
            for si in range(20):
                use=d[si,f]>=0;rows=np.flatnonzero(use);dd=d[si,f,use]
                if np.any(rows==dd) or not np.array_equal(coverage.type_index.to_numpy()[rows],coverage.type_index.to_numpy()[dd]) or not np.array_equal(donors['fold_bins'][f,rows],donors['fold_bins'][f,dd]):raise ValueError('invalid donor permutation')
        recomputed=decide(uncertainty['1000'])
        if decision['science_status']!=recomputed['science_status'] or rec['science_status']!=decision['science_status'] or decision['parameters']!=PARAMETERS:raise ValueError('A1 decision/gate drift')
        if not np.isfinite(scores.loc[paired,['core','type','shuffle_mean']].to_numpy()).all():raise ValueError('invalid paired finite scores')
        if common.read(out/'input_manifest.json')['DAPI_loaded']:raise ValueError('A1 irrelevant DAPI load')
    elif node=='A2_CORE':
        from .evaluation import validate_endpoint
        results=[validate_endpoint(cfg,name) for name in A2_ENDPOINTS]
        if len({(r['eligible'],r['assigned']) for r in results})!=1:raise ValueError('A2 capture not constant')
        domain=np.load(out/'fixed_domain.npz');full=domain['full_count_donors'];rows=domain['candidate_rows']
        if np.any(full[rows]<0) or not domain['eligible'][rows].all():raise ValueError('A2 shared candidate domain drift')
        baseline=tifffile.imread(cfg['endpoints']['BASE_MATCH']['final']);machine=common.read(cfg['endpoints']['BASE_MATCH']['machine']);nuc=tifffile.imread(machine['nuclei_mask'])
        allowed=np.zeros(baseline.shape,bool);allowed.ravel()[domain['pixels']]=True
        for name in A2_ENDPOINTS[1:]:
            mask=tifffile.imread(common.endpoint_spec(cfg,name)['final']);changed=mask!=baseline
            if not np.array_equal(mask>0,baseline>0) or np.any(changed&~allowed) or np.any(changed&(nuc>0)):raise ValueError('A2 mask/foreground/domain invariant')
    else:raise ValueError(node)
    return rec


def _make_mini(cfg,out):
    """Deterministic real-data ROI chosen by input-only maximum support, before scores.

    Native coordinates are first mapped in the historical full image. The mini
    stores those exact native integer coordinates translated to a local origin;
    its scale=1 machine is explicitly an engineering fixture, not new biology.
    Only cells whose complete nucleus AND baseline final fit inside are retained.
    """
    out=Path(out);manifest_path=out/'inputs/manifest.json'
    if manifest_path.exists():
        for obj in common.read(manifest_path)['outputs'].values():common.check_identity(obj)
        return str(manifest_path)
    data=_inputs(cfg);src=out/'inputs';src.mkdir(parents=True,exist_ok=True)
    frozen=np.load(Path(cfg['inputs']['frozen_strata'])/'support_counts.npy',mmap_mode='r')
    centre=np.unravel_index(int(np.argmax(frozen)),frozen.shape);side=384
    y0=max(0,min(int(centre[0])-side//2,frozen.shape[0]-side));x0=max(0,min(int(centre[1])-side//2,frozen.shape[1]-side));y1=y0+side;x1=x0+side
    from scipy import ndimage as ndi
    final_boxes=ndi.find_objects(data['baseline']);nuclear_boxes=data['support'].boxes;retained=[]
    for cid in np.unique(data['nuclei'][y0:y1,x0:x1]):
        if cid<=0:continue
        nb=nuclear_boxes[cid-1];fb=final_boxes[cid-1] if cid<=len(final_boxes) else None
        if fb and all(b[0].start>=y0 and b[0].stop<=y1 and b[1].start>=x0 and b[1].stop<=x1 for b in (nb,fb)):retained.append(int(cid))
    if len(retained)<20:raise ValueError('real mini has too few complete cells')
    retained=np.asarray(sorted(retained),np.int64);rr=np.searchsorted(data['ids'],retained)
    nuclei=data['nuclei'][y0:y1,x0:x1].copy();baseline=data['baseline'][y0:y1,x0:x1].copy()
    nuclei[~np.isin(nuclei,retained)]=0;baseline[~np.isin(baseline,retained)]=0
    tifffile.imwrite(src/'nuclei.tif',nuclei.astype(np.uint32));tifffile.imwrite(src/'baseline.tif',baseline.astype(np.uint32))
    atomic_npz(src/'metadata.npz',ids=retained.astype(np.int32),type_index=data['types'][rr],area=data['support'].area[rr],
        centroid_xy=data['support'].centroids[rr]-[x0,y0])
    selected=[]
    for _,_,a in _coordinates(data):
        keep=(a['x']>=x0)&(a['x']<x1)&(a['y']>=y0)&(a['y']<y1);b=np.array(a[keep],copy=True);b['x']-=x0;b['y']-=y0;selected.append(b)
    a=np.concatenate(selected);records=[]
    for i,b in enumerate(np.array_split(a,4)):
        p=src/f'coordinates_{i:04d}.npy';atomic_npy(p,b);start=sum(r['rows'] for r in records)
        records.append(dict(coordinate_file=str(p),coordinate_sha256=common.sha256(p),rows=len(b),row_start=start,row_stop=start+len(b)))
    pd.DataFrame(dict(transcript_id=a['transcript_id'],x_location=a['x'],y_location=a['y'],feature_name=np.asarray(data['genes'])[a['gene']],qv=np.full(len(a),20))).to_csv(src/'transcripts.csv',index=False)
    machine=dict(data['machine']);machine.update(final_mask=str(src/'baseline.tif'),pre_mask=str(src/'baseline.tif'),nuclei_mask=str(src/'nuclei.tif'),
        transcripts=str(src/'transcripts.csv'),scale_x=1.,scale_y=1.,chunksize=max(1,len(a)//4),final_matrix=str(src/'unused_expr_mat.csv'),baseline_final_mask=str(src/'baseline.tif'))
    common.atomic_json(src/'machine.json',machine)
    outputs={p.name:common.identity(p) for p in src.iterdir() if p.is_file() and p.name!='manifest.json'}
    common.atomic_json(manifest_path,dict(paths=dict(machine=str(src/'machine.json'),metadata=str(src/'metadata.npz')),coordinate_records=records,
        molecules=len(a),nuclei=len(retained),provenance=dict(original=data['source'],roi_yxyx=[y0,x0,y1,x1],
        original_native_maximum_support_K=int(frozen.max()),ROI_selection='384 native px around input-only maximum complete-support count',
        local_coordinates='historical exact native lookup, then integer translation; mini scale=1',complete_cells=True,formal=False),outputs=outputs))
    return str(manifest_path)


def validate_mini(config,output):
    cfg=common.load_run(config);out=Path(output).resolve()
    root=Path(cfg['output_root']).resolve()
    if root!=out and root not in out.parents:raise ValueError('mini output must be under config output_root')
    if 'mini' not in out.parts:raise ValueError('mini must be isolated under mini directory')
    out.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    # Runtime never calls this under its formal lock: standalone admission holds one shared lock.
    with common.compute_lock(timeout=3600):
        acquired=time.monotonic()
        mini_cfg=dict(cfg);mini_cfg['output_root']=str(out);mini_cfg['run_id']='mini-a-core'
        mini_cfg['a_core_mini_inputs']=_make_mini(cfg,out)
        mini_cfg['source_root']=str(Path(__file__).resolve().parents[2])
        common.atomic_json(out/'mini_run.json',mini_cfg)
        a1=run_node(mini_cfg,'A1_CORE');data=_inputs(mini_cfg)
        counts=np.load(out/'A1_CORE/core_counts.npz')['counts'];eligible=pd.read_csv(out/'A1_CORE/cell_coverage.csv').evaluable.to_numpy(bool)
        partial,partial_info=_count_core(mini_cfg,data,out/'count_recovery',stop_after=1)
        if partial is not None and not (out/'count_recovery/split_manifest.json').exists():raise ValueError('count recovery did not interrupt')
        recovered,recovered_info=_count_core(mini_cfg,data,out/'count_recovery')
        if not np.array_equal(counts,recovered):raise ValueError('real Core interruption/recovery counts differ')
        # Actual complete-image oracle checks every proposal in this bounded real fixture.
        direct,dinfo=_a2_masks(mini_cfg,data,counts,eligible,out/'a2_uninterrupted',oracle=True)
        interrupted,interruption=_a2_masks(mini_cfg,data,counts,eligible,out/'a2_recovered',oracle=True,interrupt_after=1)
        resumed,rinfo=_a2_masks(mini_cfg,data,counts,eligible,out/'a2_recovered',oracle=True)
        for name in A2_ENDPOINTS[1:]:
            if not np.array_equal(tifffile.imread(direct[name]['final']),tifffile.imread(resumed[name]['final'])):raise ValueError('real A2 recovered final differs')
            for artifact in ('proposals.npy','candidate_scores.npy'):
                if common.sha256(out/'a2_uninterrupted/endpoints'/name/artifact)!=common.sha256(out/'a2_recovered/endpoints'/name/artifact):raise ValueError('recovered proposals or frozen scores differ')
        from .evaluation import build_and_evaluate
        evaluations={};own_counts={}
        # Four actual mini endpoints, with full own matrix/evaluator. Mini-native is
        # a separately named ROI fixture, never a fabricated formal A2_NATIVE reuse.
        for arm in ('NATIVE','GT','GTC','SHUFFLED'):
            name='A2_MINI_'+arm;base=out/'evaluation_endpoints'/name
            spec=dict(endpoint_id=name,root=str(base),machine=str(base/'configs/machine.json'),matrix=str(base/'matrix'/name),evaluation=str(base/'evaluation'/name),cost_root=str(base))
            final=data['machine']['final_mask'] if arm=='NATIVE' else resumed['A2_'+arm]['final']
            machine=dict(data['machine']);machine.update(final_mask=final,pre_mask=final,final_matrix=str(Path(spec['matrix'])/'expr_mat.csv'))
            common.atomic_json(spec['machine'],machine);evaluations[arm]=build_and_evaluate(mini_cfg,name,spec)
            own_counts[arm]=np.load(Path(spec['matrix'])/'counts.npy')
        if len({(v['eligible'],v['assigned']) for v in evaluations.values()})!=1:raise ValueError('real mini four-arm capture differs')
        dimensions={name:list(value.shape) for name,value in own_counts.items()}
        receipt=dict(status='passed',scope='real engineering mini; not a formal experiment or A2 scientific GO',
            source=common.source_manifest(Path(__file__).resolve().parents[2]),config=common.identity(out/'mini_run.json'),
            input=common.identity(mini_cfg['a_core_mini_inputs']),A1=common.identity(out/'A1_CORE/complete.json'),
            counts_recovery_equal=True,count_interruption=partial_info,A2_recovery_equal=True,A2_interruption=interruption,
            A2_actual_interruption_exercised=interrupted is None,complete_image_topology_oracle='every actual proposal in every mini arm',
            four_arm_same_domain=True,constant_capture=True,own_matrix_shapes=dimensions,evaluations=evaluations,
            maximum_K=max(dinfo['support']['maximum_K'],a1['maximum_K']),
            wall_seconds=time.monotonic()-start,compute_wall_seconds=time.monotonic()-acquired,resource_wait_seconds=acquired-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            parameters=PARAMETERS,limitations=LIMITATIONS)
        common.atomic_json(out/'mini_receipt.json',_json_safe(receipt));return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--node',choices=['A1_CORE','A2_CORE']);p.add_argument('--validate',action='store_true');p.add_argument('--mini-output')
    args=p.parse_args()
    result=validate_mini(args.config,args.mini_output) if args.mini_output else (validate_node(args.config,args.node) if args.validate else run_node(args.config,args.node))
    print(json.dumps(_json_safe(result),indent=2),flush=True)
