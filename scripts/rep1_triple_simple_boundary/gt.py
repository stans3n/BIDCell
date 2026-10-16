"""Input-aware GT only, retaining the historical A1 eligibility and topology.

Every parent constructs its own domain. No Core-q score, P2 replay, training,
or evaluation is reachable from this module.
"""
from __future__ import annotations
from pathlib import Path
from collections import Counter
import os
import h5py
import numpy as np
import pandas as pd
import tifffile
import yaml
from scipy import ndimage as ndi, sparse
from scripts.boundary_three.core_refine import (NuclearSupport, interface_band, score_proposals,
                                               CompleteCellTopology, topology_oracle, rooted)
from scripts.boundary_three.core_scores import (reference_arrays, legal_types, matched_donors,
                                               SHUFFLE_SEEDS, PARAMETERS, MIN_FOLD)
from . import PARENTS, APPLICABLE_TYPES, N_CELLS, N_GENES, ELIGIBLE
from .inputs import resolve_parent
from .common import (atomic, atomic_csv, atomic_json, atomic_npy, atomic_npz, atomic_tiff, check_identity,
                     check_receipt, chunk_receipt, dependencies, digest, identity, inventory,
                     progress, read, resume_chunk, within, write_receipt, committed)

REASONS = ('unprocessed', 'accepted', 'nuclear_pixel', 'owner_or_background',
           'donor_components_increased', 'donor_nucleus_connection_lost', 'recipient_not_nucleus_connected')


def load_data(cfg, manifest, endpoint):
    parent = PARENTS[endpoint]
    spec = resolve_parent(cfg,manifest,endpoint)
    machine = read(spec['machine'])
    reference = read(manifest['reference_region'])
    meta = np.load(manifest['metadata'])
    ids, types = meta['ids'].astype(np.int64), meta['type_index'].astype(int)
    genes = Path(machine['genes']).read_text().splitlines()
    indices, theta, valid = reference_arrays(reference, genes)
    if len(ids) != N_CELLS or len(genes) != N_GENES:
        raise ValueError('GT requires the complete 107955 by 313 universe')
    decision = read(manifest['a1']['decision.json'])
    if decision['science_status'] != 'GO_A2' or decision['A2_applicable_type_indices'] != APPLICABLE_TYPES:
        raise ValueError('frozen A1 applicability changed')
    if decision['parameters'] != PARAMETERS:
        raise ValueError('A1 parameter contract differs from frozen algorithm')
    saved = np.load(manifest['a1']['core_counts.npz'])
    coverage = pd.read_csv(manifest['a1']['cell_coverage.csv'])
    counts = saved['counts']
    if not np.array_equal(saved['ids'], ids) or not np.array_equal(coverage.cell_id, ids) or not np.array_equal(saved['gene_indices'], indices):
        raise ValueError('A1 metadata/core/coverage gene or cell association mismatch')
    if counts.shape != (2, len(ids), len(indices)) or np.any(counts < 0):
        raise ValueError('invalid frozen A1 counts')
    expected_evaluable = legal_types(types, valid) & np.all(counts.sum(2) >= MIN_FOLD, axis=0)
    if not np.array_equal(expected_evaluable, coverage.evaluable.to_numpy(bool)) or not np.array_equal(coverage.type_index, types):
        raise ValueError('A1 coverage eligibility differs from frozen rule')
    eligible = expected_evaluable & np.isin(types, APPLICABLE_TYPES)
    donor, bins = matched_donors(counts.sum(0), types, eligible, SHUFFLE_SEEDS[0])
    # GT uses only the frozen type prior. counts above decide eligibility/donors only.
    prior = theta[np.where(legal_types(types, valid), types, 0)]
    nuclei = tifffile.imread(spec['nuclei'])
    baseline = tifffile.imread(spec['final'])
    if nuclei.shape != baseline.shape or machine['pixel_um'] != 1.0 or machine['scale_x'] != .2125 or machine['scale_y'] != .2125:
        raise ValueError('native GT geometry/coordinate scale drift')
    if not np.array_equal(baseline[nuclei > 0], nuclei[nuclei > 0]):
        raise ValueError('GT parent violates nuclear owner')
    with h5py.File(machine['preannotation']) as f:
        aid, at = f['ids'][:].astype(np.int64), f['data'][:]
    order = np.argsort(aid)
    if not np.array_equal(aid[order], ids) or not np.array_equal(at[order], types):
        raise ValueError('frozen type/ID association mismatch')
    settings = yaml.safe_load(Path(machine['baseline_config']).read_text())
    support = NuclearSupport(nuclei,ids,types,reference['type_names'],settings['model_params']['elongated'])
    if not np.array_equal(support.area,meta['area']) or not np.allclose(support.centroids,meta['centroid_xy'],rtol=0,atol=1e-8):
        raise ValueError('complete nuclear geometry/metadata differs')
    return dict(parent=parent, spec=spec, machine=machine, reference=reference,
                ids=ids, types=types, genes=genes, gene_indices=indices, prior=prior,
                eligible=eligible, donor=donor, bins=bins, support=support,
                nuclei=nuclei, baseline=baseline, coordinates=manifest['coordinates'])


def coordinates(data):
    cursor = 0
    for i, rec in enumerate(data['coordinates']):
        check_identity(rec)
        arr = np.load(rec['path'], mmap_mode='r')
        if arr.dtype.names != ('transcript_id','x','y','gene') or len(arr) != rec['rows'] or rec['row_start'] != cursor or rec['row_stop'] != cursor + len(arr):
            raise ValueError('coordinate schema/chunk coverage mismatch')
        good = (arr['x'] >= 0) & (arr['y'] >= 0)
        if np.any((arr['x'] >= 0) != (arr['y'] >= 0)) or np.any(arr['x'][good] >= data['nuclei'].shape[1]) or np.any(arr['y'][good] >= data['nuclei'].shape[0]) or np.any(arr['gene'] < 0) or np.any(arr['gene'] >= len(data['genes'])):
            raise ValueError('coordinate bounds/gene mismatch')
        cursor += len(arr)
        yield i,rec,arr
    if cursor != data.get('expected_molecules', ELIGIBLE):
        raise ValueError('coordinate eligible denominator differs')


def sparse_save(path, matrix):
    atomic(path, lambda f: sparse.save_npz(f, matrix))


def _chunk(out, kind, i, deps):
    folder = Path(out)/kind
    folder.mkdir(parents=True,exist_ok=True)
    path = folder/f'{i:06d}.npz'
    receipt = path.with_suffix('.json')
    saved = resume_chunk(receipt,deps)
    return path,receipt,saved


def build_domain(data, out, deps, callback=None):
    out=Path(out)
    complete=out/'complete.json'
    if complete.exists():
        check_receipt(complete,deps)
        z=np.load(out/'domain.npz')
        return {k:z[k] for k in z.files},sparse.load_npz(out/'expression.npz'),read(out/'domain.json')
    out.mkdir(parents=True,exist_ok=True)
    baseline,nuclei,sup=data['baseline'],data['nuclei'],data['support']
    band=interface_band(baseline)
    width=baseline.shape[1]
    query=[]
    for i,rec,a in coordinates(data):
        cd=dict(deps,coordinate=rec['sha256'],kind='query')
        p,r,saved=_chunk(out,'query',i,cd)
        if saved is None:
            good=(a['x']>=0)&(a['y']>=0)
            y,x=a['y'][good],a['x'][good]
            keep=band[y,x]&(baseline[y,x]>0)&(nuclei[y,x]==0)
            pixels=np.unique(y[keep].astype(np.int64)*width+x[keep])
            atomic_npz(p,pixels=pixels)
            chunk_receipt(r,cd,[p],molecules=len(a))
        query.append(np.load(p)['pixels'])
        progress(callback,'GT_query',i+1,len(data['coordinates']))
    pixels=np.unique(np.concatenate(query)) if query else np.empty(0,np.int64)
    atomic_npy(out/'query_pixels.npy',pixels)
    lookup=np.full(len(data['genes']),-1,int)
    lookup[data['gene_indices']]=np.arange(len(data['gene_indices']))
    ng=len(data['gene_indices'])
    allkeys=[];allcounts=[];full_pixel_counts=np.zeros(len(pixels),np.int64)
    for i,rec,a in coordinates(data):
        cd=dict(deps,coordinate=rec['sha256'],kind='expression',pixels=identity(out/'query_pixels.npy')['sha256'])
        p,r,saved=_chunk(out,'expression_chunks',i,cd)
        if saved is None:
            ix=np.flatnonzero((a['x']>=0)&(a['y']>=0))
            flat=a['y'][ix].astype(np.int64)*width+a['x'][ix]
            loc=np.searchsorted(pixels,flat)
            keep=loc<len(pixels)
            ix,flat,loc=ix[keep],flat[keep],loc[keep]
            if len(pixels):
                keep=pixels[loc]==flat
                ix,loc=ix[keep],loc[keep]
            else:
                ix,loc=ix[:0],loc[:0]
            pk,pn=np.unique(loc,return_counts=True)
            gg=lookup[a['gene'][ix]]
            ok=gg>=0
            keys,vals=np.unique(loc[ok]*ng+gg[ok],return_counts=True)
            atomic_npz(p,keys=keys,counts=vals,pixel_keys=pk,pixel_counts=pn)
            chunk_receipt(r,cd,[p])
        z=np.load(p)
        allkeys.append(z['keys']);allcounts.append(z['counts'])
        full_pixel_counts[z['pixel_keys']]+=z['pixel_counts']
        progress(callback,'GT_expression',i+1,len(data['coordinates']))
    keys=np.concatenate(allkeys) if allkeys else np.empty(0,np.int64)
    vals=np.concatenate(allcounts) if allcounts else np.empty(0,np.int64)
    expression=sparse.coo_matrix((vals,(keys//ng,keys%ng)),shape=(len(pixels),ng),dtype=np.int64).tocsr()
    support_p=[];support_r=[];support_stats=Counter()
    for start in range(0,len(data['ids']),1000):
        stop=min(start+1000,len(data['ids']))
        cd=dict(deps,kind='support',start=start,stop=stop,pixels=identity(out/'query_pixels.npy')['sha256'])
        p,r,saved=_chunk(out,'support_chunks',start//1000,cd)
        if saved is None:
            pp=[];rr=[];oob=0;fallback=0
            for row in range(start,stop):
                y,x,_,fb=sup.support(int(data['ids'][row]))
                inside=(y>=0)&(x>=0)&(y<baseline.shape[0])&(x<width)
                oob+=int((~inside).sum());fallback+=int(fb)
                flat=y[inside].astype(np.int64)*width+x[inside]
                where=np.searchsorted(pixels,flat)
                keep=where<len(pixels);where,flat=where[keep],flat[keep]
                if len(pixels):
                    where=where[pixels[where]==flat]
                if len(where):
                    pp.append(where);rr.append(np.full(len(where),row,np.int32))
            atomic_npz(p,pixel_index=np.concatenate(pp) if pp else np.empty(0,np.int64),
                       rows=np.concatenate(rr) if rr else np.empty(0,np.int32))
            saved=chunk_receipt(r,cd,[p],stats=dict(support_oob_pixels=oob,ellipse_fallbacks=fallback))
        z=np.load(p);support_p.append(z['pixel_index']);support_r.append(z['rows']);support_stats.update(saved['stats'])
        progress(callback,'GT_complete_nuclear_support',stop,len(data['ids']))
    pi=np.concatenate(support_p) if support_p else np.empty(0,np.int64);rows=np.concatenate(support_r) if support_r else np.empty(0,np.int32)
    order=np.argsort(pi,kind='stable');pi,rows=pi[order],rows[order]
    offsets=np.r_[0,np.cumsum(np.bincount(pi,minlength=len(pixels)),dtype=np.int64)]
    support_stats['maximum_K']=int(np.diff(offsets).max(initial=0))
    positions=[];candidate_rows=[];lengths=[];reasons=Counter()
    for start in range(0,len(pixels),5000):
        stop=min(start+5000,len(pixels))
        cd=dict(deps,kind='domain',start=start,stop=stop)
        p,r,saved=_chunk(out,'domain_chunks',start//5000,cd)
        if saved is None:
            pos=[];rr=[];nn=[];why=Counter()
            for q in range(start,stop):
                flat=int(pixels[q]);y,x=divmod(flat,width);owner=int(baseline[y,x])
                support=rows[offsets[q]:offsets[q+1]]
                reason=None
                if not owner:reason='background'
                elif nuclei[y,x]:reason='nuclear'
                elif not band[y,x]:reason='outside_interface_band'
                elif len(support)<2:reason='fewer_than_two_complete_supports'
                else:
                    adjacent=np.unique(baseline[max(0,y-1):min(baseline.shape[0],y+2),max(0,x-1):min(width,x+2)])
                    cand=support[np.isin(data['ids'][support],adjacent[adjacent>0])]
                    if owner not in data['ids'][cand]:reason='owner_not_supported'
                    elif len(cand)<2:reason='fewer_than_two_adjacent_supported_candidates'
                    elif not data['eligible'][cand].all():reason='candidate_not_A1_evaluable'
                    elif np.any(data['donor'][cand]<0):reason='candidate_missing_full_count_donor'
                    elif expression.getrow(q).sum()==0:reason='no_intersection_gene_expression'
                if reason:why[reason]+=1
                else:pos.append(q);rr.extend(cand);nn.append(len(cand))
            atomic_npz(p,positions=np.asarray(pos,np.int64),rows=np.asarray(rr,np.int32),lengths=np.asarray(nn,np.int64))
            saved=chunk_receipt(r,cd,[p],reasons=dict(why))
        z=np.load(p);positions.append(z['positions']);candidate_rows.append(z['rows']);lengths.append(z['lengths']);reasons.update(saved['reasons'])
        progress(callback,'GT_parent_legal_domain',stop,len(pixels))
    pos=np.concatenate(positions) if positions else np.empty(0,np.int64)
    rows=np.concatenate(candidate_rows) if candidate_rows else np.empty(0,np.int32)
    lengths=np.concatenate(lengths) if lengths else np.empty(0,np.int64)
    offsets=np.r_[0,np.cumsum(lengths,dtype=np.int64)]
    query_count=len(pixels);pixels=pixels[pos];expression=expression[pos];full_pixel_counts=full_pixel_counts[pos]
    geometry_parts=[]
    for start in range(0,len(pixels),5000):
        stop=min(start+5000,len(pixels));lo,hi=offsets[start],offsets[stop]
        cd=dict(deps,kind='geometry',start=start,stop=stop)
        p,r,saved=_chunk(out,'geometry_chunks',start//5000,cd)
        if saved is None:
            geometry=sup.candidate_geometry(pixels[start:stop],offsets[start:stop+1]-lo,rows[lo:hi])
            atomic_npz(p,geometry=geometry)
            chunk_receipt(r,cd,[p])
        geometry_parts.append(np.load(p)['geometry'])
        progress(callback,'GT_candidate_geometry',stop,len(pixels))
    geometry=np.concatenate(geometry_parts) if geometry_parts else np.empty(0,float)
    domain=dict(pixels=pixels,offsets=offsets,candidate_rows=rows,candidate_ids=data['ids'][rows],
                geometry=geometry,full_count_donors=data['donor'],full_count_bins=data['bins'],eligible=data['eligible'],
                ids=data['ids'],full_pixel_molecules=full_pixel_counts)
    atomic_npz(out/'domain.npz',**domain)
    sparse_save(out/'expression.npz',expression)
    info=dict(parent=data.get('parent','fixture'),query_pixels=query_count,eligible_pixels=len(pixels),
              eligible_G_molecules=int(expression.sum()),eligible_all_gene_molecules=int(full_pixel_counts.sum()),
              eligible_cells=int(np.unique(rows).size),candidate_edges=len(rows),support_stats=dict(support_stats),
              abstentions=dict(reasons),donor_seed=SHUFFLE_SEEDS[0],parameters=PARAMETERS,
              applicable_types=APPLICABLE_TYPES,score_uses_cell_specific_q=False,
              domain='parent positive-positive 8-neighbour interface expanded 2 um; nuclear exterior; complete support; current-parent adjacency; frozen A1/type/donor eligibility')
    atomic_json(out/'domain.json',info)
    write_receipt(complete,deps,inventory(out),**info)
    return domain,expression,info


def score_domain(data,domain,expression,out,deps,callback=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():
        check_receipt(out/'complete.json',deps)
        return np.load(out/'proposals.npy'),read(out/'summary.json')
    allproposals=[];allscores=[];below=0
    pixels,offsets,rows=domain['pixels'],domain['offsets'],domain['candidate_rows']
    for start in range(0,len(pixels),5000):
        stop=min(start+5000,len(pixels));lo,hi=offsets[start],offsets[stop]
        cd=dict(deps,start=start,stop=stop)
        p,r,saved=_chunk(out,'chunks',start//5000,cd)
        if saved is None:
            proposals,scores,under=score_proposals(data['baseline'],pixels[start:stop],offsets[start:stop+1]-lo,
                                                  rows[lo:hi],domain['geometry'][lo:hi],expression[start:stop],data['ids'],data['prior'])
            atomic_npz(p,proposals=proposals,scores=scores)
            saved=chunk_receipt(r,cd,[p],below=under)
        z=np.load(p);allproposals.append(z['proposals']);allscores.append(z['scores']);below+=saved['below']
        progress(callback,'GT_scores',stop,len(pixels))
    if allproposals:
        proposals=np.concatenate(allproposals);scores=np.concatenate(allscores)
        order=np.lexsort((proposals['target'],proposals['x'],proposals['y'],-proposals['margin']))
        proposals=proposals[order]
    else:
        proposals,scores,below=score_proposals(data['baseline'],pixels,offsets,rows,domain['geometry'],expression,data['ids'],data['prior'])
    atomic_npy(out/'proposals.npy',proposals);atomic_npy(out/'candidate_scores.npy',scores)
    summary=dict(raw_proposals=len(proposals),below_margin_or_current_best=int(below),beta=1.0,margin=float(np.log(2)),
                 tie_rule='current owner wins exact ties, else smallest ID; global order -margin,y,x,target',score='geometry + frozen type mean log likelihood')
    atomic_json(out/'summary.json',summary)
    write_receipt(out/'complete.json',deps,inventory(out),**summary)
    return proposals,summary


def apply_gt(baseline,nuclei,proposals,out,deps,callback=None,stop_after=None,oracle=False):
    """Recover exact sorted proposal cursor, mask, and each rejection reason."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    statepath=out/'state.json';mask=baseline.copy();cursor=0;decisions=np.zeros(len(proposals),np.uint8)
    if statepath.exists():
        state=check_receipt(statepath,deps)
        mask=np.load(state['mask']['path']);decisions=np.load(state['decisions']['path']);cursor=state['cursor']
        if mask.shape!=baseline.shape or len(decisions)!=len(proposals) or not 0<=cursor<=len(proposals) or np.any(decisions[:cursor]==0) or np.any(decisions[cursor:]!=0):
            raise ValueError('GT committed recovery cursor/shape/decisions invalid')
    topology=CompleteCellTopology(baseline,nuclei,proposals)
    initial=cursor
    def checkpoint():
        prior=read(statepath) if statepath.exists() else None
        mp=out/f'mask_{cursor:09d}.npy';dp=out/f'decisions_{cursor:09d}.npy'
        atomic_npy(mp,mask);atomic_npy(dp,decisions)
        mi,di=identity(mp),identity(dp)
        atomic_json(statepath,dict(status='succeeded',dependencies=deps,cursor=cursor,mask=mi,decisions=di,outputs={'mask':mi,'decisions':di}))
        committed(statepath)
        if prior:
            for key in ('mask','decisions'):
                obj=prior[key];check_identity(obj)
                if Path(obj['path']) not in (mp,dp):Path(obj['path']).unlink()
        progress(callback,'GT_apply_topology',cursor,len(proposals))
    while cursor<len(proposals):
        p=proposals[cursor];y,x,source,target=map(int,(p['y'],p['x'],p['source'],p['target']))
        ok,reason=topology.check(mask,y,x,source,target)
        if oracle and (ok,reason)!=topology_oracle(mask,nuclei,y,x,source,target):
            raise ValueError('complete-cell topology differs from full-image oracle')
        decisions[cursor]=REASONS.index(reason)
        if ok:mask[y,x]=target
        cursor+=1
        if cursor%1000==0 or cursor==len(proposals) or (stop_after is not None and cursor-initial>=stop_after):
            checkpoint()
            if stop_after is not None and cursor-initial>=stop_after and cursor<len(proposals):
                return None,dict(cursor=cursor,complete=False)
    if not statepath.exists():checkpoint()
    summary=dict(cursor=cursor,complete=True,accepted=int(np.sum(decisions==1)),rejected=int(np.sum(decisions>1)),
                 rejection_counts={name:int(np.sum(decisions==i)) for i,name in enumerate(REASONS) if i})
    return mask,summary


def validate_topology(baseline,mask,nuclei,ids,proposals,out,deps,callback=None):
    if not np.array_equal(mask>0,baseline>0) or not np.array_equal(mask[nuclei>0],nuclei[nuclei>0]):
        raise ValueError('GT final foreground/nuclear owner invariant failed')
    actual=np.unique(mask);actual=actual[actual>0]
    if not np.array_equal(actual,ids):raise ValueError('GT final ID invariant failed')
    changed=mask!=baseline
    if np.any(changed&((mask==0)|(baseline==0))):raise ValueError('GT changed background ownership')
    topology=CompleteCellTopology(baseline,nuclei,proposals)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    chunks=[]
    for start in range(0,len(ids),1000):
        stop=min(start+1000,len(ids));path=out/f'cells_{start:06d}.csv';receipt=path.with_suffix('.json')
        cd=dict(deps,start=start,stop=stop)
        saved=resume_chunk(receipt,cd)
        if saved is None:
            rows=[]
            for cid in ids[start:stop]:
                box=topology.bounds[int(cid)]
                a,b,n=baseline[box]==cid,mask[box]==cid,nuclei[box]==cid
                ar,na=rooted(a,n);br,nb=rooted(b,n)
                lost=int(np.count_nonzero(ar&b&~br));new_unrooted=int(np.count_nonzero(b&~a&~br))
                if nb>na or lost or new_unrooted:raise ValueError('GT final complete-cell topology invariant failed')
                rows.append(dict(cell_id=int(cid),components_before=na,components_after=nb,retained_nuclear_connection_lost=lost,new_unrooted_pixels=new_unrooted))
            atomic_csv(path,pd.DataFrame(rows));chunk_receipt(receipt,cd,[path])
        chunks.append(path)
        progress(callback,'GT_final_topology',stop,len(ids))
    result=dict(n_cells=len(ids),changed_pixels=int(changed.sum()),foreground_equal=True,nuclear_owner_exact=True,
                all_ids_equal=True,positive_to_positive_only=True,donor_components_do_not_increase=True,
                nuclear_connections_preserved=True,new_recipient_pixels_nucleus_connected=True)
    atomic_json(out/'invariants.json',result)
    return result


def refine(cfg,manifest,endpoint,callback=None):
    out=within(cfg['run_root'],'endpoints',endpoint,'gt')
    deps=dependencies(cfg,endpoint+'_GT')
    if (out/'complete.json').exists():return check_receipt(out/'complete.json',deps,cfg['run_root'])
    data=load_data(cfg,manifest,endpoint)
    out.mkdir(parents=True,exist_ok=True)
    domain,expression,domain_info=build_domain(data,out/'domain',dict(deps,parent=data['spec']['parent_evidence']),callback)
    proposals,score_info=score_domain(data,domain,expression,out/'scores',dict(deps,domain=identity(out/'domain/complete.json')),callback)
    mask,apply_info=apply_gt(data['baseline'],data['nuclei'],proposals,out/'recovery',dict(deps,proposals=identity(out/'scores/proposals.npy')),callback)
    final=within(cfg['run_root'],'endpoints',endpoint,'final.tif')
    if not final.exists():atomic_tiff(final,mask.astype(np.uint32))
    elif not np.array_equal(tifffile.imread(final),mask):raise ValueError('completed final mask differs from recovery')
    validation=validate_topology(data['baseline'],mask,data['nuclei'],data['ids'],proposals,out/'validation',dict(deps,final=identity(final)),callback)
    decisions=np.load(read(out/'recovery/state.json')['decisions']['path'])
    transfers=pd.DataFrame.from_records(proposals)
    transfers['decision']=[REASONS[int(v)] for v in decisions]
    transfers['accepted']=decisions==1
    if len(transfers):
        pp=np.searchsorted(domain['pixels'],transfers.y.to_numpy(np.int64)*mask.shape[1]+transfers.x.to_numpy(np.int64))
        transfers['G_molecules']=np.asarray(expression.sum(1)).ravel()[pp]
        transfers['all_gene_molecules']=domain['full_pixel_molecules'][pp]
    else:
        transfers['G_molecules']=pd.Series(dtype=np.int64);transfers['all_gene_molecules']=pd.Series(dtype=np.int64)
    atomic_csv(out/'transfers.csv',transfers)
    summary=dict(endpoint=endpoint,parent=data['parent'],domain=domain_info,scoring=score_info,application=apply_info,invariants=validation,
                 final=identity(final),A1_inputs={k:manifest['files'][str(Path(v).resolve())] for k,v in manifest['a1'].items()},
                 type_reference=manifest['files'][manifest['reference_region']],coordinate_cache=manifest['files'][manifest['coordinate_cache']],
                 training='parent checkpoint provenance only; no optimizer or new training',GT='Geometry+Type, not ground truth')
    atomic_json(out/'summary.json',summary)
    return write_receipt(out/'complete.json',deps,dict(inventory(out),final=identity(final)),**summary)
