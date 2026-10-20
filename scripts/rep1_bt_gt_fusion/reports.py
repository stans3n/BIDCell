"""Prespecified dual-F1 populations, full B/T comparisons, and local RNA costs."""
from __future__ import annotations
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from . import EXISTING, NEW, PARENTS, comparisons
from .common import (atomic_csv, atomic_json, atomic_npz, check_receipt, chunk_receipt,
                     dependencies, digest, identity, id_hash, inventory, progress, read,
                     require_manual_resume, resume_chunk, within, write_receipt)
from .bt_curves import PANELS, MARKERS, checked_ids, write_curves
from .inputs import matrix_arrays

METRICS=('positive_F1','negative_F1','positive_Precision','positive_Recall','negative_Precision','negative_Recall')


def paired_f1(cfg,specs,pop_root,out,methods,callback=None,manual_resume=None):
    """Four-way finite population per metric: both methods × both label profiles."""
    if any(m in NEW for m in methods):require_manual_resume(cfg,manual_resume)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    deps=dependencies(cfg,'PAIRED_F1_'+str(len(methods)))
    if (out/'complete.json').exists():return check_receipt(out/'complete.json',deps,cfg['run_root'])
    pop_root=Path(pop_root)
    tables={m:{p:checked_ids(pd.read_csv(specs[m][key]),m+' '+p) for p,key in [('sc','sc_scores'),('fixed','fixed_scores')]} for m in methods}
    fixed_ids=set(tables[methods[0]]['fixed'].index)
    q={m:set(tables[m]['sc'].index) for m in methods}
    summaries=[]
    pairs=comparisons(methods)
    for i,(a,b) in enumerate(pairs):
        dest=out/(a+'__'+b);dest.mkdir(exist_ok=True)
        common=np.asarray(sorted(fixed_ids&q[a]&q[b]),np.int64)
        atomic_csv(dest/'F_intersect_pairwise_QC.csv',pd.DataFrame({'cell_id':common}))
        for metric in METRICS:
            values={p+'_'+m:tables[m][p][metric].reindex(common).to_numpy(float) for m in (a,b) for p in ('sc','fixed')}
            frame=pd.DataFrame(values,index=pd.Index(common,name='cell_id'))
            finite=np.isfinite(frame.to_numpy()).all(1)
            frame['four_way_finite']=finite
            for column in values:frame[column+'_nonfinite']=~np.isfinite(frame[column])
            frame['sc_delta']=frame['sc_'+a]-frame['sc_'+b]
            frame['fixed_delta']=frame['fixed_'+a]-frame['fixed_'+b]
            atomic_csv(dest/(metric+'_cells.csv'),frame,index=True)
            selected=common[finite]
            atomic_csv(dest/(metric+'_four_way_finite_ids.csv'),pd.DataFrame({'cell_id':selected}))
            for p in ('sc','fixed'):
                delta=frame.loc[finite,p+'_delta']
                summaries.append(dict(candidate=a,control=b,metric=metric,profile=p,
                                      population='F_intersect_Qab_then_two_methods_times_two_profiles_finite',
                                      n_before_finite=len(common),n_finite=len(selected),n_excluded=len(common)-len(selected),
                                      ids_sha256=id_hash(selected),mean_candidate=float(frame.loc[finite,p+'_'+a].mean()) if len(selected) else None,
                                      mean_control=float(frame.loc[finite,p+'_'+b].mean()) if len(selected) else None,
                                      mean_delta=float(delta.mean()) if len(selected) else None,
                                      direction='candidate minus control; positive F1 higher, negative F1 lower',
                                      reason='' if len(selected) else 'no_four_way_finite_IDs'))
        progress(callback,'paired_dual_F1',i+1,len(pairs),candidate=a,control=b)
    all_common=np.asarray(sorted(fixed_ids&set.intersection(*(q[m] for m in methods))),np.int64)
    for metric in METRICS:
        finite=np.ones(len(all_common),bool)
        reasons=pd.DataFrame(index=pd.Index(all_common,name='cell_id'))
        for m in methods:
            for p in ('sc','fixed'):
                ok=np.isfinite(tables[m][p][metric].reindex(all_common).to_numpy(float))
                finite&=ok;reasons[m+'_'+p+'_finite']=ok
        selected=all_common[finite]
        atomic_csv(out/f'all{len(methods)}_{metric}_common_finite_ids.csv',pd.DataFrame({'cell_id':selected}))
        atomic_csv(out/f'all{len(methods)}_{metric}_finite_exclusions.csv',reasons,index=True)
        for m in methods:
            for p in ('sc','fixed'):
                vals=tables[m][p][metric].reindex(selected)
                summaries.append(dict(candidate=m,control='',metric=metric,profile=p,
                                      population=f'F_intersect_Q{len(methods)}_then_all_methods_and_both_profiles_finite',
                                      n_before_finite=len(all_common),n_finite=len(selected),n_excluded=len(all_common)-len(selected),
                                      ids_sha256=id_hash(selected),mean_candidate=float(vals.mean()) if len(vals) else None,
                                      mean_control=None,mean_delta=None,reason='' if len(vals) else 'no_all_method_finite_IDs'))
    atomic_csv(out/'paired_summary.csv',pd.DataFrame(summaries))
    return write_receipt(out/'complete.json',deps,inventory(out),methods=list(methods),pairwise_four_way_finite=True,
                         original_ownQC_and_fixed_full_panel_remain_primary=True)


def block_interval(delta,xy,out,deps,callback=None,side=1000,replicates=2000,seed=20261013):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    delta=np.asarray(delta,float);xy=np.asarray(xy,float)
    if len(delta)!=len(xy) or not np.isfinite(delta).all() or not np.isfinite(xy).all():raise ValueError('BT bootstrap paired population invalid')
    blocks,index=np.unique(np.floor(xy/side).astype(np.int64),axis=0,return_inverse=True)
    n=np.bincount(index,minlength=len(blocks));sums=np.bincount(index,weights=delta,minlength=len(blocks))
    atomic_npz(out/'blocks.npz',xy=blocks,n=n,delta_sum=sums)
    base=dict(side_um=side,origin_um=[0,0],seed=seed,replicates=replicates,n_cells=len(delta),n_blocks=len(blocks),
              mean_delta=float(delta.mean()) if len(delta) else None,paired_weights=True,minimum_blocks=10)
    if len(blocks)<10:
        return dict(base,ci95=[None,None],valid_replicates=0,status='NA_insufficient_spatial_blocks')
    statepath=out/'state.json';samples=np.full(replicates,np.nan);cursor=0
    rng=np.random.Generator(np.random.PCG64(seed))
    if statepath.exists():
        state=check_receipt(statepath,deps);cursor=state['cursor']
        samples=np.load(state['samples']['path'])['samples'];rng.bit_generator.state=state['rng_state']
        if not 0<=cursor<=replicates or samples.shape!=(replicates,):raise ValueError('BT bootstrap cursor invalid')
    while cursor<replicates:
        stop=min(cursor+100,replicates)
        for j in range(cursor,stop):
            chosen=rng.integers(0,len(blocks),size=len(blocks));weights=np.bincount(chosen,minlength=len(blocks))
            den=int(weights@n);samples[j]=float(weights@sums/den) if den else np.nan
        cursor=stop;p=out/f'samples_{cursor:04d}.npz';atomic_npz(p,samples=samples)
        item=identity(p)
        atomic_json(statepath,dict(status='succeeded',dependencies=deps,cursor=cursor,rng_state=rng.bit_generator.state,
                                   samples=item,outputs={'samples':item}))
        for old in out.glob('samples_*.npz'):
            if old!=p:old.unlink()
        progress(callback,'BT_spatial_bootstrap',cursor,replicates,side_um=side)
    valid=samples[np.isfinite(samples)]
    return dict(base,ci95=np.quantile(valid,[.025,.975]).tolist() if len(valid) else [None,None],
                valid_replicates=len(valid),status='computed' if len(valid) else 'NA_no_valid_resamples')


def bt_comparisons(cfg,frames,pop,out,grant,callback=None):
    require_manual_resume(cfg,grant)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    deps=dependencies(cfg,'BT8_COMPARISONS')
    if (out/'complete.json').exists():return check_receipt(out/'complete.json',deps,cfg['run_root'])
    nuclear=checked_ids(pd.read_csv(Path(pop)/'nuclear_reference.csv'),'nuclear reference')
    fixed=set(pd.read_csv(Path(pop)/'fixed_commonID6.csv').cell_id)
    result=[]
    for a,b in comparisons():
        q=fixed&set(frames[a].index[frames[a].own_qc])&set(frames[b].index[frames[b].own_qc])
        for panel in PANELS:
            distances=checked_ids(pd.read_csv(Path(pop)/(panel['direction']+'_frozen_distances.csv')),'frozen distances')
            ids=np.asarray(sorted(q&set(distances.index[distances.distance_um.le(40)])),np.int64)
            xy=nuclear.loc[ids,['x_um','y_um']].to_numpy(float)
            for marker in panel['markers']:
                delta=(frames[a].loc[ids,marker].to_numpy()>0).astype(float)-(frames[b].loc[ids,marker].to_numpy()>0).astype(float)
                target=out/(a+'__'+b)/marker;target.mkdir(parents=True,exist_ok=True)
                atomic_csv(target/'paired_expression.csv',pd.DataFrame(dict(cell_id=ids,candidate_positive=frames[a].loc[ids,marker].to_numpy()>0,
                           control_positive=frames[b].loc[ids,marker].to_numpy()>0,delta=delta)))
                for side in (1000,500,2000):
                    dd=dict(deps,candidate=a,control=b,marker=marker,side=side,population=id_hash(ids),paired_expression=identity(target/'paired_expression.csv'))
                    ci=block_interval(delta,xy,target/str(side),dd,callback,side=side)
                    result.append(dict(candidate=a,control=b,direction=panel['direction'],marker=marker,window_um='0-40 inclusive',
                                       profile='fixed_pairwise_commonQC_nuclear_ref',population_ids_sha256=id_hash(ids),**ci))
    atomic_csv(out/'paired_0_40um_summary.csv',pd.DataFrame(result))
    return write_receipt(out/'complete.json',deps,inventory(out),comparisons=[list(x) for x in comparisons()],
                         six_markers_prespecified=True,no_significance_selection_gate=True)


def local_transfers(cfg,manifest,spec,name,out,grant,callback=None):
    require_manual_resume(cfg,grant)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    deps=dependencies(cfg,name+'_LOCAL_RNA')
    if (out/'complete.json').exists():return check_receipt(out/'complete.json',deps,cfg['run_root'])
    parent=manifest['endpoints'][PARENTS[name]]
    ids,genes,parent_counts=matrix_arrays(parent);_,_,own=matrix_arrays(spec)
    meta=np.load(manifest['metadata']);ref=read(manifest['reference_region']);types=meta['type_index'].astype(int)
    type_names=['Unknown']+list(ref['type_names']);t=types+1;nt=len(type_names);ng=len(genes)
    if not np.array_equal(meta['ids'],ids):raise ValueError('local RNA nuclear metadata ID mismatch')
    xy=meta['centroid_xy'];spatial=np.floor(xy/1000).astype(int)
    coverage=pd.read_csv(manifest['a1']['cell_coverage.csv']).set_index('cell_id').loc[ids]
    density_names=sorted(coverage.density_group.astype(str).unique())
    density=pd.Index(density_names).get_indexer(coverage.density_group.astype(str))
    pc=read(Path(parent['matrix'])/'complete.json');mc=read(Path(spec['matrix'])/'complete.json')
    incoming=np.zeros(len(ids),np.int64);outgoing=incoming.copy();edge_counts=Counter();stratum_edges=Counter()
    for i,(pr,mr) in enumerate(zip_longest_local(pc['receipts'],mc['receipts'])):
        if pr is None or mr is None or pr['rows']!=mr['rows'] or pr['coordinate_sha256']!=mr['coordinate_sha256']:
            raise ValueError('parent/GT molecule row association differs')
        part=out/f'chunk_{i:04d}.npz';rp=part.with_suffix('.json');cd=dict(deps,chunk=i,parent_owner=pr['sha256'],new_owner=mr['sha256'])
        saved=resume_chunk(rp,cd)
        if saved is None:
            a=np.load(Path(parent['matrix'])/pr['owner_file']);b=np.load(Path(spec['matrix'])/mr['owner_file']);coord=np.load(mr['coordinate_file'],mmap_mode='r')
            change=a!=b
            if np.any(change&((a==0)|(b==0))):raise ValueError('local transfer capture invariant broken')
            ar=np.searchsorted(ids,a[change]);br=np.searchsorted(ids,b[change]);gene=coord['gene'][change]
            outkeys,outvals=np.unique(ar,return_counts=True);inkeys,invals=np.unique(br,return_counts=True)
            edge,ev=np.unique((t[ar]*nt+t[br])*ng+gene,return_counts=True)
            detailed=np.column_stack((t[ar],t[br],gene,spatial[ar],spatial[br],density[ar],density[br]))
            se,sv=np.unique(detailed,axis=0,return_counts=True)
            atomic_npz(part,outkeys=outkeys,outvals=outvals,inkeys=inkeys,invals=invals,edges=edge,edge_counts=ev,
                       stratum_edges=se,stratum_edge_counts=sv)
            chunk_receipt(rp,cd,[part])
        z=np.load(part);outgoing[z['outkeys']]+=z['outvals'];incoming[z['inkeys']]+=z['invals'];edge_counts.update(dict(zip(z['edges'].tolist(),z['edge_counts'].tolist())))
        stratum_edges.update({tuple(k):v for k,v in zip(z['stratum_edges'].tolist(),z['stratum_edge_counts'].tolist())})
        progress(callback,'local_RNA_transfer',i+1,len(mc['receipts']),endpoint=name)
    total=parent_counts.sum(1,dtype=np.int64);newtotal=own.sum(1,dtype=np.int64)
    if not np.array_equal(total-outgoing+incoming,newtotal):raise ValueError('local RNA transfer owner ledger does not reconstruct own counts')
    cells=pd.DataFrame(dict(cell_id=ids,type=[type_names[i] for i in t],parent_RNA=total,new_RNA=newtotal,
                           outgoing_RNA=outgoing,incoming_RNA=incoming,retained_original_RNA=total-outgoing,
                           local_retention=np.divide(total-outgoing,total,out=np.full(len(ids),np.nan),where=total>0),
                           block_x=spatial[:,0],block_y=spatial[:,1],density_group=coverage.density_group.to_numpy()))
    atomic_csv(out/'cells.csv',cells)
    strata=[]
    for family,cols in [('type',['type']),('type_spatial',['type','block_x','block_y']),('type_density',['type','density_group'])]:
        for key,part in cells.groupby(cols,dropna=False):
            sums=part[['parent_RNA','new_RNA','outgoing_RNA','incoming_RNA','retained_original_RNA']].sum()
            den=int(sums.parent_RNA);ret=float(sums.retained_original_RNA/den) if den else None
            strata.append(dict(family=family,group=str(key),n_cells=len(part),**sums.to_dict(),local_retention=ret,
                               below_unchanged_0_995_guardrail=ret is not None and ret<.995))
    atomic_csv(out/'strata.csv',pd.DataFrame(strata))
    edges=[]
    for key,value in sorted(edge_counts.items()):
        pair,gene=divmod(key,ng);a,b=divmod(pair,nt)
        edges.append(dict(source_type=type_names[a],target_type=type_names[b],gene=genes[gene],molecules=value))
    atomic_csv(out/'type_to_type_gene_transfers.csv',pd.DataFrame(edges,columns=['source_type','target_type','gene','molecules']))
    detailed=[]
    for (a,b,g,ax,ay,bx,by,ad,bd),value in sorted(stratum_edges.items()):
        detailed.append(dict(source_type=type_names[a],target_type=type_names[b],gene=genes[g],
             source_block_x=ax,source_block_y=ay,target_block_x=bx,target_block_y=by,
             source_density_group=density_names[ad],target_density_group=density_names[bd],molecules=value))
    atomic_csv(out/'type_spatial_density_gene_transfers.csv',pd.DataFrame(detailed,columns=[
        'source_type','target_type','gene','source_block_x','source_block_y','target_block_x','target_block_y',
        'source_density_group','target_density_group','molecules']))
    atomic_json(out/'interpretation.json',dict(endpoint=name,parent=PARENTS[name],capture_loss_pp=0.0,capture_loss_guardrail_pp=.5,
                local_retention_guardrail=.995,global_conservation_does_not_imply_local_RNA_retention=True,
                highlighted_types=['Myofibroblasts','Endothelial','Mono/Macro'],
                historical_GT_alerts={'Myofibroblasts':.962530,'Endothelial':.974211,'Mono/Macro':.977627},
                interpretation='GT reassigns all RNA at accepted pixels using frozen geometry/type likelihood. Below-.995 retention remains a limitation; gain/loss and genes/adjacent types are explicitly recorded. No independent owner truth establishes these transfers as correct or incorrect.'))
    return write_receipt(out/'complete.json',deps,inventory(out),capture_exact=True,guardrails_unchanged=True)


def zip_longest_local(a,b):
    from itertools import zip_longest
    return zip_longest(a,b)


def endpoint_comparison_tables(cfg,manifest,out):
    """All prespecified raw structure/reference/spatial contrasts, with NA reasons."""
    from .publication import read_strings,new_table_row
    from .inputs import TABLE
    columns,old=read_strings(TABLE)
    selected={r['method']:r for r in old if r['method'] in EXISTING}
    selected.update({m:new_table_row(cfg,manifest,m,columns) for m in NEW})
    metrics=['n_segmented_cells','capture_pct','mean_transcripts_per_cell','mean_genes_per_cell',
             'mean_cell_area_um2','median_cell_area_um2','p95_cell_area_um2','disconnected_cells_4',
             'disconnected_cells_8','nuclear_preservation_fraction','new_IDs','scclassify_qc_cells',
             'scclassify_qc_retention_pct','scclassify_unassigned_pct','fixed_population_cells',
             'scclassify_expression_pearson_type_mean','scclassify_detection_pearson_type_mean',
             'scclassify_expression_pearson_types_n','scclassify_detection_pearson_types_n']
    metrics += [c for c in columns if c.startswith('scclassify_spatial_')]
    raw=[];pairs=[]
    def numeric(value):
        try:
            v=float(value)
            return v if np.isfinite(v) else None
        except (ValueError,TypeError):return None
    for m in EXISTING+NEW:
        for metric in metrics:
            literal=selected[m].get(metric,'');value=numeric(literal)
            raw.append(dict(method=m,metric=metric,value=value,raw_string=literal,
                            availability='finite' if value is not None else 'unavailable_in_original_endpoint_summary'))
    for a,b in comparisons():
        for metric in metrics:
            av=numeric(selected[a].get(metric,''));bv=numeric(selected[b].get(metric,''))
            pairs.append(dict(candidate=a,control=b,metric=metric,candidate_value=av,control_value=bv,
                              candidate_minus_control=None if av is None or bv is None else av-bv,
                              population='native full cells for structure; original ownQC for sc/reference/spatial',
                              interpretation='pipeline descriptive contrast; no independent true-boundary accuracy claim',
                              reason='' if av is not None and bv is not None else 'at_least_one_original_scalar_unavailable'))
    atomic_csv(Path(out)/'structure_reference_spatial_raw.csv',pd.DataFrame(raw))
    atomic_csv(Path(out)/'structure_reference_spatial_comparisons.csv',pd.DataFrame(pairs))


def evaluate_eight(cfg,manifest,grant,callback=None):
    require_manual_resume(cfg,grant)
    from .evaluation import new_spec
    out=within(cfg['run_root'],'evaluation8');deps=dependencies(cfg,'EVALUATION8')
    if (out/'complete.json').exists():return check_receipt(out/'complete.json',deps,cfg['run_root'])
    out.mkdir(parents=True,exist_ok=True)
    specs=dict(manifest['endpoints']);specs.update({m:new_spec(cfg,m) for m in NEW})
    old_pop=within(cfg['run_root'],'bt_existing6','populations')
    frames={m:checked_ids(pd.read_csv(old_pop/m/'cells.csv'),m) for m in EXISTING}
    for m in NEW:
        spec=specs[m];ids,genes,counts=matrix_arrays(spec)
        ann=checked_ids(pd.read_csv(Path(spec['evaluation'])/'paper_aligned_v1/annotation.csv'),m+' annotation')
        props=checked_ids(pd.read_csv(Path(spec['evaluation'])/'paper_aligned_v1/own_cell_centroids.csv'),m+' centroid')
        frame=props.copy();frame['own_qc']=frame.index.isin(ann.index);frame['sc_label']=ann.evaluation_label.reindex(frame.index)
        for gene in MARKERS:frame[gene]=counts[:,genes.index(gene)]
        if not np.array_equal(frame.index,ids):raise ValueError('new coordinate/own matrix mapping mismatch')
        frames[m]=frame
    frozen_ids=set(pd.read_csv(old_pop/'fixed_commonID6.csv').cell_id)
    if any(not frozen_ids.issubset(frames[m].index) for m in NEW):raise ValueError('new endpoints do not cover frozen F intersect M6 target domain')
    fixed=checked_ids(pd.read_csv(manifest['fixed_population']),'fixed labels')
    nuclear=checked_ids(pd.read_csv(old_pop/'nuclear_reference.csv'),'nuclear reference')
    common_id=set.intersection(*(set(f.index) for f in frames.values()))
    common_qc=set.intersection(*(set(f.index[f.own_qc]) for f in frames.values()))
    populations={'commonID8':sorted(common_id),'commonQC8':sorted(common_qc),
                 'fixed_commonID8':sorted(common_id&set(fixed.index)),'fixed_commonQC8':sorted(common_qc&set(fixed.index))}
    pop=out/'populations';pop.mkdir(exist_ok=True)
    atomic_csv(pop/'population_availability.csv',pd.DataFrame([
        dict(method=m,n_full_ids=len(frame),n_own_qc=int(frame.own_qc.sum()),
             n_missing_sc_labels_in_QC=int(frame.loc[frame.own_qc,'sc_label'].isna().sum()),
             n_unknown_sc_labels_in_QC=int(frame.loc[frame.own_qc,'sc_label'].eq('Unknown').sum()),
             n_unassigned_sc_labels_in_QC=int(frame.loc[frame.own_qc,'sc_label'].eq('unassigned').sum()),
             n_missing_fixed_labels=int(fixed.evaluation_label.reindex(frame.index).isna().sum()),
             fixed_full_panel_n=len(fixed),own_QC_ids_sha256=id_hash(frame.index[frame.own_qc]))
        for m,frame in frames.items()]))
    qc_summary=[];label_summary=[]
    for a,b in comparisons():
        q=set(frames[a].index[frames[a].own_qc])&set(frames[b].index[frames[b].own_qc])
        populations[f'pairwise_commonQC__{a}__{b}']=sorted(q)
        populations[f'fixed_pairwise_commonQC__{a}__{b}']=sorted(set(fixed.index)&q)
        ids=np.asarray(sorted(set(frames[a].index)&set(frames[b].index)),np.int64)
        transition=pd.DataFrame(dict(cell_id=ids,candidate_qc=frames[a].loc[ids,'own_qc'].to_numpy(),control_qc=frames[b].loc[ids,'own_qc'].to_numpy()))
        transition['status']=np.select([transition.candidate_qc&transition.control_qc,transition.candidate_qc,transition.control_qc],['retained','entered','exited'],default='both_out')
        atomic_csv(pop/f'QC_transition__{a}__{b}.csv',transition)
        for key,n in transition.status.value_counts().items():qc_summary.append(dict(candidate=a,control=b,status=key,n_cells=int(n)))
        pairs=pd.DataFrame(dict(cell_id=sorted(q)))
        pairs['control_label']=frames[b].sc_label.reindex(pairs.cell_id).fillna('__missing__').to_numpy()
        pairs['candidate_label']=frames[a].sc_label.reindex(pairs.cell_id).fillna('__missing__').to_numpy()
        atomic_csv(pop/f'label_transition__{a}__{b}.csv',pairs)
        for (control_label,candidate_label),part in pairs.groupby(['control_label','candidate_label']):
            label_summary.append(dict(candidate=a,control=b,control_label=control_label,candidate_label=candidate_label,n_cells=len(part)))
    atomic_csv(pop/'QC_transition_summary.csv',pd.DataFrame(qc_summary))
    atomic_csv(pop/'label_transition_summary.csv',pd.DataFrame(label_summary))
    for key,ids in populations.items():atomic_csv(pop/(key+'.csv'),pd.DataFrame({'cell_id':ids}))
    atomic_json(pop/'sets.json',{k:dict(n=len(v),ids_sha256=id_hash(v),file=identity(pop/(k+'.csv'))) for k,v in populations.items()})
    for i,(m,frame) in enumerate(frames.items()):
        dest=out/'methods'/m;md=dict(deps,method=m)
        if (dest/'complete.json').exists():check_receipt(dest/'complete.json',md,cfg['run_root']);continue
        profiles=[('official_sc_ownQC_owncentroid',frame.index[frame.own_qc].to_numpy(),False),
                  ('fixed_commonID8_nuclear_ref',np.asarray(populations['fixed_commonID8']),True),
                  ('fixed_commonQC8_nuclear_ref',np.asarray(populations['fixed_commonQC8']),True)]
        for a,b in comparisons():
            if m in (a,b):profiles.append((f'fixed_pairwise_commonQC_nuclear_ref__{a}__{b}',np.asarray(populations[f'fixed_pairwise_commonQC__{a}__{b}']),True))
        for j,(profile,ids,is_fixed) in enumerate(profiles):
            target=dest/profile;pddeps=dict(md,profile=profile,population=id_hash(ids))
            if (target/'complete.json').exists():check_receipt(target/'complete.json',pddeps,cfg['run_root'])
            else:
                write_curves(frame,target,profile,ids,nuclear if is_fixed else frame,old_pop if is_fixed else None)
                write_receipt(target/'complete.json',pddeps,inventory(target),profile=profile)
            progress(callback,'BT8_profiles',j+1,len(profiles),endpoint=m,profile=profile)
        write_receipt(dest/'complete.json',md,inventory(dest),method=m)
        progress(callback,'BT8_endpoints',i+1,len(frames),endpoint=m)
    paired_f1(cfg,specs,pop,out/'paired_f1',EXISTING+NEW,callback,grant)
    bt_comparisons(cfg,frames,old_pop,out/'bt_comparisons',grant,callback)
    for m in NEW:local_transfers(cfg,manifest,specs[m],m,out/'local_RNA'/m,grant,callback)
    d0=np.load(within(cfg['run_root'],'endpoints',NEW[0],'gt/domain/domain.npz'))['pixels']
    d1=np.load(within(cfg['run_root'],'endpoints',NEW[1],'gt/domain/domain.npz'))['pixels']
    common_pixels=np.intersect1d(d0,d1);atomic_npz(out/'common_eligible_pixels.npz',pixels=common_pixels)
    molecule_files=[];total_common=0
    shape=read(within(cfg['run_root'],'endpoints',NEW[0],'matrix/generated.json'))
    import tifffile
    width=tifffile.imread(specs[NEW[0]]['final']).shape[1]
    for i,rec in enumerate(manifest['coordinates']):
        p=out/f'common_eligible_molecules_{i:04d}.npz';cd=dict(deps,kind='common_eligible',coordinate=rec['sha256']);rp=p.with_suffix('.json')
        saved=resume_chunk(rp,cd)
        if saved is None:
            a=np.load(rec['path'],mmap_mode='r');ok=(a['x']>=0)&(a['y']>=0)
            flat=a['y'][ok].astype(np.int64)*width+a['x'][ok];selected=np.isin(flat,common_pixels)
            tids=a['transcript_id'][ok][selected];atomic_npz(p,transcript_id=tids)
            saved=chunk_receipt(rp,cd,[p],molecules=len(tids))
        total_common+=saved['molecules'];molecule_files.append(identity(p));progress(callback,'common_eligible_molecule_coverage',i+1,len(manifest['coordinates']))
    atomic_json(out/'common_eligible_coverage.json',dict(common_pixels=len(common_pixels),common_original_molecules=total_common,
                parent_domains=[len(d0),len(d1)],molecule_sets=molecule_files,
                interpretation='coverage diagnostic only; pipeline comparisons use each actual legal domain; no same-domain pure interaction claim'))
    summary=[]
    for m,spec in specs.items():
        for profile,key in [('scclassify','sc_scores'),('fixed','fixed_scores')]:
            table=pd.read_csv(spec[key])
            for metric in METRICS:
                v=table[metric].to_numpy(float);finite=np.isfinite(v)
                summary.append(dict(method=m,profile=profile,metric=metric,n_population=len(v),n_finite=int(finite.sum()),
                                    n_unavailable=int((~finite).sum()),mean=float(v[finite].mean()) if finite.any() else None))
    atomic_csv(out/'dual_F1_raw_summary.csv',pd.DataFrame(summary))
    endpoint_comparison_tables(cfg,manifest,out)
    atomic_json(out/'limitations.json',dict(B0='historical unmatched performance reference; not a paired mechanism control',
               BASE_MATCH='paired mechanism control',GT='Geometry+Type with retained A1/type/donor eligibility, not ground truth',
               dual_F1='fixed and scClassify positive/negative metrics are parallel; primary raw populations retained',
               local_RNA='unchanged 0.5 pp capture and .995 retention guardrails; unexplained transfer loss remains a limitation',
               boundary_truth='Rep1 DAPI/vendor boundaries do not supply independent true owner or physical boundary accuracy',
               outcomes='negative/no-op results remain registered; no required winner and no multimetric selection score'))
    return write_receipt(out/'complete.json',deps,inventory(out),methods=list(EXISTING+NEW),comparisons=[list(x) for x in comparisons()],
                         existing_full_evaluations_reused=True,new_full_evaluations=2,fixed_target_pool_preserved=True)
