"""Frozen evaluator, distinct anchor consumers, and Rep2-only numerical tables."""
from pathlib import Path
import numpy as np
import pandas as pd
from .common import *

def annotation(name):
    from scripts.rep1_triple_simple_boundary.evaluation import source_runtime,run_r
    from scripts.rep1_triple_simple_boundary.inputs import matrix_arrays
    c=config();s=spec(name);out=Path(s['evaluation']);paper=out/'paper_aligned_v1';paper.mkdir(parents=True,exist_ok=True)
    d=dict(base_deps(),endpoint=name,matrix=identity(Path(s['matrix'])/'complete.json'))
    runtime=source_runtime(c,d);rp=out/'r_runtime.json';atomic_json(rp,runtime)
    complete=paper/'complete.json'
    if complete.exists():check_receipt(complete,d)
    else:
        run_r(c,SOURCE/'scripts/rep1_triple_simple_boundary/paper_evaluation.R',
            [c['evaluation']['protocol'],c['evaluation']['prepared'],machine(name)['final_matrix'],paper,rp],
            paper,inherited_fds(),validate_runtime,cfg_progress,digest(d))
        ids,genes,counts=matrix_arrays(s);total=counts.sum(1);detected=(counts>0).sum(1);expected=(total>=20)&(total<=2000)&(detected>=20)
        qc=pd.read_csv(paper/'qc_all_cells.csv');ann=pd.read_csv(paper/'annotation.csv');score=pd.read_csv(paper/'markers_per_cell.csv')
        if qc.cell_id.duplicated().any() or ann.cell_id.duplicated().any() or score.cell_id.duplicated().any():raise ValueError('duplicate evaluator IDs')
        if not np.array_equal(qc.cell_id,ids) or not np.array_equal(qc.qc_pass,expected) or not np.array_equal(ann.cell_id,ids[expected]) or not np.array_equal(score.cell_id,ids[expected]):raise ValueError('own matrix QC/annotation/F1 population differs')
        if not np.isfinite(score[['positive_F1','negative_F1']]).all().all():raise ValueError('frozen scClassify F1 contract violated')
        outputs=stage_outputs(paper);validate_outputs(outputs);write_receipt(complete,d,outputs,own_qc_n=int(expected.sum()),reference_model=c['evaluation']['reference_model'])
    return stage_outputs(paper),dict(own_qc_n=len(pd.read_csv(paper/'annotation.csv')))

def fixed_panel(name):
    from scripts.new_methods.author_marker import fixed_scores,METRICS
    from scripts.rep1_triple_simple_boundary.inputs import matrix_arrays
    fixed=pd.read_csv(ROOT/'frozen_labels/historical_fixed_population.csv').set_index('cell_id')
    if fixed.index.has_duplicates:raise ValueError('fixed population duplicate IDs')
    labels=fixed.evaluation_label.fillna('__unavailable__:unavailable_other').copy();reasons=fixed.unmatched_reason.fillna('')
    labels.loc[reasons!='']='__unavailable__:'+reasons.loc[reasons!='']
    ids,genes,counts=matrix_arrays(spec(name));prepared=Path(config()['evaluation']['prepared'])
    scores=fixed_scores(counts,ids,genes,labels,read(prepared/'markers.json'),(prepared/'reference_genes.txt').read_text().splitlines()).reindex(fixed.index)
    scores.index.name='cell_id';missing=~scores.index.isin(ids);scores.loc[missing,'label_availability']='missing_matrix_row';scores.loc[missing,METRICS]=np.nan
    out=Path(spec(name)['evaluation'])/'fixed';atomic_csv(out/'fixed_population.csv',scores,index=True)
    summary=[]
    for group,frame in [('__all__',scores),*list(scores.groupby('anchor_label',dropna=False))]:
        for metric in METRICS:
            vals=frame[metric].to_numpy(float);ok=np.isfinite(vals)
            summary.append(dict(cell_type=str(group),metric=metric,n_population=len(vals),n_finite=int(ok.sum()),n_unavailable=int((~ok).sum()),mean=float(vals[ok].mean()) if ok.any() else None))
    atomic_csv(out/'fixed_summary.csv',pd.DataFrame(summary))
    return dict(population=len(fixed),missing_rows=int(missing.sum()))

def evaluate(name):
    from scripts.rep1_triple_simple_boundary.evaluation import structural,run_r
    c=config();s=spec(name);out=Path(s['evaluation']);d=dict(base_deps(),endpoint=name,matrix=identity(Path(s['matrix'])/'complete.json'))
    structural(s,input_manifest(),out/'structure',d,cfg_progress)
    annotation(name)
    paper=out/'paper_aligned_v1';centroids=pd.read_csv(out/'structure/cells_distribution.csv')[['cell_id','x_um','y_um']]
    atomic_csv(paper/'own_cell_centroids.csv',centroids)
    fixed=fixed_panel(name);spdeps=dict(d,part='spatial')
    if (out/'spatial_complete.json').exists():check_receipt(out/'spatial_complete.json',spdeps)
    else:
        run_r(c,SOURCE/'scripts/rep1_triple_simple_boundary/spatial_diversity.R',[out/'r_runtime.json',out],out,inherited_fds(),validate_runtime,cfg_progress,status_file='spatial_execution_status.txt')
        write_receipt(out/'spatial_complete.json',spdeps,{p:identity(paper/p) for p in ('spatial_diversity_bins.csv','spatial_diversity_correlations.csv')})
    return stage_outputs(out),dict(fixed=fixed)

def paired_f1(specs,pairs,ctx,out,W,region='GLOBAL',membership=None):
    rows=[];n=W.shape[1]
    for a,b in pairs:
        tables={m:{p:pd.read_csv(specs[m][p+'_scores']).set_index('cell_id') for p in ('fixed','sc')} for m in (a,b)}
        common=np.intersect1d(np.intersect1d(tables[a]['sc'].index,tables[b]['sc'].index),ctx['fixed'].index)
        if membership is not None:common=np.intersect1d(common,membership)
        for metric in ('positive_F1','negative_F1'):
            selected=common.copy()
            for m in (a,b):
                for profile in ('fixed','sc'):selected=selected[np.isfinite(tables[m][profile][metric].reindex(selected))]
            pop=Path(out)/'paired_membership'/(region+'__'+a+'__TO__'+b+'__'+metric+'.csv')
            atomic_csv(pop,pd.DataFrame(dict(cell_id=selected)))
            bi=ctx['cells'].loc[selected,'block_1000'].to_numpy();den=np.bincount(bi,minlength=n);draw_den=W@den
            for profile in ('fixed','sc'):
                x=tables[a][profile].loc[selected,metric].to_numpy(float);y=tables[b][profile].loc[selected,metric].to_numpy(float)
                sums=np.bincount(bi,weights=y-x,minlength=n);draw=np.divide(W@sums,draw_den,out=np.full(len(W),np.nan),where=draw_den>0);good=np.isfinite(draw)
                lo,hi=np.quantile(draw[good],[.025,.975]) if good.any() else (np.nan,np.nan)
                rows.append(dict(source=a,target=b,region=region,profile=profile,metric=metric,population='fixed membership intersect own QC a/b and four-way finite for this F1 metric',
                    n=len(selected),source_mean=x.mean() if len(x) else np.nan,target_mean=y.mean() if len(x) else np.nan,delta=(y-x).mean() if len(x) else np.nan,
                    ci_low=lo,ci_high=hi,valid_bootstrap=int(good.sum()),replicates=len(W),blocks=n,contributing_blocks=int((den>0).sum()),block_um=1000,seed=20261019,
                    status='COMPUTED' if good.any() else 'INSUFFICIENT_SUPPORT',NA_reason='' if good.any() else 'empty_finite_pair_population_or_zero_resampled_denominator',membership_sha256=identity(pop)['sha256']))
    return pd.DataFrame(rows)

def statistics():
    from scripts.boundary_goal_v3 import local
    from scripts.rep1_triple_simple_boundary.inputs import matrix_arrays
    out=ROOT/'statistics';out.mkdir(exist_ok=True);specs={n:spec(n) for n in ENDPOINTS}
    roi=read(ROOT/'frozen_roi/manifest.json');manifest=input_manifest()
    # Local and fixed-F1 consumers are named and bound separately, even where
    # their membership currently has the same B1 annotation producer.
    manifest['fixed_population']=str(ROOT/'frozen_labels/local_B1_scClassify_labels.csv')
    cfg=dict(inputs=manifest,old_config=dict(domain_manifest=str(ROOT/'frozen_roi/manifest.json')),specs=specs,
        eligible=config()['eligible'],interface_expected=roi['interface_all_gene'],seed=20261019,bootstrap_replicates=2000)
    ctx=local.context(cfg,out)
    pairs=[(B0,BETA1),(P2,BETA1),(BASE,BETA1),(P2,B0),(BASE,P2),(V2,P2)]
    local.stream(cfg,specs,pairs,out,ctx,cfg_progress);mass,flows=local.summaries(out);intervals=local.intervals(cfg,out,ctx)
    # Report support of the V4 direct estimator without importing older gates.
    sufficient=np.load(out/'block_sums.npz');schema=read(out/'schema.json');diagnostics=[]
    for _,row in intervals.iterrows():
        ri=schema['regions'].index(row.region);mi=schema['methods'].index(row.source)
        den=sufficient['mass'][mi,ri].sum(1)
        if row.metric=='R_off_minus_R_on':
            pi=schema['pairs'].index([row.source,row.target]);e=sufficient['exits'][pi,ri,0];den=e[:,0]+e[:,2]
            zero=not e[:,0].sum() or not e[:,2].sum()
        elif row.metric=='favorability_balance_per_all_transfer':
            pi=schema['pairs'].index([row.source,row.target]);den=sufficient['transfers'][pi,ri,:,0];zero=not den.sum()
        elif row.metric=='type_score_selectivity_increment_same_parent_origins':
            pi=schema['pairs'].index([P2,B0]);e=sufficient['exits'][pi,ri,0];den=e[:,0]+e[:,2];zero=not e[:,0].sum() or not e[:,2].sum()
        else:zero=not den.sum()
        reason='NA_zero_denominator' if zero else 'NA_no_valid_spatial_draws' if row.valid_bootstrap==0 else ''
        diagnostics.append(dict(contributing_blocks=int((den>0).sum()),NA_reason=reason,seed=20261019))
    intervals=pd.concat([intervals.reset_index(drop=True),pd.DataFrame(diagnostics)],axis=1)
    intervals['unit']='fraction'
    for source,dest in [('estimate','estimate_percentage_points'),('ci_low','ci_low_percentage_points'),('ci_high','ci_high_percentage_points')]:intervals[dest]=100*intervals[source]
    atomic_csv(out/'intervals.csv',intervals)
    local.distance_curves(cfg,specs,out,ctx)
    # Independently prove the archived generator used the frozen global weights.
    saved=np.load(ROOT/'frozen_roi/bootstrap_weights_1000.npz');W=saved['W'];rng=np.random.default_rng(20261019);n=len(saved['blocks'])
    expected=np.vstack([np.bincount(rng.integers(n,size=n),minlength=n) for _ in range(2000)])
    if not np.array_equal(W,expected):raise ValueError('paired bootstrap weights changed')
    membership=pd.read_csv(ROOT/'frozen_roi/cell_memberships.csv')
    paired=[];raw_local=[]
    for region,group in membership.groupby('region',sort=False):
        chosen=group.cell_id.to_numpy();paired.append(paired_f1(specs,pairs,ctx,out,W,region,chosen))
        for name,s in specs.items():
            for profile,key in (('fixed','fixed_scores'),('scclassify','sc_scores')):
                frame=pd.read_csv(s[key]);frame=frame[frame.cell_id.isin(chosen)]
                for metric in ('positive_F1','negative_F1'):
                    values=frame[metric].to_numpy(float);finite=np.isfinite(values)
                    raw_local.append(dict(method=name,region=region,profile=profile,metric=metric,n_population=len(values),n_finite=int(finite.sum()),n_NA=int((~finite).sum()),mean=values[finite].mean() if finite.any() else np.nan,NA_reason='' if finite.any() else 'absent_finite_scores'))
    combined=pd.concat(paired,ignore_index=True);atomic_csv(out/'paired_F1.csv',combined)
    atomic_csv(out/'paired_global_F1.csv',combined[combined.region.eq('GLOBAL')])
    atomic_csv(out/'raw_local_F1.csv',pd.DataFrame(raw_local))
    globalrows=[]
    for name,s in specs.items():
        ids,genes,counts=matrix_arrays(s);total=counts.sum(1);detected=(counts>0).sum(1);qc=(total>=20)&(total<=2000)&(detected>=20)
        row=dict(method=name,protocol_identifier=NAMES[name],role='Tier1' if name in (BASE,P2,B0,BETA1) else 'diagnostic',n_cells=len(ids),matrix_genes=len(genes),
            eligible_RNA_D=config()['eligible'],assigned_RNA=int(total.sum()),capture=float(total.sum()/config()['eligible']),qc_pass=int(qc.sum()),qc_rejected=int((~qc).sum()))
        row.update(read(Path(s['evaluation'])/'structure/structure.json'))
        for profile,key in (('fixed','fixed_scores'),('scclassify','sc_scores')):
            frame=pd.read_csv(s[key]);row[profile+'_population_n']=len(frame)
            for metric in ('positive_F1','negative_F1'):
                v=frame[metric].to_numpy(float);ok=np.isfinite(v);row[profile+'_'+metric]=float(v[ok].mean()) if ok.any() else None
                row[profile+'_'+metric+'_n_finite']=int(ok.sum());row[profile+'_'+metric+'_n_NA']=int((~ok).sum())
        globalrows.append(row)
    atomic_csv(out/'global_metrics.csv',pd.DataFrame(globalrows))
    globaltable=pd.DataFrame(globalrows).set_index('method');costs=[]
    for a,b in pairs:
        x=globaltable.loc[a];y=globaltable.loc[b]
        record=dict(source=a,target=b,assigned_RNA_delta=int(y.assigned_RNA-x.assigned_RNA),capture_percentage_point_delta=100*(y.capture-x.capture),
            historical_capture_monitor_percentage_points=.5,historical_retention_monitor=.995,threshold_role='historical monitoring only, never outcome acceptance')
        for field in ('disconnected_cells_4','disconnected_cells_8'):
            if field in x:record[field+'_delta']=int(y[field]-x[field])
        costs.append(record)
    atomic_csv(out/'structural_capture_changes.csv',pd.DataFrame(costs))
    reference=read(TRAINING/'shared/configs/reference_region.json');evalgenes=Path(config()['evaluation']['prepared'],'reference_genes.txt').read_text().splitlines()
    atomic_csv(out/'MARKER_PANEL_AND_DEPENDENCY.csv',pd.DataFrame([dict(gene=g,lineage='B' if lineage==1 else 'T',in_313_matrix=g in ctx['genes'],in_type_prior=g in reference['gene_names'],in_307_evaluator=g in evalgenes,
        reference_labels='frozen Rep2 B1-derived scClassify; exact B Cells/CD4 T/CD8 T map',label_view_sha256=identity(ROOT/'frozen_labels/local_B1_scClassify_labels.csv')['sha256'],
        prior_sha256=identity(TRAINING/'shared/configs/reference_region.json')['sha256'],evaluation_markers_sha256=config()['frozen_inputs']['evaluation_markers.json']['sha256'],evidence='REFERENCE-CONSISTENCY PROXY') for g,lineage in local.MARKERS.items()]))
    contrasts=[]
    for a,b in pairs:
        left=mass[mass.method.eq(a)].set_index(['region','state']);right=mass[mass.method.eq(b)].set_index(['region','state'])
        for key,x in left.iterrows():
            y=right.loc[key];den=int(x.fixed_raw_marker_D)
            if den!=int(y.fixed_raw_marker_D):raise ValueError('method-specific marker denominator')
            contrasts.append(dict(source=a,target=b,region=key[0],state=key[1],source_count=int(x['count']),target_count=int(y['count']),fixed_raw_marker_D=den,
                delta_fraction=float((y['count']-x['count'])/den) if den else np.nan,delta_percentage_points=float(100*(y['count']-x['count'])/den) if den else np.nan,
                NA_reason='' if den else 'zero_fixed_marker_denominator'))
    atomic_csv(out/'local_contrasts.csv',pd.DataFrame(contrasts))
    for (name,region),frame in mass.groupby(['method','region']):
        if set(frame.state)!=set(local.STATES) or int(frame['count'].sum())!=int(frame.fixed_raw_marker_D.iloc[0]):raise ValueError('four-state denominator conservation')
    for _,frame in flows[flows.kind.eq('RECIPIENT_FATE')].groupby(['source','target','region']):
        if set(frame.fate)!=set(local.FATES) or int(frame['count'].sum())!=int(frame.all_transferred.iloc[0]):raise ValueError('six-fate transfer conservation')
    for name in ENDPOINTS:
        group=mass[mass.method.eq(name)&mass.region.eq('INTERFACE_0_2')]
        if int(group.fixed_raw_marker_D.iloc[0])!=roi['interface_six_marker']:raise ValueError('all-interface marker frozen denominator')
    atomic_json(out/'validation.json',dict(status='succeeded',raw_rows=config()['eligible'],interface_all_gene=roi['interface_all_gene'],interface_six_marker=roi['interface_six_marker'],
        all_four_states_conserved=True,all_six_fates_conserved=True,bootstrap_weights_exact=True,required_contrasts=pairs,
        science_interpretation='deferred until manual resume',published_tables_modified=False,
        historical_monitoring={'capture_threshold_percentage_points':0.5,'on_target_retention_threshold':.995,'role':'reported historical monitoring only; no outcome-sign completion gate'}))
    return stage_outputs(out),dict(contrasts=len(pairs),endpoints=list(ENDPOINTS),validation=identity(out/'validation.json'))
