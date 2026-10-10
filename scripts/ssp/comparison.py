"""Fixed historical B1 population; explicit NA and frozen four-method purity."""
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scripts.cos import diagnostics
from scripts.crgf import evaluate_egr as accepted
from scripts.crgf.common import atomic_json,identity,sha256,verify_outputs
from .evaluation import read,PAIRS,PROFILES,publish,validate_arm,method_coverage

def population_scores(methods,queue,out,population,label):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);labels=diagnostics.load_anchor(queue['anchors']['B1'])
    markers=read(Path(queue['prepared'])/'markers.json');genes_ref=(Path(queue['prepared'])/'reference_genes.txt').read_text().splitlines()
    values={};summary=[];counts_summary=[]
    for name,spec in methods.items():
        matrix=Path(spec['matrix']);ids=pd.read_csv(matrix/'cell_mapping.csv').canonical_cell_id.to_numpy(int);genes=pd.read_csv(matrix/'gene_mapping.csv').gene.tolist();idx=pd.Index(ids).get_indexer(population)
        if np.any(idx<0):raise ValueError('frozen population ID missing from own matrix '+name)
        x=np.load(matrix/'counts.npy',mmap_mode='r')[idx];f=diagnostics.fixed_scores(x,population,genes,labels,markers,genes_ref)
        # Empty expression has no rank prediction domain; preserve explicit NA.
        empty=x[:,pd.Index(genes).get_indexer(genes_ref)].sum(1)==0
        f.loc[empty,diagnostics.METRICS]=np.nan;f['zero_reference_counts']=empty
        qc=set(pd.read_csv(Path(spec['evaluation'])/'paper_aligned_v1/markers_per_cell.csv').cell_id)
        f['own_qc_pass']=f.index.isin(qc);f['population']=label;f.to_csv(out/f'{name}.csv',na_rep='NA');values[name]=f
        for typ,part in [('__all__',f),*list(f.groupby('anchor_label',dropna=False))]:
            for metric in diagnostics.METRICS:
                v=part[metric];finite=np.isfinite(v);counts_summary.append(dict(population=label,method=name,type=typ,metric=metric,n_eligible=len(v),n_evaluable=int(finite.sum()),n_unavailable=int((~finite).sum()),n_zero_counts=int(part.zero_reference_counts.sum()),n_own_qc=int(part.own_qc_pass.sum()),mean=float(v[finite].mean()) if finite.any() else None))
    for a,b in PAIRS:
        if a not in values or b not in values:continue
        delta=values[a][diagnostics.METRICS]-values[b][diagnostics.METRICS];delta['type']=values[a].anchor_label
        delta.to_csv(out/f'{a}_minus_{b}.csv',na_rep='NA')
        for typ,part in [('__all__',delta),*list(delta.groupby('type',dropna=False))]:
            for metric in diagnostics.METRICS:
                v=part[metric];ok=np.isfinite(v);summary.append(dict(population=label,comparison=a+'-'+b,contrast_role='primary' if (a,b) in PAIRS[:2] else 'auxiliary',type=typ,metric=metric,n_eligible=len(v),n_evaluable=int(ok.sum()),n_unavailable=int((~ok).sum()),mean_delta=float(v[ok].mean()) if ok.any() else None))
    macro=[]
    for name in methods:
        for metric in diagnostics.METRICS:
            rows=[r for r in counts_summary if r['method']==name and r['metric']==metric and r['type'] in markers['positive'] and r['n_evaluable']>0]
            macro.append(dict(population=label,method=name,metric=metric,n_evaluable_types=len(rows),type_macro_mean=float(np.mean([r['mean'] for r in rows])) if rows else None))
    pd.DataFrame(macro).to_csv(out/'type_macro_metrics.csv',index=False,na_rep='NA')
    pd.DataFrame(summary).to_csv(out/'paired_metrics.csv',index=False,na_rep='NA');pd.DataFrame(counts_summary).to_csv(out/'metric_summary.csv',index=False,na_rep='NA')
    return summary

def aggregate(config):
    q=read(config);root=Path(q['output_root']);out=root/'comparison';out.mkdir(exist_ok=True);start=time.monotonic()
    receipts={a:validate_arm(q,a) for a in ('S','SP')};methods={**q['historical_methods'],**q['methods']}
    oldq=read(Path(q['historical_root'])/'configs/queue.json')
    for a in ('A','B'):receipts[a]=validate_arm(oldq,a)
    verify_outputs(read(Path(oldq['methods']['C']['evaluation'])/'complete.json'))
    if set(methods)!={'A','B','S','SP'}:raise ValueError('fixed purity set mismatch')
    if sha256(q['population']['path'])!=q['population']['sha256']:raise ValueError('frozen B1 population drift')
    population=pd.read_csv(q['population']['path']).cell_id.astype(int).tolist()
    population_scores(methods,q,out/'fixed_B1QC',population,'historical_B1QC')
    qc={a:set(pd.read_csv(Path(s['evaluation'])/'paper_aligned_v1/markers_per_cell.csv').cell_id) for a,s in methods.items()}
    oldq=read(Path(q['historical_root'])/'configs/queue.json');qc_c=set(pd.read_csv(Path(oldq['methods']['C']['evaluation'])/'paper_aligned_v1/markers_per_cell.csv').cell_id)
    sensitivities={'historical_ABC_commonQC':qc['A']&qc['B']&qc_c,'B_S_SP_commonQC':qc['B']&qc['S']&qc['SP']}
    for a,b in PAIRS:sensitivities[a+'_'+b+'_commonQC']=qc[a]&qc[b]
    for name,ids in sensitivities.items():population_scores(methods,q,out/'sensitivity'/name,sorted(ids),name)
    for a in methods:population_scores({a:methods[a]},q,out/'sensitivity'/('ownQC_'+a),sorted(qc[a]),'ownQC_'+a)
    raw_purity={a:{k:read(Path(s['evaluation'])/'summary.json')[k] for k in ('positive_F1','negative_F1')} for a,s in methods.items()}
    pur=accepted.purity(raw_purity,methods=['A','B','S','SP']);pur.update(method_set=['A','B','S','SP'],historical_sets_not_comparable=True);atomic_json(out/'purity.json',pur)
    coverage=[];raw=[];snapshot=out/'resource_cost_snapshot.json';atomic_json(snapshot,read(root/'run_status.json'))
    for a in ('S','SP'):
        c,r=method_coverage(q,a,cost_snapshot=snapshot);coverage+=c;raw+=r
    for a,b in PAIRS:
        dest=out/'diagnostics'/f'{a}_minus_{b}'
        diagnostics.ownership_pair(methods[a],methods[b],dest,q['anchors']['B1'],q['prepared'],a+'-'+b)
        ac=read(methods[a]['machine']);bc=read(methods[b]['machine']);accepted.changes(ac['final_mask'],bc['final_mask'],dest/'mask_changes',a+'-'+b,ac['pixel_um'])
    diagnostics.postprocess_increment(methods,q['anchors'],PAIRS,population,out/'diagnostics')
    coverage.extend(dict(metric_id=k,profile='SSP',comparison='A/B/S/SP',required_this_run=True,status='computed',artifact_path=str(p)) for k,p in [('fixed_B1QC',out/'fixed_B1QC/paired_metrics.csv'),('purity_fixed_set',out/'purity.json'),('B_cost',root/'b_cost/facts.json')])
    pd.DataFrame(coverage).to_csv(out/'metric_coverage.csv',index=False,na_rep='NA');pd.DataFrame(raw).to_csv(out/'raw_metrics.csv',index=False,na_rep='NA')
    bad=[x for x in coverage if x['required_this_run'] and x['status'] not in ('computed','computed_distribution','partial_domain_NA','not_identifiable_zero_range')]
    if bad:raise ValueError('required metric coverage failures '+str(bad))
    publish(out,ready=True,paper_protocol_complete=True,profiles=PROFILES,arm_receipts=receipts,primary_comparisons=['S-B','SP-S'],auxiliary_comparisons=['SP-B','S-A','SP-A'],fixed_population=q['population'],independent_scientific_acceptance=False,next_state='awaiting_user_resume',wall_seconds=time.monotonic()-start)
    return True
