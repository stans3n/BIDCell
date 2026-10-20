"""Own output three profiles, immutable historical fixed-label panel, machine summary."""
from pathlib import Path
import numpy as np
import pandas as pd
from .common import *
from .accepted_profiles import run_profiles
from .author_marker import fixed_scores,METRICS
from .provenance import component_source


def fixed_panel(spec,queue,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    pop=pd.read_csv(queue['population']['path'])
    if sha256(queue['population']['path'])!=queue['population']['sha256']:raise ValueError('fixed population drift')
    if len(pop)!=103000:raise ValueError('fixed population size mismatch')
    # The frozen table explicitly includes historical label availability.
    if 'cell_id' not in pop:raise ValueError('population cell ID missing')
    labelcol=next((c for c in ['evaluation_label','anchor_label','cell_type','label'] if c in pop),None)
    if labelcol is None:raise ValueError('historical fixed label column missing: '+str(pop.columns.tolist()))
    labels=pop.set_index('cell_id')[labelcol].fillna('__unavailable__:unavailable_other')
    if 'unmatched_reason' in pop:
        reasons=pop.set_index('cell_id').unmatched_reason.fillna('')
        for reason in reasons[reasons!=''].unique():labels.loc[reasons==reason]='__unavailable__:'+str(reason)
    mapping=pd.read_csv(Path(spec['matrix'])/'cell_mapping.csv');ids=mapping.canonical_cell_id.to_numpy(np.int64)
    genes=pd.read_csv(Path(spec['matrix'])/'gene_mapping.csv').gene.tolist();counts=np.load(Path(spec['matrix'])/'counts.npy',mmap_mode='r')
    prepared=Path(queue['prepared']);markers=read(prepared/'markers.json');refs=(prepared/'reference_genes.txt').read_text().splitlines()
    scores=fixed_scores(counts,ids,genes,labels,markers,refs)
    panel=scores.reindex(pop.cell_id.to_numpy(np.int64));panel.index.name='cell_id'
    panel['anchor_label']=labels.reindex(panel.index)
    missing=~panel.index.isin(ids);panel.loc[missing,'label_availability']='missing_matrix_row'
    panel.loc[missing,METRICS]=np.nan
    panel.to_csv(out/'fixed_population.csv',na_rep='NA')
    rows=[]
    for typ,frame in [('__all__',panel),*list(panel.groupby('anchor_label',dropna=False))]:
        for metric in METRICS:
            vals=frame[metric].to_numpy(float);finite=np.isfinite(vals)
            rows.append(dict(cell_type=str(typ),metric=metric,n_population=len(frame),n_finite=int(finite.sum()),n_unavailable=int((~finite).sum()),
                             mean=float(vals[finite].mean()) if finite.any() else np.nan))
    summary=pd.DataFrame(rows);summary.to_csv(out/'fixed_summary.csv',index=False,na_rep='NA')
    macro=summary[summary.cell_type!='__all__'].groupby('metric')['mean'].mean()
    atomic_json(out/'fixed_summary.json',json_safe(dict(population=103000,missing_rows=int(missing.sum()),
                overall=summary[summary.cell_type=='__all__'].to_dict('records'),type_macro=macro.to_dict(),
                semantics='author matched zero F1/Recall=0, Precision stays NaN; missing and historically unavailable remain NA')))
    return read(out/'fixed_summary.json')


def evaluate(config):
    cfg=read(config) if isinstance(config,(str,Path)) else config;queue=read(cfg['queue_config']);arm=cfg['arm'];spec=queue['methods'][arm]
    return evaluate_spec(spec,queue,arm)


def evaluate_spec(spec,queue,arm):
    out=Path(spec['evaluation']);out.mkdir(parents=True,exist_ok=True)
    prep=read(Path(queue['prepared'])/'complete.json');protocol=prep['dependencies']['protocol']['path']
    dependencies=dict(matrix=identity(Path(spec['matrix'])/'complete.json'),machine=identity(spec['machine']),
                      prepared=identity(Path(queue['prepared'])/'complete.json'),source=component_source('evaluate'))
    cached=out/'profiles_complete.json'
    if cached.exists():
        rec=read(cached)
        if rec['dependencies']!=dependencies:raise ValueError('profile source/input drift')
        verify_outputs(rec);summary=rec['summary']
    else:
        summary=run_profiles(spec,queue,protocol,out,arm)
        atomic_json(cached,dict(status='succeeded',dependencies=dependencies,summary=summary,
                               outputs={str(f.relative_to(out)):identity(f) for folder in ['engineering','paper_aligned_v1'] for f in (out/folder).rglob('*') if f.is_file()}))
    fixed=fixed_panel(spec,queue,out/'fixed')
    summary.update(fixed_population=fixed,method=arm)
    summary['engineering_metrics']=pd.read_csv(out/'engineering/metrics.csv').to_dict('records')
    summary['structure']=read(out/'engineering/structure/structure.json')
    summary['reference_by_type']=pd.read_csv(out/'paper_aligned_v1/type_correlations.csv').to_dict('records')
    summary['neighbour_curves']=pd.read_csv(out/'paper_aligned_v1/neighbour_curves.csv').to_dict('records')
    atomic_json(out/'summary.json',json_safe(summary))
    atomic_json(out/'complete.json',dict(status='succeeded',profiles=['engineering_nuclear_v1','engineering_pearson_v1','paper_aligned_v1'],
                dependencies=dependencies,paper_method_complete=True,outputs=outputs_under(out)))
    return summary

