"""Separately recoverable final report; does not produce models or evaluations."""
from pathlib import Path
import numpy as np
import pandas as pd
from .common import *
from .author_marker import METRICS

def active_coverage(root,arm,dest):
    import json
    retained={}
    for line in (root/arm/'training.jsonl').open():
        row=json.loads(line)
        if row.get('event')=='update':retained[row['updates']]=row
    if set(retained)!=set(range(1,3626)):raise ValueError('retained training diagnostic budget incomplete')
    rows=[]
    for row in retained.values():
        for candidate in row.get('candidate_auxiliary',[]):rows.append(dict(view_cursor=row['cursor']-1,**candidate))
    if not rows:return dict(status='not_applicable',reason='morphology has no new reference loss')
    frame=pd.DataFrame(rows);static=pd.read_csv(root/'admission/coverage_candidates.csv')
    frame=frame.merge(static,on=['view_cursor','cell_id'],how='left',validate='one_to_one',suffixes=('','_static'))
    if frame['type'].isna().any():raise ValueError('active coverage cannot join frozen candidate population')
    result={}
    for key in ['type','elongated','K_bin','visibility_bin','count_bin','area_quintile','major_quintile','edge_distance']:
        result[key]={str(name):dict(appearances=len(g),unique_ids=int(g.cell_id.nunique()),eligible=int(g.eligible.sum()),
                        active=int(g.active.sum()),effective_weight=float(g.loc[g.eligible,'weight'].sum()),
                        active_weight=float(g.loc[g.active,'weight'].sum())) for name,g in frame.groupby(key)}
    frame.to_csv(dest/f'{arm}_training_active_coverage.csv',index=False)
    return result

def report(root):
    root=Path(root);queue=read(root/'configs/queue.json');rows=[];artifacts={}
    for method,spec in queue['comparison_methods'].items():
        out=Path(spec['evaluation']);rec=read(out/'complete.json')
        if rec.get('status')!='succeeded':raise ValueError('method evaluation incomplete '+method)
        verify_outputs(rec);artifacts[method]=identity(out/'complete.json')
        fixed=pd.read_csv(out/'fixed/fixed_summary.csv')
        for row in fixed.to_dict('records'):rows.append(dict(method=method,**row))
    frame=pd.DataFrame(rows);dest=root/'report';dest.mkdir(exist_ok=True);frame.to_csv(dest/'fixed_all_methods.csv',index=False,na_rep='NA')
    baseline_panel=pd.read_csv(Path(queue['comparison_methods']['B0']['evaluation'])/'fixed/fixed_population.csv').set_index('cell_id')
    diffs=[]
    for arm in ARMS:
        own=pd.read_csv(Path(queue['comparison_methods'][arm]['evaluation'])/'fixed/fixed_population.csv').set_index('cell_id').reindex(baseline_panel.index)
        for typ in ['__all__',*baseline_panel.anchor_label.dropna().unique().tolist()]:
            group=np.ones(len(own),bool) if typ=='__all__' else baseline_panel.anchor_label.eq(typ).to_numpy()
            for metric in METRICS:
                a=own[metric].to_numpy(float);b=baseline_panel[metric].to_numpy(float);valid=group&np.isfinite(a)&np.isfinite(b)
                diffs.append(dict(method=arm,cell_type=typ,metric=metric,value=float(a[valid].mean()) if valid.any() else np.nan,
                                  B0=float(b[valid].mean()) if valid.any() else np.nan,delta=float((a[valid]-b[valid]).mean()) if valid.any() else np.nan,
                                  paired_finite_n=int(valid.sum()),fixed_population_n=int(group.sum()),
                                  arm_finite_n=int((group&np.isfinite(a)).sum()),B0_finite_n=int((group&np.isfinite(b)).sum())))
    pd.DataFrame(diffs).to_csv(dest/'differences_vs_B0.csv',index=False,na_rep='NA')
    active={arm:active_coverage(root,arm,dest) for arm in ARMS}
    atomic_json(dest/'summary.json',json_safe(dict(state='compute_complete',scientific_review='awaiting_user_resume',training_active_coverage=active,
                limitations=['one seed; historical B0 exact initial RNG/order unmatched','training reference metrics are same-source diagnostics','no independent owner ground truth'],
                fixed_comparisons=diffs,methods={m:read(Path(s['evaluation'])/'summary.json') for m,s in queue['comparison_methods'].items()})))
    atomic_json(dest/'complete.json',dict(status='succeeded',state='compute_complete',scientific_review='awaiting_user_resume',dependencies=artifacts,outputs=outputs_under(dest)))
