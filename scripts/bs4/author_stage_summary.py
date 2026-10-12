"""Publish fixed-population author stage means and bind required coverage to them."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from .author_marker import METRICS
from scripts.crgf.common import identity,sha256,atomic_json

def records(aligned,arm,anchor,stage):
    rows=[]
    for metric in METRICS:
        values=aligned[metric];finite=np.isfinite(values)
        if not np.array_equal(finite,aligned[metric+'_reason'].eq('available')):raise ValueError('author stage validity/finite inconsistency')
        rows.append(dict(method=arm,anchor=anchor,stage=stage,metric=metric,n_eligible=len(aligned),n_evaluable=int(finite.sum()),n_unavailable=int((~finite).sum()),mean=float(values[finite].mean()) if finite.any() else None,population='frozen_historical_B1QC',n_missing_stage_id=int(aligned[metric+'_reason'].eq('missing_stage_id').sum()),n_label_ineligible=int(aligned[metric+'_reason'].eq('label_ineligible').sum()),n_zero_reference_descriptive=int(aligned.zero_reference_counts.fillna(False).sum()),n_valid_zero=int(values.eq(0).sum())))
    return rows

def publish(rows,authority,comparison):
    comparison=Path(comparison);frame=pd.DataFrame(rows)
    expected={(arm,anchor,stage,metric) for key in authority for arm,anchor,stage in [key.split('/')] for metric in METRICS}
    observed=set(map(tuple,frame[['method','anchor','stage','metric']].to_numpy()))
    if len(frame)!=len(expected) or observed!=expected:raise ValueError('incomplete author stage summary inventory')
    if not (frame.n_evaluable+frame.n_unavailable).eq(frame.n_eligible).all() or (frame[['n_eligible','n_evaluable','n_unavailable','n_valid_zero']]<0).any().any() or (frame.n_valid_zero>frame.n_evaluable).any():raise ValueError('invalid author stage summary denominators')
    if not np.array_equal(np.isfinite(frame['mean'].to_numpy(float)),frame.n_evaluable.gt(0)):raise ValueError('stage summary mean/evaluable mismatch')
    outputs={}
    for arm,part in frame.groupby('method',sort=True):
        path=comparison/'author_stage_summaries'/arm/'stage_fixed_anchor_summary.csv';path.parent.mkdir(parents=True,exist_ok=True);part.to_csv(path,index=False,na_rep='NA');outputs[arm]=identity(path)
    atomic_json(comparison/'author_stage_summaries/contract.json',dict(version='author-stage-fixed-summary-v1',n_rows=len(frame),n_native_stage_domains=len(authority),population='original frozen historical B1QC order and labels; each stage aligned to the full fixed population',zero_reference_counts='descriptive only',metric_domain='each author metric independently finite; legal F1/Recall zero included',outputs=outputs))
    return outputs

def remap_coverage(coverage,methods,summaries):
    coverage=coverage.copy();which=coverage.metric_id.eq('COS:stage_fixed_anchor');part=coverage.loc[which]
    if len(part)!=4 or set(part.comparison)!={'P1','P2','T1','T2'}:raise ValueError('four required stage summary coverage rows missing')
    if not {'P1','P2','T1','T2'}<=set(summaries):raise ValueError('new required stage summaries missing')
    previous={}
    for i,row in part.iterrows():
        arm=row.comparison;expected=Path(methods[arm]['evaluation'])/'diagnostics/stages/stage_fixed_anchor_summary.csv'
        if Path(row.artifact_path).resolve()!=expected.resolve():raise ValueError('original stage summary coverage path drift')
        receipt=json.loads((Path(methods[arm]['evaluation'])/'complete.json').read_text());matches=[v for v in receipt['outputs'].values() if Path(v['path']).resolve()==expected.resolve()]
        if len(matches)!=1 or matches[0]['sha256']!=sha256(expected):raise ValueError('original stage summary authority drift')
        target=summaries[arm]
        if target['sha256']!=sha256(target['path']):raise ValueError('new author stage summary drift')
        coverage.loc[i,'artifact_path']=target['path'];coverage.loc[i,'definition_version']='author-stage-fixed-summary-v1';coverage.loc[i,'implementation']='scripts/bs4/author_stage_summary.py';coverage.loc[i,'reason']='author marker scores; full frozen historical B1 population; per-metric finite domain; original evaluation summary retained'
        previous[str(expected)]=identity(expected)
    if any(Path(p).resolve() not in {Path(v['path']).resolve() for v in summaries.values()} for p in coverage.loc[which,'artifact_path']):raise ValueError('required stage summary coverage not rebound')
    return coverage,previous
