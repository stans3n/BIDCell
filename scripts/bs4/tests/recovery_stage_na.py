"""Zero-reference stage availability: counts authority and strict negative cases."""
import argparse,copy,json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.bs4.stage_reference import normalize_reference_domain,load_stage_table
from scripts.bs4.stage_population import align_stage
from scripts.cos.diagnostics import METRICS,load_anchor
from scripts.crgf.common import atomic_json,identity,source_identity
SOURCES=['scripts/bs4/stage_reference.py','scripts/bs4/stage_population.py','scripts/bs4/tests/recovery_stage_na.py']

def run(root,out):
    execution=source_identity(SOURCES)
    refs=['g'+str(i) for i in range(307)];genes=refs+['outside'];counts=np.zeros((3,308),np.int64);counts[0,-1]=9;counts[1,0]=2
    base=pd.DataFrame(dict(cell_id=[1,2,3],anchor_label=['known','known','unknown'],label_availability=['known','known','unavailable_other'],**{m:[0.,.5,0.] for m in METRICS}))
    actual,summary=normalize_reference_domain(base,[1,2,3],counts,genes,refs)
    assert actual.zero_reference_counts.tolist()==[True,False,True]
    assert actual.loc[[0,2],METRICS].isna().all().all() and (actual.loc[1,METRICS]==.5).all()
    aligned,domain=align_stage(actual,[1,2,3,4],pd.Series(['known','known','unknown','known'],index=[1,2,3,4]))
    assert aligned.positive_F1_reason.tolist()==['zero_reference_counts','available','label_ineligible','missing_stage_id']
    faults={}
    cases={
        'duplicate_count_ID':(base,[1,1,3],counts,genes,refs),
        'duplicate_stage_ID':(base.assign(cell_id=[1,1,3]),[1,2,3],counts,genes,refs),
        'missing_count_ID':(base,[1,2,4],counts,genes,refs),
        'duplicate_gene':(base,[1,2,3],counts,genes[:-1]+['g0'],refs),
        'duplicate_reference':(base,[1,2,3],counts,genes,refs[:-1]+['g0']),
        'missing_reference':(base,[1,2,3],counts,genes,refs[:-1]+['absent']),
        'negative_count':(base,[1,2,3],-np.ones_like(counts),genes,refs),
        'noninteger_count':(base,[1,2,3],counts.astype(float)+.5,genes,refs),
        'wrong_zero_flag':(base.assign(zero_reference_counts=[False,False,True]),[1,2,3],counts,genes,refs),
        'nan_zero_flag':(base.assign(zero_reference_counts=[np.nan,False,True]),[1,2,3],counts,genes,refs),
        'shape':(base,[1,2,3],counts[:,:-1],genes,refs)}
    for name,args in cases.items():
        try:normalize_reference_domain(*args)
        except ValueError as e:faults[name]=str(e)
        else:raise AssertionError('accepted '+name)
    try:align_stage(actual,[1,2,3],pd.Series(['changed','known','unknown'],index=[1,2,3]))
    except ValueError as e:faults['label_drift']=str(e)
    else:raise AssertionError('label drift')
    root=Path(root);q=json.loads((root/'configs/queue.json').read_text());table,authority=load_stage_table(q['historical_methods']['S'],'B1','pre',q['prepared']);ids=pd.read_csv(q['population']['path']).cell_id.tolist();aligned,_=align_stage(table,ids,load_anchor(q['anchors']['B1']))
    zero=aligned.zero_reference_counts.fillna(False)&aligned.label_availability.eq('known')
    assert int(zero.sum())==54 and aligned.loc[zero,METRICS].isna().all().all()
    assert source_identity(SOURCES)==execution
    atomic_json(out,dict(status='passed',execution_source=execution,synthetic_reference_only_zero_with_nonzero_outside_reference=True,negative_cases=faults,real_S_pre_known_zero_IDs=aligned.index[zero].tolist(),real_source=authority,outputs={}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.root,a.output)
