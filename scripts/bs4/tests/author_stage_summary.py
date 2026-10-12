"""All 59 real stage summaries plus required coverage authority and fault cases."""
from pathlib import Path
import argparse,json,copy
import numpy as np,pandas as pd
from scripts.bs4.author_stage_summary import records,publish,remap_coverage
from scripts.crgf.common import atomic_json,identity,source_identity
SOURCES=['scripts/bs4/author_stage_summary.py','scripts/bs4/author_marker.py','scripts/bs4/tests/author_stage_summary.py']
def run(root,integration,out):
 root=Path(root);integration=Path(integration);out=Path(out);out.mkdir(parents=True,exist_ok=True);start=source_identity(SOURCES);q=json.loads((root/'configs/queue.json').read_text());authority=json.loads((integration/'stage_input_authority.json').read_text());rows=[];inputs={}
 for key in authority:
  arm,anchor,stage=key.split('/');p=integration/'normalized_stages'/arm/f'fixed_{anchor}_{stage}.csv.gz';frame=pd.read_csv(p,float_precision='round_trip').set_index('cell_id');inputs[str(p)]=identity(p);part=records(frame,arm,anchor,stage)
  for rec in part:
   v=frame[rec['metric']].to_numpy();finite=np.isfinite(v);assert rec['n_evaluable']==int(finite.sum()) and rec['n_eligible']==len(v)
   if finite.any():np.testing.assert_allclose(rec['mean'],v[finite].mean(),atol=0,rtol=0)
  rows+=part
 assert len(authority)==59 and len(rows)==354
 summaries=publish(rows,authority,out/'comparison');coverage=pd.read_csv(root/'comparison/metric_coverage.csv');new,prior=remap_coverage(coverage,q['methods'],summaries);new.to_csv(out/'metric_coverage.csv',index=False)
 required=new[new.metric_id=='COS:stage_fixed_anchor'];assert len(required)==4 and all(str(out) in p for p in required.artifact_path)
 for arm,obj in summaries.items():
  saved=pd.read_csv(obj['path'],float_precision='round_trip');assert len(saved)==sum(r['method']==arm for r in rows)
  expected=pd.DataFrame([r for r in rows if r['method']==arm]);pd.testing.assert_frame_equal(saved,expected,check_dtype=False,check_exact=True)
 failures=[]
 for case in ['missing_required_row','wrong_old_path','missing_new_summary','wrong_new_hash','duplicate_summary_row','wrong_denominator','undefined_nonempty_mean']:
  try:
   if case in ['duplicate_summary_row','wrong_denominator','undefined_nonempty_mean']:
    rr=copy.deepcopy(rows)
    if case=='duplicate_summary_row':rr.append(rr[0])
    if case=='wrong_denominator':rr[0]['n_evaluable']+=1
    if case=='undefined_nonempty_mean':rr[0]['mean']=None
    publish(rr,authority,out/'faults'/case)
   else:
    c=coverage.copy();ss=copy.deepcopy(summaries)
    if case=='missing_required_row':c=c.drop(c[c.metric_id=='COS:stage_fixed_anchor'].index[0])
    if case=='wrong_old_path':c.loc[c.metric_id=='COS:stage_fixed_anchor','artifact_path']='/tmp/not-an-authorized-stage-summary.csv'
    if case=='missing_new_summary':ss.pop('P2')
    if case=='wrong_new_hash':ss['P1']['sha256']='0'*64
    remap_coverage(c,q['methods'],ss)
  except ValueError:failures.append(case)
  else:raise AssertionError(case)
 assert len(failures)==7 and source_identity(SOURCES)==start
 for p,obj in inputs.items():assert identity(p)==obj
 atomic_json(out/'complete.json',dict(status='passed',execution_source=start,input_source=json.loads((integration/'execution_source.json').read_text()),n_stages=59,n_rows=354,required_coverage_rows=4,negative_cases=failures,inputs=inputs,old_summary_authority=prior,outputs={str(p.relative_to(out)):identity(p) for p in out.rglob('*') if p.is_file() and p!=out/'complete.json'}));print('real stage summary passed 59/354/4 and 7 negatives',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--integration',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.root,a.integration,a.output)
