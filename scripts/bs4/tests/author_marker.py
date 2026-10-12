"""Run the archived author's actual R function and compare independent Python adapter."""
from pathlib import Path
import argparse,json,subprocess
import numpy as np
import pandas as pd
from scripts.bs4.author_marker import marker_metrics,fixed_scores
from scripts.crgf.common import atomic_json,identity,source_identity
SOURCES=['scripts/bs4/author_marker.py','scripts/bs4/tests/author_marker.py','scripts/bs4/tests/author_marker_fixture.R']
def run(out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);start=source_identity(SOURCES)
 official=Path('/home/zwan0726/work/outputs/bidcell_metric_protocol_review_20261008/publish_expression_metrics.R')
 assert identity(official)['sha256']=='2d7f0e0da83fed939e5cc2316ea159f5962477e214d00685c73f0f01142775ec'
 rng=np.random.default_rng(20261012);genes=['g'+str(i) for i in range(13)];x=np.vstack([np.zeros(13),[8,0,0,0,0,0,0,0,0,0,0,0,0],np.zeros(13),rng.poisson(1,(101,13))]);labels=['Known','Known','Unmatched','Empty']+['Known' if i%2 else 'Unmatched' for i in range(100)];markers={'Known':['g0','g3','g5'],'Empty':[]};ids=[str(i+1) for i in range(len(x))]
 fixture=dict(genes=genes,ids=ids,labels=labels,markers=markers,expression=x.tolist());atomic_json(out/'fixture.json',fixture)
 cmd=['/home/zwan0726/work/envs/bidcell_egr_r/bin/Rscript',str(Path(__file__).with_name('author_marker_fixture.R')),str(official),str(out/'fixture.json'),str(out/'R_author.csv')]
 with (out/'R.log').open('w') as f:subprocess.run(cmd,check=True,timeout=180,stdout=f,stderr=subprocess.STDOUT)
 r=pd.read_csv(out/'R_author.csv',float_precision='round_trip');actual=marker_metrics(x,genes,labels,markers);np.testing.assert_allclose(actual,r[['F1','Precision','Recall']].to_numpy(),atol=0,rtol=0,equal_nan=True)
 assert actual[0,0]==0 and np.isnan(actual[0,1]) and actual[0,2]==0
 assert np.array_equal(actual[2],[0,0,0]);assert actual[3,0]==0 and np.isnan(actual[3,1:]).all()
 failures=[]
 for name,xx,gg,ll in [('missing_gene',x,genes[:-1],labels),('missing_row',x[:-1],genes,labels),('nan_expression',np.where(np.indices(x.shape)[0]==0,np.nan,x),genes,labels),('missing_label',x,genes,[None]+labels[1:]),('duplicate_gene',x,genes[:-1]+['g0'],labels)]:
  try:marker_metrics(xx,gg,ll,markers)
  except ValueError:failures.append(name)
  else:raise AssertionError(name)
 fixed=fixed_scores(x[:3],[1,2,3],genes,pd.Series(['Known','__unavailable__:classifier_rejected'],index=[1,2]),{'positive':markers,'negative':markers},genes)
 assert fixed.loc[1,'positive_F1']==0 and fixed.loc[1,'positive_Recall']==0 and np.isnan(fixed.loc[1,'positive_Precision'])
 assert fixed.loc[[2,3],['positive_F1','positive_Precision','positive_Recall']].isna().all().all()
 assert source_identity(SOURCES)==start
 atomic_json(out/'complete.json',dict(status='passed',execution_source=start,official_source=identity(official),actual_R_cases=len(x),strict_negative_cases=failures,fixed_historical_ineligible_preserved=True,outputs={p.name:identity(p) for p in out.iterdir() if p.is_file() and p.name!='complete.json'}))
 print('author fixture passed',len(x),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();run(a.output)
