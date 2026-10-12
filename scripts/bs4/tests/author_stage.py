"""Strict author-stage count contracts and per-metric four-endpoint domains."""
from pathlib import Path
import argparse,copy,json
import numpy as np,pandas as pd
from scripts.bs4.author_marker import METRICS,fixed_scores
from scripts.bs4.author_stage import recompute_stage,load_stage_table
from scripts.bs4.stage_population import postprocess_increment
from scripts.bs4.tests.population_missing import synthetic
from scripts.crgf.common import atomic_json,identity,source_identity
SOURCES=['scripts/bs4/author_marker.py','scripts/bs4/author_stage.py','scripts/bs4/stage_population.py','scripts/bs4/comparison.py','scripts/bs4/diagnostics.py','scripts/bs4/tests/author_stage.py','scripts/bs4/tests/population_missing.py']
def run(root,out):
 root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True);start=source_identity(SOURCES);genes=['g'+str(i) for i in range(307)];markers={'positive':{'T':['g0','g1','g2']},'negative':{'T':['g5','g6','g7']}};labels=pd.Series({1:'T',2:'T',3:'T',4:'__unavailable__:classifier_rejected',5:'T'});prep=out/'prepared';prep.mkdir();anchor=out/'anchor.csv';labels.rename('cell_type').rename_axis('cell_id').to_csv(anchor);atomic_json(prep/'markers.json',markers);(prep/'reference_genes.txt').write_text('\n'.join(genes)+'\n');atomic_json(prep/'complete.json',dict(outputs={p.name:identity(p) for p in prep.iterdir()}));methods={}
 for arm in ['P1','B']:
  ev=out/arm/'evaluation';d=ev/'diagnostics/stages';d.mkdir(parents=True);matrix=out/arm/'matrix';matrix.mkdir();pd.DataFrame({'gene_column':range(307),'gene':genes}).to_csv(matrix/'gene_mapping.csv',index=False);atomic_json(matrix/'complete.json',dict(outputs={'gene_mapping.csv':identity(matrix/'gene_mapping.csv')}));tables={};counts_manifest={}
  for stage in ['pre','final']:
   ids=np.array([1,2,4] if arm=='P1' and stage=='final' else [1,2,3,4]);counts=np.zeros((len(ids),307),dtype=np.int64);counts[ids!=1,0]=10;counts[ids!=1,5]=5
   table=fixed_scores(counts,ids,genes,labels,markers,genes).reset_index();table.loc[table.zero_reference_counts,METRICS]=np.nan # legacy defective table, corrected only from actual counts
   p=d/f'fixed_B1_{stage}.csv';table.to_csv(p,index=False,na_rep='NA');tables[str(p)]=identity(p);p=d/f'counts_{stage}.npz';np.savez_compressed(p,cell_id=ids,counts=counts);counts_manifest[stage]=identity(p)
  atomic_json(d/'validation.json',dict(status='succeeded',integer_conservation=True,count_artifacts=counts_manifest));atomic_json(ev/'complete.json',dict(outputs=tables));methods[arm]=dict(evaluation=str(ev),matrix=str(matrix))
 postprocess_increment(methods,{'B1':identity(anchor)},[('P1','B')],[3,2,1,4,5],out/'four_endpoint',prepared=prep)
 frame=pd.read_csv(out/'four_endpoint/postprocess_cos_increment.csv');assert (frame.n_population==5).all()
 for _,row in frame.iterrows():assert row.n_evaluable==(1 if row.metric.endswith('Precision') else 2)
 cells=pd.read_csv(out/'four_endpoint/postprocess_P1_minus_B_B1_per_cell.csv.gz').set_index('cell_id');assert cells.loc[1,'positive_F1_joint_evaluable'] and cells.loc[1,'positive_Recall_joint_evaluable'] and not cells.loc[1,'positive_Precision_joint_evaluable'];assert cells.loc[1,'positive_F1_postprocess_rewrite']==0
 assert cells.loc[[3,5],'positive_F1_reason'].eq('missing_stage_id').all() and cells.loc[4,'positive_F1_reason']=='label_ineligible'
 postprocess_increment(methods,{'B1':identity(anchor)},[('P1','B')],[3,5],out/'empty_domain',prepared=prep);assert pd.read_csv(out/'empty_domain/postprocess_cos_increment.csv').n_evaluable.eq(0).all()
 raw=pd.read_csv(out/'B/evaluation/diagnostics/stages/fixed_B1_pre.csv');saved=np.load(out/'B/evaluation/diagnostics/stages/counts_pre.npz');ids=saved['cell_id'];counts=saved['counts'];failures=[]
 for case in ['duplicate_stage_id','duplicate_count_id','duplicate_gene','missing_reference_gene','missing_counts_row','negative_count','fractional_count','wrong_zero_flag']:
  t=raw.copy();ii=ids.copy();xx=counts.copy();gg=genes.copy();rr=genes.copy()
  if case=='duplicate_stage_id':t.loc[1,'cell_id']=t.loc[0,'cell_id']
  if case=='duplicate_count_id':ii[1]=ii[0]
  if case=='duplicate_gene':gg[-1]=gg[0]
  if case=='missing_reference_gene':rr[-1]='absent'
  if case=='missing_counts_row':ii=ii[:-1];xx=xx[:-1]
  if case=='negative_count':xx[0,0]=-1
  if case=='fractional_count':xx=xx.astype(float);xx[0,0]=.5
  if case=='wrong_zero_flag':t.loc[0,'zero_reference_counts']=False
  try:recompute_stage(t,ii,xx,gg,rr,markers)
  except ValueError:failures.append(case)
  else:raise AssertionError(case)
 small=synthetic();q=json.loads((root/'configs/queue.json').read_text());real,provenance=load_stage_table(q['historical_methods']['S'],'B1','pre',q['prepared']);knownzero=real.zero_reference_counts & real.label_availability.eq('known');assert real.loc[knownzero,['positive_F1','negative_F1','positive_Recall','negative_Recall']].eq(0).all().all();assert real.loc[knownzero,['positive_Precision','negative_Precision']].isna().all().all()
 assert source_identity(SOURCES)==start
 atomic_json(out/'complete.json',dict(status='passed',execution_source=start,negative_cases=failures,synthetic_population=small,independent_metric_finite_domains=True,actual_S_pre_known_zero_count=int(knownzero.sum()),actual_stage_inputs=provenance,outputs={str(p.relative_to(out)):identity(p) for p in out.rglob('*') if p.is_file() and p!=out/'complete.json'}));print('author stage passed',len(failures),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.root,a.output)
