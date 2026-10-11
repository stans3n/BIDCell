from scripts.bs4.tables import read_table
"""Recovery regression tests; synthetic asymmetry plus actual 103000-ID tables."""
import argparse,tempfile,unittest,json
from pathlib import Path
import numpy as np,pandas as pd
from scripts.bs4.stage_population import align_stage,population_index,load_population,postprocess_increment
from scripts.cos.diagnostics import METRICS,load_anchor
from scripts.crgf.common import atomic_json,identity,sha256,source_identity


def table(ids,values=None):
 d=pd.DataFrame({'cell_id':ids,'anchor_label':['T']*len(ids),'label_availability':['known']*len(ids)})
 for metric in METRICS:d[metric]=values if values is not None else np.arange(len(ids))*.1
 return d

class Tests(unittest.TestCase):
 def test_asymmetric_ids_preserve_order_and_missing_NA(self):
  labels=pd.Series({1:'T',2:'T',3:'T'});x,n=align_stage(table([3,1,99]),[2,1,3],labels)
  self.assertEqual(x.index.tolist(),[2,1,3]);self.assertEqual(n['n_extra_stage_ids'],1);self.assertEqual(n['n_missing_stage_rows'],1)
  self.assertTrue(x.loc[2,METRICS].isna().all());self.assertEqual(x.loc[2,METRICS[0]+'_reason'],'missing_stage_id')
  self.assertEqual(x.loc[1,METRICS[0]],.1)
 def test_zero_finite_is_not_missing_and_NA_reasons(self):
  labels=pd.Series({1:'T',2:'T',3:'T',4:'__unavailable__:classifier_rejected'});t=table([1,2,3,4],[0,np.nan,np.inf,.4]);t.loc[3,'anchor_label']=labels[4];t.loc[3,'label_availability']='classifier_rejected'
  x,_=align_stage(t,[1,2,3,4],labels);self.assertEqual(x.loc[1,METRICS[0]],0)
  self.assertEqual(x[METRICS[0]+'_reason'].tolist(),['available','undefined_original_metric','nonfinite_original_metric','label_ineligible']);self.assertTrue(x.loc[[2,3,4],METRICS].isna().all().all())
 def test_duplicate_invalid_population_and_label_drift(self):
  labels=pd.Series({1:'T',2:'T'})
  for ids in ([1,1],[1,1.5],[1,np.nan],[0,1]):
   with self.assertRaises(ValueError):population_index(ids)
  with self.assertRaisesRegex(ValueError,'duplicate'):align_stage(table([1,1]),[1,2],labels)
  t=table([1,2]);t.loc[1,'anchor_label']='wrong'
  with self.assertRaisesRegex(ValueError,'frozen historical label'):align_stage(t,[1,2],labels)
 def test_population_hash_drift(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'population.csv';pd.DataFrame({'cell_id':[3,1,2]}).to_csv(p,index=False);spec=identity(p);self.assertEqual(load_population(spec).tolist(),[3,1,2]);p.write_text('cell_id\n1\n2\n')
   with self.assertRaisesRegex(ValueError,'population drift'):load_population(spec)
 def test_joint_four_endpoint_denominator_and_empty(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);anchor=root/'anchor.csv';pd.DataFrame({'cell_id':[1,2,3,4],'cell_type':['T']*4}).to_csv(anchor,index=False);methods={}
   for arm in ['S','B']:
    ev=root/arm;d=ev/'diagnostics/stages';d.mkdir(parents=True);methods[arm]={'evaluation':str(ev)}
    for stage in ['pre','final']:
     ids=[1,2,3] if arm=='S' else [1,2,4];v=[.8,.5,.4] if arm=='S' else [.2,.1,.2]
     frame=table(ids,v)
     if arm=='B' and stage=='final':frame.loc[1,METRICS]=np.nan
     if arm=='S' and stage=='final':frame.loc[0,METRICS]=.9
     frame.to_csv(d/f'fixed_B1_{stage}.csv',index=False,na_rep='NA')
   out=root/'out';postprocess_increment(methods,{'B1':identity(anchor)},[('S','B')],[4,2,1,3],out)
   rows=pd.read_csv(out/'postprocess_cos_increment.csv');self.assertTrue((rows.n_population==4).all());self.assertTrue((rows.n_evaluable==1).all());self.assertTrue((rows.n_missing_stage_id==2).all());self.assertTrue((rows.n_undefined_original_metric==1).all());np.testing.assert_allclose(rows.pre_cos_increment,.6);np.testing.assert_allclose(rows.final_cos_increment,.7);np.testing.assert_allclose(rows.postprocess_rewrite,.1)
   cells=read_table(out/'postprocess_S_minus_B_B1_per_cell.csv');self.assertEqual(cells.cell_id.tolist(),[4,2,1,3]);self.assertTrue(cells.loc[cells.cell_id!=1,METRICS[0]+'_postprocess_rewrite'].isna().all())
   postprocess_increment(methods,{'B1':identity(anchor)},[('S','B')],[3,4],root/'empty');empty=pd.read_csv(root/'empty/postprocess_cos_increment.csv');self.assertTrue((empty.n_evaluable==0).all());self.assertTrue(empty.pre_cos_increment.isna().all())


def actual(root,out):
 q=json.loads((root/'configs/queue.json').read_text());methods={**q['historical_methods'],**q['methods']};ids=load_population(q['population']);assert len(ids)==103000
 pairs=[('S','B'),('SP','S'),('SP','B'),('S','A'),('SP','A')]
 postprocess_increment(methods,q['anchors'],pairs,ids,out)
 summary=pd.read_csv(out/'postprocess_cos_increment.csv');assert len(summary)==30 and (summary.n_population==103000).all()
 for a,b in pairs:
  cells=read_table(out/f'postprocess_{a}_minus_{b}_B1_per_cell.csv');assert cells.cell_id.tolist()==ids.tolist()
  for metric in METRICS:
   good=cells[metric+'_joint_evaluable'];row=summary.loc[(summary.comparison==a+'-'+b)&(summary.metric==metric)].iloc[0]
   assert int(good.sum())==row.n_evaluable and int((~good).sum())==row.n_unavailable
   assert cells.loc[~good,metric+'_postprocess_rewrite'].isna().all()
   if good.any():np.testing.assert_allclose(cells.loc[good,metric+'_postprocess_rewrite'].mean(),row.postprocess_rewrite,rtol=1e-12,atol=1e-14)
 return dict(population_count=len(ids),summary_rows=len(summary),all_pairs_preserve_exact_order=True,all_missing_scores_NA=True,domains=pd.read_csv(out/'stage_population_domains.csv').to_dict('records'))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);args=p.parse_args();root=Path(args.root);sources=['scripts/bs4/stage_population.py','scripts/bs4/tables.py','scripts/bs4/tests/report_population.py','scripts/cos/diagnostics.py'];execution_source=source_identity(sources);r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
 if not r.wasSuccessful():raise SystemExit(1)
 out=root/'tests/report_population_gzip_v2';result=actual(Path('/home/zwan0726/work/outputs/cos_ssp_20261011T005741+1100'),out)
 if source_identity(sources)!=execution_source:raise ValueError('population regression execution source changed')
 atomic_json(root/'tests/report_population_gzip_v2.json',dict(execution_source=execution_source,status='passed',tests_run=r.testsRun,actual=result,outputs={str(p.relative_to(out)):identity(p) for p in out.rglob('*') if p.is_file()}));print('Actual frozen population stage validation passed',flush=True)
