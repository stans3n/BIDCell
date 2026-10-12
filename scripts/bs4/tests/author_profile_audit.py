"""Audit existing profile scoring from immutable own counts and frozen labels; no classifier."""
from pathlib import Path
import argparse,json,subprocess
import numpy as np
import pandas as pd
from scripts.bs4.author_marker import METRICS,marker_metrics
from scripts.crgf.common import identity,sha256,atomic_json,source_identity
SOURCES=['scripts/bs4/author_marker.py','scripts/bs4/tests/author_profile_audit.py','scripts/bs4/tests/author_marker_fixture.R']
def read(p):return json.loads(Path(p).read_text())
def assert_declared(p,receipt):
 matches=[v for v in receipt['outputs'].values() if Path(v['path']).resolve()==Path(p).resolve()]
 if len(matches)!=1 or matches[0]['sha256']!=sha256(p):raise ValueError('input authority mismatch '+str(p))
def run(root,out):
 root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True);start=source_identity(SOURCES);q=read(root/'configs/queue.json');methods={**q['historical_methods'],**q['methods']};ref=read(q['reference']);prepared=Path(q['prepared']);paper_markers=read(prepared/'markers.json');refs=(prepared/'reference_genes.txt').read_text().splitlines();report=read(root/'comparison/complete.json');inputs={};rows=[]
 for p in [root/'configs/queue.json',Path(q['reference']),prepared/'complete.json',Path('/home/zwan0726/work/outputs/bidcell_metric_protocol_review_20261008/publish_expression_metrics.R')]:inputs[str(p)]=identity(p)
 for name in ['markers.json','reference_genes.txt']:
  assert_declared(prepared/name,read(prepared/'complete.json'));inputs[str(prepared/name)]=identity(prepared/name)
 admission=root/'tests/admission.json';review=read(root/'source/review_receipt.json')
 if review['evidence'][str(admission)]!=sha256(admission):raise ValueError('original admission identity drift')
 freeze=root/'T1/configs/frozen.json'
 if read(admission)['arm_frozen']['T1']['sha256']!=sha256(freeze):raise ValueError('frozen marker authority drift')
 frozen=read(freeze)['identities'];inputs[str(admission)]=identity(admission);inputs[str(freeze)]=identity(freeze)
 if frozen['config_reference.json']['sha256']!=sha256(q['reference']):raise ValueError('frozen reference configuration differs')
 for name in ['positive_markers','negative_markers','reference']:
  declared=frozen['input_'+name];p=Path(ref[name])
  if Path(declared['path']).resolve()!=p.resolve() or declared['sha256']!=sha256(p):raise ValueError('frozen engineering marker/reference changed')
  inputs[str(p)]=identity(p)
 for arm,spec in methods.items():
  matrix=Path(spec['matrix']);ev=Path(spec['evaluation']);mr=read(matrix/'complete.json');er=read(ev/'complete.json')
  for kind,p in [('matrix',matrix/'complete.json'),('evaluate',ev/'complete.json')]:
   declared=report['arm_receipts'][arm][kind];assert declared['sha256']==sha256(p);inputs[str(p)]=identity(p)
  for name in ['counts.npy','cell_mapping.csv','gene_mapping.csv']:assert_declared(matrix/name,mr);inputs[str(matrix/name)]=identity(matrix/name)
  ids=pd.read_csv(matrix/'cell_mapping.csv').canonical_cell_id.to_numpy();genes=pd.read_csv(matrix/'gene_mapping.csv').gene.tolist();x=np.load(matrix/'counts.npy',mmap_mode='r');assert x.shape==(len(ids),len(genes)) and not pd.Index(ids).has_duplicates
  prpath=ev/'profiles_complete.json';assert_declared(prpath,er);pr=read(prpath);inputs[str(prpath)]=identity(prpath)
  if pr['dependencies']['prepared']['sha256']!=sha256(prepared/'complete.json'):raise ValueError('frozen scoring reference drift')
  engineering=pr['dependencies']['engineering_reference']
  if engineering['sha256']!=sha256(engineering['path']) or engineering['sha256']!=sha256(q['reference']):raise ValueError('published engineering reference configuration drift')
  inputs[engineering['path']]=identity(engineering['path'])
  for profile in ['remote_nuclear_annotation_raw','windows_style_maxpearson_raw','paper_aligned_v1']:
   if profile=='paper_aligned_v1':
    folder=ev/profile;tablefile=folder/'markers_per_cell.csv';table=pd.read_csv(tablefile);ann=pd.read_csv(folder/'annotation.csv');sf=pd.read_csv(folder/'size_factors.csv');qc=pd.read_csv(folder/'qc_all_cells.csv')
    for p in [tablefile,folder/'annotation.csv',folder/'size_factors.csv',folder/'qc_all_cells.csv']:assert_declared(p,pr);inputs[str(p)]=identity(p)
    assert np.array_equal(qc.cell_id,ids) and np.array_equal(table.cell_id,qc.loc[qc.qc_pass,'cell_id']) and np.array_equal(ann.cell_id,table.cell_id) and np.array_equal(sf.cell_id,table.cell_id)
    assert np.array_equal(ann.evaluation_label,table.cell_type) and ann.evaluation_label.notna().all() and np.isfinite(sf.size_factor).all() and (sf.size_factor>0).all()
    idx=pd.Index(ids).get_indexer(table.cell_id);assert np.all(idx>=0);expr=np.log2(x[idx][:,pd.Index(genes).get_indexer(refs)]/sf.size_factor.to_numpy()[:,None]+1);gg=refs;markers=paper_markers;eligible=np.ones(len(table),bool)
   else:
    folder=ev/'engineering/cellspa'/profile;tablefile=folder/'official_markers_per_cell.csv';table=pd.read_csv(tablefile);assert_declared(tablefile,pr);inputs[str(tablefile)]=identity(tablefile);assert np.array_equal(ids,table.cell_id)
    labelpath=ev/'engineering/structure/annotation_nuclear.csv' if profile=='remote_nuclear_annotation_raw' else ev/'engineering/cellspa/official_expression_max_pearson.csv'
    assert_declared(labelpath,pr);inputs[str(labelpath)]=identity(labelpath);labels=pd.read_csv(labelpath)
    labelcol='cell_type' if profile=='remote_nuclear_annotation_raw' else 'celltype_correlation'
    if not np.array_equal(labels.cell_id,table.cell_id) or not labels[labelcol].fillna('__missing__').equals(table.cell_type.fillna('__missing__')):raise ValueError('frozen profile label drift')
    expr=np.asarray(x);gg=genes;markers={}
    for sign in ['positive','negative']:
     p=Path(ref[sign+'_markers']);m=pd.read_csv(p,index_col=0);markers[sign]={str(k):m.columns[row.to_numpy()>0].tolist() for k,row in m.iterrows()};inputs[str(p)]=identity(p)
    eligible=table.cell_type.isin(set(markers['positive'])&set(markers['negative'])).to_numpy()
   actual=table[METRICS].to_numpy(float);expected=np.full_like(actual,np.nan)
   for j,sign in enumerate(['positive','negative']):expected[eligible,j*3:j*3+3]=marker_metrics(expr[eligible],gg,table.cell_type.to_numpy()[eligible],markers[sign])
   same_na=np.isnan(actual)==np.isnan(expected);same_inf=np.isinf(actual)==np.isinf(expected);finite=np.isfinite(actual)&np.isfinite(expected);delta=np.abs(actual[finite]-expected[finite]);numerical=np.isclose(actual,expected,atol=5e-15,rtol=0,equal_nan=True)
   for j,metric in enumerate(METRICS):rows.append(dict(method=arm,profile=profile,metric=metric,n_population=len(table),n_label_eligible=int(eligible.sum()),n_finite_author=int(np.isfinite(expected[:,j]).sum()),n_finite_original=int(np.isfinite(actual[:,j]).sum()),n_semantic_difference=int((~numerical[:,j]).sum()),n_zero_reference=int(np.all(expr==0,axis=1).sum()),n_valid_zero_author=int((expected[:,j]==0).sum()),max_abs_difference=float(np.max(np.abs(actual[finite[:,j],j]-expected[finite[:,j],j]),initial=0))))
   if not numerical.all():
    corrected=table.copy();corrected[METRICS]=expected;corrected.to_csv(out/f'{arm}_{profile}_corrected.csv',index=False,na_rep='NA')
   # Direct archived R helper on a deterministic spread of actual complete rows.
   sample=np.flatnonzero(eligible)[np.linspace(0,max(0,int(eligible.sum())-1),min(64,int(eligible.sum())),dtype=int)]
   assert len(sample)
   for sign in ['positive','negative']:
    f=out/f'{arm}_{profile}_{sign}_fixture.json';target=out/f'{arm}_{profile}_{sign}_R.csv';atomic_json(f,dict(expression=expr[sample].tolist(),genes=gg,ids=table.cell_id.iloc[sample].astype(str).tolist(),labels=table.cell_type.iloc[sample].tolist(),markers=markers[sign]))
    cmd=['/home/zwan0726/work/envs/bidcell_egr_r/bin/Rscript',str(Path(__file__).with_name('author_marker_fixture.R')),'/home/zwan0726/work/outputs/bidcell_metric_protocol_review_20261008/publish_expression_metrics.R',str(f),str(target)]
    with target.with_suffix('.log').open('w') as log:subprocess.run(cmd,check=True,timeout=180,stdout=log,stderr=subprocess.STDOUT)
    result=pd.read_csv(target,float_precision='round_trip');j=0 if sign=='positive' else 3;np.testing.assert_allclose(result[['F1','Precision','Recall']],expected[sample,j:j+3],atol=0,rtol=0,equal_nan=True)
   print('audited',arm,profile,flush=True)
 frame=pd.DataFrame(rows);frame.to_csv(out/'profile_audit.csv',index=False);assert source_identity(SOURCES)==start
 for p,obj in inputs.items():assert identity(p)==obj
 atomic_json(out/'complete.json',dict(status='passed',execution_source=start,scope='all 18 frozen profiles checked; actual author R scoring only, no scClassify or upstream execution',numerical_serialization_tolerance=5e-15,semantic_differences=int(frame.n_semantic_difference.sum()),inputs=inputs,outputs={p.name:identity(p) for p in out.iterdir() if p.is_file() and p.name!='complete.json'}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.root,a.output)
