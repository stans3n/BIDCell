import json,tempfile,unittest
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
from scripts.boundary_goal_v3 import local
from scripts.boundary_goal_v3.config import STATES,FATES
from scripts.boundary_goal_v3.status import inspect

class Contracts(unittest.TestCase):
 def test_lineage_and_complete_fate_partition(self):
  lineage=np.array([0,1,2,0]);so=np.array([1,1,1,3,2,2,1]);ta=np.array([2,2,1,2,0,1,2]);gene=np.array([2,1,1,2,2,2,0])
  self.assertEqual(local.state_codes(so,lineage,gene).tolist(),[1,0,0,2,0,0,2])
  f=local.fate_codes(so,ta,lineage,gene)
  self.assertEqual(f.tolist(),[0,1,2,3,4,1,5]);self.assertEqual(np.bincount(f,minlength=6).sum(),len(so))
 def test_stream_reconstructs_matrix_and_separate_exit_risks(self):
  with tempfile.TemporaryDirectory() as td:
   t=Path(td);frozen=t/'frozen';(frozen/'molecule_strata').mkdir(parents=True);(frozen/'interfaces').mkdir();legacy=t/'endpoint/gt/domain';legacy.mkdir(parents=True)
   src=np.array([1,1,2,2,0,3,3,1],np.uint32);dst=np.array([2,1,1,2,2,1,0,3],np.uint32);gene=np.array([1,0,0,1,1,1,0,2]);xs=np.r_[np.arange(5),np.arange(3)];ys=np.r_[np.ones(5),np.full(3,2)].astype(int)
   nuc=np.zeros((3,5),np.uint32);nuc[0,:3]=[1,2,3];npth=t/'nuc.tif';tifffile.imwrite(npth,nuc)
   co=np.zeros(8,dtype=[('transcript_id','<u8'),('x','<i4'),('y','<i4'),('gene','<i4')]);co['transcript_id']=np.arange(8)+100;co['x']=xs;co['y']=ys;co['gene']=gene;cp=t/'coords.npy';np.save(cp,co)
   np.savez(frozen/'molecule_strata/chunk_0000.npz',region_bits=np.full(8,4,np.uint16),block_1000=np.r_[np.zeros(4),np.ones(4)].astype(int),interface_id=np.array([0,0,0,0,1,1,1,1]))
   pd.DataFrame(dict(cell_i=[1,1],cell_j=[2,3])).to_csv(frozen/'interfaces/pairs.csv',index=False)
   np.savez(legacy/'domain.npz',pixels=(ys*5+xs)[:3])
   specs={}
   for name,owners in [('P',src),('T',dst)]:
    mask=nuc.copy();mask[ys,xs]=owners;mp=t/(name+'.tif');tifffile.imwrite(mp,mask);matrix=t/name;matrix.mkdir();counts=np.zeros((3,3),np.int64)
    for o,g in zip(owners,gene):
     if o:counts[o-1,g]+=1
    np.save(matrix/'counts.npy',counts);pd.DataFrame(dict(matrix_row=range(3),mask_label=[1,2,3],canonical_cell_id=[1,2,3])).to_csv(matrix/'cell_mapping.csv',index=False);pd.DataFrame(dict(gene_column=range(3),gene=['CD79A','CD8A','OTHER'])).to_csv(matrix/'gene_mapping.csv',index=False);specs[name]=dict(final=str(mp),matrix=str(matrix))
   cfg=dict(inputs=dict(coordinates=[dict(path=str(cp))],endpoints=dict(BASE_MATCH=dict(nuclei=str(npth)))),specs=dict(P2_MORPH_GT=dict(root=str(t/'endpoint'))),eligible=8,interface_expected=8)
   distance=np.full(15,255,np.uint8);distance[ys*5+xs]=[0,1,2,4,0,1,2,4]
   cells=pd.DataFrame(dict(cell_id=[1,2,3],block_1000=[0,0,1])).set_index('cell_id');ctx=dict(cells=cells,frozen=frozen,shape=nuc.shape,lineage=np.array([0,1,2,0],np.int8),gene_lineage=np.array([1,2,0],np.int8),distance=distance,genes=['CD79A','CD8A','OTHER'])
   out=t/'out';local.stream(cfg,specs,[('P','T')],out,ctx);mass,flow=local.summaries(out)
   for _,p in mass.groupby(['method','region']):self.assertEqual(p['count'].sum(),p.fixed_raw_marker_D.iloc[0])
   self.assertEqual(mass.query("method=='P' and region=='GLOBAL'").fixed_raw_marker_D.iloc[0],7)
   baseline=mass.query("method=='P' and region=='GLOBAL'").iloc[0];self.assertAlmostEqual(baseline.fixed_roi_off_target_fraction,2/7);self.assertEqual(baseline.BLCR_status,'NOT_DEFINED / NOT_APPLICABLE')
   intervals=local.intervals(dict(seed=1,bootstrap_replicates=20),out,ctx);self.assertTrue({'fixed_roi_off_target_fraction_delta','fixed_roi_consistent_fraction_delta','fixed_roi_other_unmatched_fraction_delta','fixed_roi_unassigned_fraction_delta'}.issubset(set(intervals.metric)))
   h=flow.query("kind=='HOST_EXIT' and region=='GLOBAL' and subset=='ALL_FIXED_ROI'").iloc[0];self.assertEqual((h.on_N,h.off_N,h.R_on,h.R_off),(2,2,0,1))
   r=flow.query("kind=='RECIPIENT_FATE' and region=='GLOBAL'");self.assertEqual(r['count'].sum(),6);self.assertEqual(set(r.fate),set(FATES))
   den=mass.query("method=='P'").groupby('region').fixed_raw_all_gene_D.first();self.assertEqual(den.INTERFACE_0_1+den.INTERFACE_1_2,den.INTERFACE_0_2)
 def test_missing_final_receipt_is_not_completion(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td);(p/'STATUS.json').write_text(json.dumps(dict(state='COMPLETED',run_id='fixture',pid=None,exit_code=0)))
   self.assertEqual(inspect(p)['state'],'INVALID_TERMINAL_RECEIPT')

if __name__=='__main__':unittest.main()

class OrchestrationFixes(unittest.TestCase):
 def test_legacy_callback_contract(self):
  from scripts.boundary_goal_v3.pipeline import Runner
  from scripts.rep1_triple_simple_boundary.common import progress
  with tempfile.TemporaryDirectory() as td:
   r=Runner(dict(run_dir=td,run_id='callback-fixture',source_files=[],disk_reserve_bytes=0,ram_reserve_bytes=0))
   progress(r.progress,'legacy_GT',4,10,receipt='fixture')
   got=json.loads((Path(td)/'STATUS.json').read_text())['progress'];self.assertEqual((got['name'],got['cursor'],got['total']),('legacy_GT',4,10));r.lock.close()
 def test_stage04_resumes_existing_partial_directory(self):
  from scripts.boundary_goal_v3 import stages
  from unittest.mock import patch
  with tempfile.TemporaryDirectory() as td:
   t=Path(td);(t/'stages/03').mkdir(parents=True);(t/'stages/04/stats').mkdir(parents=True)
   np.savez(t/'stages/03/block_sums.npz',x=np.array([1]));(t/'stages/03/schema.json').write_text('{}')
   with patch.object(stages.local,'context',return_value={}),patch.object(stages.local,'intervals'):
    stages.initial_ci(dict(run_dir=td),lambda *a:None);stages.initial_ci(dict(run_dir=td),lambda *a:None)
 def test_bundle_excludes_self_and_mutable_exit_files(self):
  import zipfile
  from scripts.boundary_goal_v3 import stages
  from scripts.rep1_triple_simple_boundary.common import identity
  with tempfile.TemporaryDirectory() as td:
   t=Path(td)
   names=['RUN_MANIFEST.json','EVALUATION_LOCK.json','ROI_AND_REFERENCE_PROVENANCE.md','ETA_ESTIMATE.md','MARKER_PANEL_AND_DEPENDENCY.csv','CODE_REVIEW.md','EXISTING_METHODS_MASTER_METRICS.csv','BIDCELL_STYLE_DISTANCE_CURVE.csv','BOUNDARY_LOCAL_MARKER_RESULTS.csv','FIXED_ROI_OFF_TARGET_RESULTS.csv','MARKER_AWARE_TRANSFERS.csv','MARKER_AWARE_TRANSFERS_PER_GENE.csv','B0_FINAL_LOCAL_COMPARISON.csv','METHOD_CORRESPONDENCE_AND_COVERAGE.csv','BOUNDARY_SPATIAL_INTERVALS.csv','ROI_RISKSET_COUNTS.csv','FIGURE_DATA.csv','RESULT_VALIDATION.md','RESULTS_SUMMARY.json','EVIDENCE_FOR_MANUSCRIPT.md','PRO_HANDOFF.md','status.py','check_status.sh']
   for name in names:(t/name).write_text('count\n1\n' if name.endswith('.csv') else '{}\n')
   for d in ('receipts','CODE_SNAPSHOT','stages'):(t/d).mkdir()
   (t/'OS_EXIT.json').write_text('{"exit_code":1}');(t/'process_exit.json').write_text('{}');(t/'pipeline.lock').touch()
   stages.package(dict(run_dir=td),lambda *a:None,partial=True)
   self.assertLess((t/'boundary_evidence.zip').stat().st_size,100000)
   with zipfile.ZipFile(t/'boundary_evidence.zip') as z:
    self.assertNotIn('.boundary_evidence.zip.tmp',z.namelist());self.assertNotIn('OS_EXIT.json',z.namelist());self.assertIsNone(z.testzip())
   (t/'OS_EXIT.json').write_text('{"exit_code":0}')
   lines=(t/'SHA256SUMS').read_text().splitlines()
   self.assertFalse(any('OS_EXIT' in x or 'process_exit' in x or 'pipeline.lock' in x for x in lines))
   for line in lines:
    sha,relative=line.split('  ',1);self.assertEqual(identity(t/relative)['sha256'],sha)

class AggregateResume(unittest.TestCase):
 def test_stage10_repeat_rebuilds_global_local_table_without_exit_controls(self):
  from scripts.boundary_goal_v3 import stages
  from scripts.boundary_goal_v3.config import NAME
  from scripts.rep1_triple_simple_boundary.common import identity
  with tempfile.TemporaryDirectory() as td:
   t=Path(td)
   for n in (2,3,4,7,8,9):(t/'stages'/f'{n:02d}').mkdir(parents=True)
   (t/'stages/04/stats').mkdir();(t/'stages/07/matrix').mkdir()
   def save(path,rows):pd.DataFrame(rows).to_csv(t/path,index=False)
   methods=['B0','P2_MORPH','P2_MORPH_GT',NAME];pairs=[('B0','P2_MORPH_GT'),('P2_MORPH','P2_MORPH_GT'),('P2_MORPH',NAME),(NAME,'P2_MORPH_GT')]
   mass=[dict(method=m,region='GLOBAL',state=s,count=int(s=='CONSISTENT'),fixed_raw_marker_D=1,fraction=float(s=='CONSISTENT'),captured_all_gene=5,fixed_raw_all_gene_D=10,capture=.5) for m in methods for s in STATES]
   # beta0 interface row is also needed to register its global/interface capture.
   mass += [dict(x,region='INTERFACE_0_2') for x in mass]
   fate=[dict(source=a,target=b,region='GLOBAL',kind='RECIPIENT_FATE',fate=f,count=int(f=='FAVORABLE_PROXY'),all_transferred=1) for a,b in pairs for f in FATES]
   curve=[dict(method=m,population='fixed',direction='T_marker_in_B',marker='CD4',bin_index=0,n_eligible=1,n_positive=0,fraction=0) for m in methods]
   risk=[dict(method=m,region='GLOBAL',raw_marker_D=1) for m in methods];ci=[dict(source=a,target=b,region='GLOBAL',metric='fixed_roi_off_target_fraction_delta',estimate=0) for a,b in pairs]
   for n in ('03','09'):
    save(f'stages/{n}/mass.csv',mass);save(f'stages/{n}/flows.csv',fate);save(f'stages/{n}/risk.csv',risk);save(f'stages/{n}/per_gene.csv',[dict(gene='CD4',on_N=1,off_N=1)]);save(f'stages/{n}/distance.csv',curve);save(f'stages/{n}/intervals.csv',ci)
   save('stages/02/distance.csv',curve);save('stages/04/stats/intervals.csv',ci);save('stages/03/correspondence.csv',[dict(method='B0',coverage=1)])
   save('stages/09/paired_global_F1.csv',[dict(source='P2_MORPH',target=NAME,metric='positive_F1',source_mean=.3,target_mean=.31,delta=.01)])
   save('EXISTING_METHODS_MASTER_METRICS.csv',[dict(method='B0',capture_pct=50)])
   np.save(t/'stages/07/matrix/counts.npy',np.array([[5]],np.int64));(t/'stages/07/matrix/complete.json').write_text('{}');(t/'stages/07/matrix/expr_mat.csv').write_text('cell_id,gene\n1,5\n')
   (t/'stages/08/complete.json').write_text(json.dumps(dict(structure=dict(n_cells=1),capture=dict(assigned=5),n_cells=1)))
   (t/'OS_EXIT.json').write_text('{"exit_code":1}');(t/'process_exit.json').write_text('{}')
   cfg=dict(run_dir=td,eligible=10,interface_expected=10)
   a=stages.aggregate(cfg,lambda *args:None);before=identity(t/'G_ONLY_MATCHED_RESULTS.csv')['sha256'];b=stages.aggregate(cfg,lambda *args:None)
   self.assertEqual(before,identity(t/'G_ONLY_MATCHED_RESULTS.csv')['sha256']);self.assertFalse(any(p.name in ('OS_EXIT.json','process_exit.json') for p in a+b))

class ExitIdentity(unittest.TestCase):
 def test_previous_dry_run_exit_cannot_validate_current_completion(self):
  with tempfile.TemporaryDirectory() as td:
   t=Path(td);(t/'receipts').mkdir();(t/'STATUS.json').write_text(json.dumps(dict(state='COMPLETED',run_id='fixture',pid=None,ppid=456,started_epoch=20,exit_code=0)));(t/'receipts/final.json').write_text(json.dumps(dict(exit_code=0,outputs=[])));(t/'OS_EXIT.json').write_text(json.dumps(dict(exit_code=0,shell_pid=123,finished_epoch=10)))
   self.assertEqual(inspect(t)['state'],'INVALID_TERMINAL_RECEIPT')
