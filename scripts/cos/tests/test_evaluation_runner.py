"""Deterministic engineering tests; never claim these are full biological runs."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from scripts.crgf.common import atomic_json,identity,process_identity
from scripts.crgf.evaluate_egr import purity
from scripts.cos import diagnostics as d
from scripts.cos import evaluation as e
from scripts.cos import runner as r


class EvaluationTests(unittest.TestCase):
    def test_purity_requires_exact_four_set(self):
        raw={m:dict(positive_F1=i+1,negative_F1=4-i) for i,m in enumerate(e.ARMS)}
        self.assertEqual(purity(dict(list(raw.items())[:2]),e.ARMS)['status'],'incomplete_comparison_set')
        self.assertEqual(purity(raw,e.ARMS)['status'],'computed')
        raw={m:dict(positive_F1=1,negative_F1=1) for m in e.ARMS}
        result=purity(raw,e.ARMS);self.assertEqual(result['status'],'not_identifiable_zero_range');self.assertNotIn('methods',result)

    def test_owner_partition_and_empty(self):
        old=np.array([0,0,1,1,1]);new=np.array([0,1,0,1,2])
        np.testing.assert_array_equal(d.classify(new,old),np.arange(5))
        self.assertEqual(d.classify(np.array([]),np.array([])).size,0)
        with self.assertRaises(ValueError):d.classify(np.array([-1]),np.array([0]))

    def test_fixed_labels_known_missing_rejected(self):
        labels=pd.Series(['T','__unavailable__:classifier_rejected'],index=[1,2]);mk={'positive':{'T':['a']},'negative':{'T':['b']}}
        f=d.fixed_scores(np.array([[10,0],[10,0],[0,10]]),[1,2,3],['a','b'],labels,mk,['a','b'])
        self.assertEqual(f.loc[1,'positive_F1'],1)
        self.assertTrue(f.loc[[2,3],'positive_F1'].isna().all())
        self.assertEqual(f.loc[2,'label_availability'],'classifier_rejected')
        self.assertEqual(f.loc[3,'label_availability'],'missing_historical_label')
        z=d.fixed_scores(np.zeros((0,2),int),[],['a','b'],labels,mk,['a','b']);self.assertEqual(len(z),0)

    def test_anchor_duplicate_and_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'labels.csv';pd.DataFrame({'cell_id':[1,1],'evaluation_label':['T','T']}).to_csv(p,index=False)
            with self.assertRaises(ValueError):d.load_anchor(identity(p))
            spec=identity(p);p.write_text('changed')
            with self.assertRaises(ValueError):d.load_anchor(spec)

    def test_boundary_empty_and_components_all_ids(self):
        mask=np.array([[1,0,1],[0,0,0],[2,2,0]],np.uint32)
        np.testing.assert_array_equal(d.components(mask,[1,2,3]),[2,1,0])
        self.assertTrue((d.boundary_bins(np.zeros((4,4),int),1)==4).all())

    def test_probe_singleton_searcharea_channel(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=root/'probes.json';atomic_json(manifest,dict(probes=[dict(shift=0,index=0)],denominator={'1':1}))
            shape=(2,2);nuc=np.array([[1,0],[0,0]]);anchor=nuc>0;raw=np.zeros(shape,int);p=root/'shift0_000000.npz'
            binary=np.full((1,2,*shape),.5);q=np.full((2,*shape),.5)
            np.savez_compressed(p,binary_logits=np.zeros_like(binary),corrected_logits=np.zeros_like(binary),base_binary_probabilities=binary,corrected_binary_probabilities=binary,residual=np.zeros((1,1,*shape)),candidate_ids=np.array([1]),candidate_valid=np.ones(1,bool),decoder_support=np.ones((1,*shape),bool),background_score=np.full(shape,.5),q=q,margin=np.zeros(shape),cell_only_margin=np.full(shape,np.nan),entropy=np.full(shape,np.log(2)),nuclei=nuc,anchor_mask=anchor,nuclear_owner_coverage=np.ones(shape,bool),raw_owner=raw,restored_owner=nuc,k=1,search_overlap=np.ones((1,*shape)),top1_id=raw,top2_id=np.ones(shape),top1_score=np.full(shape,.5),top2_score=np.full(shape,.5),cell_top1_id=np.ones(shape),cell_top2_id=np.full(shape,-1),anchor_ce=np.array([np.log(2)]))
            result=d.probe_diagnostics(dict(manifest=identity(manifest),files=[identity(p)]),root/'result');self.assertEqual(result['n_observed'],1)
            frame=pd.read_csv(root/'result/probe_strata.csv');self.assertEqual(frame.n_pixels.max(),4);self.assertEqual(frame.n_anchor_pixels.max(),1)

    def test_source_mutation_refuses_next_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=fixture(root);code=root/'scripts/cos/example.py';code.parent.mkdir(parents=True);code.write_text('old')
            for spec in cfg['methods'].values():Path(spec['config']).parent.mkdir(parents=True,exist_ok=True);Path(spec['config']).write_text('{}')
            path=root/'queue.json';atomic_json(path,cfg);atomic_json(root/'source/frozen.json',dict(worktrees={str(root):dict(files=r.source_map(root))}))
            spec=r.stage_specs(cfg)[0];state=dict(stages={})
            r.signature(cfg,spec,state,path);code.write_text('changed')
            with self.assertRaises(ValueError):r.signature(cfg,spec,state,path)

    def test_schedule_and_retry_whitelist(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=fixture(Path(tmp));stages=r.stage_specs(cfg);names=[s['name'] for s in stages]
            self.assertEqual(names[10],'pair_B1');self.assertEqual(names[11],'G_CONT_train');self.assertEqual(names[-2:],[ 'pair_G','four'])
        self.assertTrue(r.recoverable(1,'[Errno 5] Input/output error'))
        self.assertFalse(r.recoverable(1,'NaN [Errno 5] Input/output error'))
        self.assertFalse(r.recoverable(1,'CUDA out of memory'))
        self.assertFalse(r.recoverable(-15,'hash mismatch'))
        self.assertFalse(r.is_alive(dict(process_identity(),start_ticks='0')))

    def test_receipt_exit_zero_insufficient(self):
        with self.assertRaises(ValueError):r.accept('evaluate',dict(status='succeeded'))
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'x';p.write_text('x');value=dict(status='succeeded',outputs={'x':identity(p)},paper_method_complete=True,evaluation_executed=True)
            with self.assertRaises(ValueError):r.accept('evaluate',value)
            value['profiles']=e.PROFILES;r.accept('evaluate',value)
            p.write_text('y')
            with self.assertRaises(ValueError):r.accept('evaluate',value)

    def test_actual_run_source_drift_isolates_arm_and_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg,path,script=isolated_fixture(root)
            bad=Path(cfg['methods']['B1_CONT']['worktree'])/'scripts/cos/example.py'
            def command(cfg,spec,config):return [__import__('sys').executable,str(script),spec['name'],spec['stage'],spec['receipt']],cfg['methods'][spec['method'] or ('G_CONT' if spec.get('comparison')=='G' else 'B1_CONT')]['worktree']
            with patch.object(r,'preflight',side_effect=lambda *a:bad.write_text('drift after global admission')),patch.object(r,'command_for',side_effect=command):self.assertEqual(r.run(path),1)
            state=e.read(root/'run_status.json')
            self.assertEqual(state['stages']['B1_CONT_train']['status'],'failed')
            self.assertIn('source changed',state['stages']['B1_CONT_train']['failure'])
            self.assertEqual(state['stages']['pair_B1']['status'],'blocked_dependency')
            self.assertEqual(state['stages']['G_COS_evaluate']['status'],'succeeded');self.assertEqual(state['stages']['pair_G']['status'],'succeeded')
            gpair=next(s for s in r.stage_specs(cfg) if s['name']=='pair_G')
            self.assertEqual(r.command_for(cfg,gpair,path)[1],cfg['methods']['G_CONT']['worktree'])

    def test_actual_run_corrupt_receipt_isolates_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg,path,script=isolated_fixture(root)
            def command(cfg,spec,config):return [__import__('sys').executable,str(script),spec['name'],spec['stage'],spec['receipt']],str(root)
            with patch.object(r,'preflight'),patch.object(r,'command_for',side_effect=command):
                self.assertEqual(r.run(path),0)
                state=e.read(root/'run_status.json');artifact=Path(state['stages']['B1_CONT_train']['receipt']).parent/'artifact';artifact.write_text('corruption')
                self.assertEqual(r.run(path),1)
            state=e.read(root/'run_status.json');self.assertEqual(state['stages']['B1_CONT_train']['status'],'failed')
            self.assertEqual(state['stages']['pair_B1']['status'],'blocked_dependency');self.assertEqual(state['stages']['pair_G']['status'],'succeeded')
            self.assertEqual(len(state['stages']['G_CONT_train']['attempts']),1)
            with patch.object(r,'preflight'),patch.object(r,'command_for',side_effect=command) as called:self.assertEqual(r.run(path),1);self.assertEqual(called.call_count,0)

    def test_gpu_capacity_guard_pauses_before_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg,path,script=isolated_fixture(root);cfg['mini_test']=False;cfg['admission']=str(root/'admission.json');atomic_json(path,cfg)
            atomic_json(root/'admission.json',dict(resources=dict(estimated_peak_gpu_allocator_bytes=2*1024**2)))
            with patch.object(r,'preflight'),patch.object(r.subprocess,'check_output',return_value='1'),patch.object(r,'command_for') as command:
                self.assertEqual(r.run(path),1);self.assertEqual(command.call_count,0)
            state=e.read(root/'run_status.json');self.assertEqual(state['status'],'paused_resource');self.assertEqual(state['stages']['B1_CONT_train']['attempts'],[])
            self.assertEqual(state['stages']['B1_CONT_train']['gpu_capacity_check']['free_bytes'],1024**2)

    def test_interrupted_process_records_observed_lower_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg,path,script=isolated_fixture(root);first=r.stage_specs(cfg)[0]
            stale=dict(**first,status='running',attempts=[dict(start=e.times())],child_identity=dict(process_identity(),start_ticks='0'),wall_seconds_current_attempt=12.5)
            atomic_json(root/'run_status.json',dict(run_id=cfg['run_id'],stages={first['name']:stale}))
            def command(cfg,spec,config):return [__import__('sys').executable,str(script),spec['name'],spec['stage'],spec['receipt']],str(root)
            with patch.object(r,'preflight'),patch.object(r,'command_for',side_effect=command):self.assertEqual(r.run(path),0)
            state=e.read(root/'run_status.json');attempts=state['stages'][first['name']]['attempts'];self.assertEqual(len(attempts),2)
            self.assertEqual(attempts[0]['wall_seconds'],12.5);self.assertTrue(attempts[0]['wall_seconds_is_lower_bound']);self.assertIsNone(attempts[0]['exit_code'])
            self.assertEqual(e.read(root/'compute_complete.json')['wall_time_lower_bound_attempts'],[dict(stage=first['name'],attempt=1)])

    def test_finite_transient_retries_and_lock(self):
        import fcntl
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=fixture(root);path=root/'queue.json';atomic_json(path,cfg)
            with (root/'gpu.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):r.run(path)
            script=root/'fake.py';script.write_text("import sys; print('[Errno 5] Input/output error',flush=True); sys.exit(1)\n")
            def command(cfg,spec,config):return [__import__('sys').executable,str(script)],str(root)
            with patch.object(r,'preflight'),patch.object(r,'signature',return_value='signature'),patch.object(r,'command_for',side_effect=command):
                self.assertEqual(r.run(path),1)
                state=e.read(root/'run_status.json')
                for arm in e.ARMS:self.assertEqual(len(state['stages'][arm+'_train']['attempts']),3)
                self.assertEqual(r.run(path),1)
                for arm in e.ARMS:self.assertEqual(len(e.read(root/'run_status.json')['stages'][arm+'_train']['attempts']),3)

    def test_failure_isolation_actual_child_processes_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=fixture(root);path=root/'queue.json';atomic_json(path,cfg)
            script=root/'fake.py';script.write_text('''import sys,json,pathlib,hashlib,os
name,stage,receipt=sys.argv[1:];p=pathlib.Path(receipt);p.parent.mkdir(parents=True,exist_ok=True)
if name=='B1_CONT_train':
 print('schema mismatch injected',flush=True);sys.exit(2)
a=p.parent/'artifact';a.write_text(name)
r=dict(status='succeeded',outputs={'artifact':dict(path=str(a),sha256=hashlib.sha256(a.read_bytes()).hexdigest())},profiles=['engineering_nuclear_v1','engineering_pearson_v1','paper_aligned_v1'],paper_method_complete=True,evaluation_executed=True,ready=True,paper_protocol_complete=True)
p.write_text(json.dumps(r))
''')
            def command(cfg,spec,config):return [__import__('sys').executable,str(script),spec['name'],spec['stage'],spec['receipt']],str(root)
            with patch.object(r,'preflight'),patch.object(r,'signature',return_value='fixed-signature'),patch.object(r,'command_for',side_effect=command):
                self.assertEqual(r.run(path),1)
                state=e.read(root/'run_status.json')
                self.assertEqual(state['stages']['B1_CONT_train']['status'],'failed');self.assertEqual(len(state['stages']['B1_CONT_train']['attempts']),1)
                self.assertEqual(state['stages']['pair_B1']['status'],'blocked_dependency')
                self.assertEqual(state['stages']['G_COS_evaluate']['status'],'succeeded');self.assertEqual(state['stages']['pair_G']['status'],'succeeded')
                self.assertEqual(state['stages']['four']['status'],'blocked_dependency')
                self.assertFalse(e.read(root/'pair_B1/pair_ready.json')['ready'])
                self.assertEqual(e.read(root/'compute_complete.json')['status'],'partial')
                before=len(state['stages']['G_COS_evaluate']['attempts'])
                self.assertEqual(r.run(path),1)
                self.assertEqual(len(e.read(root/'run_status.json')['stages']['G_COS_evaluate']['attempts']),before)


def isolated_fixture(root):
    cfg=fixture(root)
    for arm,spec in cfg['methods'].items():
        checkout=root/'worktrees'/arm;code=checkout/'scripts/cos/example.py';code.parent.mkdir(parents=True);code.write_text('frozen')
        spec['worktree']=str(checkout);Path(spec['config']).parent.mkdir(parents=True);Path(spec['config']).write_text('{}')
    path=root/'queue.json';atomic_json(path,cfg)
    atomic_json(root/'source/frozen.json',dict(worktrees={s['worktree']:dict(files=r.source_map(s['worktree'])) for s in cfg['methods'].values()}))
    script=root/'fake_success.py';script.write_text("""import sys,json,pathlib,hashlib
name,stage,receipt=sys.argv[1:];p=pathlib.Path(receipt);p.parent.mkdir(parents=True,exist_ok=True);a=p.parent/'artifact';a.write_text(name)
p.write_text(json.dumps(dict(status='succeeded',outputs={'artifact':dict(path=str(a),sha256=hashlib.sha256(a.read_bytes()).hexdigest())},profiles=['engineering_nuclear_v1','engineering_pearson_v1','paper_aligned_v1'],paper_method_complete=True,evaluation_executed=True,ready=True,paper_protocol_complete=True)))
""")
    return cfg,path,script


def fixture(root):
    return dict(run_id='unit_test',mini_test=True,output_root=str(root),python=__import__('sys').executable,gpu_lock=str(root/'gpu.lock'),minimum_free_gb=0,cpu_workers=1,retry_backoff_seconds=0,testing_poll_fast=True,methods={m:dict(cost_root=str(root/m),matrix=str(root/m/'matrix'/m),evaluation=str(root/m/'evaluation'/m),config=str(root/m/'config.json'),worktree=str(root),commit='test') for m in e.ARMS})

if __name__=='__main__':unittest.main()
