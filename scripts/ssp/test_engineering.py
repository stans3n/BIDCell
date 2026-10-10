"""Actual child-process orchestration tests; synthetic payloads are explicit."""
import fcntl,json,sys,tempfile,unittest,subprocess
from pathlib import Path
from unittest.mock import patch
from scripts.crgf.common import atomic_json,identity,process_identity,sha256
from . import runner as r
from .evaluation import read

def fixture(root):
    q=dict(run_id='SSP-engineering-test',mini_test=True,output_root=str(root),python=sys.executable,gpu_lock=str(root/'gpu.lock'),minimum_free_gb=0,cpu_workers=1,retry_backoff_seconds=0,testing_poll_fast=True,stage_timeouts={k:10 for k in r.TIMEOUTS},methods={a:dict(cost_root=str(root/a),matrix=str(root/a/'matrix'/a),evaluation=str(root/a/'evaluation'/a),config=str(root/a/'config.json'),worktree=str(root),commit='test') for a in ('S','SP')})
    for s in q['methods'].values():Path(s['config']).parent.mkdir();Path(s['config']).write_text('{}')
    p=root/'queue.json';atomic_json(p,q);atomic_json(root/'source/frozen.json',dict(worktrees={str(root):dict(files=r.source_map(root))}))
    script=root/'payload.py';script.write_text("""import sys,json,pathlib,hashlib
name,stage,receipt=sys.argv[1:];p=pathlib.Path(receipt);p.parent.mkdir(parents=True,exist_ok=True)
if (p.parents[2]/'fail_S').exists() and name=='S_train':print('schema mismatch');sys.exit(2)
a=p.parent/'artifact';a.write_text(name)
p.write_text(json.dumps(dict(status='succeeded',outputs={'artifact':dict(path=str(a),sha256=hashlib.sha256(a.read_bytes()).hexdigest())},profiles=['engineering_nuclear_v1','engineering_pearson_v1','paper_aligned_v1'],paper_method_complete=True,evaluation_executed=True,ready=True,paper_protocol_complete=True)))
""")
    return q,p,script

class Tests(unittest.TestCase):
    def test_fresh_method_provenance_and_real_input_freeze(self):
        from .admission import capture_method_sources
        from .entry import freeze_inputs
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);config=root/'configs/queue.json';payload=root/'input';payload.write_text('fixture')
            prepared=root/'prepared/manifest.json';atomic_json(prepared,{'outputs':{},'global_blocks':[]})
            for rel in ('prepared/coverage.json','prepared/coverage_views.json','diagnostics/calibration.json','configs/training_probes.json'):atomic_json(root/rel,{})
            source=capture_method_sources(root);before=source.read_bytes();capture_method_sources(root);self.assertEqual(before,source.read_bytes())
            methods={}
            for arm in ('S','SP'):
                ar=root/arm;run=ar/'configs/run.json';machine=ar/'configs/machine.json'
                atomic_json(run,{'prepared_manifest':str(prepared)});atomic_json(ar/'configs/inputs.json',{})
                atomic_json(machine,{k:str(payload) for k in ('baseline_final_mask','nuclei_mask','preannotation','genes','transcripts','reference','positive_markers','negative_markers')})
                methods[arm]={'worktree':str(Path(__file__).resolve().parents[2]),'cost_root':str(ar),'machine':str(machine),'config':str(run)}
            atomic_json(config,{'output_root':str(root),'methods':methods,'population':identity(payload)})
            freeze_inputs(config)
            for arm in methods:self.assertEqual(read(root/arm/'configs/frozen.json')['identities']['method_execution_sources']['sha256'],sha256(source))
            changed=read(source);changed['source'][next(iter(changed['source']))]='tampered';atomic_json(source,changed)
            with self.assertRaisesRegex(ValueError,'method execution source changed'):capture_method_sources(root)
    def test_one_command_preparation_and_idempotency(self):
        from .entry import prepare_launch
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);calls=[];config=root/'configs/queue.json';admission=root/'admission.json'
            def setup(_):calls.append('bootstrap');atomic_json(config,{'admission':str(admission)})
            def tests(_):calls.append('missing_tests')
            def inputs(_):calls.append('input_freeze')
            def freeze(_):calls.append('source_freeze');atomic_json(root/'source/frozen.json',{'fixture':'frozen'})
            def collect(_):calls.append('collect');atomic_json(admission,{'status':'passed'});return {'required_failures':[]}
            with patch('scripts.ssp.bootstrap.prepare',side_effect=setup),patch('scripts.ssp.admission.execute_missing',side_effect=tests),patch('scripts.ssp.entry.freeze_inputs',side_effect=inputs),patch('scripts.ssp.runner.freeze',side_effect=freeze),patch('scripts.ssp.admission.collect',side_effect=collect):
                self.assertEqual(prepare_launch(root),config)
                self.assertEqual(calls,['bootstrap','missing_tests','input_freeze','source_freeze','collect'])
                snapshots={p:p.read_bytes() for p in (config,admission,root/'source/frozen.json')};calls.clear()
                self.assertEqual(prepare_launch(root),config);self.assertEqual(calls,[])
                self.assertTrue(all(p.read_bytes()==b for p,b in snapshots.items()))
                self.assertFalse((root/'tests/independent_review/approval.json').exists())
                atomic_json(admission,{'status':'failed'})
                with patch('scripts.ssp.admission.collect',return_value={'required_failures':['independent review missing']}):
                    with self.assertRaisesRegex(ValueError,'required admission failures'):prepare_launch(root)
    def invoke(self,q,p,script):
        def command(c,s,cf):return [sys.executable,str(script),s['name'],s['stage'],s['receipt']],q['output_root']
        with patch.object(r,'preflight'),patch.object(r,'command_for',side_effect=command):return r.run(p)
    def test_unmocked_preflight_review_and_drift(self):
        from .entry import verify_review
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);q,p,script=fixture(root);repo=root/'repo';repo.mkdir();(repo/'scripts/ssp').mkdir(parents=True);(repo/'scripts/ssp/x.py').write_text('frozen')
            subprocess.run(['git','init','-q'],cwd=repo,check=True);subprocess.run(['git','add','.'],cwd=repo,check=True);subprocess.run(['git','-c','user.name=Fixture','-c','user.email=fixture@localhost','commit','-qm','fixture'],cwd=repo,check=True)
            commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
            anchor=root/'anchor.csv';anchor.write_text('cell_id,cell_type\n1,T\n');prepared=root/'prepared';prepared.mkdir();payload=prepared/'prototype.npy';payload.write_bytes(b'cache');atomic_json(prepared/'complete.json',dict(status='succeeded',outputs={'cache':identity(payload)}))
            for a,sp in q['methods'].items():
                sp.update(worktree=str(repo),commit=commit,machine=str(root/(a+'_machine.json')))
                f=Path(sp['cost_root'])/'configs/frozen.json';atomic_json(f,dict(identities={'prototype_cache':identity(payload),'config':identity(sp['config'])}))
            q.update(mini_test=False,testing_poll_fast=False,paper_protocol=str(anchor),prepared=str(prepared),reference=str(anchor),engineering_protocol=str(anchor),anchors={'B1':identity(anchor)},admission=str(root/'admission.json'),review_receipt=str(root/'approval.json'),minimum_free_gb=15)
            atomic_json(p,q);atomic_json(root/'source/frozen.json',dict(queue=identity(p),worktrees={str(repo):dict(commit=commit,files=r.source_map(repo))}))
            admission=dict(approved_to_launch=True,required_failures=[],run_id=q['run_id'],resources={'estimated_peak_gpu_allocator_bytes':1},frozen_source=identity(root/'source/frozen.json'),queue=identity(p),resolved_configs={a:{} for a in ('S','SP')},arm_frozen={a:identity(Path(sp['cost_root'])/'configs/frozen.json') for a,sp in q['methods'].items()});atomic_json(q['admission'],admission)
            approval=dict(approved_to_launch=True,required_failures=[],independent=True,run_id=q['run_id'],reviewer=dict(agent_id='independent-test-fixture',requested_model='gpt-6-astra',requested_reasoning_effort='high',observed_identity='synthetic fixture'),queue_sha256=sha256(p),frozen_source_sha256=sha256(root/'source/frozen.json'),evidence={q['admission']:sha256(q['admission'])});atomic_json(q['review_receipt'],approval)
            r.preflight(p,q)
            for field,value in [('independent',False),('approved_to_launch',False)]:
                atomic_json(q['review_receipt'],dict(approval,**{field:value}))
                with self.assertRaises(ValueError):r.preflight(p,q)
            atomic_json(q['review_receipt'],approval)
            for target in (payload,Path(q['methods']['S']['config']),Path(q['admission']),Path(q['methods']['S']['cost_root'])/'configs/frozen.json',repo/'scripts/ssp/x.py',p):
                original=target.read_bytes();target.write_bytes(original+b' ')
                with self.assertRaises(ValueError):r.preflight(p,q)
                target.write_bytes(original)
            r.preflight(p,q)
    def test_pid_namespace_is_unknown_not_dead(self):
        from .entry import status
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);q,p,script=fixture(root)
            atomic_json(root/'run_status.json',dict(run_id=q['run_id'],runner=process_identity(),pid_namespace='pid:[different_namespace]',status='running',stages={}))
            result=status(q);self.assertIsNone(result['runner_alive']);self.assertEqual(result['process_visibility'],'namespace_mismatch_use_host_shell')
            with self.assertRaisesRegex(RuntimeError,'Host PID namespace'):r.run(p)
    def test_serial_order(self):
        with tempfile.TemporaryDirectory() as t:
            q,_,_=fixture(Path(t));self.assertEqual([s['name'] for s in r.stage_specs(q)],['S_'+s for s in ('train','predict','postprocess','matrix','evaluate')]+['SP_'+s for s in ('train','predict','postprocess','matrix','evaluate')]+['b_cost','report'])
    def test_real_children_resume_and_receipt_corruption(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);q,p,s=fixture(root);self.assertEqual(self.invoke(q,p,s),0);before=read(root/'run_status.json');self.assertEqual(self.invoke(q,p,s),0)
            self.assertEqual([len(v['attempts']) for v in read(root/'run_status.json')['stages'].values()],[1]*12)
            rec=read(before['stages']['S_train']['receipt']);Path(rec['outputs']['artifact']['path']).write_text('corrupt')
            self.assertEqual(self.invoke(q,p,s),1);state=read(root/'run_status.json');self.assertEqual(state['stages']['S_train']['status'],'failed');self.assertEqual(state['stages']['SP_evaluate']['status'],'succeeded')
    def test_real_lock_and_pid_identity(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);q,p,s=fixture(root)
            with (root/'gpu.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):r.run(p)
            self.assertFalse(r.is_alive(dict(process_identity(),start_ticks='0')))
            first=r.stage_specs(q)[0];atomic_json(root/'run_status.json',dict(run_id=q['run_id'],stages={first['name']:dict(**first,status='running',child_identity=process_identity(),attempts=[])}))
            with self.assertRaisesRegex(RuntimeError,'still alive'):r.run(p)
    def test_nonrecoverable_and_finite_retry(self):
        self.assertFalse(r.recoverable(-15,'NaN'));self.assertFalse(r.recoverable(1,'hash mismatch [Errno 5] Input/output error'))
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);q,p,s=fixture(root);s.write_text("import sys; print('[Errno 5] Input/output error');sys.exit(1)\n")
            self.assertEqual(self.invoke(q,p,s),1);self.assertEqual(self.invoke(q,p,s),1)
            st=read(root/'run_status.json');self.assertEqual(len(st['stages']['S_train']['attempts']),3);self.assertEqual(len(st['stages']['SP_train']['attempts']),3)
    def test_watchdog(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);q,p,s=fixture(root);q['stage_timeouts']={k:.1 for k in r.TIMEOUTS};atomic_json(p,q);s.write_text('import time; time.sleep(30)\n')
            self.assertEqual(self.invoke(q,p,s),1);st=read(root/'run_status.json')
            for a in ('S','SP'):
                attempts=st['stages'][a+'_train']['attempts'];self.assertEqual(len(attempts),1);self.assertEqual(attempts[0]['termination_reason'],'watchdog_timeout')
    def test_source_drift_and_resource_pause(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);q,p,s=fixture(root);code=root/'scripts/ssp/x.py';code.parent.mkdir(parents=True);code.write_text('drift')
            self.assertEqual(self.invoke(q,p,s),1)
            st=read(root/'run_status.json');self.assertIn('source changed',st['stages']['S_train']['failure'])
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);q,p,s=fixture(root)
            with patch('scripts.ssp.resources.remaining_required',return_value=10**20):self.assertEqual(self.invoke(q,p,s),1)
            st=read(root/'run_status.json');self.assertEqual(st['status'],'paused_resource');self.assertEqual(st['stages']['S_train']['attempts'],[])

def main():
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if len(sys.argv)>1:atomic_json(Path(sys.argv[1]),dict(status='passed' if result.wasSuccessful() else 'failed',tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),source=identity(__file__),tested_sources={str(Path(__file__).with_name(x)):identity(Path(__file__).with_name(x)) for x in ('runner.py','entry.py','resources.py','evaluation.py')},scope='actual child-process tests with explicit synthetic stage payloads; not real biological mini'))
    if not result.wasSuccessful():raise SystemExit(1)
if __name__=='__main__':main()
