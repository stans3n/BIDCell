"""Actual child/process fault tests; biological receipt checks use real mini outputs."""
import argparse,copy,fcntl,json,os,signal,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from scripts.crgf.common import atomic_json,identity,process_identity,verify_outputs,source_identity
from . import runner
from .validation import validate_stage

class Engineering(unittest.TestCase):
    def fixture(self,root,payload=None):
        repo=root/'repo';repo.mkdir();script=repo/'payload.py'
        script.write_text(payload or "import sys,json,pathlib,hashlib\np=pathlib.Path(sys.argv[1]);p.parent.mkdir(parents=True,exist_ok=True);a=p.parent/'artifact';a.write_text(str(p));p.write_text(json.dumps({'status':'succeeded','outputs':{'file':{'path':str(a),'sha256':hashlib.sha256(a.read_bytes()).hexdigest()}}}))\n")
        subprocess.run(['git','init','-q'],cwd=repo,check=True,timeout=30)
        subprocess.run(['git','add','.'],cwd=repo,check=True,timeout=30)
        subprocess.run(['git','-c','user.name=Fixture','-c','user.email=fixture@local','commit','-qm','fixture'],cwd=repo,check=True,timeout=30)
        q=dict(run_id='BS4-engineering',output_root=str(root),mini_test=True,testing_poll_fast=True,python=sys.executable,gpu_lock=str(root/'gpu.lock'),retry_backoff_seconds=0,stage_timeouts={k:1 for k in runner.TIMEOUTS},cpu_workers=1,methods={})
        for a in ('P1','P2','T1','T2'):
            ar=root/a;ar.mkdir();config=ar/'config.json';atomic_json(config,dict(arm_id=a))
            q['methods'][a]=dict(config=str(config),worktree=str(repo),cost_root=str(ar),matrix=str(ar/'matrix'/a),evaluation=str(ar/'evaluation'/a),commit='fixture')
        path=root/'queue.json';atomic_json(path,q);atomic_json(root/'source/frozen.json',dict(worktrees={str(repo):dict(files=runner.source_map(repo))}))
        return q,path,script
    def invoke(self,q,path,script):
        def command(cfg,spec,config):return [sys.executable,str(script),spec['receipt']],str(script.parent)
        def synthetic_accept(stage,r,*args):
            if r.get('status')!='succeeded':raise ValueError('synthetic child failed')
            verify_outputs(r)
        with patch.object(runner,'preflight'),patch.object(runner,'command_for',side_effect=command),patch.object(runner,'accept',side_effect=synthetic_accept),patch('scripts.bs4.resources.remaining_required',return_value=0):return runner.run(path)
    def test_order(self):
        with tempfile.TemporaryDirectory() as t:
            q,_,_=self.fixture(Path(t));actual=[s['name'] for s in runner.stage_specs(q)]
            expected=[a+'_'+s for a in ('P1','P2','T1','T2') for s in (('postprocess','matrix','evaluate') if a.startswith('P') else ('train','predict','postprocess','matrix','evaluate'))]+['report']
            self.assertEqual(actual,expected)
    def test_real_children_idempotent_resume(self):
        with tempfile.TemporaryDirectory() as t:
            q,p,s=self.fixture(Path(t));self.assertEqual(self.invoke(q,p,s),0);self.assertEqual(self.invoke(q,p,s),0)
            state=json.loads((Path(t)/'run_status.json').read_text());self.assertEqual(len(state['stages']),17)
            self.assertTrue(all(len(v['attempts'])==1 for v in state['stages'].values()))
    def test_lock_pid_namespace_and_reuse(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);q,p,s=self.fixture(root)
            with (root/'runner.lock').open('a+') as f:
                fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):self.invoke(q,p,s)
            me=process_identity();self.assertTrue(runner.is_alive(me));self.assertFalse(runner.is_alive(dict(me,start_ticks='0')))
            atomic_json(root/'run_status.json',dict(run_id=q['run_id'],pid_namespace='wrong',runner=me,stages={}))
            with self.assertRaisesRegex(RuntimeError,'namespace'):self.invoke(q,p,s)
    def test_finite_retry(self):
        with tempfile.TemporaryDirectory() as t:
            q,p,s=self.fixture(Path(t),"import sys;print('[Errno 5] Input/output error');sys.exit(1)\n")
            self.assertEqual(self.invoke(q,p,s),1);self.assertEqual(self.invoke(q,p,s),1)
            state=json.loads((Path(t)/'run_status.json').read_text())
            self.assertTrue(all(len(state['stages'][a+'_'+('postprocess' if a.startswith('P') else 'train')]['attempts'])==3 for a in ('P1','P2','T1','T2')))
        self.assertFalse(runner.recoverable(-15,'NaN'));self.assertFalse(runner.recoverable(1,'hash mismatch [Errno 5] Input/output error'))
    def test_real_watchdog(self):
        with tempfile.TemporaryDirectory() as t:
            q,p,s=self.fixture(Path(t),'import time;time.sleep(10)\n');q['stage_timeouts']={k:.1 for k in runner.TIMEOUTS};atomic_json(p,q)
            self.assertEqual(self.invoke(q,p,s),1);state=json.loads((Path(t)/'run_status.json').read_text())
            self.assertTrue(all(v['attempts'][0]['termination_reason']=='watchdog_timeout' for v in state['stages'].values() if v['attempts']))
    def test_watchdog_cleans_term_ignoring_grandchild(self):
        with tempfile.TemporaryDirectory() as t:
            code="import subprocess,sys,time,pathlib\nchild=subprocess.Popen([sys.executable,'-c','import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(120)'])\npathlib.Path(sys.argv[1]).parent.mkdir(parents=True,exist_ok=True)\npathlib.Path(sys.argv[1]+'.pid').write_text(str(child.pid))\ntime.sleep(120)\n"
            q,p,s=self.fixture(Path(t),code);q['stage_timeouts']={k:.3 for k in runner.TIMEOUTS};atomic_json(p,q)
            self.assertEqual(self.invoke(q,p,s),1)
            state=json.loads((Path(t)/'run_status.json').read_text())
            for record in state['stages'].values():
                if not record['attempts']:continue
                self.assertTrue(record['attempts'][0]['descendants_terminated'])
                for pid in record['attempts'][0]['descendants_terminated']:
                    stat=Path(f'/proc/{pid}/stat')
                    if stat.exists():self.assertEqual(stat.read_text().split(') ',1)[1].split()[0],'Z')
    def test_shared_source_change_stops_queue(self):
        with tempfile.TemporaryDirectory() as t:
            q,p,s=self.fixture(Path(t));s.write_text(s.read_text()+'# changed\n')
            with self.assertRaisesRegex(ValueError,'source changed'):self.invoke(q,p,s)

def run(root):
    sources=['scripts/bs4/'+n for n in ('runner.py','validation.py','entry.py','test_engineering.py')];execution=source_identity(sources)
    root=Path(root);result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Engineering))
    if not result.wasSuccessful():raise ValueError('runner engineering failed')
    q=json.loads((root/'tests/mini_v5/configs/queue.json').read_text());rejected=[];validated=[]
    for spec in runner.stage_specs(q):
        if spec['stage']=='report':continue
        r=json.loads(Path(spec['receipt']).read_text());validate_stage(spec['stage'],r,q,spec);validated.append(spec['name'])
        for field,value in [('status','running'),('outputs',{})]:
            bad=copy.deepcopy(r);bad[field]=value
            try:validate_stage(spec['stage'],bad,q,spec)
            except ValueError:rejected.append(spec['name']+':'+field)
            else:raise AssertionError('receipt fault accepted')
        if spec['stage']=='train':
            bad=copy.deepcopy(r);bad['updates']=1
            try:validate_stage('train',bad,q,spec)
            except ValueError:rejected.append(spec['name']+':fake_updates')
            else:raise AssertionError('fake training count')
        if spec['stage']=='evaluate':
            bad=copy.deepcopy(r);bad['profiles']=['paper_aligned_v1']*3
            try:validate_stage('evaluate',bad,q,spec)
            except ValueError:rejected.append(spec['name']+':duplicate_profiles')
            else:raise AssertionError('fake profiles')
    formal=json.loads((root/'configs/queue.json').read_text());dummy=root/'tests/report_dummy';dummy.write_text('not science')
    fake=dict(status='succeeded',outputs={'dummy':identity(dummy)},ready=True,paper_protocol_complete=True,arm_receipts={a:{} for a in ('B','S','P1','P2','T1','T2')},fixed_population=formal['population'])
    try:validate_stage('report',fake,formal,dict(method=None))
    except ValueError:rejected.append('report:dummy_only')
    else:raise AssertionError('placeholder report accepted')
    if source_identity(sources)!=execution:raise ValueError('engineering execution source drift')
    atomic_json(root/'tests/runner_engineering_final.json',dict(status='passed',execution_source=execution,actual_process_tests=result.testsRun,biological_fixture='real four-arm mini, actual artifacts validated',validated_real_stages=validated,rejected_faults=rejected,source={n:identity(Path(__file__).with_name(n)) for n in ('runner.py','validation.py','entry.py','test_engineering.py')}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();run(a.root)
