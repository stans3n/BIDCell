"""Actual file/lock transactions with explicitly synthetic report payloads."""
import copy,fcntl,json,os,subprocess,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from scripts.ssp import report_recovery as rr
from scripts.ssp.runner import stage_specs,source_map
from scripts.ssp.evaluation import publish
from scripts.crgf.common import atomic_json,identity,sha256


def fixture(root):
 recovery=root/'recovery/report_v1';recovery.mkdir(parents=True)
 q=dict(run_id='synthetic-report-recovery-test',output_root=str(root),gpu_lock=str(root/'gpu.lock'),python='/usr/bin/python3',methods={a:dict(cost_root=str(root/a),matrix=str(root/a/'matrix'/a),evaluation=str(root/a/'evaluation'/a)) for a in ('S','SP')})
 (root/'runner.lock').touch();(root/'gpu.lock').touch();atomic_json(root/'configs/queue.json',q)
 state=dict(status='failed',pid_namespace=os.readlink('/proc/self/ns/pid'),stages={})
 for spec in stage_specs(q):
  path=Path(spec['receipt']);path.parent.mkdir(parents=True,exist_ok=True)
  if spec['stage']=='report':atomic_json(path,{'status':'failed','reason':'original synthetic missing stage IDs'})
  else:
   payload=path.parent/'payload';payload.write_text(spec['name']);atomic_json(path,dict(status='succeeded',outputs={'payload':identity(payload)},updates=4000,profiles=['engineering_nuclear_v1','engineering_pearson_v1','paper_aligned_v1'],paper_method_complete=True,evaluation_executed=True))
  state['stages'][spec['name']]=dict(**spec,status='failed' if spec['stage']=='report' else 'succeeded',attempts=[{'wall_seconds':1,'exit_code':1 if spec['stage']=='report' else 0}],signature='original_'+spec['name'],receipt_sha256=sha256(path))
 atomic_json(root/'run_status.json',state);atomic_json(root/'compute_complete.json',{'status':'partial'})
 protected={str(p.resolve()):identity(p) for folder in ('S','SP','b_cost') for p in (root/folder).rglob('*') if p.is_file()};protected[str(root/'configs/queue.json')]=identity(root/'configs/queue.json')
 baseline=dict(status='passed',original_commit=rr.BASE,original_run_status=identity(root/'run_status.json'),original_compute_complete=identity(root/'compute_complete.json'),successful_stages={n:dict(record=v) for n,v in state['stages'].items() if n!='report'},protected_files=protected,verified_dependency_files={},partial_comparison={str(p.relative_to(root/'comparison')):identity(p) for p in (root/'comparison').rglob('*') if p.is_file()})
 atomic_json(recovery/'baseline.json',baseline)
 return q,state,baseline,recovery


def synthetic_report(config,output_dir):
 out=Path(output_dir);out.mkdir();(out/'metric_coverage.csv').write_text('metric,status\nfixture,computed\n');publish(out,ready=True,paper_protocol_complete=True,profiles=['paper_aligned_v1','engineering_nuclear_v1','engineering_pearson_v1'])

class Tests(unittest.TestCase):
 def test_success_preserves_all_eleven_and_idempotent_no_dispatch(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);q,initial,baseline,recovery=fixture(root);provenance={'synthetic_fixture':True}
   with patch.object(rr,'check_review',return_value=provenance),patch('scripts.ssp.comparison.aggregate',side_effect=synthetic_report) as aggregate,patch('scripts.ssp.runner.command_for',side_effect=AssertionError('successful-stage dispatcher forbidden')):
    rr.execute(root);self.assertEqual(aggregate.call_count,1);after=json.loads((root/'run_status.json').read_text())
    for name in baseline['successful_stages']:self.assertEqual(after['stages'][name],initial['stages'][name])
    self.assertEqual(len(after['stages']['report']['attempts']),2);self.assertEqual(after['stages']['report']['attempts'][0],initial['stages']['report']['attempts'][0]);self.assertTrue((root/'comparison').is_symlink())
    self.assertEqual(json.loads((recovery/'failed_attempt/run_status.json').read_text()),initial)
    complete=(root/'compute_complete.json').read_bytes();rr.execute(root);self.assertEqual(aggregate.call_count,1);self.assertEqual((root/'compute_complete.json').read_bytes(),complete)
    self.assertEqual(json.loads(complete)['next_state'],'awaiting_independent_scientific_acceptance_after_user_resume')
 def test_held_lock_refuses_before_mutation(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);q,state,b,r=fixture(root);before=(root/'run_status.json').read_bytes()
   with (root/'runner.lock').open('r') as lock,patch.object(rr,'check_review',return_value={'synthetic_fixture':True}):
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with self.assertRaises(BlockingIOError):rr.execute(root)
   self.assertEqual(before,(root/'run_status.json').read_bytes());self.assertFalse((r/'attempt.json').exists())
 def test_successful_artifact_mutation_and_membership_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);q,s,b,r=fixture(root);payload=next((root/'S').rglob('payload'));old=payload.read_bytes();payload.write_text('changed')
   with self.assertRaisesRegex(ValueError,'protected successful artifact'):rr.verify_protected(root,b)
   payload.write_bytes(old);(root/'S/extra').write_text('unauthorized')
   with self.assertRaisesRegex(ValueError,'membership'):rr.verify_protected(root,b)
 def test_post_execution_source_recheck_fail_closed(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);q,s,b,r=fixture(root)
   with patch.object(rr,'check_review',side_effect=[{'synthetic_fixture':True},ValueError('source drift')]),patch('scripts.ssp.comparison.aggregate',side_effect=synthetic_report):
    with self.assertRaisesRegex(ValueError,'source drift'):rr.execute(root)
   self.assertFalse((root/'comparison').is_symlink());self.assertEqual(json.loads((root/'comparison/complete.json').read_text())['status'],'failed');self.assertTrue((r/'failure.json').exists())
   for n in b['successful_stages']:self.assertEqual(json.loads((root/'run_status.json').read_text())['stages'][n],s['stages'][n])
 def test_real_approval_source_baseline_and_evidence_rejections(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);q,s,b,r=fixture(root);bp=r/'baseline.json';repo=Path(rr.__file__).resolve().parents[2]
   with self.assertRaisesRegex(ValueError,'approval required'):rr.check_review(root,r,bp)
   commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip();evidence=r/'synthetic_test.json';evidence.write_text('synthetic evidence only')
   approval=dict(independent=True,approved_report_recovery=True,required_failures=[],reviewer=dict(agent_id='synthetic-review-fixture',requested_model='gpt-6-astra',requested_reasoning_effort='high'),run_root=str(root),scope='report_only',original_commit=rr.BASE,recovery_commit=commit,recovery_source=source_map(repo),baseline_sha256=sha256(bp),evidence={str(evidence):sha256(evidence)})
   real=subprocess.check_output
   def git(args,**kw):return '' if args==['git','status','--porcelain'] else real(args,**kw)
   with patch.object(rr.subprocess,'check_output',side_effect=git):
    atomic_json(r/'approval.json',approval);rr.check_review(root,r,bp)
    for field,value in [('independent',False),('scope','train'),('recovery_source',{}),('baseline_sha256','wrong'),('recovery_commit','wrong')]:
     a=copy.deepcopy(approval);a[field]=value;atomic_json(r/'approval.json',a)
     with self.assertRaises(ValueError):rr.check_review(root,r,bp)
    atomic_json(r/'approval.json',approval);evidence.write_text('changed')
    with self.assertRaisesRegex(ValueError,'evidence changed'):rr.check_review(root,r,bp)

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);args=p.parse_args();root=Path(args.root);tempfile.tempdir=str(root/'recovery/report_v1/tests');Path(tempfile.tempdir).mkdir(parents=True,exist_ok=True);result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
 atomic_json(root/'recovery/report_v1/tests/driver_tests.json',dict(status='passed' if result.wasSuccessful() else 'failed',tests_run=result.testsRun,scope='actual file/lock recovery transactions with explicitly synthetic report payload; never runs scientific stages',sources={x:identity(Path(__file__).resolve().parents[3]/x) for x in ['scripts/ssp/report_recovery.py','scripts/ssp/tests/report_recovery.py']}))
 if not result.wasSuccessful():raise SystemExit(1)
