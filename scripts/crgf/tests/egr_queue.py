"""Actual child subprocess failure, dependency isolation and idempotent resume."""
import argparse,json,sys
from pathlib import Path
from scripts.crgf.common import atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--child');p.add_argument('--receipt');p.add_argument('--run',action='store_true');p.add_argument('--startup-delay',type=float,default=0);a=p.parse_args();root=Path(a.root)
    if a.child:
        counts=root/'invocations';counts.mkdir(exist_ok=True);path=counts/a.child;path.write_text(str(int(path.read_text())+1 if path.exists() else 1))
        if a.child=='E_train' and not (root/'recover').exists():raise SystemExit(17)
        atomic_json(a.receipt,dict(status='succeeded',fixture_only=True,outputs={},evaluation_executed=True,paper_method_complete=not (a.child=='E_evaluate' and (root/'partial').exists()),paper_protocol_complete=True,purity_status='fixture_only_no_scores'));return
    if a.run:
        __import__('time').sleep(a.startup_delay)
        from scripts.crgf import run_egr,resources_egr
        from scripts.crgf.egr_config import QueueConfig
        cfg=QueueConfig(run_id='fixture',output_root=str(root),methods={v:{} for v in ['E','G','R']},paper_protocol='fixture',prepared='fixture',minimum_free_gb=.01)
        run_egr.read=lambda path:cfg;run_egr.verify_preflight=lambda cfg:None;resources_egr.check=lambda root:None
        specs=run_egr.stage_specs(root);paths={name:str(receipt) for name,method,stage,deps,receipt in specs}
        run_egr.signature=lambda cfg,stage,method,deps: __import__('hashlib').sha256(json.dumps([stage,method,deps],sort_keys=True).encode()).hexdigest()
        checkout=str(Path(__file__).resolve().parents[3])
        def command(cfg,stage,method):
            name=(method+'_'+stage) if method else 'report'
            return [sys.executable,'-m','scripts.crgf.tests.egr_queue','--root',str(root),'--child',name,'--receipt',paths[name]],checkout
        run_egr.command_for=command
        sleep=run_egr.time.sleep;run_egr.time.sleep=lambda seconds:sleep(.01)
        rc=run_egr.run('fixture')
        status=json.loads((root/'run_status.json').read_text())
        if (root/'partial').exists():
            assert rc==1 and status['stages']['E_evaluate']['status']=='failed'
            assert 'required evaluation profile incomplete' in status['stages']['E_evaluate']['failure']
            assert status['stages']['G_evaluate']['status']=='succeeded'
        elif not (root/'recover').exists():
            assert rc==1 and status['stages']['E_train']['status']=='failed'
            assert all(status['stages']['E_'+s]['status']=='blocked_by_dependency' for s in ['predict','postprocess','matrix','evaluate'])
            assert all(status['stages'][v+'_evaluate']['status']=='succeeded' for v in ['G','R','B1','GR'])
        else:
            assert rc==0
            assert (root/'invocations/G_train').read_text()=='1' and (root/'invocations/R_train').read_text()=='1'
        atomic_json(root/('resumed_receipt.json' if (root/'recover').exists() else 'failure_receipt.json'),dict(status='succeeded',test='actual subprocess failure/dependency isolation/idempotent resume',queue_status=status['status']))
if __name__=='__main__':main()
