"""Narrow compute controls; no publication or manual-grant creation command."""
import argparse
import json
from . import runtime as rt
from .common import require_manual_resume

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--output-root',required=True);prep.add_argument('--run-id',required=True);prep.add_argument('--plans-manifest',required=True)
    prep.add_argument('--donor-validation');prep.add_argument('--split-validation');prep.add_argument('--donor-baseline')
    prep.add_argument('--replay-donor-validation');prep.add_argument('--replay-donor-baseline')
    prep.add_argument('--p2-donor-validation');prep.add_argument('--p2-donor-baseline')
    for name in ('resume-prepare','validate','launch','status','tail','complete-check','resume-compute','publication-check','_entry','_supervise','_worker','_prepare'):
        q=sub.add_parser(name);q.add_argument('--config',required=True)
        if name=='validate':q.add_argument('--admission')
        if name=='status':q.add_argument('--json',action='store_true')
        if name=='tail':q.add_argument('--lines',type=int,default=80)
        if name=='complete-check':q.add_argument('--phase',choices=['compute'],default='compute')
        if name=='resume-compute':q.add_argument('--attempt-reason',required=True)
        if name in ('_entry','_supervise','_worker'):q.add_argument('--attempt',required=True)
        if name=='_worker':q.add_argument('--stage',required=True)
        if name in ('_worker','_prepare'):q.add_argument('--fds',required=True)
    a=p.parse_args();cmd=a.command
    if cmd=='prepare':out=rt.prepare(a.output_root,a.run_id,a.plans_manifest,a.donor_validation,a.split_validation,a.donor_baseline,a.replay_donor_validation,a.replay_donor_baseline,a.p2_donor_validation,a.p2_donor_baseline)
    elif cmd=='resume-prepare':out=rt.resume_prepare(a.config)
    elif cmd=='validate':out=rt.validate(a.config,a.admission)
    elif cmd=='launch':out=rt.launch(a.config)
    elif cmd=='resume-compute':out=rt.launch(a.config,True,a.attempt_reason)
    elif cmd=='status':out=rt.status(a.config)
    elif cmd=='tail':print(rt.tail(a.config,a.lines),end='');return
    elif cmd=='complete-check':out=rt.complete_check(a.config,a.phase)
    elif cmd=='publication-check':
        cfg=rt.load_config(a.config);require_manual_resume(cfg,str(rt.Path(cfg['run_root'])/'control/manual_resume.json'))
        raise PermissionError('Independent result analysis and root publication review are separate post-resume work; compute CLI never publishes')
    elif cmd=='_prepare':rt.prepare_snapshot(a.config,tuple(map(int,a.fds.split(','))));return
    elif cmd=='_entry':raise SystemExit(rt.entry(a.config,a.attempt))
    elif cmd=='_supervise':raise SystemExit(rt.supervise(a.config,a.attempt))
    elif cmd=='_worker':raise SystemExit(rt.worker(a.config,a.stage,a.attempt,tuple(map(int,a.fds.split(',')))))
    print(json.dumps(out,indent=2,allow_nan=False))
