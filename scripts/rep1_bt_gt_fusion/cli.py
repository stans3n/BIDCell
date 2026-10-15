"""Explicit phase commands. Read-only commands never dispatch or authorize work."""
import argparse
from collections import deque
import json
from pathlib import Path
from . import runtime as rt
from . import execution as ex


def parser():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    prepare=sub.add_parser('prepare');prepare.add_argument('--plan',required=True);prepare.add_argument('--output-root',required=True);prepare.add_argument('--run-id',required=True)
    for name in ('validate','launch','resume-production','status','tail','complete-check','grant-resume','evaluate','resume-evaluation','publish','recover-historical-mask','_supervise','_worker'):
        q=sub.add_parser(name);q.add_argument('--config',required=True)
        if name=='validate':q.add_argument('--admission',required=True)
        if name=='status':q.add_argument('--json',action='store_true')
        if name=='tail':q.add_argument('--lines',type=int,default=80)
        if name=='complete-check':q.add_argument('--phase',choices=['production'],required=True)
        if name=='grant-resume':q.add_argument('--user-instruction-utc',required=True)
        if name in ('evaluate','resume-evaluation','publish','recover-historical-mask'):q.add_argument('--manual-resume',required=True)
        if name=='recover-historical-mask':q.add_argument('--method',choices=['C','SP'],required=True);q.add_argument('--source',required=True)
        if name in ('_supervise','_worker'):q.add_argument('--attempt',required=True)
        if name=='_supervise':q.add_argument('--mode',choices=['production','evaluation'],required=True)
        if name=='_worker':q.add_argument('--node',required=True);q.add_argument('--fds',required=True)
    return p


def main(argv=None):
    args=parser().parse_args(argv);command=args.command
    if command=='prepare':result=rt.prepare(args.plan,args.output_root,args.run_id)
    elif command=='status':result=rt.status(args.config)
    elif command=='tail':
        if not 1<=args.lines<=10000:raise ValueError('tail lines must be 1..10000')
        cfg=rt.load_config(args.config)
        with (Path(cfg['run_root'])/'logs/queue.log').open() as f:print(''.join(deque(f,maxlen=args.lines)),end='')
        return
    else:
        cfg=rt.load_config(args.config);ex.install_import_guard(cfg)
        if command=='validate':result=rt.validate(args.config,args.admission)
        elif command in ('launch','resume-production'):result=rt.launch(args.config,resume=command=='resume-production')
        elif command=='complete-check':result=rt.complete_check(args.config)
        elif command=='grant-resume':result=rt.grant_resume(args.config,args.user_instruction_utc)
        elif command in ('evaluate','resume-evaluation'):
            from .common import require_manual_resume
            require_manual_resume(cfg,args.manual_resume);result=rt.launch(args.config,mode='evaluation',resume=command=='resume-evaluation')
        elif command=='publish':result=rt.publish(args.config,args.manual_resume)
        elif command=='recover-historical-mask':
            from contextlib import ExitStack
            from .common import require_manual_resume
            from .publication import recover_historical_mask
            with ExitStack() as stack:
                fds=tuple(stack.enter_context(rt.lock(p)) for p in rt.paths(cfg))
                state=rt.bound_state(cfg)
                if rt.alive(state.get('supervisor')) or rt.alive(state.get('worker')):raise RuntimeError('recovery requires an inactive queue')
                rt.verify_execution(cfg,fds,True);require_manual_resume(cfg,args.manual_resume)
                result=recover_historical_mask(cfg,args.manual_resume,args.method,args.source)
        elif command=='_supervise':result=rt.supervise(args.config,args.attempt,args.mode)
        elif command=='_worker':result=rt.worker(args.config,args.node,args.attempt,tuple(map(int,args.fds.split(','))))
        else:raise ValueError('unsupported command')
    print(json.dumps(result,indent=2,allow_nan=False))
