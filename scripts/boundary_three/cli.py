"""Operational CLI. No AI service, Markdown interpreter, or mutable science gate."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from . import runtime
from .common import load_run, output_path
from .contracts import NODES


def human_status(value):
    lines=[f"Run: {value['run_id']}  Host: {value['host']}",
           f"Queue: {value['execution_status']}  Process: {value['liveness']}",
           f"Supervisor alive: {value['process_alive']}  Worker alive: {value['worker_alive']}",
           f"Queue lock held: {value['queue_lock_held']}  Workspace compute lock held: {value['compute_lock_held']}",
           f"Current node: {value.get('current_node')}  Stage: {value.get('stage')}",
           f"Heartbeat: {value.get('heartbeat_utc')}  Last scientific progress: {value.get('last_progress_utc')}",
           f"Progress: {json.dumps(value.get('progress'),ensure_ascii=False)}",
           f"Completed: {', '.join(value['completed_nodes']) or '(none)'}",
           f"Deferred: {', '.join(value['deferred_nodes']) or '(none)'}",
           f"Scientific skips: {', '.join(value['skipped_nodes']) or '(none)'}",
           f"Failure/pause reason: {value.get('failure')}",
           f"Queue log: {value['log_path']}  Node log: {value.get('node_log_path')}"]
    for n,row in value['nodes'].items():
        lines.append(f"  {n}: {row['execution_status']} science={row.get('science_status')} receipt={row['receipt_status']['status']} reason={row.get('reason')}")
    return '\n'.join(lines)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    prepare=sub.add_parser('prepare');prepare.add_argument('--template',required=True)
    prepare.add_argument('--output-root');prepare.add_argument('--run-id')
    validate=sub.add_parser('validate');validate.add_argument('--config',required=True);validate.add_argument('--admission',required=True)
    for command in ('launch','resume','status','tail','complete-check','safe-stop','_run','_node'):
        parser=sub.add_parser(command,help=argparse.SUPPRESS if command.startswith('_') else None)
        parser.add_argument('--config',required=True)
        if command=='resume':parser.add_argument('--budget-hours',type=float)
        if command=='status':parser.add_argument('--json',action='store_true')
        if command=='tail':
            parser.add_argument('--node',choices=NODES);parser.add_argument('--lines',type=int,default=80);parser.add_argument('--follow',action='store_true')
        if command=='safe-stop':parser.add_argument('--reason',default='explicit safe-stop from CLI')
        if command=='_run':parser.add_argument('--lease',required=True)
        if command=='_node':
            parser.add_argument('--node',choices=NODES,required=True);parser.add_argument('--result',required=True)
            parser.add_argument('--lease',required=True);parser.add_argument('--queue-fd',required=True,type=int);parser.add_argument('--compute-fd',required=True,type=int)
    args=p.parse_args(argv)
    try:
        if args.command=='prepare':result=runtime.prepare(args.template,args.output_root,args.run_id)
        elif args.command=='validate':result=runtime.validate(args.config,args.admission)
        elif args.command=='launch':result=runtime.launch(args.config)
        elif args.command=='resume':result=runtime.launch(args.config,resume=True,budget_seconds=None if args.budget_hours is None else args.budget_hours*3600)
        elif args.command=='status':
            result=runtime.status(args.config)
            if not args.json:print(human_status(result));return 0
        elif args.command=='safe-stop':result=runtime.request_stop(load_run(args.config),detail=args.reason)
        elif args.command=='complete-check':
            result=runtime.complete_check(args.config);print(json.dumps(result,indent=2));return 0 if result['all_enabled_applicable_complete'] else 2
        elif args.command=='tail':
            if not 1<=args.lines<=10000:raise ValueError('--lines must be 1..10000')
            cfg=load_run(args.config);state=runtime.bound_state(cfg)
            if args.node:
                paths=sorted(output_path(cfg,'logs').glob(args.node+'_*.log'))
                if not paths:raise FileNotFoundError('node has no log yet')
                path=paths[-1]
            else:path=Path(state.get('node_log_path') or state['log_path'])
            return subprocess.call(['tail','-n',str(args.lines)]+(['-F'] if args.follow else [])+[str(path)])
        elif args.command=='_run':return runtime.run_queue(args.config,args.lease)
        elif args.command=='_node':return runtime.node_worker(args.config,args.node,args.result,args.lease,args.queue_fd,args.compute_fd)
        else:raise ValueError('unknown command')
        print(json.dumps(result,indent=2,allow_nan=False));return 0
    except Exception as exc:
        print(json.dumps(dict(status='ERROR',exception_type=type(exc).__name__,reason=str(exc))),file=sys.stderr)
        return 1


if __name__=='__main__':
    sys.exit(main())
