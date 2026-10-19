"""Paired scientific stages. Process locks/scheduling belong to the caller."""
import argparse
from .common import *


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('prepare')
    for name in ('train','predict','postprocess','matrix','evaluate','validate'):
        s=sub.add_parser(name);s.add_argument('--run-id',required=True,choices=tuple(RUNS))
        if name=='validate':s.add_argument('--stage',required=True,choices=('train','predict','postprocess','matrix','evaluate'))
        if name=='train':
            s.add_argument('--pilot-updates',type=int);s.add_argument('--pilot-root',type=Path)
    a=p.parse_args()
    if a.command=='prepare':
        from .prepare import prepare
        result=prepare(a.root)
    elif a.command=='validate':
        from .stages import validate_stage
        result=validate_stage(config_path(a.run_id,a.root),a.stage,require_bound=True)
    elif a.command=='train' and a.pilot_updates is not None:
        from .training import train
        result=train(config_path(a.run_id,a.root),a.pilot_updates,a.pilot_root)
    else:
        from .stages import run_stage
        result=run_stage(config_path(a.run_id,a.root),a.command)
    print(json.dumps(json_safe(result),indent=2))


if __name__=='__main__':main()
