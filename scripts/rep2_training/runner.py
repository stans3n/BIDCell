"""Serial persistent runner, strict scope and verified completed-stage recovery."""
import argparse
import os
import shutil
import subprocess
import sys
import time
import traceback
from .common import *

STAGES=('nuclei','expression','patches','nuclear_matrix','annotation','prepare','runtime_validation',*ARMS)

def verify_patch_inventory():
    inventory=read(ROOT/'shared/configs/patch_inventory.json')
    if len(inventory)!=read(ROOT/'manifests/resource_admission.json')['shift0_patch_files']:raise ValueError('unexpected complete patch count')
    for obj in inventory:verify_identity(obj)
    if len({x['path'] for x in inventory})!=len(inventory):raise ValueError('duplicate patch inventory')
    return len(inventory)

def stage_dependencies(stage):
    deps=base_dependencies();index=STAGES.index(stage)
    deps['previous']={name:identity(ROOT/'receipts'/f'{name}.json') for name in STAGES[:index]}
    return deps

def admission():
    cfg=read(ROOT/'configs/run.json')
    if cfg['seed']!=1234:raise ValueError('seed differs')
    for obj in cfg['input_identities'].values():verify_identity(obj)
    verify_identity(cfg['initial_identity']);verify_identity(cfg['baseline_template'])
    for obj in cfg['preflight_identities'].values():
        verify_identity(obj)
        if read(obj['path']).get('status') not in ('passed','frozen'):raise ValueError('preflight not passed '+obj['path'])
    verify_identity(cfg['authority_identity']);verify_identity(cfg['scope_identity'])
    for obj in read(ROOT/'manifests/raw_input_audit.json')['inputs'].values():verify_identity(obj)
    review=read(ROOT/'review/main_review.json')
    if review.get('status')!='passed' or review.get('source_digest')!=digest(source_identity()):raise ValueError('independent source review required for this exact implementation')
    if review.get('config_sha256')!=sha(ROOT/'configs/run.json') or review.get('native_config_sha256')!=sha(ROOT/'configs/native.yaml'):raise ValueError('review config identity mismatch')
    if read(ROOT/'manifests/resource_admission.json').get('status')!='passed':raise ValueError('resources not admitted')
    free=shutil.disk_usage(ROOT).free
    if free<6*(1<<30):raise RuntimeError('less than frozen 6 GiB free safety reserve')
    return dict(source_digest=digest(source_identity()),review=identity(ROOT/'review/main_review.json'),free_bytes=free)

def completed(stage):
    path=ROOT/'receipts'/f'{stage}.json'
    if not path.exists():return False
    if stage in ARMS:
        from .training import verify_complete
        verify_complete(stage)
    else:
        validate_receipt(path,stage_dependencies(stage))
        if stage=='patches':verify_patch_inventory()
    return True

def execute_stage(stage):
    if stage not in STAGES:raise ValueError('stage outside training-only scope')
    if completed(stage):return
    if stage in ARMS:
        from .training import train
        train(stage);return
    if stage=='prepare':
        from .prepare import prepare
        function=prepare
    elif stage=='runtime_validation':
        from .training import runtime_validation
        function=runtime_validation
    else:
        from . import preprocess
        function=getattr(preprocess,stage)
    deps=stage_dependencies(stage);outputs=function()
    candidate=dict(status='succeeded',stage=stage,dependencies=deps,outputs={k:identity(v) for k,v in outputs.items()})
    for obj in candidate['outputs'].values():verify_identity(obj)
    if stage=='patches':verify_patch_inventory()
    atomic_json(ROOT/'receipts'/f'{stage}.json',candidate)

def run():
    with lock(ROOT/'pipeline.lock'), lock('/home/zwan0726/work/outputs/cos_gpu.lock', read_only=True):
        audit=admission();state=dict(status='running',pid=os.getpid(),started=time.time(),admission=audit,stages={})
        atomic_json(ROOT/'status.json',state)
        for stage in STAGES:
            state['current_stage']=stage
            if completed(stage):
                state['stages'][stage]='verified_completed';atomic_json(ROOT/'status.json',state);continue
            state['resources']=resource_check(stage)
            state['stages'][stage]='running';atomic_json(ROOT/'status.json',state)
            logpath=ROOT/'logs'/f'{stage}.log'
            with logpath.open('a',buffering=1) as log:
                log.write(f'\nSTART {stage} {time.time()} runner_pid={os.getpid()}\n');log.flush()
                child_env=dict(os.environ,REP2_PIPELINE_PARENT=str(os.getpid()))
                child=subprocess.Popen([sys.executable,'-B','-m','scripts.rep2_training.runner','stage',stage],cwd=SOURCE,stdout=log,stderr=subprocess.STDOUT,env=child_env)
                state['child_pid']=child.pid;atomic_json(ROOT/'status.json',state)
                while child.poll() is None:
                    state['heartbeat']=time.time();atomic_json(ROOT/'status.json',state)
                    print(f'heartbeat stage={stage} pid={child.pid} time={state["heartbeat"]}',flush=True)
                    time.sleep(10)
                code=child.returncode
            if code!=0:
                state.update(status='failed',exit_code=code);state['stages'][stage]='failed';atomic_json(ROOT/'status.json',state)
                raise RuntimeError(f'{stage} failed with exit {code}; inspect {logpath}')
            if not completed(stage):raise ValueError('child exited without verified stage receipt')
            state['stages'][stage]='succeeded';atomic_json(ROOT/'status.json',state)
        # Do not replace this with an experiment-complete or downstream trigger.
        for arm in ARMS:completed(arm)
        complete=dict(status='succeeded',scope='three training tasks only',receipts={arm:identity(ROOT/'receipts'/f'{arm}.json') for arm in ARMS},
                      actual_updates={arm:ARMS[arm][1] for arm in ARMS},completed=time.time(),resume='manual only; inspect existing state first')
        atomic_json(ROOT/'training_complete.json',complete)
        state.update(status='training_complete',completed=time.time(),child_pid=None);atomic_json(ROOT/'status.json',state)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['run','stage','status']);parser.add_argument('stage',nargs='?');args=parser.parse_args()
    try:
        if args.command=='run':run()
        elif args.command=='stage':
            # The parent performs independent gate checks before dispatch.
            if os.environ.get('REP2_PIPELINE_PARENT')!=str(os.getppid()):raise RuntimeError('stage is internal; use run for locked, reviewed orchestration')
            admission()
            execute_stage(args.stage)
        else:print(json.dumps(read(ROOT/'status.json') if (ROOT/'status.json').exists() else {'status':'not_started'},indent=2))
    except BaseException as exc:
        if args.command=='run':
            path=ROOT/'status.json';state=read(path) if path.exists() else {}
            if state.get('pid')==os.getpid():
                state.update(status='failed',error=str(exc),failed=time.time());atomic_json(path,state)
        raise

if __name__=='__main__':main()
