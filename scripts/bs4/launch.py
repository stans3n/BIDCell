"""Start the frozen queue in a new session, with no client/network dependency."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from .evaluation import read,times,ARMS
from scripts.crgf.common import atomic_json,process_identity
from scripts.crgf.run_egr import is_alive


def launch(config):
    config=Path(config).resolve();cfg=read(config);root=Path(cfg['output_root']);root.mkdir(parents=True,exist_ok=True)
    from .runner import preflight
    preflight(config,cfg)
    if (root/'compute_complete.json').exists() and read(root/'compute_complete.json').get('status')=='succeeded':raise RuntimeError('compute already complete; await user resume for independent analysis')
    status=root/'run_status.json'
    if status.exists():
        old=read(status)
        if old.get('pid_namespace') and old['pid_namespace']!=os.readlink('/proc/self/ns/pid') and old.get('runner',{}).get('boot_id')==Path('/proc/sys/kernel/random/boot_id').read_text().strip():raise RuntimeError('Host PID namespace required; hidden runner cannot be assumed dead')
        if old.get('runner') and is_alive(old['runner']):raise RuntimeError('runner already active')
        for stage in old.get('stages',{}).values():
            if stage.get('child_identity') and is_alive(stage['child_identity']):raise RuntimeError('previous child still active')
    # Actual ownership is acquired and held by the child runner, not this launcher.
    for path in [root/'runner.lock',Path(cfg['gpu_lock'])]:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    logs=root/'logs';logs.mkdir(exist_ok=True);logpath=logs/f'runner_launch_{time.time_ns()}.log';cwd=cfg['methods'][ARMS[0]]['worktree']
    command=[cfg['python'],'-u','-m','scripts.bs4.runner','--config',str(config)]
    with logpath.open('w') as log:
        child=subprocess.Popen(command,cwd=cwd,env=dict(os.environ,PYTHONPATH=cwd),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    record=dict(version='SSP-detached-launch-v1',pid_namespace=os.readlink('/proc/self/ns/pid'),command=command,cwd=cwd,identity=process_identity(child.pid),session_id=os.getsid(child.pid),log=str(logpath),config=str(config),**times())
    atomic_json(root/'launches'/f'{child.pid}_{time.time_ns()}.json',record);atomic_json(root/'launch.json',record)
    print(json.dumps(record,indent=2),flush=True)
    return record


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args();launch(args.config)

if __name__=='__main__':main()
