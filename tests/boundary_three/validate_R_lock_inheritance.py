"""Tiny actual-host R FD inheritance probe, entirely engineering-only.

The test kills its own bound Python fixture parent after R starts. R sleeps eight
seconds and writes a marker, proving queue/compute locks outlive Python. It never
reads scientific data or creates a valid production launch admission.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from scripts.boundary_three import runtime
from scripts.boundary_three.common import atomic_json,identity,read,utc,digest,source_manifest
from scripts.boundary_three.contracts import PYTHON,RSCRIPT,GPU_LOCK
from scripts.boundary_three.settings import default_config


def parent(root):
    root=Path(root);cfg=read(root/'configs/run.json');state=runtime.bound_state(cfg)
    state.update(supervisor=runtime.process_identity(),current_node='P2_BASE',execution_status='RUNNING')
    runtime.save_state(cfg,state)
    with runtime.lock_file(root/'control/queue.lock') as queue,runtime.semantic_lock(cfg) as compute:
        os.environ.update(BOUNDARY_THREE_LOCK_FDS=json.dumps(dict(queue=queue.fileno(),compute=compute.fileno())),
                          BOUNDARY_THREE_RUN_ROOT=str(root),BOUNDARY_THREE_LEASE='tiny-R-lease')
        fds=runtime.inherited_lock_fds()
        atomic_json(root/'fd_map.json',dict(parent=runtime.process_identity(),queue=fds[0],compute=fds[1]))
        expression='args <- commandArgs(trailingOnly=TRUE); writeLines(as.character(Sys.getpid()),args[1]); Sys.sleep(8); writeLines("done",args[2])'
        with (root/'R.log').open('w') as log:
            subprocess.run([RSCRIPT,'-e',expression,str(root/'R.pid'),str(root/'R.done')],pass_fds=fds,
                           stdout=log,stderr=subprocess.STDOUT,check=True)


def main(root):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False)
    cfg=default_config(root,'tiny-R-lock-'+root.name)
    cfg.update(source_digest='engineering-probe-no-launch-manifest',input_digest='engineering-probe-no-launch-manifest')
    cfg['resources']['resource_wait_seconds']=30
    atomic_json(root/'configs/run.json',cfg)
    state=runtime.new_state(cfg)
    lease=dict(lease_id='tiny-R-lease',created_utc=utc(),budget_seconds=30,config_digest=digest(cfg))
    state.update(active_lease=lease['lease_id'],leases=[lease]);runtime.save_state(cfg,state)
    atomic_json(root/'control/leases/tiny-R-lease.json',lease)
    proc=subprocess.Popen([PYTHON,'-m','tests.boundary_three.validate_R_lock_inheritance','parent',str(root)])
    deadline=time.monotonic()+35
    while time.monotonic()<deadline and not (root/'R.pid').exists():
        if proc.poll() is not None:raise AssertionError('tiny R fixture parent exited before R')
        time.sleep(.1)
    if not (root/'R.pid').exists():raise AssertionError('tiny R did not start')
    rpid=int((root/'R.pid').read_text().strip());r_identity=runtime.process_identity(rpid)
    fds=read(root/'fd_map.json')
    observed={key:os.readlink(f'/proc/{rpid}/fd/{fds[key]}') for key in ('queue','compute')}
    assert Path(observed['queue']).resolve()==root/'control/queue.lock'
    assert Path(observed['compute']).resolve()==GPU_LOCK
    assert runtime.process_alive(fds['parent']) and fds['parent']['pid']==proc.pid
    os.kill(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
    for path in (root/'control/queue.lock',GPU_LOCK):
        try:
            with runtime.lock_file(path):pass
        except RuntimeError:continue
        raise AssertionError('R descendant lost inherited lock '+str(path))
    status=runtime.status(root/'configs/run.json')
    assert status['queue_lock_held'] and status['compute_lock_held']
    assert status['liveness']=='PROCESS_EXITED_DESCENDANT_LOCK_HELD'
    deadline=time.monotonic()+20
    while time.monotonic()<deadline and not (root/'R.done').exists():time.sleep(.1)
    assert (root/'R.done').read_text().strip()=='done'
    while time.monotonic()<deadline and runtime.process_alive(r_identity):time.sleep(.1)
    with runtime.lock_file(root/'control/queue.lock'):pass
    with runtime.lock_file(GPU_LOCK):pass
    result=dict(schema='boundary-three-tiny-R-lock-probe-v1',status='passed',scope='engineering-only: no evaluator or scientific experiment',
                created_utc=utc(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                source_digest=digest(source_manifest()),parent=fds['parent'],R=r_identity,inherited_R_fd_paths=observed,
                actual_parent_SIGKILL=True,locks_held_after_python_parent_exit=True,
                independent_status_liveness=status['liveness'],R_completed_naturally=True,locks_released_after_R_exit=True,
                config=identity(root/'configs/run.json'),R_log=identity(root/'R.log'))
    atomic_json(root/'receipt.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    if sys.argv[1]=='parent':parent(sys.argv[2])
    else:main(sys.argv[1])
