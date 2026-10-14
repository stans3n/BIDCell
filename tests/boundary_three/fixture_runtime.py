"""Explicit engineering fixture for actual tmux/locks/process lifecycle tests.

Never imported by production. The formal CLI rejects this fixture's marker and
interpreter. No scientific inputs, GPU, masks, matrices or metrics are produced.
"""
import argparse
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch
from scripts.boundary_three import runtime, cli
from scripts.boundary_three.common import (atomic_json, digest, identity, load_run,
    output_path, progress, read, stop_requested, validate_complete, write_complete)
from scripts.boundary_three.contracts import NODES, PYTHON
from scripts.boundary_three.settings import default_config


def run_node(cfg,node):
    root=output_path(cfg,node);root.mkdir(parents=True,exist_ok=True)
    cursor=root/'fixture_checkpoint.json'
    current=read(cursor)['cursor'] if cursor.exists() else 0
    with (root/'visits.log').open('a') as stream:
        total=40 if node=='P2_BASE' else 12
        for i in range(current,total):
            if stop_requested(cfg):raise InterruptedError('engineering fixture committed cursor safe stop')
            time.sleep(.4 if node=='P2_BASE' else .25)
            stream.write(str(i)+'\n');stream.flush();os.fsync(stream.fileno())
            atomic_json(cursor,dict(cursor=i+1,engineering_fixture=True))
            progress(cfg,node,'ENGINEERING_FIXTURE',tile_cursor=i+1,total_tiles=total)
            print(node,'fixture cursor',i+1,flush=True)
    science='NO_USEFUL_INCREMENT' if node=='A1_CORE' else 'tradeoff_or_negative' if node.startswith('P2_') else 'ENGINEERING_ONLY'
    write_complete(cfg,node,science,engineering_fixture=True)
    return validate_complete(cfg,node)


def init(root):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False)
    cfg=default_config(root,'engineering-runtime-'+root.name)
    cfg.update(engineering_fixture=True,source_root=str(Path(__file__).resolve().parents[2]),
               source_digest='engineering-only',input_digest='engineering-only',budget_seconds=120)
    wrapper=root/'fixture-python'
    wrapper.write_text('#!/usr/bin/env bash\nset -euo pipefail\nshift 2\nexec '+PYTHON+' -m tests.boundary_three.fixture_runtime "$@"\n')
    wrapper.chmod(0o700);cfg['python']=str(wrapper)
    marker=root/'fixture_evidence.json';atomic_json(marker,dict(status='ENGINEERING_FIXTURE_ONLY'))
    cfg['node_estimates']={n:dict(seconds=4,retained_bytes=0,atomic_extra_bytes=0,
                                peak_ram_bytes=0,peak_gpu_bytes=0,evidence=[identity(marker)]) for n in NODES}
    cfg['resources'].update(resource_wait_seconds=3,disk_reserve_bytes=0,min_ram_available_bytes=0,min_gpu_free_bytes=0)
    path=root/'configs/run.json';atomic_json(path,cfg)
    atomic_json(root/'admission/validated.json',dict(independent_admission=identity(marker)))
    runtime.save_state(cfg,runtime.new_state(cfg))
    print(path)


def main():
    if len(sys.argv)>1 and sys.argv[1]=='init':
        init(sys.argv[2]);return 0
    config=Path(sys.argv[sys.argv.index('--config')+1]);cfg=load_run(config)
    if cfg.get('engineering_fixture') is not True:raise ValueError('fixture cannot operate formal scientific configuration')
    module=SimpleNamespace(run_node=run_node,validate_node=validate_complete)
    resources=dict(ram_available_bytes=10**12,gpu_free_bytes=10**12,disk_free_bytes=10**12)
    with patch.object(runtime,'validated_launch',return_value={}), \
         patch.object(runtime,'verify_worker_authorization',return_value=None), \
         patch.object(runtime,'verify_source',return_value={}), \
         patch.object(runtime,'node_module',return_value=module), \
         patch.object(runtime,'resources_now',return_value=resources), \
         patch.object(runtime,'GPU_LOCK',output_path(cfg,'control/fixture_compute.lock')), \
         patch.object(runtime,'PACKAGE_SECONDS',1), \
         patch('scripts.boundary_three.provenance.verify_inputs',return_value={}), \
         patch('scripts.boundary_three.provenance.environment_check',return_value={}):
        return cli.main()


if __name__=='__main__':sys.exit(main())
