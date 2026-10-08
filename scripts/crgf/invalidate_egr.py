"""Explicit stopped-queue repair: preserve old artifacts and rerun only a stage chain."""
import argparse
import fcntl
import json
import os
import time
from pathlib import Path
from .common import atomic_json
from .run_egr import is_alive,stage_specs
from .egr_config import read


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--stage',required=True);p.add_argument('--reason',required=True);a=p.parse_args()
    cfg=read(a.config);root=Path(cfg.output_root)
    lock=(root/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=json.loads((root/'run_status.json').read_text())
    if any(v.get('child_identity') and is_alive(v['child_identity']) for v in status['stages'].values()):raise RuntimeError('child still active')
    specs=stage_specs(root);names={s[0] for s in specs}
    if a.stage not in names:raise ValueError('unknown stage')
    affected={a.stage,'report'}
    for name,method,stage,deps,receipt in specs:
        if any(d in affected for d in deps):affected.add(name)
    archive=root/'repair_history'/str(time.time_ns());archive.mkdir(parents=True)
    moved=set()
    for name,method,stage,deps,receipt in specs:
        if name not in affected:continue
        record=status['stages'][name]
        if stage=='train':targets=[root/method/'checkpoints'/method]
        elif stage=='predict':targets=[root/method/'predictions'/method]
        elif stage=='postprocess':targets=[root/method/'predictions'/method/'postprocess',root/method/'predictions'/method/'final.tif']
        elif stage=='matrix':targets=[root/method/'matrix'/method,root/method/'assignment'/method]
        elif stage=='evaluate':targets=[root/method/'evaluation'/method if method in ('E','G','R') else root/'reevaluation'/method]
        else:targets=[root/'evaluation']
        for target in targets:
            if target.exists() and target not in moved:
                destination=archive/target.relative_to(root);destination.parent.mkdir(parents=True,exist_ok=True);os.replace(target,destination);moved.add(target)
        record.setdefault('invalidations',[]).append(dict(reason=a.reason,archive=str(archive),previous_status=record['status'],receipt_sha256=record.get('receipt_sha256')))
        record.update(status='pending',receipt_sha256=None,signature=None)
    status['status']='repair_pending';atomic_json(root/'run_status.json',status)
    atomic_json(archive/'invalidation.json',dict(reason=a.reason,affected=sorted(affected),preserved_attempts=True))
    print('Invalidated:',', '.join(sorted(affected)),'; review and freeze code before resume.')
if __name__=='__main__':main()
