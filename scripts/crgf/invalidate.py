"""Explicit stopped-queue repair: preserve old artifacts and rerun only a stage chain."""
import argparse
import fcntl
import json
import os
import time
from pathlib import Path
from .common import atomic_json
from .run_stage1 import is_alive,stage_specs
from .config import read


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
        if stage=='train':targets=[root/'checkpoints'/method]
        elif stage=='predict':targets=[root/'predictions'/method]
        elif stage=='postprocess':targets=[root/'predictions'/method/'postprocess',root/'predictions'/method/'final.tif']
        elif stage=='matrix':targets=[root/'matrix'/method,root/'assignment'/method]
        elif stage=='evaluate':targets=[root/'evaluation'/method]
        else:targets=[root/f for f in ['results.json','raw_comparison.csv','report.md','report_complete.json']]
        for target in targets:
            if target.exists() and target not in moved:
                destination=archive/target.relative_to(root);destination.parent.mkdir(parents=True,exist_ok=True);os.replace(target,destination);moved.add(target)
        record.setdefault('invalidations',[]).append(dict(reason=a.reason,archive=str(archive),previous_status=record['status'],receipt_sha256=record.get('receipt_sha256')))
        record.update(status='pending',receipt_sha256=None,signature=None)
    status['status']='repair_pending';atomic_json(root/'run_status.json',status)
    atomic_json(archive/'invalidation.json',dict(reason=a.reason,affected=sorted(affected),preserved_attempts=True))
    print('Invalidated:',', '.join(sorted(affected)),'; review and freeze code before resume.')
if __name__=='__main__':main()
