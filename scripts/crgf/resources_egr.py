"""Host-visible, non-invasive resource admission for fresh launch and resume."""
import json,shutil,subprocess,time
from pathlib import Path
from .common import atomic_json
HIST=Path('/home/zwan0726/work/outputs/crgf_1008_20261008T031528+1100')
def check(root):
    root=Path(root)
    query=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    if query.stdout.strip():raise RuntimeError('GPU compute task already active; refusing resource contention: '+query.stdout.strip())
    sizes={str(p):int(subprocess.check_output(['du','-s','-B1',str(p)],text=True).split()[0]) for stage in ('checkpoints','predictions','matrix','evaluation') for p in [HIST/stage/'F1']}
    per=sum(sizes.values());existing=sum(p.stat().st_size for d in ('E','G','R') for p in (root/d).rglob('*') if p.is_file())
    # Conservative fresh 3-group upper estimate + paper annotation outputs +
    # retained failed attempts/reference/test capacity and filesystem reserve.
    remaining=max(0,3*per-existing)+4*1024**3+12*1024**3
    free=shutil.disk_usage(root).free
    result=dict(status='succeeded' if free>=remaining else 'insufficient_disk',historical_allocated_bytes=sizes,per_method_bytes=per,current_method_bytes=existing,estimated_additional_bytes=remaining,paper_outputs_allowance_bytes=4*1024**3,recovery_reserve_bytes=12*1024**3,free_bytes=free,gpu_compute_query=query.stdout,checked_ns=time.time_ns())
    atomic_json(root/'source/resource_admission.json',result)
    if free<remaining:raise OSError('insufficient space for three whole-field chains and recovery reserve')
    return result
