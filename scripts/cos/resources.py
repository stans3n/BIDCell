"""ABC storage and capacity admission using accepted COS artifacts and unique inodes."""
import argparse,json,shutil,subprocess
from pathlib import Path
from scripts.crgf.common import atomic_json,identity
GiB=1024**3

def allocated(root):
    total=0;seen=set()
    for p in Path(root).rglob('*'):
        if p.is_symlink() or not p.is_file():continue
        s=p.stat();key=(s.st_dev,s.st_ino)
        if key not in seen:total+=s.st_blocks*512;seen.add(key)
    return total

def require_space(root,mini=False,reserve_bytes=0):
    minimum=(0 if mini else 15*GiB)+reserve_bytes
    if shutil.disk_usage(root).free<minimum:
        raise RuntimeError('PAUSED_RESOURCE free disk below 15 GiB reserve plus pending atomic write')

def measure(queuepath):
    q=json.loads(Path(queuepath).read_text());root=Path(q['output_root'])
    hist=Path('/home/zwan0726/work/outputs/crgf_cos_20261009T000650+1100');hq=json.loads((hist/'configs/queue.json').read_text())
    parts={}
    for arm in ['B1_CONT','B1_COS']:
        base=Path(hq['methods'][arm]['cost_root']);parts[arm]={d:allocated(base/d) for d in ['checkpoints','predictions','matrix','evaluation']}
    historical_arm=max(sum(v.values()) for v in parts.values())
    comparison=max(allocated(hist/'four_arm'),allocated(hist/'pair_B1'))
    checkpoints=list((Path(hq['methods']['B1_COS']['cost_root'])/'checkpoints/B1_COS').glob('*.pth'))
    atomic_peak=max((p.stat().st_size for p in checkpoints),default=0)*2
    already=allocated(root);disk=shutil.disk_usage(root)
    # 35 GiB is the prespecified total workspace floor including tests/probes.
    # Measured historical full chains are used as an upward override only.
    measured_remaining=int(3*historical_arm+2*comparison+atomic_peak+2*GiB)
    workspace_bound=max(35*GiB,already+measured_remaining)
    remaining=workspace_bound-already
    memory={l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith(('MemTotal:','MemAvailable:','SwapFree:'))}
    peaks=[]
    for arm in ['B1_CONT','B1_COS']:
        b=Path(hq['methods'][arm]['cost_root'])
        for stage,folder in [('train','checkpoints'),('predict','predictions')]:
            rec=json.loads((b/folder/arm/'complete.json').read_text());peaks.append(rec['peak_vram_bytes'])
    gpu_estimate=int(max(peaks)*1.2)
    old_state=json.loads((hist/'run_status.json').read_text())
    historical_pss=max((a.get('sampled_peak_pss_bytes_sum',0) or 0 for stage in old_state['stages'].values() for a in stage.get('attempts',[])),default=0)
    cpu_peak_estimate=int(1.2*historical_pss)
    gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,capture_output=True)
    available=None
    if gpu.returncode==0:
        fields=gpu.stdout.strip().splitlines()[0].rsplit(',',3);available=(int(fields[-3])-int(fields[-2]))*1024**2
    result=dict(version='COS-LOSS-ABC-resource-v2.0',historical_component_allocated_bytes=parts,historical_source=identity(hist/'compute_complete.json'),historical_full_arm_bytes=historical_arm,comparison_upper_bytes=2*comparison,temporary_atomic_peak_bytes=atomic_peak,existing_run_allocated_bytes_unique_inodes=already,required_workspace_total_bytes=workspace_bound,remaining_workspace_bytes=remaining,minimum_free_bytes=15*GiB,disk=dict(free_bytes=disk.free,remaining_after_workspace_bytes=disk.free-remaining),memory=memory,historical_peak_pss_bytes=historical_pss,estimated_serial_cpu_peak_bytes=cpu_peak_estimate,minimum_ram_remainder_bytes=6*GiB,estimated_peak_gpu_allocator_bytes=gpu_estimate,gpu_available_bytes=available,gpu_query=gpu.stdout.strip(),gpu_query_returncode=gpu.returncode,estimate_method='max(35GiB total workspace floor, allocated test/probe bytes plus 3 accepted COS full-arm footprints + 2 comparison footprints + 2 largest atomic checkpoints + 2GiB diagnostic headroom); unique device/inode, symlinks excluded; preserve 15GiB disk and 6GiB RAM',sufficient=disk.free-remaining>=15*GiB and memory['MemAvailable']>=cpu_peak_estimate+6*GiB and available is not None and available>=gpu_estimate)
    atomic_json(root/'tests/resources.json',result);print(json.dumps(result,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();measure(a.config)
