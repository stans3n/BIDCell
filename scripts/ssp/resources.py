"""Measured two-arm storage budget, unique-inode accounting and live guards."""
import json, shutil, subprocess
from pathlib import Path
from scripts.crgf.common import atomic_json, identity
from scripts.cos.resources import allocated
GiB=1024**3

def remaining_required(cfg):
    root=Path(cfg['output_root']); path=Path(cfg.get('resource_estimate',root/'tests/resources.json'))
    if not path.exists():return (0 if cfg.get('mini_test') else 15*GiB)
    r=json.loads(path.read_text())
    return r['minimum_free_bytes']+max(0,r['planned_total_allocated_bytes']-allocated(root))

def require_space(root,mini=False,reserve_bytes=0):
    root=Path(root); queue=root/'configs/queue.json'
    if not queue.exists():queue=root.parent/'configs/queue.json'
    floor=0 if mini else 15*GiB
    needed=floor+reserve_bytes
    if queue.exists() and not mini:needed=max(needed,remaining_required(json.loads(queue.read_text())))
    if shutil.disk_usage(root).free<needed:raise RuntimeError('PAUSED_RESOURCE insufficient free disk for remaining measured outputs, 20% buffer and 15 GiB reserve')

def measure(queuepath):
    q=json.loads(Path(queuepath).read_text());root=Path(q['output_root']);hist=Path(q['historical_root']);parts={}
    for a in ('A','B'):
        r=hist/a;parts[a]={d:allocated(r/d) for d in ('checkpoints','predictions','matrix','evaluation')}
    historical_arm=max(sum(v.values()) for v in parts.values())
    comparison=3*allocated(hist/'comparison');checkpoint=max(p.stat().st_size for p in (hist/'B/checkpoints/B').glob('*.pth'))
    # Conservative additional prototype table (125k nuclei x 313 float/count),
    # two 32-step mini chains, probes and atomic checkpoint overlap.
    prototype=max(512*1024**2,allocated(root/'prepared')+allocated(root/'prepared_initial_attempt'))
    pending_numeric=0 if (root/'diagnostics/numeric_admission.json').exists() else 4*checkpoint
    pending_mini=0 if (root/'tests/mini/mini_complete.json').exists() else 2*checkpoint
    mini_probe=allocated(root/'tests')+pending_numeric+pending_mini+512*1024**2
    atomic=2*checkpoint
    planned_base=2*historical_arm+comparison+prototype+mini_probe+atomic
    buffer=int(.2*planned_base);planned=planned_base+buffer;already=allocated(root);remaining=max(0,planned-already)
    mem={l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith(('MemTotal:','MemAvailable:','SwapFree:'))}
    peaks=[]
    numeric=root/'diagnostics/numeric_admission.json'
    if numeric.exists():
        nr=json.loads(numeric.read_text())
        peaks=[nr['arms'][a]['peak_vram_bytes'] for a in ('S','SP')]
    for a in ('S','SP'):
        for folder in ('checkpoints','predictions'):
            measured=root/'tests/mini'/a/folder/a/'complete.json'
            if measured.exists():peaks.append(json.loads(measured.read_text()).get('peak_vram_bytes',0))
    historical=max(json.loads((hist/'B'/d/'B/complete.json').read_text()).get('peak_vram_bytes',0) for d in ('checkpoints','predictions'))
    gpu_peak=int(max([historical,*peaks])*1.2)
    historical_state=json.loads((hist/'run_status.json').read_text())
    historical_pss=max((a.get('sampled_peak_pss_bytes_sum',0) or 0 for stage in historical_state['stages'].values() for a in stage.get('attempts',[])),default=0)
    cpu_peak=int(historical_pss*1.2)+512*1024**2
    gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True)
    free_gpu=int(gpu.stdout.strip().splitlines()[0])*1024**2 if gpu.returncode==0 else None
    historical_timings={stage:historical_state['stages']['B_'+stage]['attempts'][-1]['wall_seconds'] for stage in ('train','predict','postprocess','matrix','evaluate')}
    estimates={}
    if numeric.exists():
        for arm in ('S','SP'):
            numeric_train=4000*sum(v['seconds'] for v in nr['arms'][arm]['rows'])/32
            rows_path=root/'tests/mini'/arm/'checkpoints'/arm/'updates.jsonl'
            mini_rows=[json.loads(line) for line in rows_path.read_text().splitlines()] if rows_path.exists() else []
            mini_train=4000*sum(v['seconds'] for v in mini_rows)/len(mini_rows) if mini_rows else 0
            train=max(numeric_train,mini_train,historical_timings['train'])
            mini=root/'tests/mini'/arm/'predictions'/arm/'complete.json'
            if mini.exists():
                pr=json.loads(mini.read_text());small=sum(v['patch_count'] for v in pr['shifts']);full=sum(len(json.loads((root/arm/f'configs/patches_shift{shift}.json').read_text())) for shift in (0,24))
                predict=pr['wall_seconds']*full/small
                estimates[arm]=dict(train=train,predict=predict,postprocess=historical_timings['postprocess'],matrix=historical_timings['matrix'],evaluate=historical_timings['evaluate'],train_basis='max of 32-update model-only timing (excludes dataset/teacher/prototype/device-transfer), mini full data+teacher+update timing, and historical full-training duration; add 20 percent plus 300 seconds checkpoint/probe reserve',numeric_model_only_seconds=numeric_train,mini_data_teacher_update_seconds=mini_train,predict_basis='actual regional double-shift patches extrapolated to full patch inventory; small ROI startup and candidate mix create substantial uncertainty; reserve 20 percent')
                estimates[arm]['train']*=1.2
                estimates[arm]['train']+=300
                estimates[arm]['predict']*=1.2
    disk=shutil.disk_usage(root);r=dict(measured_stage_estimates=estimates,version='SSP-storage-v1',historical_parts=parts,historical_arm_bytes=historical_arm,comparison_bytes=comparison,prototype_budget_bytes=prototype,mini_probe_budget_bytes=mini_probe,pending_numeric_checkpoint_bytes=pending_numeric,pending_mini_checkpoint_bytes=pending_mini,future_probe_allowance_bytes=512*1024**2,atomic_peak_bytes=atomic,prediction_error_buffer_bytes=buffer,buffer_fraction=.2,planned_total_allocated_bytes=planned,existing_allocated_unique_inodes=already,remaining_workspace_bytes=remaining,minimum_free_bytes=15*GiB,free_bytes=disk.free,required_free_bytes=remaining+15*GiB,memory=mem,historical_peak_pss_bytes=historical_pss,estimated_serial_cpu_peak_bytes=cpu_peak,minimum_ram_remainder_bytes=6*GiB,estimated_peak_gpu_allocator_bytes=gpu_peak,gpu_available_bytes=free_gpu,gpu_measurement='host nvidia-smi',sufficient=disk.free>=remaining+15*GiB and mem['MemAvailable']>=cpu_peak+6*GiB and free_gpu is not None and free_gpu>=gpu_peak)
    r['historical_B_stage_seconds']=historical_timings
    r['forecast_uncertainty']='Conservative regional extrapolation; historical full-field prediction is a separate baseline. Forecast updates from actual formal stages; eight hours remains a target, never a truncation limit.'
    atomic_json(root/'tests/resources.json',r);return r
