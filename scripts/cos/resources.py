"""Measured storage estimate; no historical artifact deletion to make room."""
import argparse,json,os,shutil,subprocess
from pathlib import Path
from scripts.crgf.common import atomic_json,identity

def allocated(root):
    total=0;seen=set()
    for p in Path(root).rglob('*'):
        if p.is_file():
            s=p.stat();key=(s.st_dev,s.st_ino)
            if key not in seen:total+=s.st_blocks*512;seen.add(key)
    return total

def measure(queuepath):
    q=json.loads(Path(queuepath).read_text());r=Path(q['output_root']);hist=Path('/home/zwan0726/work/outputs/crgf_egr_20261008T121953+1100/G')
    old={name:allocated(hist/name) for name in ['checkpoints','predictions','matrix','evaluation']}
    # Extra raw chunk maps: exactly 35,418 x 48 x 48 uint32. Five extra full
    # field maps (raw two + tile merge + split + pre-global) are uncompressed.
    inventories={s:json.loads((r/f'configs/patches_shift{s}.json').read_text()) for s in (0,24)}
    import tifffile
    machine=json.loads(Path(q['methods']['G_CONT']['machine']).read_text());shape=tifffile.TiffFile(machine['nuclei_mask']).pages[0].shape
    raw_chunk_bytes=sum(len(x) for x in inventories.values())*48*48*4
    extra_map_bytes=5*shape[0]*shape[1]*4
    probes=json.loads((r/'configs/probes.json').read_text());probe_bytes=probes['estimated_uncompressed_bytes']
    # Upper bound dense intermediate counts, irrespective of compression ratio.
    import numpy as np
    old_counts=np.load(hist/'matrix/G/counts.npy',mmap_mode='r');stage_counts_bound=9*old_counts.nbytes
    mini=r/'tests/mini_chain_v2';mini_bytes=allocated(mini) if mini.exists() else None
    stage_dir=mini/'B1_CONT/evaluation/B1_CONT/diagnostics/stages'
    fixture=json.loads((mini/'fixture.json').read_text()) if (mini/'fixture.json').exists() else {}
    sampled={key:0 for key in ('compressed_counts','cell_tables','aggregate_tables')}
    for p in stage_dir.glob('*'):
        if p.is_file():
            key='compressed_counts' if p.name.startswith('counts_') else 'cell_tables' if p.name.startswith(('fixed_','cells_')) else 'aggregate_tables'
            sampled[key]+=p.stat().st_size
    full_cells=old_counts.shape[0];sample_cells=fixture.get('cell_count',0)
    if not sample_cells or not sampled['compressed_counts'] or not sampled['cell_tables']:raise ValueError('real mini diagnostics storage measurement required')
    ratio=full_cells/sample_cells
    diagnostic_estimate=int(2*((sampled['compressed_counts']+sampled['cell_tables'])*ratio+sampled['aggregate_tables']))
    comparison_measurements={'cell_tables':0,'aggregate_and_receipts':0}
    for name in ('pair_B1','pair_G','four_arm'):
        for p in (mini/name).rglob('*'):
            if p.is_file():
                header=p.open().readline().strip().split(',') if p.suffix=='.csv' else []
                key='cell_tables' if any(x in header for x in ('cell_id','canonical_cell_id')) else 'aggregate_and_receipts'
                comparison_measurements[key]+=p.stat().st_size
    comparison_measured=sum(comparison_measurements.values())
    comparison_estimate=int(2*(comparison_measurements['cell_tables']*ratio+comparison_measurements['aggregate_and_receipts'])) if comparison_measured else None
    per_arm=sum(old.values())+raw_chunk_bytes+extra_map_bytes+probe_bytes+diagnostic_estimate
    temporary_bound=max(old['checkpoints']//9,extra_map_bytes//5,old_counts.nbytes)*2
    if comparison_estimate is None:raise ValueError('mini pair/four summary storage measurement required')
    shared_coordinates=allocated(hist.parent/'shared/coordinates')
    total_bound=4*per_arm+temporary_bound+comparison_estimate+shared_coordinates
    disk=shutil.disk_usage(r)
    memory={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:','MemAvailable:','SwapFree:'))}
    old_status_path=hist.parent/'run_status.json';old_state=json.loads(old_status_path.read_text())
    historical_resources={name:[{k:a.get(k) for k in ('wall_seconds','sampled_peak_rss_bytes_sum','sampled_peak_pss_bytes_sum')} for a in stage['attempts']] for name,stage in old_state['stages'].items() if name.startswith('G_')}
    historical_pss=max(a.get('sampled_peak_pss_bytes_sum',0) or 0 for values in historical_resources.values() for a in values)
    diagnostic_working_bound=12*shape[0]*shape[1]*4+4*old_counts.nbytes+3*shape[0]*shape[1]*8
    cpu_estimate=int(1.5*(historical_pss+diagnostic_working_bound))
    hist_train=json.loads((hist/'checkpoints/G/complete.json').read_text());hist_predict=json.loads((hist/'predictions/G/complete.json').read_text())
    gpu_estimate=int(1.5*max(hist_train['peak_vram_bytes'],hist_predict['peak_vram_bytes']))
    mini_peak_gpu=max(json.loads((Path(item['cost_root'])/'checkpoints'/arm/'complete.json').read_text())['peak_vram_bytes'] for arm,item in json.loads((mini/'configs/queue.json').read_text())['methods'].items())
    gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,capture_output=True)
    gpu_available_bytes=None
    if gpu.returncode==0:
        fields=gpu.stdout.strip().splitlines()[0].rsplit(',',3)
        gpu_available_bytes=(int(fields[-3].strip())-int(fields[-2].strip()))*1024**2
    result=dict(version='COS-storage-resource-estimate-v1',historical_component_allocated_bytes=old,raw_chunk_extra_bytes_per_arm=raw_chunk_bytes,full_map_extra_bytes_per_arm=extra_map_bytes,probe_uncompressed_estimated_bytes_per_arm=probe_bytes,intermediate_counts_dense_upper_bytes_per_arm=stage_counts_bound,mini_diagnostic_measurements_bytes=sampled,mini_to_full_cell_ratio=ratio,diagnostic_estimate_bytes_per_arm=diagnostic_estimate,comparison_measured_bytes=comparison_measured,comparison_measurements_bytes=comparison_measurements,comparison_estimate_bytes= comparison_estimate,extrapolation_safety_multiplier=2,formal_per_arm_upper_bytes=per_arm,formal_four_arm_upper_bytes=total_bound,temporary_peak_reserve_bytes=temporary_bound,shared_coordinate_cache_estimated_bytes=shared_coordinates,mini_existing_allocated_bytes=mini_bytes,disk=dict(total_bytes=disk.total,used_bytes=disk.used,free_bytes=disk.free,remaining_after_upper_estimate_bytes=disk.free-total_bound),memory=memory,historical_resource_receipt=identity(old_status_path),historical_full_field_resources=historical_resources,additional_diagnostic_working_bound_bytes=diagnostic_working_bound,estimated_peak_cpu_pss_bytes=cpu_estimate,estimated_peak_gpu_allocator_bytes=gpu_estimate,mini_measured_peak_gpu_allocator_bytes=mini_peak_gpu,cpu_estimate_scope='Full historical maximum PSS plus full-field diagnostic live-array bounds, 1.5x safety; mini front CPU sampling limited to matrix, evaluation separately monitored',gpu_estimate_scope='Historical full-field max includes full candidate K; 1.5x safety for COS, scores and checkpoint temporary tensors',gpu_query=gpu.stdout.strip(),gpu_query_returncode=gpu.returncode,gpu_available_bytes=gpu_available_bytes,minimum_free_bytes=int(q['minimum_free_gb']*1024**3),estimate_method='existing full G measured bytes plus exact raw/extra endpoints, real mini compressed counts and per-cell tables scaled to full cells with 2x safety, pair/four outputs also 2x cell scaling, two largest atomic checkpoint temporaries; hardlinks counted once. Compression extrapolation is an estimate; runtime disk guard stays active.',sufficient=disk.free-total_bound>q['minimum_free_gb']*1024**3 and memory['MemAvailable']>cpu_estimate and gpu_available_bytes is not None and gpu_available_bytes>gpu_estimate)
    atomic_json(r/'tests/resources.json',result);print(json.dumps(result,indent=2));return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();measure(a.config)
