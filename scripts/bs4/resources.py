"""Resource guard: measured v2 budget mandatory for formal computations."""
import json,shutil
from pathlib import Path
GiB=1024**3

def require_space(root,mini=False,reserve_bytes=0):
    root=Path(root);floor=15*GiB
    needed=floor+reserve_bytes
    if not mini:
        q=root/'configs/queue.json'
        if not q.exists():q=root.parent/'configs/queue.json'
        cfg=json.loads(q.read_text());r=json.loads(Path(cfg['resource_estimate']).read_text())
        if r.get('version')!='BS4-storage-v2' or not r.get('measured'):raise ValueError('missing measured v2 resource budget')
        from scripts.cos.resources import allocated
        needed=max(needed,floor+max(0,r['planned_total_allocated_bytes']-allocated(Path(cfg['output_root']))))
    free=shutil.disk_usage(root).free
    if free<needed:raise RuntimeError(f'PAUSED_RESOURCE required={needed} free={free} deficit={needed-free}')

def measure(queuepath):
    from scripts.crgf.common import atomic_json,identity
    from scripts.cos.resources import allocated
    q=json.loads(Path(queuepath).read_text());root=Path(q['output_root']);hist=Path(q['historical_root'])/'B'
    parts={d:allocated(hist/d) for d in ('checkpoints','predictions','matrix','evaluation')}
    checkpoint=max(p.stat().st_size for p in (hist/'checkpoints/B').glob('*.pth'))
    post=hist/'predictions/B/postprocess';postbytes=allocated(post)
    raster=(hist/'predictions/B/pre.tif').stat().st_size
    # P shares immutable first five maps; own tiles plus native derived final stages.
    p_chain=postbytes+4*raster+parts['matrix']+parts['evaluation']
    t_chain=sum(parts.values())
    # Added postconnect tile set + canonical + compressed proposal exposure diagnostics.
    tilebytes=sum(p.stat().st_blocks*512 for p in post.glob('tile_*.tif'))
    extra_each=tilebytes+raster+2*raster
    validation=postbytes+5*raster+extra_each
    mini_numeric=12*checkpoint+2*GiB
    aggregate=3*allocated(Path(q['historical_ssp_root'])/'comparison')
    components=dict(two_training_chains=2*t_chain,two_postprocessing_chains=2*p_chain,
        four_added_stage_diagnostics=4*extra_each,B_disabled_validation=validation,
        four_mini_numeric_checkpoint_probes=mini_numeric,comparison=aggregate,
        prepared_teacher=GiB,atomic_peak=2*checkpoint+2*raster)
    base=sum(components.values());planned=int(base*1.2);existing=allocated(root);free=shutil.disk_usage(root).free
    r=dict(version='BS4-storage-v2',measured=False,measurement_phase='historical allocated-inode baseline; actual v2 throughput/diagnostic sizes pending',
        components=components,historical_parts=parts,planned_total_allocated_bytes=planned,buffer_fraction=.2,
        minimum_free_bytes=15*GiB,existing_allocated_bytes=existing,free_bytes=free,
        required_free_bytes=max(0,planned-existing)+15*GiB,deficit_bytes=max(0,max(0,planned-existing)+15*GiB-free),
        sufficient=free>=max(0,planned-existing)+15*GiB,
        diagnostic_budget_note='Uncompressed sparse proposal upper estimate 2 full rasters per arm; replace using actual P tile throughput/bytes before admission')
    atomic_json(root/'tests/resources_preliminary.json',r);print(json.dumps(r,indent=2));return r

if __name__=='__main__':
    import sys
    measure(sys.argv[1])

def remaining_required(cfg):
    from scripts.cos.resources import allocated
    r=json.loads(Path(cfg['resource_estimate']).read_text())
    if r.get('version')!='BS4-storage-v2' or not r.get('measured'):raise ValueError('unmeasured resource budget')
    return 15*GiB+max(0,r['planned_total_allocated_bytes']-allocated(Path(cfg['output_root'])))


def measure_final(queuepath):
    """Actual full-field B diagnostics and real P production-tile rates determine v2 additions."""
    import math,subprocess
    from scripts.crgf.common import atomic_json,identity,verify_outputs
    from scripts.cos.resources import allocated
    q=json.loads(Path(queuepath).read_text());root=Path(q['output_root']);hist=Path(q['historical_root'])/'B';replay=root/'diagnostics/B_native_replay'
    required=[root/'tests/post_real_v2/complete.json',replay/'complete.json',root/'tests/report_population_gzip_v2.json',root/'diagnostics/numeric_admission_v2.json',root/'tests/mini_v5/mini_complete.json',root/'tests/aggregate_deployed_v2.json',root/'tests/deployment_test.json']
    for p in required:
        r=json.loads(p.read_text())
        if r.get('status') not in ('passed','succeeded'):raise ValueError('unpassed resource measurement '+str(p))
        if r.get('outputs'):verify_outputs(r)
    post=json.loads(required[0].read_text());oldpost=json.loads((hist/'predictions/B/postprocess/manifest.json').read_text());tiles=len(oldpost['coords']);raster=(hist/'predictions/B/pre.tif').stat().st_blocks*512
    # Native shared prediction endpoints are symlinks; inherited native tiles/finals are counted once.
    native_prediction=allocated(hist/'predictions');native_p_prediction=allocated(hist/'predictions/B/postprocess')+sum((hist/'predictions/B'/(name+'.tif')).stat().st_blocks*512 for name in ('tile_merge','split_cleanup','before_global_nuclear_restore','final'));checkpoint=allocated(hist/'checkpoints');matrix=allocated(hist/'matrix')
    profiles=sum(allocated(hist/'evaluation/B'/d) for d in ('paper_aligned_v1','engineering'))
    stage_bytes=allocated(replay/'evaluation/diagnostics/stages')
    extras={};post_seconds={}
    for arm in ('P1','P2','B'):
        rows=[r for r in post['rows'] if r['arm']==arm]
        extras[arm]=tiles*max(sum(Path(json.loads(Path(row['receipt']['path']).read_text())['outputs'][key]['path']).stat().st_blocks*512 for key in ('postconnect','diagnostics')) for row in rows)+raster
        post_seconds[arm]=math.ceil((json.loads((replay/'predictions/B/postprocess/complete.json').read_text())['wall_seconds'] if arm=='B' else max(r['wall_seconds'] for r in rows)*tiles/4)*1.2)
    # Exact table schema count for main + 10 joint-QC + 6 own-QC + all stage intervals.
    large_table_count=13+10*13+6+19+3*15
    samples=list((root/'tests/report_population_gzip_v2').rglob('*.csv.gz'))
    max_table=max(p.stat().st_blocks*512 for p in samples)
    oldcomp=Path(q['historical_ssp_root'])/'comparison/diagnostics'
    pair_dirs=[p for p in oldcomp.iterdir() if p.is_dir() and '_minus_' in p.name]
    pair_upper=max(allocated(p) for p in pair_dirs)
    comparison=large_table_count*max_table+7*pair_upper+256*1024**2
    components={}
    for arm in ('P1','P2','T1','T2'):
        components[arm+'_native_predictions']=native_p_prediction if arm.startswith('P') else native_prediction
        components[arm+'_postconnect_proposals']=extras[arm if arm.startswith('P') else 'B']
        components[arm+'_own_matrix']=matrix
        components[arm+'_three_profiles']=profiles
        components[arm+'_all_10_stage_counts_labels_transfers']=stage_bytes
        if arm.startswith('T'):components[arm+'_four_milestone_checkpoints']=checkpoint
    components['six_method_all_domains_comparison']=comparison
    largest_checkpoint=max(p.stat().st_size for p in (hist/'checkpoints/B').glob('*.pth'))
    components['atomic_peak_checkpoints_and_rasters']=2*largest_checkpoint+2*raster
    components['future_formal_logs_probes_source_bundle']=512*1024**2
    # Every already-written failed/validation attempt stays allocated. No historical deletion credit.
    admission_test_allocations={name:allocated(root/'tests'/name) for name in ('mini_v4','mini_v5','deployment','numeric_v2')};existing=allocated(root);remaining=int(math.ceil(sum(components.values())*1.2));planned=existing+remaining;disk=shutil.disk_usage(root)
    num=json.loads(required[3].read_text());gpu_peak=int(math.ceil(max(a['peak_vram_bytes'] for a in num['arms'].values())*1.2))
    historical=json.loads((Path(q['historical_ssp_root'])/'run_status.json').read_text())
    old_pss=max(a.get('sampled_peak_pss_bytes_sum',0) for stage in historical['stages'].values() for a in stage.get('attempts',[]))
    cpu_peak=int(math.ceil(max(old_pss,4*max(r['peak_rss_bytes'] for r in post['rows'] if 'peak_rss_bytes' in r))*1.2))
    memory={l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith(('MemAvailable:','SwapFree:'))}
    gpu=subprocess.run(['nvidia-smi','--id=0','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=30)
    gpu_free=int(gpu.stdout.strip())*1024**2 if gpu.returncode==0 else None
    needed=remaining+15*GiB
    result=dict(version='BS4-storage-v2',measured=True,components=components,existing_allocated_bytes=existing,admission_test_allocations=admission_test_allocations,remaining_admission_tests_bytes=0,planned_total_allocated_bytes=planned,buffer_fraction=.2,minimum_free_bytes=15*GiB,free_bytes=disk.free,required_free_bytes=needed,deficit_bytes=max(0,needed-disk.free),allocated_policy='unique device/inode; symlinks excluded; all prior attempts retained',large_comparison_tables=large_table_count,largest_measured_gzip_table_allocated=max_table,production_tile_count=tiles,postprocess_estimated_seconds=post_seconds,measurement_inputs={str(p):identity(p) for p in required},estimated_peak_gpu_allocator_bytes=gpu_peak,estimated_serial_cpu_peak_bytes=cpu_peak,minimum_ram_remainder_bytes=6*GiB,memory=memory,gpu_available_bytes=gpu_free,sufficient=disk.free>=needed and memory['MemAvailable']>=cpu_peak+6*GiB and gpu_free is not None and gpu_free>=gpu_peak)
    atomic_json(root/'tests/resources.json',result)
    if not result['sufficient']:raise RuntimeError('PAUSED_RESOURCE actual resource budget is insufficient: '+json.dumps(result))
    return result


def preview_v2(queuepath):
    """Measured schema sizes; explicitly pending native P throughput/full-B diagnostic peaks."""
    import math
    from scripts.cos.resources import allocated
    from scripts.crgf.common import atomic_json,identity
    q=json.loads(Path(queuepath).read_text());root=Path(q['output_root']);b=Path(q['historical_root'])/'B';p=b/'predictions/B';raster=(p/'pre.tif').stat().st_blocks*512;tiles=48
    native_p=allocated(p/'postprocess')+sum((p/(name+'.tif')).stat().st_blocks*512 for name in ('tile_merge','split_cleanup','before_global_nuclear_restore','final'))
    oldstage=b/'evaluation/B/diagnostics/stages';compressed=list((root/'tests/report_population_gzip_v2').rglob('*.csv.gz'));max_table=max(f.stat().st_blocks*512 for f in compressed)
    stage_count=max(f.stat().st_blocks*512 for f in oldstage.glob('counts_*.npz'))*10
    stage_cells=max(f.stat().st_blocks*512 for f in oldstage.glob('cells_*.csv'))*10
    stage_other=max(sum(f.stat().st_blocks*512 for f in oldstage.iterdir() if f.is_file() and not f.name.startswith(('counts_','cells_','fixed_'))),allocated(root/'tests/mini_v5/P2/evaluation/P2/diagnostics/stages')//2)*2
    stage_all=stage_count+stage_cells+10*max_table+stage_other
    extras={}
    for arm in ('P1','P2','T1','T2'):
        folder=root/'tests/mini_v5'/arm/'predictions'/arm/'postprocess'
        if not list(folder.glob('tile_*.json')):folder=root/'tests/mini_v4'/arm/'predictions'/arm/'postprocess'
        rec=json.loads(next(folder.glob('tile_*.json')).read_text());h,w=rec['shape'];diag=Path(rec['outputs']['diagnostics']['path']).stat().st_blocks*512
        extras[arm]=tiles*(1024*1024*4+math.ceil(diag*1024*1024/(h*w)))+raster
    profiles=sum(allocated(b/'evaluation/B'/d) for d in ('paper_aligned_v1','engineering'))
    oldcomp=Path(q['historical_ssp_root'])/'comparison/diagnostics';pairmax=max(allocated(d) for d in oldcomp.iterdir() if d.is_dir() and '_minus_' in d.name)
    components={}
    for arm in ('P1','P2','T1','T2'):
        components[arm+'_native_prediction']=native_p if arm.startswith('P') else allocated(b/'predictions')
        components[arm+'_new_postconnect_npz_proxy']=extras[arm]
        components[arm+'_matrix']=allocated(b/'matrix');components[arm+'_three_profiles']=profiles;components[arm+'_ten_stages']=stage_all
        if arm.startswith('T'):components[arm+'_four_checkpoints']=allocated(b/'checkpoints')
    components['six_method_comparison_213_large_tables_7pairs']=213*max_table+7*pairmax+256*1024**2
    checkpoint=max(f.stat().st_size for f in (b/'checkpoints/B').glob('*.pth'))
    components['B_validation_remaining']=max(0,native_p+max(extras.values())+stage_all-allocated(root/'diagnostics/B_native_replay'))
    components['numeric_four_checkpoints_pending']=max(0,4*checkpoint-allocated(root/'tests/numeric_v2'))
    measured_mini=allocated(root/'tests/mini_v4')
    components['mini_v5_remaining']=max(0,measured_mini-allocated(root/'tests/mini_v5'))
    components['deployed_real_four_arm_mini']=measured_mini+256*1024**2
    components['P_native_bench_four_tiles']=int(max(extras.values())/tiles*4)+4*1024*1024*4
    components['atomic_peak']=2*checkpoint+2*raster;components['logs_probes_deployment']=512*1024**2
    base=sum(components.values());remaining=math.ceil(base*1.2);free=shutil.disk_usage(root).free
    result=dict(version='BS4-storage-v2-preliminary',measured=False,components=components,required_remaining_with_20pct_bytes=remaining,minimum_free_bytes=15*GiB,free_bytes=free,required_free_bytes=remaining+15*GiB,deficit_bytes=max(0,remaining+15*GiB-free),headroom_bytes=free-remaining-15*GiB,existing_unique_inode_allocated_bytes=allocated(root),pending=['production1024 P1/P2 bytes/RSS/time','completed fullB10stage measured size','complete aggregate mini/deployment remaining actual allocation'],method='actual historical native paths; P shares shifts/patch chunks/probes and owns only postprocess+4 downstream rasters; old full-field counts plus measured deterministic 103000-row gzip table maximum; new NPZ currently actual mini bytes/pixel proxy, not admission')
    atomic_json(root/'tests/resources_preliminary_v3.json',result);return result
