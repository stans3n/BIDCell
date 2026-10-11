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
