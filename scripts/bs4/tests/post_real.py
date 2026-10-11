"""Native-size P executions; strict full-B replay serves only as a read-only oracle."""
import json,time,copy,argparse,resource
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np,tifffile
from scripts.bs4 import postprocess as impl
from scripts.bs4.post_rules import FullNucleusIndex
from scripts.crgf.common import identity,atomic_json,source_identity,sha256
SOURCES=['scripts/bs4/postprocess.py','scripts/bs4/post_rules.py','bidcell/model/postprocess_predictions.py']

def execute(task):
    if Path(task[4]).with_suffix('.json').exists():raise ValueError('native throughput requires a new execution attempt, not cached tiles')
    start=time.monotonic();receipt=Path(impl.tile_task(task));r=impl.validate_tile(receipt,task[-2]);seconds=time.monotonic()-start
    before={k:identity(v['path']) for k,v in r['outputs'].items()};impl.tile_task(task)
    assert before=={k:identity(v['path']) for k,v in r['outputs'].items()}
    return dict(arm=r['arm'],coords=r['coords'],wall_seconds=seconds,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,receipt=identity(receipt),bytes=sum(v['bytes'] for v in r['outputs'].values()),diagnostics=r['diagnostics'],measurement='actual new production-size tile execution')

def run(root):
    root=Path(root);out=root/'tests/post_real_v2';out.mkdir(parents=True,exist_ok=True);source=source_identity(SOURCES);execution=source_identity(SOURCES+['scripts/bs4/tests/post_real.py'])
    old=Path('/home/zwan0726/work/outputs/cos_loss_20261009T160351+1100/B/predictions/B');oldcomplete=json.loads((old/'postprocess/complete.json').read_text());deps=oldcomplete['dependencies']
    replay=root/'diagnostics/B_native_replay';validation=json.loads((replay/'validation.json').read_text());assert validation['status']=='passed'
    replay_complete=json.loads((replay/'predictions/B/postprocess/complete.json').read_text());base=replay_complete['dependencies']
    if base['source']!=source:raise ValueError('disabled oracle executed post source differs')
    for name in ('pre','nuclei'):
        if base[name]['sha256']!=deps[name]['sha256'] or sha256(base[name]['path'])!=deps[name]['sha256']:raise ValueError('disabled oracle common input mismatch')
    if base['seed']!=20261008 or base['rule']!='B' or base['arm']!='B':raise ValueError('disabled oracle native rule mismatch')
    impl._WHOLE=tifffile.imread(deps['pre']['path']);impl._NUCLEI=tifffile.imread(deps['nuclei']['path']);impl._INDEX=FullNucleusIndex(impl._NUCLEI)
    allrows=[];tasks=[]
    for y,x in [(0,0),(1024,1024)]:
        receipt=replay/'predictions/B/postprocess'/f'tile_{y}_{x}.json'
        dep=dict(base=base,coords=[y,x],shape=[1024,1024],dtype='uint32');r=impl.validate_tile(receipt,dep)
        assert np.array_equal(tifffile.imread(old/'postprocess'/f'tile_{y}_{x}.tif'),tifffile.imread(r['outputs']['postislands']['path']))
        allrows.append(dict(arm='B',coords=[y,x],receipt=identity(receipt),bytes=sum(v['bytes'] for v in r['outputs'].values()),diagnostics=r['diagnostics'],measurement='read-only strict full-field disabled replay oracle; no fabricated tile time'))
        for arm in ('P1','P2'):
            dest=out/arm/f'tile_{y}_{x}.tif';dest.parent.mkdir(exist_ok=True)
            dep=dict(base=dict(pre=deps['pre'],nuclei=deps['nuclei'],seed=20261008,arm=arm,run=root.name,rule=arm,origin=[0,0],source=source,config='real native 1024 admission'),coords=[y,x],shape=[1024,1024],dtype='uint32')
            tasks.append((y,x,1024,20261008,str(dest),dep,True))
    with ProcessPoolExecutor(max_workers=4) as pool:allrows.extend(pool.map(execute,tasks))
    for xy in ([0,0],[1024,1024]):
        rows=[r for r in allrows if r['coords']==xy]
        assert len({json.dumps(r['diagnostics']['candidate_hashes'],sort_keys=True) for r in rows})==1
        assert len({r['diagnostics']['rng_enter_islands'] for r in rows})==1
    receipt=Path(allrows[-1]['receipt']['path']);r=json.loads(receipt.read_text());dep=r['dependencies'];rejected=[]
    for key,val in [('schema','stale'),('coords',[5,6]),('dtype','uint16'),('exit_code',1),('status','running')]:
        bad=copy.deepcopy(r);bad[key]=val;file=out/f'fault_{key}.json';atomic_json(file,bad)
        try:impl.validate_tile(file,dep)
        except ValueError:rejected.append(key)
        else:raise AssertionError('fault accepted '+key)
    bad=copy.deepcopy(dep);bad['base']['rule']='B'
    try:impl.validate_tile(receipt,bad)
    except ValueError:rejected.append('dependency')
    else:raise AssertionError('changed rule accepted')
    if execution!=source_identity(SOURCES+['scripts/bs4/tests/post_real.py']):raise ValueError('post benchmark execution source changed')
    atomic_json(out/'complete.json',dict(status='passed',execution_source=execution,rows=allrows,historical_native_tiles_bitwise=True,candidates_equal=True,RNG_equal=True,strict_reuse_preserved_files=True,rejected_faults=rejected,oracle_validation=identity(replay/'validation.json'),full_B_cost=identity(replay/'predictions/B/postprocess/complete.json'),outputs={str(p.relative_to(out)):identity(p) for p in out.rglob('*') if p.is_file() and p!=out/'complete.json'}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();run(a.root)
