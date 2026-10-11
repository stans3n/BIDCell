"""Real native-size tiles, historical oracle, strict reuse and corruption faults."""
import json,random,time,copy,argparse,shutil,os
from pathlib import Path
import numpy as np
import tifffile
from scripts.bs4 import postprocess as impl
from scripts.bs4.post_rules import FullNucleusIndex
from scripts.crgf.common import identity,atomic_json,source_identity

def run(root):
    root=Path(root);out=root/'tests/post_real_v2';out.mkdir(parents=True,exist_ok=True)
    old=Path('/home/zwan0726/work/outputs/cos_loss_20261009T160351+1100/B/predictions/B')
    complete=json.loads((old/'postprocess/complete.json').read_text());deps=complete['dependencies']
    impl._WHOLE=tifffile.imread(deps['pre']['path']);impl._NUCLEI=tifffile.imread(deps['nuclei']['path']);impl._INDEX=FullNucleusIndex(impl._NUCLEI)
    allrows=[];source=source_identity(['scripts/bs4/postprocess.py','scripts/bs4/post_rules.py','bidcell/model/postprocess_predictions.py'])
    for y,x in [(0,0),(1024,1024)]:
        results={}
        for arm in ('B','P1','P2'):
            dest=out/arm/f'tile_{y}_{x}.tif';dest.parent.mkdir(exist_ok=True)
            dep=dict(base=dict(pre=deps['pre'],nuclei=deps['nuclei'],seed=20261008,arm=arm,run=root.name,rule=arm,origin=[0,0],source=source,config='real native 1024 admission'),coords=[y,x],shape=[1024,1024],dtype='uint32')
            task=(y,x,1024,20261008,str(dest),dep,True);started=time.monotonic();receipt=Path(impl.tile_task(task));seconds=time.monotonic()-started
            r=impl.validate_tile(receipt,dep);results[arm]=r
            before=identity(dest);connect_before=identity(r['outputs']['postconnect']['path'])
            impl.tile_task(task);assert identity(dest)==before and identity(r['outputs']['postconnect']['path'])==connect_before
            allrows.append(dict(arm=arm,coords=[y,x],wall_seconds=seconds,receipt=identity(receipt),bytes=sum(v['bytes'] for v in r['outputs'].values()),diagnostics=r['diagnostics']))
            if arm=='B':
                historical=tifffile.imread(old/'postprocess'/dest.name)
                assert np.array_equal(tifffile.imread(dest),historical),'disabled native tile changed'
        assert len({json.dumps(v['diagnostics']['candidate_hashes'],sort_keys=True) for v in results.values()})==1
        assert len({v['diagnostics']['rng_enter_islands'] for v in results.values()})==1
    # Strict read validation faults preserve real artifacts; faults operate only on copies.
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
    truncated=out/'truncated.tif';truncated.write_bytes(b'II')
    bad=copy.deepcopy(r);bad['outputs']['postconnect']['path']=str(truncated);file=out/'fault_hash.json';atomic_json(file,bad)
    try:impl.validate_tile(file,dep)
    except ValueError:rejected.append('truncated_hash')
    else:raise AssertionError('corrupt tile accepted')
    atomic_json(out/'complete.json',dict(status='passed',source=source,rows=allrows,historical_native_tiles_bitwise=True,candidates_equal=True,RNG_equal=True,strict_reuse_preserved_files=True,rejected_faults=rejected))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();run(a.root)
