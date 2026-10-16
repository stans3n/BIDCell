"""F01/F14 actual Simple maximum-candidate tiles; never a formal endpoint."""
import argparse,fcntl,json,multiprocessing as mp,os,random,resource,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import tifffile
from scripts.rep1_triple_simple_boundary.common import atomic_json,atomic_tiff,identity,inventory,check_identity
from scripts.rep1_triple_simple_boundary.p2 import native_connect
from scripts.boundary_three.p2_rules import FullNucleusIndex,connect,rng_hash
from bidcell.model import postprocess_predictions as native
PRE=NUC=INDEX=None

def worker(task):
    y,x,K,root,old=task;root=Path(root);root.mkdir()
    image=PRE[y:y+1024,x:x+1024];nuc=NUC[y:y+1024,x:x+1024];seed=1234+y*1000003+x
    start=time.monotonic();random.seed(seed);baseline,bd,ba=native_connect(image,nuc);state=random.getstate()
    cleaned=native.remove_islands(baseline,nuc);exit_rng=rng_hash()
    if len(bd['candidate_order'])!=K:raise ValueError('maximum tile K drift')
    if not np.array_equal(cleaned,tifffile.imread(old)):raise ValueError('Simple native max-tile historical pixels differ')
    atomic_tiff(root/'native_postconnect.tif',baseline);atomic_tiff(root/'native_postislands.tif',cleaned)
    atomic_json(root/'native.json',dict(diagnostics=bd,exit_rng=exit_rng,seconds=time.monotonic()-start))
    random.seed(seed);p2,pd,pa,_=connect(image,nuc,'P2',INDEX,(y,x))
    if random.getstate()!=state:raise ValueError('P2 entering-islands RNG changed')
    for key in ('candidate_order','candidate_hashes','rng_before_shuffle','rng_enter_islands','proposal_histogram'):
        if pd[key]!=bd[key]:raise ValueError('P2 candidate/RNG mismatch '+key)
    if not np.array_equal(pa['baseline_postconnect'],baseline):raise ValueError('native counterfactual differs')
    post=native.remove_islands(p2,nuc);atomic_tiff(root/'P2_postconnect.tif',p2);atomic_tiff(root/'P2_postislands.tif',post)
    atomic_json(root/'complete.json',dict(status='passed',coords=[y,x],K=K,full_nucleus=True,native_historical_tile_exact=True,
        candidate_order_hashes_exact=True,entering_RNG_exact=True,native_exit_RNG=exit_rng,P2_exit_RNG=rng_hash(),
        foreground_equal=bool(np.array_equal(baseline>0,p2>0)),max_query_pairs=INDEX.max_observed_pairs,
        worker_maxRSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,seconds=time.monotonic()-start,
        scope='four real tiles resource fixture; not complete replay or formal endpoint',outputs=inventory(root)))
    return str(root/'complete.json')

def run(root):
    global PRE,NUC,INDEX
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    inspected=json.load(open('/home/zwan0726/work/docs/BIDCell_Rep1_Triple_Simple_Inputs_20261017.json'));f=inspected['endpoints']['MORPH_SIMPLE_FUSION']['files']
    for key in ('pre_mask','nuclei_mask'):check_identity(f[key])
    sources={str(p):identity(p) for p in [Path(__file__),Path(native.__file__),Path(__file__).parents[1]/'scripts/boundary_three/p2_rules.py',Path(__file__).parents[1]/'scripts/rep1_triple_simple_boundary/p2.py']}
    lock=Path('/home/zwan0726/work/outputs/cos_gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.monotonic();PRE=tifffile.imread(f['pre_mask']['path']);NUC=tifffile.imread(f['nuclei_mask']['path']);INDEX=FullNucleusIndex(NUC)
    manifest=json.load(open(Path(f['pre_mask']['path']).parent/'postprocess/manifest.json'));catalog=sorted([(len(np.unique(PRE[y:y+1024,x:x+1024]))-1,y,x) for y,x in manifest['coords']],reverse=True)
    if catalog[0]!=(5199,4096,0):raise ValueError('actual Simple maximum candidate tile changed')
    tasks=[(y,x,K,str(root/f'tile_{y}_{x}'),str(Path(f['pre_mask']['path']).parent/'postprocess'/f'tile_{y}_{x}.tif')) for K,y,x in catalog[:4]]
    peaks=dict(tree_pss_bytes=0,tree_private_bytes=0,tree_rss_bytes=0,processes=0);samples=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('fork')) as pool:
        jobs=[pool.submit(worker,task) for task in tasks]
        while not all(j.done() for j in jobs):
            pids=[os.getpid()]+list(pool._processes);tot=dict(tree_pss_bytes=0,tree_private_bytes=0,tree_rss_bytes=0,processes=len(pids))
            for pid in pids:
                try:
                    stats={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc',str(pid),'smaps_rollup').read_text().splitlines()[1:] if ':' in line}
                    tot['tree_pss_bytes']+=stats['Pss'];tot['tree_private_bytes']+=stats.get('Private_Clean',0)+stats.get('Private_Dirty',0);tot['tree_rss_bytes']+=stats['Rss']
                except FileNotFoundError:pass
            for key in peaks:peaks[key]=max(peaks[key],tot[key])
            samples.append(dict(elapsed_seconds=time.monotonic()-start,**tot));atomic_json(root/'progress.json',dict(maxima=peaks,completed=sum(j.done() for j in jobs),elapsed_seconds=time.monotonic()-start))
            time.sleep(1)
        records=[identity(j.result()) for j in jobs]
    for obj in sources.values():check_identity(obj)
    atomic_json(root/'memory_samples.json',samples)
    atomic_json(root/'complete.json',dict(status='passed',real_tiles=[list(x) for x in catalog[:4]],workers=4,peaks=peaks,wall_seconds=time.monotonic()-start,
        source=sources,inputs={key:f[key] for key in ('pre_mask','nuclei_mask')},tile_receipts=records,scope='maximum Simple candidate tiles under common lock; no full-slide results or evaluation',outputs=inventory(root)))
    print(json.dumps(dict(status='passed',peaks=peaks,wall_seconds=time.monotonic()-start)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(p.parse_args().output)
