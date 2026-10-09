"""Native morphology, deterministic tile RNG, explicit tile/split receipts.
Split selection is evaluated in the full ID bounding box, exactly equivalent to
full-field 8-connected labelling (scan order and overlap tie rule preserved).
"""
import json
import os
import random
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import tifffile
from scipy import ndimage as ndi
from bidcell.model.postprocess_predictions import postprocess_connect,remove_islands
from scripts.crgf.common import atomic_json,atomic_tiff,identity,sha256,source_identity,verify_outputs,digest

from .resources import require_space

_WHOLE=None
_NUCLEI=None

def tile_task(task):
    y,x,size,seed,out,dep,mini=task;out=Path(out);receipt=out.with_suffix('.json')
    if receipt.exists():
        saved=json.loads(receipt.read_text())
        if saved['dependencies']!=dep: raise ValueError('postprocess tile dependency mismatch')
        verify_outputs(saved);return str(receipt)
    random.seed(seed+y*1000003+x)
    img=_WHOLE[y:y+size,x:x+size];nuc=_NUCLEI[y:y+size,x:x+size]
    result=remove_islands(postprocess_connect(img,nuc),nuc)
    require_space(out.parent,mini,reserve_bytes=result.nbytes)
    atomic_tiff(out,result.astype(np.uint32))
    atomic_json(receipt,dict(status='succeeded',dependencies=dep,outputs={'tile':identity(out)}))
    return str(receipt)


def split_patch(seg,nuc,cell_id,box):
    mask=(seg[box]==cell_id)
    labels,n=ndi.label(mask,structure=ndi.generate_binary_structure(2,2))
    if n>1:
        overlap=np.bincount(labels[(nuc[box]==cell_id)].ravel(),minlength=n+1)[1:]
        keep=labels==(int(overlap.argmax())+1)
    else: keep=mask
    return keep


def postprocess(cfg,method):
    global _WHOLE,_NUCLEI
    for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[key]="1"
    root=Path(cfg.output_root);pred=root/'predictions'/method;out=pred/'postprocess';out.mkdir(exist_ok=True)
    dep=dict(pre=identity(pred/'pre.tif'),nuclei=identity(Path(cfg.native().files.data_dir)/'nuclei.tif'),
             seed=cfg.seed,patch_size=cfg.postprocess_size,
             source=source_identity(['scripts/cos/postprocess.py','bidcell/model/postprocess_predictions.py']))
    _WHOLE=tifffile.imread(dep['pre']['path']);_NUCLEI=tifffile.imread(dep['nuclei']['path'])
    workers=min(4,json.loads(Path(cfg.queue_config).read_text())['cpu_workers'])
    size=min(cfg.postprocess_size,min(_WHOLE.shape));h,w=_WHOLE.shape
    hs=list(range(0,h-size,size))+[h-size];ws=list(range(0,w-size,size))+[w-size]
    coords=[(y,x) for y in hs for x in ws];tasks=[];started=time.monotonic()
    for y,x in coords:
        tasks.append((y,x,size,cfg.seed,str(out/f'tile_{y}_{x}.tif'),dict(base=dep,coords=[y,x]),cfg.mini_test))
    # Executor propagates exceptions; fork shares read-only full arrays.
    import multiprocessing as mp
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('fork')) as pool:
        receipts=[]
        for i,rec in enumerate(pool.map(tile_task,tasks)):
            receipts.append(identity(rec));atomic_json(out/'progress.json',dict(stage='tiles',completed=i+1,total=len(tasks)))
    seg=np.zeros(_WHOLE.shape,np.uint32);all_ids=[]
    for y,x in coords:
        tile=tifffile.imread(out/f'tile_{y}_{x}.tif');seg[y:y+size,x:x+size]=tile
        all_ids.extend(np.unique(tile)[np.unique(tile)>0].tolist())
    require_space(root, cfg.mini_test)
    atomic_tiff(pred/'tile_merge.tif',seg)
    windows={k for k,v in Counter(all_ids).items() if v>1}
    boundary=np.zeros(seg.shape,bool)
    # For the production shape this reproduces native negative numpy indexing at 0.
    hy=[y+i for i in range(-10,11) for y in hs if -h<=y+i<h]
    wx=[x+i for i in range(-10,11) for x in ws if -w<=x+i<w]
    boundary[hy,:]=True;boundary[:,wx]=True
    check=sorted(windows.intersection(np.unique(seg[boundary]).tolist()))
    atomic_json(out/'manifest.json',dict(dependencies=dep,tiles=receipts,coords=coords,split_ids=check,split_order='ascending canonical IDs; disjoint source labels'))
    boxes=ndi.find_objects(seg)
    splitdir=out/'splits';splitdir.mkdir(exist_ok=True)
    splitdeps=digest(dict(base=dep,tiles=receipts,ids=check))
    # Independent chunks, then rebuild from immutable merged tiles on every resume.
    for start in range(0,len(check),128):
        require_space(root,cfg.mini_test)
        ids=check[start:start+128];dest=splitdir/f'{start:06d}.npz';receipt=dest.with_suffix('.json')
        d=dict(base=splitdeps,ids=ids)
        if receipt.exists():
            saved=json.loads(receipt.read_text())
            if saved['dependencies']!=d: raise ValueError('split dependencies changed')
            verify_outputs(saved)
        else:
            arrays={}
            for cell_id in ids:
                box=boxes[cell_id-1]
                if box is None: raise ValueError('missing split ID')
                arrays[str(cell_id)]=split_patch(seg,_NUCLEI,cell_id,box)
            tmp=dest.with_suffix('.tmp')
            with tmp.open('wb') as f: np.savez_compressed(f,**arrays);f.flush();__import__('os').fsync(f.fileno())
            __import__('os').replace(tmp,dest)
            atomic_json(receipt,dict(status='succeeded',dependencies=d,outputs={'split':identity(dest)}))
        atomic_json(out/'progress.json',dict(stage='splits',completed=min(start+128,len(check)),total=len(check)))
    manifest_path=out/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    manifest['split_chunks']=[dict(start=start,ids=check[start:start+128],receipt=identity(splitdir/f'{start:06d}.json'),output=identity(splitdir/f'{start:06d}.npz')) for start in range(0,len(check),128)]
    atomic_json(manifest_path,manifest)
    # Apply only the manifest's exact chunks; no glob can collect stale files.
    for start in range(0,len(check),128):
        require_space(root,cfg.mini_test)
        with np.load(splitdir/f'{start:06d}.npz') as z:
            for cell_id in check[start:start+128]:
                box=boxes[cell_id-1];patch=seg[box];patch[patch==cell_id]=0;patch[z[str(cell_id)]]=cell_id
    require_space(root, cfg.mini_test)
    atomic_tiff(pred/'split_cleanup.tif',seg)
    require_space(root, cfg.mini_test)
    atomic_tiff(pred/'before_global_nuclear_restore.tif',seg)
    seg=np.where(_NUCLEI>0,_NUCLEI,seg).astype(np.uint32)
    require_space(root, cfg.mini_test)
    atomic_tiff(pred/'final.tif',seg)
    names=('raw_shift0','shift0','raw_shift24','shift24','pre','tile_merge','split_cleanup','before_global_nuclear_restore','final')
    atomic_json(pred/'endpoints.json',dict(version='COS-ownership-diagnostic-v1',endpoints={name:identity(pred/(name+'.tif')) for name in names},order=list(names),probes=dict(manifest=identity(root/'configs/probes.json'),files=[identity(p) for p in sorted((pred/'probes').glob('*.npz'))]),native_operation_order='postprocess_connect (cell loop close/open + nuclear restore), remove_islands, ordered tile merge, split cleanup, global nuclear restore'))

    atomic_json(out/'complete.json',dict(status='succeeded',dependencies=dep,outputs={'final_mask':identity(pred/'final.tif'),'endpoints':identity(pred/'endpoints.json'),
                'manifest':identity(out/'manifest.json'),**{name:identity(pred/(name+'.tif')) for name in ('tile_merge','split_cleanup','before_global_nuclear_restore')}},tiles=len(coords),split_ids=len(check),workers=workers,worker_blas_threads=1,wall_seconds=time.monotonic()-started))
