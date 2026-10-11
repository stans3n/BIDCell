"""Native morphology, deterministic tile RNG, explicit tile/split receipts.
Split selection is evaluated in the full ID bounding box, exactly equivalent to
full-field 8-connected labelling (scan order and overlap tie rule preserved).
"""
import json
import os
import random
import pickle,hashlib
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import tifffile
from scipy import ndimage as ndi
from bidcell.model.postprocess_predictions import remove_islands
from .post_rules import connect,FullNucleusIndex,RULE_VERSION,rng_hash
from scripts.crgf.common import atomic_json,atomic_tiff,identity,sha256,source_identity,verify_outputs,digest

from .resources import require_space

_WHOLE=None
_NUCLEI=None
_INDEX=None

def validate_tile(receipt,dep):
    saved=json.loads(Path(receipt).read_text())
    if saved.get('schema')!='BS4-tile-v2' or saved.get('status')!='succeeded' or saved.get('dependencies')!=dep:raise ValueError('tile schema/dependency mismatch')
    if saved.get('exit_code')!=0 or saved.get('dtype')!='uint32' or saved.get('coords')!=dep['coords']:raise ValueError('tile identity/exit mismatch')
    expected_seed=dep['base']['seed']+dep['coords'][0]*1000003+dep['coords'][1]
    if any(saved.get(k)!=v for k,v in dict(arm=dep['base']['arm'],run=dep['base']['run'],shape=dep['shape'],tile_seed=expected_seed,stage_names=['postconnect','postislands']).items()):raise ValueError('tile metadata mismatch')
    body={k:v for k,v in saved.items() if k!='integrity'}
    if saved.get('integrity')!=digest(body):raise ValueError('tile receipt integrity mismatch')
    diagnostics=saved.get('diagnostics',{})
    if diagnostics.get('rule')!=dep['base']['rule'] or diagnostics.get('version')!=RULE_VERSION:raise ValueError('tile rule diagnostic mismatch')
    required_rng=('rng_before_shuffle','rng_enter_islands','rng_after_islands')
    if any(not isinstance(diagnostics.get(k),str) or len(diagnostics[k])!=64 for k in required_rng):raise ValueError('tile RNG schema')
    rng=random.Random(expected_seed)
    if diagnostics['rng_before_shuffle']!=hashlib.sha256(pickle.dumps(rng.getstate())).hexdigest():raise ValueError('tile RNG seed mismatch')
    pre=tifffile.memmap(dep['base']['pre']['path'])
    y,x=dep['coords'];h,w=dep['shape'];ids=np.unique(pre[y:y+h,x:x+w]);ids=ids[ids>0].tolist()
    if sorted(diagnostics['candidate_order'])!=ids:raise ValueError('candidate IDs differ from immutable pre')
    ordered=list(ids)
    rng.shuffle(ordered)
    if ordered!=diagnostics['candidate_order'] or diagnostics['rng_enter_islands']!=hashlib.sha256(pickle.dumps(rng.getstate())).hexdigest():raise ValueError('tile RNG shuffle mismatch')
    if set(diagnostics['candidate_hashes'])!={str(i) for i in ids} or any(len(h)!=64 for h in diagnostics['candidate_hashes'].values()):raise ValueError('candidate identity schema')
    verify_outputs(saved)
    for name in ('postconnect','postislands'):
        a=tifffile.imread(saved['outputs'][name]['path'])
        if list(a.shape)!=dep['shape'] or a.dtype!=np.uint32:raise ValueError('tile shape/dtype mismatch')
        if name=='postconnect':
            island_ids=np.unique(a);island_ids=island_ids[island_ids>0];rng.shuffle(island_ids)
            if diagnostics['rng_after_islands']!=hashlib.sha256(pickle.dumps(rng.getstate())).hexdigest():raise ValueError('islands RNG mismatch')
    return saved

def tile_task(task):
    y,x,size,seed,out,dep,mini=task;out=Path(out);receipt=out.with_suffix('.json')
    if receipt.exists():validate_tile(receipt,dep);return str(receipt)
    random.seed(seed+y*1000003+x)
    img=_WHOLE[y:y+size,x:x+size];nuc=_NUCLEI[y:y+size,x:x+size]
    origin=dep['base'].get('origin',[0,0]);rule=dep['base']['rule']
    connected,diagnostics,arrays,_=connect(img,nuc,rule,_INDEX,(y+origin[0],x+origin[1]))
    if rng_hash()!=diagnostics['rng_enter_islands']:raise AssertionError('islands RNG entry changed')
    result=remove_islands(connected,nuc)
    diagnostics['rng_after_islands']=rng_hash()
    diagnostics['connect_to_islands_owner_changes']=int((connected!=result).sum())
    diagnostics['intervention_owner_unchanged_through_islands_exposure']=int(((connected!=arrays['baseline_postconnect'])&(result==connected)).sum())
    # Independently propagate the unchanged B output from exactly the same entering RNG.
    preserved=random.getstate();random.seed(seed+y*1000003+x)
    ids=np.unique(img);ids=ids[ids>0];random.shuffle(ids)
    baseline_islands=remove_islands(arrays['baseline_postconnect'],nuc);random.setstate(preserved)
    local_delta=connected!=arrays['baseline_postconnect'];later_delta=result!=baseline_islands
    diagnostics['local_difference_retained_after_islands']=int((local_delta&later_delta).sum())
    diagnostics['local_difference_cancelled_after_islands']=int((local_delta&~later_delta).sum())
    diagnostics['new_difference_after_islands']=int((~local_delta&later_delta).sum())
    diagnostics['exact_owner_swap_after_islands']=int((local_delta&later_delta&(result==arrays['baseline_postconnect'])&(baseline_islands==connected)).sum())
    arrays['baseline_postislands']=baseline_islands
    require_space(Path(dep['base']['config']['output_root']) if isinstance(dep['base'].get('config'),dict) else out.parent,mini,reserve_bytes=result.nbytes*3)
    postconnect=out.with_name(out.stem+'_postconnect.tif');atomic_tiff(postconnect,connected)
    atomic_tiff(out,result.astype(np.uint32))
    details=out.with_name(out.stem+'_diagnostics.npz');tmp=details.with_suffix('.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    os.replace(tmp,details)
    record=dict(schema='BS4-tile-v2',status='succeeded',exit_code=0,arm=dep['base']['arm'],run=dep['base']['run'],coords=[y,x],shape=list(result.shape),dtype='uint32',tile_seed=seed+y*1000003+x,stage_names=['postconnect','postislands'],dependencies=dep,diagnostics=diagnostics,outputs={'postconnect':identity(postconnect),'postislands':identity(out),'diagnostics':identity(details)})
    record['integrity']=digest(record);atomic_json(receipt,record)
    validate_tile(receipt,dep)
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
    global _WHOLE,_NUCLEI,_INDEX
    for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[key]="1"
    root=Path(cfg.output_root);pred=root/'predictions'/method;out=pred/'postprocess';out.mkdir(parents=True,exist_ok=True)
    dep=dict(pre=identity(pred/'pre.tif'),nuclei=identity(Path(cfg.native().files.data_dir)/'nuclei.tif'),
             seed=cfg.seed,patch_size=cfg.postprocess_size,
             source=source_identity(['scripts/bs4/postprocess.py','scripts/bs4/post_rules.py','bidcell/model/postprocess_predictions.py']),arm=method,run=cfg.run_id,rule=cfg.post_rule,rule_version=RULE_VERSION,config=cfg.model_dump(mode='json'),origin=list(cfg.mini_origin or (0,0)))
    _WHOLE=tifffile.imread(dep['pre']['path']);_NUCLEI=tifffile.imread(dep['nuclei']['path'])
    full=tifffile.imread(cfg.mini_full_nuclei) if cfg.mini_full_nuclei else _NUCLEI
    _INDEX=FullNucleusIndex(full) if cfg.post_rule=='P2' else None
    if cfg.mini_full_nuclei:dep['complete_nuclei']=identity(cfg.mini_full_nuclei)
    workers=min(4,json.loads(Path(cfg.queue_config).read_text())['cpu_workers'])
    size=min(cfg.postprocess_size,min(_WHOLE.shape));h,w=_WHOLE.shape
    hs=list(range(0,h-size,size))+[h-size];ws=list(range(0,w-size,size))+[w-size]
    coords=[(y,x) for y in hs for x in ws];tasks=[];started=time.monotonic()
    for y,x in coords:
        tasks.append((y,x,size,cfg.seed,str(out/f'tile_{y}_{x}.tif'),dict(base=dep,coords=[y,x],shape=list(_WHOLE[y:y+size,x:x+size].shape),dtype='uint32'),cfg.mini_test))
    # Executor propagates exceptions; fork shares read-only full arrays.
    import multiprocessing as mp
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('fork')) as pool:
        receipts=[]
        for i,rec in enumerate(pool.map(tile_task,tasks)):
            receipts.append(identity(rec));atomic_json(out/'progress.json',dict(stage='tiles',completed=i+1,total=len(tasks)))
    seg=np.zeros(_WHOLE.shape,np.uint32);canonical=np.zeros_like(seg);all_ids=[]
    for y,x in coords:
        tile=tifffile.imread(out/f'tile_{y}_{x}.tif');seg[y:y+size,x:x+size]=tile
        canonical[y:y+size,x:x+size]=tifffile.imread(out/f'tile_{y}_{x}_postconnect.tif')
        all_ids.extend(np.unique(tile)[np.unique(tile)>0].tolist())
    require_space(root, cfg.mini_test)
    atomic_tiff(pred/'tile_merge.tif',seg)
    atomic_tiff(pred/'postconnect_canonical.tif',canonical)
    del canonical
    # This canonical is diagnostic only; algorithm continues on original islands merge.
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
    names=('raw_shift0','shift0','raw_shift24','shift24','pre','postconnect_canonical','tile_merge','split_cleanup','before_global_nuclear_restore','final')
    atomic_json(pred/'endpoints.json',dict(version='COS-ownership-diagnostic-v1',endpoints={name:identity(pred/(name+'.tif')) for name in names},order=list(names),probes=dict(manifest=identity(root/'configs/probes.json'),files=[identity(p) for p in sorted((pred/'probes').glob('*.npz'))]),native_operation_order='postprocess_connect (cell loop close/open + nuclear restore), remove_islands, ordered tile merge, split cleanup, global nuclear restore'))

    atomic_json(out/'complete.json',dict(schema='BS4-postprocess-v2',arm=method,run=cfg.run_id,status='succeeded',exit_code=0,dependencies=dep,outputs={'final_mask':identity(pred/'final.tif'),'endpoints':identity(pred/'endpoints.json'),
                'manifest':identity(out/'manifest.json'),**{name:identity(pred/(name+'.tif')) for name in ('postconnect_canonical','tile_merge','split_cleanup','before_global_nuclear_restore')}},tiles=len(coords),split_ids=len(check),workers=workers,worker_blas_threads=1,wall_seconds=time.monotonic()-started))
