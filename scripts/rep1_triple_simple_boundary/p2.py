"""Native postprocess adapter with observational candidate capture and recovery.

The disabled arm calls the current native function itself. A NumPy proxy records
its final per-cell writes without changing inputs, candidates, outputs or RNG.
P2 verifies those candidate hashes and entering-islands RNG against that replay.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import hashlib
import multiprocessing as mp
import os
import random
import time

import numpy as np
import tifffile
from scipy import ndimage as ndi
from bidcell.model import postprocess_predictions as native
from .common import atomic_tiff,atomic_npz
from scripts.crgf.postprocess import split_patch
from .common import atomic_json, identity, check_identity, digest, read, progress, within, committed
from scripts.boundary_three.p2_rules import connect, FullNucleusIndex, RULE_VERSION, rng_hash

SCHEMA = 'rep1-triple-simple-p2-postprocess-v1'
_PRE = _NUC = _INDEX = None


def isolated(path,allowed_root):
    path=within(allowed_root,path)
    if path==Path(allowed_root).resolve():raise ValueError('postprocess needs a dedicated output directory')
    return path


class NativeWrites:
    """Observe only native final=np.where(filled_mask>0, scalar_ID, final)."""
    def __init__(self, shape):
        self.count = np.zeros(shape, np.uint32)
        self.hashes = {}
        self.order = []

    def __getattr__(self, name):
        return getattr(np, name)

    def where(self, condition, x=None, y=None):
        if np.isscalar(x) and isinstance(y, np.ndarray):
            mask = np.asarray(condition, dtype=bool)
            if mask.shape != self.count.shape or int(x) <= 0:
                raise ValueError('unexpected native final write')
            self.order.append(int(x))
            self.hashes[str(int(x))] = hashlib.sha256(mask.tobytes()).hexdigest()
            self.count += mask
        return np.where(condition, x, y)


def native_connect(img, nuclei):
    capture = NativeWrites(img.shape)
    original = native.np
    before = rng_hash()
    try:
        native.np = capture
        result = native.postprocess_connect(img, nuclei)
    finally:
        native.np = original
    diag = dict(rule='B', version=RULE_VERSION, candidate_order=capture.order,
                candidate_hashes=capture.hashes, rng_before_shuffle=before,
                rng_enter_islands=rng_hash(), conflict_pixels=int((capture.count >= 2).sum()),
                proposal_histogram={str(int(k)): int(v) for k, v in zip(*np.unique(capture.count, return_counts=True))})
    return result, diag, dict(proposal_count=capture.count)


def verify_record(path, dependencies=None):
    rec = read(path)
    if rec.get('status') != 'succeeded' or rec.get('schema') != SCHEMA:
        raise ValueError('invalid P2 receipt: ' + str(path))
    if dependencies is not None and rec['dependencies'] != dependencies:
        raise ValueError('P2 resume dependency drift: ' + str(path))
    if rec.get('integrity') != digest({k: v for k, v in rec.items() if k != 'integrity'}):
        raise ValueError('P2 receipt integrity mismatch: ' + str(path))
    for obj in rec['outputs'].values():
        check_identity(obj)
    return rec


def record(path, dependencies, outputs, **details):
    rec = dict(schema=SCHEMA, status='succeeded', dependencies=dependencies,
               outputs={k: identity(v) for k, v in outputs.items()}, **details)
    rec['integrity'] = digest(rec)
    if Path(path).exists():
        previous=verify_record(path,dependencies)
        if previous!=rec:raise ValueError('refusing to replace committed P2 receipt')
    else:
        atomic_json(path, rec)
        committed(path)
    return rec


def tile_task(task):
    y, x, size, dest, dep, native_pred = task
    dest = Path(dest)
    receipt = dest.with_suffix('.json')
    if receipt.exists():
        verify_record(receipt, dep)
        return str(receipt)
    random.seed(dep['base']['seed'] + y * 1000003 + x)
    img, nuc = _PRE[y:y+size, x:x+size], _NUC[y:y+size, x:x+size]
    if dep['base']['rule'] == 'B':
        connected, diagnostics, arrays = native_connect(img, nuc)
    else:
        origin = dep['base']['origin']
        connected, diagnostics, arrays, _ = connect(img, nuc, 'P2', _INDEX,
                                                   (y + origin[0], x + origin[1]))
        previous = verify_record(Path(native_pred)/'postprocess'/f'tile_{y}_{x}.json')
        for key in ('candidate_order', 'candidate_hashes', 'rng_before_shuffle', 'rng_enter_islands', 'proposal_histogram'):
            if diagnostics[key] != previous['diagnostics'][key]:
                raise ValueError('native/P2 candidate or RNG mismatch: ' + key)
        baseline = tifffile.imread(previous['outputs']['postconnect']['path'])
        if not np.array_equal(arrays['baseline_postconnect'], baseline):
            raise ValueError('native/P2 counterfactual differs')
        # These large baseline arrays are already persisted by native replay.
        del arrays['baseline_postconnect'], arrays['pre_nuclear_restore'], arrays['veto_exposures']
    if rng_hash() != diagnostics['rng_enter_islands']:
        raise ValueError('diagnostics changed entering-islands RNG')
    result = native.remove_islands(connected, nuc)
    diagnostics['rng_after_islands'] = rng_hash()
    diagnostics['connect_to_islands_owner_changes'] = int((connected != result).sum())
    diagnostics['candidate_count'] = len(diagnostics['candidate_order'])
    postconnect = dest.with_name(dest.stem + '_postconnect.tif')
    details = dest.with_name(dest.stem + '_diagnostics.npz')
    atomic_tiff(postconnect, connected.astype(np.uint32))
    atomic_tiff(dest, result.astype(np.uint32))
    atomic_npz(details, **arrays)
    record(receipt, dep, dict(postconnect=postconnect, postislands=dest, diagnostics=details),
           coords=[y, x], shape=list(result.shape), diagnostics=diagnostics)
    return str(receipt)


def tile_coordinates(shape, patch_size):
    h, w = shape
    size = min(patch_size, h, w)
    hs = list(range(0, h-size, size)) + [h-size]
    ws = list(range(0, w-size, size)) + [w-size]
    return size, hs, ws, [(y, x) for y in hs for x in ws]


def process(pre, nuclei, full_nuclei, pred, rule, *, native_pred=None,
            workers=4, seed=1234, patch_size=1024, origin=(0, 0), cfg=None,
            node=None, interrupt_after=None, allowed_root=None, dependency_context=None, callback=None, lock_fds=(), continuation_cfg=None, index_validation=None):
    """Resumable native/P2 stages. The caller owns the formal resource lock.

    interrupt_after is an admission-only deterministic fault injector. Progress
    and stop checks occur after each bounded group of at most workers tiles and
    each independent split chunk. Completed files without receipts are rebuilt.
    """
    global _PRE, _NUC, _INDEX
    if allowed_root is None or cfg is not None:raise ValueError('explicit new allowed_root/context required; legacy cfg forbidden')
    for fd in lock_fds:os.fstat(fd)
    pred = isolated(pred,allowed_root)
    pred.mkdir(parents=True, exist_ok=True)
    out = pred/'postprocess'
    out.mkdir(exist_ok=True)
    if rule not in ('B', 'P2') or (rule == 'P2' and native_pred is None):
        raise ValueError('P2 requires a completed native replay')
    workers = min(4, int(workers))
    if workers < 1:
        raise ValueError('worker bound')
    sources = [__file__, str(Path(__file__).parents[1]/'boundary_three/p2_rules.py'), native.__file__]
    dep = dict(pre=identity(pre), nuclei=identity(nuclei), full_nuclei=identity(full_nuclei),
               source={Path(p).name: identity(p) for p in sources}, rule=rule, seed=seed,
               patch_size=patch_size, origin=list(origin), rule_version=RULE_VERSION)
    if dependency_context is not None:dep['run']=dependency_context
    if rule == 'P2':
        native_receipt = Path(native_pred)/'postprocess/complete.json'
        verify_record(native_receipt)
        dep['native_replay'] = identity(native_receipt)
    if (out/'complete.json').exists():
        return verify_record(out/'complete.json', dep)
    started = time.monotonic()
    _PRE, _NUC = tifffile.imread(pre), tifffile.imread(nuclei)
    if _PRE.shape != _NUC.shape or _PRE.ndim != 2:
        raise ValueError('pre/nucleus shape differs')
    full = tifffile.imread(full_nuclei) if str(full_nuclei) != str(nuclei) else _NUC
    oy, ox = origin
    if not np.array_equal(_NUC, full[oy:oy+_NUC.shape[0], ox:ox+_NUC.shape[1]]):
        raise ValueError('cropped nuclei differ from complete global nuclear raster')
    _INDEX = FullNucleusIndex(full) if rule == 'P2' else None
    if index_validation is not None:
        from .nucleus_validation import commit
        commit(_INDEX, out, dep, index_validation, callback)
    size, hs, ws, coords = tile_coordinates(_PRE.shape, patch_size)
    tasks = [(y, x, size, str(out/f'tile_{y}_{x}.tif'),
              dict(base=dep, coords=[y, x], shape=list(_PRE[y:y+size, x:x+size].shape)), native_pred)
             for y, x in coords]
    if continuation_cfg is not None:
        from .continuation import adopt_tiles
        adopt_tiles(continuation_cfg, dep, out, tasks)
    receipts = []

    def boundary(stage, completed, total):
        state = dict(stage=stage, completed=completed, total=total, rule=rule)
        atomic_json(out/'progress.json', state)
        witness=(out/f'tile_{coords[completed-1][0]}_{coords[completed-1][1]}.json') if stage=='tiles' else (out/'splits'/f'{((completed-1)//128)*128:06d}.json')
        progress(callback,'postprocess_'+rule+'_'+stage,completed,total,receipt=str(witness))
        if interrupt_after is not None and stage == interrupt_after[0] and completed >= interrupt_after[1]:
            raise InterruptedError('admission interruption after ' + stage)

    # Bounded submissions avoid a stop request leaving a full-slide work queue.
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork')) as pool:
        for start in range(0, len(tasks), workers):
            batch = list(pool.map(tile_task, tasks[start:start+workers]))
            receipts.extend(identity(p) for p in batch)
            boundary('tiles', len(receipts), len(tasks))
    seg = np.zeros(_PRE.shape, np.uint32)
    canonical = np.zeros_like(seg)
    all_ids = []
    for y, x in coords:
        tile_record = verify_record(out/f'tile_{y}_{x}.json')
        tile = tifffile.imread(tile_record['outputs']['postislands']['path'])
        seg[y:y+size, x:x+size] = tile
        canonical[y:y+size, x:x+size] = tifffile.imread(tile_record['outputs']['postconnect']['path'])
        all_ids.extend(np.unique(tile)[np.unique(tile)>0].tolist())
    atomic_tiff(pred/'postislands_ordered_tile_merge.tif', seg)
    atomic_tiff(pred/'postconnect_canonical.tif', canonical)
    del canonical
    windows = {k for k, v in Counter(all_ids).items() if v > 1}
    boundary_mask = np.zeros(seg.shape, bool)
    h, w = seg.shape
    hy = [y+i for i in range(-10, 11) for y in hs if -h <= y+i < h]
    wx = [x+i for i in range(-10, 11) for x in ws if -w <= x+i < w]
    boundary_mask[hy, :] = True
    boundary_mask[:, wx] = True
    check = sorted(windows.intersection(np.unique(seg[boundary_mask]).tolist()))
    del boundary_mask
    boxes = ndi.find_objects(seg)
    splitdir = out/'splits'
    splitdir.mkdir(exist_ok=True)
    splitdeps = digest(dict(base=dep, tiles=receipts, ids=check))
    if continuation_cfg is not None:
        from .continuation import adopt_split
        adopt_split(continuation_cfg, dep, splitdir, splitdeps, check)
    chunks = []
    for start in range(0, len(check), 128):
        ids = check[start:start+128]
        dest = splitdir/f'{start:06d}.npz'
        receipt = dest.with_suffix('.json')
        d = dict(base=splitdeps, ids=ids)
        # The opaque content digest remains exact. Formal progress additionally
        # needs the real caller's visible current config/run identity envelope.
        if dependency_context is not None:d['run'] = dependency_context
        if receipt.exists():
            verify_record(receipt, d)
        else:
            arrays = {}
            for cid in ids:
                box = boxes[cid-1]
                if box is None:
                    raise ValueError('missing split ID')
                arrays[str(cid)] = split_patch(seg, _NUC, cid, box)
            atomic_npz(dest, **arrays)
            record(receipt, d, dict(split=dest))
        chunks.append(dict(start=start, ids=ids, receipt=identity(receipt), output=verify_record(receipt,d)['outputs']['split']))
        boundary('splits', min(start+128, len(check)), len(check))
    # Resume always rebuilds from immutable tiles, then applies exact chunks.
    for chunk in chunks:
        with np.load(chunk['output']['path']) as saved:
            for cid in chunk['ids']:
                patch = seg[boxes[cid-1]]
                patch[patch == cid] = 0
                patch[saved[str(cid)]] = cid
    atomic_tiff(pred/'split_cleanup.tif', seg)
    atomic_tiff(pred/'before_global_nuclear_restore.tif', seg)
    seg = np.where(_NUC > 0, _NUC, seg).astype(np.uint32)
    atomic_tiff(pred/'final.tif', seg)
    atomic_json(out/'manifest.json', dict(dependencies=dep, tiles=receipts, coords=coords,
                 size=size, split_ids=check, split_chunks=chunks,
                 split_order='ascending canonical IDs; immutable merged source labels'))
    paths = {name: pred/(name+'.tif') for name in STAGES}
    paths['manifest'] = out/'manifest.json'
    result = record(out/'complete.json', dep, paths, tiles=len(coords), split_ids=len(check),
                    workers=workers, worker_blas_threads=1, wall_seconds=time.monotonic()-started)
    _PRE = _NUC = _INDEX = None
    return result


STAGES = ('postconnect_canonical', 'postislands_ordered_tile_merge',
          'split_cleanup', 'before_global_nuclear_restore', 'final')


def validate_process(pred):
    pred = Path(pred)
    rec = verify_record(pred/'postprocess/complete.json')
    manifest = read(pred/'postprocess/manifest.json')
    if manifest['dependencies'] != rec['dependencies'] or len(manifest['tiles']) != rec['tiles']:
        raise ValueError('postprocess manifest lineage differs')
    for obj in manifest['tiles']:
        check_identity(obj)
        tile = verify_record(obj['path'])
        if tile['dependencies']['base'] != rec['dependencies']:
            raise ValueError('tile upstream lineage differs')
    for obj in manifest['split_chunks']:
        check_identity(obj['receipt'])
        check_identity(obj['output'])
        expected=dict(base=digest(dict(base=rec['dependencies'],tiles=manifest['tiles'],ids=manifest['split_ids'])),ids=obj['ids'])
        if 'run' in rec['dependencies']:expected['run']=rec['dependencies']['run']
        verify_record(obj['receipt']['path'],expected)
    nuc = tifffile.imread(rec['dependencies']['nuclei']['path'])
    final = tifffile.imread(pred/'final.tif')
    if final.dtype != np.uint32 or final.shape != nuc.shape or not np.array_equal(final[nuc > 0], nuc[nuc > 0]):
        raise ValueError('postprocess final nuclear identity differs')
    return rec


def assert_matrix_equal(actual,expected):
    from .inputs import matrix_arrays
    ai,ag,ac=matrix_arrays({'matrix':str(actual)});bi,bg,bc=matrix_arrays({'matrix':str(expected)})
    if not np.array_equal(ai,bi) or ag!=bg or not np.array_equal(ac,bc):
        raise ValueError('disabled Simple replay full integer matrix/ID/gene mismatch')
    a,b=read(Path(actual)/'complete.json'),read(Path(expected)/'complete.json')
    if a['totals']!=b['totals'] or len(a['receipts'])!=len(b['receipts']):raise ValueError('replay totals/coverage mismatch')
    for x,y in zip(a['receipts'],b['receipts']):
        if (x['row_start'],x['row_stop'],x['coordinate_sha256'],x['sha256'])!=(y['row_start'],y['row_stop'],y['coordinate_sha256'],y['sha256']):
            raise ValueError('replay all-transcript owner/association mismatch')
    import pandas as pd
    af=pd.read_csv(Path(actual)/'expr_mat.csv',index_col=0);bf=pd.read_csv(Path(expected)/'expr_mat.csv',index_col=0)
    if not af.equals(bf):raise ValueError('replay full native CSV differs')
    return dict(integer_counts_equal=True,cell_gene_order_equal=True,transcript_owners_equal=True,csv_equal=True,totals=a['totals'])


def run_replay(cfg,manifest,callback=None,lock_fds=()):
    from .common import dependencies,write_receipt,check_receipt,inventory
    from .inputs import matrix_evidence
    from .matrix import build
    parent=manifest['endpoints']['MORPH_SIMPLE_FUSION'];matrix_evidence(parent)
    out=within(cfg['run_root'],'replay/SIMPLE_NATIVE');dep=dependencies(cfg,'05')
    if (out/'complete.json').exists():return check_receipt(out/'complete.json',dep,cfg['run_root'])
    machine=dict(read(parent['machine']));pred=out/'predictions';mat=out/'matrix'
    # pre is the authentic pre image, read-only; never the historical final.
    if Path(machine['pre_mask']).resolve()!=Path(parent['pre']).resolve():raise ValueError('Simple native pre identity differs')
    historic=read(Path(parent['pre']).parent/'postprocess/manifest.json')['dependencies']
    if (historic['seed'],historic['patch_size'])!=(1234,1024):raise ValueError('historical Simple native seed/tile contract differs')
    for key,path in [('pre',parent['pre']),('nuclei',parent['nuclei'])]:
        check_identity(historic[key])
        if historic[key]['sha256']!=identity(path)['sha256']:raise ValueError('Simple native producer input differs')
    machine.update(final_mask=str(pred/'final.tif'),final_matrix=str(mat/'expr_mat.csv'),
                   parent_endpoint='MORPH_SIMPLE_FUSION',production_kind='native_disabled_replay')
    atomic_json(out/'machine.json',machine)
    process(parent['pre'],parent['nuclei'],parent['nuclei'],pred,'B',workers=cfg['resources']['p2_workers'],
            allowed_root=cfg['run_root'],dependency_context=dep,callback=callback,lock_fds=lock_fds,
            continuation_cfg=cfg if cfg.get('continuation') else None)
    validate_process(pred)
    a=tifffile.imread(pred/'final.tif');b=tifffile.imread(parent['final'])
    if a.dtype!=b.dtype or a.shape!=b.shape or not np.array_equal(a,b):raise ValueError('FULL Simple native replay final differs')
    build(machine,manifest['coordinates'],mat,dict(dep,part='replay_matrix'),callback)
    equality=assert_matrix_equal(mat,parent['matrix'])
    return write_receipt(out/'complete.json',dep,inventory(out),base='MORPH_SIMPLE_FUSION',scope='full_field',
                         final_pixels_equal=True,matrix=equality,source_pre=identity(parent['pre']))


def produce(cfg,manifest,callback=None,lock_fds=()):
    from .common import dependencies,write_receipt,check_receipt,atomic
    import shutil
    parent=manifest['endpoints']['MORPH_SIMPLE_FUSION'];out=within(cfg['run_root'],'endpoints/P2_SIMPLE')
    from .continuation import native_replay_root
    from .nucleus_validation import contract
    replay=native_replay_root(cfg)
    pred=out/'predictions';dep=dependencies(cfg,'06')
    result=process(parent['pre'],parent['nuclei'],parent['nuclei'],pred,'P2',native_pred=replay/'predictions',
                   workers=cfg['resources']['p2_workers'],allowed_root=cfg['run_root'],dependency_context=dep,
                   callback=callback,lock_fds=lock_fds,
                   continuation_cfg=cfg if cfg.get('continuation') else None,index_validation=contract(cfg))
    validate_process(pred)
    dest=out/'final.tif'
    if dest.exists():
        if identity(dest)['sha256']!=identity(pred/'final.tif')['sha256']:raise ValueError('existing P2 final differs')
    else:
        def copy(stream):
            with (pred/'final.tif').open('rb') as src:shutil.copyfileobj(src,stream,8*1024*1024)
        atomic(dest,copy)
    return result
