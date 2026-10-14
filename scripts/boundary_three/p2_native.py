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
from scripts.crgf.common import atomic_tiff
from scripts.crgf.postprocess import split_patch
from .common import atomic_json, identity, check_identity, digest, read, progress, stop_requested
from .contracts import OUTPUT_ROOT
from .p2_rules import connect, FullNucleusIndex, RULE_VERSION, rng_hash

SCHEMA = 'boundary-p2-postprocess-v1'
_PRE = _NUC = _INDEX = None


def isolated(path):
    path = Path(path).resolve()
    if OUTPUT_ROOT not in path.parents:
        raise ValueError('P2 output outside isolated runroot: ' + str(path))
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
    atomic_json(path, rec)
    return rec


def atomic_npz(path, **arrays):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


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
            node=None, interrupt_after=None):
    """Resumable native/P2 stages. The caller owns the formal resource lock.

    interrupt_after is an admission-only deterministic fault injector. Progress
    and stop checks occur after each bounded group of at most workers tiles and
    each independent split chunk. Completed files without receipts are rebuilt.
    """
    global _PRE, _NUC, _INDEX
    pred = isolated(pred)
    pred.mkdir(parents=True, exist_ok=True)
    out = pred/'postprocess'
    out.mkdir(exist_ok=True)
    if rule not in ('B', 'P2') or (rule == 'P2' and native_pred is None):
        raise ValueError('P2 requires a completed native replay')
    workers = min(4, int(workers))
    if workers < 1:
        raise ValueError('worker bound')
    sources = [__file__, str(Path(__file__).with_name('p2_rules.py')), native.__file__]
    dep = dict(pre=identity(pre), nuclei=identity(nuclei), full_nuclei=identity(full_nuclei),
               source={Path(p).name: identity(p) for p in sources}, rule=rule, seed=seed,
               patch_size=patch_size, origin=list(origin), rule_version=RULE_VERSION)
    if cfg is not None:
        from .common import receipt_dependencies
        dep['run'] = receipt_dependencies(cfg, node)
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
    size, hs, ws, coords = tile_coordinates(_PRE.shape, patch_size)
    tasks = [(y, x, size, str(out/f'tile_{y}_{x}.tif'),
              dict(base=dep, coords=[y, x], shape=list(_PRE[y:y+size, x:x+size].shape)), native_pred)
             for y, x in coords]
    receipts = []

    def boundary(stage, completed, total):
        state = dict(stage=stage, completed=completed, total=total, rule=rule)
        atomic_json(out/'progress.json', state)
        if cfg is not None:
            progress(cfg, node, 'postprocess_' + rule + '_' + stage, **{k:v for k,v in state.items() if k!='stage'})
            if stop_requested(cfg):
                raise InterruptedError('P2 stopped at resumable ' + stage + ' boundary')
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
        tile = tifffile.imread(out/f'tile_{y}_{x}.tif')
        seg[y:y+size, x:x+size] = tile
        canonical[y:y+size, x:x+size] = tifffile.imread(out/f'tile_{y}_{x}_postconnect.tif')
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
    chunks = []
    for start in range(0, len(check), 128):
        ids = check[start:start+128]
        dest = splitdir/f'{start:06d}.npz'
        receipt = dest.with_suffix('.json')
        d = dict(base=splitdeps, ids=ids)
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
        chunks.append(dict(start=start, ids=ids, receipt=identity(receipt), output=identity(dest)))
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
        verify_record(obj['receipt']['path'])
    nuc = tifffile.imread(rec['dependencies']['nuclei']['path'])
    final = tifffile.imread(pred/'final.tif')
    if final.dtype != np.uint32 or final.shape != nuc.shape or not np.array_equal(final[nuc > 0], nuc[nuc > 0]):
        raise ValueError('postprocess final nuclear identity differs')
    return rec
