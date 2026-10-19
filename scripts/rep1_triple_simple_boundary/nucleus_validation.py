"""Numerical validation of the actual full-nucleus P2 index, without caching it.

The independent raster scan uses reductions rather than scipy's bbox finder.
It proves complete coordinate membership and leaves the index and RNG alone.
This is input validation, not a completed P2 tile or endpoint.
"""
from pathlib import Path
import random
import numpy as np
import pandas as pd
import tifffile

from .common import (atomic_npz, check_identity, dependencies, identity, read,
                     within)

KIND = 'global_nucleus_index_validation'
EXPECTED_CELLS = 107955
FIELDS = {'ids', 'counts', 'bboxes', 'shape', 'maximum_squared_distance'}


def contract(cfg):
    from .continuation import frozen_root
    return dict(canonical_ids=identity(frozen_root(cfg)/'cells.csv'),
                expected_count=EXPECTED_CELLS)


def canonical_ids(spec):
    if spec.get('expected_count') != EXPECTED_CELLS:
        raise ValueError('global index population contract differs')
    ids = pd.read_csv(check_identity(spec['canonical_ids']), usecols=['cell_id'],
                      dtype={'cell_id': np.int64}).cell_id.to_numpy()
    ids = np.sort(ids)
    if len(ids) != EXPECTED_CELLS or not len(ids) or ids[0] <= 0 or np.any(np.diff(ids) <= 0):
        raise ValueError('global index canonical IDs missing or duplicated')
    return ids


def raster_payload(raw, ids):
    """Bounded independent count/min/max reduction over every raw nuclear pixel."""
    if raw.ndim != 2 or raw.dtype.kind not in 'ui':
        raise ValueError('global index raw raster contract')
    h, w = raw.shape
    maximum = (h-1)**2 + (w-1)**2
    if min(h, w) < 1 or maximum >= np.iinfo(np.int64).max:
        raise OverflowError('global nuclear squared distance exceeds int64')
    maxid = int(ids[-1]); counts = np.zeros(maxid+1, np.int64)
    boxes = np.empty((maxid+1, 4), np.int64)
    boxes[:] = (h, w, -1, -1)
    for start in range(0, h, 128):
        block = raw[start:start+128]
        yy, xx = np.nonzero(block)
        labels = block[yy, xx]
        if len(labels) and int(labels.max()) > maxid:
            raise ValueError('raw nuclear ID outside canonical population')
        labels = labels.astype(np.int64); yy += start
        counts += np.bincount(labels, minlength=maxid+1)
        np.minimum.at(boxes[:, 0], labels, yy)
        np.minimum.at(boxes[:, 1], labels, xx)
        np.maximum.at(boxes[:, 2], labels, yy+1)
        np.maximum.at(boxes[:, 3], labels, xx+1)
    if not np.array_equal(np.flatnonzero(counts), ids):
        raise ValueError('raw nuclear ID set differs from complete canonical population')
    return dict(ids=ids, counts=counts[ids], bboxes=boxes[ids],
                shape=np.array(raw.shape, np.int64),
                maximum_squared_distance=np.array([maximum], np.int64))


def actual_index_payload(index, ids):
    """Check the real live index, including every returned coordinate exactly."""
    before = random.getstate(); observed = index.max_observed_pairs
    payload = raster_payload(index.nuclei, ids)
    if len(index.boxes) != int(ids[-1]) or [i+1 for i,b in enumerate(index.boxes) if b is not None] != ids.tolist():
        raise ValueError('actual index ID coverage differs')
    h, w = index.nuclei.shape
    for cid, count, bounds in zip(ids, payload['counts'], payload['bboxes']):
        box = index.boxes[int(cid)-1]
        actual = np.array([box[0].start, box[1].start, box[0].stop, box[1].stop], np.int64)
        if not np.array_equal(actual, bounds):
            raise ValueError('actual nuclear index bbox differs from raw scan')
        coords = index.coordinates(cid)
        if coords.dtype != np.dtype('int64') or coords.shape != (int(count), 2):
            raise ValueError('actual nuclear coordinate count/dtype differs')
        y, x = coords.T
        if (y < 0).any() or (x < 0).any() or (y >= h).any() or (x >= w).any():
            raise ValueError('actual nuclear coordinate out of global bounds')
        linear = y*w+x
        if np.any(np.diff(linear) <= 0) or not np.all(index.nuclei[y, x] == cid):
            raise ValueError('actual nuclear coordinates incomplete, duplicated or mislabeled')
    if random.getstate() != before or index.max_observed_pairs != observed:
        raise ValueError('index validation changed RNG or query state')
    return payload


def validation_dependencies(dep, spec):
    if dep.get('rule') != 'P2' or dep.get('run', {}).get('stage') != '06':
        raise ValueError('global index validation requires actual P2 stage06')
    return dict(run=dep['run'], full_nuclei=dep['full_nuclei'], canonical_ids=spec,
                validation_source=identity(__file__),
                index_source=dep['source']['p2_rules.py'], part=KIND)


def commit(index, out, dep, spec, callback):
    from .p2 import record, verify_record, progress
    out = Path(out); dest = out/(KIND+'.npz'); receipt = dest.with_suffix('.json')
    deps = validation_dependencies(dep, spec)
    if receipt.exists():
        verify_record(receipt, deps)
        validate_payload(receipt)
        return receipt
    state = random.getstate()
    payload = actual_index_payload(index, canonical_ids(spec))
    if random.getstate() != state:raise ValueError('global index validation consumed RNG')
    atomic_npz(dest, **payload)
    record(receipt, deps, dict(numerical_payload=dest), kind=KIND,
           n_ids=len(payload['ids']), numerical_validation=True, rng_unchanged=True,
           no_distance_cache=True, recomputed=True, adopted=False)
    progress(callback, KIND, len(payload['ids']), len(payload['ids']), receipt=str(receipt))
    return receipt


def validate_payload(receipt):
    """Independent witness: reconstruct numerical truth from the entire raw input."""
    from .p2 import verify_record
    rec = verify_record(receipt); dep = rec['dependencies']
    if rec.get('kind') != KIND or rec.get('n_ids') != EXPECTED_CELLS or rec.get('numerical_validation') is not True or rec.get('rng_unchanged') is not True or rec.get('no_distance_cache') is not True or rec.get('adopted') is not False or rec.get('recomputed') is not True or rec.get('reused_from'):
        raise ValueError('global index numerical validation receipt differs')
    if set(rec['outputs']) != {'numerical_payload'}:raise ValueError('global index payload inventory differs')
    raw = tifffile.imread(check_identity(dep['full_nuclei']))
    expected = raster_payload(raw, canonical_ids(dep['canonical_ids']))
    with np.load(check_identity(rec['outputs']['numerical_payload']), allow_pickle=False) as z:
        if set(z.files) != FIELDS:raise ValueError('global index numerical fields differ')
        for name, a in expected.items():
            if z[name].dtype != np.dtype('int64') or not np.array_equal(z[name], a):
                raise ValueError('global index numerical payload differs: '+name)
    return rec


def witness(cfg, stage, receipt):
    from .inputs import load_inputs
    if stage != '06' or Path(receipt) != within(cfg['run_root'], 'endpoints/P2_SIMPLE/predictions/postprocess', KIND+'.json'):
        raise ValueError('global index witness stage/location differs')
    rec = read(receipt); dep = rec['dependencies']
    manifest = load_inputs(cfg)
    expected = dict(run=dependencies(cfg, '06'),
                    full_nuclei=identity(manifest['endpoints']['MORPH_SIMPLE_FUSION']['nuclei']),
                    canonical_ids=contract(cfg), validation_source=identity(__file__),
                    index_source=identity(Path(__file__).parents[1]/'boundary_three/p2_rules.py'), part=KIND)
    if dep != expected:raise ValueError('global index witness input/source/producer differs')
    validate_payload(receipt)
