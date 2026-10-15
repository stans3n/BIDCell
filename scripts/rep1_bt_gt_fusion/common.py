"""Strict content identities, atomic science checkpoints, and the manual phase gate."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np
from . import SCHEMA


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, np.generic):
        return safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def digest(obj):
    return hashlib.sha256(json.dumps(safe(obj), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    with Path(path).open() as stream:
        return json.load(stream)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha256(path))


def check_identity(obj):
    p = Path(obj['path'])
    if not p.is_absolute() or not p.is_file():
        raise ValueError('identity requires an existing absolute file: ' + str(p))
    if ('bytes' in obj and p.stat().st_size != obj['bytes']) or sha256(p) != obj['sha256']:
        raise ValueError('content identity drift: ' + str(p))
    return p


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic(path, writer, binary=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb' if binary else 'w') as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        fsync_dir(path.parent)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_json(path, obj):
    atomic(path, lambda f: json.dump(safe(obj), f, indent=2, sort_keys=True, allow_nan=False), False)


def atomic_npy(path, array):
    atomic(path, lambda f: np.save(f, array, allow_pickle=False))


def atomic_npz(path, **arrays):
    atomic(path, lambda f: np.savez_compressed(f, **arrays))


def atomic_csv(path, frame, index=False):
    atomic(path, lambda f: frame.to_csv(f, index=index, na_rep='NA'), False)


def within(root, *parts):
    root = Path(root).resolve()
    path = root.joinpath(*parts).resolve()
    if path != root and root not in path.parents:
        raise ValueError('output escapes run root: ' + str(path))
    return path


def dependencies(cfg, stage):
    return dict(schema=SCHEMA, run_id=cfg['run_id'], stage=stage,
                config_digest=digest(cfg), source_digest=cfg['source_digest'],
                input_digest=cfg['input_digest'], plan_sha256=cfg['plan_sha256'])


def progress(callback, stage, cursor, total, **detail):
    if callback is not None:
        callback(dict(stage=stage, cursor=int(cursor), total=int(total), detail=safe(detail)))


def inventory(root):
    root = Path(root)
    return {str(p.relative_to(root)): identity(p) for p in sorted(root.rglob('*'))
            if p.is_file() and not any(x.startswith('.') for x in p.relative_to(root).parts)
            and p not in (root / 'complete.json', root / 'production_complete.json')
            and p.name != 'progress.json'}


def write_receipt(path, deps, outputs, **fields):
    obj = dict(schema=SCHEMA, status='succeeded', dependencies=deps, outputs=outputs, **fields)
    if Path(path).exists():
        old = read(path)
        if old != safe(obj):
            raise ValueError('refusing to replace completed receipt: ' + str(path))
    else:
        atomic_json(path, obj)
    return obj


def check_receipt(path, deps=None, output_root=None):
    obj = read(path)
    if obj.get('status') != 'succeeded' or not obj.get('outputs'):
        raise ValueError('incomplete receipt: ' + str(path))
    if deps is not None and obj.get('dependencies') != deps:
        raise ValueError('receipt dependency drift: ' + str(path))
    for item in obj['outputs'].values():
        if output_root is not None:
            within(output_root, item['path'])
        check_identity(item)
    return obj


def chunk_receipt(path, deps, outputs, **fields):
    return write_receipt(path, deps, {str(p): identity(p) for p in outputs}, **fields)


def resume_chunk(path, deps):
    return check_receipt(path, deps) if Path(path).exists() else None


def id_hash(ids):
    ids = np.asarray(ids, dtype='<i8')
    if len(np.unique(ids)) != len(ids):
        raise ValueError('duplicate population ID')
    return hashlib.sha256(np.sort(ids).tobytes()).hexdigest()


def require_manual_resume(cfg, grant_path):
    """A caller cannot unlock evaluation with a flag or an old run's grant."""
    expected = within(cfg['run_root'], 'control', 'manual_resume.json')
    if grant_path is None or Path(grant_path).resolve() != expected or not expected.is_file():
        raise PermissionError('new-endpoint evaluation/publication requires the run-specific manual_resume receipt')
    grant = read(expected)
    bind = dependencies(cfg, 'MANUAL_RESUME')
    if grant.get('dependencies') != bind or grant.get('authorization') != 'explicit_user_resume':
        raise PermissionError('manual resume run/hash/authorization mismatch')
    if not grant.get('user_instruction_utc') or not grant.get('production_complete'):
        raise PermissionError('manual resume lacks the user instruction time or production identity')
    production = within(cfg['run_root'], 'production_complete.json')
    if Path(grant['production_complete']['path']).resolve() != production:
        raise PermissionError('manual resume is bound to another production result')
    check_identity(grant['production_complete'])
    check_receipt(production, dependencies(cfg, 'PRODUCTION_VALIDATE'), cfg['run_root'])
    return grant
