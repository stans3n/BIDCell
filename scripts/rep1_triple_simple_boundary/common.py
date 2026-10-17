"""Atomic scientific evidence and independent compute/publication authorization."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np
from . import SCHEMA

_LAST_WITNESS = None

def last_committed_witness():
    return _LAST_WITNESS

def committed(path):
    global _LAST_WITNESS
    _LAST_WITNESS = str(Path(path).resolve(strict=True))
    return _LAST_WITNESS


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


def writable_path(path):
    """A data reference never grants write access to its original producer."""
    p=Path(path).absolute()
    for name in ('r01','r02'):
        donor=Path('/home/zwan0726/work/outputs/bidcell_rep1_triple_simple_boundary_20261017/rep1-triple-simple-boundary-20261017-'+name)
        if p==donor or donor in p.parents:raise PermissionError('historical continuation producer is read-only')
    for parent in (p,*p.parents):
        if parent.is_symlink():raise PermissionError('scientific write through symlink forbidden')
        ref=parent/'.data-reference.json'
        if ref.is_file():
            rel=p.relative_to(parent);obj=read(ref)
            if str(rel) not in ('complete.json','initial_reuse.json','.data-reference.json') and not (obj.get('receipt_name')=='initial_reuse.json' and rel.parts[0]=='structure'):
                raise PermissionError('referenced partition leaf is read-only')
    return p


def atomic(path, writer, binary=True):
    path = writable_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb' if binary else 'w') as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        fsync_dir(path.parent)
    except BaseException:
        # Uncommitted bytes are evidence. Keep the unique same-disk temporary
        # for an audited cleanup; a later attempt uses another unique path.
        raise


def atomic_json(path, obj):
    atomic(path, lambda f: json.dump(safe(obj), f, indent=2, sort_keys=True, allow_nan=False), False)


def atomic_json_gzip(path,obj):
    import gzip
    data=json.dumps(safe(obj),sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    def writer(stream):
        with gzip.GzipFile(filename='',mode='wb',fileobj=stream,mtime=0) as compressed:compressed.write(data)
    atomic(path,writer)


def read_json_gzip(path):
    import gzip
    with gzip.open(path,'rt') as stream:return json.load(stream)


def atomic_npy(path, array):
    atomic(path, lambda f: np.save(f, array, allow_pickle=False))


def atomic_tiff(path, array):
    """Commit an exact uint32 mask through a named, same-directory TIFF.

    tifffile inspects a stream's .name; os.fdopen exposes an integer there.
    A real temporary path avoids that interface mismatch. Reopen the uncompressed
    TIFF as a read-only mapping and compare bounded row blocks before installing
    it, without creating an in-memory encoded TIFF or a second decoded mask.
    """
    import tifffile
    array = np.asarray(array)
    if array.ndim != 2 or array.dtype != np.dtype(np.uint32):
        raise ValueError('atomic TIFF requires a two-dimensional uint32 mask')
    path = writable_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tif', dir=path.parent)
    os.close(fd)
    try:
        tifffile.imwrite(tmp, array)
        with open(tmp, 'rb') as stream:
            os.fsync(stream.fileno())
        reopened = tifffile.memmap(tmp, mode='r')
        try:
            if reopened.shape != array.shape or reopened.dtype != array.dtype:
                raise ValueError('temporary TIFF shape or dtype differs from mask')
            for start in range(0, array.shape[0], 256):
                if not np.array_equal(reopened[start:start + 256], array[start:start + 256]):
                    raise ValueError('temporary TIFF pixels differ from mask')
        finally:
            reopened._mmap.close()
        os.replace(tmp, path)
        fsync_dir(path.parent)
    except BaseException:
        raise


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
                input_digest=cfg['input_digest'], domain_digest=cfg['domain_digest'],
                plan_digest=cfg['plan_digest'])


def progress(callback, stage, cursor, total, **detail):
    if callback is not None:
        if 'receipt' not in detail and _LAST_WITNESS is not None:
            detail['receipt'] = _LAST_WITNESS
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
        committed(path)
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


def reuse_partition(source, target, deps, binding, receipt_name='complete.json', *, cfg=None):
    """Validate an immutable scientific producer and install a later consumer.

    Hard links keep both named products without duplicating scientific data.
    All source bytes are verified before linking; preexisting target bytes must
    match. A new consumer receipt records the original producer and the complete
    stage-independent semantic binding (population, weights, coefficients, etc.).
    """
    source,target=Path(source),Path(target)
    old=check_receipt(source/'complete.json')
    invariant=('config_digest','source_digest','input_digest','domain_digest','plan_digest')
    if any(old['dependencies'].get(k)!=deps.get(k) for k in invariant) and cfg is not None and cfg.get('continuation'):
        from .continuation import reuse_partition as adopt_partition
        return adopt_partition(cfg,source,target,deps,binding,receipt_name)
    if any(old['dependencies'].get(k)!=deps.get(k) for k in invariant) or old.get('reuse_binding')!=safe(binding):
        raise ValueError('scientific producer partition is incompatible with consumer')
    target.mkdir(parents=True,exist_ok=True)
    for p in sorted(source.rglob('*')):
        if not p.is_file() or p==source/'complete.json' or any(x.startswith('.') for x in p.relative_to(source).parts):continue
        q=target/p.relative_to(source);q.parent.mkdir(parents=True,exist_ok=True)
        if q.exists():
            if sha256(q)!=sha256(p):raise ValueError('partial reuse target differs from committed producer')
        else:os.link(p,q)
    fields={k:v for k,v in old.items() if k not in ('schema','status','dependencies','outputs','reused_from','producer_dependencies')}
    fields.update(reused_from=identity(source/'complete.json'),producer_dependencies=old['dependencies'])
    return write_receipt(target/receipt_name,deps,inventory(target),**fields)


def partition_source(reuse_from,key):
    """Resolve a verified partition producer from INITIAL or endpoint evaluate."""
    if isinstance(reuse_from,dict):
        value=reuse_from.get(key)
        return Path(value) if value is not None else None
    return Path(reuse_from)/key if reuse_from is not None else None


def id_hash(ids):
    ids = np.asarray(ids, dtype='<i8')
    if len(np.unique(ids)) != len(ids):
        raise ValueError('duplicate population ID')
    return hashlib.sha256(np.sort(ids).tobytes()).hexdigest()


def require_manual_resume(cfg, grant_path):
    """Publication/result interpretation only. Compute evaluation has its own gate."""
    expected = within(cfg['run_root'], 'control', 'manual_resume.json')
    if grant_path is None or Path(grant_path).resolve() != expected or not expected.is_file():
        raise PermissionError('publication requires the run-specific manual_resume receipt')
    grant = read(expected)
    bind = dependencies(cfg, 'MANUAL_RESUME')
    if grant.get('dependencies') != bind or grant.get('authorization') != 'explicit_user_resume':
        raise PermissionError('manual resume run/hash/authorization mismatch')
    if not grant.get('user_instruction_utc') or not grant.get('user_instruction_evidence') or not grant.get('compute_complete'):
        raise PermissionError('manual resume lacks real user instruction or compute evidence')
    complete = within(cfg['run_root'], 'compute_complete.json')
    if Path(grant['compute_complete']['path']).resolve() != complete:
        raise PermissionError('manual resume is bound to another compute result')
    check_identity(grant['compute_complete'])
    check_receipt(complete, dependencies(cfg, '14'), cfg['run_root'])
    return grant


def require_compute_authorization(cfg, payload, stage, production=None):
    """Validate a runtime payload derived AFTER admission; never mutate config.

    Admission itself is independently validated by the controller's callback.
    Its identity, config digest and successful decision are checked again here.
    """
    from .contracts import normalize_stage, STAGE_ORDER, EVALUATION_STAGES, PRODUCTION_STAGES
    stage=normalize_stage(stage)
    declaration=cfg.get('compute_authorization',{})
    if declaration.get('mode')!='admitted_compute' or declaration.get('allowed_stages')!=list(STAGE_ORDER):
        raise PermissionError('immutable compute stage declaration differs')
    if not isinstance(payload,dict) or payload.get('authorization')!='admitted_compute':
        raise PermissionError('admitted compute payload required')
    if (payload.get('run_id'),payload.get('config_digest'),payload.get('stage'))!=(cfg['run_id'],digest(cfg),stage):
        raise PermissionError('compute payload run/config/stage mismatch')
    admission=payload.get('admission')
    if not isinstance(admission,dict):raise PermissionError('compute payload lacks admission identity')
    ap=check_identity(admission)
    within(cfg['run_root'],ap)
    audit=read(ap)
    if audit.get('config_digest')!=digest(cfg) or audit.get('status') not in ('passed','PASS','admitted','PASS_FOR_LAUNCH'):
        raise PermissionError('compute admission config/decision mismatch')
    if audit.get('open_blockers') not in ([],0):raise PermissionError('admission has unresolved blockers')
    if stage in EVALUATION_STAGES:
        if production is None:raise PermissionError('full evaluation authorization requires validated production')
        result=read(production);endpoint=EVALUATION_STAGES[stage]
        producer=next(s for s,n in PRODUCTION_STAGES.items() if n==endpoint)
        if result.get('status')!='succeeded' or result.get('endpoint')!=endpoint or result.get('stage')!=producer or result.get('dependencies',{}).get('config_digest')!=digest(cfg):
            raise PermissionError('full evaluation production run/stage/endpoint mismatch')
    expected=identity(production) if production is not None else None
    if payload.get('production')!=expected:raise PermissionError('compute production dependency mismatch')
    return payload
