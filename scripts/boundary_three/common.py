"""Atomic receipts, output isolation, source identities and shared compute lock."""
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import time
from datetime import datetime, timezone
from .contracts import *

def utc():
    return datetime.now(timezone.utc).isoformat()

def read(path):
    return json.loads(Path(path).read_text())

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def identity(path):
    path = Path(path).resolve(); st = path.stat()
    return dict(path=str(path),bytes=st.st_size,sha256=sha256(path))

def check_identity(obj):
    if identity(obj['path']) != obj:
        raise ValueError('frozen identity drift: ' + obj['path'])

def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temp.open('w') as f:
        json.dump(value,f,indent=2,allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temp,path)
    fd = os.open(path.parent,os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)

def source_manifest(repo=None):
    repo = Path(repo or Path(__file__).resolve().parents[2])
    return {str(p.relative_to(repo)):sha256(p) for folder in ('bidcell','scripts','tests','configs')
        for p in sorted((repo/folder).rglob('*')) if p.is_file() and p.suffix in ('.py','.R','.sh','.json','.yaml')
        and '__pycache__' not in p.parts}

def load_run(config):
    cfg = read(config) if isinstance(config,(str,Path)) else dict(config)
    root = Path(cfg['output_root']).resolve()
    if root != OUTPUT_ROOT and OUTPUT_ROOT not in root.parents:
        raise ValueError('output root must be isolated under new boundary root')
    if cfg['version'] != VERSION or cfg['node_order'] != list(NODES):
        raise ValueError('frozen version/node order drift')
    if cfg['endpoints'] != endpoint_registry():
        raise ValueError('historical endpoint registry drift')
    if cfg.get('source_digest') is not None:
        expected=SOURCE_ROOT/'docs/boundary_three/A_CORE_SCOPE_ADDENDUM_20261015.md'
        if cfg.get('scientific_addenda',{}).get('a2_type_scope')!=identity(expected):
            raise ValueError('frozen A2 type-scope addendum missing or changed')
    return cfg

def output_path(cfg, *parts):
    root = Path(cfg['output_root']).resolve(); path = root.joinpath(*parts).resolve()
    if root != path and root not in path.parents: raise ValueError('output escape')
    return path

def endpoint_spec(cfg,name):
    if name in cfg['endpoints']: return dict(cfg['endpoints'][name])
    if name not in NODES and name not in A2_ENDPOINTS: raise ValueError(name)
    base = output_path(cfg,'A2_CORE','endpoints',name) if name in A2_ENDPOINTS else output_path(cfg,name)
    return dict(endpoint_id=name,root=str(base),machine=str(base/'configs/machine.json'),
        pre=str(base/'predictions'/name/'pre.tif'), final=str(base/'predictions'/name/'final.tif'),
        checkpoint=str(base/'checkpoints/final.pth'), matrix=str(base/'matrix'/name),
        evaluation=str(base/'evaluation'/name),cost_root=str(base))

def inventory(root):
    root=Path(root)
    return {str(p.relative_to(root)):identity(p) for p in sorted(root.rglob('*'))
        if p.is_file() and p.name not in ('complete.json','progress.json','run.lock')
        and '.tmp.' not in p.name and not p.name.endswith('.tmp')}

def receipt_dependencies(cfg,node):
    return dict(run_id=cfg['run_id'],node=node,config_digest=digest(cfg),
        source_digest=cfg.get('source_digest'),input_digest=cfg.get('input_digest'))

def write_complete(cfg,node,science_status,artifacts=None,**details):
    root=output_path(cfg,node)
    rec=dict(schema=RECEIPT_SCHEMA,execution_status='COMPLETED',science_status=science_status,
        dependencies=receipt_dependencies(cfg,node),outputs=artifacts if artifacts is not None else inventory(root),
        completed_utc=utc(),**details)
    atomic_json(root/'complete.json',rec); return rec

def validate_complete(cfg,node):
    rec=read(output_path(cfg,node,'complete.json'))
    if rec.get('schema')!=RECEIPT_SCHEMA or rec.get('execution_status')!='COMPLETED': raise ValueError('incomplete node '+node)
    if rec['dependencies']!=receipt_dependencies(cfg,node): raise ValueError('node lineage drift '+node)
    if not rec['outputs']: raise ValueError('empty output receipt '+node)
    for obj in rec['outputs'].values(): check_identity(obj)
    return rec

@contextlib.contextmanager
def compute_lock(timeout=600):
    """One real mini or heavy scientific worker across all isolated worktrees."""
    GPU_LOCK.parent.mkdir(parents=True,exist_ok=True)
    with GPU_LOCK.open('a') as stream:
        end=time.monotonic()+timeout
        while True:
            try: fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB); break
            except BlockingIOError:
                if time.monotonic()>=end: raise RuntimeError('compute lock wait exhausted')
                time.sleep(.5)
        try: yield stream
        finally: fcntl.flock(stream,fcntl.LOCK_UN)

def progress(cfg,node,stage,**values):
    atomic_json(output_path(cfg,node,'progress.json'),dict(node=node,stage=stage,
        host=socket.gethostname(),updated_utc=utc(),**values))

def stop_requested(cfg):
    return output_path(cfg,'control','stop.json').exists()
