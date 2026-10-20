"""Local paths and durable identities; no mutation of frozen scientific modules."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'source'
ARMS = {'R2_BIDCELL_NATIVE_MATCHED': ('BASE', 3625),
        'R2_V2_MORPH_NATIVE': ('V2_MORPH', 3625),
        'R2_B1_LABEL_ANCHOR': ('BASE', 4000)}
BASE = next(iter(ARMS))
B1 = 'R2_B1_LABEL_ANCHOR'

def read(path): return json.loads(Path(path).read_text())

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()

def identity(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))

def digest(value): return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def atomic_torch(path, value):
    import torch
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:
        torch.save(value, f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def verify_identity(obj):
    if identity(obj['path']) != obj: raise ValueError('artifact identity changed: '+obj['path'])

def native_config():
    from bidcell.config import load_config
    cfg = load_config(str(ROOT/'configs/native.yaml'))
    assert Path(cfg.files.data_dir).resolve() == ROOT/'shared/data'
    assert cfg.affine.target_pix_um == 1. and cfg.model_params.patch_size == 48
    assert cfg.transcripts.shift_to_origin is False and cfg.transcripts.counts_col is None
    return cfg

def configure():
    """Inject only dataset/config locations into unchanged imported definitions."""
    import scripts.new_methods.common as common
    import scripts.new_methods.data as data
    common.BASELINE = ROOT/'shared/data'; common.BASE_CONFIG = ROOT/'configs/native.yaml'
    common.native_config = native_config
    data.BASELINE = ROOT/'shared/data'; data.native_config = native_config
    return native_config()

def source_identity():
    return {str(p.relative_to(SOURCE)): sha(p) for folder in ('bidcell', 'scripts/rep2_training',
            'scripts/new_methods', 'scripts/paired_v2', 'scripts/crgf', 'tests/new_methods')
            for p in sorted((SOURCE/folder).rglob('*')) if p.is_file() and p.suffix in ('.py','.sh')}

def base_dependencies():
    config = read(ROOT/'configs/run.json')
    return dict(config=identity(ROOT/'configs/run.json'), native=identity(ROOT/'configs/native.yaml'),
                source=source_identity(), inputs=config['input_identities'])

def validate_receipt(path, dependencies=None):
    rec = read(path)
    if rec.get('status') != 'succeeded': raise ValueError('unsuccessful receipt '+str(path))
    if dependencies is not None and rec['dependencies'] != dependencies: raise ValueError('stage dependency drift '+str(path))
    for obj in rec['outputs'].values(): verify_identity(obj)
    return rec

def receipt(stage, dependencies, outputs, **extra):
    rec = dict(status='succeeded', stage=stage, dependencies=dependencies,
               outputs={str(k): identity(v) for k,v in outputs.items()}, **extra)
    atomic_json(ROOT/'receipts'/f'{stage}.json', rec)
    return rec

def progress(stage, **extra):
    atomic_json(ROOT/'progress'/f'{stage}.json', dict(stage=stage, time=time.time(), pid=os.getpid(), **extra))

def configure_seed():
    import imgaug as ia
    from scripts.new_methods.common import seed_all
    ia.random.get_global_rng(); seed_all(1234); ia.seed(1234)

@contextlib.contextmanager
def lock(path, read_only=False):
    import fcntl
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('r' if read_only else 'a+') as stream:
        try: fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('pipeline already running; inspect existing process')
        if not read_only:
            stream.seek(0); stream.truncate(); stream.write(str(os.getpid())); stream.flush()
        yield

def allocated_run_bytes():
    seen=set();total=0
    for folder,dirs,files in os.walk(ROOT,followlinks=False):
        for name in files:
            path=Path(folder)/name
            if path.is_symlink():continue
            try:stat=path.stat()
            except FileNotFoundError:continue
            key=(stat.st_dev,stat.st_ino)
            if key not in seen:total+=stat.st_blocks*512;seen.add(key)
    return total

def resource_check(stage, minimum_extra_bytes=0):
    import shutil
    rec=read(ROOT/'manifests/resource_admission.json');reserve=6*(1<<30)
    allocated=allocated_run_bytes();free=shutil.disk_usage(ROOT).free
    required=max(reserve+minimum_extra_bytes,rec['required_peak_bytes_including_reserve']-allocated)
    if free<required:raise RuntimeError(f'{stage}: free disk {free} < required remaining {required} bytes; allocated run {allocated}')
    info={k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())}
    ram_required={'nuclei':16,'expression':17,'nuclear_matrix':12,'prepare':12,'runtime_validation':10}.get(stage,8)*(1<<30)
    if info['MemAvailable']<ram_required:raise RuntimeError(f'{stage}: available RAM {info["MemAvailable"]} < required {ram_required} bytes')
    return dict(free_bytes=free,remaining_required_bytes=required,allocated_run_bytes=allocated,ram_available_bytes=info['MemAvailable'])
