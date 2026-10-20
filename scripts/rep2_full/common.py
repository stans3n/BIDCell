"""Isolated Rep2 orchestration and identities; frozen science stays unchanged."""
import contextlib
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import time
from scripts.rep1_triple_simple_boundary.common import (read, identity, digest,
    check_identity, atomic_json, atomic_npy, atomic_npz, atomic_csv, atomic_tiff,
    check_receipt, write_receipt, inventory)

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT/'source'
TRAINING = (ROOT/'training').resolve()
BASE='BASE_MATCH'; V2='V2_MORPH'; B1='B1'; P2='P2_MORPH'
B0='G_SCORE_ONLY_REF_GATED'; BETA1='P2_MORPH_GT'
ENDPOINTS=(BASE,V2,B1,P2,B0,BETA1)
ARMS={BASE:'R2_BIDCELL_NATIVE_MATCHED',V2:'R2_V2_MORPH_NATIVE',B1:'R2_B1_LABEL_ANCHOR'}
NAMES={**ARMS,P2:'R2_P2_MORPH',B0:'R2_G_SCORE_ONLY_REF_GATED',BETA1:'R2_P2_MORPH_GT'}
AUDITS=('training_reuse_validation','label_contract_audit','r_runtime_audit',
        'resource_admission','method_reference_contract_audit','execution_scope')
STAGES=['shared','inference_capacity', 'B1_predict','B1_cleanup','B1_matrix','B1_annotate','labels',
        'BASE_MATCH_predict','BASE_MATCH_cleanup','BASE_MATCH_matrix','roi',
        'V2_MORPH_predict','V2_MORPH_cleanup','V2_MORPH_matrix','P2',
        'P2_MORPH_matrix','reference','domain','beta0','G_SCORE_ONLY_REF_GATED_matrix',
        'beta1','P2_MORPH_GT_matrix']+[n+'_evaluate' for n in ENDPOINTS]+['statistics','acceptance']

def config(): return read(ROOT/'configs/run.json')
def source_files():
    return {str(p.relative_to(SOURCE)):identity(p)['sha256'] for p in sorted(SOURCE.rglob('*'))
            if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts and p.suffix in ('.py','.R','.sh','.yaml','.json')}
def source_digest():return digest(source_files())
def base_deps():return dict(config=identity(ROOT/'configs/run.json'),source_digest=config()['source_digest'])
def deps(stage):
    i=STAGES.index(stage)
    # A hash chain binds every prior accepted stage, including input freezes.
    return dict(base_deps(),stage=stage,previous=None if i==0 else identity(ROOT/'receipts'/(STAGES[i-1]+'.json')))
def progress(stage, completed=None, total=None, **kw):
    event=dict(stage=stage,completed=completed,total=total,pid=os.getpid(),time=time.time(),**kw)
    atomic_json(ROOT/'progress/current.json',event)
    print(__import__('json').dumps(event,sort_keys=True),flush=True)
def cfg_progress(stage, completed=None, total=None, **kw):
    if isinstance(stage,dict):
        event=stage;return progress(event['stage'],event.get('cursor'),event.get('total'),**event.get('detail',{}))
    return progress(stage,completed,total,**kw)
def coordinates():return read(ROOT/'shared/coordinates/manifest.json')['records']
def input_manifest():
    return dict(metadata=str(TRAINING/'shared/prepared/nucleus_metadata.npz'),
        reference_region=str(TRAINING/'shared/configs/reference_region.json'),
        fixed_population=str(ROOT/'frozen_labels/historical_fixed_population.csv'),
        coordinates=coordinates(),evaluation=config()['evaluation'],
        endpoints={n:spec(n) for n in ENDPOINTS})
def spec(name):
    root=ROOT/'endpoints'/name
    return dict(name=name,root=str(root),final=str(root/'final.tif'),
        nuclei=str(TRAINING/'shared/data/nuclei.tif'),machine=str(root/'machine.json'),
        matrix=str(root/'matrix'),evaluation=str(root/'evaluation'),
        fixed_scores=str(root/'evaluation/fixed/fixed_population.csv'),
        sc_scores=str(root/'evaluation/paper_aligned_v1/markers_per_cell.csv'))
def machine(name):
    s=spec(name);old=TRAINING/'shared/data'
    return dict(final_mask=s['final'],nuclei_mask=s['nuclei'],genes=str(old/'all_gene_names.txt'),
        transcripts=str(old/'transcripts_processed.csv'),baseline_config=str(TRAINING/'configs/native.yaml'),
        preannotation=str(old/'nuclei_cell_type.h5'),pixel_um=1.,scale_x=.2125,scale_y=.2125,
        final_matrix=str(Path(s['matrix'])/'expr_mat.csv'),chunksize=500000)
def materialize_machine(name):
    value=machine(name);path=Path(spec(name)['machine'])
    if path.exists() and read(path)!=value:raise ValueError('immutable endpoint machine changed')
    if not path.exists():atomic_json(path,value)
    return value
def native_config():
    from bidcell.config import load_config
    return load_config(str(ROOT/'configs/inference_native.yaml'))
def configure():
    import scripts.new_methods.common as c
    import scripts.new_methods.data as d
    c.BASELINE=d.BASELINE=ROOT/'shared/data';c.BASE_CONFIG=ROOT/'configs/inference_native.yaml'
    c.native_config=d.native_config=native_config
    return native_config()
def immutable_outputs(folder):
    """Live logs, progress and R process identities are never stage products."""
    folder=Path(folder)
    return {str(p.relative_to(folder)):identity(p) for p in sorted(folder.rglob('*')) if p.is_file()
        and not p.is_symlink() and not p.name.startswith('.') and 'tmp' not in p.parts and '__pycache__' not in p.parts
        and not any(x in p.name for x in ('.tmp','.log','progress','_process.json','_exit.json'))
        and p.name!='run.lock'}
def stage_outputs(*folders):
    result={}
    for folder in folders:
        for _,obj in immutable_outputs(folder).items():result[obj['path']]=obj
    return result
def validate_outputs(outputs):
    for obj in outputs.values():check_identity(obj)
def allocated_bytes():
    seen=set();total=0
    for folder,dirs,files in os.walk(ROOT,followlinks=False):
        for name in files:
            p=Path(folder)/name
            if p.is_symlink():continue
            try:s=p.stat()
            except FileNotFoundError:continue
            key=(s.st_dev,s.st_ino)
            if key not in seen:seen.add(key);total+=s.st_blocks*512
    return total
def resources(stage, extra=0):
    admission=read(ROOT/'manifests/resource_admission.json');reserve=6*(1<<30)
    required=max(reserve+extra,admission['required_peak_bytes_including_reserve']-allocated_bytes())
    free=shutil.disk_usage(ROOT).free
    if free<required:raise RuntimeError(f'{stage}: disk free {free} < remaining admitted {required}')
    mem={k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())}
    need=(18 if stage in ('reference','statistics') else 12)*(1<<30)
    if mem['MemAvailable']<need:raise RuntimeError(f'{stage}: RAM {mem["MemAvailable"]} < admitted {need}')
    return dict(free_bytes=free,required_free=required,available_ram=mem['MemAvailable'])
@contextlib.contextmanager
def locks():
    with (ROOT/'pipeline.lock').open('a+') as own, Path('/home/zwan0726/work/outputs/cos_gpu.lock').open('r') as gpu:
        for f in (own,gpu):fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        own.seek(0);own.truncate();own.write(str(os.getpid()));own.flush()
        yield (own.fileno(),gpu.fileno())
def inherited_fds():
    fds=tuple(int(x) for x in os.environ.get('REP2_FULL_LOCK_FDS','').split(',') if x)
    if not fds:raise PermissionError('production stages require inherited pipeline and workspace locks')
    for fd in fds:os.fstat(fd)
    return fds
def validate_runtime():
    inherited_fds()
    if (ROOT/'STOP_AFTER_STAGE').exists():
        # Stop only the controller at a validated stage boundary; do not kill R.
        pass
def review_binding():
    c=config()
    if c['source_digest']!=source_digest():raise ValueError('reviewed source changed')
    review=read(ROOT/'review/main_review.json')
    if review.get('status') not in ('passed','approved') or review['source_digest']!=source_digest() or review['config_sha256']!=identity(ROOT/'configs/run.json')['sha256']:
        raise ValueError('independent implementation review/config binding missing')
    if c['endpoints']!=NAMES or c['seed']!=1234 or c['bootstrap_seed']!=20261019:raise ValueError('frozen scope changed')
def preflight():
    c=config();review_binding()
    for obj in c['frozen_inputs'].values():check_identity(obj)
    for name in AUDITS:
        rec=read(ROOT/'manifests'/(name+'.json'))
        if rec.get('status') not in ('passed','succeeded','frozen'):raise ValueError('audit not passed: '+name)
    return resources('preflight')
