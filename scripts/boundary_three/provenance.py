"""Read-only input capture with actual complete-chain content identities."""
from pathlib import Path
import importlib.metadata
import platform
import subprocess
import sys
from .common import *
from .evaluation import check_legacy_identity,verify_legacy_receipt,evaluation_contract,validate_endpoint

OBJECTIVE=Path('/home/zwan0726/.codex/attachments/c3214324-f74b-44ad-b7ed-a513d595b8a4/goal-objective.md')

def _objects(value):
    if isinstance(value,dict):
        if {'path','sha256'}<=set(value):yield value
        else:
            for item in value.values():yield from _objects(item)
    elif isinstance(value,list):
        for item in value:yield from _objects(item)

def capture_inputs(cfg,dest):
    cfg=load_run(cfg);files={};expected={};start=time.monotonic()
    def add(path,old=None):
        path=str(Path(path).resolve())
        if old is not None:
            if path in expected and expected[path]!=old['sha256']:raise ValueError('conflicting historical identities '+path)
            expected[path]=old['sha256']
        files.setdefault(path,None)
    add(OBJECTIVE)
    for name in PLAN_FILES.values():add(WORK/'docs'/name)
    add(WORK/'context/07_evaluation_dual_annotation_20261013.md')
    for key,path in cfg['inputs'].items():
        p=Path(path)
        if p.is_file():add(p)
    queue,protocol,contract=evaluation_contract(cfg)
    add(protocol);add(Path(queue['prepared'])/'complete.json')
    for obj in _objects(read(Path(queue['prepared'])/'complete.json')):add(obj['path'],obj)
    prepared=read(cfg['inputs']['prepared_manifest'])
    for obj in _objects(prepared):add(obj['path'],obj)
    for shift in ('predict_shift0','predict_shift24'):
        for row in read(cfg['inputs'][shift]):add(row['path'],row['identity'])
    for name in cfg['endpoints']:
        e=endpoint_spec(cfg,name);machine=read(e['machine'])
        for key in ('machine','pre','final','checkpoint'):add(e[key])
        for key in ('nuclei_mask','transcripts','genes','preannotation','baseline_config','reference','positive_markers','negative_markers','nuclear_matrix'):
            add(machine[key])
        for stage in ('matrix','evaluation'):
            receipt=Path(e[stage])/'complete.json';add(receipt);rec=verify_legacy_receipt(receipt)
            for obj in _objects(rec):add(obj['path'],obj)
            if stage=='matrix':
                for chunk in rec['receipts']:
                    add(chunk['coordinate_file'],dict(path=chunk['coordinate_file'],sha256=chunk['coordinate_sha256']))
                    add(Path(e[stage])/chunk['owner_file'],dict(path=str(Path(e[stage])/chunk['owner_file']),sha256=chunk['sha256']))
    for key in ('frozen_strata','frozen_counts'):
        folder=Path(cfg['inputs'][key]);add(folder/'complete.json')
        rec=verify_legacy_receipt(folder/'complete.json')
        for obj in _objects(rec):add(obj['path'],obj)
    # The historical reuse audit includes full-chain producers and files. These
    # remain original producer evidence, rather than this run's independent audit.
    reuse=read(cfg['inputs']['historical_reuse'])
    for obj in _objects(reuse):add(obj['path'],obj)
    if 'provenance' in reuse:
        historical=read(reuse['provenance']['path'])
        for obj in _objects(historical):add(obj['path'],obj)
        oldsource=WORK/'projects/BIDCell_new_methods_20261013'
        for mapping in historical.get('producer_maps',{}).values():
            for path,sha in mapping.items():add(oldsource/path,dict(path=str(oldsource/path),sha256=sha))
    total=len(files)
    for i,path in enumerate(sorted(files)):
        obj=identity(path)
        if path in expected and expected[path]!=obj['sha256']:raise ValueError('input content differs from source receipt '+path)
        files[path]=obj
        if i%1000==0:
            atomic_json(Path(dest).with_name('input_capture_progress.json'),dict(files_checked=i,total_files=total,time=utc()))
            print('input capture',i,'/',total,flush=True)
    payload=dict(schema='boundary-three-input-manifest-v1',created_utc=utc(),host=platform.node(),
        inputs=files,expected_historical_hashes=expected,plans={k:identity(WORK/'docs'/v) for k,v in PLAN_FILES.items()},
        objective=identity(OBJECTIVE),evaluation_contract=contract,endpoint_registry=cfg['endpoints'],
        original_ids=107955,eligible_molecules=34442716,spatial_genes=313,
        input_digest=digest(files),elapsed_seconds=time.monotonic()-start)
    atomic_json(dest,payload);return payload

def verify_inputs(cfg,manifest=None):
    cfg=load_run(cfg);manifest=manifest or read(output_path(cfg,'input_manifest.json'))
    if manifest['input_digest']!=cfg['input_digest'] or digest(manifest['inputs'])!=cfg['input_digest']:raise ValueError('input manifest digest mismatch')
    for obj in manifest['inputs'].values():check_identity(obj)
    if manifest['endpoint_registry']!=cfg['endpoints']:raise ValueError('endpoint identity registry changed')
    return dict(status='passed',files=len(manifest['inputs']),input_digest=cfg['input_digest'])

def environment_check(cfg,require_cuda=False):
    cfg=load_run(cfg)
    if platform.node()!='MCV-CGB4' or str(Path(sys.executable).resolve())!=str(Path(cfg['python']).resolve()):raise ValueError('unadmitted host or interpreter')
    expected=read(cfg['inputs']['environment_lock'])
    actual={d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata.get('Name')}
    if dict(sorted(actual.items()))!=expected['python_packages']:raise ValueError('Python environment drift')
    import torch
    if require_cuda and not torch.cuda.is_available():raise RuntimeError('GPU is unavailable')
    expr='pkgs <- c("scClassify","scater","SpatialExperiment","Metrics","SingleCellExperiment","Matrix","limma","jsonlite"); vals <- setNames(vapply(pkgs,function(p)as.character(packageVersion(p)),""),pkgs); cat(jsonlite::toJSON(list(R=R.version.string,packages=as.list(vals)),auto_unbox=TRUE))'
    result=read(cfg['inputs']['paper_R_environment'])
    measured=json.loads(subprocess.check_output([cfg['rscript'],'-e',expr],text=True,
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')))
    if {k:result[k] for k in ('R','packages')}!=measured:raise ValueError('R evaluation environment drift')
    return dict(status='passed',host=platform.node(),python=sys.executable,python_packages=actual,
        R=measured,gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
