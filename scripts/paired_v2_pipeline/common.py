"""Frozen execution contracts and read-only provenance checks."""
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

from scripts.paired_v2.common import (DEFAULT_ROOT, INPUT, REPO, RUNS, atomic_json,
                                    identity, read, sha256, digest, config_path,
                                    load_config, producer, verify_outputs)

PYTHON = '/home/zwan0726/miniforge3/envs/bidcell_local/bin/python'
RSCRIPT = '/home/zwan0726/work/envs/bidcell_egr_r/bin/Rscript'
METRICS = ('positive_F1','negative_F1','positive_Precision','positive_Recall',
           'negative_Precision','negative_Recall')
ENDPOINTS = ('BASE_MATCH','V2_MORPH','V2_INTERACT')
COMPARISONS = {'old_morph_increment':('V2_MORPH','BASE_MATCH'),
               'interaction_increment':('V2_INTERACT','V2_MORPH'),
               'package_adoption':('V2_INTERACT','BASE_MATCH')}
DOMAINS = ('no_support','single_candidate','unknown_competition',
           'same_type_competition','heterotypic_competition','nuclear','out_of_bounds')


def utc():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def task_graph():
    sequence = [('freeze_inputs',None,'freeze_inputs'),
                ('BASE_MATCH_train','BASE_MATCH','train'),
                ('V2_INTERACT_train','V2_INTERACT','train'),
                ('paired_training',None,'paired_training')]
    for arm in RUNS:
        for stage in ('predict','postprocess','matrix'):
            sequence.append((arm+'_'+stage,arm,stage))
        if arm=='BASE_MATCH':sequence.append(('freeze_counts',None,'freeze_counts'))
        sequence.append((arm+'_evaluate',arm,'evaluate'))
    sequence += [('reports',None,'reports'),('package',None,'package')]
    tasks=[]
    for key,arm,stage in sequence:
        tasks.append(dict(id=key,arm=arm,stage=stage,depends_on=[tasks[-1]['id']] if tasks else []))
    return tasks


def queue(root=DEFAULT_ROOT):
    root=Path(root).resolve();q=read(root/'configs/pipeline.json')
    if root!=DEFAULT_ROOT or q['root']!=str(root) or q['repo']!=str(REPO) or q['python']!=PYTHON:
        raise ValueError('unreviewed execution or output location')
    if q['tasks']!=task_graph() or q['comparisons']!={k:list(v) for k,v in COMPARISONS.items()}:
        raise ValueError('execution graph/comparison drift')
    if q['max_retries']!=1 or q['nightly_publication'] is not False or q['auto_ai_wakeup'] is not False:
        raise ValueError('nightly execution policy drift')
    return q


def check_obj(obj):
    if identity(obj['path'])!=obj:raise ValueError('frozen file identity changed: '+obj['path'])


def verify_execution(q,admitted=False):
    from scripts.paired_v2.finalize import verify_bundle
    check_obj(q['source_bundle']);verify_bundle(q['source_bundle']['path'])
    for obj in q['bindings'].values():check_obj(obj)
    if admitted:
        review=read(q['review_receipt'])
        if review.get('status')!='approved_for_launch' or not review.get('formal_admission'):
            raise ValueError('final independent integration review has not approved launch')
        if review['source_bundle']!=q['source_bundle'] or review['pipeline_config']!=identity(Path(q['root'])/'configs/pipeline.json'):
            raise ValueError('final review does not bind actual source/config')
        if review.get('blocking_findings'):raise ValueError('unresolved final review blockers')
        for arm in RUNS:
            gate=read(Path(q['root'])/'admission'/f'{arm}.json')
            if gate['status']!='admitted' or gate['review']!=identity(q['review_receipt']) or gate['pipeline_config']!=review['pipeline_config']:
                raise ValueError('arm admission drift')
            if gate['train_producer']!=producer('train') or gate['initial_snapshot']!=identity(load_config(config_path(arm))['initial_snapshot']):
                raise ValueError('admitted training producer/initial drift')
    return dict(status='passed',source_digest=read(q['source_bundle']['path'])['source_digest'])


def verify_history(q):
    check_obj(q['bindings']['historical_reuse'])
    reuse=read(q['bindings']['historical_reuse']['path'])
    check_obj(reuse['provenance']);prov=read(reuse['provenance']['path'])
    if not reuse['full_existing_chain_provenance_verified'] or prov['files_checked']!=36344:
        raise ValueError('historical full-chain reuse gate absent')
    for obj in prov['files'].values():
        if sha256(obj['path'])!=obj['sha256']:raise ValueError('historical reused artifact drift: '+obj['path'])
    from scripts.paired_v2.common import HISTORICAL_SOURCE
    for mapping in prov['producer_maps'].values():
        for path,sha in mapping.items():
            if sha256(HISTORICAL_SOURCE/path)!=sha:raise ValueError('historical producer drift: '+path)
    if read(q['bindings']['implementation_review']['path'])['status']!='implementation_approved_for_parent_integration_only':
        raise ValueError('initial implementation/reuse independently unapproved')
    return dict(status='passed',files=36344,provenance=reuse['provenance'])


def process_identity(pid=None):
    pid=pid or os.getpid()
    return dict(pid=pid,host=socket.gethostname(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                start_ticks=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[19],
                command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode().strip())


def alive(obj):
    try:return process_identity(obj['pid'])==obj
    except (OSError,ValueError,KeyError,TypeError):return False


def lock(path):
    stream=Path(path).open('a');fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB);return stream


def environment_check(q):
    import platform
    expected=read(q['bindings']['environment_lock']['path'])
    actual={d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata.get('Name')}
    if dict(sorted(actual.items()))!=expected['python_packages']:raise ValueError('Python package environment drift')
    if platform.node()!='MCV-CGB4':raise ValueError('unadmitted compute host')
    expr='pkgs <- c("scClassify","scater","SpatialExperiment","Metrics","SingleCellExperiment","Matrix","limma","jsonlite"); vals <- setNames(vapply(pkgs,function(p)as.character(packageVersion(p)),""),pkgs); cat(jsonlite::toJSON(list(R=R.version.string,packages=as.list(vals),threads=as.list(Sys.getenv(c("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")))),auto_unbox=TRUE))'
    result=json.loads(subprocess.check_output([RSCRIPT,'-e',expr],text=True,env=numeric_env()))
    if result!=read(q['bindings']['paper_R_environment']['path']):raise ValueError('paper R package/threads environment drift')
    return dict(status='passed',host=platform.node(),paper_R=result)


def numeric_env():
    return dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',
                OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4')


def require_space(q,launch=False):
    free=shutil.disk_usage(q['root']).free
    required=q['disk_forecast_remaining_bytes']+q['disk_reserve_bytes'] if launch else q['disk_reserve_bytes']+q['atomic_peak_bytes']
    if free<required:raise RuntimeError(f'resource admission failed: disk free {free} < {required}')
    mem={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()}
    if mem['MemAvailable']<q['minimum_ram_available_bytes']:raise RuntimeError('resource admission failed: available RAM')
    return dict(free_disk_bytes=free,required_free_bytes=required,available_ram_bytes=mem['MemAvailable'])


def endpoint(arm,root=DEFAULT_ROOT):
    base=(INPUT if arm=='V2_MORPH' else Path(root))/arm
    return dict(arm=arm,root=base,matrix=base/'matrix'/arm,evaluation=base/'evaluation'/arm,
                prediction=base/'predictions'/arm,machine=base/'configs/machine.json')


def inventory(folder):
    folder=Path(folder)
    return {str(p.relative_to(folder)):identity(p) for p in sorted(folder.rglob('*'))
            if p.is_file() and p.name not in ('complete.json','progress.json','run.lock') and not p.name.endswith('.tmp')}


def own_receipt(q,name):
    return Path(q['root'])/'pipeline_receipts'/f'{name}.json'


def own_dependencies(q,task):
    return dict(task=task,pipeline_config=identity(Path(q['root'])/'configs/pipeline.json'),
                source_bundle=q['source_bundle'],upstream={key:identity(own_receipt(q,key)) for key in task['depends_on']})


def validate_task(q,task):
    saved=read(own_receipt(q,task['id']))
    if saved['status']!='completed' or saved['dependencies']!=own_dependencies(q,task):raise ValueError('pipeline receipt dependencies changed: '+task['id'])
    verify_outputs(saved)
    if task['arm']:
        from scripts.paired_v2.stages import validate_stage
        result=validate_stage(config_path(task['arm']),task['stage'],require_bound=True)
        if result!=saved['semantic']:raise ValueError('pipeline/native semantic acceptance drift')
    else:
        from .stages import validate_custom
        result=validate_custom(q,task['stage'])
        if result!=saved['semantic']:raise ValueError('custom stage semantic acceptance drift')
    return saved
