"""Detached serial compute queue. No AI, automatic resume, publication or retries."""
from __future__ import annotations
from contextlib import ExitStack,contextmanager
from datetime import datetime,timezone
from collections import deque
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import threading
import time
import traceback
import uuid
from . import execution as ex,science
from .contracts import STAGE_ORDER,STAGES,EVALUATION_STAGES
from .common import (atomic_json,check_identity,check_receipt,dependencies,digest,identity,
                     inventory,last_committed_witness,read,sha256,within,write_receipt)

HOLD='COMPUTE_COMPLETE_AWAITING_MANUAL_RESUME'
GIB=1024**3
def utc():return datetime.now(timezone.utc).isoformat()
def namespace():return os.stat('/proc/self/ns/pid').st_ino
def process_identity(pid=None):
    pid=os.getpid() if pid is None else int(pid);s=Path(f'/proc/{pid}/stat').read_text();a=s[s.rfind(')')+2:].split()
    if a[0]=='Z':raise ProcessLookupError('zombie process')
    return dict(host=socket.gethostname(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),pid_namespace=namespace(),pid=pid,start_ticks=int(a[19]))
def alive(saved):
    if not saved:return False
    if saved.get('pid_namespace')!=namespace() or saved.get('host')!=socket.gethostname():return None
    try:return process_identity(saved['pid'])==saved
    except (OSError,ValueError,KeyError):return False

@contextmanager
def lock(path,create=True,wait=False):
    path=Path(path)
    if create:path.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(path,(os.O_RDWR|os.O_CREAT) if create else os.O_RDONLY,0o644)
    try:
        deadline=time.monotonic()+30
        while True:
            try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                if not wait or time.monotonic()>deadline:raise RuntimeError('compute lock held: '+str(path))
                time.sleep(.1)
        yield fd
    finally:os.close(fd) # Never LOCK_UN: descendants retain the same OFD.
def lock_held(path):
    try:
        with lock(path,False):return False
    except FileNotFoundError:return False
    except RuntimeError:return True
def paths(cfg):return [Path(cfg['run_root'])/'control/launch.lock',Path(cfg['run_root'])/'control/run.lock',ex.COMPUTE]
def check_fds(cfg,fds):
    if len(fds)!=3 or len(set(fds))!=3:raise PermissionError('three inherited OFD locks required')
    for fd,path in zip(fds,paths(cfg)):
        a,b=os.fstat(fd),path.stat()
        if (a.st_dev,a.st_ino)!=(b.st_dev,b.st_ino):raise PermissionError('inherited lock inode mismatch')
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as err:raise PermissionError('descriptor does not own inherited OFD lock') from err
        if not lock_held(path):raise PermissionError('inherited lock is not exclusive')

def load_config(path):
    path=Path(path).resolve(strict=True);cfg=read(path);root=Path(cfg['run_root']).resolve(strict=True)
    if root.parent!=ex.OUTPUT or not re.fullmatch(r'rep1-triple-simple-boundary-20261017-r[0-9]{2,}',cfg.get('run_id','')) or root.name!=cfg['run_id'] or path!=root/'configs/run.json':raise ValueError('unauthorized run/config location')
    if any(k in cfg for k in ('fixture','fixture_backend','engineering_fixture','fake_backend','test_mode')):raise ValueError('formal configuration rejects fixture options')
    science.config_check(cfg)
    if cfg['python']!=ex.PYTHON or cfg['rscript']!=ex.RSCRIPT or cfg.get('r_libs_user')!=str(ex.R_LIBRARY) or cfg.get('r_libs_site')!='':raise ValueError('fixed interpreter/library mismatch')
    for k,p in dict(source_snapshot=root/'source_snapshot',input_manifest=root/'configs/inputs_manifest.json',domain_manifest=root/'frozen_domain/complete.json',comparison_manifest=root/'configs/comparisons.json').items():
        if cfg[k]!=str(p):raise ValueError('immutable location mismatch: '+k)
    if cfg.get('no_ai_dependency') is not True or cfg.get('auto_ai_wakeup') is not False or cfg.get('publication_manual_resume_required') is not True:raise ValueError('compute/publication/AI boundary drift')
    r=cfg['resources']
    if r['reserve_bytes']<15*GIB or r['ram_reserve_bytes']<6*GIB or r['ram_allowance_bytes']!=22*GIB or r['heartbeat_seconds']!=10:raise ValueError('resource/heartbeat contract weakened')
    if r['stale_seconds']!=60 or r['no_progress_seconds']!=900 or set(r['retained_bytes'])!={'gt','p2_replay','analysis'}:raise ValueError('resource contract drift')
    if any(type(v) is not int or v<0 for v in r['retained_bytes'].values()) or r['temporary_peak_bytes']!=4*GIB or r['shared_bytes']!=2*GIB:raise ValueError('invalid resource budget')
    if r['retained_bytes']!=dict(gt=int(4.5*GIB),p2_replay=int(5.5*GIB),analysis=15*GIB):raise ValueError('unreviewed budget change')
    ex.verify_contracts(cfg);return cfg

def new_state(cfg):
    return dict(schema='rep1-triple-simple-state-v1',run_id=cfg['run_id'],config_digest=digest(cfg),
        source_commit=cfg['source_commit'],source_digest=cfg['source_digest'],input_digest=cfg['input_digest'],domain_digest=cfg['domain_digest'],plan_digest=cfg['plan_digest'],environment_digest=cfg['environment_digest'],source_snapshot=cfg['source_snapshot'],
        status='PREPARED',stage=None,attempt=None,completed=0,total=15,last_heartbeat=None,heartbeat_sequence=0,last_scientific_progress=None,progress_sequence=0,
        entry=None,supervisor=None,worker=None,children=[],exit_code=None,error=None,attempts=[],receipts={},child_exit_codes={},publication_allowed=False,compute_is_paused=False)
def bound_state(cfg):
    s=read(Path(cfg['run_root'])/'state.json')
    if s['run_id']!=cfg['run_id'] or s['config_digest']!=digest(cfg):raise ValueError('state/config mismatch')
    return s
def event(cfg,kind,**data):
    with (Path(cfg['run_root'])/'events.jsonl').open('a') as f:f.write(json.dumps(dict(time=utc(),kind=kind,**data),allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())

def tree(roots):
    records={}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            s=(p/'stat').read_text();a=s[s.rfind(')')+2:].split();records[int(p.name)]=(int(a[1]),a[0])
        except (FileNotFoundError,ProcessLookupError):continue
    found=set(int(p) for p in roots);changed=True
    while changed:
        extra={p for p,(parent,state) in records.items() if parent in found and state!='Z'}-found;changed=bool(extra);found|=extra
    return sorted(p for p in found if p in records and records[p][1]!='Z')
def owned_pids(cfg):
    roots={os.getpid()}
    # Preparation parameters and the final run configuration are intentionally
    # different immutable objects. A resumed preparer owns only this process
    # tree; it must not interpret an existing final state as partial-config state.
    if cfg.get('schema')=='rep1-triple-simple-preparation-v1':return tree(roots)
    try:
        state=bound_state(cfg)
        for key in ('entry','supervisor','worker'):
            p=state.get(key)
            if alive(p) is True:roots.add(p['pid'])
    except FileNotFoundError:pass
    return tree(roots)
def pss(cfg):
    total=private=0;observed=[]
    for pid in owned_pids(cfg):
        try:
            d={line.split(':',1)[0]:int(line.split(':',1)[1].strip().split()[0])*1024 for line in Path(f'/proc/{pid}/smaps_rollup').read_text().splitlines() if ':' in line and line.split(':',1)[0] in ('Pss','Private_Clean','Private_Dirty')}
            total+=d['Pss'];private+=d['Private_Clean']+d['Private_Dirty'];observed.append(process_identity(pid))
        except (FileNotFoundError,ProcessLookupError):continue
    return dict(pss_bytes=total,private_bytes=private,processes=observed)
def memory_available():
    return next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
def allocations(cfg):
    root=Path(cfg['run_root']);sizes=dict(gt=0,p2_replay=0,analysis=0,shared=0,uncommitted_or_temporary=0);seen=set()
    for p in root.rglob('*'):
        if not p.is_file() or p.is_symlink():continue
        try:s=p.stat()
        except FileNotFoundError:continue
        key=(s.st_dev,s.st_ino)
        if key in seen:continue
        seen.add(key);rel=p.relative_to(root);parts=rel.parts
        temporary='tmp' in parts or (p.name.startswith('.') and parts[0]!='source_snapshot') or '.tmp.' in p.name
        if temporary:category='uncommitted_or_temporary'
        elif parts[0] in ('analysis','reused') or '/evaluation/paired/' in '/'+str(rel):category='analysis'
        elif parts[0]=='replay' or parts[:2]==('endpoints','P2_SIMPLE'):category='p2_replay'
        elif parts[0]=='endpoints':category='gt'
        else:category='shared'
        sizes[category]+=s.st_blocks*512
    return sizes
def forecast(cfg,stage_start=False):
    r=cfg['resources'];used=allocations(cfg);remaining={k:max(0,v-used[k]) for k,v in r['retained_bytes'].items()};shared=max(0,r['shared_bytes']-used['shared']);free=shutil.disk_usage(cfg['run_root']).free;mem=memory_available();sample=pss(cfg)
    required=sum(remaining.values())+shared+r['temporary_peak_bytes']+r['reserve_bytes']
    ram_ok=mem+sample['pss_bytes']>=r['ram_allowance_bytes']+r['ram_reserve_bytes'] if stage_start else mem>=r['ram_reserve_bytes'] and sample['pss_bytes']<=r['ram_allowance_bytes']
    return dict(time=utc(),used_allocated_unique_bytes=used,remaining_retained_bytes=remaining,shared_remaining_bytes=shared,free_bytes=free,required_free_bytes=required,temporary_additional_bytes=r['temporary_peak_bytes'],reserve_bytes=r['reserve_bytes'],memory_available_bytes=mem,owned_tree=sample,stage_start=stage_start,admitted=free>=required and ram_ok)
class ResourceGuard:
    def __init__(self,cfg):self.cfg=cfg;self.last=0;self.saved=None
    def __call__(self,start=False):
        if start or time.monotonic()-self.last>10:
            self.saved=forecast(self.cfg,start);self.last=time.monotonic()
        else:
            self.saved=dict(self.saved,time=utc(),free_bytes=shutil.disk_usage(self.cfg['run_root']).free,memory_available_bytes=memory_available(),owned_tree=pss(self.cfg))
            r=self.cfg['resources'];self.saved['admitted']=self.saved['free_bytes']>=self.saved['required_free_bytes'] and self.saved['memory_available_bytes']>=r['ram_reserve_bytes'] and self.saved['owned_tree']['pss_bytes']<=r['ram_allowance_bytes']
        if not self.saved['admitted']:raise RuntimeError('RESOURCE_ADMISSION_FAILED '+json.dumps(self.saved))
        return self.saved

def bootstrap_command(cfg,*args):return [ex.PYTHON,'-I','-S','-B',str(Path(cfg['source_snapshot'])/'scripts/rep1_triple_simple_boundary/bootstrap.py'),*map(str,args)]
def entry_command(cfg,path,attempt):return bootstrap_command(cfg,'_entry','--config',path,'--attempt',attempt)
def supervisor_command(cfg,path,attempt):return bootstrap_command(cfg,'_supervise','--config',path,'--attempt',attempt)
def worker_command(cfg,path,stage,attempt,fds):return bootstrap_command(cfg,'_worker','--config',path,'--stage',stage,'--attempt',attempt,'--fds',','.join(map(str,fds)))
def execution_validator(cfg,fds):return ex.ExecutionValidator(cfg,fds)
def resource_contract():
    return dict(p2_workers=4,retained_bytes=dict(gt=int(4.5*GIB),p2_replay=int(5.5*GIB),analysis=15*GIB),shared_bytes=2*GIB,temporary_peak_bytes=4*GIB,reserve_bytes=15*GIB,ram_allowance_bytes=22*GIB,ram_reserve_bytes=6*GIB,heartbeat_seconds=10,stale_seconds=60,no_progress_seconds=900)
def domain_cardinalities(root):
    import numpy as np,pandas as pd
    from .local_statistics import spatial_masks
    frozen=Path(root)/'frozen_domain';groups=np.load(frozen/'cell_groups.npz');cells=pd.read_csv(frozen/'cells.csv').set_index('cell_id')
    local=len(groups.files)+sum(cells.groupby(['original_type',field],dropna=False).ngroups for field in ('block_500','block_1000','block_2000','RNA_density_quartile','nuclear_crowding_quartile'))
    empty={k:np.empty(0,np.int64) for k in ('BASE_owner','region_bits','competition_type','interface_type','K','local_density','single_unknown')}
    actual=dict(cell_groups=len(groups.files),local_groups=local,spatial_groups=len(list(spatial_masks(empty,cells))),interface_groups=len(groups.files)+4)
    bounds=dict(cell_groups=1600,local_groups=4993,spatial_groups=1836,interface_groups=1604)
    obj=dict(actual=actual,measured_projection_bounds=bounds,interface_pairs=len(pd.read_csv(frozen/'interfaces/pairs.csv')),within_measured_bounds=all(actual[k]<=bounds[k] for k in actual),scope='frozen outcome-independent inputs; no new endpoint production')
    target=Path(root)/'admission/domain_cardinalities.json'
    if target.exists():
        if read(target)!=obj:raise ValueError('committed domain cardinality proof differs')
    else:atomic_json(target,obj)
    if not obj['within_measured_bounds']:raise RuntimeError('frozen cardinalities exceed measured resource projection; new review required')
    return obj

def prepare(output_root,run_id,plans_manifest):
    if Path(output_root).resolve()!=ex.OUTPUT or Path(plans_manifest).resolve()!=ex.PLAN or not re.fullmatch(r'rep1-triple-simple-boundary-20261017-r[0-9]{2,}',run_id):raise ValueError('unauthorized preparation path')
    if ex.git('branch','--show-current')!=ex.BRANCH or ex.git('status','--porcelain','--untracked-files=all'):raise ValueError('prepare requires clean committed authorized branch')
    commit=ex.git('rev-parse','HEAD');audit=ex.preparation_audit(commit);root=ex.OUTPUT/run_id;root.mkdir(parents=True,exist_ok=False)
    for name in ('configs','control','logs','admission','tmp','cache/matplotlib','cache/numba'):(root/name).mkdir(parents=True,exist_ok=True)
    partial=dict(schema='rep1-triple-simple-preparation-v1',run_id=run_id,run_root=str(root),resources=resource_contract(),preparation_audit=audit,resource_proposal=identity(ex.WORK/'docs/BIDCell_Rep1_Triple_Simple_RootResourceRevision_20261017.json'))
    with ExitStack() as stack:
        fds=tuple(stack.enter_context(lock(p)) for p in paths(partial));ResourceGuard(partial)(True);source=ex.copy_snapshot(root,commit)
        partial.update(source_snapshot=source['snapshot_root'],source_commit=commit,source_digest=source['source_digest'])
        params=root/'control/preparation.json';atomic_json(params,partial)
        atomic_json(root/'control/preparation_binding.json',dict(params=identity(params),source_manifest=identity(root/'source_manifest.json')))
        cmd=bootstrap_command(partial,'_prepare','--config',params,'--fds',','.join(map(str,fds)))
        with (root/'logs/preparation.log').open('a') as f:
            result=subprocess.run(cmd,cwd=source['snapshot_root'],env=ex.child_environment(partial),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,pass_fds=fds)
        if result.returncode:raise RuntimeError('snapshot preparation failed; preserve all evidence: '+str(root/'logs/preparation.log'))
    return read(root/'configs/run.json')

def resume_prepare(params):
    params=Path(params).resolve(strict=True);partial=read(params);root=Path(partial['run_root']).resolve(strict=True)
    if root.parent!=ex.OUTPUT or root.name!=partial['run_id'] or not re.fullmatch(r'rep1-triple-simple-boundary-20261017-r[0-9]{2,}',root.name):raise ValueError('preparation resume requires original authorized run')
    if Path.cwd()!=Path(partial['source_snapshot']) or Path(__file__).resolve().parents[2]!=Path(partial['source_snapshot']):raise PermissionError('resume-prepare must run the original immutable snapshot')
    with ExitStack() as stack:
        fds=tuple(stack.enter_context(lock(p,False)) for p in paths(partial));ex.verify_preparation(partial,params)
        if (root/'admission/final.json').exists():raise PermissionError('admitted run cannot resume preparation')
        for name in ('endpoints','analysis','replay','reused'):
            if (root/name).exists():raise PermissionError('compute output exists; preparation cannot resume')
        if (root/'state.json').exists():
            s=read(root/'state.json')
            if s.get('attempts') or s.get('status')!='PREPARED':raise PermissionError('formal dispatch already attempted')
        prepare_snapshot(params,fds)
    return read(root/'configs/run.json')

def prepare_snapshot(params,fds):
    partial=read(params);check_fds(partial,fds);ex.verify_preparation(partial,params)
    root=Path(partial['run_root'])
    guard=ResourceGuard(partial);guard(True)
    from .inputs import freeze_inputs
    from .frozen_domains import freeze
    def progress(x):
        guard()
        atomic_json(root/'control/preparation_progress.json',dict(time=utc(),**x));print(json.dumps(x),flush=True)
    manifest=freeze_inputs(root,ex.INSPECTED,ex.PLAN,progress);freeze(root,manifest,partial['source_digest'],progress)
    domain_cardinalities(root)
    def immutable_json(path,value):
        if path.exists():
            if read(path)!=value:raise ValueError('existing preparation output differs: '+str(path))
        else:atomic_json(path,value)
    comparison=root/'configs/comparisons.json';dag=Path(__file__).with_name('approved_dag.json')
    if comparison.exists():
        if sha256(comparison)!=sha256(dag):raise ValueError('prepared DAG drift')
    else:shutil.copyfile(dag,comparison)
    immutable_json(root/'configs/resource_budget.json',read(check_identity(partial['resource_proposal'])))
    env=ex.environment(partial['source_snapshot'],fds,partial);immutable_json(root/'environment.json',env)
    cfg=dict(partial,schema='rep1-triple-simple-boundary-run-v1',input_manifest=str(root/'configs/inputs_manifest.json'),input_digest=digest(manifest),domain_manifest=str(root/'frozen_domain/complete.json'),domain_digest=sha256(root/'frozen_domain/complete.json'),plan_manifest=str(ex.PLAN),plan_digest=sha256(ex.PLAN),comparison_manifest=str(root/'configs/comparisons.json'),environment_digest=digest(env),python=ex.PYTHON,rscript=ex.RSCRIPT,r_libs_user=str(ex.R_LIBRARY),r_libs_site='',stage_order=list(STAGE_ORDER),science_audit=identity(ex.SCIENCE_AUDIT),resource_budget=identity(root/'configs/resource_budget.json'),
        resources=resource_contract(),preparation_params=identity(params),compute_authorization=dict(mode='admitted_compute',allowed_stages=list(STAGE_ORDER)),endpoint_refs={name:dict(parent=parent) for name,parent in science.PARENTS.items()},no_ai_dependency=True,auto_ai_wakeup=False,publication_manual_resume_required=True)
    immutable_json(root/'configs/run.json',cfg)
    if not (root/'state.json').exists():atomic_json(root/'state.json',new_state(cfg))
    else:
        s=bound_state(cfg)
        if s['status']!='PREPARED' or s['attempts']:raise ValueError('preparation cannot overwrite compute state')
    ex.make_preflight(cfg,fds)
    result=forecast(cfg,True);resource_receipt=root/'admission/resource_preflight.json'
    if resource_receipt.exists():resource_receipt=root/'admission'/('resource_preflight_'+uuid.uuid4().hex+'.json')
    atomic_json(resource_receipt,result)
    if not result['admitted']:raise RuntimeError('prepared but resources cannot admit; no formal compute launched')
    print(json.dumps(dict(prepared=True,config=str(root/'configs/run.json'),source_commit=cfg['source_commit'])),flush=True)

def validate(path,admission=None):
    cfg=load_config(path)
    if admission is not None and Path(admission).resolve()!=Path(cfg['run_root'])/'admission/final.json':raise ValueError('final admission must have run-specific location')
    with ExitStack() as stack:
        fds=tuple(stack.enter_context(lock(p,False)) for p in paths(cfg));v=execution_validator(cfg,fds);v();r=ResourceGuard(cfg)(True)
    return dict(status='VALIDATED_FOR_LAUNCH',config_digest=digest(cfg),admission=v.admission,resource=r)

def launch(path,resume=False,reason=None):
    cfg=load_config(path);root=Path(cfg['run_root'])
    with ExitStack() as stack:
        fds=tuple(stack.enter_context(lock(p,False)) for p in paths(cfg));v=execution_validator(cfg,fds);v();ResourceGuard(cfg)(True);s=bound_state(cfg)
        if resume:
            if not reason or s['status'] not in ('FAILED','RUNNING','STARTING'):raise PermissionError('explicit engineering resume requires failure/stale attempt and reason')
            for a in s['attempts']:
                for key in ('entry','supervisor','worker'):
                    if a.get(key) and alive(a[key]) is not False:raise RuntimeError('prior process alive or not observable')
            for key in ('entry','supervisor','worker'):
                if s.get(key) and alive(s[key]) is not False:raise RuntimeError('prior process alive or not observable')
        elif s['attempts'] or s['status']!='PREPARED':raise PermissionError('duplicate launch refused; use explicit reviewed resume-compute for engineering recovery')
        attempt=f'{len(s["attempts"])+1:02d}-{uuid.uuid4().hex[:12]}';token=root/'control'/('dispatch_'+attempt+'.json')
        atomic_json(token,dict(run_id=cfg['run_id'],config_digest=digest(cfg),attempt=attempt,created=utc(),reason=reason,resume=resume))
        s.update(status='STARTING',attempt=attempt,exit_code=None,error=None,entry=None,supervisor=None,worker=None);s['attempts'].append(dict(attempt=attempt,created=utc(),reason=reason,dispatch=identity(token)));atomic_json(root/'state.json',s)
        session='bidcell-triple-'+attempt;cmd=['tmux','new-session','-d','-s',session,'-c',cfg['source_snapshot']]
        for k,val in ex.controlled_environment(cfg).items():cmd+=['-e',k+'='+val]
        # Retain one PTY descriptor in the small entry waiter. Redirecting all
        # three streams closes the pane PTY and tmux can HUP the process before
        # Python starts. FD3 is not passed to supervisor/scientific children.
        shell='exec '+shlex.join(entry_command(cfg,path,attempt))+' 3<&0 >> '+shlex.quote(str(root/'logs/queue.log'))+' 2>&1 < /dev/null'
        result=subprocess.run([*cmd,shell],env=ex.child_environment(cfg),stdin=subprocess.DEVNULL,capture_output=True,text=True)
        atomic_json(root/'control'/('tmux_dispatch_'+attempt+'.json'),dict(session=session,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,token=identity(token)))
        if result.returncode:
            s.update(status='FAILED',error='tmux dispatch failed: '+result.stderr,exit_code=result.returncode);atomic_json(root/'state.json',s);raise RuntimeError(s['error'])
        # Release all dispatcher OFDs now; tmux cannot inherit them. Supervisor
        # waits on launch gate then independently acquires and owns all three.
    return dict(dispatched=True,attempt=attempt,session=session,config=str(path))

def entry(path,attempt):
    cfg=load_config(path);root=Path(cfg['run_root']);own=process_identity()
    atomic_json(root/'control'/('entry_'+attempt+'.json'),dict(identity=own,attempt=attempt,config_digest=digest(cfg)))
    print(utc(),'detached entry',own['pid'],'attempt',attempt,flush=True)
    child=subprocess.Popen(supervisor_command(cfg,path,attempt),cwd=cfg['source_snapshot'],env=ex.child_environment(cfg),stdin=subprocess.DEVNULL)
    code=child.wait();actual=dict(attempt=attempt,entry=own,supervisor_pid=child.pid,actual_os_exit_code=code,wait_observed=True,time=utc(),config_digest=digest(cfg))
    atomic_json(root/'control'/('supervisor_os_exit_'+attempt+'.json'),actual)
    if code:
        s=bound_state(cfg);s.update(status='FAILED',exit_code=code,error=s.get('error') or 'supervisor OS exit nonzero');atomic_json(root/'state.json',s)
    return code

def witness(cfg,stage,path):
    path=within(cfg['run_root'],path);obj=read(path)
    if path.name.endswith('.rds.committed.json'):
        prod=within(cfg['run_root'],'endpoints',EVALUATION_STAGES[stage],'production_complete.json');dep=dict(dependencies(cfg,stage),production=identity(prod))
        if obj.get('dependencies_digest')!=digest(dep) or not 0<int(obj.get('cursor',0))<=int(obj.get('total',0)):raise ValueError('R witness wrong stage/production/cursor')
        within(cfg['run_root'],check_identity(obj));return identity(path)
    if obj.get('schema')=='rep1-triple-simple-p2-postprocess-v1':
        from .p2 import verify_record
        verify_record(path)
        for item in obj['outputs'].values():within(cfg['run_root'],item['path'])
    else:check_receipt(path,output_root=cfg['run_root'])
    bindings=[]
    def find(v):
        if isinstance(v,dict):
            if 'config_digest' in v:bindings.append(v)
            for x in v.values():find(x)
        elif isinstance(v,list):
            for x in v:find(x)
    find(obj.get('dependencies',{}))
    if not bindings or any(x.get('config_digest')!=digest(cfg) or x.get('run_id')!=cfg['run_id'] for x in bindings):raise ValueError('scientific witness lacks current config binding')
    if stage=='01' and path==science.receipt_path(cfg,'01'):
        if obj.get('coordinate_rows')!=34442716 or obj.get('methods')!=list(science.EXISTING):raise ValueError('stage01 semantic completion coverage missing')
    elif stage=='01':
        if obj.get('n_cells')!=107955 or obj.get('n_genes')!=313 or obj.get('assigned',0)<=0 or obj.get('endpoint') not in science.EXISTING:raise ValueError('launch witness not a semantic old-endpoint validation')
        data=read(next(iter(obj['outputs'].values()))['path'])
        if data.get('eligible')!=34442716 or data.get('shape')!=[107955,313] or data.get('assigned')!=obj['assigned']:raise ValueError('semantic input witness incomplete')
    return identity(path)

def worker(path,stage,attempt,fds):
    cfg=load_config(path);root=Path(cfg['run_root']);v=execution_validator(cfg,fds);guard=ResourceGuard(cfg);guard(True)
    baseline={str(p):sha256(p) for name in ('endpoints','replay','validation','analysis','reused') for p in (root/name).rglob('*.json') if p.is_file()};seen=set(baseline.items());sequence=0;own=process_identity()
    def progress(data):
        nonlocal sequence
        v();resource=guard();candidate=data.get('detail',{}).get('receipt') or last_committed_witness()
        if candidate is None:return
        p=within(root,candidate);key=(str(p),sha256(p))
        if key in seen or read(p).get('reused_from'):return
        item=witness(cfg,stage,p);seen.add(key);sequence+=1
        obj=dict(time=utc(),worker=own,attempt=attempt,stage=stage,config_digest=digest(cfg),sequence=sequence,witness=item,validated_at_commit=True,progress=data,resource=resource,modules=ex.module_origins(cfg['source_snapshot']))
        dest=root/'control/witnesses'/attempt/stage/f'{sequence:08d}.json';atomic_json(dest,obj);atomic_json(root/'control'/f'progress_{attempt}_{stage}.json',dict(obj,event=identity(dest)))
    def validator():v();guard()
    prod=within(root,'endpoints',EVALUATION_STAGES[stage],'production_complete.json') if stage in EVALUATION_STAGES else None
    payload=dict(authorization='admitted_compute',run_id=cfg['run_id'],config_digest=digest(cfg),admission=v.admission,stage=stage,production=identity(prod) if prod else None)
    try:
        science.run_stage(cfg,stage,progress=progress,lock_fds=fds,validate_execution=validator,compute_authorization=payload);science.validate_stage(cfg,stage);v()
        atomic_json(root/'control'/f'worker_exit_{attempt}_{stage}.json',dict(exit_code=0,worker=own,stage=stage,receipt=identity(science.receipt_path(cfg,stage)),modules=ex.module_origins(cfg['source_snapshot'])));return 0
    except BaseException:
        error=traceback.format_exc();print(error,flush=True);atomic_json(root/'control'/f'worker_exit_{attempt}_{stage}.json',dict(exit_code=1,worker=own,stage=stage,error=error));return 1

def supervise(path,attempt):
    cfg=load_config(path);root=Path(cfg['run_root']);stop=threading.Event();mutex=threading.RLock();heartbeat_errors=[]
    with ExitStack() as stack:
        fds=(stack.enter_context(lock(paths(cfg)[0],False,True)),stack.enter_context(lock(paths(cfg)[1],False)),stack.enter_context(lock(paths(cfg)[2],False)))
        v=execution_validator(cfg,fds);guard=ResourceGuard(cfg);guard(True);s=bound_state(cfg)
        token=read(root/'control'/('dispatch_'+attempt+'.json'))
        if s['attempt']!=attempt or token['config_digest']!=digest(cfg):raise ValueError('dispatch attempt/config mismatch')
        own=process_identity();ent=read(root/'control'/('entry_'+attempt+'.json'))['identity'];s.update(status='RUNNING',supervisor=own,entry=ent,actual_cwd=str(Path.cwd()),exit_code=None)
        s['attempts'][-1].update(supervisor=own,entry=ent)
        def save():atomic_json(root/'state.json',s)
        def heartbeat():
            while not stop.is_set():
                try:
                    with mutex:
                        s.update(last_heartbeat=utc(),heartbeat_sequence=s['heartbeat_sequence']+1,children=pss(cfg)['processes']);save()
                except BaseException:heartbeat_errors.append(traceback.format_exc());return
                stop.wait(cfg['resources']['heartbeat_seconds'])
        save();thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
        try:
            for stage in STAGE_ORDER[:-1]:
                v();guard(True)
                if science.receipt_path(cfg,stage).exists():
                    science.validate_stage(cfg,stage)
                    with mutex:s['receipts'][stage]=identity(science.receipt_path(cfg,stage));s['completed']=len(s['receipts']);save()
                    continue
                with mutex:s.update(stage=stage,worker=None);save()
                print(utc(),'stage',stage,STAGES[stage]['name'],'starting',flush=True)
                logpath=root/'logs'/f'{stage}_{STAGES[stage]["name"]}_{attempt}.log'
                with logpath.open('a') as log:
                    child=subprocess.Popen(worker_command(cfg,path,stage,attempt,fds),cwd=cfg['source_snapshot'],env=ex.child_environment(cfg),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,pass_fds=fds,start_new_session=True)
                    pid=process_identity(child.pid)
                    with mutex:s['worker']=pid;s['attempts'][-1]['worker']=pid;save()
                    sequence=0
                    while child.poll() is None:
                        if heartbeat_errors:raise RuntimeError('heartbeat writer failed: '+heartbeat_errors[0])
                        pointer=root/'control'/f'progress_{attempt}_{stage}.json'
                        if pointer.exists():
                            p=read(pointer)
                            if p['attempt']!=attempt or p['stage']!=stage or p['worker']!=pid or p['config_digest']!=digest(cfg) or not p.get('validated_at_commit'):raise ValueError('worker progress binding drift')
                            if p['sequence']>sequence:
                                check_identity(p['event']);sequence=p['sequence']
                                with mutex:s.update(progress_sequence=s['progress_sequence']+1,last_scientific_progress=p['time'],last_witness=p['witness'],last_witness_event=p['event'],resource=p['resource']);save()
                                print(p['time'],'scientific progress',stage,sequence,p['progress']['stage'],p['progress'].get('cursor'),flush=True)
                        time.sleep(.25)
                    code=child.wait()
                with mutex:s['child_exit_codes'][stage]=code;save()
                if code:raise RuntimeError(f'stage {stage} actual worker OS exit {code}; see {logpath}')
                result=read(root/'control'/f'worker_exit_{attempt}_{stage}.json')
                if result['exit_code']!=0 or result['worker']!=pid:raise ValueError('worker success receipt/process mismatch')
                science.validate_stage(cfg,stage)
                with mutex:s['receipts'][stage]=identity(science.receipt_path(cfg,stage));s.update(completed=len(s['receipts']),worker=None);save()
                print(utc(),'stage',stage,'validated, actual worker exit0',flush=True)
            v();science.validate_stage(cfg,'14');complete=identity(root/'compute_complete.json')
            write_receipt(science.receipt_path(cfg,'15'),dependencies(cfg,'15'),{'compute_complete':complete},terminal_state=HOLD,exit_code=0,compute_complete=complete)
            science.validate_stage(cfg,'15')
            with mutex:s['receipts']['15']=identity(science.receipt_path(cfg,'15'));s.update(status=HOLD,stage='15',completed=15,worker=None,exit_code=0,error=None);save()
            event(cfg,'compute_terminal_intent',attempt=attempt,compute_complete=complete);return 0
        except BaseException:
            err=traceback.format_exc();print(err,flush=True)
            with mutex:s.update(status='FAILED',exit_code=1,error=err);save()
            event(cfg,'engineering_failure',attempt=attempt,error=err);return 1
        finally:stop.set();thread.join(timeout=2)

def status(path):
    cfg=load_config(path);s=bound_state(cfg);observed=s['status'];s['disk_status']=s['status'];s['process_observations']={k:alive(s.get(k)) for k in ('entry','supervisor','worker')};s['locks_held']=[lock_held(p) for p in paths(cfg)]
    age=lambda x:None if not x else (datetime.now(timezone.utc)-datetime.fromisoformat(x)).total_seconds()
    s['heartbeat_age_seconds']=age(s['last_heartbeat']);s['scientific_progress_age_seconds']=age(s['last_scientific_progress'])
    if observed=='RUNNING' and (s['process_observations']['supervisor'] is False or (s['heartbeat_age_seconds'] or 0)>cfg['resources']['stale_seconds']):observed='STALE'
    if observed==HOLD and (any(s['locks_held']) or any(x is not False for x in s['process_observations'].values())):observed='TERMINAL_INTENT_AWAITING_PROCESS_EXIT'
    s['observed_status']=observed;s['needs_progress_inspection']=(s['scientific_progress_age_seconds'] or 0)>cfg['resources']['no_progress_seconds'];s['observer_pid_namespace']=namespace();return s
def tail(path,lines=80):
    cfg=load_config(path)
    if lines<1 or lines>10000:raise ValueError('tail lines out of range')
    p=Path(cfg['run_root'])/'logs/queue.log'
    if not p.exists():return ''
    with p.open() as f:return ''.join(deque(f,maxlen=lines))
def complete_check(path,phase='compute'):
    if phase!='compute':raise PermissionError('only compute verification before genuine manual resume')
    cfg=load_config(path);s=bound_state(cfg)
    if s['status']!=HOLD or s['completed']!=15:raise RuntimeError('compute queue has not completed')
    for key in ('entry','supervisor','worker'):
        if s.get(key) and alive(s[key]) is not False:raise RuntimeError('process still alive or host namespace not observable: '+key)
    root=Path(cfg['run_root']);exitrec=read(root/'control'/('supervisor_os_exit_'+s['attempt']+'.json'))
    if exitrec.get('actual_os_exit_code')!=0 or exitrec.get('wait_observed') is not True or exitrec.get('config_digest')!=digest(cfg) or exitrec.get('supervisor_pid')!=s['supervisor']['pid']:raise ValueError('actual supervisor OS exit0 not proven')
    with ExitStack() as stack:
        fds=tuple(stack.enter_context(lock(p,False)) for p in paths(cfg));v=execution_validator(cfg,fds);v()
        for stage in STAGE_ORDER:science.validate_stage(cfg,stage)
    return dict(status=HOLD,verified=True,completed=15,total=15,compute_complete=identity(root/'compute_complete.json'),actual_supervisor_exit=identity(root/'control'/('supervisor_os_exit_'+s['attempt']+'.json')),publication_allowed=False,manual_resume_required=True)
