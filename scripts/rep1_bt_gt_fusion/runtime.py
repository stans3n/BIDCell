"""Serial detached supervisor. No clock-driven evaluation, publication, or AI."""
from __future__ import annotations
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import resource
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid
from . import NODE_ORDER
from .common import atomic_json, check_identity, check_receipt, dependencies, digest, identity, read, require_manual_resume, within
from . import execution as ex

HOLD='PRODUCTION_COMPLETE_AWAITING_MANUAL_RESUME'
TERMINAL={HOLD,'EVALUATION_COMPLETE_AWAITING_PUBLICATION','COMPLETED','FAILED'}


def utc():return datetime.now(timezone.utc).isoformat()


def process_identity(pid=None):
    pid=os.getpid() if pid is None else int(pid)
    text=Path(f'/proc/{pid}/stat').read_text();parts=text[text.rfind(')')+2:].split()
    if parts[0]=='Z':raise ProcessLookupError('zombie process')
    return dict(host=socket.gethostname(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                pid=pid,start_ticks=int(parts[19]))


def alive(saved):
    try:
        current=process_identity(saved['pid'])
        return all(current[k]==saved[k] for k in ('host','boot_id','pid','start_ticks'))
    except (OSError,KeyError,TypeError,ValueError):return False


@contextmanager
def lock(path,create=True):
    path=Path(path)
    if create:path.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(path,os.O_RDWR|os.O_CREAT if create else os.O_RDONLY,0o644)
    try:
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise RuntimeError('lock held: '+str(path)) from exc
        yield fd
    finally:os.close(fd)  # Never LOCK_UN: orphan descendants retain the same OFD.


def lock_held(path):
    try:
        with lock(path,create=False):return False
    except FileNotFoundError:return False
    except RuntimeError:return True


@contextmanager
def wait_launch_lock(path):
    """The dispatcher releases this lock immediately after tmux dispatch."""
    fd=os.open(path,os.O_RDWR);deadline=time.monotonic()+30
    try:
        while True:
            try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                if time.monotonic()>=deadline:raise RuntimeError('dispatcher launch lock did not release')
                time.sleep(.1)
        yield fd
    finally:os.close(fd)


def paths(cfg):
    root=Path(cfg['run_root'])
    return [root/'control/launch.lock',root/'control/writer.lock',ex.COMPUTE]


def check_fds(cfg,fds):
    if len(fds)!=3 or len(set(fds))!=3:raise PermissionError('three distinct inherited locks are required')
    for fd,path in zip(fds,paths(cfg)):
        st=os.fstat(fd);expected=path.stat()
        if (st.st_dev,st.st_ino)!=(expected.st_dev,expected.st_ino) or not lock_held(path):
            raise PermissionError('inherited lock descriptor identity/ownership mismatch')


def load_config(path):
    path=Path(path).resolve(strict=True);cfg=read(path)
    root=Path(cfg['run_root']).resolve()
    if root!=ex.OUTPUT/ex.RUN_ID or cfg.get('run_id')!=ex.RUN_ID or path!=root/'configs/run.json':
        raise ValueError('formal configuration/run path differs from authorized run')
    if cfg.get('schema')!='rep1-bt-gt-run-v1' or cfg.get('node_order')!=list(NODE_ORDER):raise ValueError('run schema/scope drift')
    if any(k in cfg for k in ('fixture','fixture_backend','engineering_fixture','fake_backend')):raise ValueError('fixture config is not formal')
    if cfg.get('python')!=ex.PYTHON or cfg.get('rscript')!=ex.RSCRIPT or cfg.get('r_libs_user')!=str(ex.R_LIBRARY) or cfg.get('r_libs_site')!='':
        raise ValueError('interpreter/library contract drift')
    if cfg.get('no_ai_dependency') is not True or cfg.get('auto_ai_wakeup') is not False:raise ValueError('AI/scheduler dependency forbidden')
    if Path(cfg['source_snapshot'])!=root/'source_snapshot' or Path(cfg['input_manifest'])!=root/'configs/inputs_manifest.json':
        raise ValueError('frozen file location mismatch')
    r=cfg['resources']
    if r['reserve_bytes']<15*1024**3 or r['ram_reserve_bytes']<6*1024**3 or r['heartbeat_seconds']!=10 or r['stale_seconds']!=60 or r['no_progress_seconds']!=900:
        raise ValueError('resource/watchdog contract weakened')
    for key in ('shared_bytes','temporary_peak_bytes','ram_allowance_bytes'):
        if type(r[key]) is not int or r[key]<0:raise ValueError('invalid byte budget')
    if set(r['retained_bytes'])!=set(NODE_ORDER) or any(type(n) is not int or n<0 for n in r['retained_bytes'].values()):raise ValueError('invalid retained budget')
    ex.verify_contracts(cfg)
    return cfg


def event(cfg,kind,**fields):
    with within(cfg['run_root'],'events.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(time=utc(),event=kind,**fields),allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())


def state_new(cfg):
    return dict(schema='rep1-bt-gt-run-v1',run_id=cfg['run_id'],status='PREPARED',phase='PREPARATION',current_node=None,
                source_commit=cfg['source_commit'],source_digest=cfg['source_digest'],config_digest=digest(cfg),
                input_digest=cfg['input_digest'],plan_sha256=cfg['plan_sha256'],source_snapshot=cfg['source_snapshot'],
                environment_digest=cfg['environment_digest'],supervisor=None,worker=None,heartbeat_utc=None,
                heartbeat_sequence=0,last_progress_utc=None,progress_sequence=0,new_scientific_progress_sequence=0,
                nodes={n:dict(status='pending') for n in NODE_ORDER},attempts=[],child_exit_codes={},receipts={},
                existing_bt_evaluation_allowed=True,new_endpoint_evaluation_allowed=False,publication_allowed=False,
                manual_resume_required=True,failure=None)


def bound_state(cfg):
    state=read(within(cfg['run_root'],'state.json'))
    if state.get('run_id')!=cfg['run_id'] or state.get('config_digest')!=digest(cfg):raise ValueError('state/config binding mismatch')
    return state


def memory():
    data={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key,value=line.split(':',1);data[key]=int(value.strip().split()[0])*1024
    return data['MemAvailable']


def disk_bytes(root):
    if not Path(root).exists():return 0
    return sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file() and not p.is_symlink())


def resource_forecast(cfg,state=None):
    r=cfg['resources'];root=Path(cfg['run_root']);remaining={}
    folders={'BT_EXISTING6':root/'bt_existing6','P2_BASE_GT_PRODUCE':root/'endpoints/P2_BASE_GT',
             'V2_MORPH_GT_PRODUCE':root/'endpoints/V2_MORPH_GT','PRODUCTION_VALIDATE':root/'production_complete.json'}
    for n,target in folders.items():
        used=target.stat().st_size if target.is_file() else disk_bytes(target)
        # Retain unused endpoint room for its manual evaluation too. Completed
        # production must not make future evaluation bytes disappear from budget.
        remaining[n]=max(0,r['retained_bytes'][n]-used)
    scientific_used=sum((p.stat().st_size if p.is_file() else disk_bytes(p)) for p in folders.values())
    shared_used=max(0,disk_bytes(root)-scientific_used)
    shared=max(0,r['shared_bytes']-shared_used)
    peak=sum(remaining.values())+shared+r['temporary_peak_bytes']
    free=shutil.disk_usage(root).free;available=memory()
    return dict(time=utc(),free_bytes=free,ram_available_bytes=available,remaining_retained_bytes=remaining,
                shared_remaining_bytes=shared,temporary_peak_bytes=r['temporary_peak_bytes'],total_remaining_incremental_peak_bytes=peak,
                reserve_bytes=r['reserve_bytes'],required_free_bytes=peak+r['reserve_bytes'],
                ram_required_at_stage_start_bytes=r['ram_allowance_bytes']+r['ram_reserve_bytes'],
                admitted=free>=peak+r['reserve_bytes'] and available>=r['ram_allowance_bytes']+r['ram_reserve_bytes'])


def controlled_environment(cfg):
    root=Path(cfg['run_root'])
    return dict(PYTHONPATH='',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='0',LC_ALL='C.UTF-8',
                OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
                MPLCONFIGDIR=str(root/'cache/matplotlib'),XDG_CACHE_HOME=str(root/'cache'),NUMBA_CACHE_DIR=str(root/'cache/numba'),
                TMPDIR=str(root/'tmp'),R_LIBS_USER=cfg['r_libs_user'],R_LIBS_SITE='',R_PROFILE_USER='/dev/null',R_ENVIRON_USER='/dev/null')


def child_environment(cfg):
    return dict(os.environ,**controlled_environment(cfg))


def prepare(plan,output_root,run_id):
    if Path(plan).resolve()!=ex.PLAN or Path(output_root).resolve()!=ex.OUTPUT or run_id!=ex.RUN_ID:raise ValueError('prepare paths differ from authorized plan')
    if ex.git('branch','--show-current')!='feature/rep1-bt-gt-fusion-20261016' or ex.git('status','--porcelain','--untracked-files=all'):
        raise ValueError('commit the final source on the authorized clean branch before prepare')
    commit=ex.git('rev-parse','HEAD');root=ex.OUTPUT/run_id
    if root.exists() and any(root.iterdir()):raise FileExistsError('formal run already exists; preserve it')
    root.mkdir(parents=True,exist_ok=True)
    for name in ('configs','control','logs','admission','tmp','cache'): (root/name).mkdir(exist_ok=True)
    with ExitStack() as stack:
        fds=tuple(stack.enter_context(lock(p)) for p in (root/'control/launch.lock',root/'control/writer.lock',ex.COMPUTE))
        from .inputs import freeze_inputs
        inputs=freeze_inputs(root)
        source=ex.copy_snapshot(root,commit)
        env=ex.environment(source['snapshot_root'],fds)
        atomic_json(root/'environment.json',env)
        GiB=1024**3
        cfg=dict(schema='rep1-bt-gt-run-v1',run_id=run_id,run_root=str(root),source_snapshot=source['snapshot_root'],
                 source_commit=commit,source_digest=source['source_digest'],input_manifest=str(root/'configs/inputs_manifest.json'),
                 input_digest=digest(inputs),environment_digest=digest(env),plan_sha256=ex.PLAN_SHA,prompt_sha256=ex.PROMPT_SHA,
                 python=ex.PYTHON,rscript=ex.RSCRIPT,r_libs_user=str(ex.R_LIBRARY),r_libs_site='',node_order=list(NODE_ORDER),
                 no_ai_dependency=True,auto_ai_wakeup=False,results_root='/home/zwan0726/work/outputs/bidcell_historical_dual_bt_gt_fusion_20261017',
                 resources=dict(retained_bytes=dict(zip(NODE_ORDER,[GiB,3*GiB//2,3*GiB//2,0])),shared_bytes=GiB,
                                temporary_peak_bytes=5*GiB,reserve_bytes=15*GiB,ram_allowance_bytes=20*GiB,ram_reserve_bytes=6*GiB,
                                heartbeat_seconds=10,stale_seconds=60,no_progress_seconds=900),
                 estimates=dict(existing_BT_seconds=[600,1800],two_GT_matrix_seconds=[1800,5400],
                                future_evaluation_publication_seconds=[5400,10800],retained_GiB=[3,6],
                                evidence='historical full matrix monitors, change_plan_resource_evidence, actual small science/runtime fixtures; engineering bounds, not new full-run timings'))
        atomic_json(root/'configs/run.json',cfg);atomic_json(root/'state.json',state_new(cfg))
        forecast=resource_forecast(cfg);atomic_json(root/'admission/resource_forecast.json',forecast)
        ex.verify_snapshot(cfg,False)
        if not forecast['admitted']:raise RuntimeError('resource admission failed: '+json.dumps(forecast))
        event(cfg,'PREPARED',source_commit=commit)
    return dict(status='PREPARED',config=str(root/'configs/run.json'),source_digest=cfg['source_digest'],resources=forecast)


def verify_execution(cfg,fds=(),inputs=False):
    if fds:check_fds(cfg,fds)
    origin=ex.verify_snapshot(cfg)
    ex.verify_contracts(cfg);admission=ex.verify_admission(cfg)
    ex.verify_environment(cfg,fds)
    if inputs:
        from .inputs import load_inputs
        load_inputs(cfg)
    return dict(**origin,admission=admission,python=sys.executable,environment_digest=cfg['environment_digest'])


def validate(config,admission):
    cfg=load_config(config)
    if Path(admission).resolve()!=Path(cfg['run_root'])/'admission/final.json':raise ValueError('admission must be final run-specific receipt')
    with ExitStack() as stack:
        fds=tuple(stack.enter_context(lock(p)) for p in paths(cfg))
        origin=verify_execution(cfg,fds,True);state=bound_state(cfg)
        if state['status'] not in ('PREPARED','ADMITTED'):raise ValueError('validate cannot reset an executed attempt')
        state.update(status='ADMITTED',phase='ADMISSION',execution=origin)
        atomic_json(within(cfg['run_root'],'state.json'),state)
        atomic_json(within(cfg['run_root'],'admission/validated.json'),dict(status='succeeded',config_digest=digest(cfg),
                    admission=identity(admission),source_manifest=identity(Path(cfg['run_root'])/'source_manifest.json')))
    return dict(status='ADMITTED',config_digest=digest(cfg))


class StateWriter:
    def __init__(self,cfg,state):
        self.cfg=cfg;self.state=state;self.mutex=threading.RLock();self.done=threading.Event();self.thread=None;self.error=None
    def edit(self,callback):
        with self.mutex:
            callback(self.state);atomic_json(within(self.cfg['run_root'],'state.json'),self.state)
    def update(self,**values):self.edit(lambda s:s.update(values))
    def tick(self):
        def apply(s):
            s['heartbeat_utc']=utc();s['heartbeat_sequence']+=1
            p=within(self.cfg['run_root'],'control',s['active_attempt']+'_progress.json')
            if p.exists():
                r=read(p)
                if r.get('attempt')==s['active_attempt'] and r.get('config_digest')==digest(self.cfg) and r.get('worker')==s.get('worker'):
                    for source,dest,last_key in (('progress_sequence','progress_sequence','worker_report_sequence'),
                                                ('new_scientific_progress_sequence','new_scientific_progress_sequence','worker_advance_sequence')):
                        previous=s.get(last_key,0);s[dest]+=max(0,r[source]-previous);s[last_key]=r[source]
                    for k in ('last_progress_utc','progress','committed_witness'):
                        if r.get(k) is not None:s[k]=r[k]
        self.edit(apply)
    def start(self):
        self.tick()
        def run():
            try:
                while not self.done.wait(self.cfg['resources']['heartbeat_seconds']):self.tick()
            except BaseException as exc:self.error=exc
        self.thread=threading.Thread(target=run,daemon=True);self.thread.start()
    def stop(self):
        self.done.set()
        if self.thread:self.thread.join(timeout=15)


def checkpoint_baseline(cfg):
    result={}
    for folder in ('bt_existing6','endpoints','evaluation8'):
        for p in within(cfg['run_root'],folder).rglob('*.json'):
            st=p.stat();result[str(p)]=(st.st_mtime_ns,st.st_size)
    return result


def new_checkpoint(cfg,baseline):
    witness=None
    for folder in ('bt_existing6','endpoints','evaluation8'):
        for p in within(cfg['run_root'],folder).rglob('*.json'):
            st=p.stat();key=(st.st_mtime_ns,st.st_size)
            if baseline.get(str(p))==key:continue
            baseline[str(p)]=key
            try:r=read(p)
            except (ValueError,OSError):continue
            if r.get('status')=='succeeded' and r.get('dependencies',{}).get('config_digest')==digest(cfg) and r.get('outputs'):
                check_receipt(p,r['dependencies'],cfg['run_root'])
                witness=identity(p)
            elif p.name.endswith('.rds.committed.json'):
                # R prediction commits deliberately use the scientific adapter's
                # identity format, rather than the general Python receipt schema.
                from . import NEW, N_CELLS
                relative=p.relative_to(Path(cfg['run_root']))
                if len(relative.parts)!=6 or relative.parts[0]!='endpoints' or relative.parts[1] not in NEW or relative.parts[2:5]!=('evaluation','paper_aligned_v1','prediction_chunks'):
                    raise ValueError('R prediction commit is outside the endpoint checkpoint namespace')
                target=Path(str(p)[:-len('.committed.json')])
                if Path(r['path']).resolve()!=target.resolve() or target.is_symlink():raise ValueError('R prediction commit target mismatch')
                grant=within(cfg['run_root'],'control/manual_resume.json');require_manual_resume(cfg,grant)
                deps=dependencies(cfg,relative.parts[1]+'_EVALUATION')
                deps.update(production=identity(within(cfg['run_root'],'endpoints',relative.parts[1],'production_complete.json')),manual_resume=identity(grant))
                if r.get('dependencies_digest')!=digest(deps):raise ValueError('R prediction commit endpoint dependency mismatch')
                if type(r.get('cursor')) is not int or type(r.get('total')) is not int or not 0<r['cursor']<=r['total']<=N_CELLS:
                    raise ValueError('R prediction commit cursor mismatch')
                check_identity(r);witness=identity(p)
    return witness


def worker(config,node,attempt,fds):
    cfg=load_config(config);state=bound_state(cfg)
    if state.get('active_attempt')!=attempt or state.get('current_node')!=node:raise PermissionError('worker lease mismatch')
    check_fds(cfg,fds);ex.install_import_guard(cfg)
    origin=verify_execution(cfg,fds,True);own=process_identity()
    import cv2
    cv2.setNumThreads(1)
    atomic_json(within(cfg['run_root'],'control',attempt+'_worker_ready.json'),dict(worker=own,node=node,execution=origin))
    baseline=checkpoint_baseline(cfg);seq=0;advance=0;last=None;witness=None
    def progress(value):
        nonlocal seq,advance,last,witness
        seq+=1;changed=new_checkpoint(cfg,baseline)
        if changed:advance+=1;last=utc();witness=changed
        ex.verify_snapshot(cfg);check_fds(cfg,fds)
        observed=bound_state(cfg)
        if observed.get('active_attempt')!=attempt or observed.get('status')=='FAILED':
            raise RuntimeError('supervisor lease failed after a committed checkpoint')
        atomic_json(within(cfg['run_root'],'control',attempt+'_progress.json'),dict(attempt=attempt,config_digest=digest(cfg),
                    worker=own,progress_sequence=seq,new_scientific_progress_sequence=advance,last_progress_utc=last,
                    progress=value,committed_witness=witness))
        if shutil.disk_usage(cfg['run_root']).free<cfg['resources']['reserve_bytes'] or memory()<cfg['resources']['ram_reserve_bytes']:
            raise RuntimeError('RESOURCE_LIMIT_AFTER_COMMITTED_CHECKPOINT; explicit new attempt required')
    guard=lambda:verify_execution(cfg,fds)
    try:
        if node in NODE_ORDER:
            from .science import run_stage
            result=run_stage(cfg,node,progress,fds,guard)
        elif node=='EVALUATE_NEW2':
            from .evaluation import evaluate
            grant=within(cfg['run_root'],'control/manual_resume.json');require_manual_resume(cfg,grant)
            result=evaluate(cfg,grant,progress,fds,guard)
        else:raise ValueError('worker node is outside authorized phases')
        verify_execution(cfg,fds)
        atomic_json(within(cfg['run_root'],'control',attempt+'_worker_exit.json'),dict(status='succeeded',node=node,worker=own,
                    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,execution=ex.verify_snapshot(cfg),
                    result_digest=digest(result)))
    except BaseException as exc:
        atomic_json(within(cfg['run_root'],'control',attempt+'_worker_exit.json'),dict(status='failed',node=node,worker=own,
                    error=str(exc),traceback=traceback.format_exc()))
        raise


def worker_command(cfg,node,attempt,fds,config):
    return [ex.PYTHON,'-I','-S','-B',str(Path(cfg['source_snapshot'])/'scripts/rep1_bt_gt_fusion/bootstrap.py'),
            '_worker','--config',str(config),'--node',node,'--attempt',attempt,'--fds',','.join(map(str,fds))]


def supervisor_command(cfg,attempt,mode,config):
    return [ex.PYTHON,'-I','-S','-B',str(Path(cfg['source_snapshot'])/'scripts/rep1_bt_gt_fusion/bootstrap.py'),
            '_supervise','--config',str(config),'--attempt',attempt,'--mode',mode]


def supervise(config,attempt,mode='production'):
    cfg=load_config(config);ex.install_import_guard(cfg)
    with ExitStack() as stack:
        fds=tuple(stack.enter_context(wait_launch_lock(p) if i==0 else lock(p)) for i,p in enumerate(paths(cfg)))
        state=bound_state(cfg)
        if state.get('active_attempt')!=attempt or state['status']!='STARTING' or not state['attempts'] or state['attempts'][-1]['attempt']!=attempt or state['attempts'][-1]['mode']!=mode:
            raise PermissionError('supervisor lease/phase mismatch')
        state.update(supervisor=process_identity(),worker=None,status='STARTING',phase='STARTUP',failure=None)
        writer=StateWriter(cfg,state);writer.start()
        try:
            origin=verify_execution(cfg,fds,True);writer.update(execution=origin)
            if mode=='evaluation':require_manual_resume(cfg,within(cfg['run_root'],'control/manual_resume.json'))
            nodes=list(NODE_ORDER) if mode=='production' else ['EVALUATE_NEW2']
            from .science import validate_stage,receipt_path
            for node in nodes:
                if node in NODE_ORDER and receipt_path(cfg,node).exists():
                    rec=validate_stage(cfg,node)
                    writer.edit(lambda s:s['nodes'][node].update(status='validated_complete',receipt=identity(receipt_path(cfg,node)),reused=True))
                    event(cfg,'NODE_REUSED_VALIDATED',node=node);continue
                forecast=resource_forecast(cfg,state)
                if not forecast['admitted']:raise RuntimeError('RESOURCE_NOT_ADMITTED: '+json.dumps(forecast))
                phase='EXISTING_BT_EVALUATION' if node=='BT_EXISTING6' else 'PRODUCTION_VALIDATION' if node=='PRODUCTION_VALIDATE' else 'NEW_ENDPOINT_EVALUATION' if node=='EVALUATE_NEW2' else 'NEW_ENDPOINT_PRODUCTION'
                writer.update(status='RUNNING',current_node=node,phase=phase,worker=None,resource_forecast=forecast)
                if node in NODE_ORDER:writer.edit(lambda s:s['nodes'][node].update(status='running',started_utc=utc(),attempt=attempt))
                verify_execution(cfg,fds)
                command=worker_command(cfg,node,attempt,fds,config)
                logpath=within(cfg['run_root'],'logs',attempt+'_'+node+'.log')
                with logpath.open('a') as log:
                    proc=subprocess.Popen(command,cwd=cfg['source_snapshot'],env=child_environment(cfg),stdin=subprocess.DEVNULL,
                                          stdout=log,stderr=subprocess.STDOUT,pass_fds=fds,close_fds=True,start_new_session=True)
                    own=process_identity(proc.pid);writer.update(worker=own,worker_log=str(logpath),worker_report_sequence=0,worker_advance_sequence=0)
                    event(cfg,'NODE_STARTED',node=node,worker=own,argv=command,log=str(logpath))
                    while proc.poll() is None:
                        if writer.error:raise RuntimeError('heartbeat state writer failed: '+str(writer.error))
                        time.sleep(.5)
                    writer.tick();code=proc.returncode
                writer.edit(lambda s:s['child_exit_codes'].update({node:code}))
                event(cfg,'NODE_EXIT',node=node,exit_code=code,signal=-code if code<0 else None)
                if code:raise RuntimeError('child failed: '+node+' exit='+str(code)+' log='+str(logpath))
                exited=read(within(cfg['run_root'],'control',attempt+'_worker_exit.json'))
                if exited.get('status')!='succeeded' or exited.get('node')!=node or exited.get('worker')!=own:raise ValueError('worker completion/identity mismatch')
                if node in NODE_ORDER:
                    validate_stage(cfg,node)
                    writer.edit(lambda s:s['nodes'][node].update(status='validated_complete',ended_utc=utc(),exit_code=0,receipt=identity(receipt_path(cfg,node))))
                writer.update(worker=None)
            if mode=='production':
                validate_stage(cfg,'PRODUCTION_VALIDATE')
                writer.update(status=HOLD,phase='MANUAL_HOLD',current_node=None,worker=None,completed_utc=utc(),
                              production_receipt=identity(Path(cfg['run_root'])/'production_complete.json'))
            else:writer.update(status='EVALUATION_COMPLETE_AWAITING_PUBLICATION',phase='EVALUATION_COMPLETE',current_node=None,worker=None,completed_utc=utc())
            event(cfg,'QUEUE_NORMAL_EXIT',status=writer.state['status'])
        except BaseException as exc:
            writer.update(status='FAILED',failure=dict(reason=str(exc),traceback=traceback.format_exc(),time=utc()),ended_utc=utc())
            event(cfg,'QUEUE_FAILED',reason=str(exc));raise
        finally:writer.stop()


def launch(config,mode='production',resume=False):
    cfg=load_config(config);ex.install_import_guard(cfg)
    with ExitStack() as stack:
        fds=tuple(stack.enter_context(lock(p)) for p in paths(cfg))
        state=bound_state(cfg)
        if alive(state.get('supervisor')) or alive(state.get('worker')):raise RuntimeError('run process is already active')
        verify_execution(cfg,fds,inputs=True)
        validated=read(within(cfg['run_root'],'admission/validated.json'))
        check_identity(validated['admission']);check_identity(validated['source_manifest'])
        if validated['config_digest']!=digest(cfg):raise ValueError('validated admission differs from config')
        previous_mode=state['attempts'][-1]['mode'] if state['attempts'] else None
        if resume and state['status']=='STARTING' and state['attempts'] and (datetime.now(timezone.utc)-datetime.fromisoformat(state['attempts'][-1]['created_utc'])).total_seconds()<60:
            raise RuntimeError('a fresh detached dispatch is still starting')
        if mode=='production':
            if previous_mode=='evaluation':raise PermissionError('production cannot resume a manual evaluation attempt')
            if state['status']=='ADMITTED' and not resume:pass
            elif resume and previous_mode=='production' and state['status'] in ('FAILED','RUNNING','STARTING'):
                from .science import validate_stage,receipt_path
                for n in NODE_ORDER:
                    if state['nodes'][n]['status']=='validated_complete':validate_stage(cfg,n)
                    elif receipt_path(cfg,n).exists():validate_stage(cfg,n)
            else:raise ValueError('production launch state requires admission or explicit failed-attempt recovery')
        else:
            require_manual_resume(cfg,within(cfg['run_root'],'control/manual_resume.json'))
            if state['status']==HOLD and not resume:pass
            elif resume and previous_mode=='evaluation' and state['status'] in ('FAILED','RUNNING','STARTING'):pass
            else:raise ValueError('evaluation requires exited production hold or explicit same-phase resume-evaluation')
            from .science import validate_stage
            validate_stage(cfg,'PRODUCTION_VALIDATE')
        forecast=resource_forecast(cfg,state)
        if not forecast['admitted']:raise RuntimeError('launch resources not admitted')
        attempt=uuid.uuid4().hex
        state.update(status='STARTING',phase='STARTUP',active_attempt=attempt,supervisor=None,worker=None,heartbeat_utc=None,
                     heartbeat_sequence=0,progress_sequence=0,new_scientific_progress_sequence=0,last_progress_utc=None,committed_witness=None,
                     new_endpoint_evaluation_allowed=mode=='evaluation',publication_allowed=False,failure=None)
        state['attempts'].append(dict(attempt=attempt,mode=mode,created_utc=utc(),config_digest=digest(cfg)))
        atomic_json(within(cfg['run_root'],'state.json'),state)
        session='rep1-bt-gt-'+attempt[:12]
        argv=supervisor_command(cfg,attempt,mode,config)
        command='exec '+shlex.join(argv)+' >> '+shlex.quote(str(Path(cfg['run_root'])/'logs/queue.log'))+' 2>&1'
        tmux=['tmux','new-session','-d','-s',session,'-c',cfg['source_snapshot']]
        for key,value in controlled_environment(cfg).items():tmux+=['-e',key+'='+value]
        try:subprocess.run(tmux+[command],check=True,env=child_environment(cfg),stdin=subprocess.DEVNULL)
        except BaseException as exc:
            state.update(status='FAILED',failure=dict(reason='detached launch failed: '+str(exc),time=utc()))
            atomic_json(within(cfg['run_root'],'state.json'),state);raise
        event(cfg,'DETACHED_DISPATCHED',attempt=attempt,session=session,argv=argv)
    return dict(status='DISPATCHED_NOT_YET_VERIFIED',attempt=attempt,tmux_session=session,config=str(config))


def status(config):
    cfg=load_config(config);state=bound_state(cfg);s_alive=alive(state.get('supervisor'));w_alive=alive(state.get('worker'))
    held={str(p):lock_held(p) for p in paths(cfg)}
    age=None if not state.get('heartbeat_utc') else (datetime.now(timezone.utc)-datetime.fromisoformat(state['heartbeat_utc'])).total_seconds()
    attempt_age=(datetime.now(timezone.utc)-datetime.fromisoformat(state['attempts'][-1]['created_utc'])).total_seconds() if state.get('attempts') else 0
    missing_process=not s_alive and (state.get('supervisor') is not None or state['status']=='RUNNING' or attempt_age>60)
    stale=state['status'] in ('STARTING','RUNNING') and (missing_process or (age is not None and age>60))
    result=dict(state,disk_status=state['status'],observed_status='STALE' if stale else state['status'],
                supervisor_alive=s_alive,worker_alive=w_alive,locks=held,heartbeat_age_seconds=age)
    if not s_alive and any(held.values()):result['liveness']='PROCESS_EXITED_DESCENDANT_LOCK_HELD'
    progress_anchor=state.get('last_progress_utc') or (state['attempts'][-1]['created_utc'] if state.get('attempts') else None)
    if progress_anchor:
        elapsed=(datetime.now(timezone.utc)-datetime.fromisoformat(progress_anchor)).total_seconds()
        result['scientific_progress_age_seconds']=elapsed;result['needs_progress_inspection']=state['status']=='RUNNING' and elapsed>900
    return result


def complete_check(config):
    cfg=load_config(config)
    with ExitStack() as stack:
        fds=tuple(stack.enter_context(lock(p)) for p in paths(cfg))
        state=bound_state(cfg)
        if alive(state.get('supervisor')) or alive(state.get('worker')) or state['status'] not in (HOLD,'EVALUATION_COMPLETE_AWAITING_PUBLICATION','COMPLETED'):
            raise RuntimeError('production complete-check requires an exited validated queue')
        verify_execution(cfg,fds,True)
        from .science import validate_stage
        validate_stage(cfg,'PRODUCTION_VALIDATE')
        return dict(status='PRODUCTION_VALIDATED',production_receipt=identity(Path(cfg['run_root'])/'production_complete.json'),manual_resume_required=True)


def grant_resume(config,instruction_utc):
    cfg=load_config(config)
    instant=datetime.fromisoformat(instruction_utc)
    if instant.tzinfo is None or instant>datetime.now(timezone.utc):raise ValueError('user instruction needs a real past UTC/offset timestamp')
    with ExitStack() as stack:
        fds=tuple(stack.enter_context(lock(p)) for p in paths(cfg));state=bound_state(cfg)
        if state['status']!=HOLD or alive(state.get('supervisor')) or alive(state.get('worker')):raise RuntimeError('manual resume requires exited production hold')
        verify_execution(cfg,fds,True)
        from .science import validate_stage
        validate_stage(cfg,'PRODUCTION_VALIDATE')
        p=within(cfg['run_root'],'control/manual_resume.json')
        if p.exists():raise FileExistsError('manual grant already exists; verify it rather than replacing it')
        grant=dict(authorization='explicit_user_resume',user_instruction_utc=instruction_utc,created_utc=utc(),
                   dependencies=dependencies(cfg,'MANUAL_RESUME'),production_complete=identity(Path(cfg['run_root'])/'production_complete.json'))
        atomic_json(p,grant);return grant


def publish(config,grant):
    cfg=load_config(config)
    with ExitStack() as stack:
        fds=tuple(stack.enter_context(lock(p)) for p in paths(cfg));state=bound_state(cfg)
        if alive(state.get('supervisor')) or alive(state.get('worker')) or state['status'] not in ('EVALUATION_COMPLETE_AWAITING_PUBLICATION','COMPLETED'):
            raise RuntimeError('publication requires exited complete evaluation')
        verify_execution(cfg,fds,True);require_manual_resume(cfg,grant)
        from .context_sync import analysis_review
        analysis_review(cfg)
        from .publication import publish as scientific_publish
        result=scientific_publish(cfg,grant)
        from .context_sync import synchronize
        proof=synchronize(cfg,grant,result)
        state.update(status='COMPLETED',phase='REGISTERED',publication_allowed=True,context_sync=proof,completed_utc=utc())
        atomic_json(within(cfg['run_root'],'state.json'),state)
        return proof
