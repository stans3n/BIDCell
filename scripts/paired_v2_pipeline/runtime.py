"""Bounded, single-owner remote execution, with no AI or connection dependency."""
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
from .common import *


def initial_state(q):
    return dict(status='prepared_not_started',pipeline_config=identity(Path(q['root'])/'configs/pipeline.json'),
        created_utc=utc(),stages={t['id']:dict(status='queued',arm=t['arm'],stage=t['stage'],depends_on=t['depends_on'],
            progress=None,started_utc=None,finished_utc=None,exit_code=None,logs=[],attempts=[]) for t in q['tasks']})


def progress(q,task):
    if task['arm']:
        e=endpoint(task['arm']);paths={'train':e['root']/'checkpoints/progress.json','predict':e['prediction']/'progress.json',
             'postprocess':e['prediction']/'postprocess/progress.json','matrix':e['matrix']/'progress.json'}
        p=paths.get(task['stage'])
    else:p=Path(q['root'])/'frozen_strata/progress.json' if task['stage']=='freeze_inputs' else None
    return read(p) if p and p.exists() else None


def resource_sample(pid):
    pending=[pid];seen=set();rss=0;pss=0
    while pending:
        child=pending.pop()
        if child in seen:continue
        seen.add(child)
        try:
            pending.extend(map(int,Path(f'/proc/{child}/task/{child}/children').read_text().split()))
            status=Path(f'/proc/{child}/status').read_text()
            for line in status.splitlines():
                if line.startswith('VmRSS:'):rss+=int(line.split()[1])*1024
            for line in Path(f'/proc/{child}/smaps_rollup').read_text().splitlines():
                if line.startswith('Pss:'):pss+=int(line.split()[1])*1024
        except (OSError,ValueError):pass
    return dict(rss_sum_bytes=rss,pss_sum_bytes=pss,process_count=len(seen))


def terminate_owned(child,ident):
    if child.poll() is not None:return
    if not alive(ident):raise RuntimeError('refuse signal: child process identity changed')
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=20)
    except subprocess.TimeoutExpired:
        if alive(ident):os.killpg(child.pid,signal.SIGKILL)
        child.wait(timeout=10)


def resume_training_valid(q,task):
    """Read-only validation of a durable formal snapshot, before one timeout retry."""
    import torch
    from scripts.paired_v2.training import checkpoint_header,validate_cursor
    from scripts.paired_v2.common import make_model,optimizer
    cfg=load_config(config_path(task['arm']));p=endpoint(task['arm'])['root']/'checkpoints/latest.pth'
    if not p.exists():return False
    state=torch.load(p,map_location='cpu')
    if state['pilot'] or state['header']!=checkpoint_header(cfg) or state['sequence_digest']!=digest(state['effective_sequence']):return False
    validate_cursor(state)
    if not 0<state['actual_updates']<=3625 or not state['bn_in_model_state'] or set(state['rng'])!={'python','numpy','torch','cuda','imgaug'}:return False
    model=make_model(cfg['model_kind'],'cpu');model.load_state_dict(state['model'],strict=True)
    opt=optimizer(model);opt.load_state_dict(state['optimizer'])
    if not state['optimizer']['state']:return False
    for s in state['optimizer']['state'].values():
        if set(s)!={'step','exp_avg','exp_avg_sq'} or int(s['step'])!=state['actual_updates']:return False
        if any(not torch.isfinite(v).all() for v in s.values()):return False
    if any(not torch.isfinite(v).all() for v in state['model'].values()):return False
    log=endpoint(task['arm'])['root']/'effective_sequence.jsonl'
    if not log.exists():return False
    lines=log.read_text().splitlines();rows=state['effective_sequence']
    if len(lines)<len(rows):return False
    if any(json.loads(lines[i])!=r for i,r in enumerate(rows)):return False
    # A partial/uncommitted suffix may be repaired only by the reviewed gateway.
    return True


def retry_allowed(q,task,code,attempt):
    if code!=124 or attempt>=q['max_retries'] or task['stage']!='train':return False
    if (Path(q['root'])/'worker_errors'/f"{task['id']}_attempt{attempt}.json").exists():return False
    try:return resume_training_valid(q,task)
    except (ValueError,KeyError,OSError,RuntimeError):return False


def worker(root,task_id,attempt):
    q=queue(root);task=next(t for t in q['tasks'] if t['id']==task_id)
    gpu=None
    try:
        verify_execution(q,admitted=True);require_space(q)
        if task['stage'] in ('train','predict'):gpu=lock(q['gpu_lock'])
        environment_check(q)
        from .stages import execute
        result=execute(q,task)
        # Worker completion is accepted again by the parent before its commit.
        print(json.dumps(dict(status='worker_succeeded',task=task_id,semantic=result)),flush=True)
    except BaseException as error:
        atomic_json(Path(q['root'])/'worker_errors'/f'{task_id}_attempt{attempt}.json',
            dict(status='failed',task=task_id,exception=type(error).__name__,reason=str(error),traceback=traceback.format_exc(),
                 classification='protocol_numeric_or_resource_stop_no_automatic_retry',time=utc()))
        raise
    finally:
        if gpu:gpu.close()


def semantic(q,task):
    if task['arm']:
        from scripts.paired_v2.stages import validate_stage
        return validate_stage(config_path(task['arm']),task['stage'],require_bound=True)
    from .stages import validate_custom
    return validate_custom(q,task['stage'])


def run(root):
    q=queue(root);guard=lock(Path(root)/'pipeline_run.lock');child=None;child_id=None
    state_path=Path(root)/'pipeline_status.json';state=read(state_path)
    if state['pipeline_config']!=identity(Path(root)/'configs/pipeline.json'):raise ValueError('persisted state belongs to different execution')
    def stopped(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
    try:
        verify_execution(q,admitted=True);require_space(q,launch=True);environment_check(q)
        state.update(status='running',process=process_identity(),started_utc=utc(),updated_utc=utc(),failure=None)
        atomic_json(state_path,state)
        for task in q['tasks']:
            key=task['id'];entry=state['stages'][key];path=own_receipt(q,key)
            verify_execution(q,admitted=True);require_space(q)
            if path.exists():
                accepted=validate_task(q,task);entry.update(status='completed',receipt=identity(path),semantic=accepted['semantic'],reuse='identity_and_semantics_verified')
                atomic_json(state_path,state);continue
            # Persist a pre-execution dependency identity; it bridges worker/parent crashes.
            dep=own_dependencies(q,task);pending=Path(root)/'pipeline_pending'/f'{key}.json'
            if pending.exists() and read(pending)['dependencies']!=dep:raise ValueError('pending stage lineage changed: '+key)
            if not pending.exists():atomic_json(pending,dict(status='pending',dependencies=dep))
            attempts=entry.get('attempts',[])
            # Bridge a worker success/parent commit crash without spending another attempt.
            completed=None
            if task['arm']:
                cp=endpoint(task['arm'])['root']/'stages'/f"{task['stage']}.json"
                if cp.exists():completed=semantic(q,task)
            else:
                folder={'freeze_inputs':'frozen_strata','freeze_counts':'frozen_counts','paired_training':'paired_training','reports':'results','package':'result_pack'}[task['stage']]
                if (Path(root)/folder/'complete.json').exists():completed=semantic(q,task)
            if completed is not None:
                outputs={'native_bound_receipt':identity(cp)} if task['arm'] else {'custom_completion':completed['receipt']}
                atomic_json(path,dict(status='completed',dependencies=dep,semantic=completed,attempts=attempts,outputs=outputs,recovered_worker_completion=True))
                validate_task(q,task);entry.update(status='completed',finished_utc=utc(),receipt=identity(path),semantic=completed,reuse='verified_worker_completion')
                atomic_json(state_path,state);continue
            for attempt in range(len(attempts),q['max_retries']+1):
                logpath=Path(root)/'logs'/f'pipeline_{key}_attempt{attempt}.log'
                entry.update(status='running',started_utc=utc(),attempt=attempt,exit_code=None)
                entry['logs'].append(str(logpath));state.update(current=key,updated_utc=utc());atomic_json(state_path,state)
                env=numeric_env();env['CUDA_VISIBLE_DEVICES']='0' if task['stage'] in ('train','predict') else ''
                begin=time.monotonic();peak_rss=peak_pss=0;timed_out=False
                with logpath.open('a') as stream:
                    child=subprocess.Popen([q['python'],'-B','-m','scripts.paired_v2_pipeline.cli','worker','--root',str(root),'--task',key,'--attempt',str(attempt)],
                        cwd=q['repo'],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env=env)
                    child_id=process_identity(child.pid);entry['child']=child_id;atomic_json(state_path,state)
                    while child.poll() is None:
                        sample=resource_sample(child.pid);peak_rss=max(peak_rss,sample['rss_sum_bytes']);peak_pss=max(peak_pss,sample['pss_sum_bytes'])
                        entry.update(progress=progress(q,task),resources=dict(peak_RSS_sum_bytes=peak_rss,peak_PSS_sum_bytes=peak_pss,scope='sampled owned process tree; shared RSS can double count'),elapsed_seconds=time.monotonic()-begin)
                        state['updated_utc']=utc();atomic_json(state_path,state)
                        if time.monotonic()-begin>q['stage_timeouts'][task['stage']]:
                            terminate_owned(child,child_id);timed_out=True;break
                        time.sleep(2)
                    code=124 if timed_out else child.returncode
                entry['exit_code']=code
                item=dict(attempt=attempt,exit_code=code,wall_seconds=time.monotonic()-begin,log=identity(logpath),
                    peak_RSS_sum_bytes=peak_rss,peak_PSS_sum_bytes=peak_pss,temporary_timeout=timed_out)
                attempts.append(item);entry['attempts']=attempts
                if code==0:
                    accepted=semantic(q,task)
                    outputs={'native_bound_receipt':identity(endpoint(task['arm'])['root']/'stages'/f"{task['stage']}.json")} if task['arm'] else {'custom_completion':accepted['receipt']}
                    rec=dict(status='completed',dependencies=dep,semantic=accepted,attempts=attempts,outputs=outputs)
                    atomic_json(path,rec);validate_task(q,task)
                    entry.update(status='completed',finished_utc=utc(),receipt=identity(path),semantic=accepted,progress=progress(q,task));atomic_json(state_path,state);break
                entry.update(status='failed',finished_utc=utc(),failure_reason='bounded timeout' if timed_out else 'worker failed; inspect error receipt/log')
                atomic_json(state_path,state)
                if retry_allowed(q,task,code,attempt):
                    entry['retry_reason']='one parent timeout recovery; durable complete model/BN/Adam/RNG/cursor and committed sequence verified'
                    atomic_json(state_path,state);continue
                raise RuntimeError(f'{key} failed with exit {code}; {logpath}')
            else:raise RuntimeError('attempt budget exhausted; manual resume needs explicit repair/admission')
        for task in q['tasks']:validate_task(q,task)
        cost=dict(status='all_stages_completed',started_utc=state['started_utc'],finished_utc=utc(),
            stages={k:dict(attempts=e['attempts'],started_utc=e['started_utc'],finished_utc=e['finished_utc'],logs=e['logs']) for k,e in state['stages'].items()})
        atomic_json(Path(root)/'runtime_cost.json',cost)
        atomic_json(Path(root)/'pipeline_complete.json',dict(status='completed_awaiting_ai_resume',pipeline_config=identity(Path(root)/'configs/pipeline.json'),
            source_bundle=q['source_bundle'],stage_receipts={t['id']:identity(own_receipt(q,t['id'])) for t in q['tasks']},
            manifest=identity(Path(root)/'result_pack/manifest.json'),runtime_cost=identity(Path(root)/'runtime_cost.json'),finished_utc=utc()))
        state.update(status='completed_awaiting_ai_resume',finished_utc=utc(),current=None,completion=identity(Path(root)/'pipeline_complete.json'));atomic_json(state_path,state)
    except BaseException:
        key=state.get('current')
        if key and state['stages'][key]['status']!='completed':
            state['stages'][key].update(status='failed',finished_utc=utc(),failure_reason=traceback.format_exc())
        state.update(status='failed',failure=traceback.format_exc(),updated_utc=utc());atomic_json(state_path,state);raise
    finally:
        if child is not None and child.poll() is None:terminate_owned(child,child_id)
        guard.close()


def launch(root):
    import shlex
    q=queue(root);verify_execution(q,admitted=True);require_space(q,launch=True);environment_check(q)
    launch_guard=lock(Path(root)/'pipeline_launch.lock')
    try:
        probe=lock(Path(root)/'pipeline_run.lock');probe.close()
        old=read(Path(root)/'pipeline_status.json')
        if old.get('process') and alive(old['process']):raise RuntimeError('matching runner already active')
        if any(alive(e.get('child',{})) for e in old['stages'].values()):raise RuntimeError('matching worker still active; do not launch a duplicate parent/worker')
        socket_name=q['tmux_socket'];session=q['tmux_session']
        exists=subprocess.run(['tmux','-L',socket_name,'has-session','-t',session],capture_output=True)
        if exists.returncode==0:raise RuntimeError('private hosting session already exists; do not duplicate')
        command=shlex.join([q['python'],'-B','-m','scripts.paired_v2_pipeline.cli','run','--root',str(root)])
        command+=' >> '+shlex.quote(str(Path(root)/'logs/pipeline_runner.log'))+' 2>&1'
        env=numeric_env();env['CUDA_VISIBLE_DEVICES']=''
        launch_mark=utc()
        subprocess.run(['tmux','-L',socket_name,'new-session','-d','-s',session,'-c',q['repo'],command],check=True,env=env)
        deadline=time.monotonic()+45
        while time.monotonic()<deadline:
            state=read(Path(root)/'pipeline_status.json')
            if state.get('status')=='running' and alive(state.get('process',{})):break
            if state.get('status')=='failed' and state.get('updated_utc','')>=launch_mark:raise RuntimeError('runner startup failed; inspect pipeline_runner.log')
            time.sleep(.25)
        else:raise RuntimeError('startup handshake not observed; inspect private tmux session before retry')
        receipt=dict(status='launched_verified_runner_identity',process=state['process'],source_bundle=q['source_bundle'],
            pipeline_config=identity(Path(root)/'configs/pipeline.json'),review=identity(q['review_receipt']),tmux_socket=socket_name,tmux_session=session,
            server_pid=subprocess.check_output(['tmux','-L',socket_name,'display-message','-p','-t',session,'#{pid}'],text=True).strip(),launched_utc=utc())
        atomic_json(Path(root)/'pipeline_launch.json',receipt);return receipt
    finally:launch_guard.close()


def status(root):
    q=queue(root);state=read(Path(root)/'pipeline_status.json')
    state['runner_alive']=alive(state.get('process',{}));state['current_progress']={t['id']:progress(q,t) for t in q['tasks'] if t['arm'] or t['stage']=='freeze_inputs'}
    current=state.get('current');state['child_alive']=alive(state['stages'][current].get('child',{})) if current else False
    state['free_disk_bytes']=shutil.disk_usage(root).free
    return state


def train_status(root,arm=None):
    q=queue(root);state=read(Path(root)/'pipeline_status.json');result={}
    for a in [arm] if arm else RUNS:
        entry=state['stages'][a+'_train'];p=endpoint(a)['root']/'checkpoints/progress.json';rec=endpoint(a)['root']/'checkpoints/complete.json'
        result[a]=dict(state=entry['status'],actual_updates=read(p)['actual_updates'] if p.exists() else 0,
            target_updates=3625,progress=read(p) if p.exists() else None,completion_receipt=str(rec),logs=entry['logs'],
            accepted=entry['status']=='completed' and own_receipt(q,a+'_train').exists())
    return result


def completion_check(root,trains_only=False):
    q=queue(root);verify_execution(q,admitted=True);tasks=[t for t in q['tasks'] if t['stage']=='train'] if trains_only else q['tasks']
    checked={};missing=[]
    for task in tasks:
        if not own_receipt(q,task['id']).exists():missing.append(task['id'])
        else:checked[task['id']]=validate_task(q,task)['semantic']
    if missing:return dict(complete=False,kind='two_trainings' if trains_only else 'entire_compute_and_evaluate',missing=missing,checked=checked)
    if not trains_only:
        p=Path(root)/'pipeline_complete.json'
        if not p.exists():return dict(complete=False,kind='entire_compute_and_evaluate',missing=['final_parent_acceptance'])
        rec=read(p)
        if rec['status']!='completed_awaiting_ai_resume' or rec['pipeline_config']!=identity(Path(root)/'configs/pipeline.json') or rec['source_bundle']!=q['source_bundle']:
            raise ValueError('final parent completion identity drift')
        for obj in [*rec['stage_receipts'].values(),rec['manifest'],rec['runtime_cost']]:check_obj(obj)
    return dict(complete=True,kind='two_trainings' if trains_only else 'entire_compute_and_evaluate',checked=checked,
        scientific_review='awaiting manual AI resume')


def tail(root,lines=30):
    state=status(root);paths=[Path(root)/'logs/pipeline_runner.log'];key=state.get('current')
    if key:paths.extend(map(Path,state['stages'][key]['logs'][-1:]))
    return {str(p):'\n'.join(p.read_text().splitlines()[-lines:]) if p.exists() else '(not created)' for p in paths}
