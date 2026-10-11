"""Detached, single-GPU COS queue; finite retries and independent-arm isolation."""
import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
from .evaluation import ARMS,read,times
from scripts.crgf.common import atomic_json,identity,sha256,digest,verify_outputs,process_identity
from scripts.crgf.run_egr import is_alive,process_group_memory

TIMEOUTS={'train':21600,'predict':43200,'postprocess':28800,'matrix':3600,'evaluate':14400,'report':14400}


def validate_queue(cfg):
    required={'run_id','output_root','python','methods','paper_protocol','prepared','reference','engineering_protocol','anchors','gpu_lock','minimum_free_gb','cpu_workers','stage_timeouts','retry_backoff_seconds','admission'}
    allowed=required|{'diagnostics_inputs','mini_test','testing_poll_fast','historical_root','historical_methods','population','review_receipt','resource_estimate','version','arm_order','historical_ssp_root'}
    if set(cfg)-allowed or required-set(cfg):raise ValueError('queue schema unknown/missing fields: '+str((set(cfg)-allowed,required-set(cfg))))
    if set(cfg['methods'])!=set(ARMS) or set(cfg['anchors'])!={'B1'}:raise ValueError('queue arm/anchor identity set mismatch')
    if set(cfg['stage_timeouts'])!=set(TIMEOUTS) or any(v<=0 for v in cfg['stage_timeouts'].values()):raise ValueError('explicit conservative stage watchdogs required')
    if cfg.get('testing_poll_fast') and not cfg.get('mini_test'):raise ValueError('testing poll override forbidden in formal queue')
    if cfg['minimum_free_gb']<(0 if cfg.get('mini_test') else 15) or not 1<=cfg['cpu_workers']<=4 or cfg['retry_backoff_seconds']<0:raise ValueError('invalid runner resource settings')
    for name,spec in cfg['methods'].items():
        required_method={'config','worktree','commit','machine','matrix','evaluation','cost_root'}
        if not required_method<=set(spec) or set(spec)-required_method-{'endpoints','base_variant','run_status'}:raise ValueError('method schema mismatch '+name)
    return cfg


def stage_specs(cfg):
    root=Path(cfg['output_root']);specs=[]
    for arm in ARMS:
        s=cfg['methods'][arm];r=Path(s['cost_root']);previous=[]
        paths={'train':r/'checkpoints'/arm/'complete.json','predict':r/'predictions'/arm/'complete.json','postprocess':r/'predictions'/arm/'postprocess/complete.json','matrix':Path(s['matrix'])/'complete.json','evaluate':Path(s['evaluation'])/'complete.json'}
        if arm in ('P1','P2'):
            paths.pop('train');paths.pop('predict')
        for stage,path in paths.items():
            name=arm+'_'+stage;specs.append(dict(name=name,method=arm,stage=stage,dependencies=previous,receipt=str(path)));previous=[name]
    specs.append(dict(name='report',method=None,stage='report',comparison='BS4',dependencies=[a+'_evaluate' for a in ARMS],receipt=str(root/'comparison/complete.json')))
    return specs


def source_map(checkout):
    checkout=Path(checkout)
    names=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=checkout,timeout=30).decode().split('\0')
    return {name:sha256(checkout/name) for name in sorted(set(names)) if name and (checkout/name).is_file()}


def freeze(config):
    cfg=read(config);root=Path(cfg['output_root']);root.mkdir(parents=True,exist_ok=True)
    with (root/'runner.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        frozen={s['worktree']:dict(files=source_map(s['worktree']),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=s['worktree'],text=True,timeout=30).strip(),tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=s['worktree'],text=True,timeout=30).strip()) for s in cfg['methods'].values()}
        path=root/'source/frozen.json'
        if path.exists():
            prior=read(path)
            if prior['worktrees']!=frozen:raise ValueError('freeze changed; explicit new attempt/output root required')
        atomic_json(path,dict(version='BS4-source-freeze-v1',run_id=cfg['run_id'],queue=identity(config),worktrees=frozen,**times()))


def preflight(config,cfg):
    validate_queue(cfg)
    root=Path(cfg['output_root']);admission=read(cfg.get('admission',root/'admission.json'))
    if not admission.get('approved_to_launch') or admission.get('required_failures'):raise ValueError('admission has not approved launch')
    if admission.get('run_id')!=cfg['run_id']:raise ValueError('admission run identity mismatch')
    if not cfg.get('mini_test') and not isinstance(admission.get('resources',{}).get('estimated_peak_gpu_allocator_bytes'),(int,float)):raise ValueError('admission missing GPU capacity estimate')
    frozen=read(root/'source/frozen.json')
    if not cfg.get('mini_test'):
        if admission['frozen_source']['sha256']!=sha256(root/'source/frozen.json') or admission['queue']['sha256']!=sha256(config):raise ValueError('admission source/queue identity mismatch')
        for arm,spec in cfg['methods'].items():
            if read(spec['config'])!=admission['resolved_configs'][arm]:raise ValueError('arm config changed after admission '+arm)
            arm_frozen=Path(spec['cost_root'])/'configs/frozen.json'
            if sha256(arm_frozen)!=admission['arm_frozen'][arm]['sha256']:raise ValueError('arm frozen manifest changed after admission '+arm)
            for obj in read(arm_frozen)['identities'].values():
                if sha256(obj['path'])!=obj['sha256']:raise ValueError('frozen scientific input changed '+obj['path'])
    if sha256(config)!=frozen['queue']['sha256']:raise ValueError('queue configuration changed after freeze')
    if set(cfg['methods'])!=set(ARMS):raise ValueError('BS4 arm set mismatch')
    for arm,s in cfg['methods'].items():
        actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=s['worktree'],text=True).strip()
        if actual!=s['commit'] or actual!=frozen['worktrees'][s['worktree']]['commit']:raise ValueError('worktree commit mismatch '+arm)
        if source_map(s['worktree'])!=frozen['worktrees'][s['worktree']]['files']:raise ValueError('source changed after freeze '+arm)
    for anchor in cfg['anchors'].values():
        if sha256(anchor['path'])!=anchor['sha256']:raise ValueError('historical anchor changed')
    verify_outputs(read(Path(cfg['prepared'])/'complete.json'))
    from .entry import verify_review
    verify_review(config,cfg)


def command_for(cfg,spec,config):
    python=cfg.get('python',sys.executable)
    if spec['method']:
        record=cfg['methods'][spec['method']]
        return [python,'-u','-m','scripts.bs4.stage','--config',record['config'],'--stage',spec['stage']],record['worktree']
    checkout=cfg['methods'][ARMS[0]]['worktree']
    if spec['stage']=='b_cost':return [python,'-u','-m','scripts.bs4.b_cost','--config',str(config)],checkout
    return [python,'-u','-m','scripts.bs4.evaluation','--config',str(config),'--comparison',spec['comparison']],checkout


def accept(stage,artifact,cfg=None,spec=None):
    from .validation import validate_stage
    return validate_stage(stage,artifact,cfg,spec)


def recoverable(returncode,text):
    prohibited=['out of memory','nan','nonfinite','hash','mismatch','conservation','schema','candidate','no space left']
    if any(s in text.lower() for s in prohibited):return False
    return returncode in [-signal.SIGTERM,-signal.SIGHUP,128+signal.SIGTERM,128+signal.SIGHUP] or any(s in text for s in ['[Errno 5] Input/output error','[Errno 116] Stale file handle','[Errno 110] Connection timed out'])


def signature(cfg,spec,state,config):
    frozen=read(Path(cfg['output_root'])/'source/frozen.json')
    if frozen.get('queue') and sha256(config)!=frozen['queue']['sha256']:raise ValueError('global queue configuration changed after admission')
    names=[spec['method']] if spec['method'] else ARMS
    selected={m:cfg['methods'][m] for m in names};selected_worktrees={v['worktree'] for v in selected.values()}
    for worktree in selected_worktrees:
        if source_map(worktree)!=frozen['worktrees'][worktree]['files']:raise ValueError('source changed during run; explicit new version required')
    deps={d:state['stages'][d].get('receipt_sha256') for d in spec['dependencies']}
    configs={m:sha256(cfg['methods'][m]['config']) for m in names}
    return digest(dict(config=sha256(config),frozen={w:frozen['worktrees'][w] for w in selected_worktrees},selected=selected,arm_configs=configs,stage=spec['stage'],deps=deps))


def blocked_receipt(cfg,spec,reason,status='blocked_dependency'):
    if spec['stage'] != 'report':return
    path=Path(spec['receipt']);path.parent.mkdir(parents=True,exist_ok=True)
    value=dict(status=status,ready=False,run_id=cfg['run_id'],reason=reason,**times())
    atomic_json(path,value)
    atomic_json(path.parent/'compute_summary.json',value)


def complete_receipt(cfg,state):
    root=Path(cfg['output_root'])
    for name,record in state['stages'].items():
        if record['status']!='succeeded':continue
        try:
            if sha256(record['receipt'])!=record['receipt_sha256']:raise ValueError('accepted stage receipt changed')
            accept(record['stage'],read(record['receipt']),cfg,record)
        except (OSError,ValueError,KeyError) as exc:record.update(status='failed',failure='final artifact validation: '+str(exc))
    for spec in stage_specs(cfg):
        if any(state['stages'][d]['status']!='succeeded' for d in spec['dependencies']):
            state['stages'][spec['name']].update(status='blocked_dependency',reason='final prerequisite verification failed');blocked_receipt(cfg,spec,'final prerequisite verification failed')
    seen=set();logical=allocated=0
    for artifact in root.rglob('*'):
        if not artifact.is_file():continue
        stat=artifact.lstat();key=(stat.st_dev,stat.st_ino)
        if key in seen:continue
        seen.add(key);logical+=stat.st_size;allocated+=stat.st_blocks*512
    storage=dict(logical_bytes_unique_inodes=logical,allocated_bytes_unique_inodes=allocated,free_bytes=shutil.disk_usage(root).free,scope='run output tree including attempts; symlink targets excluded; snapshot before completion receipt publication')
    missing=[k for k,v in state['stages'].items() if v['status']!='succeeded']
    artifacts={k:identity(v['receipt']) for k,v in state['stages'].items() if v['status']=='succeeded'}
    result=dict(version='BS4-compute-receipt-v1',run_id=cfg['run_id'],storage=storage,status='succeeded' if not missing else 'partial',next_state='compute_complete_awaiting_user_resume' if not missing else 'requires_repair',required_missing=missing,artifacts=artifacts,failed_attempts=[dict(stage=k,attempt=a) for k,v in state['stages'].items() for a in v['attempts'] if a.get('exit_code',0)!=0 or a.get('artifact_failure')],total_stage_wall_seconds=sum(a.get('wall_seconds',0) for v in state['stages'].values() for a in v['attempts']),wall_time_lower_bound_attempts=[dict(stage=k,attempt=i+1) for k,v in state['stages'].items() for i,a in enumerate(v['attempts']) if a.get('wall_seconds_is_lower_bound')],total_stage_wall_scope='sum of observed stage elapsed times; interrupted attempts use last persisted heartbeat lower bounds, with unknown tails explicitly listed',ai_call_time='not recorded by compute runner',**times())
    if not missing:
        shutil.copyfile(root/'comparison/metric_coverage.csv',root/'metric_coverage.csv') if (root/'comparison/metric_coverage.csv').exists() else None
    atomic_json(root/'compute_complete.json',result)
    (root/'artifact_index.md').write_text('# BS4 artifacts\n\n'+'\n'.join(f'- [{k}]({v["receipt"]}) — {v["status"]}' for k,v in state['stages'].items())+'\n')
    (root/'remaining_issues.md').write_text('# Remaining issues\n\n'+('\n'.join('- '+k+': '+state['stages'][k].get('failure',state['stages'][k].get('reason','unfinished')) for k in missing) if missing else 'Compute complete. Independent scientific analysis awaits the user resume.')+'\n')
    return not missing


def update_eta(state,cfg=None):
    defaults={'train':1555.9,'predict':4490.7,'postprocess':3375.8,'matrix':206.3,'evaluate':2230.9,'report':3600}
    observed={}
    for rec in state['stages'].values():
        if rec['status']=='succeeded':
            observed.setdefault(rec['stage'],[]).append(sum(a.get('wall_seconds',0) for a in rec.get('attempts',[])))
    remaining=0
    for rec in state['stages'].values():
        if rec['status'] not in ('pending','running','paused_resource','interrupted'):continue
        values=observed.get(rec['stage'])
        measured={}
        if cfg and Path(cfg.get('resource_estimate','')).is_file():measured=read(cfg['resource_estimate']).get('measured_stage_estimates',{}).get(rec.get('method'),{})
        estimate=measured.get(rec['stage'],sum(values)/len(values) if values else defaults[rec['stage']])
        remaining+=max(0,estimate-(rec.get('wall_seconds_current_attempt',0) if rec['status']=='running' else 0))
    now=time.time();deadline=state.get('first_started_epoch',now)+589.06*60
    state['eta_sydney']=datetime.fromtimestamp(now+remaining,timezone.utc).astimezone(ZoneInfo('Australia/Sydney')).isoformat()
    state['target_deadline_sydney']=datetime.fromtimestamp(deadline,timezone.utc).astimezone(ZoneInfo('Australia/Sydney')).isoformat()
    state['forecast_target_miss']=now+remaining>deadline
    state['target_missed']=now>deadline
    state.setdefault('initial_forecast_sydney',state['eta_sydney'])
    state['eta_basis']='arm-specific measured 32-step and real mini double-shift throughput with 20 percent allowance; completed stages or historical CPU stage estimates as fallback; regional density extrapolation uncertain'
    state['estimated_remaining_seconds']=remaining


def run(config):
    config=str(Path(config).resolve());cfg=read(config);root=Path(cfg['output_root']);root.mkdir(parents=True,exist_ok=True)
    with ExitStack() as stack:
        locks=[]
        for lockpath in [root/'runner.lock',Path(cfg['gpu_lock']),root.parent/'bs4_heavy.lock']:
            lockpath.parent.mkdir(parents=True,exist_ok=True);stream=stack.enter_context(lockpath.open('a+'));fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(stream)
            stream.seek(0);stream.truncate();json.dump(dict(run_id=cfg['run_id'],**process_identity(),**times()),stream);stream.flush();os.fsync(stream.fileno())
        path=root/'run_status.json';state=read(path) if path.exists() else dict(run_id=cfg['run_id'],stages={})
        if state['run_id']!=cfg['run_id']:raise ValueError('existing run identity mismatch')
        if state.get('pid_namespace') and state['pid_namespace']!=os.readlink('/proc/self/ns/pid') and state.get('runner',{}).get('boot_id')==Path('/proc/sys/kernel/random/boot_id').read_text().strip():raise RuntimeError('Host PID namespace required for resume')
        for name,r in state['stages'].items():
            if r.get('child_identity') and is_alive(r['child_identity']):raise RuntimeError('prior stage is still alive '+name)
            if r.get('status')=='running' and r.get('attempts'):
                attempt=r['attempts'][-1]
                if 'end' not in attempt:
                    attempt.update(end=times(),end_observation='recovery observed process absent; actual exit time unknown',exit_code=None,termination_reason='interrupted_process',recoverable=True,wall_seconds=float(r.get('wall_seconds_current_attempt',0)),wall_seconds_is_lower_bound=True,wall_scope='last persisted heartbeat elapsed only; unobserved tail not counted')
                r.update(status='interrupted',child_identity=None,reason='previous process disappeared; resume only from validated artifacts/checkpoint')
        state.setdefault('first_started_epoch',time.time())
        state.update(pid_namespace=os.readlink('/proc/self/ns/pid'),runner=process_identity(),status='preparing',config=config,heartbeat=times());atomic_json(path,state);atomic_json(root/'runner_identity.json',state['runner'])
        preflight(config,cfg)
        specs=stage_specs(cfg)
        for spec in specs:state['stages'].setdefault(spec['name'],dict(**spec,status='pending',attempts=[]))
        update_eta(state,cfg);atomic_json(path,state)
        shutdown=[];old_handlers={}
        for sig in (signal.SIGINT,signal.SIGTERM):
            old_handlers[sig]=signal.signal(sig,lambda s,f:shutdown.append(s))
        try:
            state['status']='running';logs=root/'logs';logs.mkdir(exist_ok=True)
            for spec in specs:
                name=spec['name'];record=state['stages'][name];receipt=Path(spec['receipt'])
                if record.get('validation_nonrecoverable'):
                    record.update(status='failed',reason='nonrecoverable identity validation requires explicit new attempt/version');atomic_json(path,state);continue
                if any(state['stages'][d]['status']!='succeeded' for d in spec['dependencies']):
                    record.update(status='blocked_dependency',reason='required stage unavailable');blocked_receipt(cfg,spec,record['reason']);atomic_json(path,state);continue
                try:
                    sign=signature(cfg,spec,state,config)
                    if record.get('signature') and record['signature']!=sign:raise ValueError('stage dependency changed; explicit version/invalidation required '+name)
                    if record['status']=='succeeded':
                        if sha256(receipt)!=record['receipt_sha256']:raise ValueError('completed receipt changed '+name)
                        accept(spec['stage'],read(receipt),cfg,spec);continue
                    if shutdown:break
                    # A published artifact can be recovered only under the same recorded signature.
                    if receipt.exists() and read(receipt).get('status')=='succeeded' and record.get('signature')==sign:
                        accept(spec['stage'],read(receipt),cfg,spec)
                        record.update(status='succeeded',receipt_sha256=sha256(receipt));atomic_json(path,state);continue
                except (OSError,ValueError,KeyError) as exc:
                    record.update(status='failed',failure='stage identity/artifact validation: '+str(exc),validation_nonrecoverable=True)
                    record.setdefault('validation_failures',[]).append(dict(reason=str(exc),**times()))
                    blocked_receipt(cfg,spec,record['failure'],status='failed');state['status']='failed_shared_identity';atomic_json(path,state);raise
                if record['status']=='failed' and record['attempts'] and not record['attempts'][-1].get('recoverable',False):
                    record['reason']='nonrecoverable failure retained; explicit repair/version required';atomic_json(path,state);continue
                record['signature']=sign;state['current_stage']=name
                from .resources import remaining_required
                if shutil.disk_usage(root).free<remaining_required(cfg):
                    record.update(status='paused_resource',reason='free disk below configured minimum');state['status']='paused_resource';atomic_json(path,state);break
                if not cfg.get('mini_test'):
                    available_ram=next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
                    admitted=read(cfg['admission'])['resources'];required_ram=admitted['estimated_serial_cpu_peak_bytes']+admitted['minimum_ram_remainder_bytes']
                    if available_ram<required_ram:
                        record.update(status='paused_resource',reason='available RAM below admitted serial peak plus reserve');state['status']='paused_resource';atomic_json(path,state);break
                if spec['stage'] in ('train','predict') and not cfg.get('mini_test'):
                    admission=read(cfg.get('admission',root/'admission.json'));needed=admission['resources']['estimated_peak_gpu_allocator_bytes']
                    try:
                        free=int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())*1024**2
                    except (OSError,ValueError,subprocess.CalledProcessError) as exc:
                        record.update(status='paused_resource',reason='cannot verify current GPU capacity: '+str(exc));state['status']='paused_resource';atomic_json(path,state);break
                    record['gpu_capacity_check']=dict(free_bytes=free,required_bytes=needed,device_index=0,**times())
                    if free<needed:
                        record.update(status='paused_resource',reason='current GPU free memory below admitted peak estimate');state['status']='paused_resource';atomic_json(path,state);break
                if spec['stage']=='b_cost':
                    mi={l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith(('MemAvailable:','SwapFree:'))}
                    if mi['MemAvailable']<6*1024**3:
                        record.update(status='paused_resource',reason='B diagnostics require 6 GiB available RAM');state['status']='paused_resource';atomic_json(path,state);break
                    record['b_cost_initial_swap_free']=mi['SwapFree']
                # Persisted count bounds retries across restarts; manual repair needs a new attempt/version.
                remaining=3-len(record['attempts'])
                for _ in range(max(0,remaining)):
                    cmd,cwd=command_for(cfg,spec,config);number=len(record['attempts'])+1;logpath=logs/f'{name}_attempt{number}.log'
                    attempt=dict(attempt=number,start=times(),command=cmd,log=str(logpath),sampled_peak_rss_bytes_sum=0,sampled_peak_pss_bytes_sum=0,memory_sampling_scope='stage process group sampled every 5s; RSS sums shared pages; transients may be missed')
                    record['attempts'].append(attempt);record.update(status='running',log=str(logpath));start=time.monotonic()
                    with logpath.open('w') as log:
                        threads=2 if spec['stage'] in ('train','predict') else 1 if spec['stage'] in ('postprocess','b_cost') else min(4,cfg['cpu_workers'])
                        env=dict(os.environ,PYTHONPATH=cwd,OMP_NUM_THREADS=str(threads),OPENBLAS_NUM_THREADS=str(threads),MKL_NUM_THREADS=str(threads),NUMEXPR_NUM_THREADS=str(threads))
                        child=subprocess.Popen(cmd,cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=tuple(s.fileno() for s in locks))
                        record['child_identity']=process_identity(child.pid);atomic_json(path,state);terminated=None
                        while child.poll() is None:
                            elapsed=time.monotonic()-start
                            if (shutdown or elapsed>cfg.get('stage_timeouts',TIMEOUTS).get(spec['stage'],TIMEOUTS[spec['stage']])) and terminated is None:
                                os.killpg(child.pid,signal.SIGTERM);terminated=time.monotonic();attempt['termination_reason']='runner_signal' if shutdown else 'watchdog_timeout'
                            if terminated is not None and time.monotonic()-terminated>120:os.killpg(child.pid,signal.SIGKILL)
                            mem=process_group_memory(child.pid)
                            for key in ['rss','pss']:attempt['sampled_peak_'+key+'_bytes_sum']=max(attempt['sampled_peak_'+key+'_bytes_sum'],mem[key+'_bytes_sum'])
                            if spec['stage']=='b_cost':
                                mi={l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith(('MemAvailable:','SwapFree:'))}
                                if terminated is None and (mem['pss_bytes_sum']>4*1024**3 or mi['MemAvailable']<6*1024**3 or mi['SwapFree']<record['b_cost_initial_swap_free']):
                                    os.killpg(child.pid,signal.SIGTERM);terminated=time.monotonic();attempt['termination_reason']='B_cost_resource_budget';state['status']='paused_resource'
                            progress=receipt.parent/'progress.json'
                            if progress.exists():
                                try:record['progress']=read(progress)
                                except (OSError,json.JSONDecodeError):pass
                            state['target_missed']=time.time()-state['first_started_epoch']>589.06*60
                            update_eta(state,cfg)
                            state['execution_mode']='serial_P1_then_P2_then_T1_then_T2'
                            state['heartbeat']=times();record['wall_seconds_current_attempt']=elapsed;atomic_json(path,state);time.sleep(0.05 if cfg.get('testing_poll_fast') else 5)
                        rc=child.wait()
                    attempt.update(end=times(),exit_code=rc,wall_seconds=time.monotonic()-start);record.update(child_identity=None,exit_code=rc)
                    if rc==0:
                        try:accept(spec['stage'],read(receipt),cfg,spec)
                        except Exception:attempt['artifact_failure']=traceback.format_exc()
                        else:record.update(status='succeeded',receipt_sha256=sha256(receipt));record.pop('failure',None);break
                    failure=attempt.get('artifact_failure',logpath.read_text()[-12000:]);record.update(status='failed',failure=failure);atomic_json(path,state)
                    if 'PAUSED_RESOURCE' in failure or attempt.get('termination_reason')=='B_cost_resource_budget':
                        record['status']='paused_resource';state['status']='paused_resource';attempt['recoverable']=True;atomic_json(path,state);break
                    blocked_receipt(cfg,spec,failure,status='failed')
                    if attempt.get('termination_reason')=='runner_signal':
                        record['status']='interrupted';attempt['recoverable']=True;atomic_json(path,state);break
                    retry=recoverable(rc,failure) and not shutdown and attempt.get('termination_reason')!='watchdog_timeout' and len(record['attempts'])<3
                    attempt['recoverable']=retry
                    if not retry:break
                    backoff=float(cfg['retry_backoff_seconds']);attempt['retry_backoff_seconds']=backoff;atomic_json(path,state);time.sleep(backoff)
                if remaining<=0 and record['status']!='succeeded':record.update(status='failed',failure='automatic attempts exhausted; explicit repair/new attempt required')
                atomic_json(path,state)
                if shutdown or state.get('status')=='paused_resource':break
            succeeded=complete_receipt(cfg,state);state.update(status='compute_complete_awaiting_user_resume' if succeeded else ('paused_resource' if state.get('status')=='paused_resource' else 'failed'),current_stage=None,heartbeat=times(),computation_finished=succeeded)
            atomic_json(path,state);return 0 if succeeded else 1
        finally:
            for sig,handler in old_handlers.items():signal.signal(sig,handler)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--freeze',action='store_true');a=p.parse_args()
    if a.freeze:freeze(a.config);return
    try:raise SystemExit(run(a.config))
    except BlockingIOError:print('Run/GPU resource lock held; duplicate launch refused',file=sys.stderr);raise SystemExit(3)
    except Exception:
        cfg=read(a.config);root=Path(cfg['output_root']);failure=traceback.format_exc();atomic_json(root/'runner_failure.json',dict(failure=failure,**times()))
        path=root/'run_status.json'
        if path.exists():state=read(path);state.update(status='failed',failure=failure,heartbeat=times());atomic_json(path,state)
        raise

if __name__=='__main__':main()
