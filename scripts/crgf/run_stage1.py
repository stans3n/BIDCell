"""Single-writer persistent queue. Failures propagate; unrelated method can continue."""
import argparse
import errno
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime,timezone
from pathlib import Path
from .config import read
from .common import atomic_json,identity,sha256,digest,verify_outputs,process_identity,source_identity


def now():return datetime.now(timezone.utc).isoformat()

def stage_specs(root):
    specs=[]
    for method in ('F1','B1'):specs.append((method+'_train',method,'train',[],root/'checkpoints'/method/'complete.json'))
    for method in ('F1','B1'):
        previous=method+'_train'
        for stage,path in [('predict',root/'predictions'/method/'complete.json'),('postprocess',root/'predictions'/method/'postprocess/complete.json'),('matrix',root/'matrix'/method/'complete.json'),('evaluate',root/'evaluation'/method/'complete.json')]:
            name=method+'_'+stage;specs.append((name,method,stage,[previous],path));previous=name
    specs.append(('report',None,'report',[],root/'report_complete.json'));return specs


def source_map():
    checkout=Path(__file__).resolve().parents[2]
    files=list((checkout/'scripts/crgf').rglob('*.py'))+list((checkout/'scripts/crgf').rglob('*.R'))+list((checkout/'scripts/crgf').rglob('*.sh'))+list((checkout/'scripts/crgf/configs').glob('*.json'))
    files+=list((checkout/'bidcell').rglob('*.py'))
    return {str(p.relative_to(checkout)):sha256(p) for p in sorted(files)}


def freeze(root):
    lock=(root/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=root/'run_status.json'
    if status.exists():
        old=json.loads(status.read_text())
        if any(v.get('child_identity') and is_alive(v['child_identity']) for v in old['stages'].values()):raise RuntimeError('cannot freeze with live stage')
    previous=root/'source/frozen.json'
    if previous.exists():
        history=root/'source/freeze_history';history.mkdir(exist_ok=True)
        atomic_json(history/f'{time.time_ns()}.json',json.loads(previous.read_text()))
    checkout=Path(__file__).resolve().parents[2]
    value=dict(files=source_map(),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip(),
               branch=subprocess.check_output(['git','branch','--show-current'],cwd=checkout,text=True).strip(),checkout=str(checkout),time=now())
    atomic_json(root/'source/frozen.json',value)


def signature(cfg,stage,method,dependencies):
    from .training import TRAIN_SOURCES
    sources={
      'train':TRAIN_SOURCES,
      'predict':TRAIN_SOURCES+['scripts/crgf/prediction.py','bidcell/model/utils/utils.py','bidcell/model/predict.py'],
      'postprocess':['scripts/crgf/postprocess.py','bidcell/model/postprocess_predictions.py'],
      'matrix':['scripts/crgf/matrix.py'],
      'evaluate':['scripts/crgf/evaluate.py','scripts/crgf/structural.py','scripts/crgf/legacy_benchmark.py','scripts/crgf/cellspa.R'],
      'report':['scripts/crgf/run.py']}
    return digest(dict(source=source_identity(list(dict.fromkeys(sources[stage]+['scripts/crgf/common.py']))),method=method,stage=stage,
           config=cfg.model_dump(exclude={'stage_timeouts','minimum_free_gb'}),
           inputs=sha256(Path(cfg.output_root)/'configs/inputs.json'),dependencies=dependencies))


def verify_preflight(cfg):
    root=Path(cfg.output_root);frozen=json.loads((root/'source/frozen.json').read_text())
    if frozen['files']!=source_map():raise ValueError('code changed after freeze; stop queue, inspect dependencies, explicitly freeze repaired version before resume')
    complete=json.loads((root/'configs/prepare_complete.json').read_text());verify_outputs(complete)
    inputs=json.loads((root/'configs/inputs.json').read_text())
    for name,obj in inputs.items():
        if isinstance(obj,dict) and 'path' in obj and 'sha256' in obj:
            if sha256(obj['path'])!=obj['sha256']:raise ValueError('fixed input changed '+name)
    for shift in cfg.shifts:
        inventory=json.loads((root/f'configs/patches_shift{shift}.json').read_text())
        for obj in inventory:
            p=Path(obj['path'])
            if p.stat().st_size!=obj['bytes'] or sha256(p)!=obj['sha256']:raise ValueError('expression patch changed '+str(p))
    for obj in inputs['official_sources'].values():
        if sha256(obj['path'])!=obj['sha256']:raise ValueError('official R source changed')


def is_alive(record):
    try:
        stat=Path(f"/proc/{record['pid']}/stat").read_text().split(') ',1)[1].split()
        return stat[0]!='Z' and process_identity(record['pid'])==record
    except (FileNotFoundError,ProcessLookupError):return False


def process_group_memory(pgid):
    rss=pss=0;members=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:
            fields=(entry/'stat').read_text().split(') ',1)[1].split()
            if int(fields[2])!=pgid:continue
            values={}
            for line in (entry/'smaps_rollup').read_text().splitlines():
                if ':' in line:
                    key,value=line.split(':',1)
                    if key in ('Rss','Pss'):values[key]=int(value.split()[0])*1024
            rss+=values.get('Rss',0);pss+=values.get('Pss',0);members.append(int(entry.name))
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    return dict(rss_bytes_sum=rss,pss_bytes_sum=pss,members=members)


def run(config_path):
    cfg=read(config_path);root=Path(cfg.output_root);root.mkdir(parents=True,exist_ok=True)
    lock=(root/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=root/'run_status.json';state=json.loads(path.read_text()) if path.exists() else dict(run_id=cfg.run_id,stages={},created_at=now())
    for name,record in state['stages'].items():
        if record.get('child_identity') and is_alive(record['child_identity']):raise RuntimeError('prior child still alive: '+name)
    state.update(runner=process_identity(),status='preflight',heartbeat=now(),config=str(Path(config_path).resolve()),checkout=str(Path(__file__).resolve().parents[2]))
    atomic_json(path,state);atomic_json(root/'runner_identity.json',state['runner'])
    verify_preflight(cfg)
    specs=stage_specs(root)
    for name,method,stage,deps,receipt in specs:
        state['stages'].setdefault(name,dict(status='pending',attempts=[],stage=stage,method=method,dependencies=deps,receipt=str(receipt)))
    shutdown=[]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda s,f:shutdown.append(s))
    logs=root/'logs';logs.mkdir(exist_ok=True);state['status']='running';atomic_json(path,state)
    for name,method,stage,deps,receipt in specs:
        record=state['stages'][name]
        dep_hashes={d:state['stages'][d].get('receipt_sha256') for d in deps}
        if stage=='report':
            dep_hashes={k:dict(status=v['status'],receipt=v.get('receipt_sha256'),attempts=v.get('attempts')) for k,v in state['stages'].items() if k!='report'}
        sign=signature(cfg,stage,method,dep_hashes)
        if any(state['stages'][d]['status']!='succeeded' for d in deps):
            record.update(status='blocked_by_dependency',reason='required stage failed');atomic_json(path,state);continue
        if stage=='report' and record.get('signature')!=sign and receipt.exists():
            archive=root/'report_history'/str(time.time_ns());archive.mkdir(parents=True)
            for file in ['results.json','raw_comparison.csv','report.md','report_complete.json']:
                if (root/file).exists():os.replace(root/file,archive/file)
            record['status']='pending'
        if record['status']=='succeeded':
            if record.get('signature')!=sign:raise ValueError('completed stage dependency changed; explicit invalidation required '+name)
            if sha256(receipt)!=record['receipt_sha256']:raise ValueError('stage receipt changed '+name)
            verify_outputs(json.loads(receipt.read_text()));continue
        if shutdown:break
        state['current_stage']=name
        # Stages interrupted after successful publication are accepted only under prior signature.
        if receipt.exists():
            if record.get('signature')!=sign:raise ValueError('unmatched existing receipt '+name)
            artifact=json.loads(receipt.read_text());verify_outputs(artifact)
            if artifact['status']!='succeeded':raise ValueError('unsuccessful stage artifact')
            record.update(status='succeeded',receipt_sha256=sha256(receipt),end=now());atomic_json(path,state);continue
        record['signature']=sign
        for retry in range(3):
            import shutil
            if shutil.disk_usage(root).free<cfg.minimum_free_gb*1024**3:raise OSError(errno.ENOSPC,'free disk below configured minimum')
            cmd=[sys.executable,'-u','-m','scripts.crgf.run','--config',str(Path(config_path).resolve()),'--stage',stage]
            if method:cmd+=['--method',method]
            logpath=logs/f'{name}_attempt{len(record["attempts"])+1}.log'
            attempt=dict(start=now(),command=cmd,log=str(logpath),sampled_peak_rss_bytes_sum=0,sampled_peak_pss_bytes_sum=0,memory_sampling_scope='5 second samples of stage process group including workers; RSS double counts shared pages, PSS apportions shared pages; may miss transients');record['attempts'].append(attempt)
            record.update(status='running',start=now(),log=str(logpath));start=time.monotonic()
            with logpath.open('w') as log:
                env=os.environ.copy();env.update(PYTHONPATH=state['checkout'],OMP_NUM_THREADS=str(cfg.cpu_workers),OPENBLAS_NUM_THREADS=str(cfg.cpu_workers),MKL_NUM_THREADS=str(cfg.cpu_workers))
                child=subprocess.Popen(cmd,cwd=state['checkout'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),))
                record['child_identity']=process_identity(child.pid);atomic_json(path,state);terminated=None
                while child.poll() is None:
                    elapsed=time.monotonic()-start
                    if (shutdown or elapsed>cfg.stage_timeouts[stage]) and terminated is None:
                        os.killpg(child.pid,signal.SIGTERM);terminated=time.monotonic();attempt['termination_reason']='runner_signal' if shutdown else 'stage_timeout'
                    if terminated is not None and time.monotonic()-terminated>120:os.killpg(child.pid,signal.SIGKILL)
                    memory=process_group_memory(child.pid)
                    attempt['sampled_peak_rss_bytes_sum']=max(attempt['sampled_peak_rss_bytes_sum'],memory['rss_bytes_sum'])
                    attempt['sampled_peak_pss_bytes_sum']=max(attempt['sampled_peak_pss_bytes_sum'],memory['pss_bytes_sum'])
                    record['latest_memory_sample']=memory
                    state['heartbeat']=now();record['wall_seconds_current_attempt']=elapsed
                    progress_paths={'train':root/'checkpoints'/str(method)/'progress.json','predict':root/'predictions'/str(method)/'progress.json','postprocess':root/'predictions'/str(method)/'postprocess/progress.json','matrix':root/'matrix'/str(method)/'progress.json'}
                    progress=progress_paths.get(stage)
                    if progress and progress.exists():record['progress']=json.loads(progress.read_text());record['progress_mtime_ns']=progress.stat().st_mtime_ns
                    atomic_json(path,state);time.sleep(5)
                rc=child.wait()
            attempt.update(end=now(),exit_code=rc,wall_seconds=time.monotonic()-start)
            record.update(exit_code=rc,end=now(),child_identity=None)
            if rc==0:
                try:
                    artifact=json.loads(receipt.read_text());verify_outputs(artifact)
                    if artifact['status']!='succeeded':raise ValueError('stage artifact not succeeded')
                    record.update(status='succeeded',receipt_sha256=sha256(receipt));break
                except Exception:attempt['artifact_failure']=traceback.format_exc()
            record['status']='failed';record['failure']=attempt.get('artifact_failure',logpath.read_text()[-8000:])
            atomic_json(path,state)
            # Only explicitly identified transient I/O errors; NaN/OOM/ID errors never retry.
            transient=any(x in record['failure'] for x in ['[Errno 5] Input/output error','[Errno 116] Stale file handle','[Errno 110] Connection timed out'])
            if not transient or retry==2 or shutdown:break
            time.sleep(5)
        atomic_json(path,state)
        common_failure=record.get('status')=='failed' and any(text in record.get('failure','') for text in ['No space left on device','CUDA driver initialization failed','No CUDA GPUs are available','GPU has fallen off the bus','CUDA error: unknown error','device is busy or unavailable','fixed input changed','expression patch changed'])
        if common_failure:
            state['shared_failure']=dict(stage=name,reason=record['failure'])
            for other in state['stages'].values():
                if other['status']=='pending':other.update(status='blocked_by_dependency',reason='shared input/hardware/resource failure')
            break
        if shutdown:break
    state['heartbeat']=now();state['current_stage']=None
    failed=[k for k,v in state['stages'].items() if v['status']!='succeeded']
    state.update(status='partial_failed' if failed else 'awaiting_user_analysis',computation_finished=not failed,unfinished=failed,finished_at=now())
    atomic_json(path,state)
    if failed:atomic_json(root/'failure_report.json',dict(unfinished=failed,stages=state['stages']))
    return 1 if failed else 0


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--freeze',action='store_true');a=p.parse_args()
    if a.freeze:freeze(Path(read(a.config).output_root));return
    try:raise SystemExit(run(a.config))
    except BlockingIOError:
        print("Single-writer lock held; refusing duplicate launch",file=sys.stderr);raise SystemExit(3)
    except Exception:
        cfg=read(a.config);root=Path(cfg.output_root);atomic_json(root/'runner_failure.json',dict(time=now(),traceback=traceback.format_exc()))
        path=root/'run_status.json'
        if path.exists():
            state=json.loads(path.read_text());state.update(status='runner_failed',failure=traceback.format_exc(),heartbeat=now());atomic_json(path,state)
        raise

if __name__=='__main__':main()
