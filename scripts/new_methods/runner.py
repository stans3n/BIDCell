"""Bounded, locked background computation. Never generates or modifies source."""
import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone
from .common import *
from .provenance import component_source


def lock_file(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);stream=path.open('a+')
    try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:stream.close();raise RuntimeError('exclusive lock already held: '+str(path))
    return stream


def require_space(root,minimum):
    free=shutil.disk_usage(root).free
    if free<minimum*2**30:raise RuntimeError(f'disk guard: {free/2**30:.2f}GiB free, requires {minimum}GiB')
    return free


def admission_gate(root):
    root=Path(root);q=read(root/'configs/queue.json');ad=read(q['admission']);review=read(q['review_receipt'])
    from .inputs import verify_small_contract
    verify_small_contract(root)
    if ad.get('status')!='admitted':raise ValueError('admission incomplete')
    if review.get('status')!='approved':raise ValueError('independent implementation review not approved')
    if ad['source_digest']!=source_digest() or review['source_digest']!=source_digest():raise ValueError('source differs from admitted/reviewed implementation')
    if review['admission_sha256']!=sha256(q['admission']):raise ValueError('review does not bind final admission')
    verify_outputs(ad);verify_source(read(root/'source/manifest.json')['files'])
    if sha256(root/'configs/queue.json')!=ad['queue_sha256']:raise ValueError('queue changed after admission')
    for arm in ARMS:
        cfg=read(q['methods'][arm]['config']);cal=read(cfg['calibration'])
        if cal['status']!='admitted':raise ValueError('arm calibration not admitted '+arm)
    require_space(root,q['minimum_free_gb']);return q


def plan(q):
    tasks=[]
    # One GPU; train all three first, then each native output chain.
    for arm in q['arm_order']:tasks.append((arm,'train'))
    for arm in q['arm_order']:
        for stage in ('predict','postprocess','matrix','evaluate'):tasks.append((arm,stage))
    for arm in ('B0','B1'):tasks.append((arm,'evaluate'))
    tasks.append(('all','report'));return tasks


def dependencies(root,arm,stage):
    q=read(Path(root)/'configs/queue.json');dep=dict(source=component_source(stage),queue=identity(Path(root)/'configs/queue.json'))
    prior={'predict':'train','postprocess':'predict','matrix':'postprocess','evaluate':'matrix'}
    if arm in ARMS:
        dep['config']=identity(q['methods'][arm]['config'])
        if stage in prior:dep['prerequisite']=identity(Path(root)/'receipts'/f'{arm}_{prior[stage]}.json')
    elif arm in ('B0','B1'):
        spec=q['comparison_methods'][arm];dep['historical_matrix']=identity(Path(spec['matrix'])/'complete.json');dep['machine']=identity(spec['machine'])
    if stage=='report':
        dep['evaluations']={m:identity(Path(s['evaluation'])/'complete.json') for m,s in q['comparison_methods'].items()}
        dep['coverage']=identity(Path(root)/'admission/coverage_candidates.csv')
        expected=read(q['admission'])['outputs']['coverage_candidates.csv']
        if dep['coverage']!=expected:raise ValueError('frozen coverage drift before report')
        dep['training_logs']={}
        for method in ARMS:
            expected=read(Path(root)/method/'checkpoints/complete.json')['outputs']['log']
            if identity(expected['path'])!=expected:raise ValueError('completed training log drift before report')
            dep['training_logs'][method]=expected
    return dep


def artifact_receipt(q,arm,stage):
    if stage=='report':return Path(q['output_root'])/'report/complete.json'
    spec=q['comparison_methods'][arm];root=Path(spec.get('cost_root',Path(q['output_root'])/'baselines'/arm))
    return {'train':root/'checkpoints/complete.json','predict':root/'predictions'/arm/'complete.json',
            'postprocess':root/'predictions'/arm/'postprocess/complete.json','matrix':Path(spec['matrix'])/'complete.json',
            'evaluate':Path(spec['evaluation'])/'complete.json'}[stage]


def validate_stage(q,arm,stage):
    path=artifact_receipt(q,arm,stage);rec=read(path)
    if rec.get('status')!='succeeded':raise ValueError('stage did not succeed '+str(path))
    verify_outputs(rec)
    def verify_dependency_files(value):
        if isinstance(value,dict):
            if 'path' in value and 'sha256' in value:
                if identity(value['path'])!=value:raise ValueError('artifact dependency drift '+value['path'])
            else:
                for child in value.values():verify_dependency_files(child)
        elif isinstance(value,list):
            for child in value:verify_dependency_files(child)
    verify_dependency_files(rec.get('dependencies',{}))
    if stage=='train':
        import torch
        from .training import checkpoint_header
        cfg=read(q['methods'][arm]['config']);state=torch.load(rec['outputs']['checkpoint']['path'],map_location='cpu')
        if rec['actual_updates']!=3625 or state['actual_updates']!=3625:raise ValueError('actual Adam budget mismatch')
        if state['header']!=checkpoint_header(cfg):raise ValueError('checkpoint contract mismatch')
        if state['view_cursor']!=rec['view_cursor'] or state['view_cursor']<3625:raise ValueError('checkpoint cursor mismatch')
        if not all(torch.isfinite(v).all() for v in state['model'].values()):raise ValueError('nonfinite model tensor')
    if stage=='predict' and rec['shifts']!=[0,24]:raise ValueError('both prediction shifts required')
    if stage in ('predict','postprocess'):
        import tifffile
        import numpy as np
        machine=read(q['methods'][arm]['machine']);mask=tifffile.imread(machine['pre_mask' if stage=='predict' else 'final_mask'])
        nuclei=tifffile.imread(machine['nuclei_mask'])
        if mask.shape!=nuclei.shape or mask.dtype!=np.uint32:raise ValueError('output mask shape/dtype mismatch')
        if not np.isin(np.unique(mask),np.r_[0,np.unique(nuclei)]).all():raise ValueError('output foreign nucleus ID')
        if stage=='postprocess' and not np.array_equal(mask[nuclei>0],nuclei[nuclei>0]):raise ValueError('native nucleus restoration failed')
    if stage=='matrix':
        if rec.get('validation_kind')!='independent_full_native_resize_lookup_and_conservation':raise ValueError('matrix lacks full independent validation')
        if rec['totals']['eligible']!=rec['totals']['assigned']+rec['totals']['unassigned']:raise ValueError('matrix conservation')
        import numpy as np
        counts=np.load(Path(q['comparison_methods'][arm]['matrix'])/'counts.npy',mmap_mode='r')
        if not np.issubdtype(counts.dtype,np.integer) or (counts<0).any() or int(counts.sum())!=rec['totals']['assigned']:raise ValueError('actual integer matrix conservation failed')
    if stage=='evaluate' and len(rec.get('profiles',[]))!=3:raise ValueError('three profiles required')
    return identity(path)


def stage(root,arm,name):
    root=Path(root);q=read(root/'configs/queue.json');require_space(root,q['minimum_free_gb'])
    from .inputs import verify_small_contract
    verify_source(read(root/'source/manifest.json')['files']);verify_small_contract(root)
    gpu=lock_file(q['gpu_lock']) if name in ('train','predict') else None
    try:
        if name=='report':
            from .report import report
            return report(root)
        spec=q['comparison_methods'][arm]
        if arm in ('B0','B1'):
            from .evaluation import evaluate_spec
            return evaluate_spec(spec,q,arm)
        cfg=read(spec['config'])
        if name in ('train','predict'):
            from .inputs import verify_inputs
            atomic_json(Path(cfg['output_root'])/f'{name}_input_validation.json',verify_inputs(root,shifts=(0,) if name=='train' else (0,24)))
        if name=='train':
            from .training import train
            return train(cfg)
        if name=='predict':
            from .prediction import predict
            return predict(cfg)
        if name=='postprocess':
            from scripts.crgf.postprocess import postprocess
            adapter=SimpleNamespace(output_root=cfg['output_root'],seed=1234,postprocess_size=1024,cpu_workers=q['cpu_workers'],native=native_config)
            return postprocess(adapter,arm)
        if name=='matrix':
            from scripts.crgf.matrix import build
            return build(read(spec['machine']),spec['matrix'],cache=root/'coordinate_cache',space_guard=lambda:require_space(root,q['minimum_free_gb']))
        if name=='evaluate':
            from .evaluation import evaluate
            return evaluate(cfg)
        raise ValueError(name)
    finally:
        if gpu:gpu.close()


def run(root):
    root=Path(root);guard=lock_file(root/'run.lock');q=admission_gate(root);started=time.time();(root/'receipts').mkdir(exist_ok=True)
    state=dict(state='running',scientific_review='awaiting_user_resume',process=process_identity(),started_utc=datetime.now(timezone.utc).isoformat(),stages={})
    status_path=root/'status.json'
    child=None
    def stop(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        for arm,name in plan(q):
            from .inputs import verify_small_contract
            verify_source(read(root/'source/manifest.json')['files']);verify_small_contract(root)
            key=f'{arm}_{name}';path=root/'receipts'/f'{key}.json';dep=dependencies(root,arm,name)
            if path.exists():
                saved=read(path)
                if saved['dependencies']!=dep:raise ValueError('completed stage dependencies changed '+key)
                verify_outputs(saved);validate_stage(q,arm,name)
                state['stages'][key]=dict(state='reused_verified',receipt=identity(path));continue
            pending=root/'receipts'/f'{key}_pending.json'
            artifact_path=artifact_receipt(q,arm,name)
            if artifact_path.exists() and read(artifact_path).get('status')=='succeeded':
                # Child completion and parent commit are two atomic writes. The
                # prelaunch producer contract bridges a crash between them.
                if not pending.exists() or read(pending)['dependencies']!=dep:raise ValueError('completed artifact has no matching prelaunch contract '+key)
                artifact=validate_stage(q,arm,name)
                atomic_json(path,dict(status='succeeded',dependencies=dep,attempts=[],recovered_child_completion=True,
                                     outputs={'artifact_receipt':artifact}))
                state['stages'][key]=dict(state='recovered_verified',receipt=identity(path));atomic_json(status_path,state);continue
            atomic_json(pending,dict(state='pending',dependencies=dep))
            attempts=[]
            for attempt in range(q['retries']+1):
                require_space(root,q['minimum_free_gb']);logpath=root/'logs'/f'{key}_attempt{attempt}.log'
                state.update(current=dict(arm=arm,stage=name,attempt=attempt,log=str(logpath)),last_update=time.time());atomic_json(status_path,state)
                start=time.monotonic()
                with logpath.open('a') as log:
                    child=subprocess.Popen([q['python'],'-m','scripts.new_methods.runner','stage','--root',str(root),'--arm',arm,'--stage',name],
                          cwd=REPO,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,
                          env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4'))
                    state['current']['child']=process_identity(child.pid);atomic_json(status_path,state)
                    try:code=child.wait(timeout=q['stage_timeouts'][name])
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGTERM)
                        try:child.wait(timeout=30)
                        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                        code=124
                attempts.append(dict(attempt=attempt,exit_code=code,wall_seconds=time.monotonic()-start,log=identity(logpath)))
                if code==0:
                    artifact=validate_stage(q,arm,name)
                    atomic_json(path,dict(status='succeeded',dependencies=dep,attempts=attempts,outputs={'artifact_receipt':artifact}))
                    state['stages'][key]=dict(state='succeeded',receipt=identity(path),attempts=attempts);atomic_json(status_path,state);break
                if attempt==q['retries']:raise RuntimeError(f'{key} failed after {attempt+1} attempts; see {logpath}')
                time.sleep(10)
        state.update(state='compute_complete',scientific_review='awaiting_user_resume',finished_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.time()-started)
        atomic_json(status_path,state)
    except BaseException:
        state.update(state='failed',failure=traceback.format_exc(),last_update=time.time());atomic_json(status_path,state);raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        guard.close()


def status(root):
    root=Path(root);path=root/'status.json';state=read(path) if path.exists() else dict(state='not_launched')
    process=state.get('process');alive=False
    if process:
        try:alive=process_identity(process['pid'])==process
        except (FileNotFoundError,ProcessLookupError):pass
    progress={}
    for arm in ARMS:
        for name,path in [('train',root/arm/'checkpoints/progress.json'),('predict',root/arm/'predictions'/arm/'progress.json'),('postprocess',root/arm/'predictions'/arm/'postprocess/progress.json'),('matrix',root/arm/'matrix'/arm/'progress.json')]:
            if path.exists():progress[f'{arm}_{name}']=read(path)
    return dict(**state,runner_alive=alive,progress=progress,free_disk_gib=shutil.disk_usage(root).free/2**30)


def launch(root):
    root=Path(root);q=admission_gate(root)
    launch_guard=lock_file(root/'launch.lock')
    # Reject an existing live runner by acquiring its exact exclusive lock.
    guard=lock_file(root/'run.lock');guard.close()
    try:
        with (root/'logs/runner.log').open('a') as log:
            proc=subprocess.Popen([q['python'],'-m','scripts.new_methods.runner','run','--root',str(root)],cwd=REPO,
                                  stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        ident=process_identity(proc.pid);deadline=time.monotonic()+60
        while time.monotonic()<deadline:
            if proc.poll() is not None:raise RuntimeError('runner failed its startup handshake; inspect runner.log')
            state=read(root/'status.json') if (root/'status.json').exists() else {}
            if state.get('process')==ident and state.get('state')=='running':break
            time.sleep(.25)
        else:
            os.killpg(proc.pid,signal.SIGTERM);raise RuntimeError('runner startup handshake timeout')
        receipt=dict(process=ident,runroot=str(root),source_digest=source_digest(),admission=identity(q['admission']),review=identity(q['review_receipt']))
        atomic_json(root/'launch.json',receipt);return receipt
    finally:launch_guard.close()


def complete_check(root):
    root=Path(root);q=read(root/'configs/queue.json');checks={}
    for arm,name in plan(q):
        path=root/'receipts'/f'{arm}_{name}.json';saved=read(path)
        if saved['dependencies']!=dependencies(root,arm,name):raise ValueError('receipt dependency drift')
        verify_outputs(saved);checks[f'{arm}_{name}']=validate_stage(q,arm,name)
    return dict(state='compute_complete',scientific_review='awaiting_user_resume',checked=checks)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','admit','launch','run','resume','stage','status','complete-check','tail'])
    p.add_argument('--root',required=True);p.add_argument('--arm',choices=[*ARMS,'B0','B1','all']);p.add_argument('--stage');p.add_argument('--lines',type=int,default=40)
    a=p.parse_args()
    if a.command=='prepare':
        from .prepare import prepare
        result=prepare(a.root)
    elif a.command=='admit':
        from .admission import admit
        result=admit(a.root)
    elif a.command in ('launch','resume'):result=launch(a.root)
    elif a.command=='run':result=run(a.root)
    elif a.command=='stage':result=stage(a.root,a.arm,a.stage)
    elif a.command=='status':result=status(a.root)
    elif a.command=='complete-check':result=complete_check(a.root)
    else:
        s=status(a.root);log=Path(s.get('current',{}).get('log',Path(a.root)/'logs/runner.log'))
        if log.exists():print('\n'.join(log.read_text(errors='replace').splitlines()[-a.lines:]))
        return
    print(json.dumps(json_safe(result),indent=2))


if __name__=='__main__':main()
