"""One-shot independent host-shell launch observation; never polls or schedules."""
import argparse,fcntl,json,os,socket,subprocess,time
from pathlib import Path
from .evaluation import read,times
from .status import status
from scripts.crgf.common import atomic_json,identity,process_identity
from scripts.crgf.run_egr import is_alive

def observe(config,validate_checkpoint=False):
    config=Path(config).resolve();q=read(config);root=Path(q['output_root']);state=read(root/'run_status.json');runner=state['runner']
    if runner['pid']==os.getpid() or not is_alive(runner):raise ValueError('observation requires a separate live host-shell process')
    armroot=Path(q['methods']['A']['cost_root']);progress=read(armroot/'checkpoints/A/progress.json');probe=armroot/'diagnostics/probe_000100.json'
    if progress['update']<100:raise ValueError('A has fewer than 100 actual optimizer updates')
    if not probe.exists():raise ValueError('A first lightweight probe absent')
    pr=read(probe)
    if pr['status']!='succeeded' or pr['update']!=100 or pr['arm']!='A' or len(pr['samples'])!=8:raise ValueError('first probe invalid')
    locks=[]
    for path in (root/'runner.lock',Path(q['gpu_lock'])):
        with path.open('a+') as f:
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:pass
            else:raise ValueError('runner lock is not held: '+str(path))
        owner=read(path)
        if owner['pid']!=runner['pid'] or owner['start_ticks']!=runner['start_ticks']:raise ValueError('lock owner identity mismatch')
        locks.append(dict(path=str(path),owner=owner,exclusive_lock_confirmed=True))
    cwd=q['methods']['A']['worktree'];env=dict(os.environ,PYTHONPATH=cwd,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    checked=subprocess.run([q['python'],'-m','scripts.cos.status','--config',str(config)],cwd=cwd,env=env,text=True,capture_output=True,check=True)
    if not json.loads(checked.stdout)['runner_alive']:raise ValueError('status command did not confirm runner')
    log=Path(state['stages'][state['current_stage']]['log']);tail=subprocess.run(['tail','-n','5',str(log)],text=True,capture_output=True,check=True)
    resume=Path(cwd)/'scripts/cos/resume.sh'
    subprocess.run(['bash','-n',str(resume)],check=True)
    refused=subprocess.run([str(resume),'--config',str(config)],cwd=cwd,env=env,text=True,capture_output=True)
    if refused.returncode==0 or 'runner already active' not in refused.stdout+refused.stderr:raise ValueError('resume duplicate-launch refusal not demonstrated')
    process=subprocess.check_output(['ps','-o','pid,ppid,pgid,lstart,etime,args','-p',str(runner['pid'])],text=True)
    if validate_checkpoint:
        from .validate_checkpoint import validate
        validate(q['methods']['A']['config'],100,root/'early_checkpoint_validation.json')
    cp=root/'early_checkpoint_validation.json';checkpoint_ok=cp.exists() and read(cp).get('status')=='succeeded' and read(cp).get('actual_updates')==100
    path=root/'tests/launch_observations.json';prior=read(path) if path.exists() else dict(observations=[])
    observation=dict(epoch=time.time(),**times(),observer=process_identity(),host=socket.gethostname(),runner=runner,pgid=os.getpgid(runner['pid']),process_query=process,locks=locks,update=progress['update'],elapsed_seconds=progress['elapsed_seconds'],log=str(log),log_bytes=log.stat().st_size,tail=tail.stdout,probe=identity(probe),status_cli_verified=True,resume_active_runner_rejected=True)
    entries=prior['observations']+[observation];growth=any(e['runner']==runner and observation['epoch']-e['epoch']>=20 and observation['update']>e['update'] for e in entries[:-1])
    result=dict(status='passed' if growth and checkpoint_ok else 'observing',passed=growth and checkpoint_ok,observations=entries,first_probe_passed=True,independent_shell_verified=True,checkpoint_validation=identity(cp) if checkpoint_ok else None,minimum_interval_seconds=20,strict_update_growth_verified=growth)
    atomic_json(path,result);print(json.dumps(dict(status=result['status'],update=progress['update'],observations=len(entries),passed=result['passed']),indent=2));return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--validate-checkpoint',action='store_true');a=p.parse_args();observe(a.config,a.validate_checkpoint)
