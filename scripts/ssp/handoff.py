"""Observe actual detached state and produce concrete handoff evidence, never goal actions."""
import argparse,fcntl,json,time,subprocess,os
from pathlib import Path
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
from scripts.crgf.common import atomic_json,identity,sha256,process_identity,digest
from scripts.crgf.run_egr import is_alive
from .evaluation import read

def observe(root):
    root=Path(root);q=read(root/'configs/queue.json');state=read(root/'run_status.json');runner=state['runner']
    if state.get('pid_namespace')!=os.readlink('/proc/self/ns/pid'):raise ValueError('Run this observation in host PID namespace')
    if not is_alive(runner):raise ValueError('runner identity is not alive')
    checks={}
    for p in (root/'runner.lock',Path(q['gpu_lock'])):
        with p.open('a+') as f:
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:checks[str(p)]=True
            else:raise ValueError('runner does not hold required lock '+str(p))
    progress=read(root/'S/checkpoints/S/progress.json')
    receipt=dict(epoch=time.time(),time_sydney=datetime.now(timezone.utc).astimezone(ZoneInfo('Australia/Sydney')).isoformat(),runner=runner,process_identity_verified=True,lock_checks=checks,S_update=progress['update'],progress=identity(root/'S/checkpoints/S/progress.json'),current_stage=state['current_stage'])
    path=root/'diagnostics/launch_observations.json';old=read(path) if path.exists() else {'observations':[]};old['observations'].append(receipt);atomic_json(path,old);return receipt

def duplicate_check(root):
    root=Path(root);q=read(root/'configs/queue.json');s=read(root/'run_status.json')
    if not is_alive(s['runner']):raise ValueError('no active runner to test duplicate refusal')
    wrapper=Path(q['methods']['S']['worktree'])/'scripts/ssp/prepare_and_launch.sh'
    cmd=[str(wrapper),'launch',str(root)];r=subprocess.run(cmd,text=True,capture_output=True)
    refused=r.returncode!=0 and any(x in r.stderr+r.stdout for x in ('runner already active','Resource temporarily unavailable','lock held','previous child still active'))
    record=dict(command=cmd,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,refused=refused,runner_identity_unchanged=is_alive(s['runner']))
    atomic_json(root/'diagnostics/duplicate_launch.json',record)
    if not refused or not record['runner_identity_unchanged']:raise ValueError('duplicate launch refusal not established')
    return record


def generate(root):
    import torch
    root=Path(root);q=read(root/'configs/queue.json');s=read(root/'run_status.json');obs=read(root/'diagnostics/launch_observations.json')['observations']
    valid=any(b['epoch']-a['epoch']>=20 and b['S_update']>a['S_update'] for i,a in enumerate(obs) for b in obs[i+1:])
    if not valid:raise ValueError('two >=20 second separated increasing-update observations required')
    if not is_alive(s['runner']):raise ValueError('runner no longer active')
    probe=root/'S/diagnostics/probe_000100.json';p=read(probe)
    if p['status']!='succeeded' or p.get('arm')!='S' or p.get('update')!=100 or len(p['samples'])!=8:raise ValueError('100 step eight probes incomplete')
    ck=root/'S/checkpoints/S/update_000100.pth';receipt=read(ck.with_suffix('.json'))
    if sha256(ck)!=receipt['checkpoint']['sha256']:raise ValueError('100 checkpoint digest mismatch')
    state=torch.load(ck,map_location='cpu');from .training import dependencies,make_model,optimizer,load
    from .config import read as config_read
    cfg=config_read(q['methods']['S']['config']);deps=dependencies(cfg,'S')
    model=make_model(cfg,'S',len(state['gene_order']),device='cpu');opt=optimizer(model,cfg)
    loaded=load(ck,model,opt,deps)
    if loaded[:2]!=(100,state['sampler_cursor']):raise ValueError('strict checkpoint restore counter mismatch')
    if state['dependencies']!=deps or state['update_step']!=100 or state['sampler_cursor']<100 or state['gene_order']!=deps['inputs']['gene_order']:raise ValueError('100 checkpoint complete-state identity mismatch')
    required={'model_state_dict','optimizer_state_dict','rng','architecture','BN_state','sampler_cursor','gene_order'}
    if not required<=set(state) or not state['optimizer_state_dict']['state']:raise ValueError('100 checkpoint missing full optimizer/RNG/model')
    for group in state['optimizer_state_dict']['state'].values():
        if int(group['step'])!=100:raise ValueError('Adam actual update counter not100')
    duplicate=root/'diagnostics/duplicate_launch.json'
    if not duplicate.exists() or read(duplicate).get('refused') is not True:raise ValueError('actual duplicate launch refusal receipt missing')
    wrapper=str(Path(q['methods']['S']['worktree'])/'scripts/ssp/prepare_and_launch.sh');base=f'{wrapper}'
    commands={k:f'{base} {k} {root}' for k in ('status','tail','process','resume','complete')}
    commands.update({f'{a}_evaluate':f'{base} evaluate {root} --arm {a}' for a in ('S','SP')});commands.update({f'{a}_complete':f'{base} complete {root} --arm {a}' for a in ('S','SP')})
    result=dict(version='SSP-OVERNIGHT-v1.0',run_root=str(root),runner=s['runner'],launch=read(root/'launch.json'),source=identity(root/'source/frozen.json'),queue=identity(root/'configs/queue.json'),independent_review=identity(q['review_receipt']),current_stage=s['current_stage'],S_observed_update=max(o['S_update'] for o in obs),checkpoint100=identity(ck),checkpoint100_full_state_cpu_validated=True,first_probe=identity(probe),observations=identity(root/'diagnostics/launch_observations.json'),duplicate_launch=identity(duplicate),commands=commands,eta_sydney=s.get('eta_sydney'),eta_basis=s.get('eta_basis'),target_hours=8,target_missed=s.get('target_missed',False),methods=dict(S='frozen confidence-weighted 5x5 expression soft POS, old COS and B anchor retained',SP='same S targets plus full-nucleus prototype 390→32→1 nonlinear residual with zero output init'),shared_parent='02ce88f2007eaa7a4f4de534b951a4ea0c75243bccf2804e4ea17c50d4de3b39',actual_updates_per_arm=4000,execution_order='S full chain then SP full chain then B facts and comparison',host_shell_required='Commands run in the host PID namespace; next AI resume must use require_escalated for actual process observation/launch. A sandbox-invisible PID is unknown, never proof of death.',ai_goal_status='controller must separately verify and pause after all agents end',background_terminal_state='compute_complete_awaiting_user_resume',scientific_acceptance='pending independent Astra after manual user resume')
    for name in ('initial_forecast_sydney','target_deadline_sydney','forecast_target_miss'):result[name]=s.get(name)
    atomic_json(root/'launch_handoff.json',result)
    text='# SSP overnight handoff\n\n'+f'Run: `{root}`. Actual S update observed: {result["S_observed_update"]}. Sydney ETA: {result["eta_sydney"]}.\n\n'+''.join(f'- {k}: `{v}`\n' for k,v in commands.items())+'\nS and SP independently start from the same B1 checkpoint, each 4000 real optimizer updates. Background computes both full chains and facts; independent scientific acceptance and AI goal completion require the user’s manual resume.\n'
    text+='\n'+result['host_shell_required']+'\n\nEight-hour target: '+str(result['target_deadline_sydney'])+'. Forecast target miss: '+str(result['forecast_target_miss'])+'. The runner retains every required stage if the target is missed.\n'
    (root/'launch_handoff.md').write_text(text);return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--observe',action='store_true');p.add_argument('--duplicate-check',action='store_true');a=p.parse_args();print(json.dumps(duplicate_check(a.root) if a.duplicate_check else observe(a.root) if a.observe else generate(a.root),indent=2))
