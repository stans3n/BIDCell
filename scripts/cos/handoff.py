"""Publish a real ABC handoff after independent early-run acceptance."""
import argparse,json,os,shlex,socket
from datetime import datetime,timedelta,timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from scripts.crgf.common import atomic_json,identity
from scripts.crgf.run_egr import is_alive
from .evaluation import ARMS,read

def create(config):
    config=Path(config).resolve();q=read(config);root=Path(q['output_root']);state=read(root/'run_status.json');launch=read(root/'launch.json');runner=state['runner']
    if not is_alive(runner):raise ValueError('formal runner is not alive')
    arm=Path(q['methods']['A']['cost_root']);progress=read(arm/'checkpoints/A/progress.json')
    if progress['update']<100:raise ValueError('A has fewer than 100 actual updates')
    validation=read(root/'early_checkpoint_validation.json')
    if validation['status']!='succeeded' or validation['actual_updates']<100:raise ValueError('early checkpoint not independently readable')
    observation=read(root/'tests/launch_observations.json')
    if not observation.get('passed') or len(observation.get('observations',[]))<2 or not observation.get('independent_shell_verified') or not observation.get('first_probe_passed'):raise ValueError('independent progress/lock/probe acceptance missing')
    if not read(root/'tests/admission.json')['approved_to_launch']:raise ValueError('admission not approved')
    wd=q['methods']['A']['worktree'];prefix='cd '+shlex.quote(wd)+' && '+shlex.quote(q['python']);suffix=' --config '+shlex.quote(str(config))
    current=state['stages'][state['current_stage']]['log']
    commands=dict(status=prefix+' -m scripts.cos.status'+suffix,tail='tail -n 50 -F '+shlex.quote(current),process='ps -o pid,ppid,pgid,lstart,etime,args -p '+str(runner['pid']),resume=shlex.quote(str(Path(wd)/'scripts/cos/resume.sh'))+suffix,launch=shlex.quote(str(Path(wd)/'scripts/cos/launch.sh'))+suffix)
    started=datetime.fromisoformat(launch['time_utc']);now=datetime.now(timezone.utc);eta=(started+timedelta(hours=9)).astimezone(ZoneInfo('Australia/Sydney')).isoformat()
    ready={a:str(Path(q['methods'][a]['evaluation'])/'complete.json') for a in ARMS};ready.update(comparison=str(root/'comparison/compute_summary.json'),compute_complete=str(root/'compute_complete.json'))
    result=dict(version='COS-LOSS-ABC-launch-handoff-v2.0',run_id=q['run_id'],host=socket.gethostname(),output_root=str(root),execution_mode='serial_A_then_B_then_C',time_utc=now.isoformat(),time_sydney=now.astimezone(ZoneInfo('Australia/Sydney')).isoformat(),runner=runner,runner_session_id=os.getsid(runner['pid']),launch=launch,source=identity(root/'source/frozen.json'),admission=identity(root/'tests/admission.json'),early_checkpoint=identity(root/'early_checkpoint_validation.json'),observations=identity(root/'tests/launch_observations.json'),progress=progress,locks=[str(root/'runner.lock'),q['gpu_lock']],commands=commands,ready_paths=ready,target_sydney=eta,latest_eta_sydney=state.get('eta_sydney'),eta_basis=state.get('eta_basis','remaining-stage estimate unavailable'),target_missed=now>started+timedelta(hours=9),ai_goal_action='controller may pause after checking this receipt; detached computation continues; independent final analysis requires manual user resume')
    atomic_json(root/'launch_handoff.json',result)
    lines=['# A/B/C 后台交接','',f"输出：`{root}`。PID {runner['pid']}，PGID {result['runner_session_id']}，串行 A完整链→B→C。",'',f"A已完成{progress['update']}实际更新；100步checkpoint、首个轻量probe及两次独立进度观察已通过。",'',f"九小时目标（Sydney）：{eta}。最新剩余阶段估计：{state.get('eta_sydney')}。额外probe与诊断耗时另计，超过目标继续运行。",'']
    for name,command in commands.items():lines.extend([name,'','```bash',command,'```',''])
    lines.extend(['结果路径','',*[f'- {a}: `{path}`' for a,path in ready.items()],'','AI goal暂停不停止后台。用户手动 /goal resume 后才独立科学分析与最终验收。'])
    (root/'launch_handoff.md').write_text('\n'.join(lines)+'\n');print(root/'launch_handoff.md');return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();create(a.config)
