"""Create a real launch handoff only after a stable formal early checkpoint."""
import argparse,json,os,shlex,socket,time
from datetime import datetime,timedelta,timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from scripts.crgf.common import atomic_json,identity,sha256
from scripts.crgf.run_egr import is_alive
from .config import ARMS


def read(p):return json.loads(Path(p).read_text())
def create(config):
    config=Path(config);q=read(config);root=Path(q['output_root']);state=read(root/'run_status.json');launch=read(root/'launch.json');runner=state['runner']
    if not is_alive(runner):raise ValueError('formal runner is not alive')
    if state['current_stage']!='B1_CONT_train':raise ValueError('handoff expects initial formal train stage; independently inspect advanced state')
    arm=Path(q['methods']['B1_CONT']['cost_root']);progress=read(arm/'checkpoints/B1_CONT/progress.json')
    if progress['update']<100 or progress['elapsed_seconds']<180:raise ValueError('not yet stable for several minutes and 100 actual updates')
    validation=read(root/'early_checkpoint_validation.json')
    if validation['status']!='succeeded' or validation['actual_updates']<100:raise ValueError('early checkpoint not independently readable')
    if not read(root/'admission.json')['approved_to_launch']:raise ValueError('admission not approved')
    wd=q['methods']['B1_CONT']['worktree'];python=q['python'];prefix='cd '+shlex.quote(wd)+' && '+shlex.quote(python)
    current=state['stages'][state['current_stage']]['log']
    commands=dict(status=prefix+' -m scripts.cos.status --config '+shlex.quote(str(config)),tail='tail -n 50 -F '+shlex.quote(current),attach='tail -F '+shlex.quote(current)+' '+shlex.quote(launch['log']),resume=shlex.quote(str(Path(wd)/'scripts/cos/resume.sh'))+' --config '+shlex.quote(str(config)),launch=shlex.quote(str(Path(wd)/'scripts/cos/launch.sh'))+' --config '+shlex.quote(str(config)))
    started=datetime.fromisoformat(launch['time_utc']);now=datetime.now(timezone.utc)
    # Historical base 170 minutes/arm excludes new diagnostics. The interval is
    # deliberately a planning range, not a deadline or a sampled confidence band.
    eta={name:dict(earliest_sydney=(started+timedelta(hours=lo)).astimezone(ZoneInfo('Australia/Sydney')).isoformat(),latest_sydney=(started+timedelta(hours=hi)).astimezone(ZoneInfo('Australia/Sydney')).isoformat()) for name,lo,hi in [('B1_pair',6,8),('G_pair_four',12,16)]}
    receipt=dict(version='COS-launch-handoff-v1',run_id=q['run_id'],host=socket.gethostname(),time_utc=now.isoformat(),time_sydney=now.astimezone(ZoneInfo('Australia/Sydney')).isoformat(),output_root=str(root),interpreter=python,runner=runner,runner_session_id=os.getsid(runner['pid']),launcher_receipt=identity(root/'launch.json'),launch_utc=launch['time_utc'],launch_sydney=launch['time_sydney'],locks=[str(root/'runner.lock'),q['gpu_lock']],four_worktrees={a:{k:q['methods'][a][k] for k in ('worktree','commit','config')} for a in ARMS},source=identity(root/'source/frozen.json'),admission=identity(root/'admission.json'),progress=progress,checkpoint_validation=identity(root/'early_checkpoint_validation.json'),latest_checkpoint=read(arm/'checkpoints/B1_CONT/latest.json'),commands=commands,ready_paths=dict(B1=str(root/'pair_B1/pair_ready.json'),G=str(root/'pair_G/pair_ready.json'),compute_complete=str(root/'compute_complete.json')),eta=eta,eta_basis='Historical measured 170 min/arm plus explicit allowance for new diagnostics and observed initial throughput; 12:00 Sydney is a target, not a promise.',observed_training_seconds_per_update=progress['elapsed_seconds']/progress['measured_updates'],limitations=['CPU peaks sampled, not guaranteed maxima','Storage uses measured compression extrapolation with safety factors and runtime 8 GiB guard','Detached OS session survives the launching shell; machine shutdown is not tested or guaranteed','Scientific analysis and final goal completion remain pending user resume'],ai_goal_action='controller may pause only after independent host shell acceptance; background continues')
    atomic_json(root/'launch_handoff.json',receipt)
    text=f'''# COS four-arm launch handoff\n\nRun `{q['run_id']}` on `{receipt['host']}` is running in detached session `{receipt['runner_session_id']}`, PID `{runner['pid']}` (start ticks `{runner['start_ticks']}`).\n\nStarted UTC {receipt['launch_utc']}; Sydney {receipt['launch_sydney']}. At this handoff B1_CONT has {progress['update']} actual new updates and {progress['elapsed_seconds']:.1f} measured training seconds. Its formal update-100 checkpoint passed full model/optimizer/BN/RNG/dependency and log/cursor validation.\n\nOutput: `{root}`. Full source, four worktree/commit/config identities, locks and hashes are in [launch_handoff.json]({root/'launch_handoff.json'}).\n\n'''
    for name,command in commands.items():text+=f'## {name}\n\n```bash\n{command}\n```\n\n'
    text+='## Expected results\n\n'+f"B1 ready: `{receipt['ready_paths']['B1']}`\n\nG ready: `{receipt['ready_paths']['G']}`\n\nCompute receipt: `{receipt['ready_paths']['compute_complete']}`\n\n"
    text+=f"Planning ranges in Sydney: B1 {eta['B1_pair']['earliest_sydney']} to {eta['B1_pair']['latest_sydney']}; G/four {eta['G_pair_four']['earliest_sydney']} to {eta['G_pair_four']['latest_sydney']}. New full-field diagnostics are unmeasured at launch; these are estimates, not confidence intervals. Noon remains a target and is likely exceeded.\n\n"
    text+='The queue automatically proceeds past B1 ready into G. It requires no LLM or user resume to compute. Pause the AI goal only after independent host-shell acceptance; do not signal or close the background runner. Scientific final analysis awaits the user resume.\n'
    (root/'launch_handoff.md').write_text(text);print(root/'launch_handoff.md')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();create(a.config)
