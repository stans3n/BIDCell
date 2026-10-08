"""Freeze the shared implementation and four config-only branches/worktrees."""
import argparse,json,subprocess
from pathlib import Path
from scripts.crgf.common import atomic_json,identity
from .config import ARMS

def git(checkout,*args):return subprocess.check_output(['git',*args],cwd=checkout,text=True).strip()
def freeze(queuepath):
    queuepath=Path(queuepath);q=json.loads(queuepath.read_text());root=Path(q['output_root']);common=Path(__file__).resolve().parents[2]
    if (root/'runner_identity.json').exists():raise ValueError('cannot refreeze launched run')
    if git(common,'status','--porcelain'):raise ValueError('commit shared source before freezing')
    common_commit=git(common,'rev-parse','HEAD');records={}
    for arm in ARMS:
        worktree=common.parent/('BIDCell_cos_'+arm+'_'+q['run_id'].split('+')[0]);branch='feature/cos-'+arm.lower().replace('_','-')+'-'+q['run_id'].split('+')[0]
        if worktree.exists():raise FileExistsError(worktree)
        git(common,'worktree','add','-b',branch,str(worktree),common_commit)
        p=worktree/'scripts/cos/configs/arm.json';p.parent.mkdir(exist_ok=True)
        cfg=json.loads(Path(q['methods'][arm]['config']).read_text());atomic_json(p,cfg)
        git(worktree,'add','scripts/cos/configs/arm.json');git(worktree,'commit','-m','Freeze '+arm+' COS-min-v1 continuation configuration')
        commit=git(worktree,'rev-parse','HEAD');binding=dict(arm_id=arm,worktree=str(worktree),branch=branch,commit=commit)
        atomic_json(Path(q['methods'][arm]['cost_root'])/'source/binding.json',binding)
        q['methods'][arm].update(worktree=str(worktree),commit=commit);records[arm]=dict(**binding,config=identity(q['methods'][arm]['config']))
    atomic_json(queuepath,q)
    atomic_json(root/'source/branches.json',dict(common_worktree=str(common),common_commit=common_commit,arms=records,shared_implementation=True,config_only_arm_commits=True))
    from .runner import freeze as freeze_runner
    freeze_runner(queuepath)
    print(json.dumps(records,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();freeze(a.config)
