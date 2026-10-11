"""Clone a complete bundle to an actual isolated directory and execute its entrypoints."""
import argparse,json,subprocess,sys
from pathlib import Path
from scripts.bs4.entry import export_deployment,deploy
from scripts.bs4.runner import source_map
from scripts.crgf.common import atomic_json,identity,sha256,verify_outputs

def run(root):
    root=Path(root)
    existing=root/'tests/deployment_test.json'
    if existing.exists():
        previous=json.loads(existing.read_text());verify_outputs(previous)
        if previous['source']!=source_map(Path(__file__).resolve().parents[3]):raise ValueError('deployed rehearsal source changed')
        return
    formal=json.loads((root/'configs/queue.json').read_text());review=Path(formal['review_receipt'])
    if review.exists() and json.loads(review.read_text()).get('approved_to_launch'):raise ValueError('deployment rehearsal must never call an already-approved formal launch')
    manifest=export_deployment(root);m=json.loads(manifest.read_text());target=root/'tests/deployment'/m['commit'];m['target']=str(target)
    rehearsal=root/'source'/('deployment_rehearsal_'+m['commit']+'.json');atomic_json(rehearsal,m)
    result=deploy(rehearsal);assert result==target;assert deploy(rehearsal)==target
    assert source_map(target)==m['files']
    commands={}
    for command in ('prepare','status','tail','process','complete-check'):
        result=subprocess.run([sys.executable,'-m','scripts.bs4.entry',command,'--root',str(root)],cwd=target,text=True,capture_output=True,timeout=120)
        expected=2 if command=='complete-check' else 0
        if result.returncode!=expected:raise ValueError('deployed CLI failure '+command+result.stderr)
        commands[command]=dict(exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr)
    # Exercise the deployed one-command path up to its concrete admission/review gate.
    before=(root/'runner_identity.json').exists()
    gate=subprocess.run([str(target/'scripts/bs4/prepare_and_launch.sh'),'--root',str(root),'--deployment',str(rehearsal)],cwd=target,text=True,capture_output=True,timeout=300)
    if gate.returncode==0 or (root/'runner_identity.json').exists()!=before:raise ValueError('pre-review deployed launch escaped admission gate')
    if not any(x in gate.stderr for x in ('required admission failures','independent reviewer','required before formal launch')):raise ValueError('unexpected deployed launch failure '+gate.stderr)
    commands['launch_approval_gate']=dict(exit_code=gate.returncode,stdout=gate.stdout,stderr=gate.stderr)
    # A complete real regional biological chain executes from the deployed tree.
    # Its output lives beside the checkout so no generated file changes that tree.
    mini=root/'tests/deployment'/('mini_'+m['commit'])
    log=root/'tests/deployment'/('mini_'+m['commit']+'.log')
    if not (mini/'source/review_candidate.json').exists():
        with log.open('w') as stream:
            subprocess.run([sys.executable,'-m','scripts.bs4.mini','--queue',str(root/'configs/queue.json'),'--output',str(mini)],cwd=target,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=3600)
            if not (mini/'source/baseline_preparation.json').exists():subprocess.run([sys.executable,'-m','scripts.bs4.tests.deployment_baselines','--root',str(root),'--mini',str(mini),'--prepare-queue-only'],cwd=target,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=1800)
            subprocess.run([sys.executable,'-m','scripts.bs4.mini','--queue',str(mini/'configs/queue.json'),'--stages'],cwd=target,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=14400)
    for arm in ('P1','P2','T1','T2'):
        result=subprocess.run([sys.executable,'-m','scripts.bs4.entry','complete-check','--root',str(mini),'--arm',arm],cwd=target,text=True,capture_output=True,timeout=120)
        if result.returncode:raise ValueError('deployed real mini complete-check '+arm+result.stderr)
        commands['real_mini_complete_'+arm]=dict(exit_code=result.returncode,receipt=identity(mini/arm/'evaluation'/arm/'complete.json'))
    for command,extra in [('resume',[]),('evaluate',['--arm','P1'])]:
        args=[sys.executable,'-m','scripts.bs4.entry',command,'--root',str(root),*extra]
        if command=='resume':args.extend(['--deployment',str(rehearsal)])
        result=subprocess.run(args,cwd=target,text=True,capture_output=True,timeout=300)
        if result.returncode==0 or (root/'runner_identity.json').exists()!=before:raise ValueError('unreviewed deployed '+command+' escaped gate')
        if not any(x in result.stderr for x in ('required admission failures','admission has not approved','independent reviewer','required before formal launch')):raise ValueError('unexpected '+command+' failure '+result.stderr)
        commands[command+'_approval_gate']=dict(exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr)
    # Real execution from deployed source, not just hash/import checks.
    output=root/'tests/deployment'/('post_math_'+m['commit']+'.json')
    if not output.exists():subprocess.run([sys.executable,'-m','scripts.bs4.tests.post_math','--output',str(output)],cwd=target,check=True,timeout=300)
    subprocess.run([sys.executable,'-m','scripts.bs4.tests.deployment_context','--root',str(root),'--mini',str(mini)],cwd=target,check=True,timeout=1800)
    if not (mini/'source/review_receipt.json').exists():
        atomic_json(root/'tests/deployment_awaiting_review.json',dict(status='awaiting_independent_regional_review',candidate=identity(mini/'source/review_candidate.json'),target=str(target),mini=str(mini),deployment=str(rehearsal)))
        raise ValueError('Independent review of the real deployed regional test is required before entry positive controls')
    subprocess.run([sys.executable,'-m','scripts.bs4.tests.deployment_context','--root',str(root),'--mini',str(mini),'--deployment',str(rehearsal),'--finish'],cwd=target,check=True,timeout=1800)
    atomic_json(root/'tests/deployment_test.json',dict(status='passed',commit=m['commit'],tree=m['tree'],source=m['files'],bundle=m['bundle'],target=str(target),idempotent_deploy=True,entrypoints=commands,launch_safety='formal launch deliberately gated until independent review; not a simulated biological launch',outputs={'manifest':identity(rehearsal),'executed_math':identity(output),'real_four_arm_mini':identity(mini/'mini_complete.json'),'real_mini_log':identity(log),'entry_positive_controls':identity(mini/'source/entry_positive_tests.json')}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();run(a.root)
