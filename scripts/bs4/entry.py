"""One-command reviewed deployment, preparation, admission and detached execution."""
import argparse,json,os,subprocess,sys,tarfile,time,fcntl
from contextlib import ExitStack
from pathlib import Path
from scripts.crgf.common import atomic_json,identity,sha256,verify_outputs,digest
from scripts.crgf.run_egr import is_alive
from .evaluation import read
BASE='3569aa8f76d2945bdd4eec481ccfa4728ccdf794'

def verify_review(config,cfg):
    root=Path(cfg['output_root']);p=Path(cfg['review_receipt']);r=read(p)
    if not r.get('approved_to_launch') or r.get('required_failures') or r.get('independent') is not True:raise ValueError('independent reviewer did not approve final admission')
    who=r.get('reviewer',{})
    if not who.get('agent_id') or who.get('requested_model')!='gpt-6-astra' or who.get('requested_reasoning_effort')!='high':raise ValueError('reviewer identity/role missing')
    if r['run_id']!=cfg['run_id'] or r['queue_sha256']!=sha256(config) or r['frozen_source_sha256']!=sha256(root/'source/frozen.json'):raise ValueError('review/source/queue drift')
    evidence=r.get('evidence',{})
    if str(Path(cfg['admission'])) not in evidence:raise ValueError('review must bind admission receipt')
    for path,expected in evidence.items():
        if sha256(path)!=expected:raise ValueError('reviewed evidence changed '+path)
    return r

def freeze_inputs(config):
    q=read(config);root=Path(q['output_root'])
    if (root/'runner_identity.json').exists():raise ValueError('cannot refreeze launched run')
    for arm,s in q['methods'].items():
        s['commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=s['worktree'],text=True,timeout=30).strip()
    atomic_json(config,q)
    for arm,s in q['methods'].items():
        ar=Path(s['cost_root']);items={}
        for p in (ar/'configs').glob('*.json'):
            if p.name not in ('frozen.json','queue.json'):items['config_'+p.name]=identity(p)
        for k,obj in read(ar/'configs/inputs.json').items():
            if isinstance(obj,dict) and 'path' in obj and 'sha256' in obj:items['input_'+k]=obj
        for k in ('baseline_final_mask','nuclei_mask','preannotation','genes','transcripts','reference','positive_markers','negative_markers'):items['machine_'+k]=identity(read(s['machine'])[k])
        prepared=read(s['config'])['prepared_manifest']
        items['prepared']=identity(prepared);items['population']=q['population']
        manifest=read(prepared)
        for name,obj in manifest['outputs'].items():items['prepared_'+name]=obj
        for name in ('coverage.json','coverage_views_T1.json','coverage_views_T2.json'):
            items['teacher_'+name]=identity(Path(prepared).parent/name)
        for i,obj in enumerate(manifest.get('global_blocks',[])):
            items['global_block_'+str(i)]=obj
        items['calibration']=identity(root/'diagnostics/calibration.json');items['probes']=identity(root/'configs/training_probes.json')
        items['method_execution_sources']=identity(root/'tests/method_execution_sources.json')
        atomic_json(ar/'configs/frozen.json',dict(version='BS4-frozen-input-v1',identities=items))

def export_deployment(root):
    root=Path(root);q=read(root/'configs/queue.json');repo=Path(q['methods']['P1']['worktree'])
    from .runner import source_map
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True,timeout=30).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True,timeout=30).strip():raise ValueError('commit all implementation files before export')
    subprocess.run(['git','merge-base','--is-ancestor',BASE,commit],cwd=repo,check=True,timeout=300)
    bundle=root/'source/reviewed.bundle'
    if not bundle.exists():subprocess.run(['git','bundle','create',str(bundle),'HEAD'],cwd=repo,check=True,timeout=300)
    subprocess.run(['git','bundle','verify',str(bundle)],cwd=repo,check=True,stdout=subprocess.DEVNULL,timeout=300)
    m=dict(version='BS4-reviewed-deployment-v1',base_commit=BASE,commit=commit,tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=repo,text=True,timeout=30).strip(),target=str(repo),bundle=identity(bundle),files=source_map(repo))
    path=root/'source/deployment.json'
    if path.exists() and read(path)!=m:raise ValueError('deployment export drift')
    if not path.exists():atomic_json(path,m)
    return path

def deploy(manifest):
    m=read(manifest);target=Path(m['target']).resolve()
    if m['base_commit']!=BASE:raise ValueError('deployment base mismatch')
    from .runner import source_map
    bundle=Path(m['bundle']['path'])
    if sha256(bundle)!=m['bundle']['sha256']:raise ValueError('deployment bundle changed')
    if target.exists():
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=target,text=True,timeout=30).strip()
        if commit==m['commit'] and source_map(target)==m['files']:return target
        raise ValueError('refuse overwrite existing nonidentical checkout')
    subprocess.run(['git','clone','--no-checkout',str(bundle),str(target)],check=True,timeout=300)
    subprocess.run(['git','checkout','--detach',m['commit']],cwd=target,check=True,timeout=300)
    subprocess.run(['git','merge-base','--is-ancestor',BASE,m['commit']],cwd=target,check=True,timeout=300)
    if source_map(target)!=m['files']:raise ValueError('deployed source identity mismatch')
    return target


def status(q):
    root=Path(q['output_root']);s=read(root/'run_status.json') if (root/'run_status.json').exists() else {'status':'preparing'}
    actual_namespace=os.readlink('/proc/self/ns/pid')
    expected_namespace=s.get('pid_namespace')
    if expected_namespace and expected_namespace!=actual_namespace and s.get('runner',{}).get('boot_id')==Path('/proc/sys/kernel/random/boot_id').read_text().strip():
        s['runner_alive']=None;s['process_visibility']='namespace_mismatch_use_host_shell'
    else:s['runner_alive']=is_alive(s['runner']) if s.get('runner') else False
    s['observer_pid_namespace']=actual_namespace
    return s

def prepare_launch(root):
    """Bring a fresh reviewed deployment to the independent approval gate."""
    root=Path(root);config=root/'configs/queue.json'
    from .bootstrap import prepare as setup
    from .admission import execute_missing,collect
    from .runner import freeze
    if (root/'source/frozen.json').exists():
        q=read(config);admission=Path(q['admission'])
        if not admission.exists() or read(admission).get('status')!='passed':
            result=collect(root)
            if result['required_failures']:raise ValueError('required admission failures: '+str(result['required_failures']))
        return config
    setup(root)
    execute_missing(root)
    freeze_inputs(config)
    freeze(config)
    result=collect(root)
    if result['required_failures']:raise ValueError('required admission failures: '+str(result['required_failures']))
    return config

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','freeze','launch','resume','status','tail','process','evaluate','complete','complete-check','deploy','export','admission']);p.add_argument('--root',required=True);p.add_argument('--arm',choices=['P1','P2','T1','T2']);p.add_argument('--deployment');a=p.parse_args();root=Path(a.root).resolve();config=root/'configs/queue.json'
    if a.command=='export':print(export_deployment(root));return
    if a.command=='deploy':
        if not a.deployment:raise ValueError('--deployment required')
        print(deploy(a.deployment));return
    if a.command=='prepare':
        if (root/'source/frozen.json').exists():
            from .runner import preflight
            preflight(config,read(config));print('Validated frozen reviewed preparation');return
        from .bootstrap import prepare as setup
        setup(root)
        from .prepare import prepare
        prepare(root)
        print(config);return
    if a.command in ('launch','resume'):
        deployment=Path(a.deployment) if a.deployment else root/'source/deployment.json'
        target=deploy(deployment)
        if target!=Path(__file__).resolve().parents[2]:
            # Execute the deployed implementation, including its bootstrap paths.
            env=dict(os.environ,PYTHONPATH=str(target))
            subprocess.run([sys.executable,'-m','scripts.bs4.entry',a.command,'--root',str(root),'--deployment',str(deployment.resolve())],cwd=target,env=env,check=True,timeout=14400);return
        config=prepare_launch(root)
        q=read(config)
        if not Path(q['review_receipt']).exists():raise ValueError('Preparation and admission are concrete; independent reviewer approval receipt is required before formal launch')
        from .launch import launch
        launch(config);return
    q=read(config)
    if a.command=='admission':
        from .admission import execute_missing,collect
        execute_missing(root);result=collect(root);print(json.dumps(result,indent=2))
        if result['status']!='passed':raise SystemExit(2)
        return
    if a.command=='freeze':
        freeze_inputs(config)
        from .runner import freeze
        freeze(config);return
    if a.command=='status':print(json.dumps(status(q),indent=2));return
    if a.command=='process':
        s=status(q)
        if s.get('process_visibility'):raise RuntimeError('Host PID namespace required for process inspection; use a host shell/require_escalated')
        print(json.dumps({k:s.get(k) for k in ('status','runner','runner_alive','current_stage')},indent=2))
        if s.get('runner'):subprocess.run(['ps','-o','pid,ppid,pgid,sid,etime,lstart,args','-p',str(s['runner']['pid'])],check=False,timeout=30)
        return
    if a.command=='tail':
        s=status(q);current=s.get('current_stage');log=s.get('stages',{}).get(current,{}).get('log')
        if not log and (root/'launch.json').exists():log=read(root/'launch.json')['log']
        if log:subprocess.run(['tail','-n','50',log],check=True)
        else:print('No launch log yet')
        return
    if a.command=='evaluate':
        if not a.arm:raise ValueError('--arm required')
        s=status(q)
        if s.get('process_visibility'):raise RuntimeError('Host PID namespace required; never infer a hidden runner is dead')
        if s.get('runner_alive') or any(v.get('child_identity') and is_alive(v['child_identity']) for v in s.get('stages',{}).values()):raise RuntimeError('active runner/child; standalone evaluate refused')
        receipt=Path(q['methods'][a.arm]['evaluation'])/'complete.json'
        with ExitStack() as stack:
            locks=[stack.enter_context(path.open('a+')) for path in (root/'runner.lock',Path(q['gpu_lock']))]
            for lock in locks:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            from .runner import preflight,accept
            preflight(config,q)
            if receipt.exists() and read(receipt).get('status')=='succeeded':
                accept('evaluate',read(receipt),q,dict(method=a.arm));print('Validated completed evaluation; no rerun');return
            subprocess.run([q['python'],'-m','scripts.bs4.stage','--config',q['methods'][a.arm]['config'],'--stage','evaluate'],cwd=q['methods'][a.arm]['worktree'],pass_fds=tuple(x.fileno() for x in locks),check=True,timeout=q['stage_timeouts']['evaluate'])
        return
    if a.command in ('complete','complete-check'):
        f=root/(a.arm+'/evaluation/'+a.arm+'/complete.json' if a.arm else 'compute_complete.json');r=read(f);verify_outputs(r)
        if r.get('status')!='succeeded' or r.get('required_missing'):raise ValueError('required computation incomplete')
        for artifact in r.get('artifacts',{}).values():
            if sha256(artifact['path'])!=artifact['sha256']:raise ValueError('completed stage receipt changed')
            verify_outputs(read(artifact['path']))
        print(json.dumps(r,indent=2))
if __name__=='__main__':main()
