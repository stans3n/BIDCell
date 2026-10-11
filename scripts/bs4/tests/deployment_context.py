"""Prepare and exercise an independently reviewed, explicitly regional deployment run."""
import argparse,json,subprocess,sys,time,copy,fcntl,shutil
from pathlib import Path
from scripts.crgf.common import atomic_json,identity,sha256,verify_outputs
from scripts.bs4.runner import stage_specs,signature,freeze,validate_queue
from scripts.bs4.validation import validate_stage
from scripts.bs4.entry import verify_review

def read(p):return json.loads(Path(p).read_text())
def prepare(root,mini):
    root=Path(root);mini=Path(mini);candidate=mini/'source/review_candidate.json'
    if (mini/'source/entry_positive_tests.json').exists():verify_outputs(read(mini/'source/entry_positive_tests.json'));return
    if candidate.exists():verify_outputs(read(candidate));return
    from scripts.bs4.comparison import aggregate
    from scripts.bs4.tests.aggregate_mini import SOURCES
    from scripts.crgf.common import source_identity
    execution=source_identity(SOURCES+['scripts/bs4/tests/deployment_context.py'])
    verify_outputs(read(mini/'source/baseline_preparation.json'))
    config=mini/'configs/queue.json';q=read(config);validate_queue(q)
    aggregate(config)
    if source_identity(SOURCES+['scripts/bs4/tests/deployment_context.py'])!=execution:raise ValueError('deployed aggregate source drift')
    atomic_json(mini/'aggregate_test.json',dict(status='passed',execution_source=execution,outputs={'comparison':identity(mini/'comparison/complete.json')}))
    freeze(config)
    original_status=read(mini/'run_status.json');atomic_json(mini/'source/biological_run_status.json',original_status)
    state=dict(run_id=q['run_id'],status='prepared_recovery_fixture',stages={})
    for spec in stage_specs(q):
        r=read(spec['receipt']);validate_stage(spec['stage'],r,q,spec)
        state['stages'][spec['name']]=dict(spec,status='succeeded',attempts=[],receipt_sha256=sha256(spec['receipt']),signature=signature(q,spec,state,str(config)),provenance='actual published biological mini receipt, strictly validated before independently reviewed recovery fixture')
    atomic_json(mini/'run_status.json',state);atomic_json(mini/'source/recovery_state_fixture.json',state);freeze(config)
    admission=dict(status='passed',approved_to_launch=True,required_failures=[],run_id=q['run_id'],scope='regional_deployment_test_only',outputs={s['name']:identity(s['receipt']) for s in stage_specs(q)},queue=identity(config),frozen_source=identity(mini/'source/frozen.json'))
    atomic_json(mini/'admission.json',admission)
    atomic_json(candidate,dict(status='awaiting_independent_review',review_scope='regional_deployment_test',output_root=str(mini.resolve()),run_id=q['run_id'],outputs={'queue':identity(config),'admission':identity(mini/'admission.json'),'frozen':identity(mini/'source/frozen.json'),'biological':identity(mini/'mini_complete.json'),'aggregate':identity(mini/'aggregate_test.json'),'recovery_state':identity(mini/'source/recovery_state_fixture.json')},review_receipt=q['review_receipt']))

def finish(root,mini,deployment):
    root=Path(root);mini=Path(mini);q=read(mini/'configs/queue.json');verify_review(mini/'configs/queue.json',q)
    # This exact genuine mini approval cannot authorize another root/run/config.
    formal=read(root/'configs/queue.json');bad=dict(formal,review_receipt=q['review_receipt'])
    try:verify_review(root/'configs/queue.json',bad)
    except (ValueError,KeyError):formal_rejected=True
    else:raise AssertionError('regional approval accepted for formal run')
    scope_faults=[];faultdir=mini/'tests/entry_faults';faultdir.mkdir(parents=True,exist_ok=True)
    for field,value in [('output_root',str(mini/'wrong_root')),('run_id',q['run_id']+'_wrong')]:
        bad=dict(q);bad[field]=value
        try:verify_review(mini/'configs/queue.json',bad)
        except (ValueError,KeyError):scope_faults.append(field)
        else:raise AssertionError('review accepted changed '+field)
    wrongqueue=faultdir/'different_queue.json';atomic_json(wrongqueue,dict(q,run_id=q['run_id']+'_wrong'))
    try:verify_review(wrongqueue,q)
    except ValueError:scope_faults.append('queue_hash')
    else:raise AssertionError('review accepted changed queue bytes')
    wrongreview=faultdir/'wrong_source_review.json';altered=read(q['review_receipt']);altered['frozen_source_sha256']='0'*64;atomic_json(wrongreview,altered)
    try:verify_review(mini/'configs/queue.json',dict(q,review_receipt=str(wrongreview)))
    except ValueError:scope_faults.append('source_hash')
    else:raise AssertionError('review accepted changed source signature')
    # Positive/negative input validation uses a real copied config; originals stay immutable.
    from scripts.bs4.entry import verify_prepared
    fixture=faultdir/'input_copy';(fixture/'source').mkdir(parents=True,exist_ok=True);(fixture/'arm/configs').mkdir(parents=True,exist_ok=True)
    copied=fixture/'real_input.json';shutil.copyfile(q['methods']['P1']['config'],copied)
    copiedcfg=fixture/'queue.json';atomic_json(copiedcfg,dict(output_root=str(fixture),methods={'P1':dict(cost_root=str(fixture/'arm'))}))
    atomic_json(fixture/'arm/configs/frozen.json',dict(identities={'input':identity(copied)}))
    atomic_json(fixture/'source/frozen.json',dict(queue=identity(copiedcfg),worktrees=read(mini/'source/frozen.json')['worktrees']))
    verify_prepared(copiedcfg);copied.write_bytes(copied.read_bytes()+b' ')
    try:verify_prepared(copiedcfg)
    except ValueError:scope_faults.append('real_input_copy_drift')
    else:raise AssertionError('real input drift accepted')
    before={str(p):identity(p) for arm in q['methods'].values() for base in ('checkpoints','predictions','matrix','evaluation') for p in (Path(arm['cost_root'])/base).rglob('*') if p.is_file()}
    results={}
    def command(name,*args,expected=0):
        r=subprocess.run([sys.executable,'-m','scripts.bs4.entry',name,'--root',str(mini),*args],text=True,capture_output=True,timeout=300)
        if r.returncode!=expected:raise ValueError('deployed mini '+name+' failed: '+r.stderr)
        results[name+str(len(results))]=dict(exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr);return r
    frozen_before=identity(mini/'source/frozen.json');command('prepare');command('prepare')
    if identity(mini/'source/frozen.json')!=frozen_before:raise ValueError('prepare rewrote frozen source')
    for arm in ('P1','P2','T1','T2'):command('evaluate','--arm',arm)
    process_before={str(p):identity(p) for p in [mini/'runner_identity.json',mini/'launch.json',*list((mini/'launches').glob('*'))] if p.exists()}
    with (mini/'runner.lock').open('a+') as locked:
        fcntl.flock(locked,fcntl.LOCK_EX|fcntl.LOCK_NB)
        r=subprocess.run([sys.executable,'-m','scripts.bs4.entry','resume','--root',str(mini),'--deployment',str(deployment)],text=True,capture_output=True,timeout=30)
        process_after={str(p):identity(p) for p in [mini/'runner_identity.json',mini/'launch.json',*list((mini/'launches').glob('*'))] if p.exists()}
        if r.returncode==0 or 'BlockingIOError' not in r.stderr or process_before!=process_after:raise ValueError('duplicate lock did not reject before spawning')
        results['duplicate_lock']=dict(exit_code=r.returncode,stderr=r.stderr)
    command('resume','--deployment',str(deployment))
    deadline=time.monotonic()+300
    while not (mini/'compute_complete.json').exists() and time.monotonic()<deadline:time.sleep(.25)
    if not (mini/'compute_complete.json').exists():raise TimeoutError('real deployed recovery runner did not finish')
    command('complete-check')
    while read(mini/'run_status.json').get('status')!='compute_complete_awaiting_user_resume' and time.monotonic()<deadline:time.sleep(.1)
    final=read(mini/'run_status.json')
    if final['status']!='compute_complete_awaiting_user_resume' or any(s['attempts'] for s in final['stages'].values()):raise ValueError('recovery reran completed biological stages')
    if before!={p:identity(p) for p in before}:raise ValueError('entry recovery/evaluate rewrote biological artifacts')
    for name in ('status','tail','process'):command(name)
    atomic_json(mini/'source/entry_positive_tests.json',dict(status='passed',scope='regional_deployment_test_only',approval=identity(q['review_receipt']),formal_approval_misuse_rejected=formal_rejected,rejected_scope_faults=scope_faults,prepare_frozen_identity_unchanged=True,all_biological_hashes_and_mtimes_preserved=True,actual_resumed_runner_no_stage_attempts=True,commands=results,outputs={'complete':identity(mini/'compute_complete.json')}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--mini',required=True);p.add_argument('--deployment');p.add_argument('--finish',action='store_true');a=p.parse_args()
    if a.finish:finish(a.root,a.mini,a.deployment)
    else:prepare(a.root,a.mini)
