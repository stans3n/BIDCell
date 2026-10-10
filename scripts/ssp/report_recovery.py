"""Explicitly reviewed report-only recovery. Original training source stays frozen."""
import argparse,copy,fcntl,json,os,shutil,subprocess,time
from contextlib import ExitStack
from pathlib import Path
from scripts.crgf.common import atomic_json,identity,sha256,digest,process_identity,Monitor,verify_outputs
from scripts.crgf.run_egr import is_alive
from scripts.ssp.evaluation import read,times
from scripts.ssp.runner import source_map,complete_receipt,accept

VERSION='SSP-report-recovery-v1'
BASE='f1ec2c2f6fd027760b6eacf587eff4c15cb8b28d'


def check_review(root,recovery,baseline_path):
    approval_path=recovery/'approval.json'
    if not approval_path.exists():raise ValueError('independent report recovery approval required')
    a=read(approval_path);repo=Path(__file__).resolve().parents[2]
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip():raise ValueError('recovery checkout is dirty')
    subprocess.run(['git','merge-base','--is-ancestor',BASE,commit],cwd=repo,check=True)
    if a.get('independent') is not True or a.get('approved_report_recovery') is not True or a.get('required_failures'):raise ValueError('independent recovery review did not approve')
    who=a.get('reviewer',{})
    if not who.get('agent_id') or who.get('requested_model')!='gpt-6-astra' or who.get('requested_reasoning_effort')!='high':raise ValueError('independent reviewer role identity missing')
    if a.get('run_root')!=str(root) or a.get('scope')!='report_only' or a.get('original_commit')!=BASE:raise ValueError('recovery review scope mismatch')
    if a.get('recovery_commit')!=commit or a.get('recovery_source')!=source_map(repo):raise ValueError('recovery source drift')
    if a.get('baseline_sha256')!=sha256(baseline_path):raise ValueError('baseline drift')
    for path,expected in a.get('evidence',{}).items():
        if sha256(path)!=expected:raise ValueError('reviewed recovery evidence changed '+path)
    if not a.get('evidence'):raise ValueError('recovery tests/evidence missing')
    return dict(version=VERSION,original_commit=BASE,recovery_commit=commit,recovery_source=a['recovery_source'],baseline=identity(baseline_path),approval=identity(approval_path),scope='report only; no training, prediction, native processing, matrix building, or profile evaluation')


def verify_protected(root,baseline,check_stage_records=True):
    """Hash exact full successful trees, plus frozen controls and dependencies."""
    seen={};count=0
    expected=baseline['protected_files']
    actual_paths={str(p.resolve()) for name in ('S','SP','b_cost') for p in (root/name).rglob('*') if p.is_file()}
    expected_paths={p for p in expected if any(Path(p).is_relative_to(root/name) for name in ('S','SP','b_cost'))}
    if actual_paths!=expected_paths:raise ValueError('successful artifact tree membership changed')
    for path,obj in {**baseline['verified_dependency_files'],**expected}.items():
        p=Path(path);st=p.stat();key=(st.st_dev,st.st_ino)
        if key not in seen:seen[key]=sha256(p)
        if st.st_size!=obj['bytes'] or seen[key]!=obj['sha256']:raise ValueError('protected successful artifact changed '+path)
        count+=1
    state=read(root/'run_status.json')
    if check_stage_records:
        for name,item in baseline['successful_stages'].items():
            if state['stages'][name]!=item['record']:raise ValueError('successful stage record changed '+name)
    return dict(status='passed',files_verified=count,unique_inodes_hashed=len(seen),successful_stage_records_unchanged=True,**times())


def ensure_inactive(state):
    if state.get('pid_namespace')!=os.readlink('/proc/self/ns/pid'):raise ValueError('host PID namespace required')
    if state.get('runner') and is_alive(state['runner']):raise ValueError('original runner still active')
    for stage in state['stages'].values():
        if stage.get('child_identity') and is_alive(stage['child_identity']):raise ValueError('original child still active')


def execute(root):
    root=Path(root).resolve();recovery=root/'recovery/report_v1';baseline_path=recovery/'baseline.json';baseline=read(baseline_path)
    provenance=check_review(root,recovery,baseline_path);q=read(root/'configs/queue.json');state=read(root/'run_status.json')
    if baseline['status']!='passed' or baseline['original_commit']!=BASE:raise ValueError('baseline invalid')
    ensure_inactive(state)
    with ExitStack() as stack:
        for path in (root/'runner.lock',Path(q['gpu_lock'])):
            lock=stack.enter_context(path.open('r'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (recovery/'complete.json').exists():
            done=read(recovery/'complete.json')
            if done['status']!='succeeded' or done['provenance']!=provenance:raise ValueError('prior recovery identity mismatch')
            verify_protected(root,baseline);verify_outputs(done);accept('report',read(root/'comparison/complete.json'))
            print('Validated completed report recovery; no work repeated',flush=True);return
        if (recovery/'attempt.json').exists():raise ValueError('prior recovery attempt exists; preserve failure and independently review a new recovery version')
        if identity(root/'run_status.json')['sha256']!=baseline['original_run_status']['sha256']:raise ValueError('original failed state changed before approved recovery')
        verified=verify_protected(root,baseline)
        for rel,obj in baseline['partial_comparison'].items():
            if sha256(root/'comparison'/rel)!=obj['sha256']:raise ValueError('failed partial comparison changed '+rel)
        # Conservative allocation for a new report plus per-cell reason tables; retain 15 GiB.
        additional=int(1.2*(sum(x['bytes'] for x in baseline['partial_comparison'].values())+2*1024**3))
        if shutil.disk_usage(root).free<15*1024**3+additional:raise RuntimeError('PAUSED_RESOURCE report recovery space including buffer/reserve insufficient')
        archive=recovery/'failed_attempt'
        archive.mkdir(exist_ok=False)
        for name in ('run_status.json','compute_complete.json','artifact_index.md','remaining_issues.md'):
            path=root/name
            if path.exists():shutil.copyfile(path,archive/name)
        log=state['stages']['report'].get('log')
        if log:shutil.copyfile(log,archive/'report_attempt1.log')
        shutil.copytree(root/'comparison',archive/'comparison',copy_function=os.link)
        atomic_json(archive/'preservation.json',dict(original_status=baseline['original_run_status'],original_compute_complete=baseline['original_compute_complete'],partial_comparison=baseline['partial_comparison'],policy='original failed report files retained by hardlink; recovery writes only new files in separate directory'))
        start=time.monotonic();attempt=dict(version=VERSION,scope='report_only',start=times(),process=process_identity(),provenance=provenance,baseline_verification=verified)
        atomic_json(recovery/'attempt.json',attempt)
        original_record=copy.deepcopy(state['stages']['report']);record=state['stages']['report']
        attempt_record=dict(attempt=len(record['attempts'])+1,start=times(),command=['python','-m','scripts.ssp.report_recovery','--root',str(root),'--execute'],log=str(recovery/'execution.log'),recovery_version=VERSION,original_frozen_signature=record.get('signature'),recovery_signature=digest(provenance))
        record['attempts'].append(attempt_record);record.update(status='running',recovery=provenance)
        state.update(status='report_recovery_running',current_stage='report',recovery_process=attempt['process']);atomic_json(root/'run_status.json',state)
        out=recovery/'comparison'
        try:
            from scripts.ssp.comparison import aggregate
            print('Running reviewed report only; successful stages are read-only',flush=True)
            with Monitor(recovery/'resources.json'):aggregate(root/'configs/queue.json',output_dir=out)
            receipt=read(out/'complete.json');receipt['recovery']=provenance;receipt['original_failed_report']=original_record;receipt['next_state']='awaiting_independent_scientific_acceptance_after_user_resume';atomic_json(out/'complete.json',receipt);accept('report',receipt)
            unchanged=verify_protected(root,baseline)
            atomic_json(recovery/'unchanged_after_report.json',unchanged)
            if check_review(root,recovery,baseline_path)!=provenance:raise ValueError('recovery source/approval changed during report')
            # Keep all failed partial files in archive; switch the canonical report entry atomically.
            original=root/'comparison';os.rename(original,recovery/'original_comparison')
            link=root/'comparison.recovery-link';link.symlink_to(out,target_is_directory=True);os.replace(link,original)
            attempt_record.update(end=times(),exit_code=0,wall_seconds=time.monotonic()-start)
            record.update(status='succeeded',receipt_sha256=sha256(out/'complete.json'),child_identity=None,exit_code=0)
            record.pop('failure',None);record.pop('reason',None)
            state.update(status='compute_complete_awaiting_independent_acceptance',current_stage=None,report_recovery=provenance,user_resume_already_authorized=True,heartbeat=times())
            # Existing complete_receipt validates all stage outputs and retains the failed attempt.
            if not complete_receipt(q,state):raise ValueError('required compute completion failed after report')
            result=read(root/'compute_complete.json');result['report_recovery']=provenance;result['original_failed_compute_complete']=baseline['original_compute_complete'];result['next_state']='awaiting_independent_scientific_acceptance_after_user_resume';atomic_json(root/'compute_complete.json',result)
            atomic_json(root/'run_status.json',state)
            for name,item in baseline['successful_stages'].items():
                if state['stages'][name]!=item['record']:raise ValueError('successful stage record mutation '+name)
            atomic_json(recovery/'complete.json',dict(status='succeeded',version=VERSION,provenance=provenance,successful_stages_unchanged=True,extra_optimizer_updates=0,profile_evaluation_reruns=0,original_failure_preserved=str(archive),wall_seconds=time.monotonic()-start,outputs={'comparison':identity(out/'complete.json'),'compute':identity(root/'compute_complete.json'),'unchanged':identity(recovery/'unchanged_after_report.json')},**times()))
            repo=Path(__file__).resolve().parents[2]
            command=f'cd {repo} && {q["python"]} -m scripts.ssp.report_recovery --root {root} --execute'
            (recovery/'handoff_supplement.md').write_text('Report-only recovery completed after the user manually resumed. Original training/evaluation source, admission, configurations, all eleven successful stage records and artifacts remain unchanged. Original failed report evidence is retained in failed_attempt/. The original comparison path now points to the separately versioned recovered report.\n\nStatus: '+str(root/'run_status.json')+'\nRecovery: '+str(recovery/'complete.json')+'\n\nIdempotent recovery validation command (host shell):\n`'+command+'`\n\nIndependent scientific acceptance is now authorized and pending the independent analyst; no further user resume is required.\n')
            print('REPORT RECOVERY SUCCEEDED',flush=True)
        except BaseException as exc:
            attempt_record.update(end=times(),exit_code=1,wall_seconds=time.monotonic()-start,failure=repr(exc))
            record.update(status='failed',failure=repr(exc));state.update(status='failed',current_stage=None)
            atomic_json(root/'run_status.json',state);atomic_json(recovery/'failure.json',dict(status='failed',error=repr(exc),provenance=provenance,**times()));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    if not a.execute:raise SystemExit('Use --execute only after independent recovery approval; this driver never runs any successful stage.')
    execute(a.root)
