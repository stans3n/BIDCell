"""Actual-host engineering harness; never a valid formal admission/config.

The harness patches the scientific backend and formal-config loader only in its
own subprocess. Production CLI has no fixture option. Supervisor/worker/locks,
state, progress, source verifier, import guard and real R adapter are the real code.
"""
from pathlib import Path
import sys
import runpy

# Execute the actual isolated bootstrap initialization without its CLI dispatch.
REPO=Path(__file__).resolve().parents[1]
runpy.run_path(str(REPO/'scripts/rep1_bt_gt_fusion/bootstrap.py'),run_name='engineering_bootstrap')

from contextlib import ExitStack
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
from scripts.rep1_bt_gt_fusion import NODE_ORDER
from scripts.rep1_bt_gt_fusion import runtime as rt, execution as ex, science
from scripts.rep1_bt_gt_fusion.common import atomic_json,identity,read,digest,dependencies,check_receipt,write_receipt


def fixture_receipt(cfg,node):
    return Path(cfg['run_root'])/('production_complete.json' if node=='PRODUCTION_VALIDATE' else 'endpoints/'+node+'/complete.json')


def fixture_validate(cfg,node,manifest=None):
    return check_receipt(fixture_receipt(cfg,node),dependencies(cfg,node),cfg['run_root'])


def fixture_stage(cfg,node,progress=None,lock_fds=(),validate_execution=None):
    validate_execution();root=Path(cfg['run_root']);out=root/'endpoints'/node;out.mkdir(parents=True,exist_ok=True)
    if fixture_receipt(cfg,node).exists():return fixture_validate(cfg,node)
    behavior=read(root/'fixture_behavior.json')
    for cursor in range(1,7):
        p=out/(str(cursor)+'.json');data=out/(str(cursor)+'.data.json')
        if p.exists():check_receipt(p,dependencies(cfg,node),root)
        else:
            # A committed deterministic calculation, with a per-chunk receipt.
            value=sum(i*i for i in range(cursor*1000))
            atomic_json(data,dict(cursor=cursor,sum_squares=value))
            write_receipt(p,dependencies(cfg,node),{'data':identity(data)},cursor=cursor)
            with (out/'visits.txt').open('a') as f:f.write(str(cursor)+'\n')
        progress(dict(stage='fixture_calculation',cursor=cursor,total=6,detail=dict(node=node)))
        time.sleep(.3)
        if behavior.get('fail_node')==node and cursor==2:raise RuntimeError('intentional engineering child exit')
        if behavior.get('orphan_node')==node and cursor==1:
            from scripts.rep1_bt_gt_fusion.evaluation import run_r
            dest=root/'R_child';dest.mkdir(exist_ok=True)
            run_r(cfg,REPO/'tests/rep1_bt_gt_runtime_orphan.R',[dest],dest,lock_fds,validate_execution)
    # Delayed import and real R source resolution must remain in this snapshot.
    import importlib.util
    payload=REPO/'tests/rep1_bt_gt_runtime_payload.py'
    spec=importlib.util.spec_from_file_location('fixture_delayed_payload',payload)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    atomic_json(out/'origins.json',dict(payload=mod.VALUE,payload_file=str(payload),runtime=rt.__file__,cwd=str(Path.cwd())))
    from scripts.rep1_bt_gt_fusion.evaluation import run_r
    script=REPO/'tests/rep1_bt_gt_runtime_source.R';dest=out/'R_source';dest.mkdir(exist_ok=True)
    run_r(cfg,script,[dest,script],dest,lock_fds,validate_execution)
    return write_receipt(fixture_receipt(cfg,node),dependencies(cfg,node),
                         {p.name:identity(p) for p in out.iterdir() if p.is_file() and p.name!='complete.json'})


def install_fixture(config):
    cfg=read(config)
    if cfg.get('schema')!='rep1-bt-gt-engineering-fixture-v1':raise ValueError('not an engineering fixture')
    if 'engineering' not in Path(cfg['run_root']).parts and 'runtime-fixture' not in str(cfg['run_root']):raise ValueError('unsafe fixture root')
    def load(path):
        if Path(path).resolve()!=Path(config).resolve():raise ValueError('fixture config path drift')
        actual=read(path)
        if actual!=cfg:raise ValueError('fixture config bytes drift')
        return cfg
    def verify(c,fds=(),inputs=False):
        if c!=cfg:raise ValueError('fixture config mismatch')
        if fds:rt.check_fds(c,fds)
        origin=ex.verify_snapshot(c)
        saved=read(Path(c['run_root'])/'fixture_binding.json')
        if saved['config_digest']!=digest(c):raise ValueError('fixture frozen config drift')
        return dict(origin,engineering_only=True)
    rt.load_config=load;rt.verify_execution=verify
    science.run_stage=fixture_stage;science.validate_stage=fixture_validate;science.receipt_path=fixture_receipt
    entry=str(REPO/'tests/rep1_bt_gt_runtime_fixture.py')
    rt.worker_command=lambda c,n,a,f,p:[ex.PYTHON,'-I','-S','-B',entry,'worker',str(p),n,a,','.join(map(str,f))]
    rt.supervisor_command=lambda c,a,m,p:[ex.PYTHON,'-I','-S','-B',entry,'supervise',str(p),a,m]
    return cfg


def initialize(root,behavior=None,live_copy=None):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False)
    for name in ('configs','control','logs','admission','tmp','cache'): (root/name).mkdir()
    if live_copy is None:manifest=ex.copy_snapshot(root,'ENGINEERING_ONLY_NO_FORMAL_COMMIT')
    else:
        live_copy=Path(live_copy);live_copy.mkdir(parents=True,exist_ok=False);rows=[]
        for r in ex.source_files():
            p=live_copy/r['snapshot_relative'];p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(r['path'],p)
            rows.append(dict(**identity(p),snapshot_relative=r['snapshot_relative']))
        original=ex.source_files
        try:
            ex.source_files=lambda:rows
            manifest=ex.copy_snapshot(root,'ENGINEERING_ONLY_NO_FORMAL_COMMIT')
        finally:ex.source_files=original
    GiB=1024**3
    cfg=dict(schema='rep1-bt-gt-engineering-fixture-v1',run_id=root.name,run_root=str(root),source_snapshot=manifest['snapshot_root'],
             source_commit=manifest['source_commit'],source_digest=manifest['source_digest'],input_digest='fixture-no-scientific-inputs',
             plan_sha256=ex.PLAN_SHA,python=ex.PYTHON,rscript=ex.RSCRIPT,r_libs_user=str(ex.R_LIBRARY),r_libs_site='',
             environment_digest='engineering-only-not-an-environment-admission',node_order=list(NODE_ORDER),
             resources=dict(retained_bytes={n:100000 for n in NODE_ORDER},shared_bytes=100000,temporary_peak_bytes=100000,
                            reserve_bytes=GiB,ram_allowance_bytes=GiB,ram_reserve_bytes=GiB,heartbeat_seconds=1,stale_seconds=60,no_progress_seconds=900))
    atomic_json(root/'configs/run.json',cfg);atomic_json(root/'state.json',dict(rt.state_new(cfg),status='ADMITTED'))
    atomic_json(root/'fixture_binding.json',dict(config_digest=digest(cfg)))
    atomic_json(root/'fixture_behavior.json',behavior or {})
    atomic_json(root/'admission/not_formal.json',dict(approved_for_formal_launch=False,engineering_only=True))
    atomic_json(root/'admission/validated.json',dict(config_digest=digest(cfg),admission=identity(root/'admission/not_formal.json'),source_manifest=identity(root/'source_manifest.json')))
    return cfg


def run_command(config,*args,success=True):
    cfg=read(config);entry=Path(cfg['source_snapshot'])/'tests/rep1_bt_gt_runtime_fixture.py'
    p=subprocess.run([ex.PYTHON,'-I','-S','-B',str(entry),*args,str(config)],capture_output=True,text=True,timeout=45)
    if success and p.returncode:raise AssertionError(p.stdout+p.stderr)
    if not success and p.returncode==0:raise AssertionError('fixture command unexpectedly succeeded '+str(args))
    return p


def wait_state(config,predicate,timeout=40):
    until=time.monotonic()+timeout
    while time.monotonic()<until:
        state=read(Path(config).parents[1]/'state.json')
        if predicate(state):return state
        time.sleep(.15)
    raise AssertionError('fixture state timeout: '+json.dumps(state))


def release_wait(cfg,timeout=15):
    until=time.monotonic()+timeout
    while time.monotonic()<until:
        if not any(rt.lock_held(p) for p in rt.paths(cfg)):return
        time.sleep(.1)
    raise AssertionError('fixture locks did not release naturally')


def snapshot_tree(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(root).rglob('*') if p.is_file()}


def actual_suite(root):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False);checks={};evidence=[]
    # Normal real tmux queue, two fresh heartbeats and committed calculation.
    cfg=initialize(root/'normal');config=Path(cfg['run_root'])/'configs/run.json'
    run_command(config,'launch');run_command(config,'launch',success=False)
    running=wait_state(config,lambda s:s.get('heartbeat_sequence',0)>=2 and s.get('new_scientific_progress_sequence',0)>0)
    assert rt.alive(running['supervisor'])
    held=[rt.lock_held(p) for p in rt.paths(cfg)];assert all(held)
    before=snapshot_tree(cfg['run_root']);run_command(config,'status');run_command(config,'tail')
    # Worker may compute during status. The separate completed-state check below
    # verifies exact whole-tree read-only behavior with no concurrent writer.
    finished=wait_state(config,lambda s:s['status']==rt.HOLD);release_wait(cfg)
    assert all(x['status']=='validated_complete' for x in finished['nodes'].values())
    baseline=snapshot_tree(cfg['run_root']);run_command(config,'status');run_command(config,'tail')
    assert snapshot_tree(cfg['run_root'])==baseline
    for n in NODE_ORDER:assert (Path(cfg['run_root'])/'endpoints'/n/'visits.txt').read_text().splitlines()==list(map(str,range(1,7)))
    checks.update(actual_detached_tmux=True,duplicate_start_rejected=True,two_heartbeats_and_new_checkpoint=True,
                  identity_and_three_locks=True,serial_four_nodes_and_hold=True,status_tail_read_only=True)
    # Failed child halts successors; explicit retry skips valid complete node and
    # reuses committed chunks without duplicate visits.
    failed=initialize(root/'failure',dict(fail_node='P2_BASE_GT_PRODUCE'));fc=Path(failed['run_root'])/'configs/run.json'
    run_command(fc,'launch');s=wait_state(fc,lambda s:s['status']=='FAILED');release_wait(failed)
    assert s['child_exit_codes']['P2_BASE_GT_PRODUCE']!=0
    assert not (Path(failed['run_root'])/'endpoints/V2_MORPH_GT_PRODUCE').exists()
    first=identity(Path(failed['run_root'])/'endpoints/BT_EXISTING6/complete.json')
    atomic_json(Path(failed['run_root'])/'fixture_behavior.json',{})
    run_command(fc,'resume');wait_state(fc,lambda s:s['status']==rt.HOLD);release_wait(failed)
    assert identity(first['path'])==first
    for n in NODE_ORDER:assert (Path(failed['run_root'])/'endpoints'/n/'visits.txt').read_text().splitlines()==list(map(str,range(1,7)))
    checks.update(child_nonzero_halts_successors=True,completed_receipt_unchanged_on_retry=True,cursor_retry_without_duplicate_work=True)
    # Every killed PID below is obtained from this isolated fixture's own state.
    for kill_kind in ('supervisor','worker','both'):
        orphan=initialize(root/('orphan_'+kill_kind),dict(orphan_node='BT_EXISTING6'));oc=Path(orphan['run_root'])/'configs/run.json'
        run_command(oc,'launch');pidfile=Path(orphan['run_root'])/'R_child/R.pid'
        wait_state(oc,lambda s:pidfile.exists() and s.get('worker') is not None)
        state=read(Path(orphan['run_root'])/'state.json');rpid=int(pidfile.read_text());rid=rt.process_identity(rpid)
        inherited={key:os.readlink(f'/proc/{rpid}/fd/{fd}') for key,fd in zip(('launch','writer','compute'),
                    list(map(int,read(Path(orphan['run_root'])/'R_child/rep1_bt_gt_runtime_orphan_process.json')['lock_fds'])))}
        assert all(Path(v).resolve()==p.resolve() for v,p in zip(inherited.values(),rt.paths(orphan)))
        for key in (('supervisor','worker') if kill_kind=='both' else (kill_kind,)):
            assert rt.alive(state[key]);os.kill(state[key]['pid'],signal.SIGKILL)
        time.sleep(.4)
        assert rt.alive(rid) and all(rt.lock_held(p) for p in rt.paths(orphan))
        run_command(oc,'launch',success=False)
        status=json.loads(run_command(oc,'status').stdout)
        if kill_kind in ('supervisor','both'):assert status['observed_status']=='STALE'
        until=time.monotonic()+15
        while not (Path(orphan['run_root'])/'R_child/execution_status.txt').exists() and time.monotonic()<until:time.sleep(.1)
        assert (Path(orphan['run_root'])/'R_child/execution_status.txt').read_text().strip()=='succeeded'
        release_wait(orphan)
        evidence.append(dict(case=kill_kind,R_identity=rid,inherited_fd_paths=inherited))
    checks.update(orphan_R_after_parent_death=True,orphan_R_after_worker_death=True,orphan_R_after_both_deaths=True,orphan_R_blocks_reentry=True)
    # Snapshot isolation across later worker, delayed import, real R and retry.
    live=root/'isolated_live_copy'
    isolated=initialize(root/'isolation',dict(fail_node='P2_BASE_GT_PRODUCE'),live_copy=live);ic=Path(isolated['run_root'])/'configs/run.json'
    run_command(ic,'launch');wait_state(ic,lambda s:s['status']=='FAILED');release_wait(isolated)
    (live/'tests/rep1_bt_gt_runtime_payload.py').write_text("VALUE='unreviewed_live_changed'\n")
    (live/'tests/rep1_bt_gt_runtime_source.R').write_text('stop("unreviewed live R executed")\n')
    atomic_json(Path(isolated['run_root'])/'fixture_behavior.json',{})
    run_command(ic,'resume');wait_state(ic,lambda s:s['status']==rt.HOLD);release_wait(isolated)
    for n in NODE_ORDER:
        origin=read(Path(isolated['run_root'])/'endpoints'/n/'origins.json')
        assert origin['payload']=='audited_snapshot_original' and origin['cwd']==isolated['source_snapshot']
        dest=Path(isolated['run_root'])/'endpoints'/n/'R_source'
        assert (dest/'R_value.txt').read_text().strip()=='audited_snapshot_R_original'
        assert Path((dest/'R_source.txt').read_text().strip())==Path(isolated['source_snapshot'])/'tests/rep1_bt_gt_runtime_source.R'
    # This valid synthetic grant belongs only to the fixture schema/root and
    # cannot authorize the formal run. It is created after the fixture hold.
    run_command(ic,'manual-probe')
    assert (Path(isolated['run_root'])/'manual_probe/R_value.txt').read_text().strip()=='audited_snapshot_R_original'
    payload=Path(isolated['source_snapshot'])/'tests/rep1_bt_gt_runtime_payload.py';payload.chmod(0o644);payload.write_text("VALUE='drift'\n");payload.chmod(0o444)
    run_command(ic,'launch',success=False);run_command(ic,'resume',success=False)
    checks.update(live_copy_change_isolated_from_later_worker_delayed_import_and_retry=True,
                  live_R_source_change_isolated=True,simulated_manual_resume_uses_same_snapshot=True,
                  snapshot_drift_refuses_launch_and_retry=True)
    # Corrupt a completed receipt, then explicit recovery must fail closed.
    corrupt=initialize(root/'corrupt',dict(fail_node='P2_BASE_GT_PRODUCE'));cc=Path(corrupt['run_root'])/'configs/run.json'
    run_command(cc,'launch');wait_state(cc,lambda s:s['status']=='FAILED');release_wait(corrupt)
    target=Path(corrupt['run_root'])/'endpoints/BT_EXISTING6/1.data.json';target.write_text('{}')
    run_command(cc,'resume',success=False);checks['corrupt_complete_receipt_refuses_retry']=True
    result=dict(schema='rep1-bt-gt-controller-actual-host-fixture-v1',status='passed',checks=checks,orphan_evidence=evidence,
                scope='engineering only; no formal admission, historical inputs, true BT/GT, endpoint evaluation, publication or context writes')
    atomic_json(root/'receipt.json',result);print(json.dumps(result,indent=2))


def entry():
    command=sys.argv[1]
    if command=='suite':actual_suite(sys.argv[2]);return
    config=sys.argv[2] if command in ('worker','supervise') else sys.argv[-1]
    cfg=install_fixture(config)
    if command=='worker':rt.worker(config,sys.argv[3],sys.argv[4],tuple(map(int,sys.argv[5].split(','))));return
    if command=='supervise':rt.supervise(config,sys.argv[3],sys.argv[4]);return
    if command=='launch':result=rt.launch(config)
    elif command=='resume':result=rt.launch(config,resume=True)
    elif command=='status':result=rt.status(config)
    elif command=='tail':
        with (Path(cfg['run_root'])/'logs/queue.log').open() as f:print(f.read()[-4000:])
        return
    elif command=='manual-probe':
        rt.grant_resume(config,rt.utc())
        from scripts.rep1_bt_gt_fusion.common import require_manual_resume
        require_manual_resume(cfg,Path(cfg['run_root'])/'control/manual_resume.json')
        from scripts.rep1_bt_gt_fusion.evaluation import run_r
        dest=Path(cfg['run_root'])/'manual_probe';dest.mkdir()
        with ExitStack() as stack:
            fds=tuple(stack.enter_context(rt.lock(p)) for p in rt.paths(cfg))
            rt.verify_execution(cfg,fds)
            script=REPO/'tests/rep1_bt_gt_runtime_source.R'
            run_r(cfg,script,[dest,script],dest,fds,lambda:rt.verify_execution(cfg,fds))
        result=dict(engineering_manual_phase=True,snapshot=cfg['source_snapshot'])
    else:raise ValueError('unknown fixture command')
    print(json.dumps(result,indent=2))


if __name__=='__main__':entry()
