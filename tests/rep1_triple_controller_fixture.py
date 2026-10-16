"""Actual-host fixture of real runtime, isolated synthetic science only.

Formal CLI has no fixture option and rejects this schema/path. Patches exist
only in this test entry process; detached entry/supervisor/workers re-enter it.
"""
from pathlib import Path
import runpy,sys
ROOT=Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT/'scripts/rep1_triple_simple_boundary/bootstrap.py'),run_name='fixture_bootstrap')
import hashlib,importlib,json,multiprocessing as mp,os,signal,subprocess,time
from contextlib import ExitStack
from scripts.rep1_triple_simple_boundary import runtime as rt,execution as ex,science
from scripts.rep1_triple_simple_boundary.common import atomic_json,check_identity,check_receipt,dependencies,digest,identity,inventory,read,write_receipt
from scripts.rep1_triple_simple_boundary.contracts import STAGE_ORDER

def fixture_receipt(cfg,stage):
    return Path(cfg['run_root'])/('compute_complete.json' if stage=='14' else 'control/compute_exit.json' if stage=='15' else 'products/'+stage+'/complete.json')
def fixture_validate(cfg,stage,manifest=None):
    r=check_receipt(fixture_receipt(cfg,stage),dependencies(cfg,stage),cfg['run_root'])
    if stage=='15' and (r.get('terminal_state')!=rt.HOLD or r.get('exit_code')!=0):raise ValueError('fixture terminal incomplete')
    if stage=='14':
        for previous in STAGE_ORDER[:13]:check_receipt(fixture_receipt(cfg,previous),dependencies(cfg,previous))
    return r
def fork_child(args):
    root,fds,index=args;root=Path(root)
    rt.check_fds(read(root/'configs/run.json'),fds)
    atomic_json(root/f'fork_{index}.json',dict(identity=rt.process_identity(),fds=[dict(fd=f,path=os.readlink(f'/proc/self/fd/{f}')) for f in fds],owns_OFD=True))
    time.sleep(6);atomic_json(root/f'fork_{index}_done.json',dict(result=sum(i*i for i in range(10000))))
def fixture_stage(cfg,stage,progress=None,lock_fds=(),validate_execution=None,compute_authorization=None):
    validate_execution();root=Path(cfg['run_root']);out=root/'products'/stage;out.mkdir(parents=True,exist_ok=True);behavior=read(root/'fixture_behavior.json')
    for cursor in range(1,4):
        receipt=out/f'{cursor}.json'
        if not receipt.exists():
            value=sum(i*i for i in range(cursor*1000));data=out/f'{cursor}.data.json'
            atomic_json(data,dict(eligible=34442716,shape=[107955,313],assigned=value,fixture_calculation=value,scope='engineering synthetic only'))
            write_receipt(receipt,dependencies(cfg,stage),{'data':identity(data)},n_cells=107955,n_genes=313,endpoint='BASE_MATCH',assigned=value,scientific_result='negative/no-op/guardrail flagged')
            with (out/'visits.txt').open('a') as f:f.write(str(cursor)+'\n')
        progress(dict(stage='fixture_committed_calculation',cursor=cursor,total=3,detail={'receipt':str(receipt)}));time.sleep(.12)
        if behavior.get('fail_stage')==stage and cursor==2:raise RuntimeError('intentional worker nonzero')
        if behavior.get('orphan_stage')==stage and cursor==1:
            if behavior['orphan_kind']=='R':
                from scripts.rep1_triple_simple_boundary.evaluation import run_r
                r=root/'R_child';run_r(cfg,ROOT/'tests/rep1_triple_controller_R.R',[r,'orphan'],r,lock_fds,validate_execution)
            else:
                with mp.get_context('fork').Pool(2) as pool:pool.map(fork_child,[(str(root),lock_fds,i) for i in range(2)])
    if stage=='03':
        from scripts.rep1_triple_simple_boundary.evaluation import run_r
        r=out/'R';run_r(cfg,ROOT/'tests/rep1_triple_controller_R.R',[r,'fail' if behavior.get('R_fail') else 'source'],r,lock_fds,validate_execution)
        payload=importlib.import_module('tests.rep1_triple_controller_payload')
        atomic_json(out/'origins.json',dict(value=payload.VALUE,path=payload.__file__,cwd=str(Path.cwd()),runtime=rt.__file__))
    if stage=='14' and behavior.get('incomplete'):raise ValueError('missing required synthetic coverage refuses compute_complete')
    return write_receipt(fixture_receipt(cfg,stage),dependencies(cfg,stage),inventory(out),coordinate_rows=34442716,methods=list(science.EXISTING),scope='engineering synthetic only')

def install(config):
    cfg=read(config)
    if cfg.get('schema')!='rep1-triple-engineering-fixture-v1' or 'controller-fixtures' not in str(cfg['run_root']):raise ValueError('not isolated fixture')
    def load(path):
        if Path(path).resolve()!=Path(config).resolve() or read(path)!=cfg:raise ValueError('fixture config drift')
        return cfg
    class Validator:
        def __init__(self,c,fds):self.cfg=c;self.fds=fds;self.admission=identity(Path(c['run_root'])/'admission/not_formal.json');self()
        def __call__(self):
            rt.check_fds(self.cfg,self.fds);ex.verify_snapshot(self.cfg)
            if read(Path(self.cfg['run_root'])/'fixture_binding.json')['config_digest']!=digest(self.cfg):raise ValueError('fixture binding drift')
    rt.load_config=load;rt.execution_validator=Validator;science.run_stage=fixture_stage;science.validate_stage=fixture_validate;science.receipt_path=fixture_receipt
    entry=str(ROOT/'tests/rep1_triple_controller_fixture.py')
    rt.entry_command=lambda c,p,a:[ex.PYTHON,'-I','-S','-B',entry,'entry',str(p),a]
    rt.supervisor_command=lambda c,p,a:[ex.PYTHON,'-I','-S','-B',entry,'supervise',str(p),a]
    rt.worker_command=lambda c,p,s,a,f:[ex.PYTHON,'-I','-S','-B',entry,'worker',str(p),s,a,','.join(map(str,f))]
    return cfg

def initialize(root,behavior=None,live_clone=None):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    for name in ('configs','control','logs','admission','tmp','cache'):(root/name).mkdir()
    if live_clone:
        import shutil
        rows=[]
        for r in ex.source_files():
            p=Path(live_clone)/r['snapshot_relative'];p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(r['path'],p);rows.append(dict(**identity(p),snapshot_relative=r['snapshot_relative']))
        original=ex.source_files
        try:ex.source_files=lambda:rows;m=ex.copy_snapshot(root,ex.git('rev-parse','HEAD'))
        finally:ex.source_files=original
    else:m=ex.copy_snapshot(root,ex.git('rev-parse','HEAD'))
    cfg=dict(schema='rep1-triple-engineering-fixture-v1',run_id=root.name,run_root=str(root),source_snapshot=m['snapshot_root'],source_digest=m['source_digest'],source_commit=m['source_commit'],input_digest='engineering',domain_digest='engineering',plan_digest='engineering',environment_digest='engineering',python=ex.PYTHON,rscript=ex.RSCRIPT,r_libs_user=str(ex.R_LIBRARY),r_libs_site='',resources=dict(retained_bytes=dict(gt=1_000_000,p2_replay=1_000_000,analysis=1_000_000),shared_bytes=10_000_000,temporary_peak_bytes=1_000_000,reserve_bytes=256*1024**2,ram_allowance_bytes=4*1024**3,ram_reserve_bytes=256*1024**2,heartbeat_seconds=1,stale_seconds=60,no_progress_seconds=900))
    atomic_json(root/'configs/run.json',cfg);atomic_json(root/'state.json',rt.new_state(cfg));atomic_json(root/'fixture_binding.json',dict(config_digest=digest(cfg)));atomic_json(root/'fixture_behavior.json',behavior or {});atomic_json(root/'admission/not_formal.json',dict(approved_for_formal_launch=False,engineering_only=True,config_digest=digest(cfg)));return cfg
def command(config,*args,success=True):
    cfg=read(config);entry=Path(cfg['source_snapshot'])/'tests/rep1_triple_controller_fixture.py'
    p=subprocess.run([ex.PYTHON,'-I','-S','-B',str(entry),args[0],str(config),*map(str,args[1:])],capture_output=True,text=True,timeout=50)
    if (p.returncode==0)!=success:raise AssertionError('fixture command wrong exit '+str(args)+' '+p.stdout+p.stderr)
    return p
def wait_state(config,predicate,timeout=60):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        s=read(Path(config).parents[1]/'state.json')
        if predicate(s):return s
        if s['status']=='FAILED' and not predicate(s):raise AssertionError('fixture unexpectedly failed '+json.dumps(s))
        time.sleep(.15)
    raise AssertionError('fixture timeout '+json.dumps(s))
def wait_release(cfg,timeout=15):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        s=read(Path(cfg['run_root'])/'state.json')
        if all(not rt.lock_held(p) for p in rt.paths(cfg)) and all(rt.alive(s.get(k)) is False for k in ('entry','supervisor','worker')):return
        time.sleep(.15)
    raise AssertionError('owned fixture processes/locks not released naturally')
def inventory_tree(root):
    return {str(p.relative_to(root)):(sha256(p),p.stat().st_mtime_ns,p.stat().st_size) for p in Path(root).rglob('*') if p.is_file()}
def sha256(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def suite(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=False);checks={};evidence=[]
    clone=root/'live_clone';cfg=initialize(root/'normal',live_clone=clone);config=Path(cfg['run_root'])/'configs/run.json'
    keeper='triple-fixture-env-'+str(os.getpid());prior=dict(os.environ,TMPDIR=str(root/'prior'),MPLCONFIGDIR=str(root/'prior/mpl'),LD_PRELOAD='',R_LIBS='/prior/invalid')
    (root/'prior').mkdir();subprocess.run(['tmux','new-session','-d','-s',keeper,'exec sleep 180'],env=prior,check=True)
    try:
        command(config,'launch');command(config,'launch',success=False)
        active=wait_state(config,lambda s:s['heartbeat_sequence']>=2 and s['progress_sequence']>0)
        assert rt.alive(active['supervisor']) and rt.alive(active['entry']) and all(rt.lock_held(p) for p in rt.paths(cfg))
        env=dict(x.split('=',1) for x in Path(f'/proc/{active["supervisor"]["pid"]}/environ').read_text().split('\0') if '=' in x)
        assert all(env.get(k)==v for k,v in ex.controlled_environment(cfg).items())
        atomic_json(root/'controlled_environment.json',dict(expected=ex.controlled_environment(cfg),observed={k:env.get(k) for k in ex.controlled_environment(cfg)}))
        (clone/'tests/rep1_triple_controller_payload.py').write_text("VALUE='mutated_live_clone'\n")
        with (clone/'external_sources/publish_utils.R').open('a') as f:f.write('\nstop("mutated live clone must never run")\n')
    finally:subprocess.run(['tmux','kill-session','-t',keeper],capture_output=True)
    wait_state(config,lambda s:s['status']==rt.HOLD);wait_release(cfg);done=json.loads(command(config,'complete').stdout)
    assert done['verified'];before=inventory_tree(Path(cfg['run_root']));command(config,'status');command(config,'tail');assert inventory_tree(Path(cfg['run_root']))==before
    origins=read(Path(cfg['run_root'])/'products/03/origins.json');assert origins['value']=='reviewed_snapshot_value' and origins['path'].startswith(cfg['source_snapshot'])
    r=read(Path(cfg['run_root'])/'products/03/R/R_source.json');assert r['sha256']==sha256(Path(cfg['source_snapshot'])/'external_sources/publish_utils.R')
    command(config,'publication',success=False)
    for stage in STAGE_ORDER[:-1]:assert (Path(cfg['run_root'])/'products'/stage/'visits.txt').read_text().splitlines()==['1','2','3']
    checks.update(detached_after_dispatch_exit=True,duplicate_launch_rejected=True,two_fresh_heartbeats_and_committed_progress=True,three_exclusive_locks=True,negative_noop_guardrail_continues_all15=True,actual_supervisor_OS_exit0=True,status_tail_tree_hash_mtime_read_only=True,manual_publication_gate_rejects=True,live_clone_python_R_mutations_isolated=True,old_tmux_environment_overridden=True)
    for name,behavior in [('worker_nonzero',{'fail_stage':'02'}),('R_nonzero',{'R_fail':True}),('incomplete',{'incomplete':True})]:
        c=initialize(root/name,behavior);p=Path(c['run_root'])/'configs/run.json';command(p,'launch');s=wait_state(p,lambda s:s['status']=='FAILED');wait_release(c)
        assert not fixture_receipt(c,'15').exists();exitrec=read(Path(c['run_root'])/'control'/('supervisor_os_exit_'+s['attempt']+'.json'));assert exitrec['actual_os_exit_code']!=0
        if name=='worker_nonzero':
            saved=identity(fixture_receipt(c,'01'));atomic_json(Path(c['run_root'])/'fixture_behavior.json',{});command(p,'resume','explicit owned engineering fixture recovery');wait_state(p,lambda s:s['status']==rt.HOLD);wait_release(c);assert identity(saved['path'])==saved
            assert (Path(c['run_root'])/'products/02/visits.txt').read_text().splitlines()==['1','2','3'];command(p,'complete')
        checks[name+'_propagates_and_no_false_terminal']=True
    checks['explicit_resume_reuses_committed_chunks']=True
    for kind in ('R','P2_fork'):
        c=initialize(root/('orphan_'+kind),dict(orphan_stage='01',orphan_kind=kind));p=Path(c['run_root'])/'configs/run.json';command(p,'launch');base=Path(c['run_root']);marker=base/'R_child/R_grandchild_OFD.json' if kind=='R' else base/'fork_1.json'
        s=wait_state(p,lambda s:marker.exists() and s.get('worker') is not None)
        if kind=='R':
            proof=read(marker);assert len(proof['fds'])==3 and all(x['owns_same_OFD'] for x in proof['fds']);desc=read(base/'R_child/R_grandchild.json');driver=read(base/'R_child/R_identity.json');ids=[rt.process_identity(desc['pid']),rt.process_identity(driver['pid'])]
        else:ids=[read(base/f'fork_{i}.json')['identity'] for i in range(2)]
        for key in ('supervisor','worker'):
            assert rt.alive(s[key]);os.kill(s[key]['pid'],signal.SIGKILL)
        time.sleep(.4);assert all(rt.alive(x) for x in ids) and all(rt.lock_held(x) for x in rt.paths(c));command(p,'launch',success=False);command(p,'resume','orphan still holds lock',success=False)
        deadline=time.monotonic()+15
        while any(rt.lock_held(x) for x in rt.paths(c)) and time.monotonic()<deadline:time.sleep(.15)
        assert not any(rt.lock_held(x) for x in rt.paths(c));checks[kind+'_parent_dead_descendants_hold_same_OFD_reject_reentry']=True
        atomic_json(root/(kind+'_orphan_evidence.json'),dict(parents=s,descendants=ids,proof=read(marker),locks_held_after_parent_death=True,natural_release=True))
    drift=initialize(root/'source_drift');p=Path(drift['source_snapshot'])/'tests/rep1_triple_controller_payload.py';p.chmod(0o644);p.write_text("VALUE='bad snapshot'\n");p.chmod(0o444);command(Path(drift['run_root'])/'configs/run.json','launch',success=False);checks['snapshot_drift_refuses_launch']=True
    result=dict(schema='rep1-triple-controller-fixture-v1',status='passed',source_commit=ex.git('rev-parse','HEAD'),checks=checks,scope='engineering synthetic backend only; real host tmux/OFD/R/fork/runtime, no biological endpoints',source_files={str(p.relative_to(ROOT)):identity(p) for p in (ROOT/'scripts/rep1_triple_simple_boundary').glob('*') if p.is_file()},evidence={str(p.relative_to(root)):identity(p) for p in root.glob('*.json')})
    atomic_json(root/'fixture_complete.json',result);print(json.dumps(identity(root/'fixture_complete.json')))

if __name__=='__main__':
    cmd=sys.argv[1]
    if cmd=='suite':suite(sys.argv[2])
    else:
        config=Path(sys.argv[2]);cfg=install(config)
        if cmd=='launch':print(json.dumps(rt.launch(config)))
        elif cmd=='resume':print(json.dumps(rt.launch(config,True,sys.argv[3])))
        elif cmd=='entry':raise SystemExit(rt.entry(config,sys.argv[3]))
        elif cmd=='supervise':raise SystemExit(rt.supervise(config,sys.argv[3]))
        elif cmd=='worker':raise SystemExit(rt.worker(config,sys.argv[3],sys.argv[4],tuple(map(int,sys.argv[5].split(',')))))
        elif cmd=='status':print(json.dumps(rt.status(config)))
        elif cmd=='tail':print(rt.tail(config))
        elif cmd=='complete':print(json.dumps(rt.complete_check(config)))
        elif cmd=='publication':
            from scripts.rep1_triple_simple_boundary.common import require_manual_resume
            require_manual_resume(cfg,Path(cfg['run_root'])/'control/manual_resume.json')
        else:raise ValueError('unknown engineering command')
