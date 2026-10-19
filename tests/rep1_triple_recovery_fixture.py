"""Isolated actual-host native tiles -> splits -> worker/supervisor fixtures.

Only this test entry installs the existing synthetic controller harness. The
formal CLI rejects its schema and path. Stage05 executes the real native
adapter with the real worker callback and witness; other stages stay tiny.
"""
from pathlib import Path
import runpy,sys
ROOT=Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT/'scripts/rep1_triple_simple_boundary/bootstrap.py'),run_name='fixture_bootstrap')
import copy,json,os,time
import numpy as np
import tifffile
from scripts.rep1_triple_simple_boundary import runtime as rt,execution as ex,science,p2
from scripts.rep1_triple_simple_boundary.common import atomic_json,atomic_tiff,dependencies,digest,identity,inventory,read,write_receipt
from tests import rep1_triple_controller_fixture as harness


def inputs(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    pre=np.zeros((80,80),np.uint32);nuc=np.zeros_like(pre)
    for cid,(y,x) in enumerate(((0,0),(0,40),(40,0),(40,40)),1):
        pre[max(2,y):min(78,y+40),max(2,x):min(78,x+40)]=cid
        nuc[y+18:y+23,x+18:x+23]=cid
    for name,a in (('pre',pre),('nuc',nuc)):
        if not (root/(name+'.tif')).exists():atomic_tiff(root/(name+'.tif'),a)
    return root/'pre.tif',root/'nuc.tif'


def child_fault(*args,**kwargs):
    raise RuntimeError('owned native fork deterministic child failure')


def native_stage(cfg,stage,progress=None,lock_fds=(),validate_execution=None,compute_authorization=None):
    if stage!='05':return harness.fixture_stage(cfg,stage,progress,lock_fds,validate_execution,compute_authorization)
    validate_execution();root=Path(cfg['run_root']);behavior=read(root/'fixture_behavior.json')
    pre,nuc=inputs(root/'native_inputs');pred=root/'replay/SIMPLE_NATIVE/predictions'
    def callback(data):
        if data['stage']=='postprocess_B_splits' and behavior.get('split_bad_binding'):
            path=Path(data['detail']['receipt']);rec=read(path)
            rec['dependencies'].pop('run',None);rec['integrity']=digest({k:v for k,v in rec.items() if k!='integrity'})
            atomic_json(path,rec)
        progress(data)
    if behavior.get('native_child_failure'):p2.native.postprocess_connect=child_fault
    result=p2.process(pre,nuc,nuc,pred,'B',workers=2,patch_size=48,allowed_root=root,
        dependency_context=dependencies(cfg,'05'),callback=callback,lock_fds=lock_fds,
        interrupt_after=('splits',1) if behavior.get('split_interrupt') else None)
    if result['split_ids']<1:raise AssertionError('fixture never entered actual split transition')
    p2.validate_process(pred)
    return write_receipt(harness.fixture_receipt(cfg,stage),dependencies(cfg,stage),inventory(pred),
                         native_tiles_and_splits=True,scope='small synthetic native fixture only')


def install(config):
    cfg=harness.install(config);science.run_stage=native_stage
    entry=str(ROOT/'tests/rep1_triple_recovery_fixture.py')
    rt.entry_command=lambda c,p,a:[ex.PYTHON,'-I','-S','-B',entry,'entry',str(p),a]
    rt.supervisor_command=lambda c,p,a:[ex.PYTHON,'-I','-S','-B',entry,'supervise',str(p),a]
    rt.worker_command=lambda c,p,s,a,f:[ex.PYTHON,'-I','-S','-B',entry,'worker',str(p),s,a,','.join(map(str,f))]
    return cfg


def command(config,*args,success=True):
    import subprocess
    cfg=read(config);entry=Path(cfg['source_snapshot'])/'tests/rep1_triple_recovery_fixture.py'
    p=subprocess.run([ex.PYTHON,'-I','-S','-B',str(entry),args[0],str(config),*map(str,args[1:])],capture_output=True,text=True,timeout=50)
    root=Path(cfg['run_root'])/'logs';token=str(time.time_ns())
    (root/('fixture_command_'+token+'.stdout')).write_text(p.stdout);(root/('fixture_command_'+token+'.stderr')).write_text(p.stderr)
    if (p.returncode==0)!=success:raise AssertionError('fixture command wrong exit '+str(args)+' '+p.stdout+p.stderr)
    return p


def suite(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=False);checks={};runs={}
    for name,behavior in (('native_success',{}),('split_binding_failure',{'split_bad_binding':True}),
                          ('split_interrupted',{'split_interrupt':True}),('native_fork_failure',{'native_child_failure':True})):
        cfg=harness.initialize(root/name,behavior);config=Path(cfg['run_root'])/'configs/run.json';command(config,'launch')
        command(config,'launch',success=False)
        expected=rt.HOLD if name=='native_success' else 'FAILED'
        state=harness.wait_state(config,lambda s:s['status']==expected,timeout=150);harness.wait_release(cfg)
        base=Path(cfg['run_root']);exitrec=read(base/'control'/('supervisor_os_exit_'+state['attempt']+'.json'))
        if expected=='FAILED':
            assert exitrec['actual_os_exit_code']==1 and state['child_exit_codes']['05']==1
            assert not harness.fixture_receipt(cfg,'15').exists()
        else:
            events=[read(p) for p in (base/'control/witnesses'/state['attempt']/'05').glob('*.json')]
            split=[x for x in events if x['progress']['stage']=='postprocess_B_splits']
            assert split and all(x['validated_at_commit'] for x in split)
            assert exitrec['actual_os_exit_code']==0
            checks['real_process_split_accepted_by_actual_worker_witness']=identity(split[0]['witness']['path'])
            witness=Path(split[0]['witness']['path']);bad=read(witness);bad['dependencies']['run']['stage']='06'
            bad['integrity']=digest({k:v for k,v in bad.items() if k!='integrity'});wrong=base/'tamper/wrong_stage.json';atomic_json(wrong,bad)
            for called,path in (('06',witness),('05',wrong)):
                try:rt.witness(cfg,called,path)
                except ValueError as error:assert 'P2 scientific witness stage binding differs' in str(error)
                else:raise AssertionError('P2 split stage mismatch accepted')
            checks['actual_split_called_stage_and_resigned_envelope_stage_rejected']=True
        if name=='split_binding_failure':
            log=''.join(p.read_text() for p in (base/'logs').glob('05_*.log'))
            assert 'scientific witness lacks current config binding' in log
        if name=='native_fork_failure':
            log=''.join(p.read_text() for p in (base/'logs').glob('05_*.log'))
            assert 'owned native fork deterministic child failure' in log and '_RemoteTraceback' in log
        if name=='split_interrupted':
            rec=base/'replay/SIMPLE_NATIVE/predictions/postprocess/splits/000000.json';prior=identity(rec)
            atomic_json(base/'fixture_behavior.json',{});command(config,'resume','explicit owned split interruption recovery')
            harness.wait_state(config,lambda s:s['status']==rt.HOLD,timeout=150);harness.wait_release(cfg)
            assert identity(rec)==prior
            a=tifffile.imread(base/'replay/SIMPLE_NATIVE/predictions/final.tif')
            b=tifffile.imread(root/'native_success/replay/SIMPLE_NATIVE/predictions/final.tif')
            assert np.array_equal(a,b);checks['split_resume_matches_uninterrupted_exact_final']=True
        runs[name]=dict(config=identity(config),state=identity(base/'state.json'),os_exit=identity(base/'control'/('supervisor_os_exit_'+state['attempt']+'.json')))
    checks.update(duplicate_detached_launch_rejected=True,missing_binding_propagates_worker_supervisor_exit1=True,
        actual_native_fork_exception_propagates_worker_supervisor_exit1=True,old_data_not_written=True)
    atomic_json(root/'fixture_complete.json',dict(status='passed',scope='real native process/worker/witness/supervisor; tiny synthetic masks, no formal production',
        source_commit=ex.git('rev-parse','HEAD'),source_files=ex.source_files(),checks=checks,runs=runs))
    print(json.dumps(identity(root/'fixture_complete.json')))


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
        else:raise ValueError('unknown recovery fixture command')
