"""Actual science.run_stage06 -> produce -> matrix/oracle/diagnostics host fixture.

Only this isolated test entry substitutes small input cardinalities and fixture
admission. The production entry, kernels, callbacks, witness, worker, detached
supervisor and OS exits are exercised. Formal loaders reject this schema/path.
"""
from pathlib import Path
import runpy,sys
ROOT=Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT/'scripts/rep1_triple_simple_boundary/bootstrap.py'),run_name='fixture_bootstrap')
import copy,json,os,random,time
import numpy as np
import pandas as pd
import tifffile
from scripts.rep1_triple_simple_boundary import (runtime as rt,execution as ex,science,p2,
                                                inputs,matrix,nucleus_validation as nv,contracts)
from scripts.rep1_triple_simple_boundary.common import (atomic_json,atomic_npy,atomic_npz,atomic_tiff,
    check_receipt,dependencies,digest,identity,inventory,read,write_receipt)
from scripts.crgf.matrix import grid
from tests import rep1_triple_controller_fixture as harness
from tests.rep1_triple_recovery_fixture import inputs as masks

REAL_STAGE=science.run_stage
REAL_PROCESS=p2.process
MOLECULES=64


def tiny_inputs(cfg,callback,fds):
    root=Path(cfg['run_root']);pre,nuc=masks(root/'native_inputs')
    genes=root/'native_inputs/genes.txt';genes.write_text('\n'.join('G'+str(i) for i in range(313))+'\n')
    baseline=root/'native_inputs/baseline.yaml';baseline.write_text('fixture: true\n')
    yy,xx=np.meshgrid(np.arange(8)*11,np.arange(8)*11,indexing='ij')
    frame=pd.DataFrame(dict(transcript_id=np.arange(1,65),x_location=xx.ravel(),y_location=yy.ravel(),
                            feature_name=['G'+str(i%3) for i in range(64)],qv=30))
    transcripts=root/'native_inputs/transcripts.csv';frame.to_csv(transcripts,index=False)
    coordinates=[];cursor=0
    for i,block in enumerate(pd.read_csv(transcripts,chunksize=16)):
        x,y,_=grid(block,(80,80),1.,1.)
        a=np.empty(len(block),dtype=[('transcript_id','<u8'),('x','<i4'),('y','<i4'),('gene','<i4')])
        a['transcript_id']=block.transcript_id;a['x']=x;a['y']=y;a['gene']=[int(g[1:]) for g in block.feature_name]
        path=root/f'native_inputs/coordinates_{i}.npy';atomic_npy(path,a)
        coordinates.append(dict(**identity(path),rows=len(a),row_start=cursor,row_stop=cursor+len(a)));cursor+=len(a)
    replay=root/'replay/SIMPLE_NATIVE';pred=replay/'predictions'
    p2.process(pre,nuc,nuc,pred,'B',workers=2,patch_size=48,allowed_root=root,
               dependency_context=dependencies(cfg,'05'),callback=callback,lock_fds=fds)
    machine=dict(final_mask=str(pred/'final.tif'),pre_mask=str(pre),nuclei_mask=str(nuc),
        genes=str(genes),baseline_config=str(baseline),transcripts=str(transcripts),
        scale_x=1.,scale_y=1.,pixel_um=1.,chunksize=16)
    atomic_json(replay/'machine.json',machine)
    mat=matrix.build(machine,coordinates,replay/'matrix',dict(dependencies(cfg,'05'),part='matrix'),callback)
    parent=dict(name='MORPH_SIMPLE_FUSION',pre=str(pre),nuclei=str(nuc),final=str(pred/'final.tif'),
                matrix=str(replay/'matrix'),machine=str(replay/'machine.json'))
    manifest=dict(endpoints={'MORPH_SIMPLE_FUSION':parent},coordinates=coordinates,expected_cells=4)
    frozen=root/'frozen_domain';frozen.mkdir(exist_ok=True)
    cells=pd.DataFrame(dict(cell_id=np.arange(1,5),nuclear_crowding_quartile=np.arange(4),
        BASE_count_quartile=np.arange(4),eccentricity_quartile=np.arange(4),eccentricity=[.1,.2,.3,.4],
        original_type_index=[0,1,0,-1],original_type=['A','B','A','Unknown'],RNA_density_quartile=np.arange(4),
        nuclear_area_quartile=np.arange(4),nuclear_components_group=[1]*4))
    cells.to_csv(frozen/'cells.csv',index=False);atomic_json(root/'tiny_manifest.json',manifest)
    write_receipt(replay/'complete.json',dependencies(cfg,'05'),inventory(replay),
                  final_pixels_equal=True,scope='tiny_engineering_full_field',matrix=mat['totals'])
    return manifest


def child_fault(*args,**kwargs):raise RuntimeError('actual P2 fork failure in stage06 fixture')


def tiny_stage(cfg,stage,progress=None,lock_fds=(),validate_execution=None,compute_authorization=None):
    if stage=='05':
        validate_execution();tiny_inputs(cfg,progress,lock_fds);return
    if stage!='06':return harness.fixture_stage(cfg,stage,progress,lock_fds,validate_execution,compute_authorization)
    behavior=read(Path(cfg['run_root'])/'fixture_behavior.json')
    if behavior.get('p2_child_failure'):p2.connect=child_fault
    def observe(data):
        if data['stage']=='postprocess_P2_tiles' and behavior.get('tile_bad_stage'):
            path=Path(data['detail']['receipt']);r=read(path);r['dependencies']['base']['run']['stage']='P2_SIMPLE_POSTPROCESS'
            r['integrity']=digest({k:v for k,v in r.items() if k!='integrity'});atomic_json(path,r)
        progress(data)
    return REAL_STAGE(cfg,stage,observe,lock_fds,validate_execution,compute_authorization)


def tiny_validate(cfg,stage,manifest=None):
    if stage=='05':
        rec=check_receipt(harness.fixture_receipt(cfg,stage),dependencies(cfg,stage),cfg['run_root'])
        p2.validate_process(Path(cfg['run_root'])/'replay/SIMPLE_NATIVE/predictions');return rec
    if stage=='06':
        root=Path(cfg['run_root'])/'endpoints/P2_SIMPLE'
        rec=check_receipt(root/'production_complete.json',dependencies(cfg,'06'),cfg['run_root'])
        p2.validate_process(root/'predictions')
        inputs.matrix_evidence(dict(name='P2_SIMPLE',matrix=str(root/'matrix'),final=str(root/'final.tif')))
        assert read(root/'transfer_diagnostics.json')['postconnect_support_equal']
        nv.witness(cfg,'06',root/'predictions/postprocess'/ (nv.KIND+'.json'))
        return rec
    return harness.fixture_validate(cfg,stage,manifest)


def install(config):
    cfg=harness.install(config);root=Path(cfg['run_root'])
    science.run_stage=tiny_stage;science.validate_stage=tiny_validate
    science.config_check=lambda c: c if c==cfg and c['schema']=='rep1-triple-engineering-fixture-v1' else (_ for _ in ()).throw(ValueError('tiny config drift'))
    load=lambda c,*a,**kw:read(root/'tiny_manifest.json')
    science.load_inputs=inputs.load_inputs=load
    nv.EXPECTED_CELLS=4;matrix.ELIGIBLE=MOLECULES
    matrix.build.__defaults__=(None,MOLECULES)
    inputs.matrix_evidence.__defaults__=(4,313,MOLECULES,True)
    def process(*args,**kwargs):
        kwargs['patch_size']=48
        if args[4]=='P2' and read(root/'fixture_behavior.json').get('split_interrupt'):
            kwargs['interrupt_after']=('splits',1)
        return REAL_PROCESS(*args,**kwargs)
    p2.process=process
    entry=str(ROOT/'tests/rep1_triple_stage06_fixture.py')
    rt.entry_command=lambda c,p,a:[ex.PYTHON,'-I','-S','-B',entry,'entry',str(p),a]
    rt.supervisor_command=lambda c,p,a:[ex.PYTHON,'-I','-S','-B',entry,'supervise',str(p),a]
    rt.worker_command=lambda c,p,s,a,f:[ex.PYTHON,'-I','-S','-B',entry,'worker',str(p),s,a,','.join(map(str,f))]
    return cfg


def command(config,*args,success=True):
    import subprocess
    cfg=read(config);entry=Path(cfg['source_snapshot'])/'tests/rep1_triple_stage06_fixture.py'
    p=subprocess.run([ex.PYTHON,'-I','-S','-B',str(entry),args[0],str(config),*map(str,args[1:])],capture_output=True,text=True,timeout=50)
    token=str(time.time_ns());logs=Path(cfg['run_root'])/'logs'
    (logs/('command_'+token+'.stdout')).write_text(p.stdout);(logs/('command_'+token+'.stderr')).write_text(p.stderr)
    if (p.returncode==0)!=success:raise AssertionError(str(args)+p.stdout+p.stderr)
    return p


def initialize(root,behavior):
    cfg=harness.initialize(root,behavior)
    cfg['resources']['p2_workers']=2
    cfg['compute_authorization']=dict(mode='admitted_compute',allowed_stages=list(contracts.STAGE_ORDER))
    root=Path(cfg['run_root']);atomic_json(root/'configs/run.json',cfg);atomic_json(root/'state.json',rt.new_state(cfg))
    atomic_json(root/'fixture_binding.json',dict(config_digest=digest(cfg)))
    atomic_json(root/'admission/not_formal.json',dict(status='passed',open_blockers=[],approved_for_formal_launch=False,
        engineering_only=True,config_digest=digest(cfg),scope='tiny fixture authorization only, rejected by formal loader'))
    return cfg


def suite(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=False);runs={};checks={}
    for name,behavior in (('success',{}),('tile_binding_failure',{'tile_bad_stage':True}),
                          ('split_interrupted',{'split_interrupt':True}),('P2_fork_failure',{'p2_child_failure':True})):
        cfg=initialize(root/name,behavior);base=Path(cfg['run_root']);config=base/'configs/run.json'
        command(config,'launch');command(config,'launch',success=False)
        expected=rt.HOLD if name=='success' else 'FAILED'
        state=harness.wait_state(config,lambda s:s['status']==expected,timeout=180);harness.wait_release(cfg)
        exitrec=read(base/'control'/('supervisor_os_exit_'+state['attempt']+'.json'))
        assert exitrec['actual_os_exit_code']==(0 if expected==rt.HOLD else 1)
        if expected=='FAILED':assert state['child_exit_codes']['06']==1 and not harness.fixture_receipt(cfg,'15').exists()
        events=[read(p) for p in (base/'control/witnesses'/state['attempt']/'06').glob('*.json')]
        index=[e for e in events if e['progress']['stage']==nv.KIND]
        assert len(index)==1 and index[0]['validated_at_commit']
        if name=='success':
            kinds={e['progress']['stage'] for e in events}
            assert {'postprocess_P2_tiles','postprocess_P2_splits','matrix_own_owner_and_counts','matrix_independent_full_oracle',nv.KIND}<=kinds
            checks['actual_science06_produce_tiles_splits_matrix_oracle_diagnostics']=True
            checks['new_index_numerical_witness']=index[0]['witness']
        if name=='tile_binding_failure':
            assert 'P2 scientific witness stage binding differs' in ''.join(p.read_text() for p in (base/'logs').glob('06_*.log'))
        if name=='P2_fork_failure':
            log=''.join(p.read_text() for p in (base/'logs').glob('06_*.log'))
            assert 'actual P2 fork failure' in log and '_RemoteTraceback' in log
        if name=='split_interrupted':
            split=base/'endpoints/P2_SIMPLE/predictions/postprocess/splits/000000.json';prior=identity(split)
            initial_index=identity(base/'endpoints/P2_SIMPLE/predictions/postprocess'/(nv.KIND+'.json'))
            atomic_json(base/'fixture_behavior.json',{});command(config,'resume','explicit stage06 split fixture recovery')
            state=harness.wait_state(config,lambda s:s['status']==rt.HOLD,timeout=180);harness.wait_release(cfg)
            assert identity(split)==prior and identity(initial_index['path'])==initial_index
            a=tifffile.imread(base/'endpoints/P2_SIMPLE/final.tif');b=tifffile.imread(root/'success/endpoints/P2_SIMPLE/final.tif')
            assert np.array_equal(a,b)
            resumed=[read(p) for p in (base/'control/witnesses'/state['attempt']/'06').glob('*.json')]
            assert all(e['progress']['stage']!=nv.KIND for e in resumed)
            checks['resume_preserves_committed_index_and_split_no_false_fresh_validation']=True
        runs[name]=dict(config=identity(config),state=identity(base/'state.json'),
            os_exit=identity(base/'control'/('supervisor_os_exit_'+state['attempt']+'.json')))
    checks.update(duplicate_detached_launch_rejected=True,all_three_OFDs_release_naturally=True,
        actual_P2_fork_error_propagates_OS_exit1=True,wrong_embedded_stage_propagates_OS_exit1=True,
        lower_cardinalities_only_in_isolated_rejected_fixture=True)
    atomic_json(root/'fixture_complete.json',dict(status='passed',source_commit=ex.git('rev-parse','HEAD'),
        source_files=ex.source_files(),checks=checks,runs=runs,
        scope='actual stage06 production path on tiny inputs; fixture admission/cardinalities, no formal compute'))
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
        else:raise ValueError('unknown stage06 fixture command')
