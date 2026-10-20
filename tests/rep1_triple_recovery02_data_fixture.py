"""Bounded real107955-index worker and exact two-donor consumers; no production.

The real stage06 entry is stopped before constructing the P2 pool, after its
new numerical validation witness and four explicit tile adoptions. Everything
written is a small engineering artifact in tmpfs, with a rejected fixture schema.
"""
from pathlib import Path
import runpy,sys
ROOT=Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT/'scripts/rep1_triple_simple_boundary/bootstrap.py'),run_name='fixture_bootstrap')
import copy,json,os,shutil,subprocess,time
from contextlib import ExitStack
from scripts.rep1_triple_simple_boundary import (continuation as c,continuation2 as c2,
    execution as ex,runtime as rt,science,p2,inputs,contracts,nucleus_validation as nv)
from scripts.rep1_triple_simple_boundary.common import atomic_json,dependencies,digest,identity,read
from tests.rep1_triple_continuation_fixture import partitions,reference_rejections


def assemble(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    for name in ('configs','control','logs','admission','tmp','cache'):(root/name).mkdir()
    source=ex.copy_snapshot(root,ex.git('rev-parse','HEAD'))
    resources=rt.resource_contract()
    resources.update(retained_bytes=dict(gt=0,p2_replay=32*1024**2,analysis=4*1024**2),
                     shared_bytes=64*1024**2,temporary_peak_bytes=256*1024**2,reserve_bytes=256*1024**2)
    partial=dict(schema='rep1-triple-recovery02-data-fixture-v1',run_id=root.name,run_root=str(root),
        source_snapshot=source['snapshot_root'],source_commit=source['source_commit'],source_digest=source['source_digest'],
        resources=resources,compute_authorization=dict(mode='admitted_compute',allowed_stages=list(contracts.STAGE_ORDER)))
    started=time.monotonic();cont=c2.build(partial,c2.PROOF,c2.BASELINE);build_seconds=time.monotonic()-started
    donor=read(c2.DONOR/'configs/run.json')
    for name,key in (('inputs_manifest.json','input_manifest'),('comparisons.json','comparison_manifest')):
        shutil.copyfile(donor[key],root/'configs'/name)
    cfg=dict(partial,continuation=cont,input_manifest=str(root/'configs/inputs_manifest.json'),input_digest=donor['input_digest'],
        domain_manifest=donor['domain_manifest'],domain_digest=donor['domain_digest'],plan_manifest=donor['plan_manifest'],
        plan_digest=donor['plan_digest'],comparison_manifest=str(root/'configs/comparisons.json'),
        environment_digest='engineering mutation closure; no formal admission',stage_order=list(contracts.STAGE_ORDER),
        python=ex.PYTHON,rscript=ex.RSCRIPT)
    atomic_json(root/'configs/run.json',cfg);c.install_stages(cfg);atomic_json(root/'state.json',rt.new_state(cfg))
    atomic_json(root/'admission/not_formal.json',dict(status='passed',open_blockers=[],approved_for_formal_launch=False,
        engineering_only=True,config_digest=digest(cfg),scope='bounded tmpfs numerical validation only'))
    m=c.manifest(cfg);old=read(c2.DONOR/'admission/preflight.json')
    paths=set(old['fingerprints'])|set(c.fingerprint_paths(m))
    paths.update(str(p) for p in (root/'source_snapshot').rglob('*'))
    paths.update((str(root/'source_snapshot'),str(root/'source_manifest.json'),str(root/'configs/run.json'),cont['path']))
    paths.update(item['path'] for item in c.evidence_items(m))
    atomic_json(root/'admission/fixture_mutation_set.json',dict(paths=ex.fingerprints(paths),
        scope='Complete original input/domain/environment closure, both producer science/source closures and new source/config/proofs. Formal preparation/admission does not exist.'))
    atomic_json(root/'admission/build_measurement.json',dict(seconds=build_seconds,continuation=cont,source_files=ex.source_files(),
        fixture_resources=resources,formal_resources_unchanged=rt.resource_contract(),
        storage_scope='Bounded tmpfs fixture, never a forecast or admission for a formal full queue'))
    return cfg


def worker(config,fds):
    cfg=read(config);root=Path(cfg['run_root'])
    if cfg['schema']!='rep1-triple-recovery02-data-fixture-v1' or 'recovery02-fixtures' not in str(root):raise ValueError('not isolated Recovery02 fixture')
    bound=identity(config);phases={};start=time.monotonic()
    def load(path):
        if identity(path)!=bound:raise ValueError('fixture config changed')
        return cfg
    class Validator(ex.ExecutionValidator):
        def __init__(self,cfg,fds):
            self.cfg,self.fds=cfg,fds;self.admission=identity(root/'admission/not_formal.json')
            ex.verify_snapshot(cfg);c.verify(cfg);load(config)
            proof=root/'admission/fixture_mutation_set.json';self.expected=read(proof)['paths']
            self.extra=ex.fingerprints([proof,self.admission['path']]);times=[]
            for _ in range(3):
                at=time.monotonic();super().__call__();times.append(time.monotonic()-at)
            phases['actual_ExecutionValidator_call_seconds']=times;phases['mutation_paths']=len(self.expected)
    original_config=science.config_check
    def config_check(current):
        if current!=cfg:raise ValueError('engineering config drift')
        return original_config(dict(current,schema=contracts.CONFIG_SCHEMA))
    class StopBeforePool:
        def __init__(self,*args,**kwargs):pass
        def __enter__(self):raise InterruptedError('bounded real stage06 fixture: validated index and adopted4, before remaining44 P2 pool')
        def __exit__(self,*args):pass
    def timed(module,name,label):
        original=getattr(module,name)
        def call(*args,**kwargs):
            at=time.monotonic()
            try:return original(*args,**kwargs)
            finally:phases.setdefault(label,[]).append(time.monotonic()-at)
        setattr(module,name,call)
    timed(p2,'FullNucleusIndex','actual_index_construction_seconds')
    timed(nv,'actual_index_payload','actual_complete_numeric_validation_seconds')
    timed(nv,'validate_payload','independent_witness_numeric_validation_seconds')
    timed(rt,'witness','actual_runtime_witness_seconds')
    rt.load_config=load;rt.execution_validator=Validator;science.config_check=config_check;p2.ProcessPoolExecutor=StopBeforePool
    try:return rt.worker(config,'06','engineering-01',fds)
    finally:
        phases['worker_elapsed_seconds']=time.monotonic()-start
        atomic_json(root/'admission/checkpoint_measurements.json',phases)


def rejections(cfg):
    root=Path(cfg['run_root']);original=read(cfg['continuation']['path']);badroot=root/'tamper';badroot.mkdir()
    rejected=[]
    for label in ('unknown_overlay','other_donor','base_retarget','stage05_producer','credit_double_count','fingerprint_omission'):
        obj=copy.deepcopy(original);fake=dict(cfg);target=badroot/label/'configs/continuation.json'
        fake.update(run_root=str(target.parents[1]),run_id=label)
        obj['target_run_root']=fake['run_root'];obj['target_run_id']=label
        if label=='unknown_overlay':obj['unsafe_merge']={}
        elif label=='other_donor':obj['donor2_config']=obj['base_continuation']
        elif label=='base_retarget':obj['base_continuation']=obj['donor2_config']
        elif label=='stage05_producer':obj['stage05']['config_digest']='wrong'
        elif label=='credit_double_count':obj['validated_external_reuse_bytes']['p2_replay']+=4096
        else:
            fp=read(obj['fingerprints']['path']);fp['paths'].pop(next(p for p in fp['paths'] if '/replay/' in p))
            new=target.parents[1]/'admission/continuation_fingerprints.json';atomic_json(new,fp);obj['fingerprints']=identity(new)
        # Copy only the small delta proofs to exact synthetic consumer locations.
        for key in ('source_compatibility','fingerprints'):
            dest=target.parents[1]/'admission'/('continuation_'+key+'.json')
            if not dest.exists():atomic_json(dest,read(obj[key]['path']))
            obj[key]=identity(dest)
        atomic_json(target,obj);fake['continuation']=identity(target)
        try:c.verify(fake)
        except (ValueError,PermissionError,KeyError):rejected.append(label)
        else:raise AssertionError('bad two-donor overlay accepted: '+label)
    # Re-signed actual new index receipt must fail for an incorrect called stage.
    rec=root/'endpoints/P2_SIMPLE/predictions/postprocess'/(nv.KIND+'.json')
    try:rt.witness(cfg,'05',rec)
    except ValueError:rejected.append('actual_new_index_wrong_called_stage')
    else:raise AssertionError('index accepted under wrong called stage')
    for donor in (c.DONOR,c2.DONOR):
        try:atomic_json(donor/'forbidden_fixture_write.json',{})
        except PermissionError:rejected.append('historical_write_'+donor.name)
        else:raise AssertionError('old producer writable')
    return rejected


def suite(root):
    root=Path(root);checks={}
    with ExitStack() as stack:
        workspace=stack.enter_context(rt.lock(ex.COMPUTE));cfg=assemble(root)
        fds=(stack.enter_context(rt.lock(root/'control/launch.lock')),stack.enter_context(rt.lock(root/'control/run.lock')),workspace)
        atomic_json(root/'admission/bounded_resource_fixture.json',rt.forecast(cfg,True))
        started=time.monotonic()
        cmd=[ex.PYTHON,'-I','-S','-B',str(Path(cfg['source_snapshot'])/'tests/rep1_triple_recovery02_data_fixture.py'),'worker',str(root/'configs/run.json'),','.join(map(str,fds))]
        with (root/'logs/bounded_stage06_worker.log').open('w') as log:
            result=subprocess.run(cmd,cwd=cfg['source_snapshot'],env=ex.child_environment(cfg),stdin=subprocess.DEVNULL,
                                  stdout=log,stderr=subprocess.STDOUT,pass_fds=fds,timeout=120)
        seconds=time.monotonic()-started;assert result.returncode==1
        progress=read(root/'control/progress_engineering-01_06.json');receipt=read(progress['witness']['path'])
        assert progress['sequence']==1 and progress['progress']['stage']==nv.KIND and receipt['n_ids']==107955 and seconds<120
        assert receipt['dependencies']['run']==dependencies(cfg,'06') and not receipt.get('reused_from')
        assert 'bounded real stage06 fixture: validated index and adopted4' in (root/'logs/bounded_stage06_worker.log').read_text()
        m=c.manifest(cfg)
        for item in m['generation2']['partial_p2']['tile_receipts']:
            path=root/'endpoints/P2_SIMPLE/predictions/postprocess'/Path(item['path']).name
            r=p2.verify_record(path);assert r['reused_from']==item and r['original_producer']==read(item['path'])
            assert r['dependencies']['base']['run']==dependencies(cfg,'06') and r['adopted'] and not r['recomputed']
        checks.update(actual_full_domain_stage06_worker_seconds=seconds,worker_witness=identity(progress['witness']['path']),
            actual_full_mutation_guards=read(root/'admission/checkpoint_measurements.json'),
            adopted_four_tiles_not_fresh_progress=True,no_remaining_P2_tile_computation_started=True,
            old_native_read_from_original_producer=str(c.native_replay_root(cfg)),
            original_stages01_to05_adoption_validation={s:bool(c.validate_adopted_stage(cfg,s)) for s in ('01','02','03','04','05')})
        checks['two_generation_rejections']=rejections(cfg)
        checks['all_original_partition_consumers']=partitions(cfg)
        checks['read_reference_rejections']=reference_rejections(cfg)
        c2.preserve(read(c2.BASE_BASELINE),c.DONOR,'old_files_and_directories')
        c2.preserve(read(c2.BASELINE),c2.DONOR,'paths')
        checks['both_complete_donor_baselines_preserved']=True
        atomic_json(root/'fixture_complete.json',dict(status='passed',scope='Actual107955 index science validation and all prior read-only consumers; bounded engineering only, no formal preparation/launch',
            source_commit=cfg['source_commit'],source_files=ex.source_files(),config=identity(root/'configs/run.json'),checks=checks))
    print(json.dumps(identity(root/'fixture_complete.json')))


if __name__=='__main__':
    if sys.argv[1]=='suite':suite(sys.argv[2])
    elif sys.argv[1]=='worker':raise SystemExit(worker(Path(sys.argv[2]),tuple(map(int,sys.argv[3].split(',')))))
    else:raise ValueError('unknown bounded Recovery02 fixture command')
