"""Real donor read-only continuation exercise, bounded before production.

Copies only the reviewed source and small proof/configuration objects. Executes
the actual runtime worker through adoption of48 native tiles/first split and
one new split, then intentionally stops. Also exercises actual initial partition
consumers and coverage readers. It never calls formal prepare/launch/admission.
"""
from pathlib import Path
import runpy,sys
ROOT=Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT/'scripts/rep1_triple_simple_boundary/bootstrap.py'),run_name='fixture_bootstrap')
import copy,importlib,json,os,shutil,subprocess,time
from contextlib import ExitStack
import numpy as np
import pandas as pd
from scripts.rep1_triple_simple_boundary import continuation as c,execution as ex,runtime as rt,science,p2,inputs,contracts
from scripts.rep1_triple_simple_boundary.common import atomic_json,dependencies,digest,identity,read,sha256,within

DOCS=Path('/home/zwan0726/work/docs')
PROOF=DOCS/'BIDCell_Rep1_Triple_Simple_Recovery01_RootDonorValidation_20261017.json'
SPLIT=DOCS/'BIDCell_Rep1_Triple_Simple_Recovery01_RootSplit128Numeric_20261017.json'
BASELINE=DOCS/'BIDCell_Rep1_Triple_Simple_Recovery01_DonorBaseline_20261017.json'


def worker(config,fds):
    cfg=read(config);root=Path(cfg['run_root'])
    if cfg['schema']!='rep1-triple-recovery-data-fixture-v1' or 'recovery-fixtures' not in str(root):raise ValueError('not isolated continuation fixture')
    expected=identity(config)
    def load(path):
        if str(Path(path))!=expected['path'] or identity(path)!=expected:raise ValueError('fixture config changed')
        return cfg
    class Validator:
        def __init__(self,cfg,fds):
            self.cfg,self.fds=cfg,fds;self.admission=identity(root/'admission/not_formal.json')
            ex.verify_snapshot(cfg);self()
        def __call__(self):
            rt.check_fds(self.cfg,self.fds);c.verify(self.cfg);ex.module_origins(self.cfg['source_snapshot']);load(config)
    def run_stage(cfg,stage,progress=None,lock_fds=(),validate_execution=None,compute_authorization=None):
        assert stage=='05';validate_execution();manifest=inputs.load_inputs(cfg);parent=manifest['endpoints']['MORPH_SIMPLE_FUSION']
        p2.process(parent['pre'],parent['nuclei'],parent['nuclei'],root/'replay/SIMPLE_NATIVE/predictions','B',
            workers=cfg['resources']['p2_workers'],allowed_root=root,dependency_context=dependencies(cfg,'05'),
            continuation_cfg=cfg,callback=progress,lock_fds=lock_fds,interrupt_after=('splits',256))
        raise AssertionError('bounded fixture must interrupt after first newly computed split')
    rt.load_config=load;rt.execution_validator=Validator;science.run_stage=run_stage
    return rt.worker(config,'05','engineering-01',fds)


def assemble(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    for name in ('configs','control','logs','admission','tmp','cache'):(root/name).mkdir()
    source=ex.copy_snapshot(root,ex.git('rev-parse','HEAD'))
    partial=dict(schema='rep1-triple-recovery-data-fixture-v1',run_id=root.name,run_root=str(root),source_snapshot=source['snapshot_root'],
        source_commit=source['source_commit'],source_digest=source['source_digest'],resources=rt.resource_contract())
    start=time.monotonic();cont=c.build(partial,PROOF,SPLIT,BASELINE);elapsed=time.monotonic()-start
    donor=read(c.DONOR/'configs/run.json');shutil.copyfile(donor['input_manifest'],root/'configs/inputs_manifest.json')
    shutil.copyfile(donor['comparison_manifest'],root/'configs/comparisons.json')
    cfg=dict(partial,continuation=cont,input_manifest=str(root/'configs/inputs_manifest.json'),input_digest=donor['input_digest'],
        domain_manifest=donor['domain_manifest'],domain_digest=donor['domain_digest'],plan_manifest=donor['plan_manifest'],plan_digest=donor['plan_digest'],
        comparison_manifest=str(root/'configs/comparisons.json'),environment_digest='engineering fixture; no admission',
        stage_order=list(contracts.STAGE_ORDER),python=ex.PYTHON,rscript=ex.RSCRIPT)
    atomic_json(root/'configs/run.json',cfg);c.install_stages(cfg);atomic_json(root/'state.json',rt.new_state(cfg))
    atomic_json(root/'admission/not_formal.json',dict(approved_for_formal_launch=False,engineering_only=True,config_digest=digest(cfg)))
    atomic_json(root/'admission/build_measurement.json',dict(seconds=elapsed,continuation=cont,source_files=ex.source_files()))
    return cfg


def partitions(cfg):
    from scripts.rep1_triple_simple_boundary import populations,bt,boundary_analysis as ba,coverage,evaluation_pairs as ep
    compare=importlib.import_module('scripts.rep1_triple_simple_boundary.comparisons')
    root=Path(cfg['run_root']);manifest=inputs.load_inputs(cfg,verify=False);methods=contracts.DAG['initial_methods9']
    specs={name:manifest['endpoints'][name] if name in contracts.EXISTING else inputs.new_spec(cfg,name) for name in methods}
    initial=c.initial_root(cfg);out=root/'analysis/all12';started=time.monotonic()
    tables=populations.prepare(cfg,manifest,specs,out/'populations',reuse_from=ep.reuse_sources(cfg,methods,'populations',initial))
    compare.run(cfg,manifest,specs,tables,out/'populations',out/'comparisons',reuse_from=ep.reuse_sources(cfg,methods,'comparisons',initial))
    bt.paired(cfg,manifest,specs,out/'bt',reuse_from=ep.reuse_sources(cfg,methods,'bt',initial))
    ba.run(cfg,manifest,specs,out/'boundary',reuse_from=initial/'boundary',include_structure=False)
    receipts=[p for p in out.rglob('complete.json') if read(p).get('adoption_kind')]
    assert len(receipts)==16+43+16+9 # transfer uses initial_reuse, not complete.
    assert len(list(out.rglob('initial_reuse.json')))==15
    assert not any(p.is_symlink() for p in out.rglob('*'))
    # Parse actual referenced arrays and complete table coverage at the consumer.
    groups=dict(np.load(c.frozen_root(cfg)/'cell_groups.npz'));cells=pd.read_csv(c.frozen_root(cfg)/'cells.csv').set_index('cell_id')
    a,b=contracts.comparisons(methods)[0];key=contracts.comparison_id(a,b);hashes={}
    for metric in contracts.METRICS:
        ids=populations.selected(out/'populations',key,metric)
        from scripts.rep1_triple_simple_boundary.common import id_hash
        for profile in ('sc','fixed'):hashes[(metric,profile)]=id_hash(ids)
    coverage.comparison_rows(pd.read_csv(c.read_data(out/'comparisons'/key/'stratified_intervals.csv.gz')),groups,{a:1,b:-1},'pairwise',hashes)
    q=pd.read_csv(c.read_data(out/'bt'/key/'population.csv')).cell_id.to_numpy(np.int64)
    coverage.paired_bt_coverage(out/'bt'/key,key,(a,b),True,c.frozen_root(cfg),q)
    coverage.boundary_tables(cfg,manifest,specs[a],specs[b],out/'boundary/comparisons'/key,c.frozen_root(cfg),cells,groups,False)
    return dict(seconds=time.monotonic()-started,methods=len(methods),population_partitions=16,comparison_jobs=43,BT_sets=16,
        regional_methods=9,transfer_partitions=15,representative_semantic_coverage=key,physical_donor_copies=0,
        scope='All completed donor partitions actually consumed; representative mapped score/BT/full313 transfer tables semantically validated; no all12 production claim')


def reference_rejections(cfg):
    root=Path(cfg['run_root']);target=next((root/'analysis/all12/comparisons').glob('*/.data-reference.json')).parent
    ref=target/'.data-reference.json';receipt=target/'complete.json';ref_bytes=ref.read_bytes();receipt_bytes=receipt.read_bytes()
    descriptor=read(ref);m=c.manifest(cfg);entry=m['partitions'][descriptor['source']]
    leaf=target/next(iter(entry['files']));rejected=[]
    def reject(label,operation):
        try:operation()
        except (ValueError,PermissionError):rejected.append(label)
        else:raise AssertionError('invalid reference accepted: '+label)
    for mode in ('unrelated_producer','different_approved_partition','duplicate_config_path','binding_changed','consumer_run_changed'):
        descriptor=read(ref);record=read(receipt)
        if mode=='unrelated_producer':descriptor['source']=str(root/'unapproved_producer')
        elif mode=='different_approved_partition':
            descriptor['source']=next(p for p in m['partitions'] if p!=descriptor['source'])
        elif mode=='duplicate_config_path':
            duplicate=root/'tamper/identical_config_copy.json';atomic_json(duplicate,cfg);descriptor['execution_config']=identity(duplicate)
        elif mode=='binding_changed':record['reuse_binding']['fixture_wrong_binding']=True
        else:record['dependencies']['run_id']='unrelated_consumer'
        if mode=='unrelated_producer':(root/'unapproved_producer').mkdir()
        atomic_json(ref,descriptor);record['data_reference']=identity(ref);record['outputs']['data_reference']=identity(ref)
        atomic_json(receipt,record);atomic_json(root/'tamper'/(mode+'_descriptor.json'),descriptor);atomic_json(root/'tamper'/(mode+'_receipt.json'),record)
        try:reject(mode,lambda:c.read_data(leaf))
        finally:
            ref.write_bytes(ref_bytes);receipt.write_bytes(receipt_bytes);c._DATA_CACHE.clear()
    # A local addition cannot shadow or supplement the registered external map.
    extra=target/'unregistered-extra.txt';extra.write_text('fixture addition')
    try:reject('unregistered_local_addition',lambda:c.read_data(leaf))
    finally:extra.rename(root/'tamper/unregistered-extra.txt');c._DATA_CACHE.clear()
    reject('write_to_referenced_leaf',lambda:atomic_json(leaf,{}))
    reject('write_to_original_producer',lambda:atomic_json(c.DONOR/'forbidden_fixture_write.json',{}))
    assert c.read_data(leaf)==Path(entry['files'][str(leaf.relative_to(target))]['path'])
    return rejected


def suite(root):
    root=Path(root);checks={}
    with ExitStack() as stack:
        # The actual workspace lock prevents overlap with another heavy task.
        workspace=stack.enter_context(rt.lock(ex.COMPUTE));cfg=assemble(root)
        fds=(stack.enter_context(rt.lock(root/'control/launch.lock')),stack.enter_context(rt.lock(root/'control/run.lock')),workspace)
        forecast=rt.forecast(cfg,True);atomic_json(root/'admission/resource_fixture.json',forecast)
        assert forecast['admitted'],forecast
        started=time.monotonic()
        cmd=[ex.PYTHON,'-I','-S','-B',str(Path(cfg['source_snapshot'])/'tests/rep1_triple_continuation_fixture.py'),'worker',str(root/'configs/run.json'),','.join(map(str,fds))]
        with (root/'logs/bounded_native_worker.log').open('w') as log:
            result=subprocess.run(cmd,cwd=cfg['source_snapshot'],env=ex.child_environment(cfg),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,pass_fds=fds,timeout=120)
        seconds=time.monotonic()-started;assert result.returncode==1
        progress=read(root/'control/progress_engineering-01_05.json');receipt=read(progress['witness']['path'])
        assert Path(progress['witness']['path']).name=='000128.json' and not receipt.get('reused_from')
        assert progress['sequence']==1 and progress['progress']['cursor']==256 and seconds<=120
        checks['first_genuine_new_split_seconds']=seconds
        checks['adopted48tiles_and_first128IDs_not_counted_as_progress']=True
        assert 'admission interruption after splits' in (root/'logs/bounded_native_worker.log').read_text()
        # Recompute receipt integrity around wrong envelopes: witness must still
        # reject these actual split records under the complete continuation cfg.
        rejected=[]
        for mode in ('missing','wrong_run','wrong_config'):
            bad=copy.deepcopy(receipt)
            if mode=='missing':bad['dependencies'].pop('run')
            else:bad['dependencies']['run']['run_id' if mode=='wrong_run' else 'config_digest']='unrelated'
            bad['integrity']=digest({k:v for k,v in bad.items() if k!='integrity'});path=root/'tamper'/f'{mode}.json';atomic_json(path,bad)
            try:rt.witness(cfg,'05',path)
            except ValueError:rejected.append(mode)
            else:raise AssertionError('tampered current identity accepted')
        checks['recomputed_integrity_binding_rejections']=rejected
        original=receipt['outputs']['split'];corrupt=root/'tamper/corrupt_split.npz';data=bytearray(Path(original['path']).read_bytes());data[-1]^=1;corrupt.write_bytes(data)
        bad=copy.deepcopy(receipt);bad['outputs']['split']['path']=str(corrupt);bad['integrity']=digest({k:v for k,v in bad.items() if k!='integrity'})
        corrupt_receipt=root/'tamper/corrupt_split.json';atomic_json(corrupt_receipt,bad)
        try:rt.witness(cfg,'05',corrupt_receipt)
        except ValueError:checks['actual_split_payload_corruption_rejected']=True
        else:raise AssertionError('split payload corruption accepted')
        before=time.monotonic();c.verify(cfg);checks['all_scientific_fingerprints_seconds']=time.monotonic()-before
        checks['partition_consumers']=partitions(cfg)
        checks['readonly_reference_rejections']=reference_rejections(cfg)
        baseline=read(BASELINE)
        assert {str(p.relative_to(c.DONOR)) for p in c.DONOR.rglob('*')}==set(baseline['old_files_and_directories'])
        assert all(c.fingerprint(c.DONOR/path)==fp for path,fp in baseline['old_files_and_directories'].items())
        checks['entire_donor_tree_84015_fingerprints_preserved']=True
        atomic_json(root/'fixture_complete.json',dict(status='passed',approved_for_formal_launch=False,config=identity(root/'configs/run.json'),
            source_manifest=identity(root/'source_manifest.json'),checks=checks,source_files=ex.source_files(),continuation=cfg['continuation']))
    print(json.dumps(identity(root/'fixture_complete.json')))


if __name__=='__main__':
    if sys.argv[1]=='suite':suite(sys.argv[2])
    elif sys.argv[1]=='worker':raise SystemExit(worker(Path(sys.argv[2]),tuple(map(int,sys.argv[3].split(',')))))
    else:raise ValueError('unknown isolated continuation fixture command')
