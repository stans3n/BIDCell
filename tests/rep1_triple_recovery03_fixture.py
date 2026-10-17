"""Actual three-producer adoption/digest/guard fixture; no P2 tile computation.

Writes only independent engineering source/proofs and honest adoption wrappers.
Formal configuration loaders reject its schema and its separate fixture path.
"""
from pathlib import Path
import runpy,sys
ROOT=Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT/'scripts/rep1_triple_simple_boundary/bootstrap.py'),run_name='fixture_bootstrap')
import copy,json,subprocess,time
from scripts.rep1_triple_simple_boundary import continuation as c,continuation2 as c2,continuation3 as c3,execution as ex,runtime as rt,p2
from scripts.rep1_triple_simple_boundary.common import atomic_json,dependencies,digest,identity,read


def suite(root,execute=False):
    root=Path(root)
    if not execute:
        root.mkdir(parents=True,exist_ok=False)
        for name in ('configs','admission','control','logs'):(root/name).mkdir()
        source=ex.copy_snapshot(root,ex.git('rev-parse','HEAD'))
        entry=Path(source['snapshot_root'])/'tests/rep1_triple_recovery03_fixture.py'
        with (root/'logs/snapshot_fixture.log').open('w') as log:
            result=subprocess.run([ex.PYTHON,'-I','-S','-B',str(entry),str(root),'--execute'],cwd=source['snapshot_root'],
                env=ex.child_environment(dict(run_root=str(root))),stdout=log,stderr=subprocess.STDOUT,timeout=240)
        if result.returncode:raise RuntimeError('snapshot fixture failed; preserve '+str(root/'logs/snapshot_fixture.log'))
        print(json.dumps(identity(root/'fixture_complete.json')));return
    source=read(root/'source_manifest.json')
    if Path.cwd()!=Path(source['snapshot_root']) or ROOT!=Path(source['snapshot_root']):raise ValueError('fixture must execute its own source copy')
    old=read(c3.DONOR/'configs/run.json')
    cfg=dict(schema='rep1-triple-recovery03-engineering-v1',run_id=root.name,run_root=str(root),
        source_snapshot=source['snapshot_root'],source_commit=source['source_commit'],source_digest=source['source_digest'],
        input_digest=old['input_digest'],domain_digest=old['domain_digest'],domain_manifest=old['domain_manifest'],
        plan_digest=old['plan_digest'],resources=rt.resource_contract())
    started=time.monotonic();cfg['continuation']=c3.build(cfg,c3.PROOF,c3.BASELINE)
    atomic_json(root/'configs/run.json',cfg);c.install_stages(cfg);m=c.verify(cfg)
    part=m['generation3']['partial_p2'];dep=copy.deepcopy(part['tile_base']);dep['run']=dependencies(cfg,'06')
    dep['source']={name:identity(Path(cfg['source_snapshot'])/rel) for name,rel in {'p2.py':'scripts/rep1_triple_simple_boundary/p2.py','p2_rules.py':'scripts/boundary_three/p2_rules.py','postprocess_predictions.py':'bidcell/model/postprocess_predictions.py'}.items()}
    out=root/'endpoints/P2_SIMPLE/predictions/postprocess';out.mkdir(parents=True)
    native=c.native_replay_root(cfg)/'predictions'
    tasks=[(y,x,1024,str(out/f'tile_{y}_{x}.tif'),dict(base=dep,coords=[y,x],shape=[1024,1024]),native) for y,x in part['coords']]
    with rt.lock(ex.COMPUTE):
        c.adopt_tiles(cfg,dep,out,tasks);ids=read(c3.NUMERIC)['full_split_ids']
        def split_digest():return digest(dict(base=dep,tiles=[identity(out/f'tile_{y}_{x}.json') for y,x in part['coords']],ids=ids))
        splitdir=out/'splits';splitdir.mkdir();d=split_digest();c.adopt_split(cfg,dep,splitdir,d,ids)
        committed={str(p):identity(p) for p in out.rglob('*.json')}
        c.adopt_tiles(cfg,dep,out,tasks);c.adopt_split(cfg,dep,splitdir,d,ids)
        assert committed=={str(p):identity(p) for p in out.rglob('*.json')}
        for item in [*part['tile_receipts'],*part['split_receipts']]:
            path=splitdir/Path(item['path']).name if '/splits/' in item['path'] else out/Path(item['path']).name
            rec=p2.verify_record(path);assert rec['original_producer']==read(item['path']) and rec['reused_from']==item
            assert rec['adopted'] and rec['recomputed'] is False
            rt.witness(cfg,'06',path)
        rejected=[]
        def reject(label,fn):
            try:fn()
            except (ValueError,PermissionError,KeyError):rejected.append(label)
            else:raise AssertionError('accepted bad generation3 contract: '+label)
        reject('current_split_digest',lambda:c.adopt_split(cfg,dep,splitdir,'0'*64,ids))
        reject('split_ID_order',lambda:c.adopt_split(cfg,dep,splitdir,d,list(reversed(ids))))
        reject('tile_task_order',lambda:c.adopt_tiles(cfg,dep,out,list(reversed(tasks))))
        reject('current_stage',lambda:c.adopt_tiles(cfg,dict(dep,run=dependencies(cfg,'05')),out,tasks))
        reject('called_stage_witness',lambda:rt.witness(cfg,'05',splitdir/'000000.json'))
        current=out/'tile_0_0.json';saved=read(current);bad=copy.deepcopy(saved);bad['outputs']['postislands']=bad['outputs']['postconnect']
        bad['integrity']=digest({k:v for k,v in bad.items() if k!='integrity'});atomic_json(current,bad)
        atomic_json(root/'admission/rejected_resigned_output.json',bad)
        reject('resigned_current_wrapper_output',lambda:c.adopt_tiles(cfg,dep,out,tasks));atomic_json(current,saved)
        raw=read(cfg['continuation']['path'])
        for label in ('unknown_overlay','target','source','base_retarget','donor_retarget','numeric_retarget','credit_duplicate','tile_order','old_split_digest','fingerprint_omission'):
            badcfg=dict(cfg);obj=copy.deepcopy(raw);dest=root/'tamper'/label
            badcfg.update(run_root=str(dest),run_id=label);obj.update(target_run_root=str(dest),target_run_id=label)
            for key in ('source_compatibility','fingerprints'):
                p=dest/'admission'/('continuation_'+key+'.json');atomic_json(p,read(obj[key]['path']));obj[key]=identity(p)
            if label=='unknown_overlay':obj['unsafe_deep_merge']={}
            elif label=='target':obj['target_run_id']='wrong'
            elif label=='source':obj['new_source_digest']='wrong'
            elif label=='base_retarget':obj['base_continuation']=obj['donor3_config']
            elif label=='donor_retarget':obj['donor3_config']=obj['base_continuation']
            elif label=='numeric_retarget':obj['numeric_validation']=obj['donor3_validation']
            elif label=='credit_duplicate':obj['validated_external_reuse_bytes']['p2_replay']+=4096
            elif label=='tile_order':obj['partial_p2']['tile_receipts'].reverse()
            elif label=='old_split_digest':obj['partial_p2']['old_opaque_split_digest']='0'*64
            else:
                path=Path(obj['fingerprints']['path']);fp=read(path);fp['paths'].pop(next(k for k in fp['paths'] if '/tile_' in k));atomic_json(path,fp);obj['fingerprints']=identity(path)
            p=dest/'configs/continuation.json';atomic_json(p,obj);badcfg['continuation']=identity(p)
            reject(label,lambda cfg=badcfg:c.verify(cfg))
        for donor in (c.DONOR,c2.DONOR,c3.DONOR):
            reject('historical_write_'+donor.name,lambda donor=donor:atomic_json(donor/'forbidden_fixture_write.json',{}))
        fingerprint_start=time.monotonic();expected=c.fingerprint_paths(m)
        assert ex.fingerprints(expected)==expected
        fingerprint_seconds=time.monotonic()-fingerprint_start
        for baseline,donor,key in ((c2.BASE_BASELINE,c.DONOR,'old_files_and_directories'),(c2.BASELINE,c2.DONOR,'paths'),(c3.BASELINE,c3.DONOR,'paths')):
            c2.preserve(read(baseline),donor,key)
    result=dict(status='passed',engineering_only=True,source_commit=cfg['source_commit'],source_files=ex.source_files(),config=identity(root/'configs/run.json'),
        adopted_tiles=48,adopted_split_IDs=128,full_split_IDs=2920,current_split_digest=d,all_current_receipts_actual_witness_pass=True,
        duplicate_adoption_preserved_exact_bytes=True,original_three_runs_unchanged=True,no_P2_tile_or_global_merge_computed=True,
        old_receipts_not_counted_as_fresh_progress=True,negative_cases=rejected,mutation_paths=len(expected),mutation_check_seconds=fingerprint_seconds,
        validated_external_logical_reuse_bytes=m['validated_external_reuse_bytes'],elapsed_seconds=time.monotonic()-started)
    atomic_json(root/'fixture_complete.json',result);print(json.dumps(identity(root/'fixture_complete.json')))


if __name__=='__main__':suite(sys.argv[1],len(sys.argv)>2 and sys.argv[2]=='--execute')
