"""R1: real ROI lineage/rebinding controls plus isolated child-receipt faults.

No mocks, formal training or full-slide computation. Matrix/postprocess fixtures
are genuinely generated from the admitted ROI. Scoring bytes are copied from
that same exact ROI matrix and independently checked for equality. Prediction
and split fixtures exercise receipt schemas on real historical chunk bytes;
they do not claim new-arm inference or formal training completion.
"""
import argparse
from contextlib import contextmanager
import shutil
import numpy as np
from .common import *
from .stages import (expected_native_dependencies,validate_native_lineage,
    validate_prediction_chunk,validate_postprocess_split,validate_stage,
    _bind_completed_stage,run_stage,postprocess,matrix)


@contextmanager
def preserve(*paths):
    saved=[(Path(p),Path(p).read_bytes(),Path(p).stat()) for p in paths]
    try:yield
    finally:
        for p,data,s in saved:p.write_bytes(data);os.utime(p,ns=(s.st_atime_ns,s.st_mtime_ns))


def changed_json(path,change):
    value=read(path);change(value);atomic_json(path,value)


def check(root=DEFAULT_ROOT,label='verification_1'):
    root=Path(root);dest=root/'tests/revision_r1'/label
    if dest.exists():raise ValueError('preserve previous R1 fixture; use a distinct label')
    dest.mkdir(parents=True);records=[]
    def rejected(name,call):
        try:call()
        except (ValueError,FileNotFoundError) as error:
            records.append(dict(test=name,status='rejected_as_required',exception=type(error).__name__,reason=str(error)))
        else:raise AssertionError('stale lineage accepted: '+name)
    independent=root/'tests/independent_review/cache_fixture/BASE_MATCH/configs/run.json'
    rejected('independent_B_unchanged_stale_fixture',lambda:validate_stage(independent,'evaluate'))
    out=dest/'BASE_MATCH';(out/'configs').mkdir(parents=True);(dest/'prepared').mkdir()
    for name in ('authoritative_protocol.json','experiment_contract.json','input_audit.json'):
        shutil.copy2(root/'prepared'/name,dest/'prepared'/name)
    cfg=read(config_path('BASE_MATCH',root))
    cfg.update(runroot=str(dest),output_root=str(out),config_path=str(out/'configs/run.json'),machine=str(out/'configs/machine.json'),
        initial_snapshot=str(dest/'prepared/BASE_MATCH_step_zero.pth'),contract=identity(dest/'prepared/experiment_contract.json'),
        authoritative_protocol=identity(dest/'prepared/authoritative_protocol.json'))
    atomic_json(cfg['config_path'],cfg)
    original=root/'tests/roi/BASE_MATCH';machine=read(original/'machine.json');shared=dest/'shared';shared.mkdir()
    for key,name in (('nuclei_mask','nuclei.tif'),('transcripts','transcripts.csv'),('genes','genes.txt'),('baseline_config','baseline.yaml')):
        shutil.copy2(machine[key],shared/name);machine[key]=str(shared/name)
    pred=out/'predictions/BASE_MATCH';pred.mkdir(parents=True)
    shutil.copy2(original/'predictions/BASE_MATCH/pre.tif',pred/'pre.tif')
    machine.update(pre_mask=str(pred/'pre.tif'),final_mask=str(pred/'final.tif'),baseline_final_mask=str(pred/'pre.tif'),
        final_matrix=str(out/'matrix/BASE_MATCH/expr_mat.csv'),checkpoint=str(out/'checkpoints/final.pth'))
    atomic_json(cfg['machine'],machine)
    # Native generation, rather than fabricated successful matrix/postprocess.
    postprocess(cfg);matrix(cfg)
    import tifffile
    assert np.array_equal(tifffile.imread(pred/'final.tif'),tifffile.imread(original/'predictions/BASE_MATCH/final.tif'))
    mp=out/'matrix/BASE_MATCH';mrec=read(mp/'complete.json')
    assert np.array_equal(np.load(mp/'counts.npy'),np.load(original/'matrix/BASE_MATCH/counts.npy'))
    validate_native_lineage(cfg,'postprocess',check_upstream=False)
    validate_native_lineage(cfg,'matrix',check_upstream=False)
    records.append(dict(test='real_native_ROI_postprocess_matrix_current_lineage',status='passed',counts_exact_to_original_ROI=True))
    ep=out/'evaluation/BASE_MATCH';shutil.copytree(original/'evaluation/BASE_MATCH',ep)
    deps=expected_native_dependencies(cfg,'evaluate')
    prof_outputs={str(p.relative_to(ep)):identity(p) for folder in ('engineering','paper_aligned_v1') for p in (ep/folder).rglob('*') if p.is_file()}
    atomic_json(ep/'profiles_complete.json',dict(status='succeeded',dependencies=deps,outputs=prof_outputs,summary={'scope':'receipt fixture for unchanged genuine ROI products'}))
    atomic_json(ep/'complete.json',dict(status='succeeded',dependencies=deps,paper_method_complete=True,outputs=file_tree(ep)))
    valid=validate_stage(cfg,'evaluate');bound=out/'stages/evaluate.json'
    _bind_completed_stage(cfg,'evaluate')
    assert run_stage(cfg,'evaluate')==valid
    records.append(dict(test='valid_missing_bound_commit_and_existing_bound_gateway_reuse',status='passed',qc_ids=valid['semantic']['qc_ids'],
        scope='actual gateway commit helper on real ROI evaluation bytes and newly generated identical matrix; no full formal-stage execution claimed'))
    bound_bytes=bound.read_bytes();bound_stat=bound.stat()
    cache=Path(mrec['coordinate_cache']);coordinate=Path(mrec['receipts'][0]['coordinate_file'])
    for mode in ('bound','unbound'):
        if mode=='unbound':bound.unlink()
        action=(lambda:run_stage(cfg,'evaluate')) if mode=='bound' else (lambda:_bind_completed_stage(cfg,'evaluate'))
        mutations=[('machine_scale',Path(cfg['machine']),lambda:changed_json(cfg['machine'],lambda d:d.update(scale_x=.5))),
                   ('machine_pixel_um',Path(cfg['machine']),lambda:changed_json(cfg['machine'],lambda d:d.update(pixel_um=2.))),
                   ('current_matrix_receipt',mp/'complete.json',lambda:changed_json(mp/'complete.json',lambda d:d.update(fixture_new_revision=True))),
                   ('actual_matrix_counts',mp/'counts.npy',lambda:(mp/'counts.npy').write_bytes((mp/'counts.npy').read_bytes()+b'x')),
                   ('actual_matrix_CSV',mp/'expr_mat.csv',lambda:(mp/'expr_mat.csv').write_bytes((mp/'expr_mat.csv').read_bytes()+b'\n')),
                   ('actual_coordinate_chunk',coordinate,lambda:coordinate.write_bytes(coordinate.read_bytes()+b'x'))]
        for name,path,mutate in mutations:
            with preserve(path):
                mutate();rejected(mode+'/'+name,action)
                assert bound.exists()==(mode=='bound')
                if mode=='bound':assert bound.read_bytes()==bound_bytes
        with preserve(ep/'profiles_complete.json',ep/'complete.json'):
            changed_json(ep/'profiles_complete.json',lambda d:d['dependencies']['machine'].update(sha256='0'*64))
            changed_json(ep/'complete.json',lambda d:d['outputs'].update({'profiles_complete.json':identity(ep/'profiles_complete.json')}))
            rejected(mode+'/profiles_dependency_rehashed_output',action)
            assert bound.exists()==(mode=='bound')
    # Restored valid unbound cache must still bind, then remain read-only on reuse.
    restored=_bind_completed_stage(cfg,'evaluate');assert restored==valid
    assert run_stage(cfg,'evaluate')==valid
    records.append(dict(test='valid_recovery_after_all_dependency_faults_removed',status='passed'))
    # Matrix native dependencies and source/coordinate metadata.
    matcall=lambda:validate_native_lineage(cfg,'matrix',check_upstream=False)
    for key in ('transcripts','genes','baseline_config'):
        path=Path(machine[key])
        with preserve(path):path.write_bytes(path.read_bytes()+b'\n');rejected('matrix/'+key,matcall)
    for key in ('scale_x','scale_y'):
        with preserve(cfg['machine']):
            changed_json(cfg['machine'],lambda d:d.update({key:.5}));rejected('matrix/'+key,matcall)
    for key in ('source_sha256','coordinate','filter','kind','shape'):
        bad=copy.deepcopy(mrec);bad['contract'][key]='intentional_stale_contract'
        rejected('matrix/contract_'+key,lambda bad=bad:validate_native_lineage(cfg,'matrix',bad,check_upstream=False))
    with preserve(cache/'complete.json'):
        changed_json(cache/'complete.json',lambda d:d['dependencies'].update(scale_x=.5));rejected('matrix/coordinate_cache_scale',matcall)
    owner_receipt=mp/Path(mrec['receipts'][0]['owner_file']).with_suffix('.json')
    with preserve(owner_receipt):
        changed_json(owner_receipt,lambda d:d['dependencies'].update(coordinates='0'*64));rejected('matrix/owner_coordinate_dependency',matcall)
    with preserve(mp/'validation.json'):
        changed_json(mp/'validation.json',lambda d:d['source'].update(sha256='0'*64))
        bad=copy.deepcopy(mrec);bad['validation']=identity(mp/'validation.json')
        rejected('matrix/validation_source_rehashed',lambda:validate_native_lineage(cfg,'matrix',bad,check_upstream=False))
    # Postprocessing uses a real new native ROI manifest. Rehash parent inventories
    # when corrupting children so failures exercise dependency semantics, not just hashes.
    pp=pred/'postprocess';pprec=read(pp/'complete.json');manifest=read(pp/'manifest.json')
    ppcall=lambda:validate_native_lineage(cfg,'postprocess',check_upstream=False)
    for key in ('nuclei','source'):
        bad=copy.deepcopy(pprec);bad['dependencies'][key]={'intentional_stale_dependency':True}
        rejected('postprocess/'+key,lambda bad=bad:validate_native_lineage(cfg,'postprocess',bad,check_upstream=False))
    tile=Path(manifest['tiles'][0]['path'])
    with preserve(tile,pp/'manifest.json',pp/'complete.json'):
        changed_json(tile,lambda d:d['dependencies']['base']['nuclei'].update(sha256='0'*64))
        changed_json(pp/'manifest.json',lambda d:d['tiles'].__setitem__(0,identity(tile)))
        changed_json(pp/'complete.json',lambda d:d['outputs'].update(manifest=identity(pp/'manifest.json')))
        rejected('postprocess/tile_base_rehashed_parents',ppcall)
    # A real archived split byte payload exercises the split receipt helper.
    historical=read(INPUT/'V2_MORPH/predictions/V2_MORPH/postprocess/manifest.json')['split_chunks'][0]
    sp=dest/'split_fixture/000000.json';sp.parent.mkdir();shutil.copy2(historical['output']['path'],sp.with_suffix('.npz'))
    splitbase='test dependency token for genuine historical split bytes';ids=historical['ids']
    atomic_json(sp,dict(status='succeeded',dependencies=dict(base=splitbase,ids=ids),outputs={'split':identity(sp.with_suffix('.npz'))}))
    validate_postprocess_split(sp,splitbase,ids)
    for key,value in [('base','stale'),('ids',list(reversed(ids)))]:
        with preserve(sp):
            changed_json(sp,lambda d:d['dependencies'].update({key:value}))
            rejected('postprocess/split_'+key,lambda:validate_postprocess_split(sp,splitbase,ids))
    # Identity-only checkpoint fixture; never loaded or described as trained.
    ckpt=out/'checkpoints/final.pth';ckpt.parent.mkdir();ckpt.write_bytes(b'R1 identity-only checkpoint fixture; never a model')
    pdeps=expected_native_dependencies(cfg,'predict')
    for key in ('checkpoint','prepared','producer','run_id','model_kind'):
        bad=dict(status='succeeded',dependencies=copy.deepcopy(pdeps),outputs={});bad['dependencies'][key]='stale'
        rejected('predict/top_'+key,lambda bad=bad:validate_native_lineage(cfg,'predict',bad,check_upstream=False))
    cp=dest/'prediction_chunk/000000.json';cp.parent.mkdir()
    hcp=INPUT/'V2_MORPH/predictions/V2_MORPH/shift0_patches/000000.npy';shutil.copy2(hcp,cp.with_suffix('.npy'))
    patches=[x['identity'] for x in read(INPUT/'configs/predict_shift0.json')[:128]]
    atomic_json(cp,dict(status='succeeded',dependencies=dict(base=pdeps,start=0,stop=128,patches=patches),
        outputs={'masks':identity(cp.with_suffix('.npy'))}))
    validate_prediction_chunk(cp,pdeps,0,128,patches)
    for key in pdeps:
        with preserve(cp):
            changed_json(cp,lambda d:d['dependencies']['base'].update({key:'stale'}))
            rejected('predict/chunk_'+key,lambda:validate_prediction_chunk(cp,pdeps,0,128,patches))
    with preserve(cp):
        changed_json(cp,lambda d:d['dependencies'].update(patches=list(reversed(patches))))
        rejected('predict/chunk_patch_order',lambda:validate_prediction_chunk(cp,pdeps,0,128,patches))
    records.append(dict(test='prediction_and_split_child_valid_schema_controls',status='passed',
        scope='real historical chunk payloads with isolated schema fixture receipts; not new inference'))
    assert read(root/'prepared/step_zero_receipt.json')['source']==producer('train')
    result=dict(status='passed',fixture=str(dest),records=records,rejections=sum(x['status']=='rejected_as_required' for x in records),
        no_mocks=True,no_formal_compute=True,scientific_train_producer_unchanged=True,producer=identity(REPO/'scripts/paired_v2/stages.py'),
        test_source=identity(__file__),independent_original_failure=identity(root/'tests/independent_review/cache_dependency_negative.json'))
    atomic_json(dest/'result.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=DEFAULT_ROOT);p.add_argument('--label',default='verification_1');a=p.parse_args()
    print(json.dumps(check(a.root,a.label),indent=2))
