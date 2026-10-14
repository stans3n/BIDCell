"""Own-mask integer counting and frozen authoritative complete evaluation."""
from pathlib import Path
import time
import numpy as np
import pandas as pd
import tifffile
from .common import *

METRICS=('positive_F1','negative_F1','positive_Precision','positive_Recall','negative_Precision','negative_Recall')

def check_legacy_identity(obj):
    """Old receipts include mtime; content and length are their stable identity."""
    if sha256(obj['path'])!=obj['sha256'] or ('bytes' in obj and Path(obj['path']).stat().st_size!=obj['bytes']):
        raise ValueError('historical artifact drift: '+obj['path'])

def verify_legacy_receipt(path):
    rec=read(path)
    if rec.get('status') not in ('succeeded','passed','completed'):
        raise ValueError('historical/native receipt incomplete: '+str(path))
    if not rec.get('outputs'):raise ValueError('receipt has no artifacts: '+str(path))
    for obj in rec['outputs'].values():check_legacy_identity(obj)
    return rec

def evaluation_contract(cfg):
    queue=read(cfg['inputs']['evaluation_queue']);authority=read(cfg['inputs']['authoritative_protocol'])
    prep=verify_legacy_receipt(Path(queue['prepared'])/'complete.json')
    protocol=prep['dependencies']['protocol']['path'];p=read(protocol)
    if {k:p[k] for k in authority['core']}!=authority['core']:
        raise ValueError('authoritative scientific evaluation core changed')
    check_legacy_identity(authority['source_protocol']);check_legacy_identity(authority['reference_model'])
    if sha256(protocol)!=authority['source_protocol']['sha256'] or prep['model']['sha256']!=authority['reference_model']['sha256']:
        raise ValueError('scClassify protocol/model changed')
    check_legacy_identity(authority['fixed_population']);check_legacy_identity(queue['population'])
    if queue['population']['sha256']!=authority['fixed_population']['sha256']:raise ValueError('fixed panel identity drift')
    pop=pd.read_csv(queue['population']['path'])
    if len(pop)!=103000 or pop.cell_id.duplicated().any():raise ValueError('fixed panel invalid')
    for key in ('reference','official_processing','official_expression','rscript'):
        if sha256(p[key])!=prep['dependencies'][key]['sha256']:raise ValueError('evaluation input/producer drift '+key)
    return queue,protocol,dict(authority=identity(cfg['inputs']['authoritative_protocol']),
        runtime_protocol=identity(protocol),reference_model=identity(authority['reference_model']['path']),
        prepared=identity(Path(queue['prepared'])/'complete.json'),fixed_population=identity(queue['population']['path']),
        source_digest=cfg.get('source_digest'))

def _spec(cfg,name,spec=None):
    e=dict(spec or endpoint_spec(cfg,name))
    for key in ('root','machine','matrix','evaluation'):
        p=Path(e[key]).resolve();root=Path(cfg['output_root']).resolve()
        if p!=root and root not in p.parents:raise ValueError('new endpoint output outside run: '+key)
    e.setdefault('cost_root',e['root'])
    return e

def validate_endpoint(cfg,endpoint_id,spec=None):
    cfg=load_run(cfg)
    if endpoint_id in cfg['endpoints']:
        e=endpoint_spec(cfg,endpoint_id)
    elif endpoint_id=='A2_NATIVE' and spec is None:
        rec=read(output_path(cfg,'A2_CORE','endpoints','A2_NATIVE','validated_reuse.json'))
        if rec['config_digest']!=digest(cfg):raise ValueError('native reuse config drift')
        for obj in rec['bindings'].values():check_identity(obj)
        return validate_endpoint(cfg,rec['source_endpoint'])
    else:e=_spec(cfg,endpoint_id,spec)
    machine=read(e['machine']);matrix=verify_legacy_receipt(Path(e['matrix'])/'complete.json')
    evaluation=verify_legacy_receipt(Path(e['evaluation'])/'complete.json')
    if not evaluation.get('paper_method_complete'):raise ValueError('full evaluation missing')
    if matrix.get('validation_kind')!='independent_full_native_resize_lookup_and_conservation':raise ValueError('own matrix lacks independent native oracle')
    if matrix['dependencies']['mask']['sha256']!=sha256(machine['final_mask']):raise ValueError('matrix mask linkage wrong')
    if evaluation['dependencies']['matrix']['sha256']!=sha256(Path(e['matrix'])/'complete.json'):raise ValueError('evaluation matrix linkage wrong')
    if sha256(machine['final_matrix'])!=sha256(Path(e['matrix'])/'expr_mat.csv'):raise ValueError('own native matrix CSV mismatch')
    from scripts.paired_v2.evaluation import validate_evaluation
    result=validate_evaluation(e)
    genes=pd.read_csv(Path(e['matrix'])/'gene_mapping.csv').gene.tolist()
    if genes!=Path(machine['genes']).read_text().splitlines() or len(genes)!=313:raise ValueError('spatial gene order mismatch')
    counts=np.load(Path(e['matrix'])/'counts.npy',mmap_mode='r')
    if not np.issubdtype(counts.dtype,np.integer) or (counts<0).any() or int(counts.sum())!=matrix['totals']['assigned']:raise ValueError('integer conservation failure')
    _,_,contract=evaluation_contract(cfg)
    summary=read(Path(e['evaluation'])/'summary.json')
    if summary['model_sha256']!=contract['reference_model']['sha256']:raise ValueError('endpoint classifier mismatch')
    return dict(**result,endpoint_id=endpoint_id,matrix=identity(Path(e['matrix'])/'complete.json'),
        evaluation=identity(Path(e['evaluation'])/'complete.json'),final=identity(machine['final_mask']),
        eligible=matrix['totals']['eligible'],assigned=matrix['totals']['assigned'])

def build_and_evaluate(cfg,endpoint_id,spec=None):
    cfg=load_run(cfg);e=_spec(cfg,endpoint_id,spec);machine=read(e['machine']);start=time.monotonic()
    if Path(machine['final_matrix']).resolve()!=Path(e['matrix'],'expr_mat.csv').resolve():raise ValueError('machine final_matrix must be own CSV')
    from scripts.crgf.matrix import build
    matrix_path=Path(e['matrix']);matrix_path.mkdir(parents=True,exist_ok=True)
    if (matrix_path/'complete.json').exists():
        matrix=verify_legacy_receipt(matrix_path/'complete.json')
        if matrix['dependencies']['mask']['sha256']!=sha256(machine['final_mask']):raise ValueError('stale completed matrix')
    else:
        matrix=build(machine,matrix_path,cache=output_path(cfg,'coordinate_cache'))
    queue,protocol,contract=evaluation_contract(cfg);out=Path(e['evaluation']);out.mkdir(parents=True,exist_ok=True)
    deps=dict(matrix=identity(matrix_path/'complete.json'),machine=identity(e['machine']),**contract)
    if (out/'complete.json').exists() and read(out/'complete.json').get('paper_method_complete'):
        saved=read(out/'complete.json')
        if saved['dependencies']!=deps:raise ValueError('completed evaluation drift')
        return validate_endpoint(cfg,endpoint_id,e)
    from scripts.new_methods.accepted_profiles import run_profiles
    from scripts.new_methods.evaluation import fixed_panel
    from scripts.new_methods.common import json_safe
    profiles=out/'profiles_complete.json'
    if profiles.exists():
        saved=read(profiles)
        if saved['dependencies']!=deps:raise ValueError('profile identity drift')
        for obj in saved['outputs'].values():check_identity(obj)
        summary=saved['summary']
    else:
        summary=run_profiles(e,queue,protocol,out,endpoint_id)
        outputs={str(p.relative_to(out)):identity(p) for folder in ('engineering','paper_aligned_v1') for p in sorted((out/folder).rglob('*')) if p.is_file()}
        atomic_json(profiles,dict(status='succeeded',dependencies=deps,summary=json_safe(summary),outputs=outputs))
    summary['fixed_population']=fixed_panel(e,queue,out/'fixed')
    summary['authoritative_protocol']=contract['authority'];summary['wall_seconds']=time.monotonic()-start
    atomic_json(out/'summary.json',json_safe(summary))
    from scripts.paired_v2.evaluation import validate_evaluation
    validate_evaluation(e)
    atomic_json(out/'complete.json',dict(status='succeeded',paper_method_complete=True,
        profiles=['engineering_nuclear_v1','engineering_pearson_v1','paper_aligned_v1','fixed_auxiliary'],
        dependencies=deps,outputs=inventory(out)))
    return validate_endpoint(cfg,endpoint_id,e)

def validated_reuse(cfg,endpoint_id,source_id='BASE_MATCH'):
    cfg=load_run(cfg)
    if endpoint_id!='A2_NATIVE' or source_id!='BASE_MATCH':raise ValueError('unsupported reuse mapping')
    validation=validate_endpoint(cfg,source_id);e=endpoint_spec(cfg,source_id)
    bindings={k:identity(e[k]) for k in ('machine','final','checkpoint')}
    bindings.update(matrix=identity(Path(e['matrix'])/'complete.json'),evaluation=identity(Path(e['evaluation'])/'complete.json'))
    rec=dict(status='validated_reuse',source_endpoint=source_id,endpoint_id=endpoint_id,
        config_digest=digest(cfg),bindings=bindings,validation=validation,
        scope='original frozen endpoint; no new prediction or fabricated own matrix')
    atomic_json(output_path(cfg,'A2_CORE','endpoints',endpoint_id,'validated_reuse.json'),rec)
    return rec
