"""Authoritative 18-method scientific core, own matrix and own scClassify."""
import subprocess
import time
import numpy as np
import pandas as pd
from .common import *
from scripts.new_methods.accepted_profiles import run_profiles
from scripts.new_methods.evaluation import fixed_panel


def evaluation_contract(root=DEFAULT_ROOT):
    frozen_authority=Path(root)/'prepared/authoritative_protocol.json'
    authority=read(frozen_authority);queue=read(INPUT/'configs/queue.json')
    prepared=read(Path(queue['prepared'])/'complete.json');verify_outputs(prepared)
    protocol=Path(prepared['dependencies']['protocol']['path']);p=read(protocol)
    core={k:p[k] for k in authority['core']}
    if core!=authority['core']:raise ValueError('runtime evaluation scientific core differs from authoritative18 protocol')
    if sha256(protocol)!=authority['source_protocol']['sha256']:raise ValueError('authority source protocol changed')
    if prepared['model']['sha256']!=authority['reference_model']['sha256'] or sha256(prepared['model']['path'])!=authority['reference_model']['sha256']:
        raise ValueError('scClassify model differs from authoritative18')
    if queue['population']['sha256']!=authority['fixed_population']['sha256']:raise ValueError('fixed population differs')
    for key in ('reference','official_processing','official_expression','rscript'):
        if sha256(p[key])!=prepared['dependencies'][key]['sha256']:raise ValueError('evaluation producer/input changed: '+key)
    return queue,protocol,dict(authority=identity(frozen_authority),runtime_protocol=identity(protocol),
        scientific_core=core,reference_model=prepared['model'],prepared=identity(Path(queue['prepared'])/'complete.json'),
        producer=producer('evaluate'))


def spec_for(cfg, output=None):
    root=Path(cfg['output_root']);run=cfg['run_id']
    return dict(machine=cfg['machine'],matrix=str(root/'matrix'/run),
                evaluation=str(output or root/'evaluation'/run),cost_root=str(root))


def evaluate(config):
    cfg=load_config(config);queue,protocol,contract=evaluation_contract(cfg['runroot']);spec=spec_for(cfg)
    out=Path(spec['evaluation']);out.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    matrix=read(Path(spec['matrix'])/'complete.json');verify_outputs(matrix)
    if not matrix.get('validation_kind')=='independent_full_native_resize_lookup_and_conservation':raise ValueError('matrix lacks full native acceptance')
    dependencies=dict(matrix=identity(Path(spec['matrix'])/'complete.json'),machine=identity(spec['machine']),**contract)
    cached=out/'profiles_complete.json'
    if cached.exists():
        saved=read(cached)
        if saved['dependencies']!=dependencies:raise ValueError('evaluation dependencies changed')
        verify_outputs(saved);summary=saved['summary']
    else:
        summary=run_profiles(spec,queue,str(protocol),out,cfg['run_id'])
        atomic_json(cached,dict(status='succeeded',dependencies=dependencies,summary=summary,
            outputs={str(p.relative_to(out)):identity(p) for folder in ('engineering','paper_aligned_v1') for p in (out/folder).rglob('*') if p.is_file()}))
    summary['fixed_population']=fixed_panel(spec,queue,out/'fixed')
    summary['authoritative_protocol']=contract['authority']
    summary['wall_seconds']=time.monotonic()-start
    atomic_json(out/'summary.json',json_safe(summary))
    validate_evaluation(spec)
    rec=dict(status='succeeded',dependencies=dependencies,paper_method_complete=True,summary=summary,outputs=file_tree(out))
    atomic_json(out/'complete.json',json_safe(rec));return rec


def validate_evaluation(spec):
    matrix=Path(spec['matrix']);out=Path(spec['evaluation']);paper=out/'paper_aligned_v1'
    ids=pd.read_csv(matrix/'cell_mapping.csv').canonical_cell_id.to_numpy(np.int64)
    counts=np.load(matrix/'counts.npy',mmap_mode='r');totals=counts.sum(1);ng=(counts>0).sum(1)
    expected=(totals>=20)&(totals<=2000)&(ng>=20)
    qc=pd.read_csv(paper/'qc_all_cells.csv');ann=pd.read_csv(paper/'annotation.csv');scores=pd.read_csv(paper/'markers_per_cell.csv')
    if qc.cell_id.duplicated().any() or ann.cell_id.duplicated().any() or scores.cell_id.duplicated().any():raise ValueError('duplicate evaluation IDs')
    if not np.array_equal(qc.cell_id,ids) or not np.array_equal(qc.qc_pass,expected):raise ValueError('own matrix QC population differs')
    if set(ann.cell_id)!=set(ids[expected]) or set(scores.cell_id)!=set(ids[expected]):raise ValueError('scClassify or marker population differs')
    metrics=['positive_F1','negative_F1','positive_Precision','positive_Recall','negative_Precision','negative_Recall']
    if not set(metrics)<=set(scores):raise ValueError('required marker metrics missing')
    if not np.isfinite(scores[['positive_F1','negative_F1']].to_numpy()).all():raise ValueError('nonfinite primary F1')
    for name in ('type_correlations.csv','neighbour_curves.csv','size_factors.csv'):
        if not (paper/name).exists():raise ValueError('required evaluation artifact absent: '+name)
    structure=read(out/'engineering/structure/structure.json')
    if not {'disconnected_cell_count_8'}<=structure.keys():raise ValueError('required structure metrics absent')
    return dict(status='passed',all_ids=len(ids),qc_ids=int(expected.sum()),metrics=metrics)


def replay_historical_scores(dest):
    """Fresh official scoring of E1 own matrix with identity-verified labels."""
    queue,protocol,contract=evaluation_contract();dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    historical=INPUT/'V2_MORPH/evaluation/V2_MORPH';matrix=INPUT/'V2_MORPH/matrix/V2_MORPH'
    old=read(historical/'complete.json');verify_outputs(old);mrec=read(matrix/'complete.json');verify_outputs(mrec)
    if old['dependencies']['matrix']['sha256']!=sha256(matrix/'complete.json'):raise ValueError('historical matrix linkage changed')
    if read(historical/'summary.json')['model_sha256']!=contract['reference_model']['sha256']:raise ValueError('historical classifier identity changed')
    script=Path('/home/zwan0726/work/scripts/bidcell_benchmark/recompute_historical_paper_metrics.R')
    expected=read(AUTHORITY)['implementations'][script.name]['sha256']
    if sha256(script)!=expected:raise ValueError('authoritative replay scorer changed')
    for k in ('annotation.csv','markers_per_cell.csv','size_factors.csv'):
        obj=old['outputs']['paper_aligned_v1/'+k]
        if sha256(obj['path'])!=obj['sha256']:raise ValueError('historical annotation reuse dependency changed')
    spec=dict(method='V2_MORPH',protocol=str(protocol),prepared=queue['prepared'],native_csv=str(matrix/'expr_mat.csv'),
        annotation=str(historical/'paper_aligned_v1/annotation.csv'),previous_scores=str(historical/'paper_aligned_v1/markers_per_cell.csv'),
        size_factors=str(historical/'paper_aligned_v1/size_factors.csv'),output=str(dest/'paper'))
    atomic_json(dest/'input.json',spec)
    with (dest/'scoring.log').open('w') as log:
        subprocess.run([read(protocol)['rscript'],str(script),str(dest/'input.json')],stdout=log,stderr=subprocess.STDOUT,check=True,
            env={**os.environ,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4'})
    comparisons={};historical_table=AUTHORITY.parent/'methods/V2_MORPH/paper'
    for name in ('qc_all_cells.csv','annotation.csv','markers_per_cell.csv','size_factors.csv'):
        a=pd.read_csv(dest/'paper'/name).set_index('cell_id').sort_index()
        source=historical_table/name if (historical_table/name).exists() else historical/'paper_aligned_v1'/name
        b=pd.read_csv(source).set_index('cell_id').sort_index()
        if not a.index.equals(b.index) or not set(a.columns)<=set(b.columns):raise ValueError('historical score columns/IDs differ: '+name)
        b=b[a.columns]
        col_results={}
        for c in a:
            av,bv=a[c],b[c]
            if pd.api.types.is_numeric_dtype(av) and pd.api.types.is_numeric_dtype(bv):
                x,y=av.to_numpy(float),bv.to_numpy(float)
                finite=np.isfinite(x)
                if not np.array_equal(finite,np.isfinite(y)) or not np.allclose(x[finite],y[finite],atol=1e-10,rtol=1e-8):
                    raise ValueError('historical per-cell numeric/finite mismatch: '+name+'/'+c)
                col_results[c]=dict(finite=int(finite.sum()),na=int((~finite).sum()),max_abs=float(np.max(np.abs(x[finite]-y[finite]),initial=0)))
            elif not av.fillna('__NA__').equals(bv.fillna('__NA__')):raise ValueError('historical annotation/state mismatch: '+name+'/'+c)
        comparisons[name]=dict(rows=len(a),columns=col_results,source=identity(source))
    scores=pd.read_csv(dest/'paper/markers_per_cell.csv');table=pd.read_csv(AUTHORITY.parent/'main_table.csv').set_index('method')
    aggregates={}
    for metric in ('positive_F1','negative_F1','positive_Precision','positive_Recall','negative_Precision','negative_Recall'):
        value=float(scores[metric].mean());previous=float(table.loc['V2_MORPH','scclassify_'+metric])
        if not np.isclose(value,previous,atol=1e-10,rtol=1e-8):raise ValueError('historical aggregate mismatch: '+metric)
        aggregates[metric]=dict(replayed=value,authoritative=previous)
    rec=dict(status='passed',dependencies=contract,matrix=identity(matrix/'complete.json'),scorer=identity(script),
        comparisons=comparisons,aggregates=aggregates,atol=1e-10,rtol=1e-8,
        classification='reused only after matrix/model/core/QC/size-factor identity verified; scores freshly computed by authoritative R',outputs=file_tree(dest))
    atomic_json(dest/'complete.json',rec);return rec
