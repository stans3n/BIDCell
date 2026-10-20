"""Scientific API for the separately implemented supervisor (no launch/publication)."""
from pathlib import Path
import os
import numpy as np
import tifffile
from .contracts import (SCHEMA,CONFIG_SCHEMA,STAGES,STAGE_ORDER,NEW,EXISTING,METHODS,GT_ENDPOINTS,
                        PARENTS,PRODUCTION_STAGES,EVALUATION_STAGES,normalize_stage,DAG,DAG_SHA256)
from .common import (read,digest,identity,check_identity,dependencies,check_receipt,write_receipt,
                     within,atomic_json,inventory,progress as emit_progress,require_compute_authorization,sha256)
from .inputs import load_inputs,new_spec,resolve_parent,matrix_arrays

def config_check(cfg):
    required=('run_id','run_root','source_snapshot','source_digest','input_manifest','input_digest','domain_manifest','domain_digest','plan_manifest','plan_digest','comparison_manifest','python','rscript','resources')
    if cfg.get('schema')!=CONFIG_SCHEMA or any(not cfg.get(k) for k in required):raise ValueError('science config schema/frozen identity fields missing')
    if cfg.get('stage_order')!=list(STAGE_ORDER):raise ValueError('compute stage order/scope drift')
    if not Path(cfg['run_root']).is_absolute():raise ValueError('run root must be absolute')
    for key in ('source_snapshot','input_manifest','comparison_manifest'):within(cfg['run_root'],cfg[key])
    if cfg.get('continuation'):
        from .continuation import manifest
        manifest(cfg)
    else:within(cfg['run_root'],cfg['domain_manifest'])
    if sha256(cfg['comparison_manifest'])!=DAG_SHA256:raise ValueError('comparison DAG differs from root-approved bytes')
    if sha256(cfg['plan_manifest'])!=cfg['plan_digest']:raise ValueError('plan digest mismatch')
    if cfg['resources'].get('p2_workers') not in (1,2,4):raise ValueError('P2 frozen worker count must be 1,2 or4')
    return cfg

def receipt_path(cfg,stage):
    return within(cfg['run_root'],STAGES[normalize_stage(stage)]['required_receipt'])

def _validate_production(cfg,name,stage,manifest):
    root=within(cfg['run_root'],'endpoints',name);rec=check_receipt(root/'production_complete.json',dependencies(cfg,stage),cfg['run_root'])
    if rec.get('endpoint')!=name or rec.get('parent')!=PARENTS[name] or rec.get('stage')!=stage:raise ValueError('production endpoint/parent/stage mismatch')
    parent=resolve_parent(cfg,manifest,name);matrix=check_receipt(root/'matrix/complete.json',dependencies(cfg,name+'_MATRIX'),cfg['run_root'])
    if matrix.get('validation_kind')!='independent_full_native_resize_lookup_and_conservation':raise ValueError('full matrix oracle absent')
    ids,genes,counts=matrix_arrays(new_spec(cfg,name));mask=tifffile.imread(root/'final.tif');nuc=tifffile.imread(parent['nuclei'])
    if len(ids)!=107955 or len(genes)!=313 or not np.array_equal(mask[nuc>0],nuc[nuc>0]) or not np.array_equal(np.unique(mask)[1:],ids):raise ValueError('production complete ID/nuclear semantics differ')
    if name in GT_ENDPOINTS:
        gt=check_receipt(root/'gt/complete.json',dependencies(cfg,name+'_GT'),cfg['run_root'])
        for field in ('foreground_equal','nuclear_owner_exact','all_ids_equal','positive_to_positive_only','donor_components_do_not_increase','nuclear_connections_preserved','new_recipient_pixels_nucleus_connected'):
            if gt['invariants'].get(field) is not True:raise ValueError('GT invariant absent: '+field)
        from .matrix import check_parent_conservation
        check_parent_conservation(parent,matrix,root/'matrix')
        baseline=tifffile.imread(parent['final'])
        if not np.array_equal(mask>0,baseline>0):raise ValueError('GT parent foreground changed')
    else:
        from .p2 import validate_process
        validate_process(root/'predictions')
        diag=read(root/'transfer_diagnostics.json')
        if not diag['postconnect_support_equal'] or not diag['postconnect_changes_confined_to_extranuclear_conflicts']:raise ValueError('P2 postconnect invariants missing')
    return rec

def validate_stage(cfg,stage,manifest=None):
    cfg=read(cfg) if isinstance(cfg,(str,Path)) else cfg;config_check(cfg);stage=normalize_stage(stage)
    if cfg.get('continuation'):
        from .continuation import validate_adopted_stage,manifest as continuation_manifest
        if stage in continuation_manifest(cfg)['stages']:return validate_adopted_stage(cfg,stage)
    manifest=manifest or load_inputs(cfg,verify=False)
    rec=check_receipt(receipt_path(cfg,stage),output_root=cfg['run_root'])
    if rec['dependencies'].get('config_digest')!=digest(cfg):raise ValueError('stage config identity mismatch')
    if stage in PRODUCTION_STAGES:return _validate_production(cfg,PRODUCTION_STAGES[stage],stage,manifest)
    if stage in EVALUATION_STAGES:
        from .coverage import validate_endpoint_products
        name=EVALUATION_STAGES[stage];spec=new_spec(cfg,name)
        expected=dict(dependencies(cfg,stage),production=identity(Path(spec['root'])/'production_complete.json'))
        if rec['dependencies']!=expected or rec.get('full_evaluate') is not True:raise ValueError('full evaluate stage identity/coverage mismatch')
        validate_endpoint_products(spec,manifest=manifest,cfg=cfg);return rec
    if stage in ('04','13'):
        from .coverage import validate_analysis
        methods=DAG['initial_methods9'] if stage=='04' else list(METHODS)
        return validate_analysis(cfg,'initial' if stage=='04' else 'all12',methods)
    if stage=='01':
        from .frozen_domains import validate_frozen
        validate_frozen(cfg)
        if rec.get('methods')!=list(EXISTING) or rec.get('coordinate_rows')!=34442716:raise ValueError('input semantic coverage missing')
    elif stage=='05':
        from .p2 import assert_matrix_equal,validate_process
        root=within(cfg['run_root'],'replay/SIMPLE_NATIVE');validate_process(root/'predictions')
        if not rec.get('final_pixels_equal') or rec.get('scope')!='full_field':raise ValueError('full Simple replay missing')
        assert_matrix_equal(root/'matrix',manifest['endpoints']['MORPH_SIMPLE_FUSION']['matrix'])
        if not np.array_equal(tifffile.imread(root/'predictions/final.tif'),tifffile.imread(manifest['endpoints']['MORPH_SIMPLE_FUSION']['final'])):raise ValueError('Simple disabled replay final differs')
    elif stage=='12':
        from .coverage import validate_endpoint_products
        if rec.get('methods')!=list(EXISTING):raise ValueError('old8 completion coverage differs')
        for name in EXISTING:
            root=within(cfg['run_root'],'reused',name);check_receipt(root/'complete.json')
            reuse=read(root/'complete.json')
            validate_endpoint_products(manifest['endpoints'][name],root/'structure',root/'bt',reuse['spatial_root'],manifest)
    elif stage=='14':
        if rec.get('full_evaluations')!=list(NEW) or rec.get('methods')!=list(METHODS) or rec.get('comparisons')!=26 or rec.get('factorial_contrasts')!=19:raise ValueError('compute complete coverage mismatch')
    elif stage=='15':
        if rec.get('terminal_state')!='COMPUTE_COMPLETE_AWAITING_MANUAL_RESUME' or rec.get('exit_code')!=0:raise ValueError('compute exit not successful manual hold')
        check_identity(rec['compute_complete'])
    return rec

def ensure_reused(cfg,manifest,callback=None,lock_fds=(),validate_execution=None):
    from .evaluation import structural,source_runtime,run_r
    from .bt import run_endpoint
    from .coverage import validate_endpoint_products
    for i,name in enumerate(EXISTING):
        spec=manifest['endpoints'][name];out=within(cfg['run_root'],'reused',name);dep=dependencies(cfg,'REUSE_'+name)
        if (out/'complete.json').exists():check_receipt(out/'complete.json',dep);continue
        out.mkdir(parents=True,exist_ok=True)
        structural(spec,manifest,out/'structure',dict(dep,part='structure'),callback)
        run_endpoint(cfg,manifest,spec,out/'bt',callback)
        spatial=Path(spec.get('spatial_root',Path(spec['evaluation'])/'paper_aligned_v1'))
        if not (spatial/'spatial_diversity_correlations.csv').exists():
            spdep=dict(dep,part='missing_author_spatial',paper_source=identity(Path(spec['evaluation'])/'complete.json'))
            if (out/'spatial_complete.json').exists():check_receipt(out/'spatial_complete.json',spdep)
            else:
                runtime=source_runtime(cfg,spdep);runtime['paper_input']=str(spatial)
                rp=out/'r_runtime.json';atomic_json(rp,runtime)
                run_r(cfg,Path(__file__).with_name('spatial_diversity.R'),[rp,out],out,lock_fds,validate_execution,callback,status_file='spatial_execution_status.txt')
                write_receipt(out/'spatial_complete.json',spdep,{f:identity(out/'paper_aligned_v1'/f) for f in ('spatial_diversity_bins.csv','spatial_diversity_correlations.csv')})
            spatial=out/'paper_aligned_v1'
        cov=validate_endpoint_products(spec,out/'structure',out/'bt',spatial,manifest);atomic_json(out/'coverage.json',cov)
        write_receipt(out/'complete.json',dep,inventory(out),endpoint=name,classification_reused=True,source_evaluation=identity(Path(spec['evaluation'])/'complete.json'),spatial_root=str(spatial),coverage=cov)
        emit_progress(callback,'reused_endpoint_completion',i+1,len(EXISTING),receipt=str(out/'complete.json'))

def analyze(cfg,manifest,phase,callback=None):
    from . import populations,comparisons as compare_module,boundary_analysis,bt,roi
    # Import by name to avoid the contracts function sharing the module name.
    import importlib
    compare_module=importlib.import_module(__package__+'.comparisons')
    methods=DAG['initial_methods9'] if phase=='initial' else list(METHODS)
    specs={m:manifest['endpoints'][m] if m in EXISTING else new_spec(cfg,m) for m in methods}
    out=within(cfg['run_root'],'analysis',phase);stage='04' if phase=='initial' else '13';dep=dependencies(cfg,stage)
    from .continuation import initial_root
    initial=initial_root(cfg) if phase=='all12' else None
    if initial is not None:
        from .coverage import validate_analysis
        validate_analysis(cfg,'initial',DAG['initial_methods9'])
        validate_stage(cfg,'12',manifest)
    from .evaluation_pairs import reuse_sources
    tables=populations.prepare(cfg,manifest,specs,out/'populations',callback,reuse_sources(cfg,methods,'populations',initial))
    compare_module.run(cfg,manifest,specs,tables,out/'populations',out/'comparisons',callback,reuse_sources(cfg,methods,'comparisons',initial))
    boundary_analysis.run(cfg,manifest,specs,out/'boundary',callback,initial/'boundary' if initial else None,include_structure=phase=='all12')
    bt.paired(cfg,manifest,specs,out/'bt',callback,reuse_sources(cfg,methods,'bt',initial))
    roi.render(cfg,manifest,specs,out/'roi',callback)
    atomic_json(out/'availability.json',dict(methods=methods,not_yet_available=[m for m in METHODS if m not in methods],
                frozen_domain=identity(cfg['domain_manifest']),initial_not_final=phase=='initial',
                pending_old_endpoint_completion=list(EXISTING) if phase=='initial' else [],
                pending_products=['old8_full_4_and_8_structure','old8_official_six_marker_BT','structure_local_costs'] if phase=='initial' else [],
                initial_producer=identity(initial/'complete.json') if initial else None))
    return write_receipt(out/'complete.json',dep,inventory(out),methods=methods,phase=phase,selection_unchanged=True)

def run_stage(config,stage,progress=None,lock_fds=(),validate_execution=None,compute_authorization=None):
    cfg=read(config) if isinstance(config,(str,Path)) else dict(config);config_check(cfg);stage=normalize_stage(stage)
    if stage=='15':raise ValueError('stage15 is the supervisor terminal commit after science14; no scientific work')
    if not lock_fds or validate_execution is None:raise PermissionError('science stages require inherited compute locks and snapshot validator')
    for fd in lock_fds:os.fstat(fd)
    validate_execution()
    production=within(cfg['run_root'],'endpoints',EVALUATION_STAGES[stage],'production_complete.json') if stage in EVALUATION_STAGES else None
    require_compute_authorization(cfg,compute_authorization,stage,production)
    manifest=load_inputs(cfg)
    # Every predecessor is an accepted immutable receipt; values do not control execution.
    for previous in STAGE_ORDER[:STAGE_ORDER.index(stage)]:validate_stage(cfg,previous,manifest)
    if receipt_path(cfg,stage).exists():return validate_stage(cfg,stage,manifest)
    if stage=='01':
        from .inputs import validate_semantics
        from .frozen_domains import validate_frozen
        out=validate_semantics(cfg,manifest,progress);validate_frozen(cfg)
        result=write_receipt(receipt_path(cfg,stage),dependencies(cfg,stage),dict(inventory(out),domain=identity(cfg['domain_manifest'])),methods=list(EXISTING),coordinate_rows=34442716)
    elif stage in PRODUCTION_STAGES:
        from .matrix import produce_matrix
        name=PRODUCTION_STAGES[stage];out=within(cfg['run_root'],'endpoints',name)
        if name in GT_ENDPOINTS:
            from .gt import refine
            refine(cfg,manifest,name,progress)
        else:
            from .p2 import produce
            produce(cfg,manifest,progress,lock_fds)
        mat,conservation=produce_matrix(cfg,manifest,name,progress)
        if name=='P2_SIMPLE':
            from .p2_diagnostics import transfer_diagnostics
            from .continuation import native_replay_root
            transfer_diagnostics(cfg,name,out/'predictions',native_replay_root(cfg)/'predictions',read(out/'machine.json'),out/'matrix')
        result=write_receipt(out/'production_complete.json',dependencies(cfg,stage),inventory(out),endpoint=name,parent=PARENTS[name],stage=stage,
            conservation=conservation,evaluation_complete=False,no_op_with_parent=bool(np.array_equal(tifffile.imread(out/'final.tif'),tifffile.imread(resolve_parent(cfg,manifest,name)['final']))))
    elif stage in EVALUATION_STAGES:
        from .evaluation import evaluate_endpoint
        result=evaluate_endpoint(cfg,manifest,EVALUATION_STAGES[stage],compute_authorization,progress,lock_fds,validate_execution)
    elif stage in ('04','13'):result=analyze(cfg,manifest,'initial' if stage=='04' else 'all12',progress)
    elif stage=='05':
        from .p2 import run_replay
        result=run_replay(cfg,manifest,progress,lock_fds)
    elif stage=='12':
        ensure_reused(cfg,manifest,progress,lock_fds,validate_execution)
        result=write_receipt(receipt_path(cfg,stage),dependencies(cfg,stage),{n:identity(within(cfg['run_root'],'reused',n,'complete.json')) for n in EXISTING},methods=list(EXISTING),classification_reused=True)
    elif stage=='14':
        from .coverage import complete_check
        result=complete_check(cfg,manifest)
    validate_stage(cfg,stage,manifest)
    emit_progress(progress,'stage_'+stage+'_validated',1,1,receipt=str(receipt_path(cfg,stage)))
    return result
