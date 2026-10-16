"""Manual-only 28+2 table publication, preserving every old non-Purity field.

Context synchronization is an explicit controller integration step after this
module returns a verified publication receipt. Publication is never goal completion.
"""
from __future__ import annotations
import csv
import hashlib
import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
from . import EXISTING, NEW, PARENTS, N_CELLS, ELIGIBLE
from .common import (atomic, atomic_json, check_identity, check_receipt, dependencies, digest,
                     fsync_dir, identity, progress, read, require_manual_resume, within)
from .inputs import load_inputs, load_manifest, TABLE, REGISTRY, TABLE_SHA, REGISTRY_SHA, matrix_arrays

RESULT_ROOT=Path('/home/zwan0726/work/outputs/bidcell_historical_dual_bt_gt_fusion_20261017')
PROTOCOL='DUAL-F1-BTGT-v1.0-30methods-20261017'
PURITY_VERSION='PURITY-30methods-v1.0-20261017'
PURITY_COLUMNS={'scclassify_Purity_F1','fixed_Purity_F1','purity_scaling_version','purity_scaling_path'}
MISSING_HISTORICAL_MASK_ALLOWLIST=frozenset(('C','SP'))
MISSING_MASK_REASON='original historical mask deleted; original identity retained; actual pixels unavailable for verification'
SPATIAL_COLUMNS={
    'cv_total_transciprts':'scclassify_spatial_entropy_vs_cv_total_transciprts',
    'cv_total_genes':'scclassify_spatial_entropy_vs_cv_total_genes',
    'cv_cell_area':'scclassify_spatial_entropy_vs_cv_cell_area',
    'sd_total_transciprts':'scclassify_spatial_entropy_vs_sd_total_transciprts',
    'sd_total_genes':'scclassify_spatial_entropy_vs_sd_total_genes',
    'sd_cell_area':'scclassify_spatial_entropy_vs_sd_cell_area',
}


def read_strings(path):
    with Path(path).open(newline='',encoding='utf-8-sig') as stream:
        reader=csv.DictReader(stream);rows=list(reader)
        return reader.fieldnames,rows


def write_strings(path,columns,rows):
    def writer(stream):
        output=csv.DictWriter(stream,fieldnames=columns,extrasaction='raise')
        output.writeheader();output.writerows(rows)
    atomic(path,writer,False)


def purity(rows):
    result={'version':PURITY_VERSION,'comparison_set':[r['method'] for r in rows],'profiles':{}}
    for profile in ('scclassify','fixed'):
        values=np.asarray([[float(r[profile+'_positive_F1']),float(r[profile+'_negative_F1'])] for r in rows])
        if not np.isfinite(values).all():raise ValueError('30-method Purity raw means must all be finite')
        lo=values.min(0);hi=values.max(0)
        entry=dict(positive_min=float(lo[0]),positive_max=float(hi[0]),negative_min=float(lo[1]),negative_max=float(hi[1]),methods={})
        if np.any(hi==lo):
            entry['status']='not_identifiable_zero_range'
            for r in rows:entry['methods'][r['method']]=None
        else:
            z=(values-lo)/(hi-lo);a=z[:,0];b=1-z[:,1]
            scores=np.divide(2*a*b,a+b,out=np.zeros(len(a)),where=a+b!=0)
            entry['status']='computed'
            for row,value in zip(rows,scores):entry['methods'][row['method']]=float(value)
        result['profiles'][profile]=entry
    return result


def preserve_old_raw(old_rows,new_rows,columns):
    new={r['method']:r for r in new_rows}
    if len(new)!=len(new_rows):raise ValueError('duplicate method rows')
    for row in old_rows:
        for column in columns:
            if column not in PURITY_COLUMNS and new[row['method']][column]!=row[column]:
                raise ValueError('old 28 non-Purity raw string changed: '+row['method']+'/'+column)


def mask_content(path):
    array=tifffile.imread(path)
    if array.ndim!=2 or not np.issubdtype(array.dtype,np.integer) or np.any(array<0):raise ValueError('invalid mask content')
    h=hashlib.sha256()
    h.update(np.asarray(array.shape,dtype='<i8').tobytes())
    for row in range(0,array.shape[0],256):h.update(np.asarray(array[row:row+256],dtype='<u4').tobytes())
    return h.hexdigest()


def mask_verification_summary(records):
    """Count readable method entries and their contents, never impute missing masks."""
    missing=sorted(name for name,record in records.items() if record['available'] is False)
    if any(type(record.get('available')) is not bool for record in records.values()):
        raise ValueError('mask availability must be explicit')
    if not set(missing)<=MISSING_HISTORICAL_MASK_ALLOWLIST or set(missing)&set(EXISTING+NEW):
        raise ValueError('missing mask is outside the frozen C/SP exception')
    contents=set()
    for record in records.values():
        if record['available']:
            content=record.get('content_sha256')
            if not isinstance(content,str) or len(content)!=64:
                raise ValueError('readable mask lacks an actual content hash')
            contents.add(content)
        elif record.get('content_sha256') is not None or record.get('alias_of') is not None:
            raise ValueError('missing mask cannot have a content hash or alias')
    return dict(available_mask_method_entries=len(records)-len(missing),missing_mask_methods=missing,
                available_unique_mask_contents=len(contents),
                unique_mask_contents=None if missing else len(contents),
                unique_mask_contents_verified=not missing,
                unique_mask_contents_status='not_fully_verified_missing_historical_masks' if missing else 'verified_all_method_masks')


def new_table_row(cfg,manifest,name,columns):
    from .evaluation import new_spec
    spec=new_spec(cfg,name);evaluation=Path(spec['evaluation']);root=Path(spec['root']);matrix=Path(spec['matrix'])
    summary=read(evaluation/'summary.json');structure=summary['structure'];mr=read(matrix/'complete.json');er=identity(evaluation/'complete.json')
    row={c:'' for c in columns}
    row.update(method=name,experiment_group=cfg['run_id'],n_segmented_cells=str(N_CELLS),
               capture_pct=str(100*mr['totals']['assigned']/ELIGIBLE),
               mean_transcripts_per_cell=str(structure['mean_transcripts_per_cell']),
               mean_genes_per_cell=str(structure['mean_genes_per_cell']),mean_cell_area_um2=str(structure['mean_cell_area_um2']),
               median_cell_area_um2=str(structure['median_cell_area_um2']),p95_cell_area_um2=str(structure['p95_cell_area_um2']),
               disconnected_cells_4=str(structure['disconnected_cells_4']),disconnected_cells_8=str(structure['disconnected_cells_8']),
               scclassify_qc_cells=str(summary['n_after_qc']),scclassify_qc_retention_pct=str(100*summary['qc_retention']),
               scclassify_unassigned_pct=str(100*summary['unassigned_fraction']),fixed_population_cells='103000',
               geometry_source='own_final_native_mask',evaluation_wall_seconds=str(summary['wall_seconds']),
               matrix_path=str(matrix),matrix_sha256=identity(matrix/'counts.npy')['sha256'],evaluation_path=str(evaluation),
               evaluation_sha256=er['sha256'],nuclear_preservation_fraction=str(structure['nuclear_preservation_fraction']),new_IDs='0',
               model_kind='GT_POSTPROCESS',training_identity_scope='no_new_training; parent checkpoint provenance only',
               matrix_counts_path=str(matrix/'counts.npy'),matrix_counts_sha256=identity(matrix/'counts.npy')['sha256'],
               matrix_receipt_path=str(matrix/'complete.json'),matrix_receipt_sha256=identity(matrix/'complete.json')['sha256'],
               matrix_mapping_sha256=identity(matrix/'cell_mapping.csv')['sha256'],matrix_gene_mapping_sha256=identity(matrix/'gene_mapping.csv')['sha256'],
               evaluation_receipt_path=str(evaluation/'complete.json'),evaluation_receipt_sha256=er['sha256'],
               annotation_path=str(evaluation/'paper_aligned_v1/annotation.csv'),annotation_sha256=identity(evaluation/'paper_aligned_v1/annotation.csv')['sha256'],
               artifact_registry_path=str(RESULT_ROOT/'artifact_registry.json'),canonical_artifact_registry_path=str(RESULT_ROOT/'artifact_registry.json'),
               reporting_protocol=PROTOCOL,execution_status='COMPLETED',boundary_run_id=cfg['run_id'],boundary_node=name,
               scientific_source_commit=cfg.get('source_commit',''),scientific_config_digest=digest(cfg),
               boundary_full_receipt=str(root/'production_complete.json'),checkpoint_parent=PARENTS[name])
    checkpoint=read(manifest['endpoints'][PARENTS[name]]['machine'])['checkpoint']
    row['checkpoint_path']=checkpoint;row['checkpoint_sha256']=identity(checkpoint)['sha256']
    for key,val in summary['metrics'].items():
        row[key]='NA' if val['mean'] is None else str(val['mean']);row[key+'_n']=str(val['n_finite'])
    corr=pd.read_csv(evaluation/'paper_aligned_v1/type_correlations.csv')
    for short,definition in [('expression','mean_logcounts'),('detection','detection')]:
        values=corr.loc[corr.definition.eq(definition),'pearson'].to_numpy(float);finite=np.isfinite(values)
        row[f'scclassify_{short}_pearson_type_mean']=str(float(values[finite].mean())) if finite.any() else 'NA'
        row[f'scclassify_{short}_pearson_types_n']=str(int(finite.sum()))
    spatial=pd.read_csv(evaluation/'paper_aligned_v1/spatial_diversity_correlations.csv')
    if spatial.metric.duplicated().any() or set(spatial.metric)!=set(SPATIAL_COLUMNS):
        raise ValueError('author spatial statistics must contain each of the six frozen metrics exactly once')
    if spatial.n_bins.nunique(dropna=False)!=1 or not np.isfinite(spatial.n_bins).all() or (spatial.n_bins<0).any() or (spatial.n_bins%1!=0).any():
        raise ValueError('author spatial bin denominator is invalid or inconsistent')
    for rec in spatial.to_dict('records'):
        key=SPATIAL_COLUMNS[rec['metric']]
        if key not in row:raise ValueError('unrecognized author spatial statistic: '+key)
        row[key]=str(rec['pearson']) if np.isfinite(rec['pearson']) else 'NA'
    row['scclassify_spatial_bins_n']=str(int(spatial.n_bins.iloc[0]))
    return row


def prepare_publication(cfg,grant,callback=None):
    require_manual_resume(cfg,grant)
    manifest=load_inputs(cfg)
    resolved_masks=require_publication_masks(cfg,manifest)
    check_receipt(within(cfg['run_root'],'evaluation8/complete.json'),dependencies(cfg,'EVALUATION8'),cfg['run_root'])
    stage=within(cfg['run_root'],'publication_staging');stage.mkdir(parents=True,exist_ok=True)
    deps=dependencies(cfg,'PUBLICATION')
    deps.update(manual_resume=identity(grant),evaluation8=identity(within(cfg['run_root'],'evaluation8/complete.json')))
    if (stage/'staging_complete.json').exists():
        validate_bundle(stage,deps);return stage
    columns,old_rows=read_strings(TABLE)
    if len(old_rows)!=28 or len({r['method'] for r in old_rows})!=28:raise ValueError('old table has wrong 28-method domain')
    rows=[dict(r) for r in old_rows]+[new_table_row(cfg,manifest,name,columns) for name in NEW]
    if len(rows)!=30 or len({r['method'] for r in rows})!=30 or sum(r['method']=='B0' for r in rows)!=1:
        raise ValueError('published method domain must be 28+2=30 with one B0')
    scaling=purity(rows)
    for row in rows:
        for p in ('scclassify','fixed'):
            value=scaling['profiles'][p]['methods'][row['method']]
            row[p+'_Purity_F1']='NA' if value is None else str(value)
        row['purity_scaling_version']=PURITY_VERSION;row['purity_scaling_path']=str(RESULT_ROOT/'purity_scaling.json')
    preserve_old_raw(old_rows,rows,columns)
    write_strings(stage/'main_table.csv',columns,rows)
    atomic_json(stage/'purity_scaling.json',scaling)
    mask_records={}
    from .evaluation import new_spec
    for i,name in enumerate([r['method'] for r in rows]):
        if name in NEW:
            spec=new_spec(cfg,name);mask=identity(spec['final']);mr=identity(Path(spec['matrix'])/'complete.json')
        else:
            hist=manifest['historical_masks'][name];mask=resolved_masks[name];mr=hist['matrix_receipt']
        if mask is None:
            mask_records[name]=dict(mask=hist['mask'],matrix_receipt=mr,available=False,
                content_sha256=None,alias_of=None,unavailable_reason=MISSING_MASK_REASON)
            progress(callback,'publication_mask_content',i+1,30,method=name,available=False)
            continue
        check_identity(mask);check_identity(mr)
        content=mask_content(mask['path'])
        alias=next((other for other,obj in mask_records.items() if obj['available'] and obj['content_sha256']==content),None)
        mask_records[name]=dict(mask=mask,matrix_receipt=mr,available=True,content_sha256=content,alias_of=alias)
        progress(callback,'publication_mask_content',i+1,30,method=name,available=True)
    mask_summary=mask_verification_summary(mask_records)
    registry=dict(protocol=PROTOCOL,status='verified_30_method_extension',methods=[r['method'] for r in rows],new_methods=list(NEW),
                  historical28_registry=identity(REGISTRY),historical28_table=identity(TABLE),
                  historical28_chain=read(REGISTRY),mask_content=mask_records,
                  frozen_input_manifest=identity(cfg['input_manifest']),run_root=str(Path(cfg['run_root']).resolve()),
                  missing_historical_mask_allowlist=sorted(MISSING_HISTORICAL_MASK_ALLOWLIST),
                  author_reference={k:manifest['files'][str(Path(v).resolve())] for k,v in manifest['author_reference'].items()},
                  historical_mask_recovery_receipts={name:identity(recovery_paths(cfg,name,record)[1])
                      for name,record in manifest['historical_masks'].items() if not record['available'] and resolved_masks[name] is not None},
                  **mask_summary,
                  comparison_entries=30,existing_six_B_T=identity(within(cfg['run_root'],'bt_existing6/complete.json')),
                  evaluation8=identity(within(cfg['run_root'],'evaluation8/complete.json')),dependencies=deps,new_endpoints={})
    for name in NEW:
        spec=new_spec(cfg,name);matrix=Path(spec['matrix']);ev=Path(spec['evaluation'])
        ids,genes,counts=matrix_arrays(spec)
        registry['new_endpoints'][name]=dict(parent=PARENTS[name],mask=identity(spec['final']),machine=identity(spec['machine']),
              matrix=identity(matrix/'counts.npy'),matrix_shape=list(counts.shape),matrix_dtype=str(counts.dtype),
              cell_mapping=identity(matrix/'cell_mapping.csv'),gene_mapping=identity(matrix/'gene_mapping.csv'),
              matrix_receipt=identity(matrix/'complete.json'),matrix_validation=identity(matrix/'validation.json'),
              owner_files={p.name:identity(p) for p in sorted(matrix.glob('owners_*.npy'))},
              annotation=identity(ev/'paper_aligned_v1/annotation.csv'),sc_scores=identity(spec['sc_scores']),fixed_scores=identity(spec['fixed_scores']),
              evaluation_receipt=identity(ev/'complete.json'),production_receipt=identity(Path(spec['root'])/'production_complete.json'),
              no_op_with_parent=mask_records[name]['content_sha256']==mask_records[PARENTS[name]]['content_sha256'])
    atomic_json(stage/'artifact_registry.json',registry)
    atomic_json(stage/'publication_notes.json',dict(protocol=PROTOCOL,methods=30,**mask_summary,
                missing_historical_mask_allowlist=sorted(MISSING_HISTORICAL_MASK_ALLOWLIST),
                missing_mask_policy='only frozen unavailable C/SP; historical rows, raw metrics and original identity chains retained; no substitute masks',
                fixed_and_sc_F1_parallel=True,negative_results_included=True,old28_nonpurity_strings_preserved=True,
                CSV_encoding='UTF-8; input BOM accepted and output header written without BOM; all non-Purity data strings unchanged',
                boundary_accuracy_claim=False,GT='Geometry+Type, not ground truth',
                context_sync='required controller step after verified publication; not yet complete'))
    outputs={p.name:{k:v for k,v in identity(p).items() if k!='path'} for p in stage.iterdir() if p.is_file() and p.name!='staging_complete.json'}
    atomic_json(stage/'staging_complete.json',dict(status='succeeded',dependencies=deps,outputs=outputs))
    validate_bundle(stage,deps)
    return stage


def recovery_paths(cfg,name,record):
    relative='recovered_historical_masks/'+name+'/final.tif'
    if record.get('available') or record.get('recovery_relative')!=relative:
        raise ValueError('historical mask is not a frozen missing artifact with an exact recovery location')
    path=within(cfg['run_root'],relative)
    if path.is_symlink():raise ValueError('recovered mask must be a real copy')
    return path,path.with_name('recovery_complete.json')


def _check_historical_mask_binding(record):
    if type(record.get('available')) is not bool:
        raise ValueError('historical availability is not frozen as a boolean')
    check_identity(record['matrix_receipt'])
    bound=read(record['matrix_receipt']['path'])['dependencies']['mask']
    if any(bound[k]!=record['mask'][k] for k in ('path','bytes','sha256')):
        raise ValueError('frozen original mask identity is not bound by its original matrix receipt')


def _recovery_dependencies(base,name,record):
    deps={key:base[key] for key in ('schema','run_id','config_digest','source_digest','input_digest','plan_sha256')}
    return dict(deps,stage='HISTORICAL_MASK_RECOVERY_'+name,
                original_matrix_receipt=record['matrix_receipt'],expected_mask=record['mask'])


def _allow_frozen_missing(name,record):
    if name not in MISSING_HISTORICAL_MASK_ALLOWLIST or record['available'] is not False or name in EXISTING+NEW:
        raise ValueError('missing mask is outside the frozen C/SP exception: '+name)


def _copy_frozen_mask(cfg,name,record,source):
    """Recover known artifact bytes only; this does not admit a new mask identity."""
    target,receipt=recovery_paths(cfg,name,record)
    original=record['mask'];_check_historical_mask_binding(record)
    source=Path(source).resolve(strict=True)
    source_identity=identity(source)
    if any(source_identity[k]!=original[k] for k in ('bytes','sha256')):
        raise ValueError('historical recovery source differs from the exact frozen original bytes')
    deps=_recovery_dependencies(dependencies(cfg,'PUBLICATION'),name,record)
    if receipt.exists():
        saved=check_receipt(receipt,deps,cfg['run_root'])
        if saved['outputs']['mask']['path']!=str(target):raise ValueError('recovery receipt targets another mask')
        return saved
    if target.exists():
        check_identity(dict(original,path=str(target)))
    else:
        def writer(stream):
            with source.open('rb') as inp:shutil.copyfileobj(inp,stream,8*1024*1024)
        atomic(target,writer)
    check_identity(dict(original,path=str(target)))
    result=dict(status='succeeded',dependencies=deps,method=name,outputs={'mask':identity(target)},
                recovered_from=source_identity,exact_original_artifact=True)
    atomic_json(receipt,result)
    return result


def recover_historical_mask(cfg,manual_resume,name,source):
    require_manual_resume(cfg,manual_resume)
    manifest=load_inputs(cfg)
    if name not in manifest['historical_masks']:raise ValueError('unregistered historical method')
    return _copy_frozen_mask(cfg,name,manifest['historical_masks'][name],source)


def require_publication_masks(cfg,manifest):
    """Resolve actual bytes; only frozen unavailable C/SP may remain explicit None.

    The original manifest is never rewritten. An incomplete or corrupt recovery,
    any other missing method, and drift of a previously available mask still fail.
    """
    result={}
    for name,record in manifest['historical_masks'].items():
        _check_historical_mask_binding(record)
        if record['available']:
            check_identity(record['mask']);result[name]=record['mask'];continue
        target,receipt=recovery_paths(cfg,name,record)
        if not target.is_file() or not receipt.is_file():
            if target.exists() or receipt.exists() or Path(record['mask']['path']).exists():
                raise ValueError('incomplete or unregistered historical mask recovery: '+name)
            _allow_frozen_missing(name,record)
            result[name]=None;continue
        deps=_recovery_dependencies(dependencies(cfg,'PUBLICATION'),name,record)
        restored=check_receipt(receipt,deps,cfg['run_root'])
        if not restored.get('exact_original_artifact') or restored['outputs']['mask']['path']!=str(target):
            raise ValueError('historical mask recovery receipt identity mismatch')
        check_identity(dict(record['mask'],path=str(target)))
        result[name]=restored['outputs']['mask']
    return result


def validate_mask_records(reg,methods,deps):
    """Independently verify the readable subset and the exact frozen exceptions."""
    records=reg['mask_content']
    if reg['comparison_entries']!=30 or reg['methods']!=methods or set(records)!=set(methods):
        raise ValueError('publication mask/method coverage mismatch')
    if reg.get('dependencies')!=deps or reg.get('missing_historical_mask_allowlist')!=sorted(MISSING_HISTORICAL_MASK_ALLOWLIST):
        raise ValueError('publication dependency or missing-mask policy mismatch')
    frozen=check_identity(reg['frozen_input_manifest'])
    manifest=load_manifest(frozen,verify=False)
    if digest(manifest)!=deps['input_digest']:
        raise ValueError('publication frozen input digest differs')
    run_root=Path(reg['run_root']).resolve()
    if frozen!=run_root/'configs/inputs_manifest.json':
        raise ValueError('publication frozen input manifest is outside its run')
    if set(manifest['historical_masks'])|set(NEW)!=set(methods) or set(reg['new_endpoints'])!=set(NEW):
        raise ValueError('publication historical/new method domain mismatch')
    summary=mask_verification_summary(records)
    if any(key not in reg or reg[key]!=value for key,value in summary.items()):
        raise ValueError('mask availability or unique-content counts inconsistent')
    recoveries=reg.get('historical_mask_recovery_receipts',{})
    expected_recoveries=set();seen={}
    for name in methods:
        record=records[name]
        if name in NEW:
            expected=reg['new_endpoints'][name]
            if not record['available'] or any(record[key]!=expected[key] for key in ('mask','matrix_receipt')):
                raise ValueError('new scientific endpoint mask identity/availability mismatch')
        else:
            hist=manifest['historical_masks'][name]
            _check_historical_mask_binding(hist)
            if record['matrix_receipt']!=hist['matrix_receipt']:
                raise ValueError('historical matrix receipt differs from frozen identity')
            if hist['available']:
                if not record['available'] or record['mask']!=hist['mask']:
                    raise ValueError('previously available historical mask cannot be waived')
            else:
                target,receipt=recovery_paths({'run_root':str(run_root)},name,hist)
                if not record['available']:
                    _allow_frozen_missing(name,hist)
                    if record['mask']!=hist['mask'] or record.get('unavailable_reason')!=MISSING_MASK_REASON:
                        raise ValueError('missing historical mask lost its original identity or reason')
                    if Path(hist['mask']['path']).exists() or target.exists() or receipt.exists():
                        raise ValueError('missing mask has unverified original or recovery bytes')
                else:
                    expected_recoveries.add(name)
                    if name not in recoveries or check_identity(recoveries[name])!=receipt:
                        raise ValueError('recovered historical mask lacks its exact recovery receipt')
                    saved=check_receipt(receipt,_recovery_dependencies(deps,name,hist),run_root)
                    if not saved.get('exact_original_artifact') or saved['outputs']['mask']!=record['mask'] or record['mask']['path']!=str(target):
                        raise ValueError('recovered historical mask binding mismatch')
                    check_identity(dict(hist['mask'],path=str(target)))
        check_identity(record['matrix_receipt'])
        if record['available']:
            check_identity(record['mask'])
            content=mask_content(record['mask']['path'])
            if content!=record['content_sha256'] or record['alias_of']!=seen.get(content):
                raise ValueError('actual mask content or alias differs from publication')
            seen.setdefault(content,name)
    if set(recoveries)!=expected_recoveries:
        raise ValueError('historical recovery receipt coverage mismatch')
    if any(not records[name]['available'] for name in EXISTING+NEW):
        raise ValueError('all eight scientific endpoint masks must be available')
    for name in NEW:
        if reg['new_endpoints'][name]['no_op_with_parent']!=(records[name]['content_sha256']==records[PARENTS[name]]['content_sha256']):
            raise ValueError('new endpoint no-op status differs from actual parent pixels')
    return summary


def validate_bundle(root,deps=None):
    root=Path(root);receipt=read(root/'staging_complete.json')
    if receipt.get('status')!='succeeded' or (deps is not None and receipt['dependencies']!=deps):raise ValueError('publication staging dependency mismatch')
    for name,obj in receipt['outputs'].items():check_identity(dict(obj,path=str(root/name)))
    columns,old=read_strings(TABLE);new_columns,new=read_strings(root/'main_table.csv')
    if columns!=new_columns or len(new)!=30 or len({r['method'] for r in new})!=30:raise ValueError('30-method table schema/domain mismatch')
    preserve_old_raw(old,new,columns)
    recomputed=purity(new)
    if recomputed!=read(root/'purity_scaling.json'):raise ValueError('independent Purity recomputation differs')
    for r in new:
        for profile in ('scclassify','fixed'):
            expected=recomputed['profiles'][profile]['methods'][r['method']]
            actual=r[profile+'_Purity_F1']
            if actual!=('NA' if expected is None else str(expected)):raise ValueError('table/Purity inconsistency')
    reg=read(root/'artifact_registry.json')
    summary=validate_mask_records(reg,[r['method'] for r in new],receipt['dependencies'])
    notes=read(root/'publication_notes.json')
    if notes.get('methods')!=30 or any(key not in notes or notes[key]!=value for key,value in summary.items()):
        raise ValueError('publication notes mask verification differs from registry')
    if reg['historical28_registry']!=identity(REGISTRY) or reg['historical28_table']!=identity(TABLE) or reg['historical28_chain']!=read(REGISTRY):
        raise ValueError('historical28 identity chain changed')
    for name in NEW:
        row=next(r for r in new if r['method']==name)
        item=reg['new_endpoints'][name]
        for key in ('matrix','matrix_receipt','matrix_validation','cell_mapping','gene_mapping','evaluation_receipt','annotation','sc_scores','fixed_scores','mask','production_receipt'):
            check_identity(item[key])
        for profile,key in [('scclassify','sc_scores'),('fixed','fixed_scores')]:
            f=pd.read_csv(item[key]['path'])
            for metric in ('positive_F1','negative_F1'):
                vals=f[metric].to_numpy(float);finite=np.isfinite(vals)
                if int(row[profile+'_'+metric+'_n'])!=int(finite.sum()) or not np.isclose(float(row[profile+'_'+metric]),vals[finite].mean(),rtol=0,atol=1e-12):
                    raise ValueError('independent published F1 recomputation differs')
    check_identity(dict(path=str(TABLE),sha256=TABLE_SHA));check_identity(dict(path=str(REGISTRY),sha256=REGISTRY_SHA))
    return receipt


def publish(cfg,manual_resume,progress=None):
    require_manual_resume(cfg,manual_resume)
    # This is the only authorized publication destination, never an old table dir.
    if cfg.get('results_root',str(RESULT_ROOT))!=str(RESULT_ROOT):raise ValueError('publication destination differs from plan')
    if RESULT_ROOT.exists():
        rec=validate_bundle(RESULT_ROOT)
        if rec['dependencies']['config_digest']!=digest(cfg):raise ValueError('existing publication belongs to another config')
    else:
        stage=prepare_publication(cfg,manual_resume,progress)
        os.replace(stage,RESULT_ROOT);fsync_dir(RESULT_ROOT.parent)
    rec=validate_bundle(RESULT_ROOT)
    proof=dict(status='succeeded',dependencies=rec['dependencies'],published_root=str(RESULT_ROOT),
               outputs={n:identity(RESULT_ROOT/n) for n in ('main_table.csv','artifact_registry.json','purity_scaling.json','staging_complete.json')},
               **mask_verification_summary(read(RESULT_ROOT/'artifact_registry.json')['mask_content']),
               independent_raw_and_purity_recomputation=True,old28_unchanged=True,context_sync_complete=False)
    atomic_json(RESULT_ROOT/'publication_complete.json',proof)
    return proof
