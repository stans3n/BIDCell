"""Science stage API for the separately audited supervisor; no background runner."""
from __future__ import annotations
from pathlib import Path
import os
import numpy as np
import tifffile
from . import EXISTING, NEW, PARENTS, ASSIGNED, NODE_ORDER, N_CELLS, N_GENES
from .common import (atomic_json, check_identity, check_receipt, dependencies, identity,
                     progress as emit_progress, read, within, write_receipt)
from .inputs import freeze_inputs, load_inputs, matrix_arrays


def config_check(cfg):
    required=('run_id','run_root','source_digest','input_digest','input_manifest','plan_sha256','source_snapshot')
    if any(not cfg.get(k) for k in required):raise ValueError('science config lacks required frozen identity fields')
    if cfg.get('node_order') != list(NODE_ORDER):raise ValueError('science node order/scope drift')
    if not Path(cfg['run_root']).is_absolute():raise ValueError('science run_root must be absolute')
    within(cfg['run_root'],cfg['input_manifest'])
    within(cfg['run_root'],cfg['source_snapshot'])
    return cfg


def receipt_path(cfg,node):
    if node=='BT_EXISTING6':return within(cfg['run_root'],'bt_existing6','complete.json')
    if node=='PRODUCTION_VALIDATE':return within(cfg['run_root'],'production_complete.json')
    if node in (n+'_PRODUCE' for n in NEW):return within(cfg['run_root'],'endpoints',node.removesuffix('_PRODUCE'),'production_complete.json')
    raise ValueError('unsupported production node: '+str(node))


def validate_stage(cfg,node,manifest=None):
    """Read-only identity and semantic acceptance; never runs evaluation/publication."""
    config_check(cfg)
    manifest=manifest or load_inputs(cfg)
    rec=check_receipt(receipt_path(cfg,node),dependencies(cfg,node),cfg['run_root'])
    if node=='BT_EXISTING6':
        if rec.get('methods')!=list(EXISTING):raise ValueError('BT6 completion method coverage mismatch')
        pop=read(within(cfg['run_root'],'bt_existing6/populations/complete.json'))
        if pop.get('fixed_population_n')!=103000 or set(pop.get('populations',{}))!=set(EXISTING):
            raise ValueError('BT6 population receipt semantic mismatch')
        for name in EXISTING:
            row=pop['populations'][name]
            if row['full_ids_n']!=N_CELLS or row['missing_matrix_rows'] or row['missing_coordinates']:
                raise ValueError('BT6 common nuclear namespace invariant failed')
        return rec
    if node=='PRODUCTION_VALIDATE':
        if rec.get('new_endpoints')!=list(NEW) or not rec.get('production_only'):
            raise ValueError('overall production receipt semantic mismatch')
        validate_stage(cfg,'BT_EXISTING6',manifest)
        for name in NEW:validate_stage(cfg,name+'_PRODUCE',manifest)
        return rec
    name=node.removesuffix('_PRODUCE');parent=manifest['endpoints'][PARENTS[name]]
    root=within(cfg['run_root'],'endpoints',name)
    if rec.get('endpoint')!=name or rec.get('parent')!=PARENTS[name] or not rec.get('production_only') or rec.get('evaluation_complete') is not False:
        raise ValueError('endpoint production receipt phase/parent mismatch')
    gt=check_receipt(root/'gt/complete.json',dependencies(cfg,name+'_GT'),cfg['run_root'])
    for field in ('foreground_equal','nuclear_owner_exact','all_ids_equal','positive_to_positive_only',
                  'donor_components_do_not_increase','nuclear_connections_preserved','new_recipient_pixels_nucleus_connected'):
        if gt['invariants'].get(field) is not True:raise ValueError('GT invariant missing: '+field)
    child=check_receipt(root/'matrix/complete.json',dependencies(cfg,name+'_MATRIX'),cfg['run_root'])
    if child.get('validation_kind')!='independent_full_native_resize_lookup_and_conservation':raise ValueError('GT matrix lacks full oracle')
    check_identity(child['validation'])
    from .matrix_stage import check_parent_conservation
    check_parent_conservation(parent,child,root/'matrix',ASSIGNED[name])
    mask=tifffile.imread(root/'final.tif');base=tifffile.imread(parent['final']);nuc=tifffile.imread(parent['nuclei'])
    if not np.array_equal(mask>0,base>0) or not np.array_equal(mask[nuc>0],nuc[nuc>0]):raise ValueError('completed GT mask foreground/nuclear semantics failed')
    ids,genes,_=matrix_arrays(dict(matrix=str(root/'matrix')))
    actual=np.unique(mask);actual=actual[actual>0]
    if not np.array_equal(ids,actual) or len(ids)!=N_CELLS or len(genes)!=N_GENES:raise ValueError('completed GT matrix/mask ID/gene semantics failed')
    return rec


def run_stage(config,node,progress=None,lock_fds=(),validate_execution=None):
    """BT6 -> first production -> second production -> validation. No R tonight."""
    cfg=read(config) if isinstance(config,(str,Path)) else dict(config)
    config_check(cfg)
    if node not in NODE_ORDER:raise ValueError('unsupported production stage')
    if not lock_fds or validate_execution is None:
        raise PermissionError('formal science stages require inherited lock descriptors and execution/snapshot validator')
    for fd in lock_fds:os.fstat(fd)
    validate_execution()
    manifest=load_inputs(cfg)
    pos=NODE_ORDER.index(node)
    for previous in NODE_ORDER[:pos]:validate_stage(cfg,previous,manifest)
    if receipt_path(cfg,node).exists():return validate_stage(cfg,node,manifest)
    if node=='BT_EXISTING6':
        from .bt_curves import run_existing6
        result=run_existing6(cfg,manifest,progress)
    elif node.endswith('_PRODUCE'):
        name=node.removesuffix('_PRODUCE')
        from .gt_refine import refine
        from .matrix_stage import produce_matrix
        root=within(cfg['run_root'],'endpoints',name)
        refine(cfg,manifest,name,progress)
        matrix,conservation=produce_matrix(cfg,manifest,name,progress)
        result=write_receipt(root/'production_complete.json',dependencies(cfg,node),
                  {p:identity(root/p) for p in ('gt/complete.json','matrix/complete.json','machine.json','final.tif','matrix_conservation.json')},
                  endpoint=name,parent=PARENTS[name],production_only=True,evaluation_complete=False,conservation=conservation)
    else:
        # Admission permits old B/T plus new production only. A rogue evaluator's
        # directory is evidence of an invalid overnight phase, not silently accepted.
        if any(within(cfg['run_root'],'endpoints',m,'evaluation').exists() for m in NEW):
            raise ValueError('new endpoint evaluation appeared during production-only phase')
        result=write_receipt(receipt_path(cfg,node),dependencies(cfg,node),
                 {m:identity(receipt_path(cfg,m)) for m in NODE_ORDER[:-1]},production_only=True,
                 new_endpoints=list(NEW),existing_B_T=list(EXISTING),manual_resume_required=True,
                 intended_terminal_state='PRODUCTION_COMPLETE_AWAITING_MANUAL_RESUME')
    validate_stage(cfg,node,manifest)
    emit_progress(progress,node+'_validated',1,1,receipt=str(receipt_path(cfg,node)))
    return result
