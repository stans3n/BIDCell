"""Two-base frozen P2 transfer: native replay -> P2 -> own matrix/full evaluate."""
from pathlib import Path
import argparse
import os
import resource
import shutil
import time

import numpy as np
import pandas as pd
import tifffile
from scripts.crgf.common import atomic_tiff
from scripts.crgf.matrix import build, grid
from .common import (load_run, output_path, endpoint_spec, read, atomic_json, identity,
                     sha256, digest, check_identity, compute_lock, progress,
                     stop_requested, write_complete, validate_complete, inventory)
from .contracts import PAIRED_INPUT
from .p2_native import process, validate_process, verify_record, isolated, STAGES
from .p2_diagnostics import transfer_diagnostics

BASES = dict(P2_BASE='BASE_MATCH', P2_MORPH='V2_MORPH')


def stop_boundary(cfg, next_stage):
    """Leave classification of user/budget/resource stops to the queue owner."""
    if stop_requested(cfg):
        raise InterruptedError('P2 stopped at resumable boundary before ' + next_stage)


def copy_exact(source, dest):
    dest = isolated(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if sha256(dest) != sha256(source):
            raise ValueError('existing isolated input differs: ' + str(dest))
    else:
        temp = dest.with_name(dest.name+'.tmp.'+str(os.getpid()))
        shutil.copyfile(source,temp)
        os.replace(temp,dest)


def legacy_outputs(path):
    from .evaluation import verify_legacy_receipt
    return verify_legacy_receipt(path)


def matrix_only(machine, path, cfg):
    path = Path(path)
    if (path/'complete.json').exists():
        rec = legacy_outputs(path/'complete.json')
        for key, value in rec['dependencies'].items():
            expected = machine['final_mask'] if key == 'mask' else machine[key]
            if value['sha256'] != sha256(expected):
                raise ValueError('native replay matrix input drift: ' + key)
        if rec.get('validation_kind') != 'independent_full_native_resize_lookup_and_conservation':
            raise ValueError('native replay matrix oracle absent')
        return rec
    return build(machine,path,cache=output_path(cfg,'coordinate_cache'))


def assert_matrix_equal(actual, expected):
    actual, expected = Path(actual),Path(expected)
    for name in ('counts.npy','cell_mapping.csv','gene_mapping.csv','expr_mat.csv'):
        if sha256(actual/name) != sha256(expected/name):
            raise ValueError('disabled native replay matrix differs: ' + name)
    a,b = read(actual/'complete.json'),read(expected/'complete.json')
    if a['totals'] != b['totals'] or len(a['receipts']) != len(b['receipts']):
        raise ValueError('disabled replay molecule coverage differs')
    for x,y in zip(a['receipts'],b['receipts']):
        if (x['row_start'],x['row_stop'],x['sha256']) != (y['row_start'],y['row_stop'],y['sha256']):
            raise ValueError('disabled replay transcript owner identities differ')
    return dict(integer_counts_equal=True, cell_gene_order_equal=True,
                transcript_owners_equal=True, totals=a['totals'])


def prepare_node(config,node):
    cfg = load_run(config)
    if node not in BASES:
        raise ValueError('unknown P2 node')
    e,base = endpoint_spec(cfg,node),endpoint_spec(cfg,BASES[node])
    root = Path(e['root'])
    machine = read(base['machine'])
    original = dict(machine)
    for key, path in [('pre_mask',base['pre']),('final_mask',base['final']),
                      ('checkpoint',base['checkpoint']),('final_matrix',str(Path(base['matrix'])/'expr_mat.csv'))]:
        if Path(machine[key]).resolve() != Path(path).resolve():
            raise ValueError('historical endpoint registry/machine differs: ' + key)
    if machine['pixel_um'] != 1.0 or machine['scale_x'] != .2125 or machine['scale_y'] != .2125:
        raise ValueError('native isotropic grid changed')
    historical_pred = Path(base['pre']).parent
    post = legacy_outputs(historical_pred/'postprocess/complete.json')
    manifest = read(historical_pred/'postprocess/manifest.json')
    if manifest['dependencies']['seed'] != 1234 or manifest['dependencies']['patch_size'] != 1024:
        raise ValueError('historical native seed/tile settings differ')
    from .evaluation import check_legacy_identity
    for name,actual in [('pre',base['pre']),('nuclei',machine['nuclei_mask'])]:
        check_legacy_identity(manifest['dependencies'][name])
        if manifest['dependencies'][name]['sha256'] != sha256(actual):
            raise ValueError('historical postprocess input mismatch')
    oldmatrix = legacy_outputs(Path(base['matrix'])/'complete.json')
    if post['outputs']['final_mask']['sha256'] != sha256(base['final']) or oldmatrix['dependencies']['mask']['sha256'] != sha256(base['final']):
        raise ValueError('historical final/matrix linkage differs')
    copy_exact(base['pre'],e['pre'])
    machine.update(pre_mask=e['pre'],final_mask=e['final'],
                   final_matrix=str(Path(e['matrix'])/'expr_mat.csv'),baseline_final_mask=base['final'])
    if Path(e['machine']).exists() and read(e['machine']) != machine:
        raise ValueError('P2 machine changed after preparation')
    atomic_json(e['machine'],machine)
    dep = dict(endpoint=BASES[node],original_machine=original,
        inputs={key:identity(path) for key,path in dict(machine=base['machine'],pre=base['pre'],final=base['final'],
            checkpoint=base['checkpoint'],nuclei=original['nuclei_mask'],transcripts=original['transcripts'],
            genes=original['genes'],postprocess=historical_pred/'postprocess/complete.json',
            postprocess_manifest=historical_pred/'postprocess/manifest.json',matrix=Path(base['matrix'])/'complete.json').items()},
        settings=dict(seed=1234,tile_size=1024,rule='P2',global_grid_um=1,workers=min(4,cfg['resources']['cpu_workers'])),
        own_machine=identity(e['machine']))
    dest = root/'input_manifest.json'
    if dest.exists() and read(dest) != dep:
        raise ValueError('P2 input manifest drift')
    atomic_json(dest,dep)
    return cfg,e,base,machine


def run_node(config,node):
    cfg = load_run(config)
    if output_path(cfg,node,'complete.json').exists():
        return validate_node(cfg,node)
    stop_boundary(cfg,'preparation')
    started = time.monotonic()
    cfg,e,base,machine = prepare_node(cfg,node)
    root,pred = Path(e['root']),Path(e['final']).parent
    replay_root = root/'native_replay'
    replay_pred = replay_root/'predictions'/BASES[node]
    replay_matrix = replay_root/'matrix'/BASES[node]
    replay_machine = dict(machine,pre_mask=str(replay_pred/'pre.tif'),
        final_mask=str(replay_pred/'final.tif'),final_matrix=str(replay_matrix/'expr_mat.csv'))
    copy_exact(e['pre'],replay_machine['pre_mask'])
    atomic_json(replay_root/'configs/machine.json',replay_machine)
    progress(cfg,node,'native_disabled_replay')
    native_rec = process(replay_machine['pre_mask'],machine['nuclei_mask'],machine['nuclei_mask'],
        replay_pred,'B',workers=cfg['resources']['cpu_workers'],cfg=cfg,node=node)
    validate_process(replay_pred)
    # Never rebase attribution onto a newly obtained native result.
    if not np.array_equal(tifffile.imread(replay_pred/'final.tif'),tifffile.imread(base['final'])):
        raise ValueError('FULL native replay differs from historical final; migration refused')
    stop_boundary(cfg,'native replay matrix')
    progress(cfg,node,'native_replay_matrix')
    matrix_only(replay_machine,replay_matrix,cfg)
    equality = assert_matrix_equal(replay_matrix,base['matrix'])
    replay = dict(status='succeeded',base=BASES[node],final_pixels_equal=True,
        matrix=equality,source=identity(root/'input_manifest.json'),
        outputs=dict(postprocess=identity(replay_pred/'postprocess/complete.json'),
                     final=identity(replay_pred/'final.tif'),matrix=identity(replay_matrix/'complete.json')))
    atomic_json(replay_root/'complete.json',replay)
    stop_boundary(cfg,'P2 postprocess')
    progress(cfg,node,'P2_postprocess')
    p2 = process(e['pre'],machine['nuclei_mask'],machine['nuclei_mask'],pred,'P2',
        native_pred=replay_pred,workers=cfg['resources']['cpu_workers'],cfg=cfg,node=node)
    validate_process(pred)
    stop_boundary(cfg,'own matrix and full evaluation')
    progress(cfg,node,'own_matrix_and_full_evaluation')
    from .evaluation import build_and_evaluate
    evaluation = build_and_evaluate(cfg,node)
    stop_boundary(cfg,'stage transfer diagnostics')
    progress(cfg,node,'stage_transfer_diagnostics')
    diagnostics = transfer_diagnostics(cfg,node,pred,replay_pred,machine,e['matrix'])
    stop_boundary(cfg,'complete scientific comparison')
    from .reports import compare_endpoints,factorial_report
    comparison = compare_endpoints(cfg,node,BASES[node],out=root/'comparison')
    atomic_json(root/'comparison.json',comparison)
    stop_boundary(cfg,'factorial comparison or completion')
    if node == 'P2_MORPH':
        validate_node(cfg,'P2_BASE')
        factorial_report(cfg,out=root/'factorial')
    stop_boundary(cfg,'node completion')
    tile_recs = [read(p['path']) for p in read(pred/'postprocess/manifest.json')['tiles']]
    cost = dict(wall_seconds=time.monotonic()-started,native_replay_postprocess_seconds=native_rec['wall_seconds'],
        P2_postprocess_seconds=p2['wall_seconds'],full_native_replay_included=True,
        max_K=max(r['diagnostics']['candidate_count'] for r in tile_recs),
        max_query_pairs=max(r['diagnostics'].get('max_query_pairs',0) for r in tile_recs),
        peak_parent_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        peak_child_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024,
        resumed_cost_note='postprocess receipts retain original completed-stage cost; wall is this invocation')
    atomic_json(root/'runtime_cost.json',cost)
    write_complete(cfg,node,comparison['status'],native_replay=identity(replay_root/'complete.json'),
                   evaluation=evaluation,transfer=diagnostics)
    return validate_node(cfg,node)


def validate_node(config,node):
    cfg = load_run(config)
    if node not in BASES:
        raise ValueError(node)
    rec = validate_complete(cfg,node)
    e,base = endpoint_spec(cfg,node),endpoint_spec(cfg,BASES[node])
    root = Path(e['root'])
    manifest = read(root/'input_manifest.json')
    for obj in manifest['inputs'].values():
        check_identity(obj)
    replay = legacy_outputs(root/'native_replay/complete.json')
    if replay['base'] != BASES[node] or not replay.get('final_pixels_equal'):
        raise ValueError('native replay gate absent')
    pred = Path(e['final']).parent
    replay_pred = root/'native_replay/predictions'/BASES[node]
    validate_process(pred)
    validate_process(replay_pred)
    if not np.array_equal(tifffile.imread(replay_pred/'final.tif'),tifffile.imread(base['final'])):
        raise ValueError('native replay identity lost')
    assert_matrix_equal(root/'native_replay/matrix'/BASES[node],base['matrix'])
    from .evaluation import validate_endpoint
    validate_endpoint(cfg,node)
    diagnostic = read(root/'transfer_diagnostics.json')
    if not diagnostic['postconnect_support_equal'] or not diagnostic['postconnect_changes_confined_to_extranuclear_conflicts']:
        raise ValueError('P2 intervention invariants failed')
    from .reports import validate_comparison
    validate_comparison(cfg,root/'comparison')
    return rec


def validate_mini(config,output):
    """Real historical inputs, both bases, whole-chain mini and recovery oracle.

    Mini is a development ROI with preserved original IDs and exact coordinate
    lookup. Full-slide disabled replay remains mandatory at each formal node.
    """
    from .p2_mini import validate_mini as implementation
    return implementation(config,output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=['run','validate','mini'])
    parser.add_argument('--config',required=True)
    parser.add_argument('--node',choices=list(BASES))
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.action == 'mini':
        result = validate_mini(args.config,args.output)
    elif args.action == 'run':
        result = run_node(args.config,args.node)
    else:
        result = validate_node(args.config,args.node)
    print(__import__('json').dumps(result,indent=2))
