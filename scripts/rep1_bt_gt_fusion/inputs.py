"""Explicit six-endpoint adapters and immutable input/source discovery.

This module never runs a classifier or derives scientific measurements. Historical
receipts are checked as historical bytes; their producer hashes are not rewritten.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from . import EXISTING, NEW, N_CELLS, N_GENES, ELIGIBLE, comparisons
from .common import (read, identity, check_identity, digest, atomic_json, progress,
                     within, sha256)

WORK = Path('/home/zwan0726/work')
OLD = WORK / 'outputs/bidcell_boundary_three_20261015/boundary-three-20261015-r02'
REGISTRY = WORK / 'outputs/bidcell_historical_dual_boundary_three_20261016/artifact_registry.json'
TABLE = REGISTRY.with_name('main_table.csv')
FIXED = WORK / 'outputs/bidcell_new_methods_20261013/configs/evaluation_population.csv'
REGISTRY_SHA = 'c14cdf60a098a04f12f43cb1c973ee0c3a94d460739c4795f100984f45b283bb'
TABLE_SHA = 'ab4f91d8026d1c78306cf240b851b5b978a82595924db72e75a3b42fb1225cfe'
FIXED_SHA = '6c3505d5b0a70d3977bd88d4cda61281ecec3fa924fdf4cfa63cf2b8720abd0b'


class Catalog:
    def __init__(self):
        self.files = {}

    def add(self, path, expected=None):
        p = Path(path).resolve(strict=True)
        key = str(p)
        if key not in self.files:
            self.files[key] = identity(p)
        actual = self.files[key]
        if expected is not None:
            if actual['sha256'] != expected['sha256'] or ('bytes' in expected and actual['bytes'] != expected['bytes']):
                raise ValueError('historical receipt identity mismatch: ' + key)
        return key

    def receipt(self, path, expected=None, outputs=True):
        path = self.add(path, expected)
        rec = read(path)
        if rec.get('status') not in ('succeeded', 'passed', 'completed') and rec.get('execution_status') != 'COMPLETED':
            raise ValueError('historical receipt incomplete: ' + path)
        if outputs:
            if not rec.get('outputs'):
                raise ValueError('historical receipt has no output inventory: ' + path)
            for item in rec['outputs'].values():
                if isinstance(item, dict) and 'path' in item and 'sha256' in item:
                    self.add(item['path'], item)
        return rec


def discover_adapters(cat):
    """B0 deliberately follows a different chain from the five registry entries."""
    cat.add(REGISTRY, dict(sha256=REGISTRY_SHA))
    cat.add(TABLE, dict(sha256=TABLE_SHA))
    reg = read(REGISTRY)
    result = {}
    for name in EXISTING:
        if name == 'B0':
            ent = reg['B0_reference']
            inp = read(cat.add(ent['input']['path'], ent['input']))
            if inp.get('method') != 'B0':
                raise ValueError('B0 adapter points to another historical method')
            evaluation = Path(inp['source_evaluation'])
            if evaluation != WORK / 'outputs/bidcell_new_methods_20261013/baselines/B0/evaluation':
                raise ValueError('unexpected B0 source evaluation')
            er = cat.receipt(evaluation / 'complete.json')
            matrix = Path(er['dependencies']['matrix']['path']).parent
            if matrix != WORK / 'outputs/remote_prep_20261007T122955/matrix_final':
                raise ValueError('B0 is not the native full B0 matrix')
            if Path(inp['native_csv']).resolve() != matrix / 'expr_mat.csv':
                raise ValueError('B0 native CSV chain mismatch')
            sc = cat.add(ent['sc_scores']['path'], ent['sc_scores'])
            fixed = cat.add(ent['fixed_scores']['path'], ent['fixed_scores'])
            scoring = read(cat.add(ent['scoring_receipt']['path'], ent['scoring_receipt']))
            if scoring.get('status') != 'verified_existing_full_evaluation' or Path(scoring['source']['path']).resolve() != evaluation / 'complete.json':
                raise ValueError('B0 historical scoring reuse chain mismatch')
            cat.add(scoring['source']['path'], scoring['source'])
            expected_metrics = ent['raw_metric_check']
        else:
            ent = reg['new_and_control_endpoints'][name]
            evaluation = Path(ent['evaluation_receipt']['path']).parent
            er = cat.receipt(ent['evaluation_receipt']['path'], ent['evaluation_receipt'])
            matrix = Path(ent['matrix_receipt']['path']).parent
            sc = cat.add(ent['sc_scores']['path'], ent['sc_scores'])
            fixed = cat.add(ent['fixed_scores']['path'], ent['fixed_scores'])
            expected_metrics = ent['metrics']
            for key in ('matrix', 'matrix_receipt', 'cell_mapping', 'gene_mapping', 'annotation'):
                cat.add(ent[key]['path'], ent[key])
        if not er.get('paper_method_complete'):
            raise ValueError('old endpoint lacks complete evaluation: ' + name)
        machine_path = cat.add(er['dependencies']['machine']['path'], er['dependencies']['machine'])
        machine = read(machine_path)
        mr = cat.receipt(matrix / 'complete.json', er['dependencies']['matrix'])
        if mr.get('validation_kind') != 'independent_full_native_resize_lookup_and_conservation':
            raise ValueError('old matrix lacks full independent native acceptance: ' + name)
        cat.add(mr['validation']['path'], mr['validation'])
        if mr['totals']['eligible'] != ELIGIBLE:
            raise ValueError('old eligible molecule denominator drift: ' + name)
        for key, item in mr['dependencies'].items():
            cat.add(item['path'], item)
        if sha256(machine['final_mask']) != mr['dependencies']['mask']['sha256']:
            raise ValueError('machine/matrix final mask chain mismatch: ' + name)
        if Path(machine['final_matrix']).resolve() != matrix / 'expr_mat.csv':
            raise ValueError('machine is not bound to its own matrix: ' + name)
        for key in ('nuclei_mask', 'preannotation', 'baseline_config', 'checkpoint', 'reference', 'positive_markers', 'negative_markers'):
            cat.add(machine[key])
        needed = ['summary.json', 'paper_aligned_v1/annotation.csv', 'paper_aligned_v1/qc_all_cells.csv',
                  'paper_aligned_v1/own_cell_centroids.csv', 'engineering/structure/annotation_nuclear.csv',
                  'paper_aligned_v1/neighbour_curves.csv']
        for rel in needed:
            cat.add(evaluation / rel)
        # Parent owners are immutable input to local transfer and conservation checks.
        if name in ('P2_BASE', 'V2_MORPH'):
            for rec in mr['receipts']:
                cat.add(matrix / rec['owner_file'], dict(sha256=rec['sha256']))
        result[name] = dict(name=name, machine=machine_path, matrix=str(matrix), evaluation=str(evaluation),
                            sc_scores=sc, fixed_scores=fixed, expected_metrics=expected_metrics,
                            final=machine['final_mask'], nuclei=machine['nuclei_mask'],
                            assigned=mr['totals']['assigned'])
    return result


def source_dependencies():
    """Executable bytes required in the source snapshot, including external R.

    Historical protocol JSON remains data. Its executable path fields must be
    remapped to these snapshot copies by the new evaluation adapter.
    """
    root = Path(__file__).resolve().parents[1]
    paths = [p for p in (root / 'rep1_bt_gt_fusion').rglob('*') if p.is_file() and p.suffix in ('.py', '.R')]
    relative = ['__init__.py', 'boundary_three/__init__.py', 'boundary_three/core_refine.py', 'boundary_three/core_scores.py',
                'boundary_three/common.py', 'boundary_three/contracts.py', 'boundary_three/settings.py',
                'crgf/__init__.py', 'crgf/matrix.py', 'crgf/common.py',
                'new_methods/__init__.py', 'new_methods/author_marker.py']
    paths += [root / p for p in relative]
    external = [WORK / 'outputs/bidcell_metric_protocol_review_20261008/publish_expression_metrics.R',
                WORK / 'outputs/bidcell_metric_protocol_review_20261008/publish_processing.R',
                WORK / 'outputs/bidcell_metric_protocol_review_20261008/publish_utils.R',
                WORK / 'outputs/bidcell_historical_dual_20261013/sources/publish_spatialDiversity_metrics.R']
    entries = []
    repo = root.parent
    for p in paths + external:
        p = p.resolve(strict=True)
        target = str(p.relative_to(repo)) if repo in p.parents else 'external_sources/' + p.name
        entries.append(dict(**identity(p), snapshot_relative=target))
    return sorted(entries, key=lambda x: x['snapshot_relative'])


def freeze_inputs(output_root, progress=None):
    """Hash and bind read-only inputs; no B/T, GT, matrix, or evaluation is run."""
    root = Path(output_root).resolve()
    out = within(root, 'configs', 'inputs_manifest.json')
    if out.exists():
        return load_manifest(out, verify=True)
    cat = Catalog()
    endpoints = discover_adapters(cat)
    # The publication phase preserves 28 raw rows and counts unique mask content.
    # Freeze those masks through their own historical matrix receipts now. The
    # old C/SP count CSVs were restored after their final masks were removed.
    # Missing publication-only bytes must not block the six live scientific
    # endpoints, and must never be replaced by a historical unique-mask claim.
    historical_masks = {}
    historical_root = WORK / 'outputs/bidcell_historical_dual_20261013'
    historical_table = pd.read_csv(TABLE, dtype=str, keep_default_na=False)
    if len(historical_table) != 28 or historical_table.method.duplicated().any():
        raise ValueError('historical publication table must contain 28 unique methods')
    registry = read(REGISTRY)
    for key in ('historical20_registry', 'historical20_table'):
        if key in registry:
            cat.add(registry[key]['path'], registry[key])
    for row in historical_table.to_dict('records'):
        if row['matrix_receipt_path']:
            receipt_path = row['matrix_receipt_path']
        elif row['matrix_path']:
            receipt_path = str(Path(row['matrix_path']) / 'complete.json')
        else:
            ip = cat.add(historical_root / 'methods' / row['method'] / 'input.json')
            historical_input = read(ip)
            receipt_path = str(Path(historical_input['native_csv']).parent / 'complete.json')
            if row['method'] in ('C','SP'):
                # These two historical CSVs were restored without a copied
                # complete.json. Their original source evaluation still binds
                # the original full-matrix receipt and exact mask identity.
                source_eval = Path(historical_input['annotation']).parents[1] / 'complete.json'
                source_receipt = cat.receipt(source_eval, outputs=False)
                original = source_receipt['dependencies']['matrix']
                receipt_path = cat.add(original['path'], original)
        mr = cat.receipt(receipt_path, outputs=False)
        mask_identity = mr['dependencies']['mask']
        available = Path(mask_identity['path']).is_file()
        if available:
            cat.add(mask_identity['path'], mask_identity)
            frozen_mask = cat.files[str(Path(mask_identity['path']).resolve())]
        else:
            frozen_mask = mask_identity
        historical_masks[row['method']] = dict(mask=frozen_mask, available=available,
                                              unavailable_reason='' if available else 'original_final_mask_missing; publication_unique_content_verification_blocked',
                                              recovery_relative=None if available else 'recovered_historical_masks/'+row['method']+'/final.tif',
                                              matrix_receipt=cat.files[str(Path(receipt_path).resolve())])
    cat.add(FIXED, dict(sha256=FIXED_SHA))
    old_cfg = read(cat.add(OLD / 'configs/run.json'))
    metadata = cat.add(old_cfg['inputs']['nucleus_metadata'])
    region = cat.add(old_cfg['inputs']['reference_region'])
    queue_path = cat.add(old_cfg['inputs']['evaluation_queue'])
    authority = cat.add(old_cfg['inputs']['authoritative_protocol'])
    queue = read(queue_path)
    prep_path = Path(queue['prepared']) / 'complete.json'
    prep = cat.receipt(prep_path)
    for key in ('reference', 'official_processing', 'official_expression', 'rscript', 'protocol'):
        cat.add(prep['dependencies'][key]['path'], prep['dependencies'][key])
    model = cat.add(prep['model']['path'], prep['model'])
    auth = read(authority)
    if auth['reference_model']['sha256'] != prep['model']['sha256'] or auth['fixed_population']['sha256'] != FIXED_SHA:
        raise ValueError('reference model/fixed-panel authority mismatch')
    protocol_path = prep['dependencies']['protocol']['path']
    protocol = read(protocol_path)
    if {k: protocol[k] for k in auth['core']} != auth['core']:
        raise ValueError('authoritative scientific core mismatch')
    a1 = {}
    a1_receipt = cat.receipt(OLD / 'A1_CORE/complete.json', outputs=False)
    for key in ('core_counts.npz', 'cell_coverage.csv', 'decision.json', 'input_manifest.json', 'type_scope.json'):
        a1[key] = cat.add(OLD / 'A1_CORE' / key, a1_receipt['outputs'][key])
    a1['complete.json'] = str(OLD / 'A1_CORE/complete.json')
    # Only this explicitly absolute, independently validated cache is reused.
    matrix = read(Path(endpoints['BASE_MATCH']['matrix']) / 'complete.json')
    cache = Path(matrix['coordinate_cache'])
    if not cache.is_absolute():
        raise ValueError('BASE_MATCH coordinate cache must be absolute')
    cm = read(cat.add(cache / 'complete.json'))
    mc = read(endpoints['BASE_MATCH']['machine'])
    cdeps = cm['dependencies']
    expected = dict(transcripts=matrix['dependencies']['transcripts']['sha256'],
                    genes=matrix['dependencies']['genes']['sha256'], scale_x=0.2125, scale_y=0.2125)
    if any(cdeps[k] != v for k, v in expected.items()) or cm.get('duplicate_ids') != 0:
        raise ValueError('historical coordinate cache key/uniqueness mismatch')
    if len(cm['chunks']) != len(matrix['receipts']):
        raise ValueError('cache and matrix chunk count differ')
    coordinates = []
    cursor = 0
    for i, (cc, rec) in enumerate(zip(cm['chunks'], matrix['receipts'])):
        path = cat.add(cache / cc['file'], dict(sha256=cc['sha256']))
        if Path(rec['coordinate_file']).resolve() != Path(path) or rec['coordinate_sha256'] != cc['sha256']:
            raise ValueError('coordinate cache/matrix receipt association mismatch')
        if rec['row_start'] != cursor or rec['rows'] != cc['rows'] or rec['row_stop'] != cursor + cc['rows']:
            raise ValueError('coordinate chunk coverage gap/overlap')
        cursor += cc['rows']
        coordinates.append(dict(path=path, sha256=cc['sha256'], rows=cc['rows'], row_start=rec['row_start'], row_stop=cursor))
    if cursor != ELIGIBLE:
        raise ValueError('coordinate eligible denominator mismatch')
    for ep in endpoints.values():
        mach = read(ep['machine'])
        if any(mach[k] != mc[k] for k in ('scale_x', 'scale_y', 'pixel_um')):
            raise ValueError('coordinate scale differs between endpoints')
        mr = read(Path(ep['matrix']) / 'complete.json')
        if any(mr['dependencies'][k]['sha256'] != matrix['dependencies'][k]['sha256'] for k in ('transcripts', 'genes')):
            raise ValueError('methods do not share transcript/gene universe')
    for source in source_dependencies():
        if source['snapshot_relative'].startswith('external_sources/'):
            cat.add(source['path'], source)
    author_root = WORK / 'outputs/bidcell_isbi_decision_20261016/official_reference'
    author_reference = {name:cat.add(author_root / name) for name in
                        ('author_Fig3e.csv','author_SuppFig14.csv','author_supplement_page17.png','reference_audit.json')}
    obj = dict(schema='rep1-bt-gt-inputs-v1', endpoints=endpoints, fixed_population=str(FIXED),
               metadata=metadata, reference_region=region, a1=a1, coordinates=coordinates,
               coordinate_cache=str(cache / 'complete.json'), expected_cells=N_CELLS,
               expected_genes=N_GENES, expected_molecules=ELIGIBLE,
               evaluation=dict(queue=queue_path, authority=authority, prepared=str(prep_path.parent),
                               protocol=protocol_path, reference_model=model),
               comparisons=[list(x) for x in comparisons()], existing_order=list(EXISTING), new_order=list(NEW),
               historical_masks=historical_masks,
               publication_input_status=dict(complete=all(x['available'] for x in historical_masks.values()),
                    missing_masks=sorted(k for k,v in historical_masks.items() if not v['available']),
                    scope='publication_only; no missing scientific production inputs accepted',
                    recovery='only exact original expected bytes/SHA at frozen run-relative recovery path with a bound recovery receipt; no archive located during preparation'),
               author_reference=author_reference,
               files=cat.files)
    atomic_json(out, obj)
    # A frozen, checked catalog is concrete preparation progress, not science progress.
    if progress is not None:
        progress(dict(stage='input_freeze', cursor=len(cat.files), total=len(cat.files), detail={'manifest':str(out)}))
    return obj


def load_manifest(path, verify=True):
    obj = read(path)
    if obj.get('schema') != 'rep1-bt-gt-inputs-v1' or obj.get('existing_order') != list(EXISTING) or obj.get('new_order') != list(NEW):
        raise ValueError('input manifest schema/scope mismatch')
    if obj['comparisons'] != [list(x) for x in comparisons()]:
        raise ValueError('comparison set drift')
    if set(obj.get('endpoints',{})) != set(EXISTING) or len(obj.get('historical_masks',{})) != 28:
        raise ValueError('input manifest endpoint/publication-mask coverage mismatch')
    if (obj.get('expected_cells'),obj.get('expected_genes'),obj.get('expected_molecules')) != (N_CELLS,N_GENES,ELIGIBLE):
        raise ValueError('input manifest scientific universe changed')
    if verify:
        for item in obj['files'].values():
            check_identity(item)
    return obj


def load_inputs(cfg, verify=True):
    obj = load_manifest(cfg['input_manifest'], verify=verify)
    if digest(obj) != cfg['input_digest']:
        raise ValueError('config/input manifest digest mismatch')
    return obj


def matrix_arrays(spec):
    root = Path(spec['matrix'])
    cm = pd.read_csv(root / 'cell_mapping.csv')
    gm = pd.read_csv(root / 'gene_mapping.csv')
    if cm.canonical_cell_id.duplicated().any() or gm.gene.duplicated().any():
        raise ValueError('duplicate matrix IDs/genes')
    ids = cm.canonical_cell_id.to_numpy(np.int64)
    if not np.array_equal(cm.matrix_row, np.arange(len(ids))) or not np.array_equal(cm.mask_label, ids):
        raise ValueError('cell mapping is not an exact one-to-one row mapping')
    if not np.array_equal(gm.gene_column, np.arange(len(gm))):
        raise ValueError('gene mapping order invalid')
    counts = np.load(root / 'counts.npy', mmap_mode='r')
    if counts.shape != (len(ids), len(gm)) or not np.issubdtype(counts.dtype, np.integer) or (counts < 0).any():
        raise ValueError('matrix shape/type/count violation')
    return ids, gm.gene.tolist(), counts
