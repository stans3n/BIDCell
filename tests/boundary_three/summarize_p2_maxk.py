"""Validate maximum-K artifacts and record explicit resource-estimate semantics."""
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

ROOT = Path('/home/zwan0726/work/outputs/bidcell_boundary_three_20261015/mini/p2')
REPO = Path(__file__).resolve().parents[2]
GIB = 1024 ** 3


def read(path):
    return json.loads(Path(path).read_text())


def identity(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=h.hexdigest())


def check(obj):
    if identity(obj['path']) != obj:
        raise ValueError('artifact identity changed: ' + obj['path'])


def total(path):
    return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file())


def run():
    receipt = ROOT/'maxK_5201/complete.json'
    record = read(receipt)
    if record['status'] != 'passed' or not record['both_maximum_tiles_executed']:
        raise ValueError('both maximum-K tiles must finish')
    for obj in record['outputs'].values():
        check(obj)
    for obj in record['source'].values():
        check(obj)
    check(record['prior_full_chain_receipt'])
    observer_path = ROOT/'maxK_5201_process_memory.json'
    observer = read(observer_path)
    if observer['status'] != 'finished':
        raise ValueError('resource observer has not finished')
    if observer['source_sha256'] != identity(REPO/'tests/boundary_three/observe_p2_memory.py')['sha256']:
        raise ValueError('resource observer source changed')
    log_path = ROOT/'maxK_5201_execution.log'
    lines = log_path.read_text().splitlines()
    begin = next(s.removeprefix('maximum-K compute lock acquired ') for s in lines
                 if s.startswith('maximum-K compute lock acquired '))
    end = next(s.removeprefix('both maximum-K tests complete ') for s in lines
               if s.startswith('both maximum-K tests complete '))
    locked_wall = (datetime.fromisoformat(end) - datetime.fromisoformat(begin)).total_seconds()
    cfg = read('/home/zwan0726/work/outputs/bidcell_boundary_three_20261015/draft_run.json')
    old = read(ROOT/'resource_estimate.json')
    reports_root = Path('/home/zwan0726/work/outputs/bidcell_v2_paired_20261014/results')
    report_bytes = total(reports_root)
    workers = []
    nodes = {}
    for base, node, retained in [('BASE_MATCH', 'P2_BASE', 14), ('V2_MORPH', 'P2_MORPH', 16)]:
        check(record['bases'][base])
        rec = read(record['bases'][base]['path'])
        if rec['status'] != 'passed' or rec['maximum_K_executed'] != 5201:
            raise ValueError('maximum-K worker did not pass')
        for obj in rec['dependencies']['inputs'].values():
            check(obj)
        for obj in rec['dependencies']['source'].values():
            check(obj)
        sample = observer['bases'][base]
        workers.append(max(rec['peak_worker_RSS_bytes'], sample['peak_sampled_VmHWM_bytes']))
        phases = rec['phases']
        cost = record['cost_estimates'][base]
        components = dict(old[node]['components_seconds'])
        components['native_disabled_replay'] = cost['native_full_seconds']
        components['P2_postprocess'] = cost['P2_full_seconds']
        if not math.isclose(sum(components.values()), cost['full_node_seconds'], abs_tol=1e-6):
            raise ValueError('complete chain component sum differs')
        e = cfg['endpoints'][base]
        historical_matrix = total(e['matrix'])
        historical_evaluation = total(e['evaluation'])
        raster_bytes = Path(e['pre']).stat().st_size
        tile_bytes = (ROOT/'maxK_5201'/base/'P2_postconnect.tif').stat().st_size
        disk_components = dict(
            twelve_full_rasters=12 * raster_bytes,
            two_runs_two_tile_rasters_each=48 * 4 * tile_bytes,
            two_matrices_with_10percent_size_allowance=math.ceil(2.2 * historical_matrix),
            own_evaluation_twice_historical_size=2 * historical_evaluation,
            reports_allowance=(4 if node == 'P2_MORPH' else 2) * report_bytes,
            diagnostics_splits_receipts_unmeasured_allowance=2 * GIB)
        nodes[node] = dict(
            complete_chain_seconds=cost['full_node_seconds'], seconds=cost['full_node_seconds'],
            components_seconds=components, estimate_basis=cost['basis'],
            queue_1p25_contingency_included=False, package_1800second_reserve_included=False,
            retained_bytes=retained * GIB, new_disk_bytes=retained * GIB,
            peak_ram_bytes=20 * GIB, RAM_reservation_kind='analytical whole-node reservation, not a measured peak',
            measured_maxK_wall_seconds=rec['wall_seconds'], measured_phase_seconds=phases,
            measured_compute_worker_HWM_bytes=rec['peak_worker_RSS_bytes'],
            sampled_worker_HWM_bytes=sample['peak_sampled_VmHWM_bytes'],
            artifact_disk_forecast=dict(
                bytes=sum(disk_components.values()), components_bytes=disk_components,
                historical_matrix_bytes=historical_matrix, historical_evaluation_bytes=historical_evaluation,
                prior_three_endpoint_reports_bytes=report_bytes, prior_reports_root=str(reports_root),
                semantics='artifact-derived forecast with explicit allowances; full-slide conflict CSV and split sizes remain unmeasured. This forecast does not replace the conservative retained-byte reservation. Shared coordinate cache and atomic-write/disk reserve are accounted separately by the queue.'),
            maximum_tile_receipt=record['bases'][base])
    peak = max(workers)
    result = dict(status='passed', nodes=nodes, maximum_K=5201,
        maximum_tile_receipt=identity(receipt), memory_observer_receipt=identity(observer_path),
        locked_run_wall_from_UTC_log_seconds=locked_wall, execution_log=identity(log_path),
        wall_semantics='locked-run elapsed includes both fresh-worker startups, input validation, computation and output writing; excludes lock wait. Per-base receipt wall begins after input loading/identity checks.',
        peak_worker_HWM_measured_or_sampled_bytes=peak,
        sampled_peak_concurrent_driver_and_worker_RSS_bytes=observer['peak_sampled_driver_and_worker_RSS_bytes'],
        concurrent_sample_semantics='maximum sampled sum of live RSS, not the sum of separate process HWM values and not an exact unsampled process-tree peak',
        four_worker_plus_parent_analytical_bytes=5 * peak,
        four_worker_estimate_semantics='five copies of the largest observed single-worker HWM (four workers plus one parent-sized allowance), conservatively ignoring fork sharing; not a full-slide measured peak',
        whole_node_analytical_RAM_bytes=20 * GIB,
        RAM_allocation=dict(four_workers=8 * GIB, parent_and_stage_buffers=4 * GIB, evaluator_or_R_child=8 * GIB),
        RAM_assumptions='2 GiB per worker rounds observed ~1.12 GiB upward. Allocation includes an additional 8 GiB evaluator/R child even though postprocess workers finish before evaluation/reporting. Runtime resource checks remain required.',
        separate_available_RAM_reserve_bytes=6 * GIB,
        formal_experiment_started=False, independent_code_audit=False,
        prior_full_chain_receipt=record['prior_full_chain_receipt'],
        scientific_source=record['source'], summarizer=identity(__file__))
    (ROOT/'resource_estimate_maxk.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: dict(seconds=v['seconds'], retained_GiB=v['retained_bytes']/GIB,
         artifact_forecast_GiB=v['artifact_disk_forecast']['bytes']/GIB) for k,v in nodes.items()}, indent=2))


if __name__ == '__main__':
    run()
