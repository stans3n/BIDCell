"""Targeted real maximum-K resource admission; no formal endpoint/scoring.

Both pre rasters' actual maximum is 5,201 candidates at (4096,0). Execute each
base in a fresh worker, serially under the common lock. Reuse no numerical
outputs from the earlier representative-tile mini and preserve all its receipts.
"""
import argparse
import ast
import os
from pathlib import Path
import random
import resource
import subprocess
import sys
import time

import numpy as np
import tifffile
from bidcell.model import postprocess_predictions as native
from scripts.crgf.common import atomic_tiff
from scripts.boundary_three.common import (atomic_json,check_identity,compute_lock,
    digest,identity,inventory,read,sha256,utc)
from scripts.boundary_three.contracts import SOURCE_ROOT
from scripts.boundary_three.settings import default_config
from scripts.boundary_three.p2_native import native_connect,isolated,atomic_npz
from scripts.boundary_three.p2_rules import connect,FullNucleusIndex,rng_hash


def sources(repo):
    names = ('scripts/boundary_three/p2_rules.py','scripts/boundary_three/p2_native.py',
             'bidcell/model/postprocess_predictions.py')
    result = {name:identity(repo/name) for name in names}
    # Integration may add orchestration callbacks around the unchanged native
    # tile body. Here the actual postprocess_connect/remove_islands source must
    # match in its entirety; the reused split helper must match mathematically.
    for name in names:
        if sha256(SOURCE_ROOT/name) != result[name]['sha256']:
            raise ValueError('integration scientific producer differs: '+name)
    def function(path,name):
        return ast.dump(next(n for n in ast.parse(path.read_text()).body
            if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
    split = 'scripts/crgf/postprocess.py'
    for name in ('split_patch','tile_task'):
        if function(repo/split,name) != function(SOURCE_ROOT/split,name):
            raise ValueError('integration current native mathematical AST differs: '+name)
    result['integration_native_wrapper'] = identity(SOURCE_ROOT/split)
    result['maxK_validator'] = identity(__file__)
    return result


def worker(base,out):
    out = isolated(out)
    out.mkdir(parents=True,exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    cfg = default_config()
    endpoint = cfg['endpoints'][base]
    machine = read(endpoint['machine'])
    source = sources(repo)
    pre = tifffile.memmap(endpoint['pre'])
    full = tifffile.imread(machine['nuclei_mask'])
    manifest = read(Path(endpoint['pre']).parent/'postprocess/manifest.json')
    counts = [(len(np.unique(pre[y:y+1024,x:x+1024]))-1,y,x) for y,x in manifest['coords']]
    if max(counts) != (5201,4096,0):
        raise ValueError('expected true maximum candidate inventory changed')
    y,x,size = 4096,0,1024
    image = np.asarray(pre[y:y+size,x:x+size])
    nuclei = full[y:y+size,x:x+size]
    seed = 1234+y*1000003+x
    inputs = dict(pre=identity(endpoint['pre']),complete_nuclei=identity(machine['nuclei_mask']),
        historical_tile=identity(Path(endpoint['pre']).parent/'postprocess'/f'tile_{y}_{x}.tif'))
    deps = dict(base=base,coords=[y,x],size=size,K=5201,seed=seed,inputs=inputs,source=source)
    if (out/'complete.json').exists():
        rec = read(out/'complete.json')
        if rec['dependencies'] != deps:
            raise ValueError('completed maximum-K test drift')
        for obj in rec['outputs'].values():
            check_identity(obj)
        return rec
    phases = {}
    start = time.monotonic()
    atomic_json(out/'progress.json',dict(stage='native_connect',updated_utc=utc(),K=5201))
    random.seed(seed)
    t = time.monotonic()
    baseline,nd,na = native_connect(image,nuclei)
    phases['native_connect_seconds'] = time.monotonic()-t
    before_islands = random.getstate()
    if len(nd['candidate_order']) != 5201:
        raise ValueError('maximum-K was not actually executed')
    atomic_json(out/'progress.json',dict(stage='native_islands',updated_utc=utc(),K=5201,phases=phases))
    t = time.monotonic()
    native_islands = native.remove_islands(baseline,nuclei)
    phases['native_islands_seconds'] = time.monotonic()-t
    native_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    if not np.array_equal(native_islands,tifffile.imread(inputs['historical_tile']['path'])):
        raise ValueError('actual maximum-K disabled native differs from historical production tile')
    atomic_tiff(out/'native_postconnect.tif',baseline)
    atomic_tiff(out/'native_postislands.tif',native_islands)
    atomic_json(out/'native_diagnostics.json',nd)
    atomic_json(out/'progress.json',dict(stage='P2_full_nucleus_connect',updated_utc=utc(),K=5201,phases=phases))
    index_start = time.monotonic()
    index = FullNucleusIndex(full)
    phases['full_nucleus_index_seconds'] = time.monotonic()-index_start
    random.seed(seed)
    t = time.monotonic()
    p2,pd,pa,_ = connect(image,nuclei,'P2',index,(y,x))
    phases['P2_connect_seconds'] = time.monotonic()-t
    p2_connect_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    if random.getstate() != before_islands:
        raise ValueError('maximum-K P2 changed RNG entering islands')
    for key in ('candidate_order','candidate_hashes','rng_before_shuffle','rng_enter_islands','proposal_histogram'):
        if pd[key] != nd[key]:
            raise ValueError('maximum-K native/P2 candidate or RNG mismatch: '+key)
    if not np.array_equal(na['proposal_count'],pa['proposal_count']):
        raise ValueError('maximum-K proposal counts differ')
    if not np.array_equal(pa['baseline_postconnect'],baseline):
        raise ValueError('maximum-K P2 native counterfactual differs')
    conflict = pa['proposal_count'] >= 2
    if not np.array_equal(p2>0,baseline>0) or np.any((p2!=baseline)&~((nuclei==0)&conflict)):
        raise ValueError('maximum-K P2 local invariants failed')
    if index.max_observed_pairs > index.max_pairs:
        raise ValueError('maximum-K query allocation exceeded bound')
    atomic_json(out/'progress.json',dict(stage='P2_islands',updated_utc=utc(),K=5201,phases=phases))
    t = time.monotonic()
    p2_islands = native.remove_islands(p2,nuclei)
    phases['P2_islands_seconds'] = time.monotonic()-t
    p2_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    pd['rng_after_islands'] = rng_hash()
    atomic_tiff(out/'P2_postconnect.tif',p2)
    atomic_tiff(out/'P2_postislands.tif',p2_islands)
    atomic_json(out/'P2_diagnostics.json',pd)
    atomic_npz(out/'P2_conflict_details.npz',proposal_count=pa['proposal_count'],
               conflict_resolution=pa['conflict_resolution'],conflict_proposals=pa['conflict_proposals'])
    rec = dict(status='passed',dependencies=deps,phases=phases,wall_seconds=time.monotonic()-start,
        maximum_K_executed=5201,native_historical_tile_pixels_equal=True,
        candidate_masks_exact=True,candidate_order_exact=True,RNG_enter_islands_exact=True,
        counterfactual_exact=True,postconnect_foreground_equal=True,changes_only_extranuclear_conflicts=True,
        max_query_pairs=index.max_observed_pairs,query_pair_limit=index.max_pairs,
        conflict_pixels=int(conflict.sum()),exact_ties=pd['exact_tie_pixels'],
        postconnect_changed_pixels=int((p2!=baseline).sum()),
        peak_RSS_after_native_bytes=native_peak,peak_RSS_after_P2_connect_bytes=p2_connect_peak,
        peak_worker_RSS_bytes=p2_peak,RSS_semantics='per fresh worker process high-water mark; includes imports, full input and retained diagnostic arrays',
        one_worker_threads=1,scope='maximum-K development tile resource admission; no new formal endpoint, matrix or scoring',
        outputs=inventory(out))
    atomic_json(out/'complete.json',rec)
    print(base,'maximum-K native/P2 passed',phases,'peak_RSS',p2_peak,flush=True)
    return rec


def run(out):
    out = isolated(out)
    out.mkdir(parents=True,exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    with compute_lock(timeout=3600):
        print('maximum-K compute lock acquired',utc(),flush=True)
        results = {}
        for base in ('BASE_MATCH','V2_MORPH'):
            env = dict(os.environ,PYTHONPATH=str(repo),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
            subprocess.run([sys.executable,__file__,'--base',base,'--output',str(out/base)],
                           cwd=repo,env=env,check=True)
            rec = read(out/base/'complete.json')
            for obj in rec['outputs'].values():
                check_identity(obj)
            results[base] = rec
        peak = max(r['peak_worker_RSS_bytes'] for r in results.values())
        cost = {}
        cfg = default_config()
        for base,rec in results.items():
            hist = read(Path(cfg['endpoints'][base]['pre']).parent/'postprocess/complete.json')
            phases = rec['phases']
            # Worst observed tile for all48, 4 serial-worker slots, plus25%
            # internal overhead. Parent queue contingency is additional.
            native_tile = phases['native_connect_seconds']+phases['native_islands_seconds']
            p2_tile = phases['P2_connect_seconds']+phases['P2_islands_seconds']+phases['full_nucleus_index_seconds']
            n = max(hist['wall_seconds'],native_tile*48/4*1.25)
            p = max(2.5*hist['wall_seconds'],p2_tile*48/4*1.25)
            fixed = 480+2400+1800+600+(1800 if base=='V2_MORPH' else 0)
            cost[base] = dict(native_full_seconds=n,P2_full_seconds=p,
                full_node_seconds=n+p+fixed,
                basis='all48 tiles charged observed max-K tile cost /4 workers with1.25 internal overhead; also no less than previous historical-native/2.5xP2 bound; separate queue contingency remains additional')
        record = dict(status='passed',source=sources(repo),bases={k:identity(out/k/'complete.json') for k in results},
            maximum_K=5201,both_maximum_tiles_executed=True,
            peak_single_worker_RSS_bytes=peak,four_worker_plus_parent_RSS_upper_estimate_bytes=5*peak,
            memory_semantics='conservative 5x independent worker peak (4 workers + parent), ignoring fork-shared memory savings; full matrix/evaluation memory covered separately',
            cost_estimates=cost,prior_full_chain_receipt=identity(out.parent/'attempt_01/complete.json'),
            formal_experiment_started=False,independent_code_audit=False,outputs=inventory(out))
        atomic_json(out/'complete.json',record)
        print('both maximum-K tests complete',utc(),flush=True)
        return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--base',choices=('BASE_MATCH','V2_MORPH'))
    args = parser.parse_args()
    worker(args.base,args.output) if args.base else run(args.output)
