"""Conservative real-N/group-count resource probe, not full-slide analysis.

Uses the actual 107955 nuclear IDs/coordinates and frozen W algorithm. Group
membership is an explicit worst-case bound (every ID in every group), not a
claim to have prepared the scientific fixed domains. Independently dense block
statistics measure a conservative compressed-output bound.
"""
import argparse,fcntl,time,gzip,os
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.rep1_triple_simple_boundary.common import *
from scripts.rep1_triple_simple_boundary.frozen_domains import bootstrap_weights,CELL_DIMENSIONS
from scripts.rep1_triple_simple_boundary.spatial_bootstrap import GroupProjector,block_intervals
from scripts.rep1_triple_simple_boundary.comparisons import atomic_csv_gzip
from scripts.rep1_triple_simple_boundary.contracts import SCALES,REGIONS
from tests.rep1_triple_real_fixtures import MemorySampler

def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);ins=read('/home/zwan0726/work/docs/BIDCell_Rep1_Triple_Simple_Inputs_20261017.json');meta=np.load(ins['shared']['metadata'])
    ids=meta['ids'].astype(np.int64);cells=pd.DataFrame(index=pd.Index(ids,name='cell_id'));weights={}
    for side in SCALES:
        blocks,bi,W=bootstrap_weights(meta['centroid_xy'],side);cells['block_'+str(side)]=bi;weights[side]=(blocks,W)
    # 9 spatial regions with 3 type classes, 6 properties x 5 values, K/density,
    # plus single-Unknown; one oob cell group; all real type/block marginals.
    region_count=(len(REGIONS)-1)*(1+3*(1+len(CELL_DIMENSIONS)*5)+5+4+1+len(CELL_DIMENSIONS)*5)+1+(len(REGIONS)-1)
    marginal_bound=len(np.unique(meta['type_index']))+len(CELL_DIMENSIONS)*5+sum(len(b) for b,w in weights.values())
    G=max(1600,region_count+marginal_bound);groups={f'group_{i:04d}':ids for i in range(G)}
    started=time.monotonic();timings=[];rows=[]
    with Path('/home/zwan0726/work/outputs/cos_gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with MemorySampler((lock.fileno(),)) as memory:
            t=time.monotonic();projector=GroupProjector(cells,groups);timings.append(dict(operation='worst_case_membership',seconds=time.monotonic()-t))
            population_hash=id_hash(ids);values=np.linspace(-1,1,len(ids));rng=np.random.default_rng(20261017);files=[]
            for side in SCALES:
                blocks,W=weights[side];t=time.monotonic();s,n=projector.blocks(ids,values,side,len(blocks));timings.append(dict(operation='one_metric_projection',scale=side,seconds=time.monotonic()-t))
                # No reuse of identical group columns in this independent dense
                # interval/output bound. Empty groups are covered by unit tests.
                s=rng.normal(size=s.shape);n=rng.integers(20,2000,size=n.shape).astype(float)
                folder=out/f'scale_{side}';t=time.monotonic();res=block_intervals(s,n,W,list(groups),checkpoint=(folder,dict(fixture='synthetic_dense_statistics_real_global_blocks',population_ids_sha256=population_hash)))
                timings.append(dict(operation='one_dense_metric_bootstrap_commit',scale=side,seconds=time.monotonic()-t));files.append(identity(folder/'block_sufficient_statistics.npz'))
                for row in res:row.update(profile='sc',metric='positive_F1',scale_um=side,comparison='resource_bound',population_ids_sha256=population_hash)
                rows.extend(res)
            atomic_csv_gzip(out/'one_metric_all_scales.csv.gz',pd.DataFrame(rows))
    # Final comparison jobs = 26 pairs +19 factorial +12 raw +12 common-all12.
    per_measure_bytes=sum(x['bytes'] for x in files);table_bytes=(out/'one_metric_all_scales.csv.gz').stat().st_size
    result=dict(status='passed_resource_probe',scope='no domain freeze, no formal endpoint or full analysis',actual_n_cells=len(ids),group_count=G,
        prespecified_group_count_upper_bound=region_count+marginal_bound,group_membership='all actual IDs in every group; conservative sparse-projector allocation bound',
        real_global_blocks={s:len(b) for s,(b,w) in weights.items()},resource=memory.result(),wall_seconds=time.monotonic()-started,timings=timings,
        comparison_jobs=69,comparison_profiles_metrics=12,comparison_intervals_rows=69*12*3*G,
        dense_block_statistics_measured_bytes_per_metric_three_scales=per_measure_bytes,
        comparison_dense_block_statistics_69job_12metric_projection_bytes=per_measure_bytes*69*12,
        comparison_table_gzip_measured_bytes_per_metric_three_scales=table_bytes,
        comparison_table_69job_12metric_projection_bytes=table_bytes*69*12,
        limitations=['runtime projection is conservative membership but machine load affects timing','full boundary transfer and R evaluation resources are measured separately','actual frozen domain groups and spatial ledger volume must be bound at admission; these are not completed analysis outputs'],
        sources={str(p):identity(p) for p in (Path(__file__),Path(__file__).parents[1]/'scripts/rep1_triple_simple_boundary/spatial_bootstrap.py')},metadata=identity(ins['shared']['metadata']),outputs=inventory(out))
    atomic_json(out/'fixture_receipt.json',result);print({k:result[k] for k in ('status','wall_seconds','group_count','comparison_intervals_rows','resource')},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();run(a.output)
