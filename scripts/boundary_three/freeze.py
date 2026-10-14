"""Validate and reuse immutable historical domains; freeze this comparison before new scores."""
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
from .common import *
from .evaluation import check_legacy_identity, verify_legacy_receipt
from scripts.paired_v2_pipeline.math import quartiles

def validate_historical_domains(cfg):
    cfg=load_run(cfg);folder=Path(cfg['inputs']['frozen_strata']);rec=verify_legacy_receipt(folder/'complete.json')
    for key in ('metadata','prepared','reference','nuclei','historical_matrix'):check_legacy_identity(rec['dependencies'][key])
    spec=read(folder/'strata.json');pop=pd.read_csv(folder/'fixed_cell_groups.csv')
    meta=np.load(cfg['inputs']['nucleus_metadata'])
    if len(pop)!=107955 or pop.cell_id.duplicated().any() or not np.array_equal(pop.cell_id,meta['ids']):raise ValueError('frozen ID population invalid')
    if sum(spec['domain_molecule_denominators'])!=34442716:raise ValueError('frozen molecular denominator differs')
    nuc=tifffile.imread(rec['dependencies']['nuclei']['path']);domain=np.load(folder/'domains.npy',mmap_mode='r')
    support=np.load(folder/'support_counts.npy',mmap_mode='r');unknown=np.load(folder/'any_unknown.npy',mmap_mode='r');hetero=np.load(folder/'heterotypic.npy',mmap_mode='r')
    expected=np.zeros(nuc.shape,np.uint8);expected[support==1]=1;expected[(support>=2)&unknown]=2
    expected[(support>=2)&~unknown&~hetero]=3;expected[(support>=2)&~unknown&hetero]=4;expected[nuc>0]=5
    if not np.array_equal(domain,expected):raise ValueError('frozen support domain semantics differ')
    edges,group=quartiles(pop.density_30um)
    if edges.tolist()!=spec['density_edges'] or not np.array_equal(group,pop.density_quartile):raise ValueError('frozen density quantiles differ')
    for side in (500,1000,2000):
        blocks,index=np.unique(np.floor(pop[['x_um','y_um']].to_numpy()/side).astype(np.int64),axis=0,return_inverse=True)
        if blocks.tolist()!=spec['grids'][str(side)]['blocks'] or not np.array_equal(index,pop[f'block_{side}']):raise ValueError('frozen blocks differ')
    counts=Path(cfg['inputs']['frozen_counts']);crec=verify_legacy_receipt(counts/'complete.json')
    for obj in crec['dependencies'].values():check_legacy_identity(obj)
    e=endpoint_spec(cfg,'BASE_MATCH');matrix=verify_legacy_receipt(Path(e['matrix'])/'complete.json')
    totals=np.load(Path(e['matrix'])/'counts.npy',mmap_mode='r').sum(1);groups=pd.read_csv(counts/'fixed_count_groups.csv');cspec=read(counts/'strata.json')
    ids=pd.read_csv(Path(e['matrix'])/'cell_mapping.csv').canonical_cell_id.to_numpy();edges,index=quartiles(totals)
    if not np.array_equal(pop.cell_id,ids) or not np.array_equal(groups.cell_id,ids) or not np.array_equal(groups.BASE_MATCH_counts,totals) or not np.array_equal(groups.count_quartile,index) or edges.tolist()!=cspec['edges']:
        raise ValueError('baseline fixed count strata differ')
    return dict(status='passed',original_ids=len(pop),eligible_molecules=34442716,
        historical_strata=identity(folder/'complete.json'),historical_counts=identity(counts/'complete.json'),
        original_generation_utc=spec['time'],reuse_scope='historical BASE_MATCH population; earlier scores were seen; newly frozen before this run outcomes')

def freeze_inputs(cfg):
    cfg=load_run(cfg)
    for n in NODES:
        if output_path(cfg,n,'complete.json').exists():raise ValueError('new formal node already exists before freeze')
    result=validate_historical_domains(cfg)
    for name,key in [('frozen_strata','frozen_strata'),('frozen_counts','frozen_counts')]:
        src=Path(cfg['inputs'][key]);dest=output_path(cfg,name);dest.mkdir(parents=True,exist_ok=True)
        for p in sorted(src.iterdir()):
            if p.is_file() and p.name not in ('complete.json','progress.json'):
                target=dest/p.name
                if target.exists() and target.resolve()!=p.resolve():raise ValueError('unexpected freeze output '+str(target))
                if not target.exists():target.symlink_to(p)
        atomic_json(dest/'complete.json',dict(status='succeeded',validated_reuse=True,
            original_receipt=identity(src/'complete.json'),validation=result,outputs=inventory(dest)))
    atomic_json(output_path(cfg,'prepared','frozen_inputs.json'),dict(result,config_digest=digest(cfg)))
    return result

def validate_inputs(cfg):
    cfg=load_run(cfg);result=validate_historical_domains(cfg)
    saved=read(output_path(cfg,'prepared','frozen_inputs.json'))
    if saved!=dict(result,config_digest=digest(cfg)):raise ValueError('frozen comparison receipt drift')
    for folder in ('frozen_strata','frozen_counts'):
        rec=verify_legacy_receipt(output_path(cfg,folder,'complete.json'))
        if rec['validation']!=result:raise ValueError('frozen reuse receipt drift')
    return result

def stream_coordinates(cfg):
    rec=read(Path(endpoint_spec(cfg,'BASE_MATCH')['matrix'])/'complete.json');cursor=0
    for chunk in rec['receipts']:
        if chunk['row_start']!=cursor:raise ValueError('coordinate cursor gap')
        path=Path(chunk['coordinate_file'])
        if sha256(path)!=chunk['coordinate_sha256']:raise ValueError('coordinate chunk drift')
        a=np.load(path,mmap_mode='r')
        if len(a)!=chunk['rows']:raise ValueError('coordinate row count differs')
        cursor=chunk['row_stop'];yield chunk,a
    if cursor!=34442716:raise ValueError('coordinate final denominator differs')
