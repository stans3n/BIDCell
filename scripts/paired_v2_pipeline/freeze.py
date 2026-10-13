"""Freeze outcome-independent populations before formal endpoint scoring."""
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
from scipy.spatial import cKDTree
from scripts.crgf.common import atomic_npy
from scripts.new_methods.common import json_safe
from scripts.new_methods.data import Evidence
from .common import *
from .math import quartiles


def classify_domains(nuclei,counts,unknown,heterotypic):
    d=np.zeros(nuclei.shape,np.uint8)
    d[counts==1]=1;d[(counts>=2)&unknown]=2
    d[(counts>=2)&~unknown&~heterotypic]=3;d[(counts>=2)&~unknown&heterotypic]=4
    d[nuclei>0]=5
    return d


def stream_coordinates(q):
    """Identity-verified native coordinate records; do not infer a new grid."""
    e=endpoint('V2_MORPH');rec=read(e['matrix']/'complete.json')
    for chunk in rec['receipts']:
        p=Path(chunk['coordinate_file'])
        if sha256(p)!=chunk['coordinate_sha256']:raise ValueError('historical coordinate cache changed')
        a=np.load(p,mmap_mode='r')
        if len(a)!=chunk['rows']:raise ValueError('historical coordinate coverage changed')
        yield chunk,a


def freeze_dependencies(q):
    machine=read(endpoint('V2_MORPH')['machine'])
    return dict(source_bundle=q['source_bundle'],contract=q['bindings']['contract'],
        metadata=identity(INPUT/'prepared/nucleus_metadata.npz'),prepared=identity(INPUT/'prepared/manifest.json'),
        reference=identity(INPUT/'configs/reference_region.json'),nuclei=identity(machine['nuclei_mask']),
        historical_matrix=identity(endpoint('V2_MORPH')['matrix']/'complete.json'),
        historical_reuse=q['bindings']['historical_reuse'])


def freeze_inputs(q):
    out=Path(q['root'])/'frozen_strata';out.mkdir(exist_ok=True)
    deps=freeze_dependencies(q)
    if (out/'complete.json').exists():return validate_inputs(q)
    for arm in RUNS:
        if (endpoint(arm)['evaluation']/'complete.json').exists():raise ValueError('formal scores exist before frozen input strata')
    verify_history(q);ev=Evidence(INPUT);nuc=ev.nuclei;ids=ev.ids.astype(np.int64)
    if len(ids)!=107955 or len(np.unique(ids))!=len(ids):raise ValueError('original ID universe differs')
    typ=ev.meta['type_index'];counts=np.zeros(nuc.shape,np.uint32)
    unknown=np.zeros(nuc.shape,bool);hetero=np.zeros(nuc.shape,bool);first=np.full(nuc.shape,-2,np.int16)
    outside=0;fallback=0
    for i,cid in enumerate(ids):
        y,x,_,fb=ev.support(int(cid));valid=(y>=0)&(x>=0)&(y<nuc.shape[0])&(x<nuc.shape[1])
        outside+=int((~valid).sum());fallback+=bool(fb);y=y[valid];x=x[valid]
        counts[y,x]+=1;t=int(typ[i])
        if t==-1:unknown[y,x]=True
        else:
            seen=first[y,x];hetero[y,x]|=(seen>=0)&(seen!=t)
            first[y,x]=np.where(seen==-2,t,seen)
        if i%2000==0:
            atomic_json(out/'progress.json',dict(nuclei_completed=i,total=len(ids),time=utc()))
            print('freeze global support',i,'/',len(ids),flush=True)
    domains=classify_domains(nuc,counts,unknown,hetero)
    single_unknown=(domains==1)&unknown
    for name,a in [('domains',domains),('support_counts',counts),('any_unknown',unknown),
                   ('heterotypic',hetero),('first_known_type',first),('single_unknown',single_unknown)]:
        atomic_npy(out/(name+'.npy'),a)
    centroids=ev.centroids.astype(float);pixel=read(endpoint('V2_MORPH')['machine'])['pixel_um']
    if pixel!=1.0 or not np.isfinite(centroids).all():raise ValueError('centroid coordinate protocol drift')
    density=cKDTree(centroids*pixel).query_ball_point(centroids*pixel,30,return_length=True,workers=1)-1
    edges,groups=quartiles(density)
    populations=pd.DataFrame(dict(cell_id=ids,original_type_index=typ,original_type=[ev.type_names[t] if t>=0 else 'Unknown' for t in typ],
        x_um=centroids[:,0]*pixel,y_um=centroids[:,1]*pixel,density_30um=density,density_quartile=groups))
    grids={}
    for side in (500,1000,2000):
        xy=np.floor(centroids*pixel/side).astype(np.int64);universe,bi=np.unique(xy,axis=0,return_inverse=True)
        populations[f'block_{side}']=bi
        grids[str(side)]=dict(side_um=side,origin_um=[0,0],blocks=universe.tolist(),n_blocks=len(universe),population=np.bincount(bi).tolist())
    populations.to_csv(out/'fixed_cell_groups.csv',index=False)
    denom=np.zeros(7,np.int64);single_n=0;block_denoms={str(s):{} for s in (500,1000,2000)}
    for _,a in stream_coordinates(q):
        valid=(a['x']>=0)&(a['y']>=0);d=np.full(len(a),6,np.uint8)
        d[valid]=domains[a['y'][valid],a['x'][valid]];denom+=np.bincount(d,minlength=7)
        single_n+=int(single_unknown[a['y'][valid],a['x'][valid]].sum())
        for side in (500,1000,2000):
            xy=np.stack((a['x'][valid]//side,a['y'][valid]//side),1)
            pairs,nn=np.unique(xy,axis=0,return_counts=True)
            for pair,n in zip(pairs,nn):
                key=f'{pair[0]},{pair[1]}';dest=block_denoms[str(side)];dest[key]=dest.get(key,0)+int(n)
    if int(denom.sum())!=34442716:raise ValueError('fixed native molecule denominator differs')
    spec=dict(status='frozen_before_formal_scores',density_radius_um=30,density_edges=edges.tolist(),
        quartile_algorithm='linear quantiles, <=lower group, duplicate edges merged',grids=grids,
        original_type_names=ev.type_names,domains=list(DOMAINS),domain_molecule_denominators=denom.tolist(),
        single_unknown_molecule_denominator=single_n,block_molecule_denominators=block_denoms,
        support_universe='all complete original global nuclei; frozen Evidence.support; nuclear exterior only',
        support_oob_pixels_before_crop=outside,support_ellipse_fallback_nuclei=fallback,
        pixel_class_counts=np.bincount(domains.ravel(),minlength=6).tolist(),original_ids=len(ids),
        unknown_ids=int((typ==-1).sum()),time=utc())
    atomic_json(out/'strata.json',spec)
    atomic_json(out/'complete.json',dict(status='succeeded',dependencies=deps,outputs=inventory(out)))
    return validate_inputs(q)


def validate_inputs(q):
    out=Path(q['root'])/'frozen_strata';rec=read(out/'complete.json')
    if rec['status']!='succeeded' or rec['dependencies']!=freeze_dependencies(q):raise ValueError('frozen input stratum dependencies changed')
    verify_outputs(rec);spec=read(out/'strata.json');pop=pd.read_csv(out/'fixed_cell_groups.csv')
    meta=np.load(INPUT/'prepared/nucleus_metadata.npz')
    if not np.array_equal(pop.cell_id,meta['ids']) or len(pop)!=107955 or int(spec['unknown_ids'])!=1040:raise ValueError('fixed populations differ')
    nuc=tifffile.imread(rec['dependencies']['nuclei']['path']);d=np.load(out/'domains.npy',mmap_mode='r')
    n=np.load(out/'support_counts.npy',mmap_mode='r');u=np.load(out/'any_unknown.npy',mmap_mode='r');h=np.load(out/'heterotypic.npy',mmap_mode='r')
    if not np.array_equal(d,classify_domains(nuc,n,u,h)) or not np.array_equal(d==5,nuc>0):raise ValueError('fixed domain exclusivity/nuclear classification differs')
    edges,g=quartiles(pop.density_30um)
    if not np.array_equal(edges,spec['density_edges']) or not np.array_equal(g,pop.density_quartile):raise ValueError('density strata differ')
    if sum(spec['domain_molecule_denominators'])!=34442716:raise ValueError('native domain denominator coverage differs')
    for side in (500,1000,2000):
        universe,bi=np.unique(np.floor(pop[['x_um','y_um']].to_numpy()/side).astype(int),axis=0,return_inverse=True)
        if universe.tolist()!=spec['grids'][str(side)]['blocks'] or not np.array_equal(bi,pop[f'block_{side}']):raise ValueError('fixed block universe differs')
    return dict(status='passed',receipt=identity(out/'complete.json'),original_ids=107955,
                native_molecules=34442716,domain_pixels=spec['pixel_class_counts'],density_edges=spec['density_edges'])


def count_dependencies(q):
    e=endpoint('BASE_MATCH')
    return dict(source_bundle=q['source_bundle'],frozen_inputs=identity(Path(q['root'])/'frozen_strata/complete.json'),
                matrix=identity(e['matrix']/'complete.json'),counts=identity(e['matrix']/'counts.npy'),mapping=identity(e['matrix']/'cell_mapping.csv'))


def freeze_counts(q):
    out=Path(q['root'])/'frozen_counts';out.mkdir(exist_ok=True)
    if (out/'complete.json').exists():return validate_counts(q)
    for arm in RUNS:
        if (endpoint(arm)['evaluation']/'complete.json').exists():raise ValueError('formal scoring began before E0 count-quartile freeze')
    from scripts.paired_v2.stages import validate_stage
    validate_stage(config_path('BASE_MATCH'),'matrix',require_bound=True);validate_inputs(q)
    e=endpoint('BASE_MATCH');ids=pd.read_csv(e['matrix']/'cell_mapping.csv').canonical_cell_id.to_numpy()
    counts=np.load(e['matrix']/'counts.npy',mmap_mode='r').sum(1)
    pop=pd.read_csv(Path(q['root'])/'frozen_strata/fixed_cell_groups.csv')
    if not np.array_equal(ids,pop.cell_id):raise ValueError('E0 count strata IDs differ')
    edges,g=quartiles(counts)
    pd.DataFrame(dict(cell_id=ids,BASE_MATCH_counts=counts,count_quartile=g)).to_csv(out/'fixed_count_groups.csv',index=False)
    atomic_json(out/'strata.json',dict(status='frozen_before_new_formal_scores',edges=edges.tolist(),
        group_counts=np.bincount(g,minlength=len(edges)+1).tolist(),algorithm='linear, <=lower, duplicate edges merged',time=utc()))
    atomic_json(out/'complete.json',dict(status='succeeded',dependencies=count_dependencies(q),outputs=inventory(out)))
    return validate_counts(q)


def validate_counts(q):
    out=Path(q['root'])/'frozen_counts';rec=read(out/'complete.json')
    if rec['status']!='succeeded' or rec['dependencies']!=count_dependencies(q):raise ValueError('frozen E0 count groups changed')
    verify_outputs(rec);pop=pd.read_csv(out/'fixed_count_groups.csv');spec=read(out/'strata.json')
    e=endpoint('BASE_MATCH');actual=np.load(e['matrix']/'counts.npy',mmap_mode='r').sum(1)
    ids=pd.read_csv(e['matrix']/'cell_mapping.csv').canonical_cell_id.to_numpy();edges,g=quartiles(actual)
    if not np.array_equal(pop.cell_id,ids) or not np.array_equal(pop.BASE_MATCH_counts,actual) or not np.array_equal(pop.count_quartile,g) or edges.tolist()!=spec['edges']:
        raise ValueError('E0 count quartile populations or boundaries differ')
    return dict(status='passed',receipt=identity(out/'complete.json'),edges=spec['edges'],groups=spec['group_counts'])
