"""Complete prespecified spatial strata, gross transfer rates and local cost CI."""
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
from .contracts import REGIONS,SCALES
from .frozen_domains import in_region,block_indices,CELL_DIMENSIONS
from .common import (identity,check_identity,check_receipt,atomic_json,atomic_npz,read,
                     inventory,write_receipt,committed,progress,id_hash)
from .comparisons import atomic_csv_gzip
from .spatial_bootstrap import GroupProjector,block_intervals

KIND_NAMES=('background_unchanged','unchanged_cell','cell_to_cell','cell_to_background','background_to_cell')

def kinds(source,target):
    return np.where(source==target,np.where(source>0,1,0),np.where(source==0,4,np.where(target==0,3,2))).astype(np.int8)

def spatial_masks(st,cells):
    """Position-defined fixed strata, with BASE-owner cell properties.

    BASE background stays in the region/type/K/local-density denominators. It is
    explicitly NA for cell properties, and never becomes a fabricated cell.
    """
    owner=st['BASE_owner'];props={}
    for key in CELL_DIMENSIONS+('original_type_index',):
        lut=np.full(int(cells.index.max())+1,-1,np.int16);lut[cells.index]=cells[key].to_numpy(np.int16)
        props[key]=lut[owner]
    for region in REGIONS:
        use=in_region(st['region_bits'],region);yield region,use
        typ=st['interface_type'] if region.startswith('interface') else st['competition_type']
        for ti,label in ((-1,'not_competing'),(0,'same'),(1,'heterotypic'),(2,'Unknown')):
            t=use&(typ==ti);name=region+'__type_'+label;yield name,t
            for dim in CELL_DIMENSIONS:
                for q in (-1,0,1,2,3):yield name+'__'+dim+'_'+str(q),t&(props[dim]==q)
        for dim in CELL_DIMENSIONS:
            for q in (-1,0,1,2,3):yield region+'__'+dim+'_'+str(q),use&(props[dim]==q)
        for k in range(5):yield region+'__K_'+str(k),use&(st['K']==k)
        for q in (-1,0,1,2,3):yield region+'__local_density_'+str(q),use&(st['local_density']==q)
        if 'single_unknown' in st:yield region+'__single_unknown',use&st['single_unknown']
    for t in sorted(cells.original_type_index.unique()):
        use=(owner>0)&(props['original_type_index']==t);name='BASE_owner_type_'+str(t);yield name,use
        for dim in ('RNA_density_quartile','nuclear_crowding_quartile'):
            for q in (-1,0,1,2,3):yield name+'__'+dim+'_'+str(q),use&(props[dim]==q)

def _statistics(v):
    # Five exclusive ownership transitions, derivations preserve all gross flows.
    den=v.sum(-1);source=v[...,1]+v[...,2]+v[...,3];target=v[...,1]+v[...,2]+v[...,4]
    return dict(capture_delta_pp=(100*(target-source),den),owner_retention=(v[...,1],source),
                cell_to_cell_rate=(v[...,2],den),cell_to_background_rate=(v[...,3],den),
                background_to_cell_rate=(v[...,4],den),gross_transfer_rate=(v[...,2:].sum(-1),den))

def spatial_families(st,cells):
    """Equivalent categorical partitions avoid thousands of full-length masks."""
    owner=st['BASE_owner'];props={}
    for key in CELL_DIMENSIONS+('original_type_index',):
        lut=np.full(int(cells.index.max())+1,-1,np.int16);lut[cells.index]=cells[key].to_numpy(np.int16);props[key]=lut[owner]
    for region in REGIONS:
        ix=np.flatnonzero(in_region(st['region_bits'],region));zero=np.zeros(len(ix),np.int8)
        yield [region],ix,zero
        typ=st['interface_type'] if region.startswith('interface') else st['competition_type'];t=typ[ix]+1
        labels=('not_competing','same','heterotypic','Unknown');prefix=[region+'__type_'+label for label in labels]
        yield prefix,ix,t
        for dim in CELL_DIMENSIONS:
            d=props[dim][ix]+1
            yield [p+'__'+dim+'_'+str(q) for p in prefix for q in (-1,0,1,2,3)],ix,t*5+d
            yield [region+'__'+dim+'_'+str(q) for q in (-1,0,1,2,3)],ix,d
        yield [region+'__K_'+str(k) for k in range(5)],ix,st['K'][ix]
        yield [region+'__local_density_'+str(q) for q in (-1,0,1,2,3)],ix,st['local_density'][ix]+1
        if 'single_unknown' in st:
            chosen=ix[st['single_unknown'][ix]];yield [region+'__single_unknown'],chosen,np.zeros(len(chosen),np.int8)
    ts=np.asarray(sorted(cells.original_type_index.unique()));ix=np.flatnonzero(owner>0);t=np.searchsorted(ts,props['original_type_index'][ix])
    prefix=['BASE_owner_type_'+str(x) for x in ts];yield prefix,ix,t
    for dim in ('RNA_density_quartile','nuclear_crowding_quartile'):
        yield [p+'__'+dim+'_'+str(q) for p in prefix for q in (-1,0,1,2,3)],ix,t*5+props[dim][ix]+1

def accumulate_spatial(st,cells,kind,names,totals,acc):
    lookup={name:i for i,name in enumerate(names)};seen=[]
    for group,ix,cat in spatial_families(st,cells):
        seen.extend(group);rows=np.asarray([lookup[k] for k in group]);G=len(group)
        if np.any(cat<0) or np.any(cat>=G):raise ValueError('fixed spatial categorical value outside definition')
        code=cat.astype(np.int64)*5+kind[ix]
        totals[rows]+=np.bincount(code,minlength=G*5).reshape(G,5)
        for side,blocks in acc.items():
            bi=st['block_'+str(side)][ix];good=bi>=0
            if np.any(bi[good]>=len(blocks)):raise ValueError('fixed spatial block outside global W')
            values=np.bincount(bi[good].astype(np.int64)*G*5+code[good],minlength=len(blocks)*G*5).reshape(len(blocks),G,5)
            blocks[:,rows,:]+=values
    if len(seen)!=len(names) or set(seen)!=set(names):raise ValueError('categorical spatial stratum coverage drift')

def spatial_accounts(cfg,manifest,a,b,cells,out,deps,callback=None):
    """Full molecule/pixel fixed strata, exact cursor, shared global block W."""
    from .boundary_analysis import _owner_chunk
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return check_receipt(out/'complete.json',deps)
    frozen=Path(cfg['run_root'])/'frozen_domain';weights={s:dict(np.load(frozen/f'bootstrap_weights_{s}.npz')) for s in SCALES}
    source=tifffile.imread(b['final']);target=tifffile.imread(a['final']);base=tifffile.imread(manifest['endpoints']['BASE_MATCH']['final'])
    grids={key:np.load(frozen/file,mmap_mode='r') for key,file in [('region_bits','region_bits.npy'),('local_density','local_density_group.npy'),('competition_type','competition_type.npy'),('interface_type','interface_type.npy')]}
    K=np.load(Path(manifest['historical_strata'])/'support_counts.npy',mmap_mode='r');unknown=np.load(Path(manifest['historical_strata'])/'any_unknown.npy',mmap_mode='r')
    for unit in ('molecule','pixel'):
        folder=out/unit;folder.mkdir(exist_ok=True);cd=dict(deps,unit=unit);state=folder/'state.json'
        if (folder/'complete.json').exists():check_receipt(folder/'complete.json',cd);continue
        cursor=0;totals=acc=names=None
        if state.exists():
            saved=check_receipt(state,cd);z=dict(np.load(saved['outputs']['snapshot']['path']));cursor=saved['cursor'];totals=z['totals'];names=saved['names'];acc={s:z['blocks_'+str(s)] for s in SCALES}
        if unit=='molecule':
            ar=read(Path(a['matrix'])/'complete.json')['receipts'];br=read(Path(b['matrix'])/'complete.json')['receipts'];total=len(ar)
        else:total=(len(source)+127)//128
        for index in range(cursor,total):
            if unit=='molecule':
                st=dict(np.load(frozen/'molecule_strata'/f'chunk_{index:04d}.npz'))
                so=_owner_chunk(b,br[index],manifest['coordinates'][index]);ta=_owner_chunk(a,ar[index],manifest['coordinates'][index])
            else:
                start=index*128;stop=min(start+128,len(source));yy,xx=np.mgrid[start:stop,:source.shape[1]]
                st={key:np.asarray(grid[start:stop]).ravel() for key,grid in grids.items()};st['BASE_owner']=base[start:stop].ravel()
                st['K']=np.minimum(K[start:stop],4).ravel();st['single_unknown']=((K[start:stop]==1)&unknown[start:stop]).ravel()
                for side in SCALES:st['block_'+str(side)]=block_indices(np.c_[xx.ravel(),yy.ravel()],weights[side]['blocks'],side)
                so=source[start:stop].ravel();ta=target[start:stop].ravel()
            kind=kinds(so,ta)
            # Allocate from the frozen definitions once; empty strata are retained.
            if names is None:
                names=[n for n,_ in spatial_masks(st,cells)];totals=np.zeros((len(names),5),np.int64)
                acc={s:np.zeros((len(weights[s]['blocks']),len(names),5),np.int64) for s in SCALES}
            accumulate_spatial(st,cells,kind,names,totals,acc)
            snap=folder/f'state_{index+1:04d}.npz';atomic_npz(snap,totals=totals,**{'blocks_'+str(s):v for s,v in acc.items()})
            prior=read(state) if state.exists() else None
            atomic_json(state,dict(status='succeeded',dependencies=cd,outputs={'snapshot':identity(snap)},cursor=index+1,total=total,names=names))
            committed(state)
            if prior:
                old=prior['outputs']['snapshot'];check_identity(old);Path(old['path']).unlink()
            progress(callback,'fixed_'+unit+'_strata',index+1,total,receipt=str(state))
        records=[]
        globalstats=_statistics(totals)
        for side in SCALES:
            stats=_statistics(acc[side])
            for metric,(num,den) in stats.items():
                rows=block_intervals(num,den,weights[side]['W'],names,minimum_units=0,
                    checkpoint=(folder/'bootstrap'/f'{metric}_{side}',dict(cd,metric=metric,scale=side)),callback=callback)
                gnum,gden=globalstats[metric]
                for j,row in enumerate(rows):
                    row.update(unit=unit,metric=metric,scale_um=side,full_numerator=int(gnum[j]),full_denominator=int(gden[j]),
                        full_point=float(gnum[j]/gden[j]) if gden[j] else None,excluded_from_spatial_denominator=int(gden[j]-den[:,j].sum()),
                        spatial_population='fixed native grid; OOB has no block and explicit NA; overlapping strata not additive')
                records.extend(rows)
        atomic_csv_gzip(folder/'stratified_intervals.csv.gz',pd.DataFrame(records))
        atomic_csv_gzip(folder/'stratified_counts.csv.gz',pd.DataFrame(totals,columns=KIND_NAMES).assign(stratum=names,unit=unit))
        write_receipt(folder/'complete.json',cd,inventory(folder),all_prespecified_strata=len(names),unit=unit)
    return write_receipt(out/'complete.json',deps,inventory(out),units=['molecule','pixel'],fixed_strata_complete=True)

def cell_cost_intervals(cfg,cell,cells,groups,out,deps,callback=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);projector=GroupProjector(cells,groups);rows=[]
    if (out/'complete.json').exists():return check_receipt(out/'complete.json',deps)
    ids=cells.index.to_numpy(np.int64)
    measures={m:(cell[m].to_numpy(float),None) for m in ('incoming_RNA','outgoing_RNA','net_RNA')}
    measures['pooled_RNA_retention']=(cell.retained_RNA.to_numpy(float),cell.previous_RNA.to_numpy(float))
    for side in SCALES:
        W=np.load(Path(cfg['run_root'])/'frozen_domain'/f'bootstrap_weights_{side}.npz')['W']
        for metric,(v,d) in measures.items():
            s,n=projector.blocks(ids,v,side,W.shape[1],d)
            result=block_intervals(s,n,W,projector.names,minimum_units=20 if d is None else 0,
                checkpoint=(out/'bootstrap'/f'{metric}_{side}',dict(deps,metric=metric,scale=side,population_ids_sha256=id_hash(ids))),callback=callback)
            for row in result:row.update(metric=metric,scale_um=side,unit='cell' if d is None else 'molecule_pooled_by_frozen_cell_block')
            rows.extend(result)
    atomic_csv_gzip(out/'intervals.csv.gz',pd.DataFrame(rows));return write_receipt(out/'complete.json',deps,inventory(out))
