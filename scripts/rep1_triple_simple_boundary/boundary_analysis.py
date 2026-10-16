"""Whole and fixed-domain pixel/molecule/cell/interface costs; complete sparse ledger."""
from pathlib import Path
from itertools import zip_longest
import numpy as np
import pandas as pd
import tifffile
from .contracts import REGIONS,SCALES,comparisons,comparison_id
from .common import (read,identity,check_identity,check_receipt,dependencies,atomic_json,atomic_npz,
                     atomic_csv,inventory,write_receipt,chunk_receipt,progress,reuse_partition,id_hash)
from .inputs import matrix_arrays
from .frozen_domains import in_region,block_indices,CELL_DIMENSIONS
from .comparisons import pooled_interval,atomic_csv_gzip

TRANSFER_COLUMNS=('gene_index','source_id','target_id','source_type_index','target_type_index','type_relation',
    'region_bits','local_RNA_density_quartile','support_K','interface_id','molecule_block_500','molecule_block_1000','molecule_block_2000',
    'source_BASE_count','target_BASE_count','source_RNA_density','target_RNA_density','source_crowding','target_crowding',
    'source_nuclear_area','target_nuclear_area','source_eccentricity','target_eccentricity',
    'source_cell_block_500','target_cell_block_500','source_cell_block_1000','target_cell_block_1000','source_cell_block_2000','target_cell_block_2000')

def transfer_kind(source,target):
    a=np.asarray(source);b=np.asarray(target);changed=a!=b
    return dict(unchanged_cell=(a>0)&~changed,background_unchanged=(a==0)&~changed,
                cell_to_cell=changed&(a>0)&(b>0),cell_to_background=(a>0)&(b==0),background_to_cell=(a==0)&(b>0))

def sparse_transfer(source,target,gene,strata,cells):
    """All observed gene×ID/type×density×space combinations, including owner0.

    One lossless region bitset records overlapping region membership. Consumers
    expand that bitset by regions.json, never add overlapping domain totals.
    """
    source=np.asarray(source,np.int64);target=np.asarray(target,np.int64);gene=np.asarray(gene,np.int64)
    use=source!=target;a,b=source[use],target[use];n=len(a)
    if not n:return np.empty((0,len(TRANSFER_COLUMNS)),np.int64),np.empty(0,np.int64)
    if not np.isin(a[a>0],cells.index).all() or not np.isin(b[b>0],cells.index).all():raise ValueError('transfer missing frozen owner ID/type')
    def values(ids,key,default=-2):
        return cells[key].reindex(ids).fillna(default).to_numpy(np.int64)
    at,bt=values(a,'original_type_index'),values(b,'original_type_index')
    relation=np.where((a==0)|(b==0),3,np.where((at<0)|(bt<0),2,np.where(at==bt,0,1)))
    arrays=[gene[use],a,b,at,bt,relation,strata['region_bits'][use],strata['local_density'][use],strata['K'][use],strata['interface_id'][use]]
    for s in (500,1000,2000):arrays.append(strata['block_'+str(s)][use])
    for key in ('BASE_count_quartile','RNA_density_quartile','nuclear_crowding_quartile','nuclear_area_quartile','eccentricity_quartile'):
        arrays.extend((values(a,key),values(b,key)))
    for s in (500,1000,2000):arrays.extend((values(a,'block_'+str(s)),values(b,'block_'+str(s))))
    keys,counts=np.unique(np.column_stack(arrays),axis=0,return_counts=True)
    return keys,counts.astype(np.int64)

def _owner_chunk(spec,record,coordinate):
    p=Path(spec['matrix'])/record['owner_file'];check_identity(dict(path=str(p),sha256=record['sha256']))
    if record['coordinate_sha256']!=coordinate['sha256'] or record['rows']!=coordinate['rows'] or record['row_start']!=coordinate['row_start'] or record['row_stop']!=coordinate['row_stop']:
        raise ValueError('transfer transcript ID row alignment differs')
    owner=np.load(p,mmap_mode='r')
    if owner.shape!=(record['rows'],) or owner.dtype!=np.dtype('uint32'):raise ValueError('owner chunk shape/dtype differs')
    return owner

def endpoint_regions(cfg,manifest,spec,out,callback=None):
    """Fixed full denominators include BASE background, plus explicit OOB."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True);deps=dependencies(cfg,'REGIONS_'+spec['name'])
    if (out/'complete.json').exists():return check_receipt(out/'complete.json',deps)
    frozen=Path(cfg['run_root'])/'frozen_domain';bits=np.load(frozen/'region_bits.npy',mmap_mode='r');mask=tifffile.imread(spec['final'])
    mr=read(Path(spec['matrix'])/'complete.json');rows=[];acc={r:[0,0] for r in REGIONS}
    for i,(rec,coord) in enumerate(zip_longest(mr['receipts'],manifest['coordinates'])):
        if rec is None or coord is None:raise ValueError('regional capture chunk coverage differs')
        owner=_owner_chunk(spec,rec,coord);st=np.load(frozen/'molecule_strata'/f'chunk_{i:04d}.npz')
        for r in REGIONS:
            use=in_region(st['region_bits'],r);acc[r][0]+=int(use.sum());acc[r][1]+=int(np.count_nonzero(owner[use]>0))
    for r in REGIONS:
        use=in_region(bits,r);den=int(use.sum());assigned=int(np.count_nonzero(mask[use]>0))
        rows.append(dict(endpoint=spec['name'],region=r,unit='pixel',denominator=den,numerator=assigned,value=assigned/den if den else None,NA_reason='' if den else 'empty_fixed_pixel_region'))
        den,assigned=acc[r];rows.append(dict(endpoint=spec['name'],region=r,unit='molecule',denominator=den,numerator=assigned,value=assigned/den if den else None,NA_reason='' if den else 'empty_fixed_molecule_region'))
    expected=pd.read_csv(frozen/'denominators.csv')
    for row in rows:
        v=expected[(expected.region==row['region'])&(expected.unit==row['unit'])]
        if len(v)!=1 or int(v.iloc[0].denominator)!=row['denominator']:raise ValueError('fixed regional denominator drift')
    atomic_csv(out/'capture_foreground.csv',pd.DataFrame(rows))
    return write_receipt(out/'complete.json',deps,inventory(out),endpoint=spec['name'],regions=list(REGIONS),reuse_binding=endpoint_binding(cfg,spec))

def endpoint_binding(cfg,spec):
    return dict(final=identity(spec['final']),matrix=identity(Path(spec['matrix'])/'complete.json'),
                frozen_domain=identity(cfg['domain_manifest']))

def pair_transfers(cfg,manifest,a,b,out,callback=None,reuse_from=None,include_structure=True):
    """Control b→candidate a; exact full per-cell/per-gene matrix reconstruction."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True);key=comparison_id(a['name'],b['name']);dep=dependencies(cfg,'TRANSFER_'+key)
    if (out/'complete.json').exists():return check_receipt(out/'complete.json',dep)
    frozen=Path(cfg['run_root'])/'frozen_domain';cells=pd.read_csv(frozen/'cells.csv').set_index('cell_id')
    binding=dict(candidate=endpoint_binding(cfg,a),control=endpoint_binding(cfg,b),
                 weights={str(s):identity(frozen/f'bootstrap_weights_{s}.npz') for s in SCALES})
    groupfile=np.load(frozen/'cell_groups.npz');groups={k:groupfile[k] for k in groupfile.files}
    for dimension in ('block_500','block_1000','block_2000','RNA_density_quartile','nuclear_crowding_quartile'):
        for k,part in cells.groupby(['original_type',dimension],dropna=False):groups['type__'+dimension+'__'+str(k)]=part.index.to_numpy(np.int64)
    if reuse_from is not None and (Path(reuse_from)/'complete.json').exists():
        if not (out/'initial_reuse.json').exists():reuse_partition(reuse_from,out,dep,binding,'initial_reuse.json')
        else:check_receipt(out/'initial_reuse.json',dep)
        if include_structure:structure_costs(cfg,a,b,cells,groups,out/'structure',callback)
        return write_receipt(out/'complete.json',dep,inventory(out),comparison=key,matrix_gene_cell_delta_exact=True,all313_gene_ledger=True,
            sparse_chunks=read(Path(reuse_from)/'complete.json')['sparse_chunks'],scientific_alarms_do_not_block=True,
            reuse_binding=binding,reused_from=identity(Path(reuse_from)/'complete.json'),structure_completed=include_structure)
    ids,genes,pc=matrix_arrays(b);ai,ag,cc=matrix_arrays(a)
    if not np.array_equal(ai,ids) or genes!=ag or not np.array_equal(ids,cells.index):raise ValueError('transfer complete matrix/mapping mismatch')
    mr=read(Path(a['matrix'])/'complete.json');pr=read(Path(b['matrix'])/'complete.json');N=len(ids);G=len(genes)
    net=np.zeros((N,G),np.int64);incoming=np.zeros(N,np.int64);outgoing=np.zeros(N,np.int64)
    regional={r:{k:0 for k in ('denominator','source_assigned','target_assigned','retained','cell_to_cell','cell_to_background','background_to_cell','unchanged_cell','background_unchanged')} for r in REGIONS}
    weights={s:dict(np.load(frozen/f'bootstrap_weights_{s}.npz')) for s in SCALES}
    blockstats={(r,s):np.zeros((len(weights[s]['blocks']),7),np.int64) for r in REGIONS for s in SCALES}
    # [all molecules, source-assigned, target-assigned, retained, c2c,c2bg,bg2c]
    for i,(ar,br,coord) in enumerate(zip_longest(mr['receipts'],pr['receipts'],manifest['coordinates'])):
        if ar is None or br is None or coord is None:raise ValueError('pair transcript rows missing')
        st=dict(np.load(frozen/'molecule_strata'/f'chunk_{i:04d}.npz'));co=np.load(coord['path'],mmap_mode='r')
        source=_owner_chunk(b,br,coord);target=_owner_chunk(a,ar,coord);change=source!=target
        cd=dict(dep,chunk=i,source_owner=br['sha256'],target_owner=ar['sha256'],coordinate=coord['sha256'])
        p=out/'chunks'/f'{i:04d}.npz';rp=p.with_suffix('.json')
        if rp.exists():check_receipt(rp,cd)
        else:
            keys,n=sparse_transfer(source,target,co['gene'],st,cells);atomic_npz(p,keys=keys,molecules=n)
            chunk_receipt(rp,cd,[p],columns=list(TRANSFER_COLUMNS),rows=len(co),changed=int(change.sum()),unit='molecule')
        z=np.load(p);keys=z['keys'];vals=z['molecules']
        if int(vals.sum())!=int(change.sum()):raise ValueError('sparse transfer omitted changed molecule')
        for owner,col,sign in [('source_id',1,-1),('target_id',2,1)]:
            use=keys[:,col]>0;row=np.searchsorted(ids,keys[use,col]);g=keys[use,0]
            np.add.at(net,(row,g),sign*vals[use]);np.add.at(outgoing if sign<0 else incoming,row,vals[use])
        kinds=transfer_kind(source,target)
        for r in REGIONS:
            use=in_region(st['region_bits'],r);v=regional[r];v['denominator']+=int(use.sum());v['source_assigned']+=int(np.count_nonzero(source[use]>0));v['target_assigned']+=int(np.count_nonzero(target[use]>0));v['retained']+=int(np.count_nonzero(use&(source>0)&~change))
            for k,m in kinds.items():v[k]+=int(np.count_nonzero(use&m))
            val=[np.ones(len(co),bool),source>0,target>0,(source>0)&~change,kinds['cell_to_cell'],kinds['cell_to_background'],kinds['background_to_cell']]
            for side in SCALES:
                bi=st['block_'+str(side)];ok=use&(bi>=0);dest=blockstats[(r,side)]
                for j,m in enumerate(val):dest[:,j]+=np.bincount(bi[ok],weights=m[ok].astype(np.int64),minlength=len(dest)).astype(np.int64)
        progress(callback,'complete_sparse_gene_transfer',i+1,len(mr['receipts']),receipt=str(rp))
    if not np.array_equal(cc-pc,net):raise ValueError('full gene-by-cell matrix delta differs from transcript transfer ledger')
    if not np.array_equal(net.sum(1),incoming-outgoing):raise ValueError('gross/net transfer row mismatch')
    total=pc.sum(1,dtype=np.int64);new=cc.sum(1,dtype=np.int64);cell=cells.copy()
    cell['previous_RNA']=total;cell['new_RNA']=new;cell['incoming_RNA']=incoming;cell['outgoing_RNA']=outgoing;cell['net_RNA']=incoming-outgoing
    cell['retained_RNA']=total-outgoing;cell['retention']=np.divide(total-outgoing,total,out=np.full(N,np.nan),where=total>0)
    cell['retention_alarm_0_995']=cell.retention.lt(.995);atomic_csv_gzip(out/'cells.csv.gz',cell.reset_index())
    atomic_csv(out/'gene_totals.csv',pd.DataFrame(dict(gene=genes,source_total=pc.sum(0),target_total=cc.sum(0),net=net.sum(0))))
    local=[]
    for name,pop in groups.items():
        part=cell.loc[pop];sums=part[['previous_RNA','new_RNA','incoming_RNA','outgoing_RNA','net_RNA','retained_RNA']].sum();den=int(sums.previous_RNA)
        retention=float(sums.retained_RNA/den) if den else None
        local.append(dict(stratum=name,n_cells=len(part),**sums.to_dict(),pooled_retention=retention,cell_mean_retention=float(part.retention.mean()) if len(part) else None,
                          retention_alarm=retention is not None and retention<.995,NA_reason='' if den else 'no_previous_assigned_molecules'))
    atomic_csv_gzip(out/'local_costs.csv.gz',pd.DataFrame(local))
    from .local_statistics import spatial_accounts,cell_cost_intervals
    cell_cost_intervals(cfg,cell,cells,groups,out/'cell_cost_intervals',dict(dep,part='cell_cost_intervals'),callback)
    spatial_accounts(cfg,manifest,a,b,cells,out/'spatial_strata',dict(dep,part='fixed_spatial_strata'),callback)
    regrows=[];intervals=[]
    for r,v in regional.items():
        den=v['denominator'];prior=v['source_assigned'];regrows.append(dict(region=r,unit='molecule',**v,capture_delta_pp=100*(v['target_assigned']-prior)/den if den else None,
            pooled_retention=v['retained']/prior if prior else None,capture_loss_guardrail_pp=.5,local_retention_guardrail=.995,NA_reason='' if den else 'empty_fixed_domain',overlap='not additive'))
        regrows[-1].update(capture_loss_alarm=bool(den and 100*(v['target_assigned']-prior)/den < -.5),retention_alarm=bool(prior and v['retained']/prior < .995))
        for side in SCALES:
            arr=blockstats[(r,side)];p=out/'blocks'/f'{r}_{side}.npz';atomic_npz(p,values=arr,columns=np.array(['denominator','source_assigned','target_assigned','retained','cell_to_cell','cell_to_background','background_to_cell']))
            for metric,num,dn in [('capture_delta_pp',100*(arr[:,2]-arr[:,1]),arr[:,0]),('pooled_retention',arr[:,3],arr[:,1]),('gross_transfer_rate',arr[:,4]+arr[:,5]+arr[:,6],arr[:,0])]:
                intervals.append(dict(region=r,unit='molecule',scale_um=side,metric=metric,**pooled_interval(num,dn,weights[side]['W'],minimum_units=0)))
    atomic_csv(out/'regions.csv',pd.DataFrame(regrows));atomic_csv(out/'region_spatial_intervals.csv',pd.DataFrame(intervals))
    pixel_result=pixel_transfers(cfg,manifest,a,b,cells,out/'pixels',dep,callback)
    if include_structure:structure_costs(cfg,a,b,cells,groups,out/'structure',callback)
    # Per-interface transfers use the unique frozen assignment; all overlapping
    # exposures remain available in the immutable interface exposure tables.
    interface_account(cfg,manifest,a,b,out/'interfaces',dep,callback)
    atomic_json(out/'schema.json',dict(columns=list(TRANSFER_COLUMNS),gene_names=genes,background_owner=0,background_type_index=-2,unknown_type_index=-1,
         region_bits={r:1<<i for i,r in enumerate(REGIONS)},type_relation={'0':'same','1':'heterotypic','2':'Unknown','3':'background'},
         partition='chunk sparse keys sum additively across chunks; region bits expand to overlapping named memberships',
         cell_strata_source=identity(frozen/'cells.csv'),all313_genes=True,top_N_truncation=False,
         matrix_gene_cell_delta_exact=True,gross_and_net_separate=True,scientific_guardrails_block_compute=False))
    return write_receipt(out/'complete.json',dep,inventory(out),comparison=key,matrix_gene_cell_delta_exact=True,all313_gene_ledger=True,
                         sparse_chunks=len(mr['receipts']),scientific_alarms_do_not_block=True,reuse_binding=binding,structure_completed=include_structure)

def pixel_transfers(cfg,manifest,a,b,cells,out,deps,callback=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);frozen=Path(cfg['run_root'])/'frozen_domain'
    source=tifffile.imread(b['final']);target=tifffile.imread(a['final']);bits=np.load(frozen/'region_bits.npy',mmap_mode='r');rows=[]
    kinds=transfer_kind(source,target)
    for r in REGIONS:
        use=in_region(bits,r);den=int(use.sum());rows.append(dict(region=r,unit='pixel',denominator=den,**{k:int(np.count_nonzero(use&m)) for k,m in kinds.items()},
            source_foreground=int(np.count_nonzero(use&(source>0))),target_foreground=int(np.count_nonzero(use&(target>0))),NA_reason='' if den else 'empty_fixed_pixel_domain'))
    atomic_csv(out/'regions.csv',pd.DataFrame(rows))
    density=np.load(frozen/'local_density_group.npy',mmap_mode='r');K=np.load(Path(manifest['historical_strata'])/'support_counts.npy',mmap_mode='r')
    assignment=np.load(frozen/'interfaces/assignment.npy',mmap_mode='r');weights={s:dict(np.load(frozen/f'bootstrap_weights_{s}.npz')) for s in SCALES}
    # Complete ID-to-ID pixel ledger, with fixed regions and spatial location.
    for start in range(0,len(source),128):
        stop=min(start+128,len(source));ch=source[start:stop]!=target[start:stop];yy,xx=np.nonzero(ch);yy+=start
        rp=out/f'chunk_{start:05d}.json';cd=dict(deps,part='pixel_transfer',start=start,stop=stop)
        if rp.exists():check_receipt(rp,cd)
        else:
            st=dict(region_bits=bits[yy,xx],local_density=density[yy,xx],K=np.minimum(K[yy,xx],4),interface_id=assignment[yy,xx])
            for side in SCALES:st['block_'+str(side)]=block_indices(np.c_[xx,yy],weights[side]['blocks'],side)
            keys,n=sparse_transfer(source[yy,xx],target[yy,xx],np.full(len(yy),-1,np.int32),st,cells)
            path=rp.with_suffix('.npz');atomic_npz(path,keys=keys[:,1:],pixels=n);chunk_receipt(rp,cd,[path],changed_pixels=int(ch.sum()))
        progress(callback,'full_pixel_transfer_ledger',stop,len(source),receipt=str(rp))
    atomic_json(out/'schema.json',dict(columns=list(TRANSFER_COLUMNS[1:]),unit='pixel',all_changed_pixels=True,background_type_index=-2,unknown_type_index=-1))
    return rows

def structure_path(cfg,spec):
    candidate=Path(spec['evaluation'])/'structure/cells_distribution.csv'
    if candidate.exists():return candidate
    return Path(cfg['run_root'])/'reused'/spec['name']/'structure/cells_distribution.csv'

def structure_costs(cfg,a,b,cells,groups,out,callback=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    dep=dict(dependencies(cfg,'STRUCTURE_COST_'+comparison_id(a['name'],b['name'])),source=identity(structure_path(cfg,b)),target=identity(structure_path(cfg,a)))
    if (out/'complete.json').exists():return check_receipt(out/'complete.json',dep)
    x=pd.read_csv(structure_path(cfg,a)).set_index('cell_id');y=pd.read_csv(structure_path(cfg,b)).set_index('cell_id')
    if not np.array_equal(x.index,cells.index) or not np.array_equal(y.index,cells.index):raise ValueError('structure full ID set differs')
    metrics=('area_um2','components_4','components_8','transcripts_per_cell','genes_per_cell');rows=[]
    for name,pop in groups.items():
        for metric in metrics:
            d=x.loc[pop,metric]-y.loc[pop,metric]
            rows.append(dict(stratum=name,metric=metric,n_cells=len(pop),mean_source=float(y.loc[pop,metric].mean()) if len(pop) else None,
                mean_target=float(x.loc[pop,metric].mean()) if len(pop) else None,mean_delta=float(d.mean()) if len(pop) else None,
                NA_reason='' if len(pop) else 'empty_fixed_cell_group'))
    atomic_csv_gzip(out/'costs.csv.gz',pd.DataFrame(rows))
    from .spatial_bootstrap import GroupProjector,block_intervals
    projector=GroupProjector(cells,groups);intervals=[]
    for metric in metrics:
        delta=(x[metric]-y[metric]).to_numpy(float)
        for side in SCALES:
            W=np.load(Path(cfg['run_root'])/'frozen_domain'/f'bootstrap_weights_{side}.npz')['W'];s,n=projector.blocks(cells.index.to_numpy(),delta,side,W.shape[1])
            result=block_intervals(s,n,W,projector.names,checkpoint=(out/'bootstrap'/f'{metric}_{side}',dict(dep,metric=metric,scale=side,population_ids_sha256=id_hash(cells.index))),callback=callback)
            for row in result:row.update(metric=metric,scale_um=side,unit='cell')
            intervals.extend(result)
    atomic_csv_gzip(out/'intervals.csv.gz',pd.DataFrame(intervals));return write_receipt(out/'complete.json',dep,inventory(out))

def interface_account(cfg,manifest,a,b,out,deps,callback=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);frozen=Path(cfg['run_root'])/'frozen_domain';pairs=pd.read_csv(frozen/'interfaces/pairs.csv');n=len(pairs)
    values=np.zeros((n,7),np.int64);ar=read(Path(a['matrix'])/'complete.json');br=read(Path(b['matrix'])/'complete.json')
    for i,(ra,rb) in enumerate(zip(ar['receipts'],br['receipts'])):
        sa=np.load(Path(b['matrix'])/rb['owner_file']);ta=np.load(Path(a['matrix'])/ra['owner_file']);st=np.load(frozen/'molecule_strata'/f'chunk_{i:04d}.npz');group=st['interface_id'];good=group>=0;k=transfer_kind(sa,ta)
        for j,m in enumerate((np.ones(len(sa),bool),sa>0,ta>0,(sa>0)&(sa==ta),k['cell_to_cell'],k['cell_to_background'],k['background_to_cell'])):values[:,j]+=np.bincount(group[good],weights=m[good],minlength=n).astype(np.int64)
    names=['denominator','source_assigned','target_assigned','retained','cell_to_cell','cell_to_background','background_to_cell']
    for j,name in enumerate(names):pairs[name]=values[:,j]
    atomic_csv_gzip(out/'unique_assignment_molecule_transfers.csv.gz',pairs)
    # Full overlapping interface exposure ledger (all molecules/pixels), streamed
    # from frozen pair exposure chunks; no double-count claim for union totals.
    raw=np.load(frozen/'molecule_count.npy',mmap_mode='r');sm=tifffile.imread(b['final']);tm=tifffile.imread(a['final']);allrows=[]
    for p in sorted((frozen/'interfaces').glob('exposure_*.npz')):
        z=np.load(p);flat=z['pixel'];pair=z['interface'];k=transfer_kind(sm.ravel()[flat],tm.ravel()[flat])
        for unit,weight in [('pixel',np.ones(len(flat),np.int64)),('molecule',raw.ravel()[flat].astype(np.int64))]:
            vals=[np.bincount(pair,weights=weight,minlength=n)]
            vals+=[np.bincount(pair,weights=weight*m,minlength=n) for m in k.values()]
            active=np.unique(pair)
            for i in active:allrows.append(dict(interface_id=int(i),unit=unit,denominator=int(vals[0][i]),**{key:int(vals[j+1][i]) for j,key in enumerate(k)}))
    atomic_csv_gzip(out/'overlapping_exposure_transfers.csv.gz',pd.DataFrame(allrows,columns=['interface_id','unit','denominator','unchanged_cell','background_unchanged','cell_to_cell','cell_to_background','background_to_cell']))
    from .spatial_bootstrap import GroupProjector,block_intervals
    from .local_statistics import _statistics
    cells=pd.read_csv(frozen/'cells.csv').set_index('cell_id');groupfile=np.load(frozen/'cell_groups.npz')
    groups={'all':pairs.interface_id.to_numpy(np.int64)}
    for kind in ('same','heterotypic','Unknown'):groups['type_'+kind]=pairs.loc[pairs.type_class==kind,'interface_id'].to_numpy(np.int64)
    for key in groupfile.files:
        pop=groupfile[key];groups['pair_exposed_to_'+key]=pairs.loc[pairs.cell_i.isin(pop)|pairs.cell_j.isin(pop),'interface_id'].to_numpy(np.int64)
    geometry=pairs.set_index('interface_id');weights={side:dict(np.load(frozen/f'bootstrap_weights_{side}.npz')) for side in SCALES}
    for side in SCALES:geometry['block_'+str(side)]=block_indices(pairs[['midpoint_x_um','midpoint_y_um']].to_numpy(float),weights[side]['blocks'],side)
    projector=GroupProjector(geometry,groups);intervals=[]
    for unit in ('pixel','molecule'):
        frame=pd.DataFrame(allrows)
        table=(frame[frame.unit==unit].set_index('interface_id').reindex(geometry.index).fillna(0) if len(frame) else pd.DataFrame(0,index=geometry.index,columns=['denominator','unchanged_cell','cell_to_cell','cell_to_background','background_to_cell']))
        den=table.denominator.to_numpy(float);source=table.unchanged_cell+table.cell_to_cell+table.cell_to_background;target=table.unchanged_cell+table.cell_to_cell+table.background_to_cell
        measures={'capture_delta_pp':(100*(target-source).to_numpy(float),den),'owner_retention':(table.unchanged_cell.to_numpy(float),source.to_numpy(float)),
                  'gross_transfer_rate':((table.cell_to_cell+table.cell_to_background+table.background_to_cell).to_numpy(float),den)}
        for metric,(num,dn) in measures.items():
            observed=np.array([np.count_nonzero(dn[np.searchsorted(geometry.index.to_numpy(),pop)]>0) for pop in groups.values()])
            for side in SCALES:
                W=weights[side]['W'];s,n=projector.blocks(geometry.index.to_numpy(),num,side,W.shape[1],dn)
                result=block_intervals(s,n,W,projector.names,minimum_units=20,observations=observed,
                    checkpoint=(out/'bootstrap'/f'{unit}_{metric}_{side}',dict(deps,part='interface_pair_statistics',unit=unit,metric=metric,scale=side,interface_ids_sha256=id_hash(geometry.index))),callback=callback)
                for row in result:row.update(unit='interface_pair',exposure_unit=unit,metric=metric,scale_um=side,overlap='all pair exposures; not additive as union counts')
                intervals.extend(result)
    atomic_csv_gzip(out/'paired_spatial_intervals.csv.gz',pd.DataFrame(intervals))
    atomic_json(out/'semantics.json',dict(unit='interface',unique_assignment='minimum(distance_squared,i,j)',overlap_exposures_retained=True,overlap_not_additive=True,empty_pairs=n==0))

def run(cfg,manifest,specs,out,callback=None,reuse_from=None,include_structure=True):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);deps=dependencies(cfg,'BOUNDARY_'+str(len(specs)))
    if (out/'complete.json').exists():return check_receipt(out/'complete.json',deps)
    reused=[];computed=[]
    for name,spec in specs.items():
        dest=out/'methods'/name;source=Path(reuse_from)/'methods'/name if reuse_from else None
        if source is not None and (source/'complete.json').exists() and not (dest/'complete.json').exists():
            reuse_partition(source,dest,dependencies(cfg,'REGIONS_'+name),endpoint_binding(cfg,spec));reused.append('method:'+name)
        else:endpoint_regions(cfg,manifest,spec,dest,callback);computed.append('method:'+name)
    pairs=comparisons(tuple(specs))
    for i,(a,b) in enumerate(pairs):
        p=out/'comparisons'/comparison_id(a,b);source=Path(reuse_from)/'comparisons'/comparison_id(a,b) if reuse_from else None
        pair_transfers(cfg,manifest,specs[a],specs[b],p,callback,source,include_structure)
        (reused if source is not None and (source/'complete.json').exists() else computed).append('pair:'+comparison_id(a,b))
        progress(callback,'fixed_boundary_comparison',i+1,len(pairs),receipt=str(p/'complete.json'))
    atomic_json(out/'coverage.json',dict(methods=list(specs),pairs=[comparison_id(a,b) for a,b in pairs],regions=list(REGIONS),units=['pixel','molecule','cell','interface'],full313_sparse_transfers=True,
        reused=reused,computed=computed,structure_costs_complete=include_structure,pending=[] if include_structure else ['full_4_and_8_structure_local_costs_until_stage12']))
    return write_receipt(out/'complete.json',deps,inventory(out),methods=list(specs),pairs=[comparison_id(a,b) for a,b in pairs])
