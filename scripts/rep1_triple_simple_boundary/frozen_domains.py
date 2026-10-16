"""Outcome-independent BASE domains, all fixed strata, ROI and spatial weights."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import tifffile
import yaml
import h5py
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scripts.boundary_three.core_refine import NuclearSupport,interface_band
from .contracts import NEW,SCALES,REGIONS,N_CELLS,N_GENES,ELIGIBLE
from .common import (read,identity,digest,check_identity,check_receipt,within,inventory,write_receipt,
                     atomic_json,atomic_npy,atomic_npz,atomic_csv,chunk_receipt,progress,id_hash)
from .inputs import matrix_arrays

TYPE_CLASSES=('same','heterotypic','Unknown')
CELL_DIMENSIONS=('BASE_count_quartile','RNA_density_quartile','nuclear_crowding_quartile',
                 'nuclear_area_quartile','eccentricity_quartile','nuclear_components_group')

def quartiles(values,edges=None):
    x=np.asarray(values,float);ok=np.isfinite(x)
    e=np.unique(np.quantile(x[ok],[.25,.5,.75],method='linear')) if edges is None and ok.any() else np.asarray([] if edges is None else edges,float)
    g=np.full(x.shape,-1,np.int8);g[ok]=np.searchsorted(e,x[ok],side='left')
    return e,g

def type_class(unknown,heterotypic):
    """Unknown takes priority even when two other known types disagree."""
    return np.where(unknown,2,np.where(heterotypic,1,0)).astype(np.int8)

def region_arrays(base,nuc,support_count,interface):
    ext=nuc==0;c=ext&(support_count>=2);i=ext&interface
    return dict(all=np.ones(base.shape,bool),competition=c,interface=i,nuclear=nuc>0,
                noncompetitive=ext&(support_count<2),interface_positive=i&(base>0),
                competition_interface=c&i,competition_only=c&~i,interface_only=i&~c,
                oob=np.zeros(base.shape,bool))

def region_bits(regions):
    bits=np.zeros(next(iter(regions.values())).shape,np.uint16)
    for j,name in enumerate(REGIONS):bits[regions[name]]|=np.uint16(1<<j)
    return bits

def in_region(bits,name):
    return (np.asarray(bits)&(1<<REGIONS.index(name)))!=0

def block_indices(xy,blocks,side,allow_oob=False):
    xy=np.asarray(xy);grid=np.floor(xy/side).astype(np.int64)
    lookup={tuple(k):i for i,k in enumerate(np.asarray(blocks).tolist())}
    # Native grid is compact. LUT avoids a Python operation for every molecule.
    mx=max(int(np.max(blocks[:,0],initial=0))+1,1);my=max(int(np.max(blocks[:,1],initial=0))+1,1)
    lut=np.full((mx,my),-1,np.int16)
    for k,i in lookup.items():lut[k]=i
    valid=(xy[:,0]>=0)&(xy[:,1]>=0)&(grid[:,0]<mx)&(grid[:,1]<my)
    result=np.full(len(xy),-1,np.int16);result[valid]=lut[grid[valid,0],grid[valid,1]]
    if np.any(result[valid]<0) or (not allow_oob and np.any(~valid)):
        raise ValueError('coordinate not covered by frozen nuclear block universe')
    return result

def bootstrap_weights(centroids,side,replicates=2000,seed=20261013):
    blocks,index=np.unique(np.floor(np.asarray(centroids)/side).astype(np.int64),axis=0,return_inverse=True)
    rng=np.random.Generator(np.random.PCG64(seed));n=len(blocks)
    W=np.zeros((replicates,n),np.int16)
    for r in range(replicates):W[r]=np.bincount(rng.integers(0,n,size=n),minlength=n)
    return blocks,index,W

def local_density(counts,radius=10,stripe=128):
    """Closed disk and cropped image denominator; stripe halo is exactly radius."""
    yy,xx=np.mgrid[-radius:radius+1,-radius:radius+1];kernel=(yy*yy+xx*xx<=radius*radius).astype(np.float64)
    out=np.empty(counts.shape,np.float32)
    for start in range(0,len(counts),stripe):
        stop=min(start+stripe,len(counts));lo=max(0,start-radius);hi=min(len(counts),stop+radius)
        part=np.asarray(counts[lo:hi],np.float64)
        num=ndi.convolve(part,kernel,mode='constant',cval=0)
        den=ndi.convolve(np.ones(part.shape,np.float64),kernel,mode='constant',cval=0)
        out[start:stop]=(num/den)[start-lo:stop-lo]
    return out

def interface_edges(base):
    h,w=base.shape;pieces=[]
    for dy,dx in ((0,1),(1,-1),(1,0),(1,1)):
        ya=slice(max(0,-dy),min(h,h-dy));xa=slice(max(0,-dx),min(w,w-dx))
        yb=slice(max(0,dy),min(h,h+dy));xb=slice(max(0,dx),min(w,w+dx))
        a,b=base[ya,xa],base[yb,xb];y,x=np.nonzero((a>0)&(b>0)&(a!=b))
        i,j=a[y,x],b[y,x];gy=y+ya.start;gx=x+xa.start
        pieces.append(np.column_stack((np.minimum(i,j),np.maximum(i,j),gy,gx,gy+dy,gx+dx)).astype(np.int32))
    e=np.concatenate(pieces) if pieces else np.empty((0,6),np.int32)
    if len(e):e=e[np.lexsort((e[:,5],e[:,4],e[:,3],e[:,2],e[:,1],e[:,0]))]
    return e

def build_interfaces(base,nuc,types_by_id,out,deps,callback=None):
    """All pair exposures retained, unique ownership=min(distance²,i,j)."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():
        check_receipt(out/'complete.json',deps)
        return {k:np.load(out/(k+'.npy'),mmap_mode='r') for k in ('assignment','multiplicity','any_unknown','known_heterotypic','multiple_known_types')},pd.read_csv(out/'pairs.csv')
    for committed_path in sorted(out.glob('exposure_*.json')):
        check_receipt(committed_path,dict(deps,start=int(committed_path.stem.split('_')[-1])))
    edges=interface_edges(base);atomic_npz(out/'edges.npz',edges=edges)
    pair,starts,counts=np.unique(edges[:,:2],axis=0,return_index=True,return_counts=True) if len(edges) else (np.empty((0,2),int),np.array([],int),np.array([],int))
    h,w=base.shape;assignment=np.full(h*w,-1,np.int32);best=np.full(h*w,255,np.uint8)
    mult=np.zeros(h*w,np.uint16);unknown=np.zeros(h*w,bool);hetero=np.zeros(h*w,bool)
    first_type=np.full(h*w,-2,np.int16);multiple_types=np.zeros(h*w,bool)
    offsets=[(dy,dx,dy*dy+dx*dx) for dy in range(-2,3) for dx in range(-2,3) if dy*dy+dx*dx<=4]
    rows=[];flatnuc=nuc.ravel()
    for begin in range(0,len(pair),1000):
        # Validate committed exposure before regenerating deterministic state;
        # never overwrite corruption and thereby make an old receipt match.
        existing=out/f'exposure_{begin:06d}.json'
        saved=None
        if existing.exists():
            check_receipt(existing,dict(deps,start=begin));saved=dict(np.load(existing.with_suffix('.npz')))
        pp=[];ii=[];dd=[]
        for k in range(begin,min(begin+1000,len(pair))):
            e=edges[starts[k]:starts[k]+counts[k]];points=np.unique(np.concatenate((e[:,2:4],e[:,4:6])),axis=0)
            if saved is not None:
                left,right=np.searchsorted(saved['interface'],[k,k+1]);f=saved['pixel'][left:right];d=saved['distance_squared'][left:right]
            else:
                flat=[];dist=[]
                for dy,dx,d2 in offsets:
                    y,x=points[:,0]+dy,points[:,1]+dx;ok=(y>=0)&(x>=0)&(y<h)&(x<w)
                    f=y[ok].astype(np.int64)*w+x[ok];ok2=flatnuc[f]==0;f=f[ok2]
                    flat.append(f);dist.append(np.full(len(f),d2,np.uint8))
                f=np.concatenate(flat);d=np.concatenate(dist)
                order=np.lexsort((d,f));f,d=f[order],d[order];unique=np.r_[True,f[1:]!=f[:-1]] if len(f) else np.array([],bool)
                f,d=f[unique],d[unique]
            i,j=map(int,pair[k]);ti,tj=types_by_id[i],types_by_id[j]
            wins=d<best[f] # Pairs iterated lexicographically, equal distance retains minimum (i,j).
            assignment[f[wins]]=k;best[f[wins]]=d[wins];mult[f]+=1
            unknown[f]|=(ti<0 or tj<0);hetero[f]|=(ti>=0 and tj>=0 and ti!=tj)
            for typ in (ti,tj):
                if typ>=0:
                    old=first_type[f];multiple_types[f]|=(old>=0)&(old!=typ);first_type[f]=np.where(old==-2,typ,old)
            pp.append(f.astype(np.int32));ii.append(np.full(len(f),k,np.int32));dd.append(d)
            rows.append(dict(interface_id=k,cell_i=i,cell_j=j,contact_edges=int(counts[k]),exposure_pixels=len(f),
                             type_class='Unknown' if min(ti,tj)<0 else 'same' if ti==tj else 'heterotypic',
                             midpoint_x_um=float(np.mean(np.r_[e[:,3],e[:,5]])),midpoint_y_um=float(np.mean(np.r_[e[:,2],e[:,4]]))))
        p=out/f'exposure_{begin:06d}.npz';r=p.with_suffix('.json')
        if not r.exists():atomic_npz(p,pixel=np.concatenate(pp),interface=np.concatenate(ii),distance_squared=np.concatenate(dd))
        chunk_receipt(r,dict(deps,start=begin),[p],unit='interface_pixel_exposure',overlap='all pairs retained; not additive as domain counts')
        progress(callback,'freeze_interface_exposure',min(begin+1000,len(pair)),len(pair),receipt=str(r))
    arrays=dict(assignment=assignment.reshape(base.shape),multiplicity=mult.reshape(base.shape),any_unknown=unknown.reshape(base.shape),known_heterotypic=hetero.reshape(base.shape),multiple_known_types=multiple_types.reshape(base.shape))
    for k,a in arrays.items():atomic_npy(out/(k+'.npy'),a)
    table=pd.DataFrame(rows,columns=['interface_id','cell_i','cell_j','contact_edges','exposure_pixels','type_class','midpoint_x_um','midpoint_y_um'])
    atomic_csv(out/'pairs.csv',table)
    if not np.array_equal(assignment.reshape(base.shape)>=0,interface_band(base)&(nuc==0)):
        raise ValueError('pair-specific exposure union differs from BASE global 2um band')
    write_receipt(out/'complete.json',deps,inventory(out),pairs=len(pair),edges=len(edges),tie='distance_squared,i,j',unit='interface')
    return arrays,table

def membership_groups(base,regions,competition_type,interface_type,K,local_group,cells,single_unknown=None):
    """BASE-owned full-cell spectrum groups, permitting documented overlap."""
    ids=cells.index.to_numpy(np.int64);result={};definitions={}
    def add(name,chosen,definition):
        chosen=np.asarray(chosen,np.int64);chosen=np.unique(chosen[chosen>0]);result[name]=chosen
        definitions[name]=dict(unit='cell',n=len(chosen),ids_sha256=id_hash(chosen),definition=definition,overlap=True)
    for region,mask in regions.items():
        selected=np.unique(base[mask]);add(region,selected,'exists BASE-owned pixel in '+region)
        if region=='oob':continue
        typ=interface_type if region.startswith('interface') else competition_type
        for tc,label in enumerate(TYPE_CLASSES):
            local=np.unique(base[mask&(typ==tc)]);name=region+'__type_'+label;add(name,local,'fixed pixel type projected by BASE owner')
            for dim in CELL_DIMENSIONS:
                for group in (-1,0,1,2,3):
                    vals=cells.index[cells[dim].eq(group)];add(name+'__'+dim+'_'+str(group),np.intersect1d(local,vals),'fixed cell property intersection')
        for k in range(5):add(region+'__K_'+str(k),base[mask&(np.minimum(K,4)==k)],'frozen SearchArea support multiplicity, 4 means >=4')
        for q in range(4):add(region+'__local_density_'+str(q),base[mask&(local_group==q)],'fixed local RNA density projected by BASE owner')
        if single_unknown is not None:add(region+'__single_unknown',base[mask&single_unknown],'one frozen Unknown support; separate from competition Unknown')
        for dim in CELL_DIMENSIONS:
            for group in (-1,0,1,2,3):add(region+'__'+dim+'_'+str(group),np.intersect1d(selected,cells.index[cells[dim].eq(group)]),'fixed property intersect region membership')
    for dim in ('original_type',*CELL_DIMENSIONS,*['block_'+str(s) for s in SCALES]):
        for value,part in cells.groupby(dim,dropna=False):add('cell_'+dim+'_'+str(value),part.index,'frozen cell marginal')
    return result,definitions

def freeze(run_root,manifest,source_digest,progress=None):
    """Prepare only from the approved old BASE/support inputs, before any new final."""
    out=within(run_root,'frozen_domain');dep=dict(schema='rep1-frozen-boundary-v1',input_digest=digest(manifest),source_digest=source_digest)
    if (out/'complete.json').exists():return check_receipt(out/'complete.json',dep,run_root)
    if any(within(run_root,'endpoints',n,'final.tif').exists() for n in NEW):raise ValueError('new result exists before fixed-domain freeze')
    if set(manifest['endpoints'])-set(__import__(__package__,fromlist=['EXISTING']).EXISTING):raise ValueError('new endpoints forbidden in domain input')
    out.mkdir(parents=True,exist_ok=True);spec=manifest['endpoints']['BASE_MATCH'];machine=read(spec['machine'])
    # Validate every committed child before any deterministic reconstruction can
    # replace an uncommitted output. Corrupt committed evidence is never hidden.
    for p in sorted(out.rglob('*.json')):
        obj=read(p)
        if isinstance(obj,dict) and obj.get('status')=='succeeded' and obj.get('outputs'):check_receipt(p)
    base=tifffile.imread(spec['final']);nuc=tifffile.imread(spec['nuclei']);ids,genes,counts=matrix_arrays(spec)
    meta=np.load(manifest['metadata']);types=meta['type_index'].astype(int);ref=read(manifest['reference_region'])
    if not np.array_equal(ids,meta['ids']):raise ValueError('freeze metadata/matrix ID mismatch')
    with h5py.File(machine['preannotation']) as f:aid=f['ids'][:];at=f['data'][:]
    order=np.argsort(aid)
    if not np.array_equal(aid[order],ids) or not np.array_equal(at[order],types):raise ValueError('freeze engineering type/preannotation mismatch')
    settings=yaml.safe_load(Path(machine['baseline_config']).read_text())
    support=NuclearSupport(nuc,ids,types,ref['type_names'],settings['model_params']['elongated'])
    if not np.array_equal(support.area,meta['area']) or not np.allclose(support.centroids,meta['centroid_xy'],atol=1e-8,rtol=0):raise ValueError('complete nuclear metadata differs')
    historic=Path(manifest['historical_strata']);K=np.load(historic/'support_counts.npy',mmap_mode='r')
    unk=np.load(historic/'any_unknown.npy',mmap_mode='r');het=np.load(historic/'heterotypic.npy',mmap_mode='r')
    Ctype=type_class(unk,het);Ctype[K<2]=-1;regions=region_arrays(base,nuc,K,interface_band(base));bits=region_bits(regions)
    # Reconstruct old SearchArea counts/types from the literal support implementation.
    kcheck=np.zeros(base.shape,np.uint32);ucheck=np.zeros(base.shape,bool);hcheck=np.zeros(base.shape,bool);first=np.full(base.shape,-2,np.int16)
    exposures={r:[] for r in REGIONS if r!='oob'};fallback=oob=0
    for begin in range(0,len(ids),1000):
        end=min(begin+1000,len(ids));p=out/'support_progress'/f'{begin:06d}.json';data=p.with_suffix('.npz');cd=dict(dep,start=begin,stop=end)
        if p.exists():rec=check_receipt(p,cd)
        else:
            flat=[];offset=[0];chunk_oob=chunk_fallback=0
            for cid in ids[begin:end]:
                y,x,_,fb=support.support(int(cid));valid=(y>=0)&(x>=0)&(y<len(base))&(x<base.shape[1])
                chunk_oob+=int((~valid).sum());chunk_fallback+=int(fb)
                f=(y[valid].astype(np.int64)*base.shape[1]+x[valid]).astype(np.int32);flat.append(f);offset.append(offset[-1]+len(f))
            atomic_npz(data,pixel=np.concatenate(flat),offsets=np.asarray(offset,np.int64))
            rec=chunk_receipt(p,cd,[data],ellipse_fallbacks=chunk_fallback,out_of_bounds=chunk_oob,nuclei=end-begin)
        saved=dict(np.load(data));fallback+=rec['ellipse_fallbacks'];oob+=rec['out_of_bounds']
        for index,row in enumerate(range(begin,end)):
            cid=ids[row];f=saved['pixel'][saved['offsets'][index]:saved['offsets'][index+1]];y,x=np.divmod(f,base.shape[1]);t=types[row]
            kcheck[y,x]+=1
            if t<0:ucheck[y,x]=True
            else:
                seen=first[y,x];hcheck[y,x]|=(seen>=0)&(seen!=t);first[y,x]=np.where(seen==-2,t,seen)
            for r in exposures:
                if np.any(regions[r][y,x]):exposures[r].append(int(cid))
        globals()['progress'](progress,'freeze_complete_nuclear_support',end,len(ids),receipt=str(p))
    hs=read(historic/'strata.json')
    if not np.array_equal(kcheck,K) or not np.array_equal(ucheck,unk) or not np.array_equal(hcheck,het) or not np.array_equal(first,np.load(historic/'first_known_type.npy')) or (oob,fallback)!=(hs['support_oob_pixels_before_crop'],hs['support_ellipse_fallback_nuclei']):
        raise ValueError('historical support raster/algorithm reconstruction differs')
    del kcheck,ucheck,hcheck,first
    types_by_id=np.full(int(ids.max())+1,-2,np.int16);types_by_id[ids]=types
    interfaces,pairs=build_interfaces(base,nuc,types_by_id,out/'interfaces',dict(dep,part='BASE_interfaces'),progress)
    Itype=np.full(base.shape,-1,np.int8);assignment=interfaces['assignment'];valid=assignment>=0
    if len(pairs):Itype[valid]=pd.Categorical(pairs.type_class,categories=TYPE_CLASSES).codes[assignment[valid]]
    # Complete nuclear second moments and 4/8 components are validated independently.
    moments=[]
    for row,cid in enumerate(ids):
        y,x=support.nucleus(int(cid));xy=np.c_[x,y].astype(float);delta=xy-xy.mean(0);cov=delta.T@delta/len(xy)
        eig=np.linalg.eigvalsh(cov);ecc=np.sqrt(max(0,1-eig[0]/eig[1])) if eig[1]>0 else np.nan
        box=support.boxes[int(cid)-1];ncomp=ndi.label(nuc[box]==cid,np.ones((3,3)))[1]
        moments.append((ecc,ncomp))
        if not np.allclose(cov,meta['covariance'][row],rtol=0,atol=1e-8):raise ValueError('complete nuclear second moments differ')
    vals=np.asarray(moments);tot=counts.sum(1,dtype=np.int64);area=np.bincount(base.ravel(),minlength=int(ids.max())+1)[ids]
    crowd=cKDTree(support.centroids).query_ball_point(support.centroids,30,return_length=True,workers=1)-1
    cells=pd.DataFrame(dict(cell_id=ids,original_type_index=types,original_type=[ref['type_names'][t] if t>=0 else 'Unknown' for t in types],
                            x_um=support.centroids[:,0],y_um=support.centroids[:,1],BASE_count=tot,BASE_area=area,RNA_density=tot/area,
                            nuclear_area=support.area,eccentricity=vals[:,0],nuclear_components=vals[:,1].astype(int),
                            nuclear_components_group=(vals[:,1]>1).astype(np.int8),nuclear_crowding=crowd)).set_index('cell_id')
    threshold={}
    for field,dim in [('BASE_count','BASE_count_quartile'),('RNA_density','RNA_density_quartile'),('nuclear_crowding','nuclear_crowding_quartile'),('nuclear_area','nuclear_area_quartile'),('eccentricity','eccentricity_quartile')]:
        edge,group=quartiles(cells[field]);cells[dim]=group;threshold[dim]=edge.tolist()
    old=pd.read_csv(historic/'fixed_cell_groups.csv');oldc=pd.read_csv(Path(manifest['historical_counts'])/'fixed_count_groups.csv')
    if threshold['BASE_count_quartile']!=[78,140,272] or threshold['nuclear_crowding_quartile']!=[8,11,14] or not np.array_equal(cells.BASE_count_quartile,oldc.count_quartile) or not np.array_equal(crowd,old.density_30um):raise ValueError('historical BASE count/crowding strata differ')
    fixed=pd.read_csv(manifest['fixed_population']).set_index('cell_id');cells['fixed_B1_label']=fixed.evaluation_label.reindex(ids);cells['fixed_B1_unmatched_reason']=fixed.unmatched_reason.reindex(ids)
    for side in SCALES:
        blocks,index,W=bootstrap_weights(support.centroids,side);cells['block_'+str(side)]=index
        atomic_npz(out/f'bootstrap_weights_{side}.npz',blocks=blocks,W=W)
    atomic_csv(out/'cells.csv',cells,index=True)
    from .bt import freeze_targets
    freeze_targets(cells,fixed,out/'bt_targets',dict(dep,part='fixed_BT_targets'))
    raw=np.zeros(base.shape,np.uint32);cursor=0
    for i,rec in enumerate(manifest['coordinates']):
        check_identity(rec);a=np.load(rec['path'],mmap_mode='r');good=(a['x']>=0)&(a['y']>=0)
        if rec['row_start']!=cursor:raise ValueError('freeze molecule row coverage mismatch')
        np.add.at(raw,(a['y'][good],a['x'][good]),1);cursor+=len(a)
    if cursor!=ELIGIBLE:raise ValueError('freeze eligible denominator differs')
    atomic_npy(out/'molecule_count.npy',raw);density=local_density(raw)
    union=regions['competition']|regions['interface'];edges,_=quartiles(density[union]);_,localgroup=quartiles(density,edges)
    if not union.any():localgroup[:]=-1
    threshold['local_RNA_density_quartile']=edges.tolist()
    atomic_npy(out/'local_RNA_density.npy',density);atomic_npy(out/'local_density_group.npy',localgroup)
    atomic_npy(out/'region_bits.npy',bits);atomic_npy(out/'competition_type.npy',Ctype);atomic_npy(out/'interface_type.npy',Itype)
    atomic_npy(out/'competition_known_heterotypy_with_unknown.npy',unk&het)
    atomic_npy(out/'single_unknown.npy',(K==1)&unk)
    groups,definitions=membership_groups(base,regions,Ctype,Itype,K,localgroup,cells,(K==1)&unk)
    for r,selected in exposures.items():
        name=r+'__support_exposure';groups[name]=np.asarray(selected,np.int64)
        definitions[name]=dict(unit='cell',n=len(selected),ids_sha256=id_hash(selected),definition='any complete nuclear SearchArea support in '+r,overlap=True,supplementary=True)
    atomic_npz(out/'cell_groups.npz',**groups);atomic_json(out/'cell_groups.json',definitions)
    memberships=[]
    for r in REGIONS:
        for cid in groups[r]:memberships.append((r,'BASE_owner',int(cid)))
        for cid in exposures.get(r,[]):memberships.append((r,'support_exposure',int(cid)))
    atomic_csv(out/'cell_memberships.csv',pd.DataFrame(memberships,columns=['region','membership','cell_id']))
    molecule_den={r:0 for r in REGIONS};Kcap=np.minimum(K,4)
    for i,rec in enumerate(manifest['coordinates']):
        rp=out/'molecule_strata'/f'chunk_{i:04d}.json'
        if rp.exists():check_receipt(rp,dict(dep,coordinate=rec))
        a=np.load(rec['path'],mmap_mode='r');good=(a['x']>=0)&(a['y']>=0);y,x=a['y'][good],a['x'][good]
        mbits=np.full(len(a),np.uint16((1<<REGIONS.index('all'))|(1<<REGIONS.index('oob'))),np.uint16);mbits[good]=bits[y,x]
        arr=dict(region_bits=mbits,BASE_owner=np.zeros(len(a),np.uint32),local_density=np.full(len(a),-1,np.int8),competition_type=np.full(len(a),-1,np.int8),
                 interface_type=np.full(len(a),-1,np.int8),K=np.zeros(len(a),np.uint8),interface_id=np.full(len(a),-1,np.int32),single_unknown=np.zeros(len(a),bool))
        for key,grid in [('BASE_owner',base),('local_density',localgroup),('competition_type',Ctype),('interface_type',Itype),('K',Kcap),('interface_id',assignment)]:arr[key][good]=grid[y,x]
        arr['single_unknown'][good]=(K[y,x]==1)&unk[y,x]
        for side in SCALES:
            z=np.load(out/f'bootstrap_weights_{side}.npz');arr['block_'+str(side)]=block_indices(np.c_[a['x'],a['y']],z['blocks'],side,True)
        for r in REGIONS:molecule_den[r]+=int(in_region(mbits,r).sum())
        p=out/'molecule_strata'/f'chunk_{i:04d}.npz'
        if not rp.exists():atomic_npz(p,**arr)
        chunk_receipt(rp,dict(dep,coordinate=rec),[p],rows=len(a),unit='molecule')
        globals()['progress'](progress,'freeze_molecule_strata',i+1,len(manifest['coordinates']),receipt=str(rp))
    denominators=[]
    for r in REGIONS:
        for unit,n in [('pixel',int(regions[r].sum())),('molecule',molecule_den[r]),('cell',len(groups[r]))]:denominators.append(dict(region=r,unit=unit,denominator=n,overlap_policy='N,C,O partition; interface overlay; all includes OOB molecules'))
    atomic_csv(out/'denominators.csv',pd.DataFrame(denominators))
    from .roi import select_rois
    selections=select_rois(regions,Ctype,Itype,localgroup)
    atomic_json(out/'roi_selection.json',selections)
    from .roi import freeze_raw_transcripts
    freeze_raw_transcripts(manifest,selections,out/'roi_transcripts',dict(dep,part='original_ROI_transcripts'),progress)
    dapi=tifffile.imread(manifest['dapi']);lo,hi=np.quantile(np.log1p(dapi.astype(float)),[.01,.99],method='linear')
    info=dict(regions=list(REGIONS),unit_definitions={'pixel':'native 1um grid','molecule':'all313 original eligible transcript row','cell':'BASE membership and full expression spectrum','interface':'unordered positive BASE owner pair'},
              thresholds=threshold,unknown_priority=True,source_roles=['BASE_final','complete_nuclei','original_engineering_types','BASE_matrix','raw_coordinates','fixed_B1_panel'],
              support_inputs={k:manifest['files'][str(Path(v).resolve())] for k,v in [('mask',spec['nuclei']),('metadata',manifest['metadata'])]},
              support_reconstructed_exact=True,support_oob=oob,support_ellipse_fallback=fallback,
              input_dapi=identity(manifest['dapi']),dapi_log1p_limits=[float(lo),float(hi)],
              expected_molecules=ELIGIBLE,pairs=len(pairs),ROI=selections['summary'],
              bootstrap=dict(scales=list(SCALES),replicates=2000,seed=20261013,origin=[0,0],same_global_weights_all_methods=True),
              input_selection='historical frozen inputs prespecified before four new results, not unseen-history claim')
    atomic_json(out/'regions.json',info)
    return write_receipt(out/'complete.json',dep,inventory(out),**info)

def validate_frozen(cfg):
    dep=dict(schema='rep1-frozen-boundary-v1',input_digest=cfg['input_digest'],source_digest=cfg['source_digest'])
    p=within(cfg['run_root'],'frozen_domain/complete.json');rec=check_receipt(p,dep,cfg['run_root'])
    if identity(p)['sha256']!=cfg['domain_digest']:raise ValueError('fixed-domain digest mismatch')
    return rec
