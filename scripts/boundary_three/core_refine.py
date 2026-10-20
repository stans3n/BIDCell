"""Complete nuclear support, fixed A2 domain, and exact global-cell topology.

No Evidence object is constructed: DAPI and training raster counts are irrelevant.
"""
from __future__ import annotations
import functools
import math
import os
from pathlib import Path
import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy import sparse
from . import common
from .core_scores import CORE_EDT, MARGIN, BETA

CONNECTIVITY=np.ones((3,3),np.uint8)


def atomic_npy(path, array):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    with tmp.open('wb') as f:np.save(f,array,allow_pickle=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def atomic_npz(path, **arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def support_kernel(mask,elongated):
    """Literal frozen new_methods.data.support_kernel, without importing Evidence."""
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(20,20));fallback=False
    if elongated:
        try:
            contours=cv2.findContours(mask.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
            _,axes,orientation=cv2.fitEllipse(np.squeeze(contours[0]))
            eccentricity=np.sqrt(1-(min(axes)/max(axes))**2)
            kh=int(round(.9*eccentricity*60));kv=max(60-kh,3)
            kernel=ndi.rotate(cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kh,kv)),90-orientation,reshape=True)
        except Exception:fallback=True
    return kernel,fallback


class NuclearSupport:
    def __init__(self,nuclei,ids,types,type_names,elongated=()):
        self.nuclei=np.asarray(nuclei);self.ids=np.asarray(ids,np.int64);self.types=np.asarray(types,int)
        self.names=list(type_names);self.elongated=set(elongated);self.boxes=ndi.find_objects(self.nuclei)
        self.row={int(cid):i for i,cid in enumerate(ids)}
        actual=np.unique(nuclei);actual=actual[actual>0]
        if not np.array_equal(actual,self.ids) or len(np.unique(ids))!=len(ids):raise ValueError('complete nuclear ID mismatch')
        self.area=np.zeros(len(ids),np.int64);self.centroids=np.zeros((len(ids),2),float)
        for i,cid in enumerate(ids):
            y,x=self.nucleus(int(cid));self.area[i]=len(y);self.centroids[i]=[x.mean(),y.mean()]
        self.radius=np.maximum(np.sqrt(self.area/np.pi),.5)

    @functools.lru_cache(maxsize=256)
    def nucleus(self,cid):
        box=self.boxes[cid-1]
        if box is None:raise ValueError('missing complete nucleus')
        y,x=np.where(self.nuclei[box]==cid)
        return y+box[0].start,x+box[1].start

    def core_image(self):
        out=np.zeros_like(self.nuclei);area=np.zeros(len(self.ids),np.int64)
        for i,cid in enumerate(self.ids):
            box=self.boxes[cid-1];mask=self.nuclei[box]==cid
            # Explicit zero halo, including specimen boundary; another ID is exterior.
            inside=ndi.distance_transform_edt(np.pad(mask,1))[1:-1,1:-1]>=CORE_EDT
            out[box][inside]=cid;area[i]=int(inside.sum())
        return out,area

    @functools.lru_cache(maxsize=256)
    def support(self,cid):
        y,x=self.nucleus(cid);typ=int(self.types[self.row[cid]])
        elongated=typ>=0 and self.names[typ] in self.elongated;pad=80
        while True:
            y0=int(y.min())-pad;x0=int(x.min())-pad
            mask=np.zeros((int(np.ptp(y))+1+2*pad,int(np.ptp(x))+1+2*pad),np.uint8)
            mask[y-y0,x-x0]=1;kernel,fallback=support_kernel(mask,elongated)
            sup=cv2.dilate(mask,kernel,iterations=1)>0
            if not (sup[[0,-1],:].any() or sup[:,[0,-1]].any()):break
            pad*=2
            if pad>1024:raise RuntimeError('global support halo failed')
        sy,sx=np.where(sup)
        return sy+y0,sx+x0,kernel,fallback

    def candidate_csr(self,pixels,notify=None):
        """Exact support membership on requested native pixels; no centroid truncation."""
        pixels=np.asarray(pixels,np.int64)
        if len(pixels)>1 and np.any(pixels[1:]<=pixels[:-1]):raise ValueError('query pixels not sorted unique')
        h,w=self.nuclei.shape;ii=[];rr=[];oob=0;fallback=0
        if not len(pixels):return np.zeros(1,np.int64),np.empty(0,np.int32),dict(maximum_K=0,support_oob_pixels=0,ellipse_fallbacks=0)
        for row,cid in enumerate(self.ids):
            y,x,_,fb=self.support(int(cid));inside=(y>=0)&(x>=0)&(y<h)&(x<w)
            oob+=int((~inside).sum());fallback+=int(fb);flat=y[inside].astype(np.int64)*w+x[inside]
            where=np.searchsorted(pixels,flat);keep=where<len(pixels)
            where=where[keep];flat=flat[keep];where=where[pixels[where]==flat]
            if len(where):ii.append(where);rr.append(np.full(len(where),row,np.int32))
            if notify is not None and row%1000==0:notify(row,len(self.ids))
        index=np.concatenate(ii) if ii else np.empty(0,np.int64)
        rows=np.concatenate(rr) if rr else np.empty(0,np.int32)
        order=np.argsort(index,kind='stable');index=index[order];rows=rows[order]
        offsets=np.r_[0,np.cumsum(np.bincount(index,minlength=len(pixels)),dtype=np.int64)]
        return offsets,rows,dict(maximum_K=int(np.diff(offsets).max(initial=0)),support_oob_pixels=oob,ellipse_fallbacks=fallback)

    def candidate_geometry(self,pixels,offsets,rows,notify=None):
        query=np.repeat(np.arange(len(pixels)),np.diff(offsets));out=np.empty(len(rows),float)
        order=np.argsort(rows,kind='stable');sr=rows[order]
        starts=np.r_[0,1+np.flatnonzero(sr[1:]!=sr[:-1]),len(sr)] if len(sr) else np.array([0])
        w=self.nuclei.shape[1]
        for k,(a,b) in enumerate(zip(starts[:-1],starts[1:])):
            row=int(sr[a]);ix=order[a:b];flat=pixels[query[ix]];y,x=self.nucleus(int(self.ids[row]))
            distance=cKDTree(np.c_[x,y]).query(np.c_[flat%w,flat//w],workers=1)[0]
            out[ix]=-np.minimum(distance/self.radius[row],4)
            if notify is not None and k%1000==0:notify(k,len(starts)-1)
        return out


def interface_band(baseline):
    """Both sides of differing positive 8-neighbour interfaces, Euclidean 2 um."""
    baseline=np.asarray(baseline);boundary=np.zeros(baseline.shape,bool);h,w=baseline.shape
    for dy,dx in ((0,1),(1,-1),(1,0),(1,1)):
        ya=slice(max(0,-dy),min(h,h-dy));xa=slice(max(0,-dx),min(w,w-dx))
        yb=slice(max(0,dy),min(h,h+dy));xb=slice(max(0,dx),min(w,w+dx))
        a=baseline[ya,xa];b=baseline[yb,xb];different=(a>0)&(b>0)&(a!=b)
        boundary[ya,xa]|=different;boundary[yb,xb]|=different
    if not boundary.any():return boundary
    return ndi.distance_transform_edt(~boundary)<=2.0


def fixed_domain(baseline,nuclei,pixels,offsets,rows,ids,eligible,donor,pixel_counts,notify=None):
    """One frozen candidate universe and abstention decision for all A2 arms."""
    h,w=baseline.shape;band=interface_band(baseline);pos=[];candidate_rows=[];out_offsets=[0];reasons={}
    def reject(reason):reasons[reason]=reasons.get(reason,0)+1
    for p,flat in enumerate(pixels):
        if notify is not None and p%1000==0:notify(p,len(pixels))
        y,x=divmod(int(flat),w);owner=int(baseline[y,x])
        if not owner:reject('background');continue
        if nuclei[y,x]:reject('nuclear');continue
        if not band[y,x]:reject('outside_interface_band');continue
        support=rows[offsets[p]:offsets[p+1]]
        if len(support)<2:reject('fewer_than_two_complete_supports');continue
        adjacent=np.unique(baseline[max(0,y-1):min(h,y+2),max(0,x-1):min(w,x+2)])
        cand=support[np.isin(ids[support],adjacent[adjacent>0])]
        if owner not in ids[cand]:reject('owner_not_supported');continue
        if len(cand)<2:reject('fewer_than_two_adjacent_supported_candidates');continue
        if not eligible[cand].all():reject('candidate_not_A1_evaluable');continue
        if np.any(donor[cand]<0):reject('candidate_missing_full_count_donor');continue
        if pixel_counts.getrow(p).sum()==0:reject('no_intersection_gene_expression');continue
        pos.append(p);candidate_rows.extend(cand);out_offsets.append(len(candidate_rows))
    return np.asarray(pos,np.int64),np.asarray(out_offsets,np.int64),np.asarray(candidate_rows,np.int32),reasons


def score_proposals(baseline,pixels,offsets,rows,geometry,counts,ids,profiles,disabled=False,notify=None,beta=BETA):
    """All gene records at a pixel share one owner. Scores use mean log likelihood."""
    proposals=[];under_margin=0;w=baseline.shape[1]
    score_dtype=[('pixel','<i8'),('candidate_id','<u4'),('geometry','<f8'),('expression','<f8'),('score','<f8')]
    scores=np.empty(len(rows),dtype=score_dtype)
    for p,flat in enumerate(pixels):
        if notify is not None and p%1000==0:notify(p,len(pixels))
        lo,hi=offsets[p:p+2];cand=rows[lo:hi];current=int(baseline.ravel()[flat]);pc=counts.getrow(p)
        if not len(cand) or pc.sum()<=0:raise ValueError('frozen scoring domain lacks candidate/expression')
        expression=np.log(profiles[cand][:,pc.indices])@pc.data/float(pc.sum())
        value=geometry[lo:hi]+beta*expression;current_index=np.flatnonzero(ids[cand]==current)
        if len(current_index)!=1:raise ValueError('current owner not exactly once in frozen candidates')
        best=float(value.max());ci=int(current_index[0]);ties=np.flatnonzero(value==best)
        winner=ci if ci in ties else int(ties[np.argmin(ids[cand[ties]])])
        margin=best-float(value[ci]);target=int(ids[cand[winner]])
        scores['pixel'][lo:hi]=flat;scores['candidate_id'][lo:hi]=ids[cand]
        scores['geometry'][lo:hi]=geometry[lo:hi];scores['expression'][lo:hi]=expression;scores['score'][lo:hi]=value
        if target!=current and margin>=MARGIN and not disabled:
            y,x=divmod(int(flat),w);proposals.append((margin,y,x,current,target))
        else:under_margin+=1
    proposals.sort(key=lambda t:(-t[0],t[1],t[2],t[4]))
    return np.asarray(proposals,dtype=[('margin','<f8'),('y','<i4'),('x','<i4'),('source','<u4'),('target','<u4')]), scores,under_margin


def rooted(mask,nucleus):
    labels,n=ndi.label(mask,CONNECTIVITY);roots=np.unique(labels[nucleus&mask]);roots=roots[roots>0]
    return np.isin(labels,roots)&mask,n


def topology_oracle(mask,nuclei,y,x,source,target):
    """Independent complete-image oracle for test/mini; no finite crop assumption."""
    if nuclei[y,x]!=0:return False,'nuclear_pixel'
    if mask[y,x]!=source or not source or not target:return False,'owner_or_background'
    donor=mask==source;recipient=mask==target
    before,n0=rooted(donor,nuclei==source);donor[y,x]=False;after,n1=rooted(donor,nuclei==source)
    if n1>n0:return False,'donor_components_increased'
    before[y,x]=False
    if np.any(before&~after):return False,'donor_nucleus_connection_lost'
    connected,_=rooted(recipient,nuclei==target)
    if not connected[max(0,y-1):y+2,max(0,x-1):x+2].any():return False,'recipient_not_nucleus_connected'
    return True,'accepted'


class CompleteCellTopology:
    """Certified full-cell bounding boxes, not arbitrary image crops.

    Every proposal pixel was adjacent to its target in frozen baseline, so all
    reachable masks lie within that ID's baseline bbox padded by one pixel.
    We include every proposed coordinate as a second explicit certification.
    """
    def __init__(self,baseline,nuclei,proposals):
        self.nuclei=nuclei;self.boxes=ndi.find_objects(baseline);h,w=baseline.shape
        self.bounds={}
        for i,box in enumerate(self.boxes):
            if box:self.bounds[i+1]=[box[0].start,box[0].stop,box[1].start,box[1].stop]
        for p in proposals:
            for cid in (int(p['source']),int(p['target'])):
                if cid not in self.bounds:raise ValueError('proposal unknown ID')
                b=self.bounds[cid];y=int(p['y']);x=int(p['x'])
                b[0]=min(b[0],y);b[1]=max(b[1],y+1);b[2]=min(b[2],x);b[3]=max(b[3],x+1)
        self.bounds={cid:(slice(max(0,b[0]-1),min(h,b[1]+1)),slice(max(0,b[2]-1),min(w,b[3]+1))) for cid,b in self.bounds.items()}

    def check(self,mask,y,x,source,target):
        if self.nuclei[y,x]!=0:return False,'nuclear_pixel'
        if mask[y,x]!=source or not source or not target:return False,'owner_or_background'
        db=self.bounds[source];rb=self.bounds[target];dy=y-db[0].start;dx=x-db[1].start
        donor=mask[db]==source;dn=self.nuclei[db]==source
        before,n0=rooted(donor,dn);donor[dy,dx]=False;after,n1=rooted(donor,dn)
        if n1>n0:return False,'donor_components_increased'
        before[dy,dx]=False
        if np.any(before&~after):return False,'donor_nucleus_connection_lost'
        recipient=mask[rb]==target;connected,_=rooted(recipient,self.nuclei[rb]==target)
        ry=y-rb[0].start;rx=x-rb[1].start
        if not connected[max(0,ry-1):ry+2,max(0,rx-1):rx+2].any():return False,'recipient_not_nucleus_connected'
        return True,'accepted'


def apply_proposals(baseline,nuclei,proposals,out,deps,notify=None,stop_after=None,oracle=False):
    """Atomic generation snapshots bind exact sorted proposal cursor and mask.

    A crash before checkpoint publication recomputes at most one batch; it never
    reapplies an accepted proposal to a committed snapshot. No RNG is consumed.
    """
    out=Path(out);out.mkdir(parents=True,exist_ok=True);topology=CompleteCellTopology(baseline,nuclei,proposals)
    statepath=out/'state.json';mask=baseline.copy();cursor=0;decisions=np.zeros(len(proposals),np.uint8);reasons={}
    if statepath.exists():
        state=common.read(statepath)
        if state['dependencies']!=deps:raise ValueError('A2 recovery dependencies drift')
        for obj in (state['mask'],state['decisions']):common.check_identity(obj)
        mask=np.load(state['mask']['path']);decisions=np.load(state['decisions']['path']);cursor=state['cursor'];reasons=state['reasons']
        if not 0<=cursor<=len(proposals) or len(decisions)!=len(proposals):raise ValueError('A2 recovery cursor')
    initial_cursor=cursor
    def checkpoint():
        mp=out/f'mask_{cursor:09d}.npy';dp=out/f'decisions_{cursor:09d}.npy'
        atomic_npy(mp,mask);atomic_npy(dp,decisions)
        common.atomic_json(statepath,dict(dependencies=deps,cursor=cursor,mask=common.identity(mp),decisions=common.identity(dp),reasons=reasons))
        # Last published snapshot remains authoritative; earlier versions are redundant.
        for p in out.glob('mask_*.npy'):
            if p!=mp:p.unlink()
        for p in out.glob('decisions_*.npy'):
            if p!=dp:p.unlink()
    while cursor<len(proposals):
        p=proposals[cursor];y=int(p['y']);x=int(p['x']);source=int(p['source']);target=int(p['target'])
        ok,reason=topology.check(mask,y,x,source,target)
        if oracle and (ok,reason)!=topology_oracle(mask,nuclei,y,x,source,target):raise ValueError('incremental/full-image topology mismatch')
        decisions[cursor]=1 if ok else 2;reasons[reason]=reasons.get(reason,0)+1
        if ok:mask[y,x]=target
        cursor+=1
        if cursor%1000==0 or cursor==len(proposals) or (stop_after is not None and cursor-initial_cursor>=stop_after):
            checkpoint()
            if notify is not None:notify(cursor,len(proposals))
            if stop_after is not None and cursor-initial_cursor>=stop_after and cursor<len(proposals):return None,dict(cursor=cursor,complete=False,reasons=reasons)
    if not statepath.exists():checkpoint()
    if not np.array_equal(mask>0,baseline>0) or not np.array_equal(mask[nuclei>0],baseline[nuclei>0]):raise ValueError('A2 capture/nuclear invariant')
    return mask,dict(cursor=cursor,complete=True,reasons=reasons,accepted=int(np.count_nonzero(decisions==1)),rejected=int(np.count_nonzero(decisions==2)))
