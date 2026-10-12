"""Loss-only native dataset sidecars and full-nucleus morphology evidence.

Original DataProcessing.__getitem__ performs all native input/marker/support
work. The subclass only records and deterministically replays its one transform.
Counts come independently from its original HDF5 before candidate replication.
"""
import functools
import re
from pathlib import Path
import cv2
import h5py
import imgaug.augmenters as iaa
import numpy as np
import torch
import tifffile
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from bidcell.model.dataio.dataset_input import DataProcessing
from .common import BASELINE, native_config, read


def coords(path):return tuple(map(int,re.findall(r'\d+',Path(path).stem)))


class IndexedIDs(list):
    """Same immutable unique-ID order and .index semantics, with O(1) lookup."""
    def __init__(self,values):
        super().__init__(values);self.lookup={value:i for i,value in enumerate(values)}
        if len(self.lookup)!=len(self):raise ValueError('duplicate annotated nucleus IDs')
    def index(self,value,*bounds):
        if bounds:return super().index(value,*bounds)
        try:return self.lookup[value]
        except KeyError:raise ValueError(f'{value!r} is not in list') from None


def transform(array, flip, rot):
    # imgaug's Rot90 is clockwise; use the actual library as the authority.
    result=array[None]
    if flip:result=iaa.Flipud(1.0)(images=result)
    if rot:result=iaa.Rot90(int(rot),keep_size=True)(images=result)
    return np.ascontiguousarray(result[0])


@functools.lru_cache(8)
def transform_matrix(flip,rot):
    # Infer signed coordinate-basis transform from a labelled centre and bases.
    fixture=np.zeros((5,5,1),np.int32);fixture[2,2,0]=1;fixture[2,3,0]=2;fixture[3,2,0]=3
    out=transform(fixture,flip,rot)[...,0]
    centre=np.argwhere(out==1)[0][::-1]
    return np.stack([np.argwhere(out==v)[0][::-1]-centre for v in (2,3)],axis=1).astype(float)


def support_kernel(mask,elongated):
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


class Evidence:
    def __init__(self,root,nuclei=None):
        self.root=Path(root);self.manifest=read(self.root/'prepared/manifest.json')
        self.nuclei=tifffile.imread(BASELINE/'nuclei.tif').astype(np.int32) if nuclei is None else nuclei
        p=self.root/'prepared'
        self.meta=np.load(p/'nucleus_metadata.npz')
        self.full_counts=np.load(p/'full_nuclear_counts.npy',mmap_mode='r')
        self.boxes=ndi.find_objects(self.nuclei)
        self.dapi=np.load(p/'dapi_features.npy',mmap_mode='r')
        self.ids=self.meta['ids'];self.centroids=self.meta['centroid_xy']
        self.id_to_row={int(cid):i for i,cid in enumerate(self.ids)}
        self.centroid_tree=cKDTree(self.centroids)
        self.ref=read(self.root/'configs/reference_region.json')
        self.type_names=self.ref['type_names'];self.ref_valid=np.asarray(self.ref['valid'],bool)
        self.elongated=set(native_config().model_params.elongated)
        self.stats=self.manifest['geometry_stats']

    @functools.lru_cache(maxsize=512)
    def nucleus(self,cid):
        box=self.boxes[cid-1]
        if box is None:raise ValueError('missing full global nucleus '+str(cid))
        yy,xx=np.where(self.nuclei[box]==cid);yy+=box[0].start;xx+=box[1].start
        return yy,xx

    @functools.lru_cache(maxsize=512)
    def support(self,cid):
        yy,xx=self.nucleus(cid);row=self.id_to_row[cid];typ=int(self.meta['type_index'][row])
        elongated=typ>=0 and self.type_names[typ] in self.elongated
        pad=80
        while True:
            y0=int(yy.min())-pad;x0=int(xx.min())-pad
            mask=np.zeros((int(np.ptp(yy))+1+2*pad,int(np.ptp(xx))+1+2*pad),np.uint8)
            mask[yy-y0,xx-x0]=1
            kernel,fallback=support_kernel(mask,elongated)
            sup=cv2.dilate(mask,kernel,iterations=1)>0
            if not (sup[[0,-1],:].any() or sup[:,[0,-1]].any()):break
            pad*=2
            if pad>1024:raise RuntimeError('global support halo failed')
        sy,sx=np.where(sup);sy+=y0;sx+=x0
        return sy,sx,kernel,fallback

    def crop_support(self,cid,y,x):
        sy,sx,_,_=self.support(int(cid));inside=(sy>=y)&(sy<y+48)&(sx>=x)&(sx<x+48)
        mask=np.zeros((48,48),bool);mask[sy[inside]-y,sx[inside]-x]=True
        censored=bool(((sy<0)|(sy>=self.nuclei.shape[0])|(sx<0)|(sx>=self.nuclei.shape[1])).any())
        return mask,len(sy),censored

    def other_distance(self,cid,y,x,query):
        # Expand until unseen pixels outside the crop cannot be nearer to any
        # query point. Global image edges are not fictitious nuclei.
        H,W=self.nuclei.shape;halo=32
        while True:
            y0=max(0,y-halo);x0=max(0,x-halo);y1=min(H,y+48+halo);x1=min(W,x+48+halo)
            patch=self.nuclei[y0:y1,x0:x1]
            yy,xx=np.where((patch>0)&(patch!=cid))
            if len(yy):
                dist=cKDTree(np.stack((xx+x0,yy+y0),1)).query(query,workers=1)[0]
                # Pixel centres just beyond the cropped rectangle.
                lower=np.full(len(query),np.inf)
                if y0>0:lower=np.minimum(lower,query[:,1]-(y0-1))
                if y1<H:lower=np.minimum(lower,y1-query[:,1])
                if x0>0:lower=np.minimum(lower,query[:,0]-(x0-1))
                if x1<W:lower=np.minimum(lower,x1-query[:,0])
                if np.all(dist<=lower):return dist.reshape(48,48),False
            if y0==0 and x0==0 and y1==H and x1==W:
                return np.full((48,48),np.inf),True
            halo*=2

    def geometry(self,ids,y,x,flip,rot,native_support):
        yy,xx=np.mgrid[y:y+48,x:x+48];query=np.stack((xx.ravel(),yy.ravel()),1)
        R=transform_matrix(flip,rot);geometry=[];no_neighbour=0
        multiplicity=np.minimum(np.log1p(native_support.sum(0))/np.log(33),1)
        for i,cid in enumerate(ids):
            cid=int(cid);row=self.id_to_row[cid];ny,nx=self.nucleus(cid)
            radius=max(float(np.sqrt(len(ny)/np.pi)),.5);mu=self.centroids[row]
            own=(self.nuclei[y:y+48,x:x+48]==cid)
            other=(self.nuclei[y:y+48,x:x+48]>0)&~own
            # Exact full-mask EDT includes a zero halo around the full nucleus.
            box=self.boxes[cid-1];full=(self.nuclei[box]==cid)
            interior=ndi.distance_transform_edt(np.pad(full,1))[1:-1,1:-1]
            d_in=np.zeros((48,48))
            vy,vx=np.where(own);d_in[vy,vx]=interior[vy+y-box[0].start,vx+x-box[1].start]
            d_out=cKDTree(np.stack((nx,ny),1)).query(query,workers=1)[0].reshape(48,48)
            d_other,missing=self.other_distance(cid,y,x,query)
            no_neighbour+=missing
            offset=np.stack((xx-mu[0],yy-mu[1]),-1)/radius
            offset=offset@R.T
            covariance=self.meta['covariance'][row];cov=R@covariance@R.T
            trace=np.trace(cov)
            vals=np.linalg.eigvalsh(cov)
            if trace<=1e-12:cxx,cxy,cyy,ecc=.5,0.,.5,0.
            else:
                cxx,cxy,cyy=cov[0,0]/trace,cov[0,1]/trace,cov[1,1]/trace
                ecc=np.sqrt(max(0,1-vals[0]/max(vals[1],1e-8)))
            if len(self.ids)==1:vector=np.zeros(2)
            else:
                dist,indices=self.centroid_tree.query(mu,k=min(3,len(self.ids)))
                nearest=float(np.min(np.asarray(dist)[np.asarray(indices)!=row]))
                neighbours=self.centroid_tree.query_ball_point(mu,nearest+1e-10)
                neighbours=[j for j in neighbours if j!=row]
                j=min(neighbours,key=lambda j:(float(np.linalg.norm(self.centroids[j]-mu)),int(self.ids[j])))
                vector=R@((self.centroids[j]-mu)/radius)
            radius_feature=np.clip((np.log1p(radius)-self.stats['mean_log_radius'])/max(self.stats['sd_log_radius'],1e-6),-3,3)/3
            def scalar(a):return transform(np.asarray(a)[...,None],flip,rot)[...,0]
            geo=np.zeros((16,48,48),np.float32)
            geo[0]=scalar(own.astype(np.uint8));geo[1]=scalar(other.astype(np.uint8))
            geo[2]=scalar(np.clip((d_out-d_in)/radius,-4,4)/4)
            geo[3]=scalar(np.clip(d_other/radius,0,4)/4)
            geo[4]=scalar(np.clip(offset[...,0],-4,4)/4);geo[5]=scalar(np.clip(offset[...,1],-4,4)/4)
            geo[6]=radius_feature;geo[7]=cxx;geo[8]=cxy;geo[9]=cyy;geo[10]=ecc
            geo[11]=native_support[i];geo[12]=multiplicity
            geo[13]=np.clip(vector[0],-4,4)/4;geo[14]=np.clip(vector[1],-4,4)/4
            geo[15]=scalar(np.clip((d_out-d_other)/radius,-4,4)/4)
            geometry.append(geo)
        return np.stack(geometry) if geometry else np.zeros((0,16,48,48),np.float32),no_neighbour

    def aux(self,native,raw,flip,rot,morphology):
        y,x=int(native[5]),int(native[6]);labels=np.asarray(native[7],np.int64)
        ids=np.unique(labels);ids=ids[ids>0];rows=np.asarray([self.id_to_row[int(cid)] for cid in ids],np.int64)
        support=np.asarray(native[2]).transpose(2,0,1).astype(bool)
        global_masks=[];global_area=[];visible=[];censored=[];full_pixels=[];visible_pixels=[]
        for cid in ids:
            mask,area,censor=self.crop_support(int(cid),y,x)
            global_masks.append(transform(mask[...,None].astype(np.uint8),flip,rot)[...,0].astype(bool))
            global_area.append(area);visible.append(int(mask.sum()));censored.append(censor)
            full_pixels.append(len(self.nucleus(int(cid))[0]));visible_pixels.append(int((labels==cid).sum()))
        types=self.meta['type_index'][rows]
        valid=(types>=0)&self.ref_valid[np.maximum(types,0)]
        aux=dict(raw_counts=np.ascontiguousarray(raw.transpose(2,0,1),dtype=np.float32),candidate_ids=ids,
                 nuclear_labels=labels,full_nuclear_counts=np.asarray(self.full_counts[rows],np.float32),
                 global_support_visible=np.stack(global_masks) if len(ids) else np.zeros((0,48,48),bool),
                 support_area_global=np.asarray(global_area,np.int64),support_area_visible=np.asarray(visible,np.int64),
                 support_image_censored=np.asarray(censored,bool),full_nucleus_pixels=np.asarray(full_pixels,np.int64),
                 visible_nucleus_pixels=np.asarray(visible_pixels,np.int64),type_index=types,ref_valid=valid,
                 specimen_clipped=self.meta['specimen_clipped'][rows],native_support=support)
        if morphology:
            geo,missing=self.geometry(ids,y,x,flip,rot,support)
            dapi=transform(np.asarray(self.dapi[:,y:y+48,x:x+48]).transpose(1,2,0),flip,rot).transpose(2,0,1)[None]
            aux.update(geometry=geo,dapi=np.ascontiguousarray(dapi))
        return {k:torch.from_numpy(np.ascontiguousarray(v)) for k,v in aux.items()}


class NativeViews(DataProcessing):
    def __init__(self,root,training=True,shift=0,auxiliary=True,morphology=False):
        super().__init__(native_config(),isTraining=training,all_patches=True,shift_patches=shift,total_steps=4000 if training else None)
        self.root=Path(root);self.forced_transform=None;self.recorded=(0,0)
        self.morphology_enabled=morphology
        self.evidence=Evidence(root,self.nuclei) if auxiliary else None
        # Only accelerate immutable identity lookup; original __getitem__, the
        # candidate ordering, negative type indexing and marker arithmetic run
        # unchanged. Independent eight-transform comparison verifies equivalence.
        self.nuclei_types_ids=IndexedIDs(self.nuclei_types_ids)
        self.nuclei_ids_elong=set(self.nuclei_ids_elong)

    def augment_data(self,batch_raw):
        if self.forced_transform is None:
            result,flip,rot=super().augment_data(batch_raw)
        else:
            flip,rot=self.forced_transform;result=transform(batch_raw,flip,rot)[None]
        self.recorded=(int(flip),int(rot));return result,flip,rot

    def get_view(self,view,device='cpu'):
        index=int(view['index']);self.forced_transform=(int(view.get('flip',0)),int(view.get('rot90',0)))
        native=super().__getitem__(index)
        flip,rot=self.recorded if self.isTraining else (0,0)
        with h5py.File(self.fp_patches[index],'r') as f:raw=f['data'][:].astype(np.float32)
        raw=transform(raw,flip,rot)
        if not np.array_equal(raw.sum(-1),np.asarray(native[8])):raise ValueError('raw/native expression alignment failure')
        ids=np.unique(native[7]);ids=ids[ids>0].astype(np.int64)
        if 'candidate_ids' in view and ids.tolist()!=view['candidate_ids']:raise ValueError('frozen candidate identity mismatch')
        batch=dict(input=native[0].permute(3,2,0,1),nuclei=native[1].permute(2,0,1)[:,None],
                   support=native[2].permute(2,0,1)[:,None],positive=native[3].permute(2,0,1)[:,None],
                   negative=native[4].permute(2,0,1)[:,None],labels=torch.from_numpy(np.asarray(native[7],np.int64)),
                   xy=(int(native[5]),int(native[6])),candidate_ids=ids)
        if self.evidence:batch['aux']=self.evidence.aux(native,raw,flip,rot,self.morphology_enabled)
        for key,value in batch.items():
            if torch.is_tensor(value):batch[key]=value.to(device)
        if 'aux' in batch:batch['aux']={k:v.to(device) for k,v in batch['aux'].items()}
        return batch
