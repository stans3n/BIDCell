"""Freeze Rep2 views and unchanged full-grid morphology features once."""
import gc
import os
from pathlib import Path
import h5py
import natsort
import numpy as np
import tifffile
import torch
from scipy import ndimage as ndi
from .common import *
from .preprocess import coords
from .data import Views

def make_views(dataset, updates, b1=False):
    rng=np.random.RandomState(1234);gen=torch.Generator().manual_seed(1234)
    views=[];nonempty=epoch=0
    while nonempty<updates:
        order=rng.permutation(len(dataset)) if b1 else torch.randperm(len(dataset),generator=gen).tolist()
        before=nonempty
        for index in order:
            path=str(dataset.fp_patches[int(index)]);y,x=coords(path)
            ids=np.unique(dataset.nuclei[y:y+48,x:x+48]);ids=ids[ids>0].astype(int).tolist()
            views.append(dict(cursor=len(views),index=int(index),path=path,row=y,col=x,epoch=epoch,
                              flip=int(rng.randint(2)),rot90=int(rng.randint(4)),candidate_ids=ids,K=len(ids)))
            nonempty+=bool(ids)
            if nonempty==updates:break
        if nonempty==before:raise ValueError('entire native train pool empty')
        epoch+=1
    return dict(seed=1234,train_pool=list(map(str,dataset.fp_patches)),actual_updates=updates,formal_visit_count=len(views),
                empty_skips=sum(v['K']==0 for v in views),max_K=max(v['K'] for v in views),views=views,
                shuffle='numpy.RandomState permutation' if b1 else 'torch.randperm separate Generator',
                augmentation='numpy.RandomState1234 flip/quarter-turn')

def prepare():
    cfg=configure();configure_seed();root=ROOT/'shared';p=root/'prepared';data=Path(cfg.files.data_dir)
    p.mkdir(exist_ok=True);(root/'configs').mkdir(exist_ok=True)
    genes=(data/cfg.files.fp_gene_names).read_text().splitlines()
    from scripts.new_methods.reference import build_reference
    atomic_json(root/'configs/reference_region.json',build_reference(cfg.files.fp_ref,genes,'V1_REGION'))
    dataset=Views();nuc=dataset.nuclei
    matched=make_views(dataset,3625);atomic_json(root/'configs/views_matched.json',matched)
    full=Views(full_pool=True);anchor=make_views(full,4000,b1=True);del full;gc.collect()
    atomic_json(root/'configs/views_b1.json',anchor)
    union=np.zeros(nuc.shape,bool);unique_train=set()
    for v in matched['views']:
        unique_train.update(v['candidate_ids']);union[v['row']:v['row']+48,v['col']:v['col']+48]=True
    ids=np.unique(nuc);ids=ids[ids>0];boxes=ndi.find_objects(nuc)
    with h5py.File(data/cfg.files.fp_nuclei_anno) as f:tmap=dict(zip(f['ids'][:].astype(int),f['data'][:].astype(int)))
    if set(map(int,ids))!=set(tmap):raise ValueError('annotation nucleus identity mismatch')
    centroid=[];covariance=[];area=[];clipped=[]
    for cid in ids:
        box=boxes[cid-1];yy,xx=np.where(nuc[box]==cid);yy+=box[0].start;xx+=box[1].start
        xy=np.stack((xx,yy),1).astype(float);mu=xy.mean(0);centroid.append(mu)
        covariance.append((xy-mu).T@(xy-mu)/len(xy));area.append(len(xy))
        clipped.append(bool((yy==0).any() or (xx==0).any() or (yy==nuc.shape[0]-1).any() or (xx==nuc.shape[1]-1).any()))
    area=np.asarray(area,np.int64);id_to_row=np.full(int(ids.max())+1,-1,np.int64);id_to_row[ids]=np.arange(len(ids))
    metadata=dict(ids=ids,centroid_xy=np.asarray(centroid),covariance=np.asarray(covariance),area=area,
                  type_index=np.asarray([tmap[int(cid)] for cid in ids],np.int64),specimen_clipped=np.asarray(clipped))
    with (p/'nucleus_metadata.tmp').open('wb') as f:np.savez(f,**metadata)
    os.replace(p/'nucleus_metadata.tmp',p/'nucleus_metadata.npz')
    train_rows=id_to_row[np.asarray(sorted(unique_train))];logr=np.log1p(np.maximum(np.sqrt(area[train_rows]/np.pi),.5))
    geom_stats=dict(unique_nuclei=len(train_rows),mean_log_radius=float(logr.mean()),sd_log_radius=float(logr.std(ddof=0)))
    from scripts.new_methods.prepare import add_counts
    counts=np.zeros((len(ids),len(genes)),np.int64);coverage=np.zeros(nuc.shape,np.uint8);maps=[];raw_total=nuclear_total=0
    for file in natsort.natsorted((data/'expr_maps').glob('all_genes_*.hdf5')):
        y0,y1,x0,x1=coords(file);y1=min(y1,nuc.shape[0]);x1=min(x1,nuc.shape[1])
        if coverage[y0:y1,x0:x1].any():raise ValueError('overlapping expression cores')
        coverage[y0:y1,x0:x1]=1
        with h5py.File(file) as f:
            for sy in range(y0,y1,64):
                ey=min(sy+64,y1);values=f['data'][sy-y0:ey-y0,:x1-x0,:];labels=nuc[sy:ey,x0:x1]
                add_counts(counts,labels,values,id_to_row)
                raw_total+=int(values.sum(dtype=np.int64));nuclear_total+=int(values[labels>0].sum(dtype=np.int64))
        maps.append(dict(source=identity(file),core=[y0,y1,x0,x1]));progress('prepare',part='nuclear_counts',map=file.name)
    if not coverage.all() or counts.sum()!=nuclear_total:raise ValueError('morphology count conservation failed')
    np.save(p/'full_nuclear_counts.tmp.npy',counts);os.replace(p/'full_nuclear_counts.tmp.npy',p/'full_nuclear_counts.npy')
    del counts,coverage;gc.collect()
    dapi=tifffile.imread(data/cfg.files.fp_rdapi)
    if dapi.shape!=nuc.shape:raise ValueError('DAPI/nuclei shape mismatch')
    intensity=np.log1p(dapi.astype(np.float32));del dapi
    q1,q99=np.quantile(intensity[union],[.01,.99],method='linear')
    if q99<=q1:raise ValueError('no DAPI training contrast')
    intensity=np.clip((intensity-q1)/(q99-q1),0,1)
    features=np.lib.format.open_memmap(p/'dapi_features.tmp.npy',mode='w+',dtype=np.float32,shape=(3,*nuc.shape))
    for y in range(0,nuc.shape[0],256):
        y0=max(0,y-3);y1=min(nuc.shape[0],y+256+3);stop=min(y+256,nuc.shape[0]);im=intensity[y0:y1]
        gx=ndi.sobel(im,axis=1,mode='reflect');gy=ndi.sobel(im,axis=0,mode='reflect')
        edge=np.clip(np.sqrt(gx*gx+gy*gy)/4,0,1);local=im-ndi.uniform_filter(im,size=5,mode='reflect')
        sl=slice(y-y0,stop-y0);features[:,y:stop]=np.stack((im[sl],edge[sl],local[sl]))
    features.flush();del features;os.replace(p/'dapi_features.tmp.npy',p/'dapi_features.npy')
    initial=Path(read(ROOT/'configs/run.json')['shared_initial'])
    link=p/'shared_initial.pth'
    if link.is_symlink():
        if link.resolve()!=initial.resolve():raise ValueError('initial link identity differs')
    elif link.exists():raise ValueError('unexpected local initial copy')
    else:link.symlink_to(initial)
    manifest=dict(status='succeeded',geometry_stats=geom_stats,shape=list(nuc.shape),genes=genes,raw_maps=maps,
                  raw_count_total=raw_total,nuclear_count_total=nuclear_total,unknown_nuclei=int((metadata['type_index']==-1).sum()),
                  dapi=dict(q1=float(q1),q99=float(q99),training_unique_pixels=int(union.sum()),quantile='linear exact unique training pixels'),
                  matched=dict(visits=len(matched['views']),empty_skips=matched['empty_skips'],max_K=matched['max_K']),
                  b1=dict(visits=len(anchor['views']),empty_skips=anchor['empty_skips'],max_K=anchor['max_K']))
    atomic_json(p/'manifest.json',manifest)
    return {name:p/name for name in ('manifest.json','nucleus_metadata.npz','full_nuclear_counts.npy','dapi_features.npy','shared_initial.pth')} | {name:root/'configs'/name for name in ('views_matched.json','views_b1.json','reference_region.json')}
