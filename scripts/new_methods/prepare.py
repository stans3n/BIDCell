"""Prepare immutable fresh-state, view sequence, reference and full-grid evidence."""
import json
import os
import re
import shutil
import time
from pathlib import Path
import h5py
import natsort
import numpy as np
import pandas as pd
import tifffile
import torch
from numba import njit
from scipy import ndimage as ndi
from .common import *
from .data import coords, NativeViews, transform_matrix
from .reference import build_reference


@njit(cache=True)
def add_counts(counts,labels,values,id_to_row):
    for y in range(labels.shape[0]):
        for x in range(labels.shape[1]):
            cid=labels[y,x]
            if cid>0:
                row=id_to_row[cid]
                for g in range(values.shape[2]):counts[row,g]+=int(values[y,x,g])


def prepare(root,rebuild=False):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    for name in ['configs','prepared','admission','source','logs']:(root/name).mkdir(exist_ok=True)
    cfg=native_config();p=root/'prepared';start=time.monotonic()
    producers=['scripts/new_methods/prepare.py','scripts/new_methods/reference.py','scripts/new_methods/data.py',
               'scripts/new_methods/common.py','bidcell/model/model/model.py','bidcell/model/model/intialisation.py',
               'bidcell/model/model/layers.py','bidcell/model/dataio/dataset_input.py','bidcell/config.py']
    preparation_source={name:sha256(REPO/name) for name in producers}
    if (p/'manifest.json').exists() and not rebuild:
        rec=read(p/'manifest.json');verify_outputs(rec)
        for name,obj in rec['dependencies'].items():
            if sha256(obj['path'])!=obj['sha256']:raise ValueError('prepared input changed: '+name)
        for entry in rec['raw_maps']:
            obj=entry['source']
            if sha256(obj['path'])!=obj['sha256']:raise ValueError('prepared raw expression core changed')
        if rec.get('preparation_source')!=preparation_source:raise ValueError('prepared algorithm changed; create a new preparation attempt')
        for obj in rec['configs'].values():
            if sha256(obj['path'])!=obj['sha256']:raise ValueError('prepared configuration changed')
        write_configs(root)
        return rec
    seed_all();nuc=tifffile.imread(BASELINE/'nuclei.tif').astype(np.int32)
    genes=(BASELINE/'all_gene_names.txt').read_text().splitlines()
    if len(genes)!=313 or len(set(genes))!=313:raise ValueError('expected frozen 313 genes')
    ref_r=build_reference(cfg.files.fp_ref,genes,'V1_REGION');ref_b=build_reference(cfg.files.fp_ref,genes,'V1_BOUNDARY')
    atomic_json(root/'configs/reference_region.json',ref_r);atomic_json(root/'configs/reference_boundary.json',ref_b)
    dataset=NativeViews(root,auxiliary=False)
    if len(dataset)!=4726:raise ValueError('historical 14536->4726 pool changed')
    del dataset.nuclei;dataset.nuclei=nuc
    generator=torch.Generator().manual_seed(1234);order=torch.randperm(len(dataset),generator=generator).tolist()
    rng=np.random.RandomState(1234);views=[];unique_train=set();union=np.zeros(nuc.shape,bool)
    for cursor,index in enumerate(order):
        path=Path(dataset.fp_patches[index]);y,x=coords(path)
        ids=np.unique(nuc[y:y+48,x:x+48]);ids=ids[ids>0].astype(int).tolist()
        unique_train.update(ids);union[y:y+48,x:x+48]=True
        views.append(dict(cursor=cursor,index=index,path=str(path),row=y,col=x,flip=int(rng.randint(2)),rot90=int(rng.randint(4)),candidate_ids=ids,K=len(ids)))
    if sum(v['K']>0 for v in views)<3625:raise ValueError('one frozen epoch cannot supply budget')
    configured_order=views
    formal_stop=int(np.flatnonzero(np.cumsum([v['K']>0 for v in views])==3625)[0])+1
    views=views[:formal_stop]
    unique_train=set();union[:]=False
    for v in views:
        unique_train.update(v['candidate_ids']);union[v['row']:v['row']+48,v['col']:v['col']+48]=True
    atomic_json(root/'configs/views.json',dict(seed=1234,train_pool=14536,configured_views=4726,actual_updates=3625,
                shuffle='torch.randperm separate Generator(seed1234)',augmentation='NumPy RandomState1234 flip/quarter-turn',
                historical_limitation='B0 exact initial RNG and shuffled visit order were not saved; only new arms are exactly paired',
                formal_visit_count=len(views),configured_order=configured_order,views=views))
    nonempty=[v for v in views if v['K']>0];boundary=[]
    for lo,hi,n in [(1,1,16),(2,4,24),(5,100000,24)]:
        boundary.extend([v for v in nonempty if lo<=v['K']<=hi][:n])
    for v in nonempty:
        if len(boundary)==64:break
        if v not in boundary:boundary.append(v)
    boundary_fixed=[v for v in boundary if v['K']==1][:2]+[v for v in boundary if 2<=v['K']<=4][:3]+[v for v in boundary if v['K']>=5][:3]
    atomic_json(root/'configs/probes.json',dict(region=nonempty[:32],development=nonempty[:128],boundary=boundary,boundary_fixed=boundary_fixed,morphology=nonempty[:8]))
    for shift in (0,24):
        files=natsort.natsorted((BASELINE/f'expr_maps/expr_maps_input_patches_48x48_shift_{shift}').glob('*.hdf5'))
        atomic_json(root/f'configs/predict_shift{shift}.json',[dict(index=i,path=str(path),row=coords(path)[0],col=coords(path)[1],identity=identity(path)) for i,path in enumerate(files)])
    ids=np.unique(nuc);ids=ids[ids>0];boxes=ndi.find_objects(nuc)
    with h5py.File(BASELINE/'nuclei_cell_type.h5') as f:tmap=dict(zip(f['ids'][:].astype(int),f['data'][:].astype(int)))
    if set(map(int,ids))!=set(tmap):raise ValueError('nucleus annotation identity mismatch')
    centroid=[];covariance=[];area=[];clipped=[]
    for cid in ids:
        box=boxes[cid-1];yy,xx=np.where(nuc[box]==cid);yy+=box[0].start;xx+=box[1].start
        xy=np.stack((xx,yy),1).astype(float);mu=xy.mean(0);centroid.append(mu)
        covariance.append((xy-mu).T@(xy-mu)/len(xy));area.append(len(xy))
        clipped.append(bool((yy==0).any() or (xx==0).any() or (yy==nuc.shape[0]-1).any() or (xx==nuc.shape[1]-1).any()))
    area=np.asarray(area,np.int64);id_to_row=np.full(int(ids.max())+1,-1,np.int64);id_to_row[ids]=np.arange(len(ids))
    metadata=dict(ids=ids,centroid_xy=np.asarray(centroid),covariance=np.asarray(covariance),area=area,
                  type_index=np.asarray([tmap[int(cid)] for cid in ids],np.int64),specimen_clipped=np.asarray(clipped))
    with (p/'nucleus_metadata.tmp').open('wb') as stream:np.savez(stream,**metadata)
    os.replace(p/'nucleus_metadata.tmp',p/'nucleus_metadata.npz')
    train_rows=id_to_row[np.asarray(sorted(unique_train))];logr=np.log1p(np.maximum(np.sqrt(area[train_rows]/np.pi),.5))
    geom_stats=dict(unique_nuclei=len(train_rows),mean_log_radius=float(logr.mean()),sd_log_radius=float(logr.std(ddof=0)))
    counts=np.zeros((len(ids),len(genes)),np.int64);coverage=np.zeros(nuc.shape,np.uint8);maps=[];raw_total=0;nuclear_total=0
    for file in natsort.natsorted((BASELINE/'expr_maps').glob('all_genes_*.hdf5')):
        y0,y1,x0,x1=coords(file);y1=min(y1,nuc.shape[0]);x1=min(x1,nuc.shape[1])
        if coverage[y0:y1,x0:x1].any():raise ValueError('expression map core overlaps')
        coverage[y0:y1,x0:x1]=1
        with h5py.File(file) as f:
            data=f['data']
            if data.shape[0]<y1-y0 or data.shape[1]<x1-x0 or data.shape[2]!=len(genes):raise ValueError('raw map shape mismatch')
            for start_row in range(y0,y1,64):
                end=min(start_row+64,y1)
                values=data[start_row-y0:end-y0,:x1-x0,:]
                labels=nuc[start_row:end,x0:x1]
                if not np.isfinite(values).all() or (values<0).any():raise ValueError('invalid raw expression values')
                if not np.equal(values,np.floor(values)).all():raise ValueError('noninteger raw HDF5 counts')
                add_counts(counts,labels,values,id_to_row)
                raw_total+=int(values.sum(dtype=np.int64));nuclear_total+=int(values[labels>0].sum(dtype=np.int64))
        maps.append(dict(source=identity(file),core=[y0,y1,x0,x1],shape=list(data.shape) if False else None))
        print('full nucleus counts core',file.name,'nuclear total',nuclear_total,flush=True)
    if not coverage.all() or int(counts.sum())!=nuclear_total:raise ValueError('full-grid coverage/count conservation failed')
    atomic_npy(p/'full_nuclear_counts.npy',counts)
    # Independently read fixed patch samples, including near map remainder edges.
    patches=[views[i] for i in np.linspace(0,len(views)-1,32,dtype=int)]
    alignment=[]
    for view in patches:
        y,x=view['row'],view['col']
        a=np.zeros((48,48,len(genes)),np.uint8);seen=np.zeros((48,48),bool)
        for source in maps:
            sy,ey,sx,ex=source['core'];yy0=max(y,sy);yy1=min(y+48,ey);xx0=max(x,sx);xx1=min(x+48,ex)
            if yy1<=yy0 or xx1<=xx0:continue
            with h5py.File(source['source']['path']) as f:a[yy0-y:yy1-y,xx0-x:xx1-x]=f['data'][yy0-sy:yy1-sy,xx0-sx:xx1-sx]
            seen[yy0-y:yy1-y,xx0-x:xx1-x]=True
        if not seen.all():raise ValueError('patch not covered by expression cores')
        with h5py.File(view['path']) as f:b=f['data'][:]
        if not np.array_equal(a,b):raise ValueError('raw-map/native-patch coordinate mismatch')
        alignment.append(dict(path=view['path'],counts=int(b.sum()),equal=True))
    dapi=tifffile.imread(BASELINE/'dapi_resized.tif')
    if dapi.shape!=nuc.shape or not np.isfinite(dapi).all() or (dapi<0).any():raise ValueError('DAPI grid/value mismatch')
    intensity=np.log1p(dapi.astype(np.float32));q1,q99=np.quantile(intensity[union],[.01,.99],method='linear')
    if q99<=q1:raise ValueError('DAPI has no training contrast')
    intensity=np.clip((intensity-q1)/(q99-q1),0,1)
    features=np.lib.format.open_memmap(p/'dapi_features.tmp.npy',mode='w+',dtype=np.float32,shape=(3,*nuc.shape))
    for y in range(0,nuc.shape[0],256):
        y0=max(0,y-3);y1=min(nuc.shape[0],y+256+3);stop=min(y+256,nuc.shape[0]);im=intensity[y0:y1]
        gx=ndi.sobel(im,axis=1,mode='reflect');gy=ndi.sobel(im,axis=0,mode='reflect')
        edge=np.clip(np.sqrt(gx*gx+gy*gy)/4,0,1);local=im-ndi.uniform_filter(im,size=5,mode='reflect')
        sl=slice(y-y0,stop-y0);features[:,y:stop]=np.stack((im[sl],edge[sl],local[sl]))
    features.flush();del features
    os.replace(p/'dapi_features.tmp.npy',p/'dapi_features.npy')
    # DAPI provenance + overlays: frozen resized image and actual nuclear edges,
    # expression intensity at corners/centre and density extremes.
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    points=[(0,0),(0,nuc.shape[1]-256),(nuc.shape[0]-256,0),(nuc.shape[0]-256,nuc.shape[1]-256),
            (nuc.shape[0]//2-128,nuc.shape[1]//2-128)]
    density=[(len(v['candidate_ids']),v['row'],v['col']) for v in views]
    for _,y,x in (min(density),max(density),sorted(density)[len(density)//2],*sorted(density)):
        point=(min(y,nuc.shape[0]-256),min(x,nuc.shape[1]-256))
        if point not in points:points.append(point)
        if len(points)==8:break
    fig,axes=plt.subplots(2,4,figsize=(16,8));overlays=[]
    for ax,(y,x) in zip(axes.ravel(),points):
        im=intensity[y:y+256,x:x+256];labels=nuc[y:y+256,x:x+256]
        ax.imshow(im,cmap='gray',vmin=0,vmax=1);ax.contour(labels>0,levels=[.5],colors=['cyan'],linewidths=.25)
        ax.set_title(f'row {y}, col {x}');ax.axis('off')
        overlays.append(dict(row=y,col=x,nucleus_intensity=float(im[labels>0].mean()) if (labels>0).any() else None,
                             background_intensity=float(im[labels==0].mean()) if (labels==0).any() else None))
    fig.tight_layout();fig.savefig(p/'dapi_alignment_overlays.png',dpi=120);plt.close(fig)
    seed_all();from bidcell.model.model.model import SegmentationModel
    model=SegmentationModel(n_channels=313);atomic_torch(p/'shared_initial.pth',dict(model=model.state_dict(),seed=1234,rng=rng_state(),trained=False,updates=0))
    dependencies={k:identity(v) for k,v in dict(nuclei=BASELINE/'nuclei.tif',annotation=BASELINE/'nuclei_cell_type.h5',
        genes=BASELINE/'all_gene_names.txt',dapi=BASELINE/'dapi_resized.tif',baseline_config=BASE_CONFIG,
        reference=cfg.files.fp_ref,positive=cfg.files.fp_pos_markers,negative=cfg.files.fp_neg_markers).items()}
    write_configs(root)
    manifest=dict(status='succeeded',version=VERSION,dependencies=dependencies,preparation_source=preparation_source,
                  configs={f.name:identity(f) for f in (root/'configs').glob('*.json') if f.name!='queue.json'},shape=list(nuc.shape),genes=genes,
                  coordinate='resized global row,column; 1um/pixel; exact raw map nonoverlapping cores',
                  raw_maps=maps,raw_count_total=raw_total,nuclear_count_total=nuclear_total,map_pixels=int(coverage.sum()),
                  patch_alignment=alignment,geometry_stats=geom_stats,unknown_nuclei=int((metadata['type_index']==-1).sum()),
                  dapi=dict(q1=float(q1),q99=float(q99),training_unique_pixels=int(union.sum()),quantile='linear exact unique training pixels',
                            clipped_low=float((intensity==0).mean()),clipped_high=float((intensity==1).mean()),overlays=overlays,
                            source='frozen baseline dapi_resized.tif from morphology_mip; no resize/shift'),
                  transforms={f'{f},{r}':transform_matrix(f,r).tolist() for f in (0,1) for r in range(4)},
                  wall_seconds=time.monotonic()-start,
                  outputs={name:identity(p/name) for name in ['nucleus_metadata.npz','full_nuclear_counts.npy','dapi_features.npy','dapi_alignment_overlays.png','shared_initial.pth']})
    atomic_json(p/'manifest.json',manifest)
    return manifest


def write_configs(root):
    root=Path(root);old=read(WORK/'outputs/cos_bs4_20261012T013909+1100/configs/queue.json')
    def put(path,value):
        path=Path(path)
        if path.exists():
            if read(path)!=value:raise ValueError('existing configuration differs: '+str(path))
        else:atomic_json(path,value)
    for key in ('population','paper_protocol','reference','engineering_protocol'):
        src=Path(old[key]['path'] if key=='population' else old[key]);dest=root/'configs'/src.name
        if not dest.exists():shutil.copyfile(src,dest)
    queue=dict(version=VERSION,output_root=str(root),python='/home/zwan0726/miniforge3/envs/bidcell_local/bin/python',
               source_root=str(REPO),arm_order=['V1_BOUNDARY','V2_MORPH','V1_REGION'],methods={},seed=1234,updates=3625,
               shared_initial=str(root/'prepared/shared_initial.pth'),prepared=old['prepared'],
               reference=str(root/'configs/reference.json'),engineering_protocol=str(root/'configs/protocol.json'),
               paper_protocol=str(root/'configs/evaluation_protocol.json'),population=identity(root/'configs/evaluation_population.csv'),
               gpu_lock=str(WORK/'outputs/cos_gpu.lock'),minimum_free_gb=20,
               review_receipt=str(root/'admission/independent_review.json'),admission=str(root/'admission/complete.json'),
               stage_timeouts=dict(train=43200,predict=43200,postprocess=28800,matrix=10800,evaluate=21600,report=3600),
               retries=1,cpu_workers=4)
    base_machine=read(REPO/'scripts/crgf/configs/machine.json') if (REPO/'scripts/crgf/configs/machine.json').exists() else read(WORK/'outputs/crgf_egr_20261008T121953+1100/configs/B0_machine.json')
    for arm in ARMS:
        ar=root/arm;(ar/'configs').mkdir(parents=True,exist_ok=True)
        cfg=dict(version=VERSION,arm=arm,output_root=str(ar),runroot=str(root),queue_config=str(root/'configs/queue.json'),
                 seed=1234,updates=3625,lr=1e-5,betas=[.9,.999],weight_decay=1e-4,precision='float32; no AMP; TF32 false',
                 initial=queue['shared_initial'],views=str(root/'configs/views.json'),reference=str(root/'configs'/('reference_region.json' if arm=='V1_REGION' else 'reference_boundary.json')),
                 prepared_manifest=str(root/'prepared/manifest.json'),calibration=str(root/'admission'/f'{arm}_calibration.json'))
        put(ar/'configs/run.json',cfg)
        machine=dict(base_machine,baseline_final_mask=base_machine['final_mask'],final_mask=str(ar/'predictions'/arm/'final.tif'),pre_mask=str(ar/'predictions'/arm/'pre.tif'),
                     final_matrix=str(ar/'matrix'/arm/'expr_mat.csv'),checkpoint=str(ar/'checkpoints/final.pth'))
        put(ar/'configs/machine.json',machine)
        queue['methods'][arm]=dict(config=str(ar/'configs/run.json'),machine=str(ar/'configs/machine.json'),
                    matrix=str(ar/'matrix'/arm),evaluation=str(ar/'evaluation'/arm),cost_root=str(ar))
    queue['comparison_methods']=dict(queue['methods'])
    for baseline,matrix in [('B0',WORK/'outputs/remote_prep_20261007T122955/matrix_final'),
                            ('B1',WORK/'outputs/crgf_1008_20261008T031528+1100/matrix/B1')]:
        machine=read(WORK/f'outputs/crgf_egr_20261008T121953+1100/configs/{baseline}_machine.json')
        machine.update(final_matrix=str(matrix/'expr_mat.csv'),baseline_final_mask=base_machine['final_mask'])
        machine_path=root/'configs'/f'{baseline}_machine.json';put(machine_path,machine)
        queue['comparison_methods'][baseline]=dict(machine=str(machine_path),matrix=str(matrix),evaluation=str(root/'baselines'/baseline/'evaluation'))
    put(root/'configs/queue.json',queue)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);args=parser.parse_args()
    print(json.dumps(prepare(args.root),indent=2))
