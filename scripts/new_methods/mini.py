"""Real ROI, both native shifts, native postprocess, own integer owners and profiles.

Processed transcript IDs/genes are retained. Coordinates are rebased through the
verified native global pixel lookup and the inverse local lookup, preserving each
transcript's exact global quantised pixel without floating crop-resize drift.
"""
import gc
import json
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import tifffile
import torch
from bidcell.model.utils.utils import get_seg_mask
from scripts.crgf.matrix import grid,lookup_axes,build
from scripts.crgf.postprocess import postprocess
from .common import *
from .data import NativeViews
from .prediction import merge_native
from .accepted_profiles import run_profiles
from .provenance import component_source


def mini_chain(root):
    from .runner import lock_file
    root=Path(root);queue=read(root/'configs/queue.json');canonical=root/'admission/mini';canonical.mkdir(exist_ok=True)
    dependencies=dict(pilots={arm:identity(root/'admission'/arm/'pilot_final.pth') for arm in ARMS},
                prepared=identity(root/'prepared/manifest.json'),queue=identity(root/'configs/queue.json'),
                prediction_source=component_source('predict'),evaluation_source=component_source('evaluate'),
                mini_source=identity(__file__))
    if (canonical/'complete.json').exists():
        previous=read(canonical/'complete.json')
        if previous['dependencies']==dependencies:
            verify_outputs(previous);return previous
    # Failed or superseded pilot attempts remain inspectable; own matrix paths
    # are unique, so retry never overwrites a protected completed matrix.
    dest=root/'admission/mini_attempts'/str(time.time_ns());dest.mkdir(parents=True)
    gpu=lock_file(queue['gpu_lock']);views=read(root/'configs/views.json')['views'];view=next(v for v in views if v['K']>=12)
    y0,x0=view['row'],view['col'];size=144
    original=read(queue['methods']['V1_REGION']['machine']);nuc=tifffile.imread(original['nuclei_mask'])
    y0=min(y0,nuc.shape[0]-size);x0=min(x0,nuc.shape[1]-size)
    local_nuc=nuc[y0:y0+size,x0:x0+size];shared=dest/'shared';shared.mkdir(exist_ok=True)
    atomic_tiff(shared/'nuclei.tif',local_nuc.astype(np.uint32))
    xa,ya=lookup_axes((size,size),original['scale_x'],original['scale_y'])
    invx=np.searchsorted(xa,np.arange(size));invy=np.searchsorted(ya,np.arange(size))
    tables=[];source_rows=0
    for chunk in pd.read_csv(original['transcripts'],usecols=['transcript_id','x_location','y_location','feature_name','qv'],chunksize=500000):
        x,y,valid=grid(chunk,nuc.shape,original['scale_x'],original['scale_y'])
        keep=valid&(x>=x0)&(x<x0+size)&(y>=y0)&(y<y0+size);selected=chunk.loc[keep].copy()
        if len(selected):
            localx=x[keep]-x0;localy=y[keep]-y0
            selected['x_location']=invx[localx]*original['scale_x'];selected['y_location']=invy[localy]*original['scale_y']
            checkx,checky,checkvalid=grid(selected,(size,size),original['scale_x'],original['scale_y'])
            if not checkvalid.all() or not np.array_equal(checkx,localx) or not np.array_equal(checky,localy):raise ValueError('ROI coordinate rebase failed')
            tables.append(selected);source_rows+=len(selected)
    if not tables:raise ValueError('real ROI has no transcripts')
    transcripts=pd.concat(tables,ignore_index=True);transcripts.to_csv(shared/'transcripts_processed.csv',index=False)
    roi=dict(row=y0,col=x0,size=size,global_shape=list(nuc.shape),retained_transcripts=source_rows,
             coordinate_rebase='global native pixel -> inverse local cv2 nearest lookup; exact IDs/genes/pixels preserved',
             transcripts_source=identity(original['transcripts']),nuclei_source=identity(original['nuclei_mask']))
    atomic_json(shared/'roi.json',roi)
    models={}
    for arm in ARMS:
        ar=dest/arm;pred=ar/'predictions'/arm;pred.mkdir(parents=True,exist_ok=True)
        cfg=read(queue['methods'][arm]['config']);checkpoint=root/'admission'/arm/'pilot_final.pth';state=torch.load(checkpoint,map_location='cpu')
        if not state.get('pilot') or state['arm']!=arm:raise ValueError('mini requires this arm isolated pilot')
        model=make_model(arm,cfg['initial']);model.load_state_dict(state['model'],strict=True);model.eval();del state
        maps=[];patches=[]
        for shift in (0,24):
            dataset=NativeViews(root,training=False,shift=shift,auxiliary=arm=='V2_MORPH',morphology=arm=='V2_MORPH')
            pool=read(root/f'configs/predict_shift{shift}.json');chosen=[v for v in pool if y0<=v['row'] and v['row']+48<=y0+size and x0<=v['col'] and v['col']+48<=x0+size]
            if not chosen:raise ValueError('real mini ROI lacks both native shifts')
            whole=np.zeros((size,size),np.uint32)
            with torch.inference_mode():
                for v in chosen:
                    b=dataset.get_view(dict(index=v['index']),'cuda')
                    if len(b['input']):
                        z=model(b['input'],b.get('aux') if arm=='V2_MORPH' else None)
                        patch=get_seg_mask(z.cpu().numpy(),b['labels'].cpu().numpy()).astype(np.uint32)
                    else:patch=np.zeros((48,48),np.uint32)
                    y,x=v['row']-y0,v['col']-x0;whole[y:y+48,x:x+48]=patch
            maps.append(whole);atomic_tiff(pred/f'shift{shift}.tif',whole)
            if shift==24:shifted=[f"{v['row']-y0}_{v['col']-x0}.hdf5" for v in chosen]
            patches.append(dict(shift=shift,views=chosen))
            del dataset
        pre,fill=merge_native(*maps,shifted);atomic_tiff(pred/'pre.tif',pre);atomic_tiff(pred/'fill.tif',fill)
        atomic_json(pred/'mini_prediction.json',dict(pilot=identity(checkpoint),roi=roi,patches=patches,outputs={'pre':identity(pred/'pre.tif')}))
        machine=dict(original,nuclei_mask=str(shared/'nuclei.tif'),transcripts=str(shared/'transcripts_processed.csv'),
                     pre_mask=str(pred/'pre.tif'),final_mask=str(pred/'final.tif'),baseline_final_mask=str(pred/'pre.tif'),
                     final_matrix=str(ar/'matrix'/arm/'expr_mat.csv'))
        atomic_json(ar/'machine.json',machine)
        models[arm]=dict(machine=str(ar/'machine.json'),matrix=str(ar/'matrix'/arm),evaluation=str(ar/'evaluation'/arm),cost_root=str(ar))
        del model;gc.collect();torch.cuda.empty_cache()
    gpu.close()
    summaries={};protocol=read(Path(queue['prepared'])/'complete.json')['dependencies']['protocol']['path']
    for arm,spec in models.items():
        ar=Path(spec['cost_root']);machine=read(spec['machine'])
        adapter=SimpleNamespace(output_root=str(ar),seed=1234,postprocess_size=1024,cpu_workers=1,
                                native=lambda:SimpleNamespace(files=SimpleNamespace(data_dir=str(shared))))
        postprocess(adapter,arm)
        final=tifffile.imread(machine['final_mask'])
        if not np.array_equal(final[local_nuc>0],local_nuc[local_nuc>0]):raise ValueError('ROI native nuclear restoration failed')
        matrix=build(machine,spec['matrix'],cache=shared/'coordinate_cache')
        if matrix['totals']['eligible']!=source_rows:raise ValueError('ROI transcript population changed')
        summaries[arm]=run_profiles(spec,queue,protocol,spec['evaluation'],arm)
        if not Path(spec['evaluation'],'paper_aligned_v1/markers_per_cell.csv').exists():raise ValueError('ROI paper profile missing')
        print(arm,'real ROI full native chain complete',flush=True)
    rec=dict(status='succeeded',dependencies=dependencies,scope='real 144x144 global ROI; discarded development endpoints; no performance claim',roi=roi,summaries=summaries,
             outputs=outputs_under(dest),source=source_digest())
    atomic_json(dest/'complete.json',json_safe(rec));atomic_json(canonical/'complete.json',json_safe(rec));return rec
