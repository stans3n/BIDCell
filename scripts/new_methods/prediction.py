"""Strict endpoint restore; whole-group native inference, both shifts, native merge."""
import re
import time
from pathlib import Path
import numpy as np
import torch
from bidcell.model.utils.utils import get_seg_mask
from bidcell.model.predict import gap_coords
from .common import *
from .data import NativeViews
from .training import checkpoint_header
from .provenance import component_source


def merge_native(pred,shifted,paths):
    coords=[list(map(int,re.findall(r'\d+',Path(p).name)[:2])) for p in paths]
    hs=[c[0] for c in coords];ws=[c[1] for c in coords];fill=np.zeros(pred.shape,np.uint8)
    for i in range(int(48*.35),int(48*.65)):
        fill[np.asarray(hs)+i,:]=1;fill[:,np.asarray(ws)+i]=1
    fill[gap_coords(hs,48),:]=0;fill[:,gap_coords(ws,48)]=0
    return np.where(fill,shifted,pred).astype(np.uint32),fill


def predict(config):
    cfg=read(config) if isinstance(config,(str,Path)) else config;root=Path(cfg['runroot']);arm=cfg['arm']
    out=Path(cfg['output_root'])/'predictions'/arm;out.mkdir(parents=True,exist_ok=True)
    checkpoint=Path(cfg['output_root'])/'checkpoints/final.pth'
    train=read(checkpoint.parent/'complete.json');verify_outputs(train)
    state=torch.load(checkpoint,map_location='cpu')
    if state['header']!=checkpoint_header(cfg) or state['actual_updates']!=cfg['updates']:raise ValueError('prediction endpoint identity mismatch')
    model=make_model(arm,cfg['initial']);model.load_state_dict(state['model'],strict=True);model.eval();del state
    dep=dict(checkpoint=identity(checkpoint),source=component_source('predict'),prepared=identity(cfg['prepared_manifest']))
    start_time=time.monotonic();torch.cuda.reset_peak_memory_stats();whole_maps=[];receipts=[]
    for shift in (0,24):
        data=NativeViews(root,training=False,shift=shift,auxiliary=arm=='V2_MORPH',morphology=arm=='V2_MORPH')
        expected=read(root/f'configs/predict_shift{shift}.json')
        if [str(p) for p in data.fp_patches]!=[v['path'] for v in expected]:raise ValueError('prediction pool changed')
        whole=np.zeros(data.nuclei.shape,np.uint32);chunks=out/f'shift{shift}_patches';chunks.mkdir(exist_ok=True)
        for start in range(0,len(data),128):
            stop=min(start+128,len(data));path=chunks/f'{start:06d}.npy';recpath=path.with_suffix('.json')
            chunk_dep=dict(base=dep,start=start,stop=stop,patches=[identity(v['path']) for v in expected[start:stop]])
            if recpath.exists():
                rec=read(recpath)
                if rec['dependencies']!=chunk_dep:raise ValueError('prediction chunk dependency mismatch')
                verify_outputs(rec);masks=np.load(path)
            else:
                masks=np.empty((stop-start,48,48),np.uint32)
                with torch.inference_mode():
                    for j,index in enumerate(range(start,stop)):
                        batch=data.get_view(dict(index=index),'cuda')
                        if len(batch['input']):
                            z=model(batch['input'],batch.get('aux') if arm=='V2_MORPH' else None)
                            if not torch.isfinite(z).all():raise FloatingPointError('nonfinite prediction')
                            masks[j]=get_seg_mask(z.cpu().numpy(),batch['labels'].cpu().numpy()).astype(np.uint32)
                        else:masks[j]=0
                atomic_npy(path,masks);atomic_json(recpath,dict(status='succeeded',dependencies=chunk_dep,outputs={'masks':identity(path)}))
            for j,v in enumerate(expected[start:stop]):whole[v['row']:v['row']+48,v['col']:v['col']+48]=masks[j]
            receipts.append(identity(recpath));atomic_json(out/'progress.json',dict(shift=shift,completed_patches=stop,total_patches=len(data),
                    elapsed_seconds=time.monotonic()-start_time,peak_vram_bytes=torch.cuda.max_memory_allocated(),process=process_identity()))
            print(arm,'predict shift',shift,stop,'/',len(data),flush=True)
        atomic_tiff(out/f'shift{shift}.tif',whole);whole_maps.append(whole)
        if shift==24:shifted_paths=list(data.fp_patches)
        del data
    pre,fill=merge_native(*whole_maps,shifted_paths);atomic_tiff(out/'pre.tif',pre);atomic_tiff(out/'fill.tif',fill)
    rec=dict(status='succeeded',dependencies=dep,shifts=[0,24],chunks=receipts,wall_seconds=time.monotonic()-start_time,
             peak_vram_bytes=torch.cuda.max_memory_allocated(),outputs={p.name:identity(p) for p in [out/'pre.tif',out/'shift0.tif',out/'shift24.tif',out/'fill.tif']})
    atomic_json(out/'complete.json',rec);return rec
