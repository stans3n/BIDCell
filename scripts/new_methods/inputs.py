"""Frozen input verification at stage boundaries, never in the update loop."""
from pathlib import Path
import h5py
import numpy as np
from .common import *


def verify_small_contract(root):
    root=Path(root);manifest=read(root/'prepared/manifest.json')
    for name,obj in manifest['configs'].items():
        if identity(obj['path'])!=obj:raise ValueError('prepared configuration drift '+name)
    for name,expected in manifest['preparation_source'].items():
        if sha256(REPO/name)!=expected:raise ValueError('prepared producer drift '+name)
    return identity(root/'prepared/manifest.json')


def verify_inputs(root,shifts=(0,24),raw=True):
    root=Path(root);verify_small_contract(root);manifest=read(root/'prepared/manifest.json');verify_outputs(manifest)
    checked={};shapes=[];patches=0
    for key,obj in manifest['dependencies'].items():
        if identity(obj['path'])!=obj:raise ValueError('frozen prepared input drift '+key)
        checked[key]=obj
    if raw:
        for item in manifest['raw_maps']:
            obj=item['source']
            if identity(obj['path'])!=obj:raise ValueError('frozen raw expression map drift')
            with h5py.File(obj['path']) as f:
                shape=list(f['data'].shape);dtype=str(f['data'].dtype)
            y0,y1,x0,x1=item['core']
            if len(shape)!=3 or shape[2]!=313 or shape[0]<y1-y0 or shape[1]<x1-x0:raise ValueError('raw core shape contract')
            shapes.append(dict(source=obj,shape=shape,dtype=dtype,core=item['core']))
    frozen={}
    for shift in shifts:
        path=root/f'configs/predict_shift{shift}.json'
        for view in read(path):
            obj=view['identity'];current=identity(view['path'])
            if current!=obj:raise ValueError('frozen native HDF patch drift '+view['path'])
            frozen[view['path']]=obj;patches+=1
        checked[f'predict_shift{shift}']=identity(path)
    if 0 in shifts:
        for view in read(root/'configs/views.json')['views']:
            if view['path'] not in frozen:raise ValueError('training view outside frozen native patch input')
    return dict(status='succeeded',prepared=identity(root/'prepared/manifest.json'),dependencies=checked,
                raw_shapes=shapes,patch_files_checked=patches,frozen_patch_digest=digest(frozen),
                scope='full SHA256 input and prepared evidence at stage boundary; no per-update hashes')
