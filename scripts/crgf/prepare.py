"""Resolve and freeze every dependency before launching the unattended queue."""
import argparse
import json
import os
import subprocess
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from bidcell.model.dataio.dataset_input import DataProcessing
from .common import atomic_json,identity,digest,sha256
from .config import RunConfig,read
from .training import make_views,TRAIN_SOURCES


def prepare(cfg):
    root=Path(cfg.output_root);(root/'configs').mkdir(parents=True,exist_ok=True)
    native=cfg.native();templates=Path(__file__).with_name('configs')
    machine=json.loads((templates/'machine.json').read_text())
    reference=json.loads((templates/'reference.json').read_text());protocol=json.loads((templates/'protocol.json').read_text())
    if machine['baseline_config']!=cfg.baseline_config or machine['checkpoint']!=cfg.checkpoint: raise ValueError('template/B0 mismatch')
    inputs={k:identity(machine[k]) for k in ['baseline_config','checkpoint','nuclei_mask','preannotation','genes','transcripts','reference','positive_markers','negative_markers','final_mask','final_matrix']}
    if inputs['checkpoint']['sha256']!='cf4315f485152d1ac0b02b77c36c5b3452dfaed09ecd2841d5c58c7b0314547c': raise ValueError('B0 checkpoint acceptance mismatch')
    genes=Path(machine['genes']).read_text().splitlines()
    ref=pd.read_csv(machine['reference'],index_col=0)
    if [g for g in ref.columns if g in set(genes)]!=genes:raise ValueError('reference gene order mismatch')
    inputs['gene_order']=genes
    for name in ['positive_markers','negative_markers']:
        table=pd.read_csv(machine[name],index_col=0)
        if any(g not in table.columns for g in genes):raise ValueError('marker genes mismatch')
    with h5py.File(machine['preannotation']) as f:
        inputs['unknown_training_count']=int((f['data'][:]==-1).sum())
    inputs['unknown_training_behavior']='native -1 indexes last reference type; unchanged for both methods; evaluation Unknown stays NA'
    inputs['Rscript']=identity(reference['rscript'])
    inputs['official_sources']={p.name:identity(p) for p in Path(reference['official_sources']).glob('*.R')}
    if not os.access(reference['rscript'],os.X_OK):raise ValueError('Rscript unavailable')
    rcheck=subprocess.run([reference['rscript'],'-e','suppressPackageStartupMessages(library(SpatialExperiment));library(BiocParallel);library(Matrix);cat("dependencies_ok\\n")'],capture_output=True,text=True)
    if rcheck.returncode:raise RuntimeError(rcheck.stdout+rcheck.stderr)
    inputs['R_dependencies_check']=rcheck.stdout
    inventories={}
    for shift in cfg.shifts:
        ds=DataProcessing(native,isTraining=False,shift_patches=shift,all_patches=True)
        inventory=[identity(p) for p in ds.fp_patches]
        atomic_json(root/f'configs/patches_shift{shift}.json',inventory)
        inventories[str(shift)]=dict(count=len(inventory),sha256=sha256(root/f'configs/patches_shift{shift}.json'))
    inputs['patch_inventories']=inventories
    ds=DataProcessing(native,isTraining=True,total_steps=None)
    views=make_views(ds,cfg);atomic_json(root/'configs/views.json',views)
    inputs['type_order']=ds.type_names
    atomic_json(root/'configs/inputs.json',inputs)
    atomic_json(root/'configs/resolved_native.json',native.model_dump(mode='json'))
    atomic_json(root/'configs/run.json',cfg.model_dump())
    atomic_json(root/'configs/reference.json',reference);atomic_json(root/'configs/protocol.json',protocol)
    atomic_json(root/'configs/B0_machine.json',machine)
    for method in ('F1','B1'):
        own=dict(machine);own['baseline_final_mask']=machine['final_mask']
        own['final_mask']=str(root/'predictions'/method/'final.tif')
        own['pre_mask']=str(root/'predictions'/method/'pre.tif')
        own['final_matrix']=str(root/'matrix'/method/'expr_mat.csv')
        atomic_json(root/f'configs/{method}_machine.json',own)
        atomic_json(root/f'configs/{method}_resolved.json',dict(method=method,fusion_enabled=method=='F1',config=cfg.model_dump(),native=native.model_dump(mode='json'),inputs=inputs,views=identity(root/'configs/views.json')))
    atomic_json(root/'configs/prepare_complete.json',dict(status='succeeded',inputs_hash=digest(inputs),config=cfg.model_dump(),outputs={p.name:identity(p) for p in (root/'configs').glob('*.json') if p.name!='prepare_complete.json'},train_pool=len(ds),views=len(views['views']),updates=cfg.updates))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();prepare(read(a.config))
