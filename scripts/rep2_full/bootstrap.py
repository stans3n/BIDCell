"""Freeze paths and evidence once, after independent admission is available."""
from pathlib import Path
import yaml
from .common import *

def layout():
    for name in ('configs','logs','receipts','progress','shared/data/expr_maps','shared/coordinates','tests','endpoints','review'):
        (ROOT/name).mkdir(parents=True,exist_ok=True)
    old=TRAINING/'shared'
    links={ROOT/'shared/prepared':old/'prepared',ROOT/'shared/configs':old/'configs'}
    for name in ('nuclei.tif','nuclei_cell_type.h5','all_gene_names.txt','dapi_resized.tif'):
        links[ROOT/'shared/data'/name]=old/'data'/name
    links[ROOT/'shared/data/expr_maps/expr_maps_input_patches_48x48_shift_0']=old/'data/expr_maps/expr_maps_input_patches_48x48_shift_0'
    for dst,src in links.items():
        if dst.is_symlink():
            if dst.resolve()!=src.resolve():raise ValueError('read-only input link changed')
        elif dst.exists():raise ValueError('unexpected input copy '+str(dst))
        else:dst.symlink_to(src,target_is_directory=src.is_dir())
    native=yaml.safe_load((TRAINING/'configs/native.yaml').read_text());native['files']['data_dir']=str(ROOT/'shared/data')
    p=ROOT/'configs/inference_native.yaml';text=yaml.safe_dump(native,sort_keys=False)
    if p.exists() and p.read_text()!=text:raise ValueError('native configuration changed')
    p.write_text(text)
    return p

def main():
    p=layout();old=TRAINING/'shared'
    deferred=read(TRAINING/'manifests/deferred_reference_asset_audit.json')
    prepared=Path(deferred['assets']['scClassify_model.rds']['path']).parent
    inputs={n:identity(ROOT/'manifests'/(n+'.json')) for n in AUDITS}
    inputs.update(authority=identity(ROOT/'authority/ISBI2027_Authoritative_Writing_Pack_V4/REP2_FROZEN_MINIMAL_PROTOCOL_V4.md'),
        training_complete=identity(TRAINING/'training_complete.json'),training_config=identity(TRAINING/'configs/run.json'),
        native=identity(TRAINING/'configs/native.yaml'),inference_native=identity(p),
        prepared=identity(TRAINING/'shared/prepared/manifest.json'),protocol=deferred['parameters'])
    for n,arm in ARMS.items():inputs['checkpoint_'+n]=identity(TRAINING/'runs'/arm/'checkpoints/final.pth')
    for name in ('reference_metadata.rds','scClassify_model.rds','R_packages.csv','markers.rds','markers.json','reference_genes.txt','reference_mean_logcounts.csv','reference_detection_proportions.csv','type_mapping.csv'):
        inputs['evaluation_'+name]=identity(prepared/name)
    for name in ('nuclei.tif','nuclei_cell_type.h5','all_gene_names.txt','transcripts_processed.csv'):
        inputs['shared_'+name]=identity(old/'data'/name)
    inputs['metadata']=identity(old/'prepared/nucleus_metadata.npz');inputs['reference_region']=identity(old/'configs/reference_region.json')
    cfg=dict(version='REP2-FROZEN-FULL-COMPUTE-v1',seed=1234,bootstrap_seed=20261019,bootstrap_replicates=2000,
        bootstrap_side_um=1000,endpoints=NAMES,eligible=read(old/'data/transcript_summary.json')['rows'],
        shape=read(old/'prepared/manifest.json')['shape'],source_digest=source_digest(),frozen_inputs=inputs,
        evaluation=dict(prepared=str(prepared),protocol=deferred['parameters']['path'],reference_model=str(prepared/'scClassify_model.rds')),
        rscript='/home/zwan0726/work/envs/bidcell_egr_r/bin/Rscript',evaluation_threads=4,source_snapshot=str(SOURCE),
        no_training=True,interpretation='deferred until manual resume',write_scope=str(ROOT))
    path=ROOT/'configs/run.json'
    if path.exists() and read(path)!=cfg:raise ValueError('run already frozen differently; no overwrite')
    atomic_json(path,cfg)
if __name__=='__main__':main()
