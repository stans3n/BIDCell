"""Default B replay: original immutable prediction, no training/prediction/classification."""
import argparse,json,shutil,time
from pathlib import Path
import numpy as np
import tifffile
from scripts.bs4.config import read
from scripts.bs4.postprocess import postprocess,validate_tile
from scripts.bs4.reuse import prediction
from scripts.crgf.common import atomic_json,identity,sha256,verify_outputs

def run(root):
    root=Path(root);out=root/'diagnostics/B_native_replay';(out/'configs').mkdir(parents=True,exist_ok=True)
    cfg=read(root/'T1/configs/run.json').model_copy(update=dict(arm_id='B',output_root=str(out),mini_test=True,reused_prediction_root='/home/zwan0726/work/outputs/cos_loss_20261009T160351+1100/B/predictions/B'))
    # Validation run uses production raster, seed, size, worker count, all native operations.
    atomic_json(out/'configs/run.json',cfg.model_dump(mode='json'))
    shutil.copyfile(root/'configs/probes.json',out/'configs/probes.json')
    prediction(cfg);postprocess(cfg,'B')
    historical=Path(cfg.reused_prediction_root);current=out/'predictions/B'
    endpoints=json.loads((historical/'endpoints.json').read_text())['endpoints'];rows=[]
    for name,obj in endpoints.items():
        if sha256(obj['path'])!=obj['sha256']:raise ValueError('historical endpoint drift')
        a=tifffile.imread(obj['path']);b=tifffile.imread(current/(name+'.tif'))
        if not np.array_equal(a,b) or a.dtype!=b.dtype:raise ValueError('disabled B endpoint changed '+name)
        rows.append(dict(stage=name,historical=obj,replay=identity(current/(name+'.tif')),bitwise=True,shape=list(a.shape),dtype=str(a.dtype)))
    manifest=json.loads((current/'postprocess/manifest.json').read_text())
    for tile in manifest['tiles']:
        receipt=json.loads(Path(tile['path']).read_text());validate_tile(tile['path'],receipt['dependencies'])
        old=historical/'postprocess'/Path(receipt['outputs']['postislands']['path']).name
        if not np.array_equal(tifffile.imread(old),tifffile.imread(receipt['outputs']['postislands']['path'])):raise ValueError('disabled B island tile changed')
    atomic_json(out/'validation.json',dict(status='passed',scope='default B production validation replay; not a formal fifth arm',native_masks=rows,tiles_verified=len(manifest['tiles']),new_postconnect_endpoint=identity(current/'postconnect_canonical.tif'),training='not_required',prediction='reused',classification='not_required',postprocess=identity(current/'postprocess/complete.json')))
    q=json.loads((root/'configs/queue.json').read_text());spec=dict(q['historical_methods']['B']);old_eval=Path(spec['evaluation']);evaluation=out/'evaluation';evaluation.mkdir(exist_ok=True)
    for name in ('paper_aligned_v1','engineering','summary.json'):
        p=evaluation/name
        if not p.exists():p.symlink_to(old_eval/name,target_is_directory=(old_eval/name).is_dir())
    spec.update(evaluation=str(evaluation),endpoints=str(current/'endpoints.json'),population=q['population'],base_variant='B1')
    from scripts.bs4.diagnostics import stage_diagnostics
    stage_diagnostics(spec,q['anchors'],q['prepared'],evaluation/'diagnostics/stages')
    atomic_json(out/'complete.json',dict(status='passed',outputs={'validation':identity(out/'validation.json'),'diagnostics':identity(evaluation/'diagnostics/stages/validation.json')}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();run(a.root)
