"""Both full native shifts, complete candidate groups and exact native merge."""
import gc
import time
import numpy as np
from bidcell.model.utils.utils import get_seg_mask
from scripts.new_methods.prediction import merge_native
from .common import *
from .data import PairedViews
from .training import checkpoint_header


def predict(config):
    cfg=load_config(config);check_environment()
    out=Path(cfg['output_root'])/'predictions'/cfg['run_id'];out.mkdir(parents=True,exist_ok=True)
    ckpt=Path(cfg['output_root'])/'checkpoints/final.pth'
    receipt=read(ckpt.parent/'complete.json');verify_outputs(receipt)
    state=torch.load(ckpt,map_location='cpu')
    if state['header']!=checkpoint_header(cfg) or state['pilot'] or (state['actual_updates'],state['view_cursor'],state['empty_skips'])!=(3625,3998,373):
        raise ValueError('prediction endpoint is not the validated complete formal training')
    model=make_model(cfg['model_kind']);model.load_state_dict(state['model'],strict=True);model.eval();del state
    dep=dict(checkpoint=identity(ckpt),producer=producer('predict'),prepared=identity(INPUT/'prepared/manifest.json'),
             run_id=cfg['run_id'],model_kind=cfg['model_kind'])
    start_time=time.monotonic();torch.cuda.reset_peak_memory_stats();maps=[];chunks_all=[];shift_counts={};max_k=0
    for shift in (0,24):
        data=PairedViews(cfg['model_kind'],training=False,shift=shift)
        expected=read(INPUT/f'configs/predict_shift{shift}.json')
        if [str(p) for p in data.fp_patches]!=[v['path'] for v in expected]:raise ValueError('prediction pool changed')
        whole=np.zeros(data.nuclei.shape,np.uint32);chunks=out/f'shift{shift}_patches';chunks.mkdir(exist_ok=True)
        for start in range(0,len(expected),128):
            stop=min(start+128,len(expected));path=chunks/f'{start:06d}.npy';recpath=path.with_suffix('.json')
            chunk_dep=dict(base=dep,start=start,stop=stop,patches=[v['identity'] for v in expected[start:stop]])
            for v in expected[start:stop]:
                if sha256(v['path'])!=v['identity']['sha256']:raise ValueError('prediction patch hash changed')
            if recpath.exists():
                rec=read(recpath)
                if rec['dependencies']!=chunk_dep:raise ValueError('prediction chunk dependency mismatch')
                verify_outputs(rec);masks=np.load(path)
                if masks.shape!=(stop-start,48,48):raise ValueError('prediction chunk shape mismatch')
            else:
                masks=np.empty((stop-start,48,48),np.uint32);ids=[]
                with torch.inference_mode():
                    for j,v in enumerate(expected[start:stop]):
                        b=data.get_view(v,'cuda');ids.append(b['candidate_ids'].tolist())
                        if len(b['input']):
                            z=forward(model,b,cfg['model_kind'])
                            if not torch.isfinite(z).all():raise FloatingPointError('nonfinite prediction')
                            masks[j]=get_seg_mask(z.cpu().numpy(),b['labels'].cpu().numpy()).astype(np.uint32)
                        else:masks[j]=0
                atomic_npy(path,masks)
                rec=dict(status='succeeded',dependencies=chunk_dep,candidate_ids=ids,outputs={'masks':identity(path)})
                atomic_json(recpath,rec)
            max_k=max(max_k,max(map(len,rec['candidate_ids']),default=0))
            for j,v in enumerate(expected[start:stop]):whole[v['row']:v['row']+48,v['col']:v['col']+48]=masks[j]
            chunks_all.append(identity(recpath));atomic_json(out/'progress.json',dict(status='running',shift=shift,
                completed_patches=stop,total_patches=len(expected),elapsed_seconds=time.monotonic()-start_time,
                peak_vram_bytes=torch.cuda.max_memory_allocated(),process=process_identity()))
            print(cfg['run_id'],'predict',shift,stop,'/',len(expected),flush=True)
        atomic_tiff(out/f'shift{shift}.tif',whole);maps.append(whole);shift_counts[str(shift)]=len(expected)
        if shift==24:shifted_paths=list(data.fp_patches)
        del data;gc.collect()
    pre,fill=merge_native(*maps,shifted_paths);atomic_tiff(out/'pre.tif',pre);atomic_tiff(out/'fill.tif',fill)
    rec=dict(status='succeeded',dependencies=dep,shifts=[0,24],shift_counts=shift_counts,max_K=max_k,chunks=chunks_all,
        wall_seconds=time.monotonic()-start_time,peak_vram_bytes=torch.cuda.max_memory_allocated(),
        outputs={name:identity(out/name) for name in ('pre.tif','shift0.tif','shift24.tif','fill.tif')})
    atomic_json(out/'complete.json',rec);return rec
