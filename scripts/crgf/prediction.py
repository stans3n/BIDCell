"""Full two-shift prediction, resumable atomic 128-patch groups, native stitching."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from bidcell.model.dataio.dataset_input import DataProcessing
from bidcell.model.model.model import SegmentationModel
from bidcell.model.utils.utils import get_seg_mask
from bidcell.model.predict import gap_coords
from .common import atomic_json, atomic_npy, atomic_tiff, identity, sha256, source_identity, verify_outputs
from .training import to_device, seed_all, dependencies


def decode(logits, nuclei, ids):
    actual=np.unique(nuclei);actual=actual[actual>0]
    if not np.array_equal(ids,actual): raise ValueError('decode candidate ID/order mismatch')
    if not len(ids): return np.zeros(nuclei.shape,np.uint32)
    if logits.shape[0]!=len(ids): raise ValueError('decode candidate count mismatch')
    return get_seg_mask(logits,nuclei).astype(np.uint32)


def merge_native(pred, shifted, paths, patch_size=48):
    import re
    coords=[list(map(int,re.findall(r'\d+',Path(p).name)[:2])) for p in paths]
    hs=[c[0] for c in coords];ws=[c[1] for c in coords]
    fill=np.zeros(pred.shape,np.uint8)
    for i in range(int(patch_size*.35),int(patch_size*.65)):
        fill[np.asarray(hs)+i,:]=1;fill[:,np.asarray(ws)+i]=1
    fill[gap_coords(hs,patch_size),:]=0;fill[:,gap_coords(ws,patch_size)]=0
    return np.where(fill,shifted,pred).astype(np.uint32),fill


def predict(cfg,method):
    root=Path(cfg.output_root);out=root/'predictions'/method;out.mkdir(parents=True,exist_ok=True)
    checkpoint=root/'checkpoints'/method/'final.pth'
    state=torch.load(checkpoint,map_location='cpu')
    if state['dependencies']!=dependencies(cfg,method):raise ValueError('prediction checkpoint dependency mismatch')
    if state['update_step']!=cfg.updates: raise ValueError('wrong checkpoint endpoint')
    deps=dict(checkpoint=identity(checkpoint),config=cfg.model_dump(mode="json"),
              inputs=json.loads((root/'configs/inputs.json').read_text()),
              source=source_identity(['scripts/crgf/prediction.py','bidcell/model/model/model.py',
              'bidcell/model/model/relation_geometry.py','bidcell/model/dataio/dataset_input.py',
              'bidcell/model/dataio/candidate_geometry.py','bidcell/model/utils/utils.py','bidcell/model/predict.py']))
    seed_all(cfg.seed);torch.set_num_threads(cfg.cpu_workers)
    genes=state['gene_order'];model=SegmentationModel(n_channels=len(genes),fusion=method=='F1').cuda()
    model.load_state_dict(state['model_state_dict'],strict=True);del state;model.eval()
    all_manifests=[];provider=None;started=time.monotonic();completed_total=0
    for shift in cfg.shifts:
        ds=DataProcessing(cfg.native(),isTraining=False,all_patches=True,shift_patches=shift,structured=True,geometry_provider=provider)
        expected=json.loads((root/f'configs/patches_shift{shift}.json').read_text())
        if ds.fp_patches!=[p['path'] for p in expected]: raise ValueError('inference pool changed')
        whole=np.zeros(ds.nuclei.shape,np.uint32);covered=np.zeros(ds.nuclei.shape,bool)
        chunks=out/f'shift{shift}_patches';chunks.mkdir(exist_ok=True)
        chunk_receipts=[]
        for start in range(0,len(ds),128):
            stop=min(start+128,len(ds));rec=chunks/f'{start:06d}.json';arrpath=chunks/f'{start:06d}.npy'
            chunkdeps=dict(dependencies=deps,start=start,stop=stop,patches=expected[start:stop])
            if rec.exists():
                receipt=json.loads(rec.read_text())
                if receipt['dependencies']!=chunkdeps: raise ValueError('prediction chunk dependencies changed')
                verify_outputs(receipt);masks=np.load(arrpath)
            else:
                masks=np.empty((stop-start,48,48),np.uint32)
                with torch.inference_mode():
                    for j,index in enumerate(range(start,stop)):
                        sample=ds[index]
                        if len(sample['cell_ids']):
                            device=to_device(sample)
                            logits=model(device['x'],device['geometry'],device['s']).cpu().numpy()
                            if not np.isfinite(logits).all(): raise FloatingPointError('nonfinite prediction')
                        else: logits=None
                        masks[j]=decode(logits,sample['nuclei'],sample['cell_ids'])
                        completed_total+=1
                atomic_npy(arrpath,masks)
                receipt=dict(status='succeeded',dependencies=chunkdeps,outputs={'masks':identity(arrpath)})
                atomic_json(rec,receipt)
            import re
            for j,path in enumerate(ds.fp_patches[start:stop]):
                y,x=map(int,re.findall(r'\d+',Path(path).name)[:2])
                whole[y:y+48,x:x+48]=masks[j];covered[y:y+48,x:x+48]=True
            chunk_receipts.append(identity(rec))
            progress=dict(shift=shift,completed_patches=stop,total_patches=len(ds),new_patches=completed_total,
                          elapsed_seconds=time.monotonic()-started,peak_vram_bytes=torch.cuda.max_memory_allocated())
            atomic_json(out/'progress.json',progress);print(json.dumps(progress),flush=True)
        provider=ds.geometry_provider
        # Keep native shift maps unchanged; global nucleus recovery occurs after merge.
        atomic_tiff(out/f'shift{shift}.tif',whole)
        atomic_tiff(out/f'coverage_shift{shift}.tif',covered.astype(np.uint8))
        all_manifests.append(dict(shift=shift,patch_count=len(ds),chunks=chunk_receipts,covered_fraction=float(covered.mean())))
        if shift==0: pred=whole
        else: shifted=whole;shifted_paths=ds.fp_patches
    pre,fill=merge_native(pred,shifted,shifted_paths)
    atomic_tiff(out/'pre.tif',pre);atomic_tiff(out/'fill.tif',fill)
    atomic_json(out/'complete.json',dict(status='succeeded',dependencies=deps,shifts=all_manifests,
                outputs={p.name:identity(p) for p in [out/'pre.tif',out/'shift0.tif',out/'shift24.tif',out/'fill.tif']},
                wall_seconds=time.monotonic()-started,peak_vram_bytes=torch.cuda.max_memory_allocated()))
