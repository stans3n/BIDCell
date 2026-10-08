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
from scripts.crgf.common import atomic_json, atomic_npy, atomic_tiff, identity, sha256, source_identity, verify_outputs
from .training import to_device, seed_all, dependencies, make_model
from .ownership import decode, compact_sample, log_ownership


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
    train_complete=json.loads((checkpoint.parent/'complete.json').read_text());verify_outputs(train_complete)
    state=torch.load(checkpoint,map_location='cpu')
    if state['dependencies']!=dependencies(cfg,method):raise ValueError('prediction checkpoint dependency mismatch')
    if state['update_step']!=cfg.updates: raise ValueError('wrong checkpoint endpoint')
    deps=dict(checkpoint=identity(checkpoint),config=cfg.model_dump(mode="json"),
              inputs=json.loads((root/'configs/inputs.json').read_text()),
              source=source_identity(['scripts/cos/prediction.py','scripts/cos/ownership.py','bidcell/model/model/model.py',
              'bidcell/model/model/relation_geometry.py','bidcell/model/dataio/dataset_input.py',
              'bidcell/model/dataio/candidate_geometry.py','bidcell/model/utils/utils.py','bidcell/model/predict.py']))
    seed_all(cfg.seed);torch.set_num_threads(cfg.cpu_workers)
    genes=state['gene_order'];model=make_model(cfg,method,len(genes))
    model.load_state_dict(state['model_state_dict'],strict=True);del state;model.eval()
    all_manifests=[];provider=None;started=time.monotonic();completed_total=0
    for shift in cfg.shifts:
        ds=DataProcessing(cfg.native(),isTraining=False,all_patches=True,shift_patches=shift,structured=True,geometry_provider=provider,mask_unknown_markers=True)
        expected=json.loads((root/f'configs/patches_shift{shift}.json').read_text())
        if ds.fp_patches!=[p['path'] for p in expected]: raise ValueError('inference pool changed')
        raw_whole=np.zeros(ds.nuclei.shape,np.uint32);whole=np.zeros(ds.nuclei.shape,np.uint32);covered=np.zeros(ds.nuclei.shape,bool)
        chunks=out/f'shift{shift}_patches';chunks.mkdir(exist_ok=True)
        chunk_receipts=[]
        probes=json.loads((root/'configs/probes.json').read_text())['probes']
        probe_indices={p['index'] for p in probes if p['shift']==shift}
        probe_dir=out/'probes';probe_dir.mkdir(exist_ok=True)
        for start in range(0,len(ds),128):
            stop=min(start+128,len(ds));rawpath=chunks/f'{start:06d}_raw.npy';rec=chunks/f'{start:06d}.json';arrpath=chunks/f'{start:06d}.npy'
            for item in expected[start:stop]:
                if sha256(item['path'])!=item['sha256']:raise ValueError('prediction patch content hash mismatch')
            chunkdeps=dict(dependencies=deps,start=start,stop=stop,patches=expected[start:stop])
            if rec.exists():
                receipt=json.loads(rec.read_text())
                if receipt['dependencies']!=chunkdeps: raise ValueError('prediction chunk dependencies changed')
                verify_outputs(receipt);masks=np.load(arrpath);raw_masks=np.load(rawpath)
            else:
                masks=np.empty((stop-start,48,48),np.uint32);raw_masks=np.empty_like(masks)
                with torch.inference_mode():
                    for j,index in enumerate(range(start,stop)):
                        sample=compact_sample(ds[index])
                        if len(sample['cell_ids']):
                            device=to_device(sample)
                            corrected,base,residual=model(device['x'],device['geometry'],device['s'],return_details=True)
                            logits=corrected.cpu().numpy()
                            if not np.isfinite(logits).all(): raise FloatingPointError('nonfinite prediction')
                        else: logits=None
                        raw_masks[j]=decode(logits,sample['nuclei'],sample['cell_ids'],restore=False)
                        masks[j]=decode(logits,sample['nuclei'],sample['cell_ids'])
                        if index in probe_indices:
                            save_probe(probe_dir/f'shift{shift}_{index:06d}.npz',sample,logits,base.cpu().numpy() if logits is not None else None,residual.cpu().numpy() if logits is not None else None,raw_masks[j],masks[j])
                        completed_total+=1
                atomic_npy(arrpath,masks);atomic_npy(rawpath,raw_masks)
                receipt=dict(status='succeeded',dependencies=chunkdeps,outputs={'masks':identity(arrpath),'raw':identity(rawpath),**{p.name:identity(p) for p in probe_dir.glob(f'shift{shift}_*.npz') if start<=int(p.stem.rsplit('_',1)[1])<stop}})
                atomic_json(rec,receipt)
            import re
            for j,path in enumerate(ds.fp_patches[start:stop]):
                y,x=map(int,re.findall(r'\d+',Path(path).name)[:2])
                raw_whole[y:y+48,x:x+48]=raw_masks[j];whole[y:y+48,x:x+48]=masks[j];covered[y:y+48,x:x+48]=True
            chunk_receipts.append(identity(rec))
            progress=dict(shift=shift,completed_patches=stop,total_patches=len(ds),new_patches=completed_total,
                          elapsed_seconds=time.monotonic()-started,peak_vram_bytes=torch.cuda.max_memory_allocated())
            atomic_json(out/'progress.json',progress);print(json.dumps(progress),flush=True)
        provider=ds.geometry_provider
        # Keep native shift maps unchanged; global nucleus recovery occurs after merge.
        atomic_tiff(out/f'shift{shift}.tif',whole);atomic_tiff(out/f'raw_shift{shift}.tif',raw_whole)
        atomic_tiff(out/f'coverage_shift{shift}.tif',covered.astype(np.uint8))
        all_manifests.append(dict(shift=shift,patch_count=len(ds),chunks=chunk_receipts,covered_fraction=float(covered.mean())))
        if shift==0: pred=whole
        else: shifted=whole;shifted_paths=ds.fp_patches
    pre,fill=merge_native(pred,shifted,shifted_paths)
    atomic_tiff(out/'pre.tif',pre);atomic_tiff(out/'fill.tif',fill)
    atomic_json(out/'complete.json',dict(status='succeeded',dependencies=deps,shifts=all_manifests,
                outputs={p.name:identity(p) for p in [out/'pre.tif',out/'shift0.tif',out/'shift24.tif',out/'raw_shift0.tif',out/'raw_shift24.tif',out/'fill.tif']},
                wall_seconds=time.monotonic()-started,peak_vram_bytes=torch.cuda.max_memory_allocated()))


def save_probe(path,sample,logits,base,residual,raw,restored):
    from scipy.special import softmax
    import os
    k=len(sample['cell_ids']);shape=raw.shape
    if k:
        binary=softmax(logits,axis=1);background=binary[:,0].mean(0)
        scores=np.concatenate((background[None],binary[:,1]),axis=0)
        q=scores/scores.sum(0,keepdims=True)
        logq=log_ownership(torch.from_numpy(logits)).numpy()
        sorted_scores=np.sort(scores,axis=0)
        margin=sorted_scores[-1]-sorted_scores[-2]
        cell_sorted=np.sort(binary[:,1],axis=0)
        cell_margin=cell_sorted[-1]-cell_sorted[-2] if k>1 else np.full(shape,np.nan,np.float32)
        log_owner=np.concatenate(([0],sample['cell_ids']))[logq.argmax(0)]
        log_probability_owner_mismatch=int(np.count_nonzero(log_owner!=raw))
    else:
        logits=base=np.empty((0,2,*shape),np.float32);residual=np.empty((0,1,*shape),np.float32)
        binary=logits; background=np.ones(shape,np.float32);q=np.ones((1,*shape),np.float32)
        logq=np.zeros_like(q);margin=np.ones(shape,np.float32);cell_margin=np.full(shape,np.nan,np.float32)
        log_probability_owner_mismatch=0
    entropy=-np.where(q>0,q*np.where(np.isfinite(logq),logq,0),0).sum(0)
    targets=np.zeros(shape,np.int64)
    for i,c in enumerate(sample['cell_ids']): targets[sample['nuclei']==c]=i+1
    anchor=sample['nuclei']>0
    anchor_ce=-logq[(targets[anchor],)+np.nonzero(anchor)] if anchor.any() else np.empty(0,np.float32)
    labels=np.concatenate(([0],sample['cell_ids']))
    scores=np.concatenate((background[None],binary[:,1]),axis=0) if k else background[None]
    ranking=np.argsort(-scores,axis=0,kind='stable')
    second=ranking[1] if k else np.zeros(shape,np.int64)
    cell_ranking=np.argsort(-binary[:,1],axis=0,kind='stable') if k else None
    arrays=dict(top1_id=labels[ranking[0]],top2_id=labels[second] if k else np.full(shape,-1),top1_score=np.take_along_axis(scores,ranking[:1],axis=0)[0],top2_score=np.take_along_axis(scores,second[None],axis=0)[0] if k else np.full(shape,np.nan),cell_top1_id=sample['cell_ids'][cell_ranking[0]] if k else np.full(shape,-1),cell_top2_id=sample['cell_ids'][cell_ranking[1]] if k>1 else np.full(shape,-1),binary_logits=base,corrected_logits=logits,base_binary_probabilities=softmax(base,axis=1) if k else base,corrected_binary_probabilities=binary,residual=residual,
        candidate_ids=sample['cell_ids'],candidate_valid=np.ones(k,bool),decoder_support=np.ones((k,*shape),bool),
        search_area=sample['s'].cpu().numpy(),search_overlap=sample['s'].sum(0).cpu().numpy(),
        background_score=background,q=q,log_q=logq,margin=margin,cell_only_margin=cell_margin,entropy=entropy,
        nuclei=sample['nuclei'],anchor_mask=anchor,anchor_ce=anchor_ce,nuclear_owner_coverage=np.isin(sample['nuclei'],np.concatenate(([0],sample['cell_ids']))),
        raw_owner=raw,restored_owner=restored,k=k,row=sample['row'],col=sample['col'],log_probability_owner_mismatch=log_probability_owner_mismatch)
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f: np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
