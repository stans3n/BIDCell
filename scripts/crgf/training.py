"""Matched full-pool continuation with exact-update endpoint and strict resume."""
import json
import os
import random
import signal
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from bidcell.model.dataio.dataset_input import DataProcessing
from bidcell.model.model.model import SegmentationModel
from bidcell.model.model.losses import (NucleiEncapsulationLoss, Oversegmentation,
    CellCallingLoss, OverlapLoss, PosNegMarkerLoss)
from .common import atomic_json, atomic_torch, digest, identity, sha256, source_identity

TRAIN_SOURCES = ['bidcell/model/model/model.py','bidcell/model/model/relation_geometry.py',
 'bidcell/model/dataio/dataset_input.py','bidcell/model/dataio/candidate_geometry.py',
 'bidcell/model/model/losses.py','bidcell/model/model/layers.py','bidcell/model/model/intialisation.py',
 'scripts/crgf/training.py','scripts/crgf/config.py','scripts/crgf/common.py']


def seed_all(seed):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.use_deterministic_algorithms(True, warn_only=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(s):
    random.setstate(s['python']); np.random.set_state(s['numpy']); torch.set_rng_state(s['torch'].cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in s['cuda']])


def make_model(cfg, method, genes):
    seed_all(cfg.seed)
    model = SegmentationModel(n_channels=genes, fusion=method=='F1')
    old = torch.load(cfg.checkpoint, map_location='cpu')['model_state_dict']
    result = model.load_state_dict(old, strict=method=='B1')
    if method=='F1':
        expected = {'relation_geometry.'+k for k in model.relation_geometry.state_dict()}
        if set(result.missing_keys) != expected or result.unexpected_keys:
            raise ValueError(f'warm start backbone mismatch {result}')
    return model.cuda()


def optimizer(model,cfg):
    return torch.optim.Adam(model.parameters(),lr=cfg.learning_rate,betas=cfg.betas,eps=cfg.eps,weight_decay=cfg.weight_decay)


def losses(logits, sample):
    n,s,p,q = [sample[k] for k in ('n','s','p','q')]
    device = logits.device
    ne=NucleiEncapsulationLoss(1,device)(logits,n)
    os_=Oversegmentation(1,device)(logits,n)
    cc=CellCallingLoss(1,device)(logits,s)
    ov=OverlapLoss(1,device)(logits,n)
    pn=PosNegMarkerLoss(1,1,device)(logits,p,q)
    pos=F.cross_entropy(logits,p[:,0],reduction='sum')/len(logits)
    neg=(torch.sigmoid(F.softmax(logits,1)[:,1]-.5)*q[:,0]).sum()/len(logits)
    # Backprop through exact original five-return sum. POS/NEG are logging only.
    total=ne+os_+cc+ov+pn
    return total,dict(ne=ne,os=os_,cc=cc,ov=ov,pos=pos,neg=neg)


def to_device(sample):
    return {k:(v.cuda() if torch.is_tensor(v) else v) for k,v in sample.items()}


def make_views(dataset, cfg):
    # Separate RNG; model/module construction cannot change sample order.
    rng = np.random.RandomState(cfg.seed)
    views=[]; nonempty=0; epoch=0
    while nonempty<cfg.updates:
        for index in rng.permutation(len(dataset)):
            flip=int(rng.randint(2)); rotation=int(rng.randint(4))
            path=dataset.fp_patches[index]
            import re
            y,x=map(int,re.findall(r'\d+',Path(path).name)[:2])
            k=int(np.count_nonzero(np.unique(dataset.nuclei[y:y+48,x:x+48])))
            views.append(dict(index=int(index),flip=flip,rotation=rotation,k=k,epoch=epoch,patch=path))
            nonempty += k>0
            if nonempty==cfg.updates: break
        epoch+=1
        if epoch>1 and nonempty==0: raise ValueError('entire train pool empty')
    return dict(version='numpy-RandomState-permutation-flip-rot-v1',seed=cfg.seed,
                train_pool=dataset.fp_patches,views=views,total_updates=cfg.updates)


def dependencies(cfg, method):
    root=Path(cfg.output_root)
    inputs=json.loads((root/'configs/inputs.json').read_text())
    return dict(method=method,method_version=cfg.method_version,source=source_identity(TRAIN_SOURCES),
                config=cfg.model_dump(exclude={'output_root','stage_timeouts','minimum_free_gb'}),
                inputs=inputs,views=sha256(root/'configs/views.json'),candidate_contract='ascending IDs, complete group, crop S/full nucleus d')


def save(path, model, opt, deps, step, cursor, seen):
    import subprocess
    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[2],text=True).strip()
    atomic_torch(path,dict(source_commit=source_commit,architecture={k:list(v.shape) for k,v in model.state_dict().items()},rng_version='python-numpy-torch-cuda-v1',sampler_version='frozen-views-cursor-v1',model_state_dict=model.state_dict(),optimizer_state_dict=opt.state_dict(),
                 dependencies=deps,update_step=step,visited_views=cursor,sampler_cursor=cursor,
                 unique_patches=sorted(seen),rng=rng_state(),scheduler=None,scaler=None,
                 gene_order=deps['inputs']['gene_order'],BN_state='included in strict model_state_dict'))


def load(path, model, opt, deps):
    state=torch.load(path,map_location='cpu')
    if state['dependencies'] != deps: raise ValueError('checkpoint dependency mismatch')
    if state.get('architecture')!={k:list(v.shape) for k,v in model.state_dict().items()}:raise ValueError('checkpoint architecture mismatch')
    model.load_state_dict(state['model_state_dict'],strict=True)
    opt.load_state_dict(state['optimizer_state_dict']);restore_rng(state['rng'])
    return state['update_step'],state['sampler_cursor'],set(state['unique_patches'])


def train(cfg,method):
    root=Path(cfg.output_root);out=root/'checkpoints'/method;out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(cfg.cpu_workers)
    ds=DataProcessing(cfg.native(),isTraining=True,total_steps=None,structured=True)
    manifest=json.loads((root/'configs/views.json').read_text())
    if ds.fp_patches!=manifest['train_pool']: raise ValueError('train pool changed')
    deps=dependencies(cfg,method);model=make_model(cfg,method,len(ds.gene_names));opt=optimizer(model,cfg)
    latest=out/'latest.pth';step=cursor=0;seen=set()
    if latest.exists(): step,cursor,seen=load(latest,model,opt,deps)
    model.train();stop=[]
    for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig,lambda s,f:stop.append(s))
    started=time.monotonic();initial_step=step;attempt_identity=f'{os.getpid()}-{time.time_ns()}'
    with (out/'updates.jsonl').open('a',buffering=1) as log:
        while step<cfg.updates:
            view=manifest['views'][cursor];ds.view_transform=(view['flip'],view['rotation'])
            t=time.monotonic();sample=ds[view['index']];geometry_seconds=time.monotonic()-t
            if len(sample['cell_ids'])!=view['k']: raise ValueError('candidate identity/count changed')
            cursor+=1
            if view['k']==0: continue
            sample=to_device(sample);opt.zero_grad(set_to_none=True)
            logits=model(sample['x'],sample['geometry'],sample['s'])
            total,parts=losses(logits,sample)
            if not torch.isfinite(total): raise FloatingPointError('nonfinite loss')
            total.backward()
            for name,p in model.named_parameters():
                if p.grad is None or not torch.isfinite(p.grad).all(): raise FloatingPointError('missing/nonfinite gradient '+name)
            opt.step();step+=1;seen.add(view['index']);torch.cuda.synchronize()
            entry=dict(attempt_identity=attempt_identity,resumed_from_update=initial_step,update=step,visited_views=cursor,k=view['k'],patch=view['patch'],flip=view['flip'],rotation=view['rotation'],
                       unique_patches=len(seen),pool_size=len(ds),loss=float(total),losses={k:float(v.detach()) for k,v in parts.items()},
                       seconds=time.monotonic()-t,data_and_geometry_seconds=geometry_seconds,peak_vram_bytes=torch.cuda.max_memory_allocated())
            log.write(json.dumps(entry)+'\n')
            if step%10==0 or step==1: print(json.dumps(entry),flush=True)
            atomic_json(out/'progress.json',dict(**entry,elapsed_seconds=time.monotonic()-started,
                        measured_updates=step-initial_step,remaining_updates=cfg.updates-step))
            if step%cfg.checkpoint_every==0 or step in (cfg.early_checkpoint,cfg.updates) or stop:
                save(latest,model,opt,deps,step,cursor,seen)
                # Retain fixed milestones without duplicating bytes (atomic source is replaced next save).
                checkpoint=out/f'update_{step:06d}.pth'
                if not checkpoint.exists(): os.link(latest,checkpoint)
                atomic_json(out/'latest.json',dict(update=step,checkpoint=identity(latest),dependencies_hash=digest(deps)))
            if stop: raise SystemExit(128+stop[-1])
    final=out/'final.pth'
    if not final.exists(): os.link(latest,final)
    atomic_json(out/'complete.json',dict(status='succeeded',updates=step,visited_views=cursor,unique_patches=len(seen),pool_size=len(ds),
                dependencies=deps,outputs={'checkpoint':identity(final)},wall_seconds=time.monotonic()-started,
                peak_vram_bytes=torch.cuda.max_memory_allocated()))
