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
from scripts.crgf.common import atomic_json, atomic_torch, digest, identity, sha256, source_identity

from scripts.cos.ownership import anchor_loss, compact_sample
from .data import dataset
from .method import teacher, replace_pos, stats, target_hash

TRAIN_SOURCES = ['bidcell/model/model/model.py','bidcell/model/model/relation_geometry.py',
 'bidcell/model/dataio/dataset_input.py','bidcell/model/dataio/candidate_geometry.py',
 'bidcell/model/model/losses.py','bidcell/model/model/layers.py','bidcell/model/model/intialisation.py',
 'scripts/bs4/training.py','scripts/bs4/loss_diagnostics.py','scripts/bs4/config.py','scripts/bs4/method.py','scripts/ssp/method.py','scripts/bs4/data.py','scripts/cos/ownership.py','scripts/crgf/common.py']


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


def make_model(cfg, method, genes, device='cuda'):
    if method != cfg.arm_id: raise ValueError('arm/config mismatch')
    seed_all(cfg.seed)
    model=SegmentationModel(n_channels=genes,fusion=False,variant='B1',cos_enabled=True,prototype_projection=None)
    if sha256(cfg.checkpoint) != cfg.parent_sha256: raise ValueError('parent checkpoint hash changed')
    parent=torch.load(cfg.checkpoint,map_location='cpu')
    if parent['update_step'] != cfg.historical_updates: raise ValueError('parent update mismatch')
    old=parent['model_state_dict']
    result=model.load_state_dict(old,strict=False)
    expected={'cos_residual.weight','cos_residual.bias'} if cfg.cos_enabled else set()
    if method=='SP': expected.update(k for k in model.state_dict() if k.startswith('prototype_residual.'))
    if set(result.missing_keys)!=expected or result.unexpected_keys:
        raise ValueError(f'parent state mismatch {result}')
    for name,value in old.items():
        if not torch.equal(model.state_dict()[name],value): raise ValueError('parent tensor changed '+name)
    if cfg.base_variant=='G' and not any(k.startswith('relation_geometry.') for k in old):
        raise ValueError('trained G state absent')
    model=model.to(device)
    seed_all(cfg.seed)  # runtime RNG matched after optional head construction
    return model


def optimizer(model,cfg):
    return torch.optim.Adam(model.parameters(),lr=cfg.learning_rate,betas=cfg.betas,eps=cfg.eps,weight_decay=cfg.weight_decay)


def pos_components(logits, sample, rho=1.):
    """Partition valid-masked POS, always reduced by the real candidate K."""
    k=len(logits)
    if k==0: raise ValueError('K=0 has no POS update')
    p=sample['p'][:,0]; valid=sample['type_valid'].to(logits.device).bool()[:,None,None]
    nu=torch.as_tensor(sample['nuclei'],device=logits.device)
    u=(nu[None]==0)&(sample['s'][:,0]>0)&(p==0)
    masks={'positive':p>0,'u_background':u,'other_background':(p==0)&~u}
    ce=F.cross_entropy(logits,p,reduction='none')
    unweighted={name:(ce*mask*valid).sum()/k for name,mask in masks.items()}
    weighted={name:value*(rho if name=='u_background' else 1.) for name,value in unweighted.items()}
    counts={name:dict(valid_pixels=int((mask&valid).sum()),unknown_pixels=int((mask&~valid).sum()),all_pixels=int(mask.sum())) for name,mask in masks.items()}
    return unweighted,weighted,dict(K=k,total_candidate_pixels=p.numel(),valid_candidates=int(valid.sum()),unknown_candidates=k-int(valid.sum()),U_pixels=int(u.sum()),U_fraction=float(u.float().mean()),regions=counts)


def soft_pos_components(logits,sample):
    t=teacher(sample);valid=sample['type_valid'].bool()[:,None,None];lp=F.log_softmax(logits,1)
    ce=-(t['target']*lp[:,1]+(1-t['target'])*lp[:,0])*t['weight']
    masks={'H_old_p0':t['replace']&(sample['p'][:,0]==0),'H_old_p1':t['replace']&(sample['p'][:,0]>0),'outside_H':valid&~t['replace']}
    return {key:dict(loss=float((ce[mask]).sum()/len(logits)),pixels=int(mask.sum()),weight_sum=float(t['weight'][mask].sum())) for key,mask in masks.items()}


def forward(model,sample,return_details=False):
    return model(sample['x'],sample['geometry'],sample['s'],return_details=return_details)

def losses(logits, sample, cos_enabled=False, anchor_weight=1., pos_background_rho=1., s_enabled=True):
    n,s,p,q = [sample[k] for k in ('n','s','p','q')]
    if len(logits)==0: raise ValueError('K=0 must skip optimizer update')
    valid=sample['type_valid'].to(device=logits.device,dtype=logits.dtype)
    ne=NucleiEncapsulationLoss(1,logits.device)(logits,n)
    os_=Oversegmentation(1,logits.device)(logits,n)
    cc=CellCallingLoss(1,logits.device)(logits,s)
    ov=OverlapLoss(1,logits.device)(logits,n)
    pos=(F.cross_entropy(logits,p[:,0],reduction='sum') if bool(valid.all()) else (F.cross_entropy(logits,p[:,0],reduction='none')*valid[:,None,None]).sum())/len(logits)
    if pos_background_rho != 1.:
        _,weighted,_=pos_components(logits,sample,pos_background_rho)
        pos=sum(weighted.values())
    if s_enabled: pos=replace_pos(pos,logits,sample)
    neg=(torch.sigmoid(F.softmax(logits,1)[:,1]-.5)*q[:,0]*valid[:,None,None]).sum()/len(logits)
    parts=dict(ne=ne,os=os_,cc=cc,ov=ov,pos=pos,neg=neg)
    if cos_enabled: parts['anchor']=anchor_loss(logits,sample['nuclei'],sample['cell_ids'])
    return sum(v for k,v in parts.items() if k!='anchor') + (anchor_weight*parts['anchor'] if cos_enabled else 0),parts


def to_device(sample):
    # Freeze CPU teacher before any model execution; both arms consume identical bytes.
    sample=dict(sample)
    if len(sample['cell_ids']) and '_teacher' not in sample:sample['_teacher']=teacher(sample)
    def move(v):
        if torch.is_tensor(v):return v.cuda()
        if isinstance(v,dict):return {k:move(x) for k,x in v.items()}
        return v
    return {k:move(v) for k,v in sample.items()}


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
    raw=json.loads((root/'configs/inputs.json').read_text())
    inputs={k:raw[k] for k in ('baseline_config','nuclei_mask','preannotation','genes','reference','positive_markers','negative_markers','gene_order','type_order')}
    prepared=json.loads(Path(cfg.prepared_manifest).read_text())
    from scripts.crgf.common import verify_outputs
    verify_outputs(prepared)
    return dict(method=method,method_version=cfg.method_version,source=source_identity(TRAIN_SOURCES),
                config=cfg.model_dump(exclude={'output_root','stage_timeouts','minimum_free_gb'}),
                inputs=inputs,prepared=identity(cfg.prepared_manifest),prepared_outputs=prepared['outputs'],views=sha256(root/'configs/views.json'),frozen=identity(root/'configs/frozen.json'),candidate_contract='ascending IDs, complete group, crop S/full nucleus d')


def save(path, model, opt, deps, step, cursor, seen):
    import subprocess
    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[2],text=True).strip()
    atomic_torch(path,dict(source_commit=source_commit,architecture={k:list(v.shape) for k,v in model.state_dict().items()},rng_version='python-numpy-torch-cuda-v1',sampler_version='frozen-views-cursor-v1',model_state_dict=model.state_dict(),optimizer_state_dict=opt.state_dict(),
                 dependencies=deps,update_step=step,visited_views=cursor,sampler_cursor=cursor,
                 unique_patches=sorted(seen),rng=rng_state(),scheduler=None,scaler=None,
                 gene_order=deps['inputs']['gene_order'],BN_state='included in strict model_state_dict'))


def load(path, model, opt, deps):
    receipt=json.loads(Path(path).with_suffix('.json').read_text())
    if sha256(path)!=receipt['checkpoint']['sha256'] or receipt['dependencies_hash']!=digest(deps):
        raise ValueError('checkpoint receipt hash mismatch')
    state=torch.load(path,map_location='cpu')
    if receipt.get('update')!=state['update_step'] or receipt.get('cursor')!=state['sampler_cursor']:
        raise ValueError('checkpoint receipt update/cursor mismatch')
    def finite(value):
        if torch.is_tensor(value) and value.is_floating_point() and not torch.isfinite(value).all(): raise FloatingPointError('nonfinite checkpoint state')
        if isinstance(value,dict):
            for child in value.values(): finite(child)
        elif isinstance(value,(list,tuple)):
            for child in value: finite(child)
    finite(state['model_state_dict']);finite(state['optimizer_state_dict'])
    if state['dependencies'] != deps: raise ValueError('checkpoint dependency mismatch')
    if state.get('architecture')!={k:list(v.shape) for k,v in model.state_dict().items()}:raise ValueError('checkpoint architecture mismatch')
    if not 0<=state['update_step']<=deps['config']['updates'] or state['sampler_cursor']<state['update_step']:raise ValueError('invalid update/cursor')
    if state.get('gene_order')!=deps['inputs']['gene_order']:raise ValueError('checkpoint gene order mismatch')
    if state.get('rng_version')!='python-numpy-torch-cuda-v1' or state.get('sampler_version')!='frozen-views-cursor-v1':raise ValueError('checkpoint RNG/sampler contract')
    if hasattr(model,'prototype_residual') and not torch.equal(state['model_state_dict']['prototype_residual.projection'].cpu(),model.prototype_residual.projection.detach().cpu()):raise ValueError('fixed projection changed in checkpoint')
    model.load_state_dict(state['model_state_dict'],strict=True)
    opt.load_state_dict(state['optimizer_state_dict']);restore_rng(state['rng'])
    return state['update_step'],state['sampler_cursor'],set(state['unique_patches'])


def publish_checkpoint(out,model,opt,deps,step,cursor,seen):
    out=Path(out);milestone=out/f'update_{step:06d}.pth'
    if milestone.exists():
        receipt_path=milestone.with_suffix('.json')
        if receipt_path.exists():
            receipt=json.loads(receipt_path.read_text())
            if sha256(milestone)!=receipt['checkpoint']['sha256'] or receipt['dependencies_hash']!=digest(deps):
                raise ValueError('existing milestone identity mismatch')
        else:
            # An unpublished partial checkpoint is retained for diagnosis.
            os.replace(milestone,out/f'unpublished_{step}_{time.time_ns()}.pth')
    if not milestone.exists():
        save(milestone,model,opt,deps,step,cursor,seen)
        atomic_json(milestone.with_suffix('.json'),dict(update=step,cursor=cursor,checkpoint=identity(milestone),dependencies_hash=digest(deps)))
    latest=out/'latest.pth';temp=out/'latest.link.tmp'
    if temp.exists():temp.unlink()
    os.link(milestone,temp);os.replace(temp,latest)
    atomic_json(out/'latest.json',dict(update=step,cursor=cursor,checkpoint=identity(latest),milestone=identity(milestone),dependencies_hash=digest(deps)))


def recover_checkpoint(out,deps):
    """Select newest validated milestone, retaining fail-closed corruption checks."""
    out=Path(out);latest=out/'latest.pth';receipt_path=out/'latest.json'
    old=json.loads(receipt_path.read_text()) if receipt_path.exists() else None
    latest_hash=sha256(latest) if latest.exists() else None
    latest_valid=old is not None and latest_hash==old['checkpoint']['sha256']
    candidates=[]
    for rp in sorted(out.glob('update_*.json')):
        receipt=json.loads(rp.read_text());path=rp.with_suffix('.pth')
        if receipt['dependencies_hash']!=digest(deps):raise ValueError('milestone dependency mismatch')
        if not path.exists() or sha256(path)!=receipt['checkpoint']['sha256']:raise ValueError('milestone corruption')
        candidates.append((receipt['update'],path,receipt))
    if not candidates:
        if latest.exists() or receipt_path.exists():raise ValueError('unvalidated latest checkpoint')
        return None
    update,path,receipt=max(candidates,key=lambda x:x[0])
    if latest_valid and old['update']>=update:
        return latest  # load independently verifies receipt counter vs payload.
    if latest.exists() and not latest_valid:
        if latest_hash!=receipt['checkpoint']['sha256']:raise ValueError('latest corruption, not interrupted publication')
        if old is not None and old['update']>=update:raise ValueError('latest receipt corruption')
    temp=out/'latest.recover.tmp'
    if temp.exists():temp.unlink()
    os.link(path,temp);os.replace(temp,latest)
    atomic_json(out/'latest.json',dict(update=update,cursor=receipt['cursor'],checkpoint=identity(latest),milestone=identity(path),dependencies_hash=digest(deps)))
    atomic_json(out/f'publication_recovery_{time.time_ns()}.json',dict(status='recovered_validated_milestone',update=update,checkpoint=identity(path)))
    return latest


def train(cfg,method):
    root=Path(cfg.output_root);out=root/'checkpoints'/method;out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(min(2,cfg.cpu_workers))
    ds=dataset(cfg)
    manifest=json.loads((root/'configs/views.json').read_text())
    if ds.fp_patches!=manifest['train_pool']: raise ValueError('train pool changed')
    teacher_inventory_path=Path(cfg.prepared_manifest).parent/f'coverage_views_{method}.json'
    teacher_inventory={ (r['patch'],r['flip'],r['rotation']):r['target_sha256'] for r in json.loads(teacher_inventory_path.read_text())['rows']} if not cfg.mini_test else {}
    inventory={v['path']:v for v in json.loads((root/'configs/patches_shift0.json').read_text())}
    deps=dependencies(cfg,method);model=make_model(cfg,method,len(ds.gene_names));opt=optimizer(model,cfg)
    latest=out/'latest.pth';step=cursor=0;seen=set()
    resume=recover_checkpoint(out,deps)
    if resume is not None: step,cursor,seen=load(resume,model,opt,deps)
    # A crash may leave log lines newer than the durable checkpoint. Preserve
    # the attempt, then restore one authoritative entry per committed update.
    logpath=out/'updates.jsonl'
    if step>0 and not logpath.exists(): raise ValueError('missing committed update log')
    if logpath.exists():
        original=logpath.read_text(); rows=[json.loads(x) for x in original.splitlines() if x.strip()]
        valid={r['update']:r for r in rows if r['update']<=step}
        if len(valid)!=step or sorted(valid)!=list(range(1,step+1)):
            raise ValueError('checkpoint/log committed update coverage mismatch')
        if len(rows)!=len(valid):
            archive=out/'attempt_history';archive.mkdir(exist_ok=True)
            (archive/f'updates_{time.time_ns()}.jsonl').write_text(original)
            tmp=logpath.with_suffix('.tmp');tmp.write_text(''.join(json.dumps(valid[i])+'\n' for i in sorted(valid)));os.replace(tmp,logpath)
    import hashlib
    initialization=dict(variant=method,parent_checkpoint=identity(cfg.checkpoint),fresh_optimizer_state_count=len(opt.state) if step==0 else None,base_variant=cfg.base_variant,cos_enabled=cfg.cos_enabled,parameters=sum(p.numel() for p in model.parameters()),
        added_parameters=321+(12545 if method=='SP' else 0),prototype_parameters=12545 if method=='SP' else 0,
        tensor_hashes={k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in model.state_dict().items()} if step==0 else {},
        resumed_at=step,model_import=__import__('bidcell.model.model.model',fromlist=['x']).__file__,training_import=__file__)
    if step==0:
        if (out/'initialization.json').exists() and json.loads((out/'initialization.json').read_text())!=initialization:
            raise ValueError('initialization identity changed')
        atomic_json(out/'initialization.json',initialization)
    else:
        if not (out/'initialization.json').exists():raise ValueError('missing original initialization evidence')
        atomic_json(out/f'resume_{time.time_ns()}.json',initialization)
    model.train();stop=[]
    for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig,lambda s,f:stop.append(s))
    started=time.monotonic();initial_step=step;attempt_identity=f'{os.getpid()}-{time.time_ns()}'
    with (out/'updates.jsonl').open('a',buffering=1) as log:
        while step<cfg.updates:
            if step%10==0:
                from .resources import require_space
                try: require_space(root,cfg.mini_test,reserve_bytes=400*1024**2)
                except RuntimeError as error:
                    if 'PAUSED_RESOURCE' not in str(error):raise
                    log.flush();os.fsync(log.fileno())
                    publish_checkpoint(out,model,opt,deps,step,cursor,seen)
                    raise
            view=manifest['views'][cursor]
            if sha256(view['patch'])!=inventory[view['patch']]['sha256']:raise ValueError('training patch content hash mismatch')
            ds.view_transform=(view['flip'],view['rotation'])
            t=time.monotonic();sample=compact_sample(ds[view['index']]);geometry_seconds=time.monotonic()-t
            if list(sample['cell_ids']) != view['cell_ids']: raise ValueError('candidate identity changed')
            if len(sample['cell_ids'])!=view['k']: raise ValueError('candidate identity/count changed')
            cursor+=1
            if view['k']==0: continue
            frozen_teacher=teacher(sample)
            if not cfg.mini_test and target_hash(frozen_teacher)!=teacher_inventory[(view['patch'],view['flip'],view['rotation'])]:raise ValueError('frozen teacher content mismatch')
            sample['_teacher']=frozen_teacher
            sample=to_device(sample);opt.zero_grad(set_to_none=True)
            logits,base_logits,residual=forward(model,sample,return_details=True)
            total,parts=losses(logits,sample,cfg.cos_enabled,cfg.anchor_weight,cfg.pos_background_rho)
            if not torch.isfinite(total): raise FloatingPointError('nonfinite loss')
            total.backward()
            for name,p in model.named_parameters():
                if p.grad is None or not torch.isfinite(p.grad).all(): raise FloatingPointError('missing/nonfinite gradient '+name)
            from .loss_diagnostics import groups, norm
            gradient_norms={g:norm(tuple(p.grad for p in model.parameters()),inds) for g,inds in groups(model).items()}
            decay_norms={g:norm(tuple(cfg.weight_decay*p.detach() for p in model.parameters()),inds) for g,inds in groups(model).items()}
            milestone=(step+1) in (100,1000,2000,4000)
            full_before={n:p.detach().clone() for n,p in model.named_parameters()} if milestone else None
            before=model.outconv_seg.weight.detach().clone()
            opt.step()
            if torch.equal(before,model.outconv_seg.weight): raise FloatingPointError('optimizer made no actual parameter update')
            step+=1;seen.add(view['index']);torch.cuda.synchronize()
            if milestone:
                from .loss_diagnostics import record_update, training_probe
                record_update(root,method,step,model,full_before)
                del full_before
                training_probe(root,method,step,model,ds,cfg)
            with torch.no_grad():
                pu,pw,pden=pos_components(logits,sample,cfg.pos_background_rho)
                teacher_stats=stats(teacher(sample),sample)
                pos_partition=dict(unweighted={k:float(v) for k,v in pu.items()},weighted={k:float(v) for k,v in pw.items()},denominators=pden)
            entry=dict(soft_pos_partition=soft_pos_components(logits.detach(),sample),teacher=teacher_stats,pos_partition=pos_partition,anchor_weight=cfg.anchor_weight,pos_background_rho=cfg.pos_background_rho,attempt_identity=attempt_identity,resumed_from_update=initial_step,update=step,visited_views=cursor,k=view['k'],patch=view['patch'],flip=view['flip'],rotation=view['rotation'],
                       unique_patches=len(seen),pool_size=len(ds),loss=float(total),losses={k:float(v.detach()) for k,v in parts.items()},
                       residual_abs_mean=float(residual.detach().abs().mean()),residual_abs_max=float(residual.detach().abs().max()),gradient_norms=gradient_norms,weight_decay_gradient_norms=decay_norms,valid_candidates=int(sample['type_valid'].sum()),unknown_candidates=int((~sample['type_valid']).sum()),type_reasons=dict(__import__('collections').Counter(sample['type_reasons'])),type_indices=dict(__import__('collections').Counter(map(str,sample['type_indices']))),marker_supervised_pixels=int(sample['type_valid'].sum())*48*48,marker_masked_pixels=int((~sample['type_valid']).sum())*48*48,masked_marker_contribution=0.,seconds=time.monotonic()-t,data_and_geometry_seconds=geometry_seconds,peak_vram_bytes=torch.cuda.max_memory_allocated())
            log.write(json.dumps(entry)+'\n')
            if step%10==0 or step==1: print(json.dumps(entry),flush=True)
            atomic_json(out/'progress.json',dict(**entry,elapsed_seconds=time.monotonic()-started,
                        measured_updates=step-initial_step,remaining_updates=cfg.updates-step))
            if step in (100,1000,2000,4000,cfg.updates) or stop:
                log.flush();os.fsync(log.fileno())
                from .resources import require_space
                try: require_space(root,cfg.mini_test,reserve_bytes=400*1024**2)
                except RuntimeError as error:
                    if 'PAUSED_RESOURCE' not in str(error):raise
                    publish_checkpoint(out,model,opt,deps,step,cursor,seen)
                    raise
                publish_checkpoint(out,model,opt,deps,step,cursor,seen)
            if stop: raise SystemExit(128+stop[-1])
    final=out/'final.pth'
    if not final.exists(): os.link(latest,final)
    elif sha256(final)!=sha256(latest): raise ValueError('final checkpoint differs from endpoint')
    atomic_json(out/'complete.json',dict(status='succeeded',updates=step,visited_views=cursor,unique_patches=len(seen),pool_size=len(ds),
                parameters=sum(p.numel() for p in model.parameters()),added_parameters=321+(12545 if method=='SP' else 0),prototype_parameters=12545 if method=='SP' else 0,dependencies=deps,outputs={'checkpoint':identity(final)},wall_seconds=time.monotonic()-started,
                peak_vram_bytes=torch.cuda.max_memory_allocated()))
