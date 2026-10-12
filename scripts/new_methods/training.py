"""One complete native candidate group per FP32 Adam update, strict resume."""
import copy
import gc
import json
import time
from pathlib import Path
import numpy as np
import torch
from bidcell.model.model.region_losses import RegionRootLoss,BoundaryRegionLoss,native_owner
from .common import *
from .data import NativeViews
from .provenance import component_source


def loss_module(config,device):
    arm=config['arm']
    if arm=='V1_REGION':return RegionRootLoss(read(config['reference'])).to(device)
    if arm=='V1_BOUNDARY':return BoundaryRegionLoss(read(config['reference'])).to(device)
    if arm in ('V2_MORPH','BASE'):return None
    raise ValueError(arm)


def scalar_diag(diag):
    return {k:v for k,v in diag.items() if not torch.is_tensor(v)}


def candidate_diag(batch,diag):
    if 'candidate_eligible' not in diag:return []
    values=torch.stack((batch['aux']['candidate_ids'].double(),batch['aux']['type_index'].double(),
                       diag['candidate_eligible'].double(),diag['candidate_active'].double(),diag['weights'].double()),1).detach().cpu().numpy()
    return [dict(cell_id=int(v[0]),type_index=int(v[1]),eligible=bool(v[2]),active=bool(v[3]),weight=float(v[4])) for v in values]


def forward_losses(model,batch,base,extra,arm):
    z=model(batch['input'],batch.get('aux') if arm=='V2_MORPH' else None)
    l6,parts=base(z,batch);zero=z.sum()*0
    if arm=='V1_REGION':e,r,diag=extra(z,batch['aux']);new=(e,r)
    elif arm=='V1_BOUNDARY':e,diag=extra(z,batch['aux']);new=(e,zero)
    else:new=(zero,zero);diag={}
    return z,l6,new,parts,diag


def intervention(new,calibration,step):
    lam=calibration.get('lambda_max',0.)*(.1+.9*min(step/200,1.))
    return lam*(new[0]+calibration.get('eta',0.)*new[1]),lam


def checkpoint_header(config):
    root=Path(config['runroot'])
    return dict(version=VERSION,arm=config['arm'],architecture='UNet3plus'+('+M-V2.2' if config['arm']=='V2_MORPH' else ''),
                auxiliary_protocol={'V1_REGION':'RR-V1.2','V1_BOUNDARY':'B-V1.3','V2_MORPH':'M-V2.2','BASE':'native'}[config['arm']],
                source=component_source('train'),config=identity(Path(config['output_root'])/'configs/run.json'),
                initial=identity(config['initial']),prepared=identity(config['prepared_manifest']),
                views=identity(config['views']),reference=identity(config['reference']),calibration=identity(config['calibration']),
                seed=1234,optimizer=dict(name='Adam',lr=1e-5,betas=[.9,.999],weight_decay=1e-4),precision=config['precision'])


def snapshot(model,opt,header,updates,cursor,skips,signals):
    return dict(header=header,model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state(),
                actual_updates=updates,view_cursor=cursor,epoch=0,empty_skips=skips,new_signal_updates=signals,
                bn_in_model_state=True)


def restore(path,model,opt,header):
    state=torch.load(path,map_location='cpu')
    if state['header']!=header:raise ValueError('checkpoint contract differs; cannot resume')
    model.load_state_dict(state['model'],strict=True);opt.load_state_dict(state['optimizer'])
    # Torch moves Adam tensor state to parameter device in load_state_dict.
    restore_rng(state['rng'])
    return state


def finite_gradients(model):
    return all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def train(config):
    config=read(config) if isinstance(config,(str,Path)) else config
    root=Path(config['runroot']);out=Path(config['output_root'])/'checkpoints';out.mkdir(exist_ok=True)
    if not torch.cuda.is_available():raise RuntimeError('formal training requires host CUDA')
    calibration=read(config['calibration'])
    if calibration['status']!='admitted':raise ValueError('calibration not admitted')
    header=checkpoint_header(config);arm=config['arm'];model=make_model(arm,config['initial']);opt=optimizer(model)
    data=NativeViews(root,auxiliary=arm!='BASE',morphology=arm=='V2_MORPH')
    views=read(config['views'])['views'];base=BaseSix('cuda');extra=loss_module(config,'cuda')
    updates=cursor=skips=signals=0
    if (out/'latest.pth').exists():
        state=restore(out/'latest.pth',model,opt,header)
        updates=state['actual_updates'];cursor=state['view_cursor'];skips=state['empty_skips'];signals=state['new_signal_updates']
        if updates>config['updates'] or cursor>len(views):raise ValueError('checkpoint cursor exceeds frozen budget')
        if updates!=sum(v['K']>0 for v in views[:cursor]) or skips!=sum(v['K']==0 for v in views[:cursor]):raise ValueError('checkpoint cursor/update/skip inconsistency')
    model.train();start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    log=Path(config['output_root'])/'training.jsonl'
    # Append-only log may include attempted updates lost before latest checkpoint;
    # record resume epoch explicitly; actual state and cursor are authoritative.
    with log.open('a',buffering=1) as stream:
        stream.write(json.dumps(dict(event='start_or_resume',updates=updates,cursor=cursor,process=process_identity()))+'\n')
        while updates<config['updates']:
            if cursor>=len(views):raise RuntimeError('frozen view sequence exhausted before actual budget')
            view=views[cursor];before=time.monotonic();batch=data.get_view(view,'cuda');data_seconds=time.monotonic()-before
            cursor+=1
            if not view['K']:skips+=1;continue
            opt.zero_grad(set_to_none=True)
            z,l6,new,parts,diag=forward_losses(model,batch,base,extra,arm)
            add,lam=intervention(new,calibration,updates);loss=l6+add
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite training loss')
            loss.backward()
            if not finite_gradients(model):raise FloatingPointError('nonfinite training gradients')
            opt.step();updates+=1
            if float(new[0].detach())!=0 or float(new[1].detach())!=0:signals+=1
            torch.cuda.synchronize()
            record=dict(event='update',updates=updates,cursor=cursor,nominal_index=view['index'],empty_skips=skips,
                        K=view['K'],flip=view['flip'],rot90=view['rot90'],xy=batch['xy'],loss=float(loss.detach()),
                        base6=float(l6.detach()),base_terms={k:float(v.detach()) for k,v in parts.items()},
                        expression=float(new[0].detach()),root=float(new[1].detach()),lambda_=lam,
                        data_seconds=data_seconds,step_seconds=time.monotonic()-before,new_signal_updates=signals,
                        auxiliary=scalar_diag(diag),candidate_auxiliary=candidate_diag(batch,diag),
                        morphology=getattr(getattr(model,'morphology',None),'diagnostics',{}))
            stream.write(json.dumps(record)+'\n')
            atomic_json(out/'progress.json',dict(status='running',actual_updates=updates,view_cursor=cursor,empty_skips=skips,
                        total=config['updates'],new_signal_updates=signals,elapsed_seconds=time.monotonic()-start,
                        peak_vram_bytes=torch.cuda.max_memory_allocated(),process=process_identity()))
            if updates%25==0 or updates==config['updates']:
                atomic_torch(out/'latest.pth',snapshot(model,opt,header,updates,cursor,skips,signals))
            if updates in (100,1000,2000,3625):
                from .diagnostics import fixed_probe_diagnostics
                saved_rng=rng_state();fixed_probe_diagnostics(model,data,base,extra,config,updates,calibration)
                restore_rng(saved_rng);model.train()
            if updates%10==0:print(arm,updates,'/',config['updates'],'cursor',cursor,'loss',record['loss'],flush=True)
        atomic_torch(out/'final.pth',snapshot(model,opt,header,updates,cursor,skips,signals))
    rec=dict(status='succeeded',dependencies=header,actual_updates=updates,view_cursor=cursor,empty_skips=skips,
             new_signal_updates=signals,wall_seconds=time.monotonic()-start,peak_vram_bytes=torch.cuda.max_memory_allocated(),
             parameter_count=sum(p.numel() for p in model.parameters()),outputs={'checkpoint':identity(out/'final.pth'),'log':identity(log)})
    atomic_json(out/'complete.json',rec);return rec


def gradient_stats(model,losses):
    """Full parameter vectors; FP64 sums, not last-layer gradient proxies."""
    params=[p for p in model.parameters() if p.requires_grad]
    grads=[]
    for loss in losses:
        grads.append(torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True))
    matrix=np.zeros((len(grads),len(grads)),np.float64);layers={}
    names=[n for n,p in model.named_parameters() if p.requires_grad]
    for p_index,name in enumerate(names):
        key=name.split('.')[0];local=layers.setdefault(key,np.zeros_like(matrix))
        for i in range(len(grads)):
            a=grads[i][p_index]
            if a is None:continue
            if not torch.isfinite(a).all():raise FloatingPointError('nonfinite probe gradient '+name)
            for j in range(i,len(grads)):
                b=grads[j][p_index]
                if b is None:continue
                val=float((a.double()*b.double()).sum())
                matrix[i,j]+=val;local[i,j]+=val
                if i!=j:matrix[j,i]+=val;local[j,i]+=val
    return matrix,{k:v.tolist() for k,v in layers.items()}


def capture(z,batch):
    owner=native_owner(z);non=batch['labels']==0;fg=(owner>0)&non
    total=batch['aux']['raw_counts'].sum(0)
    areas=[int(((owner==i+1)&non).sum()) for i in range(len(z))]
    return dict(capture=float(total[fg].sum()),area_median=float(np.median(areas)),area_sum=sum(areas))


def probe(model,data,base,extra,config,views,state_name):
    old_mode=model.training;model.eval();rows=[];saved_rng=rng_state()
    for n,view in enumerate(views):
        batch=data.get_view(view,'cuda');z,l6,new,parts,diag=forward_losses(model,batch,base,extra,config['arm'])
        gram,layers=gradient_stats(model,[l6,*new])
        rows.append(dict(state=state_name,view_cursor=view['cursor'],K=view['K'],gram=gram.tolist(),layer_gram=layers,
                         norms=np.sqrt(np.diag(gram)).tolist(),losses=[float(l6.detach()),*[float(v.detach()) for v in new]],
                         capture=capture(z,batch),diagnostics=scalar_diag(diag)))
        del batch,z,l6,new,parts,diag
        if (n+1)%8==0:print(config['arm'],'probe',state_name,n+1,'/',len(views),flush=True)
    restore_rng(saved_rng);model.train(old_mode);return rows


def temporary_updates(model,opt,data,base,extra,config,views,start,stop,calibration=None):
    model.train();rows=[]
    for step in range(start,stop):
        batch=data.get_view(views[step],'cuda');opt.zero_grad(set_to_none=True)
        z,l6,new,_,diag=forward_losses(model,batch,base,extra,config['arm'])
        add,lam=intervention(new,calibration or {},step)
        total=l6+add
        if not torch.isfinite(total):raise FloatingPointError('nonfinite development objective')
        total.backward()
        if not finite_gradients(model):raise FloatingPointError('nonfinite development gradients')
        opt.step();rows.append(dict(step=step+1,loss=float(total.detach()),lambda_=lam,diagnostics=scalar_diag(diag)))
    return rows
