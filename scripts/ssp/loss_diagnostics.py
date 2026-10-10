"""Fixed gradient calibration and side-effect-free eight-view training probes."""
import argparse,copy,csv,json,time,hashlib
from pathlib import Path
from contextlib import contextmanager
import numpy as np
import torch
from scripts.crgf.common import atomic_json,identity,sha256,source_identity,digest
from .training import losses,pos_components,rng_state,restore_rng,make_model,optimizer,to_device,forward
from .method import teacher,stats,replace_pos
from scripts.cos.ownership import compact_sample,log_ownership
SIX=('ne','os','cc','ov','pos','neg')

def groups(model):
    names=[n for n,p in model.named_parameters()]
    return {g:[i for i,n in enumerate(names) if pred(n)] for g,pred in {
      'new_head':lambda n:n.startswith('cos_residual.'),
      'prototype_head':lambda n:n.startswith('prototype_residual.'),
      'prototype_first':lambda n:n.startswith('prototype_residual.first.'),
      'prototype_final':lambda n:n.startswith('prototype_residual.final.'),
      'output_conv':lambda n:n.startswith('outconv_seg.'),
      'encoder':lambda n:n.startswith(('conv1.','conv2.','conv3.','conv4.','conv5.')),
      'other_decoder':lambda n:not n.startswith(('prototype_residual.','cos_residual.','outconv_seg.','conv1.','conv2.','conv3.','conv4.','conv5.')),
      'old_shared':lambda n:not n.startswith(('cos_residual.','prototype_residual.')),
      'all':lambda n:True}.items()}

def norm(gs,inds):
    return float(torch.sqrt(sum((gs[i].detach().double().square().sum() for i in inds if gs[i] is not None),torch.zeros((),dtype=torch.float64,device=next(g.device for g in gs if g is not None)))))
def cosine(a,b,inds):
    na=norm(a,inds);nb=norm(b,inds)
    if na<=1e-12 or nb<=1e-12:return None
    dot=sum(float((a[i].detach().double()*b[i].detach().double()).sum()) for i in inds if a[i] is not None and b[i] is not None)
    return dot/(na*nb)
def gradient(value,params):
    gs=torch.autograd.grad(value,params,retain_graph=True,allow_unused=True)
    if any(g is not None and not torch.isfinite(g).all() for g in gs):raise FloatingPointError('nonfinite diagnostic gradient')
    return gs

def get(ds,view):
    if sha256(view['patch'])!=view.get('patch_sha256',sha256(view['patch'])):raise ValueError('diagnostic patch changed')
    ds.view_transform=(view['flip'],view['rotation']);return to_device(compact_sample(ds[view['index']]))

@contextmanager
def preserve(model,ds=None):
    state={n:v.detach().clone() for n,v in model.state_dict().items()};rng=rng_state()
    modes={n:m.training for n,m in model.named_modules()};grads={n:None if p.grad is None else p.grad.detach().clone() for n,p in model.named_parameters()}
    transform=getattr(ds,'view_transform',None)
    try:yield
    finally:
        model.load_state_dict(state,strict=True)
        for n,m in model.named_modules():m.training=modes[n]
        for n,p in model.named_parameters():p.grad=grads[n]
        restore_rng(rng)
        if ds is not None:ds.view_transform=transform

def inspect(model,sample,rho,weight,tag):
    params=list(model.parameters());gr=groups(model)
    z=forward(model,sample)
    _,old=losses(z,sample,True,1.,1.,s_enabled=False)
    # One historical scalar path for the identity decomposition; avoid a second
    # inherited nondeterministic CUDA NLL reduction obscuring delta POS.
    actual=dict(old);actual['pos']=replace_pos(old['pos'],z,sample)
    target=teacher(sample)
    unw,we,cov=pos_components(z,sample,rho)
    old_total=sum(old[k] for k in SIX);actual_total=sum(actual[k] for k in SIX)
    values={**old,'actual_pos':actual['pos'],'delta_pos':actual['pos']-old['pos'],'old_total':old_total,'actual_six':actual_total,'actual_total':actual_total+weight*old['anchor'],**{'pos_'+k:v for k,v in unw.items()}}
    ga=gradient(old['anchor'],params);go=gradient(old_total,params);gactual=gradient(actual_total,params)
    rows=[];cos=[];regions=[];sixnorm=[]
    nu=torch.as_tensor(sample['nuclei'],device=z.device);mask_regions={'nuclear':(nu>0)[None].expand(len(z),-1,-1),'outside_nuclei':(nu==0)[None].expand(len(z),-1,-1),'positive':sample['p'][:,0]>0,'S_H':target['replace'],'S_F':target['F'],'U':(nu[None]==0)&(sample['s'][:,0]>0)&(sample['p'][:,0]==0)}
    for name,value in values.items():
        gs=ga if name=='anchor' else go if name=='old_total' else gactual if name=='actual_six' else gradient(value,params)
        for group,inds in gr.items():
            rows.append(dict(**tag,loss=name,group=group,value=float(value.detach()),norm=norm(gs,inds),actual_component_weight=(rho if name=='pos_u_background' else weight if name=='anchor' else 1.),weighted_norm=norm(gs,inds)*(rho if name=='pos_u_background' else weight if name=='anchor' else 1.)))
            cos.append(dict(**tag,loss=name,group=group,cosine_anchor=cosine(gs,ga,inds),cosine_old=cosine(gs,go,inds)))
        if name in SIX:sixnorm.append(norm(gs,gr['old_shared']))
        lg=torch.autograd.grad(value,z,retain_graph=True)[0]
        for region,mask in mask_regions.items():
            regions.append(dict(**tag,loss=name,region=region,pixels=int(mask.sum()),norm=float(lg.permute(0,2,3,1)[mask].double().norm())))
        if name=='anchor' and torch.count_nonzero(lg.permute(0,2,3,1)[mask_regions['outside_nuclei']]):raise AssertionError('anchor has direct extranuclear gradient')
    oldnorm=norm(go,gr['old_shared']);anorm=norm(ga,gr['old_shared']);ratio=anorm/oldnorm if oldnorm>1e-12 else None
    q=log_ownership(z).detach();winner=q.argmax(0);targets=torch.zeros_like(nu,dtype=torch.long)
    for i,c in enumerate(sample['cell_ids']):targets[nu==int(c)]=i+1
    nuclear=nu>0;correct=q.gather(0,targets[None])[0];competitor=q.clone();competitor.scatter_(0,targets[None],-torch.inf);margin=correct-competitor.max(0).values
    gold=torch.autograd.grad(old['pos'],z,retain_graph=True)[0];gnew=torch.autograd.grad(actual['pos'],z,retain_graph=True)[0]
    metrics=dict(H_foreground_probability=float(z.softmax(1)[:,1][target['replace']].mean()) if target['replace'].any() else None,teacher=stats(target,sample),prototype_confidence_quantiles=torch.quantile(sample['prototype_confidence'],torch.tensor([0.,.1,.5,.9,1.],device=z.device)).tolist(),pos_logit_gradient_old_norm=float(gold.norm()),pos_logit_gradient_new_norm=float(gnew.norm()),pos_logit_gradient_cosine=float(torch.nn.functional.cosine_similarity(gold.flatten(),gnew.flatten(),dim=0)),**tag,k=len(z),ratio=ratio,old_norm=oldnorm,anchor_norm=anorm,cancellation_ratio=oldnorm/sum(sixnorm) if sum(sixnorm)>0 else None,anchor_ce=float(old['anchor'].detach()),nuclear_pixels=int(nuclear.sum()),nuclear_top1=float((winner[nuclear]==targets[nuclear]).float().mean()),nuclear_margin=float(margin[nuclear].mean()),background_fraction=float((winner==0).float().mean()),outside_background_fraction=float((winner[~nuclear]==0).float().mean()),u_foreground_probability=float(z.softmax(1)[:,1][mask_regions['U']].mean()) if mask_regions['U'].any() else None,pos_unweighted={k:float(v.detach()) for k,v in unw.items()},pos_weighted={k:float(v.detach()) for k,v in we.items()},coverage=cov)
    return metrics,rows,cos,regions

def write_csv(path,rows):
    if not rows:return
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def append_unique(path,rows,keys):
    path=Path(path);old=list(csv.DictReader(path.open())) if path.exists() else []
    newkeys={tuple(str(r[k]) for k in keys) for r in rows};old=[r for r in old if tuple(str(r[k]) for k in keys) not in newkeys];write_csv(path,old+rows)

def training_probe(root,arm,step,model,ds,cfg):
    root=Path(root);out=root/'diagnostics';out.mkdir(exist_ok=True);start=time.monotonic()
    views=json.loads((root/'configs/training_probes.json').read_text())['samples'];metrics=[];norms=[];cos=[]
    with preserve(model,ds):
        model.train()
        for i,v in enumerate(views):
            # Restore model/BN/RNG for every probe to the same milestone.
            with preserve(model,ds):
                m,n,c,r=inspect(model,get(ds,v),cfg.pos_background_rho,cfg.anchor_weight,dict(arm=arm,update=step,sample=i,view_cursor=v['view_cursor']))
            metrics.append(m);norms+=n;cos+=c
    atomic_json(out/f'probe_{step:06d}.json',dict(status='succeeded',arm=arm,update=step,samples=metrics,wall_seconds=time.monotonic()-start,protocol='no optimizer access; full model/BN/modes/RNG/grad/view-transform restored'))
    flat=[{k:(json.dumps(v,sort_keys=True) if isinstance(v,dict) else v) for k,v in m.items()} for m in metrics]
    append_unique(out/'training_probe_metrics.csv',flat,['arm','update','sample'])
    append_unique(out/'training_probe_gradients.csv',norms,['arm','update','sample','loss','group'])
    append_unique(out/'training_probe_cosines.csv',cos,['arm','update','sample','loss','group'])

def record_update(root,arm,step,model,before):
    named=list(model.named_parameters());deltas=tuple(p.detach()-before[n] for n,p in named)
    rows=[dict(arm=arm,update=step,group=g,actual_adam_parameter_update_norm=norm(deltas,inds)) for g,inds in groups(model).items()]
    append_unique(Path(root)/'diagnostics/optimizer_update_norms.csv',rows,['arm','update','group'])

def calibration_diagnostics(root):
    from .config import read
    from .data import dataset
    root=Path(root);cfg=read(root/'S/configs/run.json');torch.set_num_threads(2)
    ds=dataset(cfg);model=make_model(cfg,'S',len(ds.gene_names));model.train();out=root/'diagnostics';metrics=[];norms=[];cosines=[];regions=[]
    start=time.monotonic()
    views=json.loads((out/'gradient_samples.json').read_text())['samples']
    for i,v in enumerate(views):
        with preserve(model,ds):m,n,c,r=inspect(model,get(ds,v),1.,cfg.anchor_weight,dict(sample=i,view_cursor=v['view_cursor']))
        metrics.append(m);norms+=n;cosines+=c;regions+=r
        atomic_json(out/'gradient_progress.json',dict(completed=i+1,total=len(views),elapsed=time.monotonic()-start))
    atomic_json(out/'gradient_diagnostics.json',dict(status='passed',samples=metrics,anchor_weight=cfg.anchor_weight,calibration_identity=cfg.calibration_identity,recalibrated=False,wall_seconds=time.monotonic()-start))
    write_csv(out/'gradient_norms.csv',norms);write_csv(out/'gradient_cosines.csv',cosines);write_csv(out/'gradient_regions.csv',regions)
