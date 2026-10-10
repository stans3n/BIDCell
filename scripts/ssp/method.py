"""Frozen label-free local count teacher and prototype residual, plan §§4/5."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def local_counts(expression):
    # expression is raw H,W,G, before candidate masks, with no halo.
    if expression.ndim!=3 or not torch.isfinite(expression).all() or (expression<0).any():
        raise ValueError('local expression contract')
    return F.avg_pool2d(expression.permute(2,0,1)[None],5,1,2,divisor_override=1)[0].permute(1,2,0)

@torch.no_grad()
def teacher(sample):
    if '_teacher' in sample:return sample['_teacher']
    p=sample['p'][:,0].float();q=sample['q'][:,0].bool();s=sample['s'][:,0].bool()
    valid=sample['type_valid'].bool()[:,None,None];k=len(p);h,w=p.shape[-2:]
    counts=local_counts(sample['local_expression'])
    if sample['exclusive_positive'].shape!=(k,counts.shape[-1]):raise ValueError('marker alignment')
    a=torch.einsum('hwg,kg->khw',counts,sample['exclusive_positive'].to(counts.dtype))
    u=(-torch.expm1(-a/2))*torch.exp(-2*sample['geometry'][:,1])
    u=u*(valid&s&~q&(a>0))
    denom=.25+u.sum(0);pi=u/denom;bg=.25/denom
    if k:
        top,index=u.max(0);second=u.topk(min(2,k),dim=0).values[-1] if k>1 else torch.zeros_like(top)
        atop=a.gather(0,index[None])[0]
        confidence=(-torch.expm1(-atop/3))*(top-second)/(top+1e-8)
        confidence=torch.where(top>0,confidence,0.)
        max_pi=pi.max(0).values
    else:
        index=torch.zeros((h,w),dtype=torch.long,device=p.device);confidence=torch.zeros_like(bg);max_pi=torch.zeros_like(bg)
    nuclei=torch.as_tensor(sample['nuclei'],device=p.device)
    eligible=(nuclei==0)&(valid&s).any(0)
    conflict=(valid&s&(p>0)&q).any(0)
    H=eligible&(max_pi>=.55)&(confidence>=.25)&~conflict
    replaced=H[None]&valid&s
    t=torch.where(replaced,pi,p)
    weight=torch.where(replaced&~q,.5+.5*confidence[None],torch.ones_like(p))
    J=valid&s&(nuclei==0)[None]&(p==0)&~q
    Fnew=J&H[None]&(t>0)
    return dict(target=t.detach(),weight=weight.detach(),H=H,replace=replaced,pi=pi,bg=bg,c=confidence,a=a,u=u,E=eligible,J=J,F=Fnew,conflict=conflict,winner=index,counts=counts)

def replace_pos(pos_B,logits,sample,target=None):
    t=teacher(sample) if target is None else target
    if not bool(t['H'].any()):return pos_B
    mask=t['replace'];logp=F.log_softmax(logits,dim=1)
    new=-(t['target']*logp[:,1]+(1-t['target'])*logp[:,0])*t['weight']
    old=F.cross_entropy(logits,sample['p'][:,0],reduction='none')
    return pos_B+(new[mask].sum()-old[mask].sum())/len(logits)

def stats(t,sample):
    p=sample['p'][:,0];q=sample['q'][:,0].bool();m=t['replace'];Fnew=t['F']
    def quant(v):return [float(x) for x in torch.quantile(v.float(),torch.tensor([0.,.1,.25,.5,.75,.9,1.],device=v.device))] if v.numel() else None
    ids=np.asarray(sample['cell_ids']);active=t['H']
    return dict(K=len(p),H=int(active.sum()),E=int(t['E'].sum()),J=int(t['J'].sum()),F=int(Fnew.sum()),
        F_target_sum=float(t['target'][Fnew].sum()),F_weighted_target_sum=float((t['target']*t['weight'])[Fnew].sum()),
        replaced_candidate_pixels=int(m.sum()),old_p1_modified=int((m&(p>0)).sum()),old_p0_modified=int((m&(p==0)).sum()),strong_negative_preserved=int((m&q).sum()),
        conflict_pixels=int(t['conflict'].sum()),unknown_candidates=int((~sample['type_valid']).sum()),
        weights_sum=float(t['weight'][m].sum()),weight_reduction_sum=float((1-t['weight'])[m].sum()),
        max_pi_quantiles=quant(t['pi'].max(0).values[active]),bg_quantiles=quant(t['bg'][active]),confidence_quantiles=quant(t['c'][active]),target_quantiles=quant(t['target'][m]),
        winner_ids=np.unique(ids[t['winner'][active].cpu().numpy()]).tolist() if active.any() else [],
        affected_ids=ids[Fnew.flatten(1).any(1).cpu().numpy()].tolist(),
        low_confidence_E=int((t['E']&(t['c']<.25)).sum()),low_pi_E=int((t['E']&(t['pi'].max(0).values<.55)).sum()))

class PrototypeResidual(nn.Module):
    def __init__(self,projection):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261011)
            self.first=nn.Conv2d(390,32,1,bias=True)
            nn.init.uniform_(self.first.weight,-1/math.sqrt(390),1/math.sqrt(390));nn.init.zeros_(self.first.bias)
            self.final=nn.Conv2d(32,1,1,bias=True);nn.init.zeros_(self.final.weight);nn.init.zeros_(self.final.bias)
        self.register_buffer('projection',torch.as_tensor(projection,dtype=torch.float32))
        if self.projection.ndim!=2 or self.projection.shape[1]!=16 or not torch.isfinite(self.projection).all():raise ValueError('projection contract')
        assert sum(p.numel() for p in self.parameters())==12545
    def forward(self,hd1,expression,r,confidence,geometry,search):
        k,_,h,w=hd1.shape;s=search[:,0].bool();d=geometry[:,1]
        if r.shape!=(k,self.projection.shape[0]) or confidence.shape!=(k,):raise ValueError('prototype shape mismatch')
        C=local_counts(expression);total=C.sum(-1)
        ell=torch.sqrt(C/total.clamp_min(1)[...,None]);rho=torch.sqrt(r)
        e=ell@self.projection;ep=rho@self.projection;c=confidence[:,None,None,None]
        local=e.permute(2,0,1)[None].expand(k,-1,-1,-1);proto=ep[:,:,None,None]
        masked=d.masked_fill(~s,torch.inf)
        # Full candidate group competition; min among candidates other than i.
        if k>1:
            vals,inds=masked.topk(2,dim=0,largest=False)
            other=torch.where(torch.arange(k,device=d.device)[:,None,None]==inds[0],vals[1],vals[0])
            other=torch.where(torch.isfinite(other),other,1.)
        else:other=torch.ones_like(d)
        m=s.sum(0);density=((m-1).float()/4).clamp(max=1)
        similarity=torch.einsum('hwg,kg->khw',ell,rho)
        scalars=torch.stack((confidence[:,None,None].expand(-1,h,w),
            (torch.log1p(total)/math.log(21)).clamp(max=1)[None].expand(k,-1,-1),d,
            density[None].expand(k,-1,-1),(other-d).clamp(-1,1),confidence[:,None,None]*similarity),1)
        features=torch.cat((hd1,local,c*proto.expand(-1,-1,h,w),c*(local-proto).abs(),c*(local*proto),scalars),1)
        if features.shape[1]!=390:raise AssertionError('feature count')
        return .1*c*search*self.final(F.silu(self.first(features)))


def target_hash(target):
    import hashlib
    h=hashlib.sha256()
    for key in ('target','weight','H'):
        a=target[key].detach().cpu().contiguous().numpy();h.update(key.encode());h.update(a.tobytes())
    return h.hexdigest()
