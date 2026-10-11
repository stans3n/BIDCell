"""Frozen original S pi/c/H with BS4 all-one weights and alpha/beta targets."""
import hashlib
import torch
from torch.nn import functional as F
from scripts.ssp.method import teacher as frozen_s_teacher, local_counts

@torch.no_grad()
def teacher(sample):
    if '_teacher' in sample:return sample['_teacher']
    alpha=float(sample.get('alpha',0.));beta=float(sample.get('beta',1.))
    if not 0<=alpha<=1 or not 0<=beta<=1:raise ValueError('alpha/beta bounds')
    base=frozen_s_teacher(sample)
    p=sample['p'][:,0].float();q=sample['q'][:,0].bool();s=sample['s'][:,0].bool()
    valid=sample['type_valid'].bool()[:,None,None]
    E=base['H'][None]&valid&s&~q
    target=torch.where(E,torch.where(p>0,(1-alpha)+alpha*base['pi'],beta*base['pi']),p)
    changed=E&(target!=p)
    return dict(base,target=target.detach(),weight=torch.ones_like(p),replace=changed,
        E_candidate=E,F=E&(p==0)&(target>0),alpha=alpha,beta=beta)

def replace_pos(pos_B,logits,sample,target=None):
    t=teacher(sample) if target is None else target
    mask=t['replace']
    if not bool(mask.any()):return pos_B
    lp=F.log_softmax(logits,dim=1)
    new=-(t['target']*lp[:,1]+(1-t['target'])*lp[:,0])
    old=F.cross_entropy(logits,sample['p'][:,0],reduction='none')
    return pos_B+(new[mask].sum()-old[mask].sum())/len(logits)

def target_hash(t):
    h=hashlib.sha256()
    for key in ('target','weight','H'):
        h.update(key.encode());h.update(t[key].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def frozen_teacher_hash(t):
    h=hashlib.sha256()
    for key in ('pi','c','H'):
        h.update(key.encode());h.update(t[key].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def stats(t,sample):
    from scripts.ssp.method import stats as original_stats
    d=original_stats(t,sample);p=sample['p'][:,0];E=t['E_candidate'];soft=E&(p>0)&(t['target']<1)
    d.update(alpha=t['alpha'],beta=t['beta'],all_weights_one=bool((t['weight']==1).all()),
        eligible_candidate_pixels=int(E.sum()),p1_softened_pixels=int(soft.sum()),
        p1_softening_mass=float((1-t['target'])[soft].sum()),frozen_teacher_sha256=frozen_teacher_hash(t))
    return d
