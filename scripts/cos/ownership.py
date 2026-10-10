"""Frozen full-patch candidate ownership, stable probability decoder and anchor."""
import numpy as np
import torch
from torch.nn import functional as F
from scipy.special import softmax


def candidate_order(ids, nuclei=None, valid=None):
    ids = np.asarray(ids)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError('candidate IDs must be one-dimensional integers')
    valid = np.ones(len(ids), bool) if valid is None else np.asarray(valid, bool)
    if valid.shape != ids.shape:
        raise ValueError('candidate mask shape mismatch')
    indices = np.flatnonzero(valid)
    chosen = ids[indices]
    if (chosen <= 0).any() or len(np.unique(chosen)) != len(chosen):
        raise ValueError('invalid or duplicate candidate ID')
    indices = indices[np.argsort(chosen, kind='stable')]
    if nuclei is not None:
        nuclei = np.asarray(nuclei)
        if not np.issubdtype(nuclei.dtype, np.integer) or (nuclei < 0).any():
            raise ValueError('invalid nuclear ID raster')
        actual = np.unique(nuclei[nuclei > 0])
        if not np.array_equal(ids[indices], actual):
            raise ValueError('candidate domain differs from patch nuclear IDs')
    return indices


def compact_sample(sample):
    """Interface-only padding: remove before native BN, model and all reductions."""
    valid=sample.get('candidate_valid')
    valid=valid.detach().cpu().numpy() if torch.is_tensor(valid) else valid
    indices=candidate_order(sample['cell_ids'], sample['nuclei'], valid)
    # Production records are canonical; permutation-aware public decoder remains invariant.
    if not np.array_equal(np.asarray(sample['cell_ids'])[indices], np.unique(sample['nuclei'][sample['nuclei']>0])):
        raise ValueError('candidate order differs from input')
    result=dict(sample)
    for name in ('x','n','s','p','q','geometry','type_valid','candidate_valid','exclusive_positive','prototype_r','prototype_confidence'):
        if name in result:
            result[name]=result[name][torch.as_tensor(indices,device=result[name].device)]
    for name in ('cell_ids','full_areas'):
        if name in result: result[name]=np.asarray(result[name])[indices]
    for name in ('type_reasons','type_indices'):
        if name in result: result[name]=[result[name][i] for i in indices]
    # Verify candidate tensors carry exactly the labelled corresponding nucleus.
    if len(indices):
        expected=torch.as_tensor(result['nuclei'][None]==result['cell_ids'][:,None,None], device=result['n'].device)
        if not torch.equal(result['n'][:,0].bool(),expected):
            raise ValueError('candidate tensor/ID order mismatch')
    return result


def log_ownership(logits, valid=None):
    if logits.ndim != 4 or logits.shape[1] != 2:
        raise ValueError('expected K x 2 x H x W binary logits')
    if not torch.isfinite(logits).all(): raise FloatingPointError('nonfinite logits')
    mask=torch.ones(len(logits),dtype=torch.bool,device=logits.device) if valid is None else torch.as_tensor(valid,dtype=torch.bool,device=logits.device)
    if mask.shape != (len(logits),): raise ValueError('candidate mask mismatch')
    k=int(mask.sum())
    if k==0:
        # Differentiable background with every invalid score -infinity.
        bg=logits.sum(dim=(0,1),keepdim=False)*0
        return torch.cat((bg[None],torch.full_like(logits[:,0],-torch.inf)),0)
    binary=F.log_softmax(logits,dim=1)
    background=torch.logsumexp(binary[mask,0],dim=0)-np.log(k)
    cell=binary[:,1].masked_fill(~mask[:,None,None],-torch.inf)
    scores=torch.cat((background[None],cell),dim=0)
    return scores-torch.logsumexp(scores,dim=0,keepdim=True)


def anchor_loss(logits, nuclei, ids, valid=None):
    ids=np.asarray(ids)
    mask=np.ones(len(ids),bool) if valid is None else np.asarray(valid,bool)
    candidate_order(ids,np.asarray(nuclei),mask)
    nu=torch.as_tensor(nuclei,dtype=torch.long,device=logits.device)
    anchor=nu>0
    if not anchor.any(): return logits.sum()*0
    targets=torch.zeros_like(nu)
    for i,c in enumerate(ids):
        if mask[i]: targets[nu==int(c)]=i+1
    if (targets[anchor]==0).any(): raise ValueError('missing nuclear owner')
    q=log_ownership(logits,mask)
    return -q.gather(0,targets[None])[0][anchor].mean()


def decode(logits,nuclei,ids,valid=None,restore=True):
    indices=candidate_order(ids,nuclei,valid)
    if not len(indices): return np.zeros(np.asarray(nuclei).shape,np.uint32)
    values=np.asarray(logits)
    if values.shape != (len(ids),2,*np.asarray(nuclei).shape): raise ValueError('logits shape mismatch')
    if not np.isfinite(values).all(): raise FloatingPointError('nonfinite logits')
    probabilities=softmax(values[indices],axis=1)
    scores=np.concatenate((probabilities[:,0].mean(0,keepdims=True),probabilities[:,1]),axis=0)
    labels=np.concatenate(([0],np.asarray(ids)[indices]))
    owner=labels[scores.argmax(0)].astype(np.uint32)
    return np.where(nuclei>0,nuclei,owner).astype(np.uint32) if restore else owner
