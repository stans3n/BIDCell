"""Candidate-order-equivariant, bounded foreground residuals.

The baseline decoder is deliberately separate: it competes ALL candidate
foreground probabilities against their mean background, even outside support.
Only residuals are restricted to candidate support intersected with G.
"""
import numpy as np
import torch
from torch import nn


def features(logits, support, nuclei, distances, radii, density, ids):
    """Canonical ties; geometry and score competitors are distinct.

    Returns 14 channels for both absolute and relative. Relative is an
    invertible linear re-expression of exactly the same geometry information:
    (self, competitor) -> (self, competitor-self). No extra observations.
    """
    c, _, h, w = logits.shape
    dev = logits.device
    ell = logits[:, 1] - logits[:, 0]
    support = support.bool()
    count = support.sum(0)
    gate = support & (count[None] >= 2) & (~nuclei.bool().any(0)[None])
    if c == 0:
        return logits.new_zeros((0,h,w,14)), logits.new_zeros((0,h,w,14)), gate, ell
    ids = torch.as_tensor(ids, device=dev)
    # Canonical sorting followed by min/max makes ties independent of input order.
    order = torch.argsort(ids)
    d = distances / (radii[:, None, None] + 1e-6)
    comp_d, comp_r, comp_score, pool_d, pool_r = [], [], [], [], []
    for i in range(c):
        valid = support[order] & (order[:,None,None] != i)
        dg = d[order].masked_fill(~valid, float('inf'))
        k = dg.argmin(0)
        has = valid.any(0)
        comp_d.append(torch.where(has, d[order][k,torch.arange(h,device=dev)[:,None],torch.arange(w,device=dev)[None,:]], d[i]))
        comp_r.append(torch.where(has, radii[order][k], radii[i]))
        comp_score.append(torch.where(has, ell[order].masked_fill(~valid,-float('inf')).max(0).values, ell[i]))
        n = valid.sum(0).clamp_min(1)
        pool_d.append(torch.where(has, (d[order]*valid).sum(0)/n, d[i]))
        pool_r.append(torch.where(has, (radii[order,None,None]*valid).sum(0)/n, radii[i]))
    dg, rg, es, dp, rp = map(torch.stack,(comp_d,comp_r,comp_score,pool_d,pool_r))
    rr = radii[:,None,None].expand_as(d)
    # Fixed physical scales, not fitted on dev: 10 um radius, log-count / log(17),
    # log1p density. Ell clipped only as an input feature; native logits untouched.
    base = [ell.clamp(-12,12)/4, es.clamp(-12,12)/4,
            torch.log1p(count.float())[None].expand_as(d)/np.log(17),
            torch.log1p(density)[None].expand_as(d)]
    absolute = base + [d/4,dg/4,rr/10,rg/10,dp/4,rp/10,
                       distances/40, dg*rg/40, rr.square()/100,rg.square()/100]
    relative = base + [d/4,(dg-d)/4,rr/10,(rg-rr)/10,(dp-d)/4,(rp-rr)/10,
                       distances/40,(dg*rg-distances)/40,rr.square()/100,(rg.square()-rr.square())/100]
    return torch.stack(absolute,-1), torch.stack(relative,-1), gate, dg-d


class ResidualAdapter(nn.Module):
    def __init__(self, method, beta=1., distance_scale=1.):
        super().__init__()
        self.method, self.beta, self.distance_scale = method, beta, distance_scale
        # Padding score inputs preserves parameter count, but extra dimensions
        # have identically zero input and confer no geometry information.
        self.head = nn.Sequential(nn.Linear(14,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU(),nn.Linear(32,1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, z, absolute, relative, gate, distance_advantage):
        if self.method in ('E00','E01'):
            delta = z[:,0]*0
        elif self.method == 'E03':
            # Prespecified simple absolute-distance rule from Change Plan §8.2.
            # absolute[...,4] is self normalized distance /4.
            delta = torch.clamp(-self.beta*absolute[...,4]*4+self.distance_scale,-1.,1.)*gate
        else:
            v = relative if self.method == 'E05' else absolute
            if self.method == 'E02':
                v = torch.cat((v[...,:4],torch.zeros_like(v[...,4:])),dim=-1)
            delta = torch.tanh(self.head(v).squeeze(-1))*gate
        result = torch.stack((z[:,0],z[:,1]+delta),dim=1)
        return result, delta
