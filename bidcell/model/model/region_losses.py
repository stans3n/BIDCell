"""Independent RR-V1.1 and B-V1.3 objectives. No change to the original six losses.

The discrete native decoder and graph selections are detached. Every selected
continuous graph value is gathered from the original differentiable tensor.
ST tests must check the mixed soft VJP, never hard-argmax finite differences.
"""
import heapq
import numpy as np
import torch
from torch import nn
from scipy.special import softmax
from numba import njit


def native_owner(logits):
    """Actual baseline scipy dtype, BG-channel mean, first-index argmax."""
    with torch.no_grad():
        probs = softmax(logits.detach().cpu().numpy(), axis=1)
        owner = np.concatenate((probs[:, 0].mean(0, keepdims=True), probs[:, 1])).argmax(0)
    return torch.from_numpy(owner).to(logits.device)


def probabilities(logits):
    probs = logits.softmax(1)
    return probs[:, 1], probs[:, 0].mean(0)


def js(p, q):
    m = (p+q)*.5
    return .5*((p*(p.clamp_min(1e-12).log()-m.clamp_min(1e-12).log())).sum(-1)
               +(q*(q.clamp_min(1e-12).log()-m.clamp_min(1e-12).log())).sum(-1))


@njit(cache=True)
def widest_indices(values, sources):
    """Exact eight-neighbour multi-source widest paths with deterministic ties.

    Heap key (-capacity, flat index); equal-capacity paths never replace the
    existing choice. Equal minima along the chosen path use the smallest index.
    Sources have their actual tensor values (own-nucleus sources already 1).
    """
    h, w = values.shape; v = values.ravel(); n = h*w
    cap = np.full(n, -1., dtype=np.float64)
    bottleneck = np.full(n, -1, dtype=np.int64)
    done = np.zeros(n, dtype=np.bool_)
    heap = [(-1., -1)]
    heap.pop()
    for p in range(n):
        if sources.ravel()[p]:
            cap[p] = v[p]; bottleneck[p] = p
            heapq.heappush(heap, (-np.float64(v[p]), p))
    while heap:
        neg, p = heapq.heappop(heap)
        if done[p] or -neg != cap[p]: continue
        done[p] = True
        y, x = p//w, p%w
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dy == 0 and dx == 0: continue
                yy, xx = y+dy, x+dx
                if yy < 0 or yy >= h or xx < 0 or xx >= w: continue
                q = yy*w+xx
                if done[q]: continue
                c = min(cap[p], np.float64(v[q]))
                if c <= cap[q]: continue
                if v[q] < cap[p]: bi = q
                elif v[q] > cap[p]: bi = bottleneck[p]
                else: bi = min(q, bottleneck[p])
                cap[q] = c; bottleneck[q] = bi
                heapq.heappush(heap, (-c, q))
    return bottleneck.reshape(h, w), cap.reshape(h, w)


def reachable(values, sources):
    inds, _ = widest_indices(values.detach().cpu().numpy(), np.asarray(sources, dtype=bool))
    if (inds < 0).any(): raise ValueError("widest paths require at least one source")
    index = torch.from_numpy(inds).to(values.device)
    return values.flatten().gather(0, index.flatten()).reshape_as(values)


class ReferenceLoss(nn.Module):
    def __init__(self, reference):
        super().__init__()
        self.register_buffer("theta", torch.as_tensor(reference["theta"], dtype=torch.float32))
        self.register_buffer("delta", torch.as_tensor(reference["delta"], dtype=torch.float32))
        self.register_buffer("gene_indices", torch.as_tensor(reference["gene_indices"], dtype=torch.long))

    def inputs(self, logits, aux):
        f, b = probabilities(logits)
        owner = native_owner(logits)
        nuclear = aux["nuclear_labels"] != 0
        raw = aux["raw_counts"][self.gene_indices].flatten(1).T.to(logits.dtype)
        full = aux["full_nuclear_counts"][:, self.gene_indices].to(logits.dtype)
        types = aux["type_index"]
        valid = aux["ref_valid"] & (types >= 0)
        coverage = aux["support_area_visible"].to(logits.dtype)/aux["support_area_global"].clamp_min(1)
        theta = self.theta[types.clamp_min(0)].to(logits.dtype)
        delta = self.delta[types.clamp_min(0)].to(logits.dtype) + .05*(1-coverage)
        return f, b, owner, nuclear, raw, full, types, valid, coverage, theta, delta


class RegionRootLoss(ReferenceLoss):
    version = "RR-V1.1"
    def forward(self, logits, aux, with_root=True):
        if len(logits) == 0: return logits.sum()*0, logits.sum()*0, {"eligible": 0}
        f, b, owner, nuclear, raw, full, types, valid, coverage, theta, delta = self.inputs(logits, aux)
        k, h, w = f.shape
        hard = (owner[None] == torch.arange(1, k+1, device=f.device)[:, None, None]).to(f.dtype)
        pi = (f/.1).softmax(0)
        q = hard.detach() + ((owner > 0) & ~nuclear)[None]*(pi-pi.detach())
        y = full + (q*(~nuclear)[None]).flatten(1)@raw
        static = full.sum(1) + (aux["global_support_visible"] & ~nuclear[None]).flatten(1).to(f.dtype)@raw.sum(1)
        eligible = valid & (static > 0)
        weights = coverage*(static/32).clamp_max(1)
        mass = y.sum(1)
        composition = .99*y/mass.clamp_min(1e-12)[:, None] + .01/raw.shape[1]
        divergence = js(composition, theta)
        excess = (divergence-delta.clamp_max(float(np.log(2)))).relu()
        terms = torch.where(eligible & (mass > 0), weights*excess, torch.zeros_like(excess))
        expression = terms.sum()/eligible.sum().clamp_min(1)
        root = logits.sum()*0
        diag = {"eligible": int(eligible.sum()), "expression_active": int(((excess > 0)&eligible&(mass > 0)).sum()),
                "zero_assigned_mass": int((eligible&(mass == 0)).sum()), "candidate_count": k,
                "hard_counts": y.detach(), "weights": weights.detach(), "owner": owner.detach(),
                "candidate_eligible":eligible.detach(),"candidate_active":(eligible&(mass>0)&(excess>0)).detach(),
                "root_active_pixels": 0, "root_candidates": 0}
        if with_root:
            # One bulk GPU->CPU transfer for graph input, not one scalar/vertex.
            competitors = torch.cat((b[None], f), 0)
            top = competitors.detach().argsort(dim=0, descending=True, stable=True)[:2]
            best = torch.where(top[0][None] == torch.arange(1,k+1,device=f.device)[:,None,None], top[1][None], top[0][None])
            rival = competitors.gather(0, best)
            a = torch.sigmoid((f-rival)/.1)
            own_n = aux["nuclear_labels"][None] == aux["candidate_ids"][:,None,None]
            a = torch.where(nuclear[None], own_n.to(a.dtype), a)
            av = a.detach().cpu().numpy()
            own_np = own_n.detach().cpu().numpy()
            edges = np.zeros((h,w), bool); edges[[0,-1],:] = True; edges[:,[0,-1]] = True
            ris=[]; bis=[]; use=[]
            for i in range(k):
                if not own_np[i].any(): continue
                ri, _ = widest_indices(av[i], own_np[i]); bi, _ = widest_indices(av[i], edges)
                ris.append(ri); bis.append(bi); use.append(i)
            if use and (~nuclear).any():
                ai = a[use]
                ri = torch.as_tensor(np.stack(ris), device=f.device).flatten(1)
                bi = torch.as_tensor(np.stack(bis), device=f.device).flatten(1)
                r = ai.flatten(1).gather(1,ri).reshape_as(ai)
                boundary = ai.flatten(1).gather(1,bi).reshape_as(ai)
                c = torch.where(r >= boundary, r, boundary)  # R wins a tie
                penalties = hard[use].detach()*ai*(.55-c).relu()*(~nuclear)[None]
                root = penalties.sum()/(len(use)*(~nuclear).sum()).clamp_min(1)
                diag.update(root_active_pixels=int((penalties>0).sum()), root_candidates=len(use),
                            boundary_only_pixels=int(((boundary >= .55)&(r < .55)&(hard[use]>0)&~nuclear).sum()),
                            nucleus_reachable_pixels=int(((r>=.55)&(hard[use]>0)&~nuclear).sum()),
                            root_penalty_mask=(penalties>0).detach(),root_r=r.detach(),root_boundary=boundary.detach(),
                            root_candidate_indices=torch.as_tensor(use,device=f.device))
        return expression, root, diag


class BoundaryRegionLoss(ReferenceLoss):
    version = "B-V1.3"
    def forward(self, logits, aux):
        if len(logits) == 0: return logits.sum()*0, {"eligible": 0, "omega": 0}
        f,b,owner,nuclear,raw,full,types,valid,coverage,theta,delta = self.inputs(logits,aux)
        k,h,w=f.shape; support=aux["native_support"].bool()
        valid = valid & ~aux["specimen_clipped"]
        n_support=support.sum(0)
        supported_valid=(~support | valid[:,None,None]).all(0)
        # Distinct types without assuming contiguous or positive IDs.
        typ_min=torch.where(support,types[:,None,None],torch.full_like(support, 2**30,dtype=torch.long)).min(0).values
        typ_max=torch.where(support,types[:,None,None],torch.full_like(support,-2,dtype=torch.long)).max(0).values
        omega0=(~nuclear)&(raw.sum(1).reshape(h,w)>0)&(n_support>=2)&supported_valid&(typ_min!=typ_max)
        hard=(owner[None]==torch.arange(1,k+1,device=f.device)[:,None,None]).to(f.dtype)
        owner_in_c=(hard.bool()&support).any(0)
        omega=omega0&owner_in_c
        active=support&omega[None]
        # Softmax across C only; inactive columns get a finite safe dummy domain.
        scores=(f/.1).masked_fill(~support,-torch.inf)
        scores=torch.where(omega[None],scores,torch.zeros_like(scores))
        r=scores.softmax(0)
        assignment=(hard*support).detach()+active*(r-r.detach())
        y=full+(assignment*(~nuclear)[None]).flatten(1)@raw
        static=full.sum(1)+(support&~nuclear[None]).flatten(1).to(f.dtype)@raw.sum(1)
        eligible=(support&omega0[None]).flatten(1).any(1)&valid&(static>0)
        mass=y.sum(1);p=.99*y/mass.clamp_min(1e-12)[:,None]+.01/raw.shape[1]
        divergence=js(p,theta);excess=(divergence-delta).relu()
        weights=coverage*(static/32).clamp_max(1)
        terms=torch.where(eligible&(mass>0),weights*excess.square(),torch.zeros_like(excess))
        loss=terms.sum()/eligible.sum().clamp_min(1)
        # Empty active domain is exactly connected zero, even if detached context
        # has a positive diagnostic value.
        if not omega.any():loss=logits.sum()*0
        return loss,{"eligible":int(eligible.sum()),"omega0":int(omega0.sum()),"omega":int(omega.sum()),
                     "active_candidates":int(active.flatten(1).any(1).sum()),
                     "deadzone_active":int((eligible&(excess>0)&(mass>0)).sum()),
                     "hard_counts":y.detach(),"owner":owner.detach(),"direct_domain":active.detach(),
                     "omega0_mask":omega0.detach(),"weights":weights.detach(),
                     "candidate_eligible":eligible.detach(),"candidate_active":(eligible&(mass>0)&(excess>0)&active.flatten(1).any(1)).detach(),
                     "background_skip":int((omega0&(owner==0)).sum()),
                     "outside_c_skip":int((omega0&(owner>0)&~owner_in_c).sum())}
