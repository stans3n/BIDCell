"""The single frozen V2-PAIRED candidate interaction block.

Candidates remain the network batch. Every query attends to its entire legal
key set at each position; no candidate truncation, self edge, or type filtering.
"""
import torch
from torch import nn
from torch.nn import functional as F


class CandidateInteraction(nn.Module):
    version = 'I-V2-PAIRED-20261014'

    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(256, 32)
        self.norm = nn.LayerNorm(32, eps=1e-5)
        self.q = nn.Linear(32, 32)
        self.k = nn.Linear(32, 32)
        self.v = nn.Linear(32, 32)
        self.relation = nn.Sequential(nn.Linear(4, 16), nn.SiLU(), nn.Linear(16, 2))
        self.update = nn.Linear(96, 32)
        self.output = nn.Linear(32, 256)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.diagnostics = {}
        self.record_attention = False
        self.last_attention = None

    def forward(self, h3, support, relation):
        k, c, h, w = h3.shape
        if c != 256 or (h, w) != (12, 12):
            raise ValueError('interaction requires Kx256x12x12')
        if support.shape != (k, 48, 48) or relation.shape != (k, k, 4):
            raise ValueError('interaction support/relation shape mismatch')
        if k == 0:
            raise ValueError('empty candidate groups must be skipped by the caller')
        if not torch.isfinite(relation).all():
            raise ValueError('nonfinite relation')
        coarse = F.max_pool2d(support[:, None].float(), 4, 4)[:, 0] > 0
        s = coarse.flatten(1).T  # positions, candidates
        legal = s[:, :, None] & s[:, None, :]
        legal = legal & ~torch.eye(k, dtype=torch.bool, device=h3.device)[None]
        has = legal.any(-1)
        e = self.norm(self.embedding(h3.permute(2, 3, 0, 1).reshape(h*w, k, c)))
        def heads(layer):
            return layer(e).reshape(h*w, k, 2, 16).permute(0, 2, 1, 3)
        q, keys, values = heads(self.q), heads(self.k), heads(self.v)
        scores = torch.matmul(q, keys.transpose(-1, -2)) / 4
        scores = scores + self.relation(relation).permute(2, 0, 1)[None]
        # An empty row receives finite scores before softmax, then exact zero
        # weights. It never evaluates softmax over an all-minus-infinity row.
        masked = scores.masked_fill(~legal[:, None], -torch.inf)
        safe = torch.where(has[:, None, :, None], masked, torch.zeros_like(masked))
        weights = safe.softmax(-1) * legal[:, None].to(scores.dtype)
        message = torch.matmul(weights, values).permute(0, 2, 1, 3).reshape(h*w, k, 32)
        u = F.gelu(self.update(torch.cat((e, message, e-message), -1)))
        delta = self.output(u) * has[:, :, None].to(h3.dtype)
        delta = delta.reshape(h, w, k, c).permute(2, 3, 0, 1)
        self.diagnostics = {
            'candidate_positions': int(has.numel()),
            'positions_with_neighbor': int(has.sum()),
            'legal_edges': int(legal.sum()),
            'max_neighbors': int(legal.sum(-1).max()),
            'residual_ratio': float(delta.detach().norm()/h3.detach().norm().clamp_min(1e-12)),
            'nonzero_residual_outside_neighbor': int((delta.detach().permute(2, 3, 0, 1).reshape(h*w, k, c)[~has] != 0).sum()),
        }
        self.last_attention = weights.detach() if self.record_attention else None
        return h3 + delta
