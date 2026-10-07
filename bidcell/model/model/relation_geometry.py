"""CRGF-1008: one complete ragged candidate group, channel-only LayerNorm."""
import torch
from torch import nn
from torch.nn import functional as F


class RelationGeometry(nn.Module):
    def __init__(self):
        super().__init__()
        self.wf = nn.Conv2d(128, 32, 1)
        self.wg = nn.Conv2d(4, 32, 1)
        self.norm = nn.LayerNorm(32, eps=1e-5)
        self.wm = nn.Conv2d(65, 32, 1)
        self.wo = nn.Conv2d(32, 128, 1)
        for layer in (self.wf, self.wg, self.wm, self.wo):
            nn.init.xavier_uniform_(layer.weight, gain=1)
            nn.init.zeros_(layer.bias)

    def relations(self, features, geometry, search_area):
        if features.ndim != 4 or features.shape[1] != 128 or features.shape[0] == 0:
            raise ValueError('CRGF requires one complete nonempty [K,128,H/2,W/2] group')
        if geometry.shape != (features.shape[0], 4, features.shape[2]*2, features.shape[3]*2):
            raise ValueError('geometry/group shape mismatch')
        if search_area.shape != (features.shape[0], 1, *geometry.shape[2:]):
            raise ValueError('support/group shape mismatch')
        if features.dtype != torch.float32:
            raise ValueError('stage1 requires FP32')
        g = F.avg_pool2d(geometry, 2, 2)
        a = (F.max_pool2d(search_area.float(), 2, 2) > 0).float()
        t = self.wf(features) + self.wg(g)
        z = F.gelu(self.norm(t.movedim(1, -1)).movedim(-1, 1), approximate='none')
        n_other = a.sum(0, keepdim=True) - a
        mean_other = ((a*z).sum(0, keepdim=True) - a*z) / n_other.clamp_min(1)
        relation = (n_other > 0) * (mean_other - z)
        return z, relation, n_other.log1p(), a

    def forward(self, features, geometry, search_area):
        z, relation, count, a = self.relations(features, geometry, search_area)
        v = F.gelu(self.wm(torch.cat((z, relation, count), 1)), approximate='none')
        return features + .1 * a * self.wo(v)
