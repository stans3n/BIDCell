"""Frozen M-V2.2: candidate geometry and shared DAPI local cross attention."""
import torch
from torch import nn
from torch.nn import functional as F


def conv(cin, cout, stride, groups):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
                         nn.GroupNorm(groups, cout, eps=1e-5), nn.SiLU())


def depthwise(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cin, 3, 2, 1, groups=cin, bias=True),
                         nn.Conv2d(cin, cout, 1, bias=False),
                         nn.GroupNorm(8, cout, eps=1e-5), nn.SiLU())


class LocalCrossAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_e = nn.LayerNorm(64, eps=1e-5)
        self.ln_m = nn.LayerNorm(64, eps=1e-5)
        self.ln_f = nn.LayerNorm(64, eps=1e-5)
        self.q, self.k, self.v, self.out = [nn.Linear(64, 64) for _ in range(4)]
        self.relative_bias = nn.Parameter(torch.zeros(4, 25))
        self.gate = nn.Conv2d(192, 64, 1)
        nn.init.zeros_(self.gate.weight); nn.init.zeros_(self.gate.bias)
        self.ffn = nn.Sequential(nn.Linear(64, 256), nn.GELU(), nn.Linear(256, 64))
        self.last_attention = None

    def forward(self, e, m, record=False):
        n, c, h, w = e.shape
        en = self.ln_e(e.permute(0, 2, 3, 1))
        mn = self.ln_m(m.permute(0, 2, 3, 1))
        q = self.q(en).reshape(n, h*w, 4, 16).permute(0, 2, 1, 3)
        def windows(layer):
            x = layer(mn).permute(0, 3, 1, 2)
            return F.unfold(x, 5, padding=2).reshape(n, 4, 16, 25, h*w).permute(0, 1, 4, 3, 2)
        k, v = windows(self.k), windows(self.v)
        valid = F.unfold(torch.ones((1, 1, h, w), device=e.device), 5, padding=2).transpose(1, 2).bool()
        scores = (q.float().unsqueeze(-2)*k.float()).sum(-1)/4
        scores = scores + self.relative_bias.float()[None, :, None, :]
        weights = scores.masked_fill(~valid[:, None], -torch.inf).softmax(-1)
        if record: self.last_attention = weights.detach()
        context = (weights.unsqueeze(-1)*v.float()).sum(-2).permute(0, 2, 1, 3).reshape(n, h, w, c)
        context = self.out(context.to(e.dtype)).permute(0, 3, 1, 2)
        gate = torch.sigmoid(self.gate(torch.cat((en.permute(0, 3, 1, 2), mn.permute(0, 3, 1, 2), context), 1)))
        e1 = e + gate*context
        return e1 + self.ffn(self.ln_f(e1.permute(0, 2, 3, 1))).permute(0, 3, 1, 2)


class MorphologyModule(nn.Module):
    version = "M-V2.2"
    def __init__(self, chunk_size=8):
        super().__init__()
        self.geometry = nn.Sequential(conv(16, 32, 1, 8), conv(32, 48, 2, 8), depthwise(48, 64))
        self.image = nn.Sequential(conv(3, 16, 1, 4), conv(16, 24, 2, 4), depthwise(24, 32))
        self.memory = nn.Sequential(nn.Conv2d(96, 64, 1, bias=False), nn.GroupNorm(8, 64, eps=1e-5), nn.SiLU())
        self.expression = nn.Conv2d(256, 64, 1)
        self.blocks = nn.ModuleList([LocalCrossAttention(), LocalCrossAttention()])
        self.output = nn.Conv2d(64, 256, 1)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)
        self.chunk_size = chunk_size
        self.diagnostics = {}

    def forward(self, h3, geometry, dapi):
        if geometry.shape != (len(h3), 16, 48, 48) or dapi.shape != (1, 3, 48, 48):
            raise ValueError("invalid morphology features: require Kx16x48x48 and 1x3x48x48")
        im = self.image(dapi)  # exactly once for this view, every update
        residuals = []
        chunk = self.chunk_size or len(h3)
        for start in range(0, len(h3), chunk):
            hs, gs = h3[start:start+chunk], geometry[start:start+chunk]
            memory = self.memory(torch.cat((self.geometry(gs), im.expand(len(gs), -1, -1, -1)), 1))
            e0 = self.expression(hs); e = e0
            for block in self.blocks: e = block(e, memory)
            residuals.append(self.output(e-e0))
        delta = torch.cat(residuals)
        self.diagnostics = {"residual_ratio": (delta.detach().norm()/h3.detach().norm().clamp_min(1e-12)).item()}
        return h3 + delta
