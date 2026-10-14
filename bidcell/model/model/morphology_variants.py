"""Fixed Morph ablations; all corresponding tensors derive from one full state."""
import copy
import hashlib
import json
import torch
from torch import nn
from .morphology import MorphologyModule

CANONICAL_SHA = '64c4e90ce5e14c948ce7bd3f523fb3b50f9f1734c322625826a7c3f99bcf4957'
VARIANTS = {
    'full': dict(model_kind='MORPH', morph_variant='full', use_geometry=True, use_dapi=True),
    'no_dapi': dict(model_kind='MORPH', morph_variant='no_dapi', use_geometry=True, use_dapi=False),
    'simple_fusion': dict(model_kind='MORPH', morph_variant='simple_fusion', use_geometry=True, use_dapi=True),
    'no_geo': dict(model_kind='MORPH', morph_variant='no_geo', use_geometry=False, use_dapi=True),
}

def tensor_sha(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        t = tensor.detach().cpu().contiguous()
        h.update(json.dumps([name, str(t.dtype), list(t.shape)]).encode()); h.update(t.numpy().tobytes())
    return h.hexdigest()

def canonical_full():
    with torch.random.fork_rng():
        torch.manual_seed(20261013)
        result = MorphologyModule(chunk_size=0)
    if tensor_sha(result.state_dict()) != CANONICAL_SHA:
        raise ValueError('historical Morph reconstruction hash mismatch; stop this experiment family')
    return result

class LocalConvolutionFusion(nn.Module):
    """Two independent blocks, zero padded 5x5 support; no candidate interaction."""
    def __init__(self, original):
        super().__init__()
        for name in ('ln_e', 'ln_m', 'ln_f', 'gate', 'ffn'):
            setattr(self, name, copy.deepcopy(getattr(original, name)))
        self.context = nn.Sequential(nn.Conv2d(128,64,1), nn.GroupNorm(8,64,eps=1e-5), nn.SiLU(),
            nn.Conv2d(64,64,5,padding=2,groups=64), nn.Conv2d(64,64,1))

    def forward(self,e,m):
        en=self.ln_e(e.permute(0,2,3,1)).permute(0,3,1,2)
        mn=self.ln_m(m.permute(0,2,3,1)).permute(0,3,1,2)
        context=self.context(torch.cat((en,mn),1))
        e1=e+torch.sigmoid(self.gate(torch.cat((en,mn,context),1)))*context
        return e1+self.ffn(self.ln_f(e1.permute(0,2,3,1))).permute(0,3,1,2)

class MorphologyVariant(nn.Module):
    version='MORPH-ABLATION-20261015-v1.0'
    def __init__(self,variant,canonical=None):
        super().__init__()
        if variant not in VARIANTS or variant=='full':raise ValueError(variant)
        original=canonical if canonical is not None else canonical_full()
        self.spec=dict(VARIANTS[variant]);self.variant=variant;self.diagnostics={}
        for name in ('expression','output','memory'):
            setattr(self,name,copy.deepcopy(getattr(original,name)))
        if self.spec['use_geometry']:self.geometry=copy.deepcopy(original.geometry)
        if self.spec['use_dapi']:self.image=copy.deepcopy(original.image)
        self.blocks=copy.deepcopy(original.blocks)
        if variant in ('no_dapi','no_geo'):
            columns=slice(0,64) if variant=='no_dapi' else slice(64,96)
            # Forking protects even the discarded default convolution initializer.
            with torch.random.fork_rng():
                projection=nn.Conv2d(64 if variant=='no_dapi' else 32,64,1,bias=False)
            with torch.no_grad():projection.weight.copy_(original.memory[0].weight[:,columns])
            self.memory[0]=projection
        elif variant=='simple_fusion':
            with torch.random.fork_rng():
                torch.manual_seed(2026101501)
                self.blocks=nn.ModuleList([LocalConvolutionFusion(b) for b in original.blocks])

    def forward(self,h3,geometry=None,dapi=None):
        pieces=[]
        if self.spec['use_geometry']:
            if geometry is None or geometry.shape!=(len(h3),16,48,48):raise ValueError('geometry input contract')
            pieces.append(self.geometry(geometry))
        elif geometry is not None:raise ValueError('NO_GEO must not receive geometry')
        if self.spec['use_dapi']:
            if dapi is None or dapi.shape!=(1,3,48,48):raise ValueError('DAPI input contract')
            pieces.append(self.image(dapi).expand(len(h3),-1,-1,-1))
        elif dapi is not None:raise ValueError('NO_DAPI must not receive DAPI')
        memory=self.memory(torch.cat(pieces,1) if len(pieces)>1 else pieces[0])
        e0=self.expression(h3);e=e0
        for block in self.blocks:e=block(e,memory)
        delta=self.output(e-e0)
        self.diagnostics={'residual_ratio':float(delta.detach().norm()/h3.detach().norm().clamp_min(1e-12))}
        return h3+delta

def initialization_map(module,canonical):
    src=canonical.state_dict();rows={}
    for name,t in module.state_dict().items():
        if name=='memory.0.weight' and getattr(module,'variant','full') in ('no_dapi','no_geo'):
            start,stop=(0,64) if module.variant=='no_dapi' else (64,96)
            expected=src[name][:,start:stop];origin=f'{name}[:,{start}:{stop},:,:]'
        elif name in src:
            expected=src[name];origin=name
        else:
            rows[name]=dict(origin='PyTorch default initializer',seed=2026101501,shape=list(t.shape),sha256=tensor_sha({name:t}));continue
        if not torch.equal(t.detach().cpu(),expected.detach().cpu()):raise ValueError('nonidentical common tensor '+name)
        rows[name]=dict(origin=origin,exact=True,shape=list(t.shape),sha256=tensor_sha({name:t}))
    return rows
