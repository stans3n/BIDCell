"""Immutable identities and baseline construction for the three independent arms."""
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
from scripts.crgf.common import *

REPO = Path(__file__).resolve().parents[2]
WORK = Path('/home/zwan0726/work')
BASELINE = (WORK/'baseline/latest_bidcell_official_run').resolve()
BASE_CONFIG = BASELINE/'configs/xenium_breast_official_step4000.yaml'
ARMS = ('V1_REGION', 'V1_BOUNDARY', 'V2_MORPH')
VERSION = 'BIDCELL-THREE-ARM-20261013-v1'


def read(path): return json.loads(Path(path).read_text())


def native_config():
    from bidcell.config import load_config
    cfg=load_config(str(BASE_CONFIG))
    if cfg.model_params.name!='custom' or cfg.model_params.patch_size!=48:raise ValueError('baseline architecture drift')
    return cfg


def seed_all(seed=1234):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(4)


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda']:torch.cuda.set_rng_state_all(state['cuda'])


def make_model(arm, initial, device='cuda'):
    from bidcell.model.model.model import SegmentationModel
    from bidcell.model.model.morphology import MorphologyModule
    seed_all()
    model=SegmentationModel(n_channels=313)
    state=torch.load(initial,map_location='cpu')
    model.load_state_dict(state['model'],strict=True)
    if arm=='V2_MORPH':
        # Fork module RNG so neither saved backbone nor data RNG depends on it.
        with torch.random.fork_rng():
            torch.manual_seed(20261013);model.morphology=MorphologyModule(chunk_size=0)
    elif arm not in (*ARMS, 'BASE'):raise ValueError(arm)
    return model.to(device)


def optimizer(model):
    return torch.optim.Adam(model.parameters(),lr=1e-5,betas=(.9,.999),weight_decay=1e-4)


class BaseSix:
    """Call unchanged original implementations, including their reduction order."""
    def __init__(self,device):
        from bidcell.model.model.losses import NucleiEncapsulationLoss,Oversegmentation,CellCallingLoss,OverlapLoss,PosNegMarkerLoss
        p=native_config().training_params
        self.ne=NucleiEncapsulationLoss(p.ne_weight,device);self.os=Oversegmentation(p.os_weight,device)
        self.cc=CellCallingLoss(p.cc_weight,device);self.ov=OverlapLoss(p.ov_weight,device)
        self.pn=PosNegMarkerLoss(p.pos_weight,p.neg_weight,device)
        self.p=PosNegMarkerLoss(p.pos_weight,0.,device);self.n=PosNegMarkerLoss(0.,p.neg_weight,device)
    def __call__(self,z,b):
        ne=self.ne(z,b['nuclei']).squeeze();os_=self.os(z,b['nuclei']).squeeze()
        cc=self.cc(z,b['support']).squeeze();ov=self.ov(z,b['nuclei']).squeeze()
        pn=self.pn(z,b['positive'],b['negative']).squeeze()
        total=ne+os_+cc+ov+pn
        # Diagnostic positive/negative calls do not replace the original PN sum.
        with torch.no_grad():
            pos=self.p(z,b['positive'],b['negative']);neg=self.n(z,b['positive'],b['negative'])
        return total,dict(ne=ne,os=os_,cc=cc,ov=ov,pos=pos,neg=neg,pn=pn)


def json_safe(value):
    if isinstance(value,dict):return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_safe(v) for v in value]
    if isinstance(value,np.generic):value=value.item()
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def source_manifest():
    paths=sorted([p for folder in ('bidcell','scripts','tests/new_methods') for p in (REPO/folder).rglob('*')
                  if p.is_file() and p.suffix in ('.py','.R','.sh') and '__pycache__' not in p.parts])
    return {str(p.relative_to(REPO)):sha256(p) for p in paths}


def source_digest():return digest(source_manifest())


def verify_source(manifest):
    if source_manifest()!=manifest:raise ValueError('frozen source changed; new admission/review required')


def outputs_under(path, exclude=('complete.json','run.lock')):
    root=Path(path)
    return {str(p.relative_to(root)):identity(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name not in exclude and not p.name.endswith('.tmp')}
