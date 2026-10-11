"""Exact CPU loss reference from protected accepted old checkout."""
import sys
sys.path.insert(0,'/home/zwan0726/work/projects/BIDCell_cos_loss_20261009T160351')
import torch
from scripts.cos.training import losses
from scripts.cos.tests.real_model import tensor_hashes
from scripts.crgf.common import atomic_json

torch.set_num_threads(2)
fixtures=torch.load(sys.argv[1],map_location='cpu');results={}
for name,(z,sample) in fixtures.items():
    z=z.detach().requires_grad_();loss,parts=losses(z,sample,True,34.61094491628047);g=torch.autograd.grad(loss,z)[0]
    results[name]=dict(loss=float(loss),parts={k:float(v) for k,v in parts.items()},gradient=tensor_hashes(g))
atomic_json(sys.argv[2],dict(source=sys.modules[losses.__module__].__file__,fixtures=results))
