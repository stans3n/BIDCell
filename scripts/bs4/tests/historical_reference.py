"""Runs exclusively accepted checkout modules, emitting a reference for new-source tests."""
import sys,json,subprocess
from pathlib import Path
HISTORY='/home/zwan0726/work/projects/BIDCell_cos_loss_20261009T160351'
sys.path.insert(0,HISTORY)
import torch
from scripts.cos.config import read
from scripts.cos.training import make_model,optimizer,losses,to_device,rng_state
from scripts.cos.tests.real_model import tensor_hashes
from scripts.crgf.common import atomic_json

def main(sample_path,out):
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=HISTORY,text=True,timeout=30).strip()=='541eeae61a53d84ec5543716cf4cb46bf6a292b2'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=HISTORY,text=True,timeout=30).strip()
    torch.set_num_threads(2);cfg=read('/home/zwan0726/work/outputs/cos_loss_20261009T160351+1100/B/configs/run.json');s=to_device(torch.load(sample_path,map_location='cpu'));m=make_model(cfg,'B',s['x'].shape[1]);opt=optimizer(m,cfg);m.eval()
    with torch.no_grad():z=m(s['x'],s['geometry'],s['s']);ev=tensor_hashes(z.detach())
    m.train();z=m(s['x'],s['geometry'],s['s']);loss,parts=losses(z,s,True,cfg.anchor_weight);loss.backward()
    result=dict(eval=ev,train=tensor_hashes(z.detach()),parts={k:float(v) for k,v in parts.items()},gradients={n:tensor_hashes(p.grad) for n,p in m.named_parameters()})
    opt.step();result.update(model=tensor_hashes(m.state_dict()),optimizer=tensor_hashes(opt.state_dict()),source_imports=dict(model=sys.modules[type(m).__module__].__file__,training=sys.modules[losses.__module__].__file__))
    atomic_json(out,result)
if __name__=='__main__':main(*sys.argv[1:])
