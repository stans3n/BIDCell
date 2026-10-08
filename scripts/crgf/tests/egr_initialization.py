"""Whole-model historical template identity and all h2 consumers on real max-K."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import torch,numpy as np
from scripts.crgf.common import atomic_json
from scripts.crgf.config import read
from scripts.crgf.training import make_model,optimizer,losses,to_device
from bidcell.model.dataio.dataset_input import DataProcessing

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);a=p.parse_args();cfg=read(a.config).model_copy(update={'variant':None});torch.set_num_threads(4)
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 ds=DataProcessing(cfg.native(),isTraining=True,total_steps=None,structured=True);views=json.loads((Path(cfg.output_root)/'configs/views.json').read_text());view=max(views['views'],key=lambda v:v['k']);ds.view_transform=(view['flip'],view['rotation']);sample=to_device(ds[view['index']]);report={};template=None
 for method in ('GR','E','G','R'):
  model=make_model(cfg,method,len(ds.gene_names));state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
  if method=='GR':template=state
  else:
   for k,v in state.items():
    expected=template[k]
    if k=='relation_geometry.wm.weight' and method in ('E','G'):expected=expected[:,:32]
    torch.testing.assert_close(v,expected,atol=0,rtol=0)
  hashes={k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in state.items()};captured={};hooks=[]
  for name in ('maxpool2','h2_PT_hd4','h2_PT_hd3','h2_Cat_hd2_conv'):
   hooks.append(getattr(model,name).register_forward_pre_hook(lambda m,args,n=name:captured.__setitem__(n,args[0])))
  hooks.append(model.relation_geometry.register_forward_hook(lambda m,args,value:captured.__setitem__('h2_out',value)))
  model.train();opt=optimizer(model,cfg);opt.zero_grad();y=model(sample['x'],sample['geometry'],sample['s']);total,parts=losses(y,sample);assert torch.isfinite(total);total.backward()
  for name in ('maxpool2','h2_PT_hd4','h2_PT_hd3','h2_Cat_hd2_conv'):assert captured[name] is captured['h2_out']
  assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters());opt.step()
  report[method]=dict(template_tensor_hashes=hashes,max_K=view['k'],loss=float(total),h2_all_consumers_same_object=True,finite_gradients=True)
  for h in hooks:h.remove()
  del model,opt,state,captured,y,total,parts;torch.cuda.empty_cache()
 atomic_json(out/'whole_initialization_and_h2.json',dict(status='succeeded',checks=report,legacy_template_module_equivalence='module_contract.json'))
if __name__=='__main__':main()
