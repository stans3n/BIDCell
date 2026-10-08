"""Explicit scalar-loss FP32 tolerance audit using retained G resume fixture."""
import argparse,json
from pathlib import Path
from bidcell.model.dataio.dataset_input import DataProcessing
from scripts.crgf.config import read
from scripts.crgf.training import make_model,optimizer,load,to_device,losses
from scripts.crgf.common import atomic_json
import torch

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);a=p.parse_args();cfg=read(a.config).model_copy(update={'variant':'G'});out=Path(a.out);torch.set_num_threads(4)
 ds=DataProcessing(cfg.native(),isTraining=True,total_steps=None,structured=True);views=json.loads((Path(cfg.output_root)/'configs/views.json').read_text());v=[v for v in views['views'] if v['k']>1][1];ds.view_transform=(v['flip'],v['rotation']);sample=to_device(ds[v['index']]);m=make_model(cfg,'G',len(ds.gene_names));opt=optimizer(m,cfg);m.train();values=[]
 for _ in range(2):
  load(out/'G_resume.pth',m,opt,{'inputs':{'gene_order':ds.gene_names},'test':'EGR exact resume'});opt.zero_grad();y=m(sample['x'],sample['geometry'],sample['s']);l,_=losses(y,sample);l.backward();opt.step();values.append(float(l))
 tolerance=1e-5*max(abs(values[0]),1);assert abs(values[0]-values[1])<=tolerance
 atomic_json(out/'resume_loss_tolerance.json',dict(status='succeeded',losses=values,max_absolute_difference=abs(values[0]-values[1]),allowed_absolute_difference=tolerance,definition='FP32 relative 1e-5 on unchanged native six-loss total'))
if __name__=='__main__':main()
