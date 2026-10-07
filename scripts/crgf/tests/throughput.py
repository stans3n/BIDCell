"""Small representative inference throughput and disk-boundary estimates."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from bidcell.model.dataio.dataset_input import DataProcessing
from scripts.crgf.config import read
from scripts.crgf.training import make_model,to_device
from scripts.crgf.prediction import decode
from scripts.crgf.common import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);a=p.parse_args();cfg=read(a.config);torch.set_num_threads(4)
    ds=DataProcessing(cfg.native(),isTraining=False,structured=True)
    training=DataProcessing(cfg.native(),isTraining=True,total_steps=None)
    views=json.loads((Path(cfg.output_root)/'configs/views.json').read_text())['views']
    choices=[]
    for k in (3,10,11,41):choices.append(next(v for v in views if v['k']==k))
    rows=[]
    for method in ('F1','B1'):
        model=make_model(cfg,method,len(ds.gene_names));model.eval()
        for view in choices:
            index=ds.fp_patches.index(view['patch']);s=to_device(ds[index])
            with torch.inference_mode():model(s['x'],s['geometry'],s['s'])
            torch.cuda.synchronize()
            times=[]
            for _ in range(3):
                begin=time.monotonic();sample=ds[index];loaded=time.monotonic();s=to_device(sample)
                with torch.inference_mode():logits=model(s['x'],s['geometry'],s['s']).cpu().numpy()
                pred=decode(logits,sample['nuclei'],sample['cell_ids']);torch.cuda.synchronize()
                times.append(dict(total_seconds=time.monotonic()-begin,data_geometry_seconds=loaded-begin))
            rows.append(dict(method=method,k=view['k'],patch=view['patch'],samples=times,median_seconds=float(np.median([v['total_seconds'] for v in times]))))
        del model;torch.cuda.empty_cache()
    atomic_json(a.out,dict(status='succeeded',scope='warm representative samples; includes data, full-nucleus geometry, transfer, forward, CPU native decode; excludes patch receipt I/O/stitching and input loading',measurements=rows))
if __name__=='__main__':main()
