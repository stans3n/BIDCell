"""Read-only formal milestone validation while the detached runner continues."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .config import read
from .training import dependencies,make_model,optimizer,load
from scripts.crgf.common import identity,atomic_json,digest

def validate(config,update,output):
    cfg=read(config);root=Path(cfg.output_root);path=root/'checkpoints'/cfg.arm_id/f'update_{update:06d}.pth'
    state=torch.load(path,map_location='cpu');deps=dependencies(cfg,cfg.arm_id)
    model=make_model(cfg,cfg.arm_id,len(state['gene_order']),device='cpu');opt=optimizer(model,cfg)
    actual,cursor,seen=load(path,model,opt,deps)
    if actual!=update:raise ValueError('milestone update differs')
    views=json.loads((root/'configs/views.json').read_text())['views']
    if sum(v['k']>0 for v in views[:cursor])!=update:raise ValueError('view/update conservation failed')
    rows=[json.loads(line) for line in (path.parent/'updates.jsonl').read_text().splitlines() if line.strip()]
    committed=[r for r in rows if r['update']<=update]
    if [r['update'] for r in committed]!=list(range(1,update+1)):raise ValueError('committed update log coverage')
    moments=state['optimizer_state_dict']['state']
    if not moments or any(int(v['step'])!=update for v in moments.values()):raise ValueError('optimizer update counter mismatch')
    result=dict(status='succeeded',arm=cfg.arm_id,actual_updates=update,visited_views=cursor,checkpoint=identity(path),receipt=identity(path.with_suffix('.json')),dependencies_hash=digest(deps),complete_model_and_optimizer_readable=True,model_tensor_count=len(state['model_state_dict']),optimizer_parameter_count=len(moments),BN_keys=sum('running_mean' in k or 'running_var' in k or 'num_batches_tracked' in k for k in state['model_state_dict']),committed_log_entries=len(committed),valid_candidates=sum(r['valid_candidates'] for r in committed),unknown_candidates=sum(r['unknown_candidates'] for r in committed),source_commit=state['source_commit'])
    atomic_json(output,result);print(json.dumps(result,indent=2));return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--update',type=int,required=True);p.add_argument('--output',required=True);a=p.parse_args();validate(a.config,a.update,a.output)
