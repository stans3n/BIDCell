"""Revalidate historical inputs and replay exactly the saved matched views."""
import argparse,json,shutil,subprocess
from pathlib import Path
from .config import read,RunConfig
from .common import atomic_json,identity,sha256,verify_outputs
HIST=Path('/home/zwan0726/work/outputs/crgf_1008_20261008T031528+1100')

def prepare_method(cfg,check_views=True):
    root=Path(cfg.output_root);(root/'configs').mkdir(parents=True,exist_ok=True)
    for name in ('inputs.json','views.json','patches_shift0.json','patches_shift24.json','resolved_native.json','reference.json','protocol.json','B0_machine.json'):
        shutil.copyfile(HIST/'configs'/name,root/'configs'/name)
    if check_views:
        from bidcell.model.dataio.dataset_input import DataProcessing
        from .training import make_views
        ds=DataProcessing(cfg.native(),isTraining=True,total_steps=None)
        generated=make_views(ds,cfg);saved=json.loads((root/'configs/views.json').read_text())
        if generated!=saved:raise ValueError('historical view replay differs')
        atomic_json(root/'source/view_replay.json',dict(status='succeeded',reference=identity(HIST/'configs/views.json'),actual=identity(root/'configs/views.json'),views=len(saved['views']),updates=sum(v['k']>0 for v in saved['views']),empty=sum(v['k']==0 for v in saved['views']),pool=len(saved['train_pool']),unique=len({v['index'] for v in saved['views'] if v['k']>0})))
    atomic_json(root/'configs/run.json',cfg.model_dump())
    machine=json.loads((root/'configs/B0_machine.json').read_text());machine['baseline_final_mask']=machine['final_mask']
    machine.update(final_mask=str(root/'predictions'/cfg.variant/'final.tif'),pre_mask=str(root/'predictions'/cfg.variant/'pre.tif'),final_matrix=str(root/'matrix'/cfg.variant/'expr_mat.csv'))
    atomic_json(root/f'configs/{cfg.variant}_machine.json',machine)
    atomic_json(root/'configs/prepare_complete.json',dict(status='succeeded',outputs={p.name:identity(p) for p in (root/'configs').glob('*.json') if p.name!='prepare_complete.json'}))

def main():
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--verify-method');a=p.parse_args()
    if a.verify_method:
        from .run_stage1 import verify_preflight
        verify_preflight(read(a.verify_method))
    elif a.config:prepare_method(read(a.config))
    else:p.error('--config or --verify-method required')
if __name__=='__main__':main()
