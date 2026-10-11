"""Reject receipt/payload corruption, architecture/gene/RNG/optimizer drift."""
import argparse,copy,json,tempfile
from pathlib import Path
import torch
from bidcell.model.model.model import SegmentationModel
from scripts.bs4.training import load,optimizer
from scripts.bs4.config import read

from scripts.crgf.common import atomic_json,atomic_torch,identity,digest,source_identity

def main(root):
    sources=['scripts/bs4/tests/checkpoint_faults.py','scripts/bs4/training.py'];execution=source_identity(sources)
    root=Path(root);torch.set_num_threads(2);evidence={}
    for arm in ('T1','T2'):
        path=root/'tests/numeric_v2'/f'{arm}_resume.pth';state=torch.load(path,map_location='cpu');deps=state['dependencies'];cfg=read(root/arm/'configs/run.json')
        model=SegmentationModel(n_channels=len(state['gene_order']),fusion=False,variant='B1',cos_enabled=True,prototype_projection=None);opt=optimizer(model,cfg)
        # Positive control on the exact common B1+321 architecture; all future mutations must fail.
        original=load(path,model,opt,deps)
        assert original[0]==16 and len(opt.state)>0
        with tempfile.TemporaryDirectory(dir=root/'tests') as td:
            target=Path(td)/'fault.pth';receipt=target.with_suffix('.json');checks={}
            for fault in ('receipt_hash','payload_corruption','architecture','gene_order','rng_version','sampler_version','update_counter','counter_forged_in_range','cursor_out_of_range','adam_moment_shape','adam_state_missing','adam_lr','adam_order','adam_negative_second_moment','visited_views','nonfinite'):
                bad=dict(state)
                if fault=='architecture':bad['architecture']={}
                if fault=='gene_order':bad['gene_order']=state['gene_order'][::-1]
                if fault=='rng_version':bad['rng_version']='wrong'
                if fault=='sampler_version':bad['sampler_version']='wrong'
                if fault=='update_counter':bad['update_step']=5000
                if fault=='counter_forged_in_range':bad['update_step']=17
                if fault=='cursor_out_of_range':bad['sampler_cursor']=bad['visited_views']=999999
                if fault=='visited_views':bad['visited_views']=0
                if fault in ('adam_lr','adam_order'):
                    bad['optimizer_state_dict']=copy.deepcopy(state['optimizer_state_dict'])
                    group=bad['optimizer_state_dict']['param_groups'][0]
                    if fault=='adam_lr':group['lr']*=2
                    else:group['params']=list(reversed(group['params']))
                if fault in ('adam_moment_shape','adam_state_missing','adam_negative_second_moment'):
                    bad['optimizer_state_dict']=copy.deepcopy(state['optimizer_state_dict'])
                    key=next(iter(bad['optimizer_state_dict']['state']))
                    if fault=='adam_negative_second_moment':bad['optimizer_state_dict']['state'][key]['exp_avg_sq'].view(-1)[0]=-1
                    elif fault=='adam_state_missing':del bad['optimizer_state_dict']['state'][key]
                    else:bad['optimizer_state_dict']['state'][key]['exp_avg']=torch.zeros(1)
                if fault=='nonfinite':
                    bad['model_state_dict']=dict(state['model_state_dict']);key=next(iter(bad['model_state_dict']))
                    bad['model_state_dict'][key]=bad['model_state_dict'][key].clone();bad['model_state_dict'][key].view(-1)[0]=float('nan')
                atomic_torch(target,bad);rec=dict(update=bad['update_step'],cursor=bad['sampler_cursor'],checkpoint=identity(target),dependencies_hash=digest(deps))
                if fault=='receipt_hash':rec['checkpoint']['sha256']='0'*64
                atomic_json(receipt,rec)
                if fault=='payload_corruption':
                    with target.open('r+b') as f:f.seek(-1,2);f.write(b'!')
                try:load(target,model,opt,deps)
                except (ValueError,FloatingPointError) as e:checks[fault]=str(e)
                else:raise AssertionError('accepted corrupt state '+fault)
            for field in ('alpha','beta','teacher_version','parent_sha256'):
                altered=copy.deepcopy(deps);altered['config'][field]='tampered'
                try:load(path,model,opt,altered)
                except ValueError as e:checks[field]=str(e)
                else:raise AssertionError('accepted dependency drift '+field)
            load(path,model,opt,deps)
            evidence[arm]=dict(positive_control=True,rejected=checks)
    if source_identity(sources)!=execution:raise ValueError('checkpoint fault execution source drift')
    atomic_json(root/'tests/checkpoint_faults_v2.json',dict(status='passed',execution_source=execution,rejected=evidence,source=source_identity(['scripts/bs4/tests/checkpoint_faults.py','scripts/bs4/training.py']),temporary_corrupted_files_removed=True))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();main(a.root)
