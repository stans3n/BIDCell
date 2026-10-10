"""Reject receipt/payload corruption, architecture/gene/RNG/projection drift."""
import argparse,copy,json,tempfile
from pathlib import Path
import torch
from bidcell.model.model.model import SegmentationModel
from scripts.ssp.training import load,optimizer
from scripts.ssp.config import read
from scripts.ssp.prototype import PrototypeCache
from scripts.crgf.common import atomic_json,atomic_torch,identity,digest,source_identity

def main(root):
    root=Path(root);torch.set_num_threads(2);evidence={}
    for arm in ('S','SP'):
        path=root/'tests/numeric'/f'{arm}_resume.pth';state=torch.load(path,map_location='cpu');deps=state['dependencies'];cfg=read(root/arm/'configs/run.json');cache=PrototypeCache(cfg.prepared_manifest,state['gene_order'])
        model=SegmentationModel(n_channels=len(state['gene_order']),fusion=False,variant='B1',cos_enabled=True,prototype_projection=cache.arrays['projection'].copy() if arm=='SP' else None);opt=optimizer(model,cfg)
        with tempfile.TemporaryDirectory(dir=root/'tests') as td:
            target=Path(td)/'fault.pth';receipt=target.with_suffix('.json');checks={}
            for fault in ('receipt_hash','payload_corruption','architecture','gene_order','rng_version','sampler_version','update_counter','nonfinite')+ (('projection',) if arm=='SP' else ()):
                bad=dict(state)
                if fault=='architecture':bad['architecture']={}
                if fault=='gene_order':bad['gene_order']=state['gene_order'][::-1]
                if fault=='rng_version':bad['rng_version']='wrong'
                if fault=='sampler_version':bad['sampler_version']='wrong'
                if fault=='update_counter':bad['update_step']=5000
                if fault in ('nonfinite','projection'):
                    bad['model_state_dict']=dict(state['model_state_dict']);key='prototype_residual.projection' if fault=='projection' else next(iter(bad['model_state_dict']))
                    bad['model_state_dict'][key]=bad['model_state_dict'][key].clone();bad['model_state_dict'][key].view(-1)[0]=float('nan') if fault=='nonfinite' else 9.
                atomic_torch(target,bad);rec=dict(update=bad['update_step'],cursor=bad['sampler_cursor'],checkpoint=identity(target),dependencies_hash=digest(deps))
                if fault=='receipt_hash':rec['checkpoint']['sha256']='0'*64
                atomic_json(receipt,rec)
                if fault=='payload_corruption':
                    with target.open('r+b') as f:f.seek(-1,2);f.write(b'!')
                try:load(target,model,opt,deps)
                except (ValueError,FloatingPointError) as e:checks[fault]=str(e)
                else:raise AssertionError('accepted corrupt state '+fault)
            evidence[arm]=checks
    atomic_json(root/'tests/checkpoint_faults.json',dict(status='passed',rejected=evidence,source=source_identity(['scripts/ssp/tests/checkpoint_faults.py','scripts/ssp/training.py']),temporary_corrupted_files_removed=True))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();main(a.root)
