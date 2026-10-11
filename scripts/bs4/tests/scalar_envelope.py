"""Repeat original CUDA scalar path on the exact previously divergent mixed patch."""
import argparse,json
from pathlib import Path
import torch
from scripts.bs4.config import read
from scripts.bs4.training import make_model,forward,losses
from scripts.cos.training import losses as original
from scripts.cos.tests.real_model import tensor_hashes
from scripts.crgf.common import atomic_json,identity,source_identity

def run(root):
    root=Path(root);sources=['scripts/bs4/tests/scalar_envelope.py','scripts/bs4/training.py','scripts/bs4/method.py','scripts/cos/training.py'];execution_source=source_identity(sources);torch.set_num_threads(2);cfg=read(root/'T1/configs/run.json')
    sample=torch.load(root/'tests/equivalence_final/historical_reference_sample.pth',map_location='cuda')
    model=make_model(cfg,'T1',sample['x'].shape[1]);model.train();z=forward(model,sample)
    observations=[];reference=None
    zero={k:v for k,v in sample.items() if k!='_teacher'};zero.update(alpha=0.,beta=0.)
    # Fixed in advance: 1000 pairs, exactly balanced call order. Never stop on a desired extreme.
    for rep in range(1000):
        if rep%2==0:
            lo,po=original(z,sample,True,cfg.anchor_weight,1.);ln,pn=losses(z,zero,True,cfg.anchor_weight,1.)
        else:
            ln,pn=losses(z,zero,True,cfg.anchor_weight,1.);lo,po=original(z,sample,True,cfg.anchor_weight,1.)
        go=torch.autograd.grad(lo,z,retain_graph=True)[0];gn=torch.autograd.grad(ln,z,retain_graph=True)[0]
        if not torch.equal(go,gn):raise AssertionError('identity logits gradient')
        if reference is None:reference=tensor_hashes(go)
        elif tensor_hashes(go)!=reference:raise AssertionError('repeated gradient drift')
        observations.append(dict(rep=rep,order='old_new' if rep%2==0 else 'new_old',old_NE=float(po['ne']),new_NE=float(pn['ne']),old_POS=float(po['pos']),new_POS=float(pn['pos'])))
    old=[r['old_NE'] for r in observations];new=[r['new_NE'] for r in observations]
    initial=json.loads((root/'tests/equivalence_final/equivalence_scalar_observation.json').read_text())['parts']['ne']
    envelope=[min(old),max(old)];all_values=[initial['new'],initial['historical'],initial['same_process_old'],*new]
    enclosed=all(envelope[0]<=v<=envelope[1] for v in all_values)
    if source_identity(sources)!=execution_source:raise ValueError('execution source changed')
    atomic_json(root/'tests/mixed_cuda_reduction_v4.json',dict(execution_source=execution_source,model_state_hashes=tensor_hashes(model.state_dict()),logits_hash=tensor_hashes(z.detach()),status='passed' if enclosed else 'failed',same_frozen_mixed_patch=True,fixture=identity(root/'tests/equivalence_final/historical_reference_sample.pth'),prespecified_pairs=1000,balanced_order=True,ordering_effect_hypothesis="unproven; earlier fixed-order tails retained as failures",historical_repeat_count=len(old),old_NE_envelope=envelope,initial_NE=initial,all_observed_new_inside_old_repeat_envelope=enclosed,alpha_beta_zero_enabled_path=True,logit_gradients_bitwise=True,observations=observations))
    if not enclosed:raise ValueError('actual scalar discrepancy not enclosed by independent old repeat range')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();run(a.root)
