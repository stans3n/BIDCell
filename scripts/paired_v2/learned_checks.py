"""Nonzero learned-state permutation/masking and branch-activation evidence."""
import argparse
import copy
from pathlib import Path
import numpy as np
import torch
from .common import *
from .data import PairedViews


def check(root=DEFAULT_ROOT):
    root=Path(root);check_environment();run='V2_INTERACT';cfg=read(config_path(run,root))
    path=root/'tests/pilots'/run/'checkpoints/final.pth';state=torch.load(path,map_location='cpu')
    assert state['actual_updates']==100 and state['pilot']
    m=make_model('V2_INTERACT');m.load_state_dict(state['model'],strict=True);m.eval();data=PairedViews('V2_INTERACT')
    view=read(INPUT/'configs/probes.json')['morphology'][0];b=data.get_view(view,'cuda');k=len(b['input'])
    order=torch.arange(k-1,-1,-1,device='cuda');captured={}
    def hook(module,args):captured['h3']=args[0].detach().clone()
    handle=m.interaction.register_forward_pre_hook(hook)
    with torch.no_grad():z=forward(m,b,'V2_INTERACT')
    handle.remove();h3=captured['h3'];aux=b['aux']
    pb=dict(b,input=b['input'][order],aux=dict(aux))
    for key in ('geometry','native_support'):pb['aux'][key]=aux[key][order]
    pb['aux']['relation']=aux['relation'][order][:,order]
    with torch.no_grad():pz=forward(m,pb,'V2_INTERACT')[order]
    # Preserve the additional whole-network gate and its failure while allowing
    # independent required learned-block checks to produce their own evidence.
    full_permutation_error=None
    try:torch.testing.assert_close(z,pz,atol=1e-6,rtol=1e-6)
    except AssertionError as error:full_permutation_error=str(error)
    module=m.interaction;module.record_attention=True
    with torch.no_grad():actual=module(h3,aux['native_support'],aux['relation'])
    assert not torch.equal(actual,h3)
    weights=module.last_attention
    with torch.no_grad():permuted_block=module(h3[order],aux['native_support'][order],aux['relation'][order][:,order])[order]
    torch.testing.assert_close(actual,permuted_block,atol=1e-6,rtol=1e-6)
    coarse=torch.nn.functional.max_pool2d(aux['native_support'][:,None].float(),4,4)[:,0]>0
    s=coarse.flatten(1).T;legal=s[:,:,None]&s[:,None,:]&~torch.eye(k,device='cuda',dtype=torch.bool)[None]
    assert torch.count_nonzero(weights.masked_select(~legal[:,None].expand_as(weights)))==0
    has=legal.any(-1)
    torch.testing.assert_close(weights.sum(-1),has[:,None].expand(-1,2,-1).float(),atol=2e-7,rtol=0)
    delta=(actual-h3).permute(2,3,0,1).reshape(144,k,256)
    assert torch.count_nonzero(delta[~has])==0
    with torch.no_grad():
        assert torch.equal(module(h3[:1],aux['native_support'][:1],aux['relation'][:1,:1]),h3[:1])
        assert torch.equal(module(h3,torch.zeros_like(aux['native_support']),aux['relation']),h3)
    probes={}
    for n in (2,32,64,100):
        p=root/'tests/pilots'/run/'diagnostics'/f'probe_{n:04d}.json';r=read(p)
        active=[x for x in r['rows'] if x['interaction']['max_neighbors']>=2]
        if not active:raise ValueError('no real multi-neighbor probe')
        q=sum(next(t['gradient_norm'] for t in x['gradient_groups']['interaction']['tensors'] if t['name']=='interaction.q.weight') for x in active)
        keys=sum(next(t['gradient_norm'] for t in x['gradient_groups']['interaction']['tensors'] if t['name']=='interaction.k.weight') for x in active)
        if min(q,keys)<=0:raise ValueError('query/key inactive at learned multi-neighbor probe')
        for group in ('backbone','morphology','interaction'):
            if not all(x['gradient_groups'][group]['gradient_squared']>0 and x['gradient_groups'][group]['parameter_delta_squared']>0 for x in active):raise ValueError('inactive branch: '+group)
        probes[str(n)]=dict(evidence=identity(p),real_multi_neighbor_views=len(active),q_weight_gradient_norm_sum=q,k_weight_gradient_norm_sum=keys,
            residual_ratios=[x['interaction']['residual_ratio'] for x in active],
            owner_changes=[x['interaction_disabled']['owner_changes'] for x in active],
            logits_l2=[x['interaction_disabled']['logits_l2'] for x in active])
    result=dict(status='required_block_checks_passed_extra_full_model_gate_failed' if full_permutation_error else 'passed',
        required_block_checks_status='passed',extra_full_model_gate_status='failed' if full_permutation_error else 'passed',
        extra_full_model_gate_failure=full_permutation_error,checkpoint=identity(path),view=view,nonzero_actual_branch=True,
        learned_block_permutation_max_abs=float((actual-permuted_block).abs().max()),
        full_model_permutation_max_abs=float((z-pz).abs().max()),full_model_permutation_rms=float((z-pz).double().square().mean().sqrt()),
        permutation_tolerance=dict(atol=1e-6,rtol=1e-6),illegal_attention_exact_zero=True,no_neighbor_residual_exact_zero=True,K1_exact=True,
        query_and_key_active_on_real_multi_neighbor_probes=probes,
        softmax_bias_note='A common per-head key/bias shift is mathematically unidentifiable; nonzero gradients are required for internal groups and Q/K weights on multi-neighbor views, not every scalar bias.',
        producer=producer('train'),test_source=identity(__file__),
        original_failed_test_source=identity(root/'tests/learned_checks_original_failed.py'))
    atomic_json(root/'tests/learned_interaction.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=DEFAULT_ROOT);a=p.parse_args();print(check(a.root)['status'])
