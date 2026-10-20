"""Locate the additional learned whole-network FP32 permutation test failure.

This is evidence only: it does not change the original extra 1e-6 gate or the
formal FP32 implementation. Same-checkpoint ablations isolate native layers.
"""
import argparse
import torch
from bidcell.model.model.region_losses import native_owner
from bidcell.model.utils.utils import get_seg_mask
from .common import *
from .data import PairedViews


def stats(a,b):
    d=(a-b).double()
    return dict(exact=torch.equal(a,b),max_abs=float(d.abs().max()),rms=float(d.square().mean().sqrt()),
                reference_rms=float(a.double().square().mean().sqrt()),
                extra_gate_exceedances=int((d.abs()>1e-6+1e-6*a.double().abs()).sum()),elements=a.numel())


def check(root=DEFAULT_ROOT,device='cuda'):
    root=Path(root);environment=check_environment(device=='cuda')
    path=root/'tests/pilots/V2_INTERACT/checkpoints/final.pth'
    state=torch.load(path,map_location='cpu');data=PairedViews('V2_INTERACT')
    view=read(INPUT/'configs/probes.json')['morphology'][0];b=data.get_view(view,device)
    k=len(b['input']);order=torch.arange(k-1,-1,-1,device=device);inverse=torch.argsort(order)
    pb=dict(b,input=b['input'][order],aux=dict(b['aux']))
    for key in ('geometry','native_support'):pb['aux'][key]=b['aux'][key][order]
    pb['aux']['relation']=b['aux']['relation'][order][:,order]
    def same_layout(value,reference):
        result=torch.empty_strided(reference.shape,reference.stride(),dtype=reference.dtype,device=reference.device)
        result.copy_(value);return result
    same_stride=dict(pb,input=same_layout(pb['input'],b['input']),aux=dict(pb['aux']))
    for key in ('geometry','native_support','relation'):
        same_stride['aux'][key]=same_layout(pb['aux'][key],b['aux'][key])
    result=dict(environment=environment,checkpoint=identity(path),view=view,K=k,
                original_input_stride=list(b['input'].stride()),permuted_input_stride=list(pb['input'].stride()),
                extra_gate=dict(atol=1e-6,rtol=1e-6,origin='additional implementation test, not frozen numerical admission contract'),cases={})
    for case in ('full','without_interaction','native_backbone_only','full_same_stride','without_interaction_same_stride','native_backbone_only_same_stride'):
        model=make_model('V2_INTERACT',device);model.load_state_dict(state['model'],strict=True);model.eval()
        if not case.startswith('full'):model._modules.pop('interaction')
        if case.startswith('native_backbone_only'):model._modules.pop('morphology')
        captures={};handles=[]
        def hook(name):
            def capture(module,args,out):captures[name]=out.detach().clone()
            return capture
        names=['conv1','conv2','conv3','morphology','interaction','conv4','conv5',
               'relu4d_1','relu3d_1','relu2d_1','relu1d_1','outconv_seg']
        for name in names:
            if hasattr(model,name):handles.append(getattr(model,name).register_forward_hook(hook(name)))
        saved_rng=rng_state();saved_model=tensor_sha(model.state_dict())
        with torch.no_grad():
            z=forward(model,b,'V2_INTERACT');original=dict(captures)
            repeat=forward(model,b,'V2_INTERACT');repeated=dict(captures)
            permuted_batch=same_stride if case.endswith('_same_stride') else pb
            pz=forward(model,permuted_batch,'V2_INTERACT')[inverse];permuted={n:t[inverse] for n,t in captures.items()}
        assert saved_model==tensor_sha(model.state_dict()) and state_equal(saved_rng,rng_state())
        owner=get_seg_mask(z.cpu().numpy(),b['labels'].cpu().numpy())
        powner=get_seg_mask(pz.cpu().numpy(),b['labels'].cpu().numpy())
        result['cases'][case]=dict(layers={n:dict(permutation=stats(original[n],permuted[n]),repeat=stats(original[n],repeated[n])) for n in original},
            permuted_input_stride=list(permuted_batch['input'].stride()),
            logits=stats(z,pz),repeat_logits=stats(z,repeat),candidate_owner_changes=int((native_owner(z)!=native_owner(pz)).sum()),
            native_id_owner_changes=int((owner!=powner).sum()),model_bn_rng_unchanged=True,all_modules_eval=all(not x.training for x in model.modules()))
        for h in handles:h.remove()
        del model,captures,original,repeated,permuted,z,pz,repeat
    result['status']='extra_whole_model_gate_failed' if result['cases']['full']['logits']['extra_gate_exceedances'] else 'extra_whole_model_gate_passed'
    result['interpretation']='The original extra gate is retained. Per-layer same-checkpoint ablations and repeat-forward controls localize any ordering differences; no loss/backward envelope is used to excuse forward error.'
    result['test_source']=identity(__file__)
    atomic_json(root/f'tests/permutation_diagnostic_{device}.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=DEFAULT_ROOT);p.add_argument('--device',default='cuda',choices=['cpu','cuda']);a=p.parse_args()
    print(check(a.root,a.device)['status'])
