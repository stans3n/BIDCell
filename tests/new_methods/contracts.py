"""Independent numerical acceptance fixtures, not assertion strings or existence checks."""
import copy
import importlib.util
import subprocess
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage as ndi
from bidcell.model.model.region_losses import *
from scripts.new_methods.common import *


def numerical():
    records={};torch.manual_seed(17)
    # Compare exact maximin capacities against threshold connectivity, an
    # independent characterization, including a long snake and diagonal path.
    rng=np.random.default_rng(22);maximum=0.
    for n in range(24):
        a=rng.random((9,11));sources=np.zeros(a.shape,bool);sources[0,0]=True;sources[8,9]=True
        inds,capacity=widest_indices(a,sources);expected=np.zeros_like(a)
        for threshold in sorted(np.unique(a)):
            components,_=ndi.label(a>=threshold,structure=np.ones((3,3)))
            labels=np.unique(components[sources&(a>=threshold)]);labels=labels[labels>0]
            expected[np.isin(components,labels)]=threshold
        maximum=max(maximum,float(np.max(np.abs(expected-capacity))))
        assert np.array_equal(a.ravel()[inds],capacity)
        # Production FP32 needs a separate numba specialization and identical
        # double heap keys; FP64 gradcheck alone cannot exercise this path.
        i32,c32=widest_indices(a.astype(np.float32),sources)
        assert np.array_equal(a.astype(np.float32).ravel()[i32].astype(np.float64),c32)
    assert maximum==0;records['widest_threshold_oracle_max_error']=maximum
    a=torch.tensor([[1.,.73,.61],[.24,.82,.92],[.13,.37,.46]],dtype=torch.float64,requires_grad=True)
    source=np.zeros((3,3),bool);source[0,0]=True
    assert torch.autograd.gradcheck(lambda x:reachable(x,source),(a,),eps=1e-6,atol=1e-5,rtol=1e-3)
    records['widest_non_tie_gradcheck']=True
    k,g,h,w=3,32,4,5
    ref=dict(theta=np.tile(np.full(g,.01/g),(3,1)),delta=[0.,0.,0.],gene_indices=list(range(g)))
    for i in range(3):ref['theta'][i,i]=.99+.01/g
    z=torch.randn(k,2,h,w,dtype=torch.float64,requires_grad=True);labels=torch.zeros(h,w,dtype=torch.long)
    labels[0,0]=11;labels[0,1]=22;labels[0,2]=33
    counts=torch.arange(g*h*w,dtype=torch.float64).reshape(g,h,w)%3
    support=torch.ones(k,h,w,dtype=torch.bool);support[2,2:]=False
    aux=dict(raw_counts=counts,candidate_ids=torch.tensor([11,22,33]),nuclear_labels=labels,
             full_nuclear_counts=torch.ones(k,g,dtype=torch.float64),global_support_visible=support,
             native_support=support,support_area_visible=support.flatten(1).sum(1),support_area_global=torch.full((k,),h*w),
             type_index=torch.tensor([0,1,2]),ref_valid=torch.ones(k,dtype=torch.bool),specimen_clipped=torch.zeros(k,dtype=torch.bool))
    for name,cls in [('region',RegionRootLoss),('boundary',BoundaryRegionLoss)]:
        lossmod=cls(ref).double()
        if name=='region':loss,root,diag=lossmod(z,aux)
        else:loss,diag=lossmod(z,aux)
        grad=torch.autograd.grad(loss,z,retain_graph=True)[0]
        owner=native_owner(z);hard=(owner[None]==torch.arange(1,k+1)[:,None,None]).double()
        expected=aux['full_nuclear_counts']+(hard*(labels==0)[None]*(support if name=='boundary' else 1)).flatten(1)@counts.flatten(1).T
        assert torch.equal(expected,diag['hard_counts'])
        if name=='region':blocked=(labels>0)|(owner==0);leak=float(grad[:,:,blocked].abs().max())
        else:leak=float(grad.permute(1,0,2,3)[:,~diag['direct_domain']].abs().max())
        assert leak==0 and torch.isfinite(grad).all()
        records[name]=dict(hard_count_max_error=float((expected-diag['hard_counts']).abs().max()),direct_gradient_leak=leak,gradient_norm=float(grad.norm()))
    # A single candidate's conditional expression redistribution is identically
    # constant, including Unknown competing naturally in the multicandidate case.
    one={key:(value[:1] if key not in ('raw_counts','nuclear_labels') else value) for key,value in aux.items()}
    z1=z[:1].detach().requires_grad_();e,_,_=RegionRootLoss(ref).double()(z1,one)
    assert torch.equal(torch.autograd.grad(e,z1)[0],torch.zeros_like(z1));records['K1_expression_zero']=True
    one['ref_valid'][:]=False;e,_,_=RegionRootLoss(ref).double()(z1,one)
    assert torch.isfinite(e) and float(e)==0;records['missing_reference_connected_zero']=True
    return records


def compare_tensors(reference, candidate, floor=None, kind='gradients'):
    rows={}
    for name,value in reference.items():
        actual=candidate[name];diff=(value-actual).double().abs()
        rms=float(diff.square().mean().sqrt());norm=float(value.double().norm())
        row=dict(maxabs=float(diff.max()),rms=rms,norm=norm,
                 relative_L2=float(diff.norm()/norm) if norm else None,
                 fail_atol1e6_rtol1e5=int((diff>1e-6+1e-5*value.double().abs()).sum()),n=value.numel())
        if floor is not None:
            epsilon=1e-7 if kind=='gradients' else 1e-8
            bound=2*floor[name]['rms']+epsilon
            row.update(native_repeat_rms=floor[name]['rms'],rms_bound=bound)
            row['exceeds_RMS_envelope']=rms>bound
            if rms>bound and kind=='gradients':raise AssertionError(f'{kind} exceeds native repeat RMS envelope {name}: {rms}>{bound}')
            if kind=='state' and row['maxabs']>2e-5+1e-8:raise AssertionError('next Adam exceeds two learning rates '+name)
        rows[name]=row
    return rows


def adam_formula(before,after,gradients,optimizer_before=None,optimizer_after=None):
    """Independent CPU FP32 Adam formula with actual gradients and prior moments."""
    rows={}
    for index,(name,gradient) in enumerate(gradients.items()):
        p=before[name].cpu();g=gradient.cpu()+1e-4*p
        previous=(optimizer_before or {}).get('state',{}).get(index,{})
        m=previous.get('exp_avg',torch.zeros_like(p)).cpu();v=previous.get('exp_avg_sq',torch.zeros_like(p)).cpu()
        step=int(previous.get('step',0))+1
        mnew=.9*m+.1*g;vnew=.999*v+.001*g.square()
        expected=p-(1e-5/(1-.9**step))*mnew/(vnew.sqrt()/np.sqrt(1-.999**step)+1e-8)
        actual=after[name].cpu();error=float((actual-expected).abs().max())
        torch.testing.assert_close(actual,expected,atol=1e-7,rtol=1e-6)
        row=dict(step=step,max_parameter_formula_error=error)
        if optimizer_after is not None:
            actual_m=optimizer_after['state'][index]['exp_avg'].cpu();actual_v=optimizer_after['state'][index]['exp_avg_sq'].cpu()
            # lerp and a*m+b*g round differently near cancellation. Bound
            # arithmetic error by input magnitudes, not the cancelled output.
            real_m=.9*m.double()+.1*g.double();real_v=.999*v.double()+.001*g.double().square()
            eps=torch.finfo(torch.float32).eps
            bound_m=1e-7+4*eps*(.9*m.double().abs()+.1*g.double().abs())
            bound_v=1e-7+4*eps*(.999*v.double().abs()+.001*g.double().square())
            error_m=(actual_m.double()-real_m).abs();error_v=(actual_v.double()-real_v).abs()
            if (error_m>bound_m).any() or (error_v>bound_v).any():raise AssertionError('Adam moment formula exceeds FP32 rounding bound '+name)
            row.update(max_first_moment_formula_error=float(error_m.max()),max_second_moment_formula_error=float(error_v.max()),
                       first_moment_rounding_bound_fraction=float((error_m/bound_m).max()),second_moment_rounding_bound_fraction=float((error_v/bound_v).max()))
        rows[name]=row
    return dict(parameter_atol=1e-7,parameter_rtol=1e-6,moment_rounding_bound='1e-7 + 4*float32_eps*sum(abs(each mathematical update summand)); FP64 independent formula',tensors=rows)


def adam_outlier_coordinates(original,new,comparisons,before):
    result={}
    for name,row in comparisons.items():
        if not row.get('exceeds_RMS_envelope') or name not in original['gradients']:continue
        difference=(original['state'][name]-new['state'][name]).abs();idx=int(difference.argmax())
        p=float(before[name].flatten()[idx]);ga=float(original['gradients'][name].flatten()[idx]);gb=float(new['gradients'][name].flatten()[idx])
        result[name]=dict(flat_index=idx,pre_parameter=p,original_gradient=ga,new_gradient=gb,
                         original_effective_gradient=ga+1e-4*p,new_effective_gradient=gb+1e-4*p,
                         effective_gradient_sign_flip=bool(np.sign(ga+1e-4*p)!=np.sign(gb+1e-4*p)),
                         original_after=float(original['state'][name].flatten()[idx]),new_after=float(new['state'][name].flatten()[idx]))
    return result


def baseline_oracle(root,batch):
    """Exact logits; GPU derivatives compared to the actual native noise floor."""
    import gc
    out=Path(root)/'admission/oracle';out.mkdir(exist_ok=True)
    source=subprocess.check_output(['git','show','2fae67ad74eaa9ec77b5d6fed6178f7cd9feca9d:bidcell/model/model/model.py'],cwd=REPO,text=True)
    path=out/'baseline_model.py';path.write_text(source)
    spec=importlib.util.spec_from_file_location('bidcell.model.model.baseline_oracle',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    init=Path(root)/'prepared/shared_initial.pth';state=torch.load(init,map_location='cpu')['model'];base=BaseSix('cuda')
    def execute(arm):
        seed_all()
        if arm=='ORIGINAL':model=module.SegmentationModel(n_channels=313).cuda();model.load_state_dict(state)
        else:model=make_model(arm,init)
        model.train();opt=optimizer(model)
        z=model(batch['input'],batch['aux']) if arm=='V2_MORPH' else model(batch['input'])
        loss,parts=base(z,batch);repeated=[{k:float(v.detach()) for k,v in base(z,batch)[1].items()} for _ in range(3)]
        loss.backward();grads={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if n in state}
        opt.step();after={n:v.detach().cpu().clone() for n,v in model.state_dict().items() if n in state}
        formula=adam_formula(state,after,grads)
        result=dict(logits=z.detach().cpu(),gradients=grads,state=after,Adam_formula=formula,
             terms={k:float(v.detach()) for k,v in parts.items()},same_logits_repeated_terms=repeated)
        del model,opt,z,loss,parts;gc.collect();torch.cuda.empty_cache();return result
    original=execute('ORIGINAL');floor=None;native_repeats=[]
    # Near-zero-gradient sign flips are sparse: one repeat may happen to have
    # none in a large tensor. Five prespecified repeats measure that native floor.
    for _ in range(5):
        repeat=execute('ORIGINAL');assert torch.equal(original['logits'],repeat['logits'])
        comparison={kind:compare_tensors(original[kind],repeat[kind],kind=kind) for kind in ('gradients','state')}
        native_repeats.append(dict(terms=repeat['terms'],same_logits_repeated_terms=repeat['same_logits_repeated_terms'],comparisons=comparison,Adam_formula=repeat['Adam_formula']))
        if floor is None:floor=copy.deepcopy(comparison)
        else:
            for kind in floor:
                for name in floor[kind]:
                    if comparison[kind][name]['rms']>floor[kind][name]['rms']:floor[kind][name]=comparison[kind][name]
        del repeat;gc.collect()
    records={'native_repeats':native_repeats,'native_repeat_envelope':floor}
    for arm in ('BASE','V2_MORPH'):
        new=execute(arm);assert torch.equal(original['logits'],new['logits'])
        for k,v in original['terms'].items():np.testing.assert_allclose(v,new['terms'][k],atol=1e-6,rtol=1e-6)
        comparisons={kind:compare_tensors(original[kind],new[kind],floor[kind],kind) for kind in ('gradients','state')}
        bn=[n for n in state if n.endswith(('running_mean','running_var','num_batches_tracked'))]
        assert all(torch.equal(original['state'][n],new['state'][n]) for n in bn)
        records[arm]=dict(logits_exact=True,BN_exact=True,terms=new['terms'],same_logits_repeated_terms=new['same_logits_repeated_terms'],comparisons=comparisons,
                          Adam_formula=new['Adam_formula'],Adam_RMS_outlier_coordinates=adam_outlier_coordinates(original,new,comparisons['state'],state))
        del new;gc.collect()
    return dict(status='succeeded',oracle=identity(path),checks=records,original_terms=original['terms'],
        original_same_logits_repeated_terms=original['same_logits_repeated_terms'],scalar_atol=1e-6,scalar_rtol=1e-6,
        original_Adam_formula=original['Adam_formula'],
        GPU_limit='Original CUDA backward itself is nondeterministic. Gradient tensor RMS <=2x five native repeats+1e-7; Adam parameter RMS is reported, including sparse near-zero sign flips, not falsely passed. Each side independently obeys original Adam with actual gradients; max first update difference <=2lr+1e-8. CPU independent oracle is bitwise.',source=source_digest())
