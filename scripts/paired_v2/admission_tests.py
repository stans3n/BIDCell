"""Executable admission evidence; never emits an independent approval."""
import argparse
import copy
import gc
import importlib.util
import resource
import time
import numpy as np
import torch
from torch.nn import functional as F
from .common import *
from .data import PairedViews, relations
from .training import setup, restore, checkpoint_header, train
from tests.new_methods.contracts import compare_tensors, adam_formula


def cpu_dict(state):return {n:v.detach().cpu().clone() for n,v in state.items()}


def functional_tests(root):
    from bidcell.model.model.interaction import CandidateInteraction
    from scripts.new_methods.data import transform,transform_matrix
    seed_all();module=CandidateInteraction().double()
    with torch.no_grad():module.output.weight.normal_(0,.03);module.output.bias.normal_(0,.03)
    h=torch.randn(5,256,12,12,dtype=torch.float64);s=torch.ones(5,48,48,dtype=torch.bool)
    s[3,:,24:]=False;s[4]=False;r=torch.randn(5,5,4,dtype=torch.float64).clamp(-1,1)
    module.record_attention=True;actual=module(h,s,r);weights=module.last_attention.clone()
    order=torch.tensor([4,2,0,3,1]);inverse=order.argsort()
    permuted=module(h[order],s[order],r[order][:,order])[inverse]
    torch.testing.assert_close(permuted,actual,atol=1e-12,rtol=1e-12)
    assert torch.equal(module(h[:1],s[:1],r[:1,:1]),h[:1])
    assert torch.equal(module(h,torch.zeros_like(s),r),h)
    assert torch.equal(actual[4],h[4])
    assert torch.equal(weights.diagonal(dim1=-2,dim2=-1),torch.zeros_like(weights.diagonal(dim1=-2,dim2=-1)))
    one_neighbor=module(h[:2],torch.ones_like(s[:2]),r[:2,:2])
    q_grad,k_grad=torch.autograd.grad(one_neighbor.sum(),(module.q.weight,module.k.weight))
    assert torch.count_nonzero(q_grad)==0 and torch.count_nonzero(k_grad)==0
    assert torch.equal(module.last_attention[:,:,0,1],torch.ones_like(module.last_attention[:,:,0,1]))
    assert torch.equal(module.last_attention[:,:,1,0],torch.ones_like(module.last_attention[:,:,1,0]))
    # Independent explicit legal-neighbor loop at every position/head.
    e=F.layer_norm(F.linear(h.permute(2,3,0,1).reshape(144,5,256),module.embedding.weight,module.embedding.bias),
        (32,),module.norm.weight,module.norm.bias,1e-5)
    q=F.linear(e,module.q.weight,module.q.bias).reshape(144,5,2,16)
    k=F.linear(e,module.k.weight,module.k.bias).reshape(144,5,2,16)
    v=F.linear(e,module.v.weight,module.v.bias).reshape(144,5,2,16)
    bias=F.linear(F.silu(F.linear(r,module.relation[0].weight,module.relation[0].bias)),module.relation[2].weight,module.relation[2].bias)
    coarse=s.reshape(5,12,4,12,4).any(4).any(2).flatten(1).T
    message=torch.zeros(144,5,2,16,dtype=h.dtype);has=torch.zeros(144,5,dtype=torch.bool)
    max_error=0.
    for pos in range(144):
        for i in range(5):
            js=[j for j in range(5) if i!=j and coarse[pos,i] and coarse[pos,j]]
            if not js:continue
            has[pos,i]=True
            for head in range(2):
                logits=torch.stack([(q[pos,i,head]*k[pos,j,head]).sum()/4+bias[i,j,head] for j in js])
                w=logits.softmax(0)
                message[pos,i,head]=sum(w[t]*v[pos,j,head] for t,j in enumerate(js))
                max_error=max(max_error,float((w-weights[pos,head,i,js]).abs().max()))
    m=message.flatten(2);u=F.gelu(F.linear(torch.cat((e,m,e-m),-1),module.update.weight,module.update.bias))
    d=F.linear(u,module.output.weight,module.output.bias)*has[:,:,None]
    expected=h+d.reshape(12,12,5,256).permute(2,3,0,1)
    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
    block_error=float((actual-expected).abs().max())
    transforms=[];data=PairedViews('V2_INTERACT');views=read(INPUT/'configs/views.json')['views']
    view=next(v for v in views if v['K']>=3 and any(data.evidence.meta['type_index'][data.evidence.id_to_row[i]]==-1 for i in v['candidate_ids']))
    from scripts.new_methods.data import NativeViews
    native=NativeViews(INPUT,auxiliary=False)
    for flip in (0,1):
        for rot in range(4):
            test=dict(view,flip=flip,rot90=rot);batch=data.get_view(test);other=native.get_view(test)
            for key in ('input','nuclei','support','positive','negative','labels'):
                assert torch.equal(batch[key],other[key]),key
            fixture=np.arange(25).reshape(5,5,1)
            oracle=np.rot90(np.flipud(fixture) if flip else fixture,-rot,axes=(0,1))
            assert np.array_equal(transform(fixture,flip,rot),oracle)
            ids=view['candidate_ids'];mu=data.evidence.centroids[[data.evidence.id_to_row[i] for i in ids]]
            rad=np.array([max(np.sqrt(len(data.evidence.nucleus(i)[0])/np.pi),.5) for i in ids])
            for i in range(len(ids)):
                for j in range(len(ids)):
                    delta=mu[j]-mu[i]
                    if flip:delta=delta*np.array([1,-1])
                    for _ in range(rot):delta=np.array([-delta[1],delta[0]])
                    expected=np.r_[np.clip(delta/(rad[i]+rad[j]),-4,4)/4,
                                   min(np.linalg.norm(mu[j]-mu[i])/(rad[i]+rad[j]),4)/4,
                                   np.clip(np.log(rad[j]/rad[i]),-2,2)/2]
                    np.testing.assert_allclose(batch['aux']['relation'][i,j].numpy(),expected,atol=4e-8,rtol=1e-6)
            transforms.append(dict(flip=flip,rot90=rot,native_fields_exact=True,relation_correct=True))
    rejected=[]
    for ids in ([1,1],[-1],[int(data.evidence.ids.max())+100]):
        try:relations(data.evidence,ids,0,0)
        except (ValueError,KeyError):rejected.append(ids)
        else:raise AssertionError('abnormal IDs accepted')
    result=dict(status='passed',nonzero_projection_permutation_max_error=float((actual-permuted).abs().max()),
        independent_full_key_softmax_max_error=max_error,independent_block_output_max_error=block_error,
        K1_exact=True,no_neighbor_exact=True,self_edges_absent=True,K2_single_neighbor_weight_one_and_qk_gradient_zero=True,
        unknown_native_negative_index_preserved=True,
        type_filter_absent=True,eight_transforms=transforms,abnormal_ids_rejected=rejected,view=view,
        producer=producer('train'))
    atomic_json(Path(root)/'tests/functional.json',result);return result


def initial_equivalence(root,device):
    root=Path(root);check_environment(device=='cuda');data=PairedViews('V2_INTERACT')
    view=read(INPUT/'configs/views.json')['views'][0];batch=data.get_view(view,device)
    spec=importlib.util.spec_from_file_location('bidcell.model.model.paired_frozen_oracle',HISTORICAL_SOURCE/'bidcell/model/model/model.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    initial=torch.load(INPUT/'prepared/shared_initial.pth',map_location='cpu')['model'];base=BaseSix(device)
    def execute(kind):
        seed_all()
        if kind=='NATIVE':
            m=module.SegmentationModel(n_channels=313).to(device);m.load_state_dict(initial,strict=True)
        else:m=make_model(kind,device)
        m.train();o=optimizer(m);z=m(batch['input']) if kind=='NATIVE' else forward(m,batch,kind)
        loss,parts=base(z,batch);loss.backward()
        grads={n:p.grad.detach().cpu().clone() for n,p in m.named_parameters() if n in initial}
        o.step();state={n:v.detach().cpu().clone() for n,v in m.state_dict().items() if n in initial}
        result=dict(logits=z.detach().cpu(),gradients=grads,state=state,terms={k:float(v.detach()) for k,v in parts.items()},
                    total=float(loss.detach()),adam=adam_formula(initial,state,grads))
        del m,o,z,loss,parts;gc.collect()
        if device=='cuda':torch.cuda.empty_cache()
        return result
    reference=execute('NATIVE');repeats=[];floor=None
    if device=='cuda':
        for _ in range(5):
            repeated=execute('NATIVE');assert torch.equal(reference['logits'],repeated['logits'])
            comparison=compare_tensors(reference['gradients'],repeated['gradients'])
            repeats.append(dict(gradient_differences=comparison,terms=repeated['terms'],adam=repeated['adam']))
            if floor is None:floor=copy.deepcopy(comparison)
            else:
                for name,row in comparison.items():
                    if row['rms']>floor[name]['rms']:floor[name]=row
            del repeated;gc.collect()
    arms={};bn=[n for n in initial if n.endswith(('running_mean','running_var','num_batches_tracked'))]
    for kind in ('BASE','V2_MORPH','V2_INTERACT'):
        candidate=execute(kind)
        assert torch.equal(reference['logits'],candidate['logits'])
        assert all(torch.equal(reference['state'][n],candidate['state'][n]) for n in bn)
        if device=='cpu':
            assert state_equal(reference['gradients'],candidate['gradients'])
            assert reference['terms']==candidate['terms'] and reference['total']==candidate['total']
            assert state_equal(reference['state'],candidate['state'])
            gradients={'exact':True}
        else:
            for key in reference['terms']:np.testing.assert_allclose(reference['terms'][key],candidate['terms'][key],rtol=1e-6,atol=0)
            np.testing.assert_allclose(reference['total'],candidate['total'],rtol=1e-6,atol=0)
            gradients=compare_tensors(reference['gradients'],candidate['gradients'],floor)
        arms[kind]=dict(logits_exact=True,BN_exact=True,terms=candidate['terms'],gradient_checks=gradients,adam=candidate['adam'])
        del candidate;gc.collect()
    result=dict(status='passed',device=device,view=view,native_repeats=repeats,envelope=floor,arms=arms,
                native_source=identity(HISTORICAL_SOURCE/'bidcell/model/model/model.py'),producer=producer('train'))
    atomic_json(root/'tests'/f'initial_equivalence_{device}.json',result);return result


def resume_test(root,run_id):
    cfg=load_config(config_path(run_id,root));path=Path(root)/'tests/pilots'/run_id/'checkpoints/final.pth'
    saved=torch.load(path,map_location='cpu');header=checkpoint_header(cfg)
    model,opt,data,diag=setup(cfg);views=read(INPUT/'configs/views.json')['views'];cursor=saved['view_cursor']
    while not views[cursor]['K']:cursor+=1
    v=views[cursor];base=BaseSix('cuda');before=copy.deepcopy(saved['optimizer']);before_state=saved['model']
    drift=[]
    for key in header:
        changed=copy.deepcopy(header);changed[key]={'intentional_drift':True}
        try:restore(path,model,opt,changed)
        except ValueError:drift.append(key)
        else:raise AssertionError('header drift accepted')
    def execute():
        s=restore(path,model,opt,header)
        assert state_equal(model.state_dict(),s['model']) and state_equal(opt.state_dict(),s['optimizer']) and state_equal(rng_state(),s['rng'])
        model.train();opt.zero_grad(set_to_none=True);b=data.get_view(v,'cuda');z=forward(model,b,cfg['model_kind']);loss,_=base(z,b);loss.backward()
        gradients={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters()};opt.step()
        state=cpu_dict(model.state_dict());op=copy.deepcopy(opt.state_dict())
        formula=adam_formula(before_state,state,gradients,before,op)
        return dict(gradients=gradients,state=state,rng=rng_state(),optimizer=op,formula=formula,logits=z.detach().cpu())
    anchor=execute();floor=None;repeats=[]
    for _ in range(5):
        r=execute();comp=compare_tensors(anchor['gradients'],r['gradients']);repeats.append(comp)
        if floor is None:floor=copy.deepcopy(comp)
        else:
            for name,row in comp.items():
                if row['rms']>floor[name]['rms']:floor[name]=row
        del r;gc.collect()
    # A fresh model/optimizer and fresh data object exercise construction+restore.
    del model,opt,data,diag;gc.collect();torch.cuda.empty_cache()
    model,opt,data,diag=setup(cfg);candidate=execute()
    comparison=compare_tensors(anchor['gradients'],candidate['gradients'],floor)
    assert torch.equal(anchor['logits'],candidate['logits']) and state_equal(anchor['rng'],candidate['rng'])
    bn=[n for n in anchor['state'] if n.endswith(('running_mean','running_var','num_batches_tracked'))]
    assert all(torch.equal(anchor['state'][n],candidate['state'][n]) for n in bn)
    result=dict(status='passed',checkpoint=identity(path),restored_model_BN_Adam_RNG_cursor_exact=True,
        next_view=v,next_logits_BN_RNG_exact=True,header_drift_rejected=drift,native_repeat_count=5,
        native_repeats=repeats,next_gradient_check=comparison,original_adam=anchor['formula'],resumed_adam=candidate['formula'],
        next_state_differences=compare_tensors(anchor['state'],candidate['state'],kind='next_state'),producer=producer('train'))
    atomic_json(Path(root)/'tests'/f'resume_{run_id}.json',result);return result


def resources(root):
    import tifffile
    root=Path(root);check_environment();views=read(INPUT/'configs/views.json')['views'];nonempty=[v for v in views if v['K']]
    maximum=max(nonempty,key=lambda v:v['K']);nuc=tifffile.imread(read(INPUT/'V2_MORPH/configs/machine.json')['nuclei_mask'])
    predict_max={}
    for shift in (0,24):
        best=None
        for v in read(INPUT/f'configs/predict_shift{shift}.json'):
            a=np.unique(nuc[v['row']:v['row']+48,v['col']:v['col']+48]);k=int((a>0).sum())
            if best is None or k>best['K']:best=dict(v,K=k,candidate_ids=a[a>0].tolist())
        predict_max[str(shift)]=best
    assert max(v['K'] for v in predict_max.values())==41
    del nuc;records={}
    selected=[maximum]+[v for v in nonempty[:15] if v['path']!=maximum['path']]
    for run_id,kind in RUNS.items():
        cfg=load_config(config_path(run_id,root));m,o,data,diag=setup(cfg);base=BaseSix('cuda');m.train();timings=[]
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();free,total=torch.cuda.mem_get_info();external=total-free-torch.cuda.memory_reserved()
        for v in selected:
            begin=time.monotonic();b=data.get_view(v,'cuda');o.zero_grad(set_to_none=True);z=forward(m,b,kind);loss,_=base(z,b);loss.backward();o.step();torch.cuda.synchronize()
            timings.append(dict(K=v['K'],cursor=v['cursor'],seconds=time.monotonic()-begin));del b,z,loss
        reserved=torch.cuda.max_memory_reserved();peak=external+reserved
        if total-peak<1024**3:raise MemoryError('training maximum K lacks 1GiB headroom')
        records[run_id]=dict(train=dict(views=timings,max_K=40,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=reserved,
            estimated_device_peak=peak,headroom=total-peak,minutes_per_1000=float(np.mean([r['seconds'] for r in timings])*1000/60)),prediction={})
        del m,o,data,diag;gc.collect();torch.cuda.empty_cache()
        m=make_model(kind);m.eval()
        for shift in (0,24):
            data=PairedViews(kind,training=False,shift=shift);v=predict_max[str(shift)]
            pool=read(INPUT/f'configs/predict_shift{shift}.json')
            representative=[pool[i] for i in np.linspace(0,len(pool)-1,15,dtype=int)]
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();free,total=torch.cuda.mem_get_info();external=total-free-torch.cuda.memory_reserved();ts=[]
            representative_times=[]
            with torch.inference_mode():
                for _ in range(3):
                    begin=time.monotonic();b=data.get_view(v,'cuda');z=forward(m,b,kind);torch.cuda.synchronize();ts.append(time.monotonic()-begin);del b,z
                for item in representative:
                    begin=time.monotonic();b=data.get_view(item,'cuda');k=len(b['input'])
                    if k:
                        z=forward(m,b,kind);del z
                    torch.cuda.synchronize();representative_times.append(dict(index=item['index'],K=k,seconds=time.monotonic()-begin));del b
            reserved=torch.cuda.max_memory_reserved();peak=external+reserved
            if total-peak<1024**3:raise MemoryError('prediction maximum K lacks 1GiB headroom')
            records[run_id]['prediction'][str(shift)]=dict(view=v,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=reserved,
                estimated_device_peak=peak,headroom=total-peak,seconds=ts,max_K_minutes_per_1000=float(np.mean(ts)*1000/60),
                representative_views=representative_times,minutes_per_1000=float(np.mean([t['seconds'] for t in representative_times])*1000/60))
            del data;gc.collect();torch.cuda.empty_cache()
        del m;gc.collect();torch.cuda.empty_cache()
    import shutil
    result=dict(status='passed',environment=check_environment(),arms=records,
        peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        disk_free_bytes=shutil.disk_usage(root).free,test_bytes=sum(p.stat().st_size for p in (root/'tests').rglob('*') if p.is_file()),
        measured_full_candidate_groups=True,estimated_remaining_train_seconds=sum(r['train']['minutes_per_1000']*60*3.625 for r in records.values()),
        prediction_time_scope='three repeats at actual maximum K plus 15 evenly spaced views per shift; estimate, not a guaranteed full-slide timing',producer=producer('train'))
    atomic_json(root/'tests/resources.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('test',choices=['functional','cpu','cuda','pilot','resume','resources'])
    p.add_argument('--root',type=Path,default=DEFAULT_ROOT);p.add_argument('--run-id',choices=tuple(RUNS));a=p.parse_args()
    if not (a.root/'prepared/step_zero_receipt.json').exists():raise ValueError('formal step zero must be frozen before admission tests')
    if a.test=='functional':result=functional_tests(a.root)
    elif a.test in ('cpu','cuda'):result=initial_equivalence(a.root,'cpu' if a.test=='cpu' else 'cuda')
    elif a.test=='pilot':result=train(config_path(a.run_id,a.root),100,a.root/'tests/pilots'/a.run_id)
    elif a.test=='resume':result=resume_test(a.root,a.run_id)
    else:result=resources(a.root)
    print(json.dumps(dict(status=result['status'],test=a.test),indent=2))


if __name__=='__main__':main()
