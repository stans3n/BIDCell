"""GPU real-data zero-init, gradients, Unknown construction and subprocess resume."""
import argparse,json,subprocess,sys,hashlib,copy,random
from pathlib import Path
import numpy as np
import torch
from scripts.cos.config import read
from scripts.cos.training import make_model,optimizer,seed_all,losses,to_device,save,load,dependencies,rng_state
from scripts.cos.ownership import compact_sample,decode,anchor_loss
from scripts.crgf.common import atomic_json,identity,digest
from scripts.crgf.training import losses as old_losses
from bidcell.model.dataio.dataset_input import DataProcessing
from bidcell.model.utils.utils import get_seg_mask

def draws():return dict(python=random.random(),numpy=float(np.random.rand()),torch=float(torch.rand(())),cuda=float(torch.rand((),device='cuda')))

def tensor_hashes(value):
    if torch.is_tensor(value):return dict(shape=list(value.shape),dtype=str(value.dtype),sha256=hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest())
    if isinstance(value,np.ndarray):return dict(shape=list(value.shape),sha256=hashlib.sha256(value.tobytes()).hexdigest())
    if isinstance(value,dict):return {str(k):tensor_hashes(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [tensor_hashes(v) for v in value]
    return value

def dataset(cfg):return DataProcessing(cfg.native(),isTraining=True,total_steps=None,structured=True,mask_unknown_markers=True)
def get(ds,v):
    ds.view_transform=(v['flip'],v['rotation']);sample=ds[v['index']];return to_device(compact_sample(sample) if ds.mask_unknown_markers else sample)
def update(model,opt,sample,cos):
    opt.zero_grad(set_to_none=True);z=model(sample['x'],sample['geometry'],sample['s']);total,_=losses(z,sample,cos);total.backward();opt.step()

def child(cfgpath,out):
    cfg=read(cfgpath);torch.set_num_threads(4);ds=dataset(cfg);views=json.loads((Path(cfg.output_root)/'configs/views.json').read_text())['views'];used=[v for v in views if v['k']][:2]
    model=make_model(cfg,cfg.arm_id,len(ds.gene_names));opt=optimizer(model,cfg);model.train();deps=dependencies(cfg,cfg.arm_id)
    step,cursor,seen=load(out/'latest.pth',model,opt,deps);assert step==1
    next_random=draws()
    update(model,opt,get(ds,used[1]),cfg.cos_enabled)
    atomic_json(out/'resumed.json',dict(model=tensor_hashes(model.state_dict()),optimizer=tensor_hashes(opt.state_dict()),rng=tensor_hashes(rng_state()),next_view=views[views.index(used[1])+1],consumed_view=used[1],sampler_cursor=views.index(used[1])+1,next_random=next_random))

def main(queuepath,out):
    out.mkdir(parents=True,exist_ok=True);queue=json.loads(Path(queuepath).read_text());torch.set_num_threads(4);evidence={}
    for base in ('B1','G'):
        cont=read(queue['methods'][base+'_CONT']['config']);cos=read(queue['methods'][base+'_COS']['config']);ds=dataset(cos)
        ds_old=DataProcessing(cos.native(),isTraining=True,total_steps=None,structured=True)
        views=json.loads((Path(cos.output_root)/'configs/views.json').read_text())['views'];chosen=[v for v in views if v['k']>=2][:3]
        a=make_model(cont,cont.arm_id,len(ds.gene_names));b=make_model(cos,cos.arm_id,len(ds.gene_names));a.eval();b.eval()
        assert sum(p.numel() for p in b.parameters())-sum(p.numel() for p in a.parameters())==321
        records=[]
        for v in chosen:
            s=get(ds,v)
            with torch.no_grad():
                z0=a(s['x'],s['geometry'],s['s']);z,original,residual=b(s['x'],s['geometry'],s['s'],return_details=True)
            assert torch.equal(z0,z) and torch.equal(original,z0) and torch.count_nonzero(residual)==0
            assert torch.equal(z0.softmax(1),z.softmax(1))
            raw0=decode(z0.cpu().numpy(),s['nuclei'],s['cell_ids'],restore=False);raw1=decode(z.cpu().numpy(),s['nuclei'],s['cell_ids'],restore=False)
            assert np.array_equal(raw0,raw1)
            assert np.array_equal(decode(z.cpu().numpy(),s['nuclei'],s['cell_ids']),get_seg_mask(z.cpu().numpy(),s['nuclei']))
            records.append(dict(patch=v['patch'],k=v['k'],logit_max_difference=0,raw_owner_mismatches=0,legacy_restored_owner_mismatches=0))
        del a;torch.cuda.empty_cache()
        b.train();s=get(ds,chosen[0]);opt=optimizer(b,cos);assert len(opt.state)==0
        corrected,original,residual=b(s['x'],s['geometry'],s['s'],return_details=True);total,parts=losses(corrected,s,True)
        gradients={}
        for name,value in parts.items():
            gr=torch.autograd.grad(value,(b.cos_residual.weight,b.outconv_seg.weight),retain_graph=True,allow_unused=True)
            gradients[name]=dict(residual=float(gr[0].norm()),backbone=float(gr[1].norm()))
            assert all(g is not None and torch.isfinite(g).all() for g in gr)
        outer=torch.as_tensor(s['nuclei']==0,device=corrected.device)
        grad=torch.autograd.grad(sum(v for k,v in parts.items() if k!='anchor'),residual,retain_graph=True)[0]
        assert grad[:,:,outer].abs().sum()>0
        assert gradients['anchor']['residual']>0 and gradients['anchor']['backbone']>0
        total.backward();opt.step();assert b.cos_residual.weight.abs().sum()>0
        # Intercept marker tables to prove invalid types never request the SMC row.
        unknown_id=int(s['cell_ids'][0]);oldtype=ds.type_by_id[unknown_id];ds.type_by_id[unknown_id]=-1
        unknown=get(ds,chosen[0]);idx=list(unknown['cell_ids']).index(unknown_id);assert not unknown['type_valid'][idx] and unknown['p'][idx].sum()==0 and unknown['q'][idx].sum()==0
        ds.type_by_id[unknown_id]=oldtype
        # Full all-known construction equals historical data path, then every original loss.
        known=next(v for v in views if v['k']>0 and all(ds.type_by_id.get(c) is not None and np.isfinite(ds.type_by_id[c]) and float(ds.type_by_id[c]).is_integer() and 0<=ds.type_by_id[c]<len(ds.type_names) for c in v['cell_ids']))
        sn=get(ds,known);so=get(ds_old,known)
        for key in ('x','n','s','p','q'):assert torch.equal(sn[key],so[key])
        z=b(sn['x'],sn['geometry'],sn['s']);_,pn=losses(z,sn);_,po=old_losses(z,so)
        for key in pn:assert torch.allclose(pn[key],po[key],rtol=1e-6,atol=2e-6),key
        evidence[base]=dict(zero_init=records,added_parameters=321,loss_values={k:float(v) for k,v in parts.items()},gradient_norms=gradients,nonnuclear_original_six_residual_gradient=float(grad[:,:,outer].norm()),unknown_mask_test=True,all_known_legacy_tolerance=dict(rtol=1e-6,atol=2e-6,absolute_differences={k:float((pn[k]-po[k]).abs()) for k in pn}))
        del b,opt,ds_old;torch.cuda.empty_cache()
    cfgpath=queue['methods']['B1_COS']['config'];cfg=read(cfgpath);ds=dataset(cfg);views=json.loads((Path(cfg.output_root)/'configs/views.json').read_text())['views'];used=[v for v in views if v['k']][:2]
    model=make_model(cfg,cfg.arm_id,len(ds.gene_names));opt=optimizer(model,cfg);model.train();deps=dependencies(cfg,cfg.arm_id)
    update(model,opt,get(ds,used[0]),True);cursor=views.index(used[0])+1
    draws();draws();draws()
    save(out/'latest.pth',model,opt,deps,1,cursor,{used[0]['index']});atomic_json(out/'latest.json',dict(update=1,cursor=cursor,checkpoint=identity(out/'latest.pth'),dependencies_hash=digest(deps)))
    for key in ('source','config','inputs','views','frozen'):
        bad=copy.deepcopy(deps);bad[key]='tampered'
        try:load(out/'latest.pth',model,opt,bad)
        except ValueError:pass
        else:raise AssertionError('tampered dependency accepted '+key)
    next_random=draws()
    update(model,opt,get(ds,used[1]),True)
    expected=dict(model=tensor_hashes(model.state_dict()),optimizer=tensor_hashes(opt.state_dict()),rng=tensor_hashes(rng_state()),next_view=views[views.index(used[1])+1],consumed_view=used[1],sampler_cursor=views.index(used[1])+1,next_random=next_random)
    atomic_json(out/'continuous.json',expected)
    del model,opt;torch.cuda.empty_cache()
    subprocess.run([sys.executable,'-m','scripts.cos.tests.real_model','--child','--config',cfgpath,'--out',str(out)],check=True)
    actual=json.loads((out/'resumed.json').read_text());assert actual==expected,'cross-process continuation diverged'
    atomic_json(out/'complete.json',dict(status='succeeded',real_model=evidence,resume_new_process_bitwise_equal=True,dependency_failure_injection=True,checkpoint=identity(out/'latest.pth')))
    print(json.dumps(dict(status='succeeded',output=str(out/'complete.json'))),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--queue');p.add_argument('--out',required=True);p.add_argument('--child',action='store_true');p.add_argument('--config');a=p.parse_args()
    if a.child:child(a.config,Path(a.out))
    else:main(a.queue,Path(a.out))
