"""Historical B off-switch and S/SP common initialization on real data."""
import argparse,copy,json,subprocess,sys
from pathlib import Path
import torch
from scripts.crgf.common import atomic_json,digest,source_identity
from scripts.cos.config import read as old_read
from scripts.cos.training import make_model as old_model,losses as old_loss
from scripts.cos.tests.real_model import tensor_hashes
from scripts.cos.ownership import decode
from scripts.bs4.config import read
from scripts.bs4.training import make_model,losses,dataset,forward,optimizer,rng_state
from scripts.bs4.loss_diagnostics import get
HIST=Path('/home/zwan0726/work/outputs/cos_loss_20261009T160351+1100')
def main(root):
    root=Path(root);torch.set_num_threads(2);cfg=read(root/'T1/configs/run.json');ds=dataset(cfg)
    views=json.loads((root/'configs/views.json').read_text())['views'];known=next(v for v in views if v['k']>=2 and all(ds.type_by_id.get(c,-1)>=0 for c in v['cell_ids']))
    v=next(v for v in views if v['k']>=2 and any(ds.type_by_id.get(c,-1)<0 for c in v['cell_ids']));s=get(ds,v)
    sample_file=root/'tests/historical_reference_sample.pth';torch.save({k:v.cpu() if torch.is_tensor(v) else v for k,v in s.items()},sample_file)
    subprocess.run([sys.executable,str(Path(__file__).with_name('historical_reference.py')),str(sample_file),str(root/'tests/historical_reference.json')],check=True,cwd='/home/zwan0726/work/projects/BIDCell_cos_loss_20261009T160351')
    reference=json.loads((root/'tests/historical_reference.json').read_text())
    b=old_model(old_read(HIST/'B/configs/run.json'),'B',len(ds.gene_names));a=make_model(cfg,'T1',len(ds.gene_names));rng=digest(tensor_hashes(rng_state()))
    pcfg=read(root/'T2/configs/run.json');p=make_model(pcfg,'T2',len(ds.gene_names));assert digest(tensor_hashes(rng_state()))==rng
    assert {k:v for k,v in tensor_hashes(p.state_dict()).items() if not k.startswith('prototype_residual.')}==tensor_hashes(a.state_dict())==tensor_hashes(b.state_dict())
    for m in (a,b,p):m.eval()
    with torch.no_grad():
        za=forward(a,s);zb=b(s['x'],s['geometry'],s['s']);zp=forward(p,s)
        assert torch.equal(za,zb) and torch.equal(za,zp)
        assert tensor_hashes(za.detach())==reference['eval']
        import numpy as np
        assert np.array_equal(decode(za.cpu().numpy(),s['nuclei'],s['cell_ids']),decode(zp.cpu().numpy(),s['nuclei'],s['cell_ids']))
    del p;torch.cuda.empty_cache()
    a.train();b.train();oa=optimizer(a,cfg);ob=optimizer(b,cfg)
    za=forward(a,s);zb=b(s['x'],s['geometry'],s['s']);assert torch.equal(za,zb)
    la,pa=losses(za,s,True,cfg.anchor_weight,1.,s_enabled=False);lb,pb=old_loss(zb,s,True,cfg.anchor_weight,1.)
    scalar_differences={k:dict(new=float(pa[k]),historical=reference['parts'][k],same_process_old=float(pb[k])) for k in pa}
    atomic_json(root/'tests/equivalence_scalar_observation.json',dict(status='observed',parts=scalar_differences))
    assert all(float(pa[k])==reference['parts'][k] for k in pa if k not in ('ne','pos'))
    assert all(path.startswith('/home/zwan0726/work/projects/BIDCell_cos_loss_20261009T160351/') for path in reference['source_imports'].values())
    assert all(torch.equal(pa[k],pb[k]) for k in pa if k not in ('ne','pos'))
    la.backward();lb.backward();assert tensor_hashes(za.detach())==reference['train'] and {n:tensor_hashes(p.grad) for n,p in a.named_parameters()}==reference['gradients']
    assert all(torch.equal(x.grad,y.grad) for x,y in zip(a.parameters(),b.parameters()))
    oa.step();ob.step();assert tensor_hashes(a.state_dict())==tensor_hashes(b.state_dict()) and tensor_hashes(oa.state_dict())==tensor_hashes(ob.state_dict())
    assert tensor_hashes(a.state_dict())==reference['model'] and tensor_hashes(oa.state_dict())==reference['optimizer']
    # Empty H executes same legacy loss object and next Adam exactly on real crop with zero new evidence.
    empty=dict(s,exclusive_positive=torch.zeros_like(s['exclusive_positive']));empty.pop('_teacher',None)
    from scripts.bs4.method import teacher
    assert not teacher(empty)['H'].any()
    oa.zero_grad(set_to_none=True);ob.zero_grad(set_to_none=True)
    za=forward(a,empty);zb=b(empty['x'],empty['geometry'],empty['s'])
    la,_=losses(za,empty,True,cfg.anchor_weight);lb,_=old_loss(zb,empty,True,cfg.anchor_weight)
    empty_scalar=dict(new=float(la),old=float(lb));la.backward();lb.backward();assert all(torch.equal(x.grad,y.grad) for x,y in zip(a.parameters(),b.parameters()))
    oa.step();ob.step();assert tensor_hashes(a.state_dict())==tensor_hashes(b.state_dict())
    # Original all-known CUDA reduction repeat spread, without changing the historical path.
    sk=get(ds,known);a.eval()
    zk=forward(a,sk)
    old_values=[];new_values=[];old_ne=[];new_ne=[];g_reference=None
    for rep in range(10):
        lo,po=old_loss(zk,sk,True,cfg.anchor_weight);ln,pn=losses(zk,sk,True,cfg.anchor_weight,s_enabled=False)
        old_values.append(float(po['pos']));new_values.append(float(pn['pos']));old_ne.append(float(po['ne']));new_ne.append(float(pn['ne']))
        go=torch.autograd.grad(lo,zk,retain_graph=True)[0];gn=torch.autograd.grad(ln,zk,retain_graph=True)[0]
        assert torch.equal(go,gn)
        if g_reference is None:g_reference=go
        else:assert torch.equal(go,g_reference)
        assert all(torch.equal(po[key],pn[key]) for key in po if key not in ('ne','pos'))
    def cpu_sample(sample):
        return {k:v.detach().cpu() if torch.is_tensor(v) else v for k,v in sample.items() if k!='_teacher'}
    torch.save({'all_known':(zk.detach().cpu(),cpu_sample(sk)),'mixed_unknown':(za.detach().cpu(),cpu_sample(empty))},root/'tests/cpu_loss_fixtures.pth')
    atomic_json(root/'tests/historical_cuda_reduction.json',dict(status='observed',patch=known,old_NE_repeats=old_ne,new_NE_repeats=new_ne,old_POS_repeats=old_values,new_disabled_POS_repeats=new_values,old_range=max(old_values)-min(old_values),new_range=max(new_values)-min(new_values),logits_identical=True,all_repeated_logits_gradients_bitwise=True,limitation='original all-known CUDA nll_loss2d sum uses nondeterministic reduction; no loss or kernel altered'))
    atomic_json(root/'tests/equivalence.json',dict(status='passed',historical_B_forward_grad_next_adam_bitwise=True,loss_scalar_observation=scalar_differences,empty_H_scalar_observation=empty_scalar,empty_H_POS_original_path_and_grad_next_adam_bitwise=True,scalar_reduction_review_required=True,SP_zero_init_logits_decode_and_common_RNG_bitwise=True,patch=v,independent_historical_imports=reference['source_imports'],source=source_identity(__import__('scripts.bs4.training',fromlist=['TRAIN_SOURCES']).TRAIN_SOURCES)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();main(a.root)
