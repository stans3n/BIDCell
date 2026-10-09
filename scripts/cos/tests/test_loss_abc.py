import copy
import numpy as np
import pytest
import torch
from scripts.cos.training import losses,pos_components
from scripts.cos.tests.test_math import sample

def test_exact_region_value_gradient_and_unknown_K():
    torch.manual_seed(17);s=sample(3);s['type_valid'][2]=False
    s['p'].zero_();s['p'][0,0,3,2]=1;s['s'][0,0,3,3]=0
    z=torch.randn(3,2,4,4,requires_grad=True)
    b,pb=losses(z,s,True,21.5,1.);c,pc=losses(z,s,True,21.5,.5)
    u,w,counts=pos_components(z,s,.5)
    torch.testing.assert_close(pc['pos'],pb['pos']-.5*u['u_background'])
    torch.testing.assert_close(torch.autograd.grad(c,z,retain_graph=True)[0],torch.autograd.grad(b-.5*u['u_background'],z,retain_graph=True)[0])
    diff=torch.autograd.grad(pb['pos']-pc['pos'],z,retain_graph=True)[0]
    outside=(torch.as_tensor(s['nuclei'])[None]==0)&(s['s'][:,0]>0)&(s['p'][:,0]==0)&s['type_valid'][:,None,None]
    assert torch.count_nonzero(diff.permute(0,2,3,1)[~outside])==0
    assert counts['K']==3 and counts['unknown_candidates']==1
    for k in ('ne','os','cc','ov','neg','anchor'):assert torch.equal(pb[k],pc[k])
    ce=torch.nn.functional.cross_entropy(z,s['p'][:,0],reduction='none')
    explicit=(ce*outside*.5).sum()/3
    torch.testing.assert_close(pb['pos']-pc['pos'],explicit)
    # Other nuclei are explicitly outside U, even for non-own candidate.
    assert not outside[:,0,0].any() and not outside[:,1,1].any()
    assert not outside[0,3,2] and not outside[0,3,3]

def test_empty_U_rho_identity_and_anchor_linear():
    for k in (1,3):
        s=sample(k);s['s'].zero_();z=torch.randn(k,2,4,4,requires_grad=True)
        a,pa=losses(z,s,True,0.,1.);b,pb=losses(z,s,True,1.,1.);c,pc=losses(z,s,True,1.,.5)
        torch.testing.assert_close(b,c)
        torch.testing.assert_close(torch.autograd.grad(b,z,retain_graph=True)[0],torch.autograd.grad(c,z,retain_graph=True)[0])
        torch.testing.assert_close(b-a,pa['anchor'])
        d,_=losses(z,s,True,7.,1.)
        torch.testing.assert_close(torch.autograd.grad(d,z,retain_graph=True)[0],torch.autograd.grad(a+7*pa['anchor'],z)[0])
    with pytest.raises(ValueError):losses(torch.empty(0,2,4,4),sample(0),True,0.,1.)

def test_historical_COS_value_and_gradient():
    # Load the authoritative unchanged module under a separate package-qualified name.
    import importlib.util
    path='/home/zwan0726/work/projects/BIDCell_cos_common_20261009/scripts/cos/training.py'
    spec=importlib.util.spec_from_file_location('scripts.cos.historical_training_fixture',path);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    for unknown in (False,True):
        s=sample(3);s['type_valid'][2]=not unknown;z=torch.randn(3,2,4,4,requires_grad=True)
        a,pa=losses(z,s,True,1.,1.);b,pb=old.losses(z,s,True)
        assert torch.equal(a,b)
        for key in pa:assert torch.equal(pa[key],pb[key])
        assert torch.equal(torch.autograd.grad(a,z,retain_graph=True)[0],torch.autograd.grad(b,z)[0])


def test_stratum_fallback_order():
    from scripts.cos.prepare_loss import select
    items=[dict(k=k,view_cursor=i) for i,k in enumerate([7,1,3,1,7,1,7])]
    result=select(items,[1,3,2])
    assert result['selected_counts']=={'K1':1,'K2_4':1,'K5plus':2}
    assert result['fallback_cursors']==[3,5]
    assert [v['view_cursor'] for v in result['samples']]==[0,1,2,3,4,5]
