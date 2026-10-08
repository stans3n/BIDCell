import numpy as np
import pytest
import torch
from scripts.cos.ownership import candidate_order,decode,log_ownership,anchor_loss,compact_sample
from scripts.cos.training import losses
from scripts.crgf.training import losses as legacy_losses

def sample(k=3):
    n=torch.zeros(k,1,4,4,dtype=torch.long)
    for i in range(k):n[i,0,i,i]=1
    return dict(n=n,s=torch.ones_like(n),p=torch.randint(0,2,n.shape),q=torch.randint(0,2,n.shape),type_valid=torch.ones(k,dtype=torch.bool),nuclei=(n[:,0]*torch.arange(1,k+1)[:,None,None]).sum(0).numpy(),cell_ids=np.arange(1,k+1,dtype=np.int64))

def test_k0_k1_large_and_limit():
    z=torch.empty(0,2,4,4,requires_grad=True); assert torch.equal(log_ownership(z),torch.zeros(1,4,4))
    z=torch.randn(1,2,4,4); assert torch.allclose(log_ownership(z).exp(),z.softmax(1)[0])
    for k in [2,3,20]:
        z=torch.full((k,2,1,1),-1000.);z[:,0]=1000.;z[0,0]=-1000.;z[0,1]=1000.
        q=log_ownership(z);assert torch.isfinite(q).all();assert q[1].exp().item()==pytest.approx(1/(2-1/k),rel=1e-5)
        assert -q[1].item()==pytest.approx(np.log(2-1/k),rel=1e-5)

def test_tie_permutation_padding():
    nuc=np.array([[2,1],[0,0]],np.int64);ids=np.array([1,2]);z=np.zeros((2,2,2,2),np.float32)
    assert not decode(z,nuc,ids,restore=False).any()
    z[:,1]=2;expected=np.ones((2,2),np.uint32)
    np.testing.assert_array_equal(decode(z,nuc,ids,restore=False),expected)
    np.testing.assert_array_equal(decode(z[::-1],nuc,ids[::-1],restore=False),expected)
    padded=np.concatenate((z,np.ones_like(z[:1])*20));pid=np.array([1,2,0]);mask=np.array([1,1,0],bool)
    np.testing.assert_array_equal(decode(padded,nuc,pid,mask,False),expected)
    assert torch.allclose(log_ownership(torch.from_numpy(padded),mask)[:3],log_ownership(torch.from_numpy(z)))
    for bad in [np.array([1,1]),np.array([-1,2]),np.array([1,3])]:
        with pytest.raises(ValueError):candidate_order(bad,nuc)
    with pytest.raises(FloatingPointError):log_ownership(torch.full((1,2,1,1),float('nan')))

def test_anchor_gradients_and_outside_scope():
    s=sample();z=torch.randn(3,2,4,4,requires_grad=True);loss=anchor_loss(z,s['nuclei'],s['cell_ids']);loss.backward()
    for i in range(3):
        assert z.grad[i,1,i,i]<0
        for j in range(3):
            if i!=j:assert z.grad[j,1,i,i]>0
    assert torch.equal(z.grad[:,:,3,3],torch.zeros(3,2))
    z=torch.empty(0,2,4,4,requires_grad=True);a=anchor_loss(z,np.zeros((4,4),np.int32),np.array([],dtype=np.int64))
    a.backward();assert z.grad.abs().sum()==0

def test_unknown_total_k_and_all_known_legacy():
    torch.manual_seed(5);s=sample();z=torch.randn(3,2,4,4,requires_grad=True)
    old,op=legacy_losses(z,s);new,np_=losses(z,s)
    for key in op:assert torch.equal(op[key],np_[key]),key
    assert torch.allclose(old,new)
    s['type_valid'][:]=False;total,parts=losses(z,s,True)
    assert parts['pos']==0 and parts['neg']==0 and total!=0
    g=torch.autograd.grad(parts['pos']+parts['neg'],z,retain_graph=True)[0];assert g.abs().sum()==0
    s['type_valid'][0]=True;_,parts=losses(z,s)
    expected=torch.nn.functional.cross_entropy(z[:1],s['p'][:1,0],reduction='sum')/3
    assert torch.allclose(parts['pos'],expected)


def test_frozen_transform_all_eight_matches_imgaug():
    import imgaug.augmenters as iaa
    from bidcell.model.dataio.candidate_geometry import transform
    image=np.random.RandomState(17).rand(48,48,7)
    for flip in (0,1):
        for rotation in range(4):
            legacy=iaa.Flipud(1.0)(images=image[None])[0] if flip else image.copy()
            legacy=iaa.Rot90(rotation,keep_size=True)(images=legacy[None])[0] if rotation else legacy
            np.testing.assert_array_equal(transform(image,flip,rotation),legacy)
