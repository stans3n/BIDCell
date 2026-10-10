"""Finite-logit, tied-owner, empty-support and zero-expression edge cases."""
import argparse
from pathlib import Path
import numpy as np,torch
from scripts.ssp.method import teacher,replace_pos,PrototypeResidual
from scripts.crgf.common import atomic_json,source_identity

def fixture(k=1):
    p=torch.zeros(k,1,7,7,dtype=torch.long);expr=torch.zeros(7,7,2);expr[3,5,0]=2
    return dict(p=p,q=p.clone(),s=torch.ones_like(p),geometry=torch.cat((p.float(),torch.full_like(p.float(),.1),p.float(),p.float()),1),type_valid=torch.ones(k,dtype=torch.bool),local_expression=expr,exclusive_positive=torch.tensor([[1.,0.]]).expand(k,-1),nuclei=np.zeros((7,7),np.int32))
def main(root):
    torch.set_num_threads(2);s=fixture();t=teacher(s);assert t['H'][3,3] and t['target'][0,3,3]>0
    for scale in (0.,1000.,-1000.):
        z=torch.stack((torch.full((1,7,7),scale),torch.full((1,7,7),-scale)),1).requires_grad_();base=torch.nn.functional.cross_entropy(z,s['p'][:,0],reduction='sum');loss=replace_pos(base,z,s,t);g=torch.autograd.grad(loss,z)[0];assert torch.isfinite(loss) and torch.isfinite(g).all()
    tied=teacher(fixture(2));assert not tied['H'].any() and torch.count_nonzero(tied['c'])==0
    empty=fixture();empty['local_expression'].zero_();et=teacher(empty);assert torch.equal(et['bg'],torch.ones_like(et['bg'])) and not et['H'].any()
    outside=fixture();outside['s'].zero_();assert not teacher(outside)['H'].any()
    nuclear=fixture();nuclear['nuclei']=np.ones((7,7),np.int32);assert not teacher(nuclear)['H'].any()
    weak=fixture();weak['geometry'][:,1]=1;assert not teacher(weak)['H'].any()
    head=PrototypeResidual(np.ones((2,16),np.float32)/4);r=torch.tensor([[.5,.5]])
    out=head(torch.zeros(1,320,7,7),empty['local_expression'],r,torch.zeros(1),empty['geometry'],empty['s']);assert torch.isfinite(out).all() and not torch.count_nonzero(out)
    atomic_json(Path(root)/'tests/math_edges.json',dict(status='passed',checks=['K1_new_evidence','extreme_finite_logits_and_gradients','tied_owner_zero_confidence','zero_expression_bg1','empty_support_no_H','all_nuclear_no_H','weak_geometry_no_H','zero_prototype_confidence_zero_finite_residual'],source=source_identity(['scripts/ssp/tests/math_edges.py','scripts/ssp/method.py'])))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();main(a.root)
