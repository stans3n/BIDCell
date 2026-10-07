"""Independent incremental acceptance checks, invoked before pilot training."""
import json
from pathlib import Path
import numpy as np
import tifffile
import torch
from scipy.spatial import cKDTree
from scipy import ndimage as ndi
from .adapter import features, ResidualAdapter
from bidcell.model.utils.utils import get_seg_mask


def run(out):
    from .pilot import load_view, SEED, MACHINE, save
    spec=json.loads((out/'spec.json').read_text())
    torch.manual_seed(SEED)
    c,h,w=4,3,3
    z=torch.randn(c,2,h,w);support=torch.tensor([[[1,1,1],[1,0,0],[1,1,0]],[[1,1,1],[1,1,0],[1,1,0]],[[1,1,0],[0,1,1],[1,0,0]],[[1,0,1],[1,1,1],[1,0,0]]],dtype=torch.bool)
    nuc=torch.zeros_like(support);nuc[1,0,0]=True
    radii=torch.tensor([2.,3.,5.,7.]);distance=torch.rand(c,h,w)*3
    # Exact normalized-distance tie with different radii and canonical IDs.
    distance[0,0,1]=radii[0]+1e-6;distance[1,0,1]=radii[1]+1e-6
    support[:,0,1]=True;distance[3,0,1]=20.
    z[:,0,0,1]=0.;z[:,1,0,1]=torch.tensor([2.,2.,1.,4.])
    ids=torch.tensor([81,12,43,7]);density=torch.ones(h,w)
    a,r,g,adv=features(z,support,nuc,distance,radii,density,ids)
    assert float(a[2,0,1,7])==float(torch.tensor(.3)), 'geometry tie must choose canonical ID12 radius3'
    assert float(a[2,0,1,1])==1., 'score competitor is distinct ID7 ell4, not geometry competitor'
    order=torch.tensor([2,0,3,1]);ap,rp,gp,dp=features(z[order],support[order],nuc[order],distance[order],radii[order],density,ids[order])
    torch.testing.assert_close(a[order],ap,atol=0,rtol=0);torch.testing.assert_close(r[order],rp,atol=0,rtol=0)
    for name in ('E02','E04','E05'):
        model=ResidualAdapter(name)
        torch.nn.init.normal_(model.head[-1].weight,0,.05)
        z1,d1=model(z,a,r,g,adv);z2,d2=model(z[order],ap,rp,gp,dp)
        torch.testing.assert_close(z1[order],z2,atol=1e-6,rtol=1e-6)
        canonical=torch.argsort(ids);canonical2=torch.argsort(ids[order])
        sample_n=np.array([[0,7,12],[43,81,0],[0,0,0]])
        assert np.array_equal(get_seg_mask(z1.detach().numpy()[canonical],sample_n),get_seg_mask(z2.detach().numpy()[canonical2],sample_n))
    global_nuc=tifffile.imread(MACHINE['nuclei_mask']);obj=ndi.find_objects(global_nuc)
    closure_checks=0;violations=[]
    for b in spec['blocks']:
        for cid in b['candidate_ids']:
            sl=obj[cid-1];lo=np.array([sl[0].start,sl[1].start])-56;hi=np.array([sl[0].stop,sl[1].stop])+56
            origin=np.array([b['y'],b['x']])
            offset=b['core_offset'];end=offset+b['core_size'];size=b['context_size']
            if np.all(lo<origin+end) and np.all(hi>origin+offset):
                if not (np.all(lo>=origin) and np.all(hi<=origin+size)):
                    violations.append(dict(block=b['id'],candidate_id=int(cid),full_bbox=[[sl[0].start,sl[0].stop],[sl[1].start,sl[1].stop]],expanded_lo=lo.tolist(),expanded_hi=hi.tolist(),context_origin=origin.tolist(),context_stop=(origin+size).tolist(),missing_low=np.maximum(origin-lo,0).tolist(),missing_high=np.maximum(hi-origin-size,0).tolist()))
                closure_checks+=1
    save(out/'geometry_closure_diagnosis.json',dict(violations=violations,count=len(violations),closure_checks=closure_checks))
    assert not violations,'full candidate context not closed; see geometry_closure_diagnosis.json'
    n_checked=0;edge_checked=0;g_transcript_empty=0
    for v in spec['schedule'][:50]:
        p=load_view(out,v)
        expected=p['targets']['sa'][:,0].bool() & (p['targets']['sa'][:,0].sum(0)[None]>=2) & torch.from_numpy(p['nuclei']==0)[None]
        assert torch.equal(expected,p['gate'])
        g_transcript_empty+=int((p['gate'] & torch.from_numpy(p['expression_sum']==0)[None]).sum())
        y,x=p['coords'];yy,xx=np.mgrid[y:y+48,x:x+48];q=np.stack((yy.ravel(),xx.ravel()),1)
        for j,cid in enumerate(p['ids']):
            sl=obj[cid-1]
            if not (sl[0].start<y or sl[0].stop>y+48 or sl[1].start<x or sl[1].stop>x+48):continue
            coords=np.argwhere(global_nuc[sl]==cid)+[sl[0].start,sl[1].start]
            # Independent brute force at selected pixels, not another tree call.
            chosen=np.array([0,47,1127,2303]);expected=np.sqrt(((q[chosen,None]-coords[None])**2).sum(2)).min(1)
            raw=np.arange(48*48).reshape(48,48)
            if v['flip']:raw=np.flip(raw,0)
            raw=np.rot90(raw,-v['rot90_clockwise'])
            recovered=np.empty((48*48,));recovered[raw.ravel()]=p['distances'][j].ravel()
            np.testing.assert_allclose(recovered[chosen],expected,atol=2e-6,rtol=1e-6)
            edge_checked+=1
        n_checked+=1
    assert edge_checked>0,'must exercise full nucleus extending outside native patch'
    assert g_transcript_empty>0,'G must extend beyond transcript occupied pixels'
    save(out/'acceptance_geometry.json',dict(status='PASS',nonidentical_multicandidate_reorder=True,canonical_decoded_labels_equal=True,canonical_geometry_ties=True,score_competitor_distinct=True,complete_possible_support_context_checks=closure_checks,real_views=n_checked,patch_edge_nuclei_independently_checked=edge_checked,G_on_transcript_empty_candidate_pixels=g_transcript_empty))
    print('EXTENDED GEOMETRY ACCEPTANCE PASS',flush=True)
