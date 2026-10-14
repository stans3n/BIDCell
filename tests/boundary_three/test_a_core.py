"""Scientific invariants; tests do not assert duplicated implementation literals."""
import hashlib
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from scipy import sparse
from scripts.crgf.matrix import grid
from scripts.boundary_three.core_scores import (molecule_fold,profile,matched_donors,cross_scores,
    paired_bootstrap,decide,reference_arrays,legal_types,PARAMETERS)
from scripts.boundary_three.core_refine import (NuclearSupport,interface_band,fixed_domain,
    score_proposals,apply_proposals,topology_oracle,CompleteCellTopology)


class CoreStatisticsTests(unittest.TestCase):
    def test_native_lookup_independent_two_dimensional_oracle(self):
        # Real scale plus nonintegral dimensions, ties, negative/OOB values.
        mask=np.arange(7*13,dtype=np.int32).reshape(7,13);scale=.2125
        points=pd.DataFrame(dict(x_location=[0,.5*scale,1.5*scale,2.5*scale,5.55,12.,13.,-.1,-1.],
                                y_location=[0,.5*scale,1.5*scale,2.5*scale,2.7,6.,7.,.2,-1.]))
        x,y,valid=grid(points,mask.shape,scale,scale)
        enlarged=cv2.resize(mask,(round(mask.shape[1]/scale),round(mask.shape[0]/scale)),interpolation=cv2.INTER_NEAREST)
        xi=(points.x_location/scale).round().to_numpy(int);yi=(points.y_location/scale).round().to_numpy(int)
        good=(xi>=0)&(yi>=0)&(xi<enlarged.shape[1])&(yi<enlarged.shape[0])
        self.assertTrue(np.array_equal(valid,good));self.assertTrue(np.array_equal(mask[y[valid],x[valid]],enlarged[yi[good],xi[good]]))

    def test_complete_nucleus_eroded_core_and_degenerate(self):
        nuc=np.zeros((11,17),np.uint32);nuc[0:5,0:5]=1;nuc[4:9,8:13]=2;nuc[2,15]=3
        ev=NuclearSupport(nuc,[1,2,3],[0,0,-1],['A']);core,area=ev.core_image()
        self.assertEqual(area.tolist(),[9,9,0]);self.assertEqual(core[0,0],0);self.assertEqual(core[1,1],1)
        pixels=np.arange(nuc.size);off,rows,_=ev.candidate_csr(pixels)
        for p in (0,6*17+10,nuc.size-1):
            got=set(ev.ids[rows[off[p]:off[p+1]]]);expected=set()
            for cid in (1,2,3):
                y,x,_,_=ev.support(cid)
                if np.any((y==p//17)&(x==p%17)):expected.add(cid)
            self.assertEqual(got,expected)

    def test_sha_split_reference_and_profile(self):
        ids=np.array([0,1,23,24,2**63+99],np.uint64)
        folds=molecule_fold(ids)
        self.assertEqual(folds.tolist(),[hashlib.sha256(('20261015|'+str(int(i))).encode()).digest()[0]&1 for i in ids])
        self.assertFalse(set(ids[folds==0])&set(ids[folds==1]))
        ref=dict(gene_names=['b','a'],gene_indices=[1,0],theta=[[.2,.8],[0.,0.]],type_names=['T','bad'],valid=[True,False])
        gi,theta,valid=reference_arrays(ref,['a','b','c'])
        self.assertEqual(legal_types([-1,0,1,2],valid).tolist(),[False,True,False,False])
        self.assertTrue(np.array_equal(profile(np.zeros((1,2)),theta[:1]),theta[:1]))
        q=profile(np.array([[9,1]]),theta[:1]);self.assertAlmostEqual(float(q.sum()),1)
        with self.assertRaises(ValueError):reference_arrays(ref,['b','a','c'])

    def test_donors_and_paired_cross_validation(self):
        counts=np.array([[[12,0],[0,12],[10,2],[2,10],[40,0],[4,6]],[[0,14],[14,0],[2,10],[10,2],[41,0],[5,5]]])
        types=np.array([0,0,0,0,0,-1]);theta=np.array([[.5,.5]])
        result=cross_scores(counts,types,theta,np.array([True]))
        self.assertEqual(result['paired'].tolist(),[True,True,True,True,False,False])
        for s in range(20):
            for fold in (0,1):
                d=result['donors'][s,fold,:4]
                self.assertEqual(sorted(d.tolist()),[0,1,2,3]);self.assertTrue(np.all(d!=np.arange(4)))
        # A held-out reversal must hurt; test would catch self-leakage or swapped split.
        self.assertTrue(np.all(result['core'][:4]<result['type'][:4]))
        d,b=matched_donors(counts.sum(0),types,result['eligible'],2026101500)
        self.assertEqual(d[4],-1);self.assertEqual(d[5],-1)

    def test_paired_bootstrap_and_gate_boundaries(self):
        xy=np.c_[np.repeat(np.arange(10)*1000,100),np.zeros(1000)]
        values=np.tile([-.5,-.52,-.51],(1000,1));r,a=paired_bootstrap(values,xy,1000)
        self.assertEqual(decide(r)['science_status'],'GO_A2');self.assertTrue(np.allclose(a[:,0],.02))
        self.assertTrue(np.allclose(a[:,1],.01))
        values[:,0]=-.53;r,_=paired_bootstrap(values,xy,1000)
        self.assertEqual(decide(r)['science_status'],'NO_USEFUL_INCREMENT')
        r['n_cells']=999;self.assertEqual(decide(r)['science_status'],'INCONCLUSIVE')
        r['n_cells']=1000;r['core_minus_type']['mean']=.01;r['core_minus_type']['ci95']=[0.,.03];r['core_minus_shuffle']['ci95']=[-.01,.02]
        self.assertEqual(decide(r)['science_status'],'INCONCLUSIVE')


class RefinementTests(unittest.TestCase):
    def fixture(self):
        base=np.zeros((12,17),np.uint32);base[2:10,1:8]=1;base[2:10,8:15]=2
        nuc=np.zeros_like(base);nuc[5:7,3:5]=1;nuc[5:7,11:13]=2
        return base,nuc

    def test_same_domain_abstains_all_arms(self):
        base,nuc=self.fixture();pixels=np.array([5*17+7]);off=np.array([0,2]);rows=np.array([0,1]);expr=sparse.csr_matrix([[2,3]])
        pos,_,_,reason=fixed_domain(base,nuc,pixels,off,rows,np.array([1,2]),np.array([True,True]),np.array([1,-1]),expr)
        self.assertEqual(len(pos),0);self.assertEqual(reason['candidate_missing_full_count_donor'],1)
        pos,_,r,_=fixed_domain(base,nuc,pixels,off,rows,np.array([1,2]),np.array([True,True]),np.array([1,0]),expr)
        self.assertEqual(pos.tolist(),[0]);self.assertEqual(r.tolist(),[0,1])

    def test_pixel_multigene_margin_tie_and_disabled(self):
        base,nuc=self.fixture();pixels=np.array([5*17+7,6*17+7]);off=np.array([0,2,4]);rows=np.array([0,1,0,1]);geometry=np.zeros(4)
        expr=sparse.csr_matrix([[1,9],[1,9]]);prof=np.array([[.99,.01],[.01,.99]])
        p,s,below=score_proposals(base,pixels,off,rows,geometry,expr,np.array([1,2]),prof)
        self.assertEqual(len(p),2);self.assertEqual(p['y'].tolist(),[5,6]);self.assertEqual(p['target'].tolist(),[2,2])
        p2,_,_=score_proposals(base,pixels,off,rows,geometry,expr,np.array([1,2]),prof,disabled=True);self.assertEqual(len(p2),0)
        tied=np.array([[.5,.5],[.5,.5]]);p3,_,_=score_proposals(base,pixels,off,rows,geometry,expr,np.array([1,2]),tied);self.assertEqual(len(p3),0)

    def test_complete_topology_bridge_island_nucleus_and_recovery(self):
        base,nuc=self.fixture()
        # Recipient existing island is insufficient for a new rooted attachment.
        island=base.copy();island[0,0]=2
        self.assertEqual(topology_oracle(island,nuc,2,1,1,2),(False,'recipient_not_nucleus_connected'))
        self.assertFalse(topology_oracle(base,nuc,5,3,1,2)[0])
        dtype=[('margin','f8'),('y','i4'),('x','i4'),('source','u4'),('target','u4')]
        proposals=np.array([(2.,y,7,1,2) for y in range(2,10)],dtype=dtype)
        topology=CompleteCellTopology(base,nuc,proposals);mask=base.copy()
        for p in proposals:
            values=tuple(int(p[k]) for k in ('y','x','source','target'))
            self.assertEqual(topology.check(mask,*values),topology_oracle(mask,nuc,*values))
            if topology.check(mask,*values)[0]:mask[values[0],values[1]]=values[3]
        with tempfile.TemporaryDirectory() as tmp:
            full,fs=apply_proposals(base,nuc,proposals,Path(tmp)/'full',{'fixture':'a'},oracle=True)
            partial,ps=apply_proposals(base,nuc,proposals,Path(tmp)/'resume',{'fixture':'a'},stop_after=3,oracle=True)
            self.assertIsNone(partial);self.assertEqual(ps['cursor'],3)
            resumed,rs=apply_proposals(base,nuc,proposals,Path(tmp)/'resume',{'fixture':'a'},oracle=True)
            self.assertTrue(np.array_equal(full,resumed));self.assertEqual(fs,rs)
            self.assertTrue(np.array_equal(full>0,base>0));self.assertTrue(np.array_equal(full[nuc>0],base[nuc>0]))
        # Single bridge pixel removal would split the donor's nucleus-connected arm.
        bridge=np.zeros((7,9),np.uint32);bridge[3,1:8]=1;bridge[2,4]=2
        nuclei=np.zeros_like(bridge);nuclei[3,1]=1;nuclei[2,4]=2
        self.assertEqual(topology_oracle(bridge,nuclei,3,4,1,2),(False,'donor_components_increased'))

if __name__=='__main__':unittest.main()
