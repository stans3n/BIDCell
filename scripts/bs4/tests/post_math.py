"""Independent historical-function and full-coordinate brute-force checks."""
import argparse,json,random,hashlib,time
from pathlib import Path
import numpy as np
from scripts.bs4.post_rules import connect,FullNucleusIndex,candidate
from bidcell.model.postprocess_predictions import postprocess_connect,remove_islands
from scripts.crgf.common import atomic_json,source_identity

def run(output):
    start=time.monotonic();rng=np.random.default_rng(9817);cases=0;veto=changed=0
    for z in range(24):
        n=np.zeros((44,53),np.uint32)
        for cid in (3,9,17):
            for _ in range(5):n[rng.integers(44),rng.integers(53)]=cid
        image=rng.choice([0,0,0,3,9,17],size=(24,29)).astype(np.uint32);origin=(7,11);nt=n[7:31,11:40]
        index=FullNucleusIndex(n,max_pairs=7)
        random.seed(z);baseline=postprocess_connect(image,nt);state=random.getstate();islands=remove_islands(baseline,nt)
        results={}
        for rule in ('B','P1','P2'):
            random.seed(z);out,diag,arrays,masks=connect(image,nt,rule,index,origin,capture=True)
            assert random.getstate()==state
            results[rule]=(out,diag,arrays,masks)
            if rule=='B':
                assert np.array_equal(out,baseline);assert np.array_equal(remove_islands(out,nt),islands)
            if rule=='P1':
                expected=np.zeros_like(image)
                for cid in diag['candidate_order']:expected[masks[cid]&((image==0)|(image==cid))]=cid
                expected=np.where(nt>0,nt,expected);assert np.array_equal(expected,out);veto+=diag['veto_proposal_pixels']
            if rule=='P2':
                expected=np.zeros_like(image)
                for y,x in np.ndindex(image.shape):
                    proposals=[cid for cid,m in masks.items() if m[y,x]]
                    if len(proposals)==1:expected[y,x]=proposals[0]
                    elif proposals:
                        distances={cid:int(((np.argwhere(n==cid)-[y+origin[0],x+origin[1]])**2).sum(1).min()) for cid in proposals}
                        expected[y,x]=min(proposals,key=lambda cid:(distances[cid],cid))
                expected=np.where(nt>0,nt,expected);assert np.array_equal(out,expected);assert index.max_observed_pairs<=7;changed+=diag['changed_vs_B']
        assert results['B'][1]['candidate_hashes']==results['P1'][1]['candidate_hashes']==results['P2'][1]['candidate_hashes'];cases+=1
    assert veto>0 and changed>0
    n=np.zeros((10,10),np.uint32);n[3,2]=7;n[3,4]=2;n[9,9]=7
    ix=FullNucleusIndex(n,3)
    assert ix.squared(7,[[3,3]])[0]==ix.squared(2,[[3,3]])[0]==1
    assert ix.squared(7,[[8,9]])[0]==1
    assert ix.squared(2,[[3,2]])[0]==4
    for image in [np.full((2,2),99,np.uint32),np.array([[99,2],[99,2]],np.uint32)]:
        try:connect(image,n[:2,:2],'P2',ix)
        except ValueError as e:assert 'missing complete nucleus' in str(e)
        else:raise AssertionError('missing nucleus accepted')
    evidence=dict(status='passed',random_cases=cases,veto_proposals=veto,changed_P2=changed,oracle='independent all global nuclear coordinates per candidate query',bounded_query_pairs=7,missing_unique_and_multiple_candidate_nucleus_rejected=True,wall_seconds=time.monotonic()-start,source=source_identity(['scripts/bs4/post_rules.py','scripts/bs4/tests/post_math.py','bidcell/model/postprocess_predictions.py']))
    atomic_json(output,evidence);print(json.dumps(evidence))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();run(a.output)
