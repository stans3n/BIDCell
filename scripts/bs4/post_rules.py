"""Post-candidate interventions; original candidate body retained verbatim.
All distances use complete global nuclear pixel sets and int64 squared pixels.
"""
import hashlib
import random
import pickle
import cv2
import numpy as np
from scipy import ndimage as ndi

RULE_VERSION = 'BS4-postconnect-v2-int64-full-nucleus'

def candidate(img,nuclei,i,kernel,structure):
    s=structure
    i_mask = np.where(img == i, 1, 0).astype(np.uint8)

    connected_mask = cv2.dilate(i_mask, kernel, iterations=2)
    connected_mask = cv2.erode(connected_mask, kernel, iterations=2)

    # Add nucleus as predicted by cellpose
    nucleus_mask = np.where(nuclei == i, 1, 0).astype(np.uint8)

    connected_mask = connected_mask + nucleus_mask
    connected_mask[connected_mask > 0] = 1

    unique_ids, num_ids = ndi.label(connected_mask, structure=s)
    if num_ids > 1:
        # The first element is always 0 (background)
        unique, counts = np.unique(unique_ids, return_counts=True)

        # Ensure counts in descending order
        counts, unique = (list(t) for t in zip(*sorted(zip(counts, unique))))
        counts.reverse()
        unique.reverse()
        counts = np.array(counts)
        unique = np.array(unique)

        no_overlap = False

        # Rarely, the nucleus is not the largest segment
        for i_part in range(1, len(counts)):
            if i_part > 1:
                no_overlap = True  # TODO: Helen, check this!
            largest = unique[np.argmax(counts[i_part:]) + i_part]
            connected_mask = np.where(unique_ids == largest, 1, 0)
            # Break if current largest region overlaps nucleus
            if np.sum(connected_mask * nucleus_mask) > 0.5:
                break

        # Close holes on largest section
        filled_mask = ndi.binary_fill_holes(connected_mask).astype(int)

    else:
        filled_mask = ndi.binary_fill_holes(connected_mask).astype(int)

    return filled_mask > 0

class FullNucleusIndex:
    """One global bbox per ID; query matrices bounded by max_pairs, no distances cached."""
    def __init__(self,nuclei,max_pairs=262144):
        if nuclei.ndim!=2 or nuclei.dtype.kind not in 'ui':raise ValueError('nucleus raster contract')
        h,w=nuclei.shape
        if (h-1)**2+(w-1)**2>=np.iinfo(np.int64).max:raise OverflowError('squared distance exceeds int64')
        if max_pairs<1:raise ValueError('query memory bound')
        self.nuclei=nuclei;self.boxes=ndi.find_objects(nuclei);self.max_pairs=int(max_pairs)
        self.max_observed_pairs=0
    def coordinates(self,cid):
        cid=int(cid)
        if cid<=0 or cid>len(self.boxes) or self.boxes[cid-1] is None:raise ValueError('missing complete nucleus ID '+str(cid))
        box=self.boxes[cid-1];coords=np.argwhere(self.nuclei[box]==cid).astype(np.int64)
        coords+=np.array([box[0].start,box[1].start],np.int64)
        if not len(coords):raise ValueError('empty complete nucleus')
        return coords
    def squared(self,cid,queries):
        q=np.asarray(queries,dtype=np.int64)
        if q.ndim!=2 or q.shape[1]!=2:raise ValueError('query coordinates')
        h,w=self.nuclei.shape
        if len(q) and ((q<0).any() or (q[:,0]>=h).any() or (q[:,1]>=w).any()):raise ValueError('query outside global grid')
        coords=self.coordinates(cid);answer=np.full(len(q),np.iinfo(np.int64).max,np.int64)
        nc=min(len(coords),512,self.max_pairs);nq=max(1,self.max_pairs//nc)
        for a in range(0,len(q),nq):
            result=answer[a:a+nq]
            for b in range(0,len(coords),nc):
                self.max_observed_pairs=max(self.max_observed_pairs,len(q[a:a+nq])*len(coords[b:b+nc]))
                dy=q[a:a+nq,0,None]-coords[None,b:b+nc,0]
                dx=q[a:a+nq,1,None]-coords[None,b:b+nc,1]
                result[:]=np.minimum(result,(dy*dy+dx*dx).min(1))
        return answer

def rng_hash():return hashlib.sha256(pickle.dumps(random.getstate())).hexdigest()

def connect(img,nuclei,rule='B',index=None,origin=(0,0),capture=False):
    if rule not in ('B','P1','P2'):raise ValueError('unknown rule')
    if img.shape!=nuclei.shape or img.ndim!=2:raise ValueError('tile raster contract')
    if rule=='P2' and index is None:raise ValueError('complete global nucleus index required')
    cell_ids=np.unique(img);cell_ids=cell_ids[cell_ids>0]
    if rule=='P2':
        for cid in cell_ids:index.coordinates(cid)  # Missing IDs fail even on unique proposals.
    before=rng_hash();random.shuffle(cell_ids);after=rng_hash()
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5));s=ndi.generate_binary_structure(2,2)
    final=np.zeros(img.shape,np.uint32);original=np.zeros_like(final);count=np.zeros(img.shape,np.uint32)
    hashes={};masks={} if capture else None;veto=[]
    for i in cell_ids:
        mask=candidate(img,nuclei,i,kernel,s)
        hashes[str(int(i))]=hashlib.sha256(mask.tobytes()).hexdigest()
        if capture:masks[int(i)]=mask.copy()
        count+=mask;original[mask]=i
        write=mask
        if rule=='P1':
            rejected=mask&(img!=0)&(img!=i);yy,xx=np.nonzero(rejected)
            if len(yy):veto.append(np.column_stack((yy,xx,img[yy,xx],np.full(len(yy),i),nuclei[yy,xx])).astype(np.int64))
            write=mask&((img==0)|(img==i))
        final[write]=i
    conflict=count>=2;proposals=[];maxint=np.iinfo(np.int64).max
    best=np.full(img.shape,maxint,np.int64);second=np.full_like(best,maxint);second_id=np.zeros_like(final);ties=np.zeros(img.shape,bool)
    if rule=='P2':
        # Re-generating deterministic candidates consumes no RNG; avoids K x tile storage.
        for i in cell_ids:
            mask=candidate(img,nuclei,i,kernel,s)
            if hashlib.sha256(mask.tobytes()).hexdigest()!=hashes[str(int(i))]:raise AssertionError('candidate nondeterminism')
            yy,xx=np.nonzero(mask&conflict)
            if not len(yy):continue
            d=index.squared(int(i),np.column_stack((yy+origin[0],xx+origin[1])))
            old=best[yy,xx];oldid=final[yy,xx]
            wins=(d<old)|((d==old)&(i<oldid))
            # Maintain the exact two smallest (distance, global ID) pairs.
            sd=second[yy,xx];si=second_id[yy,xx]
            sw=(~wins)&((d<sd)|((d==sd)&(i<si)))
            second[yy[wins],xx[wins]]=old[wins];second_id[yy[wins],xx[wins]]=oldid[wins]
            second[yy[sw],xx[sw]]=d[sw];second_id[yy[sw],xx[sw]]=i
            best[yy[wins],xx[wins]]=d[wins];final[yy[wins],xx[wins]]=i
            proposals.append(np.column_stack((yy,xx,np.full(len(yy),i),d)).astype(np.int64))
        ties=conflict&(best==second)
    before_restore=final.copy();final=np.where(nuclei>0,nuclei,final).astype(np.uint32)
    baseline=np.where(nuclei>0,nuclei,original).astype(np.uint32)
    if rule=='P1':
        if np.any((final>0)&(baseline==0)):raise AssertionError('P1 support not subset')
        if np.any((nuclei==0)&(img>0)&(final>0)&(final!=img)):raise AssertionError('P1 owner protection')
    if rule=='P2':
        if not np.array_equal(final>0,baseline>0):raise AssertionError('P2 foreground support')
        if np.any((final!=baseline)&~((nuclei==0)&conflict)):raise AssertionError('P2 changed nonconflict')
    if rng_hash()!=after:raise AssertionError('diagnostics consumed RNG')
    vt=np.concatenate(veto) if veto else np.empty((0,5),np.int64)
    pp=np.concatenate(proposals) if proposals else np.empty((0,4),np.int64)
    ys,xs=np.nonzero(conflict)
    diagnostics=dict(rule=rule,version=RULE_VERSION,candidate_order=cell_ids.tolist(),candidate_hashes=hashes,
        rng_before_shuffle=before,rng_enter_islands=after,proposal_histogram={str(int(k)):int(v) for k,v in zip(*np.unique(count,return_counts=True))},
        conflict_pixels=int(conflict.sum()),changed_vs_B=int((final!=baseline).sum()),exact_tie_pixels=int(ties.sum()),
        veto_proposal_pixels=len(vt),veto_unique_pixels=len(np.unique(vt[:,:2],axis=0)),veto_nuclear_positions=int(np.count_nonzero(vt[:,4])) if len(vt) else 0,
        nuclear_restore_changes=int((before_restore!=final).sum()),no_conflict_equal=bool(np.array_equal(final[~conflict],baseline[~conflict])) if rule=='P2' else None,
        distance_units='squared global pixel coordinates; isotropic native 1 um grid',max_query_pairs=index.max_observed_pairs if index else 0)
    arrays=dict(proposal_count=count,veto_exposures=vt,conflict_proposals=pp,
        conflict_resolution=np.column_stack((ys,xs,before_restore[ys,xs],best[ys,xs],second_id[ys,xs],second[ys,xs],ties[ys,xs],final[ys,xs])).astype(np.int64) if rule=='P2' else np.empty((0,8),np.int64),
        baseline_postconnect=baseline,pre_nuclear_restore=before_restore)
    return final,diagnostics,arrays,masks
