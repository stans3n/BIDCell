"""Bounded ROI pilot driver. Every phase is explicit and leaves evidence.

Run from this checkout with its absolute PYTHONPATH. No legacy code imports.
Original dataset, model, loss, decoder and morphology functions are reused.
"""
import argparse
import contextlib
from functools import lru_cache
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
from types import MethodType

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
import tifffile
import torch

from bidcell.config import load_config
from bidcell.model.dataio.dataset_input import DataProcessing
from bidcell.model.model.model import SegmentationModel
from bidcell.model.model.losses import NucleiEncapsulationLoss, Oversegmentation, CellCallingLoss, OverlapLoss, PosNegMarkerLoss
from bidcell.model.utils.utils import get_seg_mask
from bidcell.model.predict import gap_coords
from bidcell.model.postprocess_predictions import postprocess_connect, remove_islands
from .adapter import features, ResidualAdapter

TOOLS = Path('/home/zwan0726/work/projects/bidcell_remote_prep_v1')
sys.path.insert(0,str(TOOLS))
from matrix import aggregate
from common import sha256, atomic_json, identity

MACHINE = json.loads((TOOLS/'configs/machine.json').read_text())
SEED=20261007
SOURCE=Path(__file__).resolve().parents[2]


def save(path,obj):
    atomic_json(Path(path),obj)


@lru_cache(maxsize=4)
def cache_dependencies(out):
    implementation=inspect.getsource(CacheBuilder.build)+inspect.getsource(features)
    names=['checkpoint','baseline_config','nuclei_mask','preannotation','reference','positive_markers','negative_markers','genes']
    return dict(inputs={k:sha256(MACHINE[k]) for k in names},spec=sha256(out/'spec.json'),cache_implementation=hashlib.sha256(implementation.encode()).hexdigest(),dataset=sha256(SOURCE/'bidcell/model/dataio/dataset_input.py'),backbone=sha256(SOURCE/'bidcell/model/model/model.py'),dtype='float32')


def provenance():
    revision=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()
    sources={str(p.relative_to(SOURCE)):sha256(p) for p in sorted((SOURCE/'bidcell').rglob('*')) if p.suffix in ('.py','.R')}
    source_hash=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()
    diff=subprocess.check_output(['git','-C',str(SOURCE),'diff','--binary','HEAD'])
    dirty_hash=hashlib.sha256(diff+json.dumps(sources,sort_keys=True).encode()).hexdigest()
    return dict(import_path=str(SOURCE/'bidcell'),revision=revision,source_hash=source_hash,dirty_diff_hash=dirty_hash,dirty_hash_definition='git binary diff HEAD plus content hashes of all Python/R source including untracked method files',source_files=sources,seed=SEED,checkpoint=identity(MACHINE['checkpoint']),baseline_config=identity(MACHINE['baseline_config']),parent_baseline_id='bidcell_xenium_breast_official_step4000_20260723_013920',historical_execution_provenance='see inherited remote_preparation_20261007 audit; source snapshot is not asserted to be execution HEAD')


def run_identity(out,method):
    spec=json.loads((out/'spec.json').read_text())
    hashobj=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
    return dict(logical_method=method,run_id=out.name+'_'+method+'_pilot',method_config_hash=hashobj(dict(method=method,optimizer=spec['optimizer'],calibration=json.loads((out/'calibration.json').read_text()))),data_manifest_hash=hashobj(spec['patches']),split_hash=hashobj(spec['blocks']),spec_hash=sha256(out/'spec.json'),checkpoint_hash=sha256(MACHINE['checkpoint']),parent_baseline_id='bidcell_xenium_breast_official_step4000_20260723_013920',seed=SEED,method_spec_version=spec['version'],evaluation_protocol='MA-EVAL-1.0-draft+pilot-ROI-engineering-1.0')


def setup(out):
    torch.set_num_threads(4)
    np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    cfg=load_config(MACHINE['baseline_config'])
    ds=DataProcessing(cfg,isTraining=False,shift_patches=0)
    paths={}
    for shift in (0,24):
        folder=Path(ds.expr_fp.replace('_shift_0',f'_shift_{shift}'))
        paths[shift]=sorted(folder.glob('*.hdf5'),key=lambda p:tuple(map(int,re.findall(r'\d+',p.stem))))
    ds.fp_patches=[str(p) for s in (0,24) for p in paths[s]]
    meta=[]
    for s in (0,24):
        for p in paths[s]:
            y,x=map(int,re.findall(r'\d+',p.stem)[:2]);meta.append(dict(path=str(p),shift=s,y=y,x=x))
    return cfg,ds,meta


def freeze(out):
    cfg,ds,meta=setup(out)
    nuclei=ds.nuclei; objects=ndi.find_objects(nuclei)
    candidates=[]
    for y in range(48,nuclei.shape[0]-576,768):
        for x in range(48,nuclei.shape[1]-576,768):
            ids=np.unique(nuclei[y+96:y+480,x+96:x+480]);ids=ids[ids>0]
            if len(ids)<20:continue
            # Any nucleus visible in core must have complete possible native
            # patch+closing support within context, before looking at logits.
            if any(objects[i-1][0].start-56<y or objects[i-1][1].start-56<x or objects[i-1][0].stop+56>y+576 or objects[i-1][1].stop+56>x+576 for i in ids):continue
            candidates.append(dict(y=y,x=x,nuclei_count=int(len(ids))))
    candidates.sort(key=lambda b:(b['nuclei_count'],b['y'],b['x']))
    if len(candidates)<12:raise RuntimeError('insufficient legal blocks; no silent halo shrink')
    chosen=[candidates[i] for i in np.linspace(0,len(candidates)-1,12).round().astype(int)]
    blocks=[];train_ids=set();dev_ids=set()
    for i,b in enumerate(chosen):
        b=dict(b,id=f'b{i:02d}',split='dev' if i%3==2 else 'train',context_size=576,core_offset=96,core_size=384)
        y,x=b['y'],b['x']
        indexes=[k for k,m in enumerate(meta) if m['y']<y+576 and m['y']+48>y and m['x']<x+576 and m['x']+48>x]
        ids=set()
        for k in indexes:
            m=meta[k];ids.update(np.unique(nuclei[m['y']:m['y']+48,m['x']:m['x']+48]).tolist())
        ids.discard(0)
        b['patch_indexes']=indexes;b['candidate_ids']=sorted(ids)
        target=train_ids if b['split']=='train' else dev_ids;target.update(ids)
        b['eligible_ids']=[int(i) for i in sorted(ids) if objects[i-1][0].start-56>=y+96 and objects[i-1][1].start-56>=x+96 and objects[i-1][0].stop+56<=y+480 and objects[i-1][1].stop+56<=x+480]
        blocks.append(b)
    if train_ids & dev_ids:raise RuntimeError('cross-split candidate cell contamination')
    # Original native 48-pixel patch inputs, fixed random enhancement schedule.
    train=[]
    for b in blocks:
        if b['split']!='train':continue
        for k in b['patch_indexes']:
            m=meta[k]
            if m['shift']==0 and m['y']>=b['y']+96 and m['y']+48<=b['y']+480 and m['x']>=b['x']+96 and m['x']+48<=b['x']+480 and np.any(nuclei[m['y']:m['y']+48,m['x']:m['x']+48]):train.append(k)
    rng=np.random.default_rng(SEED)
    schedule=[dict(patch_index=int(rng.choice(train)),flip=int(rng.integers(2)),rot90_clockwise=int(rng.integers(4))) for _ in range(500)]
    spec=dict(version='MA-A-PILOT-1.0',seed=SEED,blocks=blocks,patches=meta,schedule=schedule,geometry_pixel_um=1.,density='9x9 mean transcript counts; reflect within native48 patch; computed after actual transformed input',normalization='fixed ell clip12/4; log1p(count)/log17; log1p(density); d/r /4; radius/10um; distance/40um; no dev fit',split='576 contexts on768 grid:192um gaps;384 cores with96halo; complete candidate IDs disjoint between train/dev',eligibility='full global nuclear bbox expanded56um inside384 core;56=48 native possible patch support+8 morphology; fixed all methods',candidate_semantics='native48 patch nucleus candidates and SearchArea, all retained; native decoder not support gated; residual gate support AND count>=2 AND outside all nuclei',postprocessing='native functions in576 ROI frame, common candidate-universe seeded priority; same-scope native E00; not equivalent to historical full-field1024 chunk phase',matrix='native validated global coordinate cache; capture all filtered core transcripts; eligible matrix contains only fixed complete IDs, zero cells included; excluded-cell assignments separately conserved',optimizer=dict(name='Adam',lr=3e-4,betas=[.9,.999],weight_decay=0.,updates=500),kappa=1.,regularization='train-only acceptance picks one common lambda from .01,.1; no dev selection',guardrails=dict(capture_drop_percentage_points=.5,foreground_relative_change=.05,median_eligible_cell_area_relative_change=.05,disconnected_increase_percentage_points=.1),guardrail_domain='capture/foreground384 cores; area and pre/post8-connected fragmentation fixed eligible IDs; pooled and per block; screen only',primary='official Purity F1 missing_definition; no primary ranking; engineering profiles separately',exposure='held-out adapter dev; frozen baseline and preannotation historically full Rep1 exposed',tolerance=dict(scores_atol=1e-6,scores_rtol=1e-6,labels='exact canonical IDs',matrix='exact integers'),provenance=provenance())
    save(out/'spec.json',spec)
    save(out/'frozen.json',dict(spec_hash=sha256(out/'spec.json'),frozen_before_method_results=True,train_patch_count=len(train),train_candidate_ids=len(train_ids),dev_candidate_ids=len(dev_ids)))
    print('FROZEN',[(b['id'],b['split'],b['nuclei_count'],len(b['eligible_ids'])) for b in blocks],flush=True)


def refreeze(out):
    """Pre-outcome closure repair: preserve original cores and training views."""
    previous=out.parent/'spec.json';spec=json.loads(previous.read_text())
    _,ds,meta=setup(out);nuclei=ds.nuclei;objects=ndi.find_objects(nuclei)
    train_ids=set();dev_ids=set();proof=[]
    for b in spec['blocks']:
        b['y']-=48;b['x']-=48;b['context_size']=672;b['core_offset']=144
        y,x=b['y'],b['x'];size=b['context_size']
        if y<0 or x<0 or y+size>nuclei.shape[0] or x+size>nuclei.shape[1]:raise RuntimeError('expanded context outside scene')
        indexes=[k for k,m in enumerate(meta) if m['y']<y+size and m['y']+48>y and m['x']<x+size and m['x']+48>x]
        ids=set()
        for k in indexes:
            m=meta[k];ids.update(np.unique(nuclei[m['y']:m['y']+48,m['x']:m['x']+48]).tolist())
        ids.discard(0);b['patch_indexes']=indexes;b['candidate_ids']=sorted(ids)
        (train_ids if b['split']=='train' else dev_ids).update(ids)
        margins=[]
        # Scan every global nucleus, including those not selected as patch
        # candidates, to demonstrate no unobserved possible core influence.
        for cid,sl in enumerate(objects,1):
            if sl is None:continue
            lo=np.array([sl[0].start,sl[1].start])-56;hi=np.array([sl[0].stop,sl[1].stop])+56;origin=np.array([y,x])
            if np.all(lo<origin+528) and np.all(hi>origin+144):
                if not (np.all(lo>=origin) and np.all(hi<=origin+size)):raise RuntimeError('expanded full global candidate closure still fails')
                margins.append(int(min((lo-origin).min(),(origin+size-hi).min())))
        proof.append(dict(block=b['id'],relevant_global_candidates=len(margins),minimum_context_margin=min(margins)))
    if train_ids & dev_ids:raise RuntimeError('expanded cross-split candidate contamination')
    spec.update(version='MA-A-PILOT-1.1',split='same frozen384core centers and500train views;672 contexts with144halo,768grid leaves96um gaps; full candidate IDs train/dev disjoint',postprocessing='native functions in672 ROI frame; common candidate-universe seeded priority; same-scope E00 not historical full-field1024 phase',provenance=provenance(),repair=dict(reason='original96halo failed full competitor bbox+56 closure;1482 violations, max34pixels outsidecontext; uniform48eachside repair preserves all core locations/density ranks/eligibility/training sequence',prior_spec=identity(previous),prior_diagnosis=identity(out.parent/'geometry_closure_diagnosis.json'),viewed_method_results=False))
    save(out/'spec.json',spec);save(out/'frozen.json',dict(spec_hash=sha256(out/'spec.json'),frozen_before_method_results=True,training_sequence_unchanged=True,core_coordinates_unchanged=True,complete_cell_eligibility_unchanged=True,train_candidate_ids=len(train_ids),dev_candidate_ids=len(dev_ids),closure_proof=proof))
    print('REFROZEN',proof,flush=True)


class CacheBuilder:
    def __init__(self,out):
        self.out=out;self.spec=json.loads((out/'spec.json').read_text())
        self.cfg,self.ds,self.meta=setup(out)
        self.objects=ndi.find_objects(self.ds.nuclei)
        self.areas=np.bincount(self.ds.nuclei.ravel())
        self.trees={}
        self.model=SegmentationModel(n_channels=len(self.ds.gene_names)).cuda()
        checkpoint=torch.load(MACHINE['checkpoint'],map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        self.model.requires_grad_(False);self.model.eval()
        self.state_before={k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()}
        self.dependency=cache_dependencies(out)
        (out/'cache').mkdir(exist_ok=True)

    def build(self,index,flip=0,rot=0):
        key=f'p{index:06d}_f{flip}_r{rot}'
        path=self.out/'cache'/f'{key}.pt'
        if path.exists():
            load_view(self.out,dict(patch_index=index,flip=flip,rot90_clockwise=rot))
            return key
        m=self.meta[index]
        self.ds.isTraining=True
        def transform(ds,img):
            a=np.flip(img,axis=0) if flip else img
            return np.rot90(a,-rot,axes=(0,1))[None].copy(),flip,rot
        self.ds.augment_data=MethodType(transform,self.ds)
        sample=self.ds[index]
        expr,n,sa,pos,neg,y,x,nuc,density=sample
        ids=np.unique(nuc);ids=ids[ids>0].astype(np.int64)
        if len(ids)==0:
            z=torch.zeros((0,2,48,48));absolute=torch.zeros((0,48,48,14));relative=absolute.clone();gate=torch.zeros((0,48,48),dtype=torch.bool);adv=torch.zeros((0,48,48));distances=np.zeros((0,48,48),np.float32)
        else:
            xx=expr.permute(3,2,0,1).cuda()
            with torch.no_grad():z=self.model(xx).detach().cpu()
            yy,xx=np.mgrid[y:y+48,x:x+48];query=np.stack((yy.ravel(),xx.ravel()),1)
            dd=[]
            for i in ids:
                if i not in self.trees:
                    sl=self.objects[i-1];coords=np.argwhere(self.ds.nuclei[sl]==i)+[sl[0].start,sl[1].start];self.trees[i]=cKDTree(coords)
                d=self.trees[i].query(query,k=1)[0].reshape(48,48)
                if flip:d=np.flip(d,axis=0)
                dd.append(np.rot90(d,-rot).copy())
            distances=np.asarray(dd,np.float32)
            rad=torch.from_numpy(np.sqrt(self.areas[ids]/np.pi).astype(np.float32))
            sm=sa.permute(2,0,1).bool();nm=n.permute(2,0,1).bool()
            den=torch.from_numpy(ndi.uniform_filter(np.asarray(density,dtype=np.float32),size=9,mode='reflect'))
            absolute,relative,gate,adv=features(z,sm,nm,torch.from_numpy(distances),rad,den,ids)
        targets={name:t.permute(2,0,1)[:,None].long() for name,t in [('n',n),('sa',sa),('pos',pos),('neg',neg)]}
        payload=dict(dependency=self.dependency,source_hdf5=identity(m['path']),key=key,patch_index=index,view=dict(flip=flip,rot90_clockwise=rot),ids=ids,z=z,absolute=absolute,relative=relative,gate=gate,advantage=adv,nuclei=np.asarray(nuc,np.uint32),expression_sum=np.asarray(density,np.float32),distances=distances,targets=targets,coords=[y,x],shift=m['shift'],input_hash=hashlib.sha256(expr.numpy().tobytes()).hexdigest())
        torch.save(payload,path)
        path.with_suffix('.sha256').write_text(sha256(path)+'\n')
        print('CACHE',key,len(ids),int(gate.sum()),flush=True)
        return key

    def verify_frozen(self):
        equal=all(torch.equal(v.cpu(),self.state_before[k]) for k,v in self.model.state_dict().items())
        if not equal:raise RuntimeError('backbone parameter or BN mutated')
        save(self.out/'backbone_frozen.json',dict(strict_load=True,all_parameters_require_grad_false=all(not p.requires_grad for p in self.model.parameters()),eval_mode=not self.model.training,state_tensors_bitwise_unchanged=True,state_tensor_count=len(self.state_before),gpu_peak_allocated=torch.cuda.max_memory_allocated(),gpu_peak_reserved=torch.cuda.max_memory_reserved()))


def cache(out,probe=False):
    b=CacheBuilder(out);spec=b.spec
    seq=spec['schedule'][:50] if probe else spec['schedule']
    for v in seq:b.build(v['patch_index'],v['flip'],v['rot90_clockwise'])
    blocks=[x for x in spec['blocks'] if x['split']=='dev']
    if probe:blocks=blocks[:1]
    for block in blocks:
        for index in block['patch_indexes']:b.build(index)
    b.verify_frozen()
    save(out/('probe_cache_complete.json' if probe else 'cache_complete.json'),dict(spec_hash=sha256(out/'spec.json'),train_views=len(seq),dev_blocks=len(blocks),dependencies=b.dependency))


def load_view(out,v):
    path=out/'cache'/f"p{v['patch_index']:06d}_f{v.get('flip',0)}_r{v.get('rot90_clockwise',0)}.pt"
    if sha256(path)!=path.with_suffix('.sha256').read_text().strip():raise RuntimeError('cache content hash mismatch')
    p=torch.load(path,map_location='cpu')
    if p['dependency']!=cache_dependencies(out):raise RuntimeError('stale cache dependencies')
    if sha256(p['source_hdf5']['path'])!=p['source_hdf5']['sha256']:raise RuntimeError('expression HDF5 changed')
    return p


def apply(model,p,device='cuda'):
    z=p['z'].to(device)
    return model(z,*[p[k].to(device) for k in ('absolute','relative','gate','advantage')])


def losses(z,targets):
    t={k:v.to(z.device) for k,v in targets.items()};d=z.device
    # Original module calls preserve formulas/weights/reductions exactly.
    ne=NucleiEncapsulationLoss(1.,d)(z,t['n'])
    os_=Oversegmentation(1.,d)(z,t['n'])
    cc=CellCallingLoss(1.,d)(z,t['sa'])
    ov=OverlapLoss(1.,d)(z,t['n'])
    pn=PosNegMarkerLoss(1.,1.,d)(z,t['pos'],t['neg'])
    # Isolate contributions using original module with the other weight zero.
    pos=PosNegMarkerLoss(1.,0.,d)(z,t['pos'],t['neg'])
    neg=PosNegMarkerLoss(0.,1.,d)(z,t['pos'],t['neg'])
    return dict(ne=ne,os=os_,cc=cc,ov=ov,pos=pos,neg=neg),ne+os_+cc+ov+pn


@contextlib.contextmanager
def stable_priority(universe,seed):
    ids=list(sorted(universe));random.Random(seed).shuffle(ids);rank={c:i for i,c in enumerate(ids)}
    original=random.shuffle
    def stable(a):
        a[:]=sorted(a,key=lambda c:rank[int(c)])
    random.shuffle=stable
    try:yield
    finally:random.shuffle=original


def merge_and_connect(out,spec,block,model,destination,save_scores=True):
    begin=time.monotonic();forward_decode_seconds=0.;score_write_seconds=0.;cache_read_seconds=0.
    y,x=block['y'],block['x'];size=block['context_size']
    masks={s:np.zeros((size,size),np.uint32) for s in (0,24)}
    nucl=np.zeros((size,size),np.uint32)
    score_dir=destination/'scores';score_dir.mkdir(parents=True,exist_ok=True)
    for index in block['patch_indexes']:
        t=time.monotonic();p=load_view(out,dict(patch_index=index));py,px=p['coords'];cache_read_seconds+=time.monotonic()-t
        t=time.monotonic()
        if len(p['ids']):
            with torch.no_grad():corrected,delta=apply(model,p)
            z=corrected.cpu().numpy();seg=get_seg_mask(z,p['nuclei'])
        else:z=p['z'].numpy();delta=p['gate'].float();seg=np.zeros((48,48),np.uint32)
        forward_decode_seconds+=time.monotonic()-t
        a,b=max(y,py),max(x,px);c,d=min(y+size,py+48),min(x+size,px+48)
        masks[p['shift']][a-y:c-y,b-x:d-x]=seg[a-py:c-py,b-px:d-px]
        nucl[a-y:c-y,b-x:d-x]=p['nuclei'][a-py:c-py,b-px:d-px]
        if save_scores:
            t=time.monotonic();np.savez_compressed(score_dir/f'{index:06d}.npz',ids=p['ids'],baseline=p['z'].numpy(),corrected=z,support=p['targets']['sa'][:,0].numpy(),gate=p['gate'].numpy(),delta=delta.cpu().numpy());score_write_seconds+=time.monotonic()-t
    # Reproduce native GLOBAL row/column fill mask, then crop; gap_coords sees
    # the same natsorted list as baseline, including its original duplicates.
    t=time.monotonic();sf=[m for m in spec['patches'] if m['shift']==24]
    hh=[m['y'] for m in sf];ww=[m['x'] for m in sf]
    hs=set(v+i for v in hh for i in range(int(48*.35),int(48*.65)))
    ws=set(v+i for v in ww for i in range(int(48*.35),int(48*.65)))
    fill=np.isin(np.arange(y,y+size),list(hs))[:,None]|np.isin(np.arange(x,x+size),list(ws))[None,:]
    fill[np.isin(np.arange(y,y+size),gap_coords(hh,48)),:]=False
    fill[:,np.isin(np.arange(x,x+size),gap_coords(ww,48))]=False
    pre=np.where(fill,masks[24],masks[0]).astype(np.uint32)
    merge_seconds=time.monotonic()-t;t=time.monotonic()
    with stable_priority(block['candidate_ids'],SEED+int(block['id'][1:])):
        final=remove_islands(postprocess_connect(pre,nucl),nucl)
    final=np.where(nucl>0,nucl,final).astype(np.uint32)
    connectivity_seconds=time.monotonic()-t
    destination.mkdir(parents=True,exist_ok=True)
    for name,a in [('shift0',masks[0]),('shift24',masks[24]),('pre_connect',pre),('final',final),('nuclei',nucl)]:tifffile.imwrite(destination/f'{name}.tif',a,photometric='minisblack')
    save(destination/'output_timing.json',dict(total_seconds=time.monotonic()-begin,cache_read_seconds=cache_read_seconds,forward_decode_seconds=forward_decode_seconds,score_write_seconds=score_write_seconds,merge_seconds=merge_seconds,connectivity_seconds=connectivity_seconds,native_patches=len(block['patch_indexes']),candidate_ids=len(block['candidate_ids']),context_pixels=size*size,accounting='nested within export/acceptance command; do not add to command wall time'))
    return pre,final,nucl


def roi_transcripts(out,spec):
    cache_dir=Path('/home/zwan0726/work/cache/bidcell_remote_prep_v1/3d552a807c41ebad532e6871ee311ee0060932a13578170aff90af6211c3ed01')
    cm=json.loads((cache_dir/'complete.json').read_text())
    expected=dict(transcripts=sha256(MACHINE['transcripts']),genes=sha256(MACHINE['genes']),shape=[5478,7526],scale_x=MACHINE['scale_x'],scale_y=MACHINE['scale_y'],builder=sha256(TOOLS/'matrix.py'))
    if cm['dependencies']!=expected:raise RuntimeError('coordinate cache contract dependencies changed')
    blocks=[b for b in spec['blocks'] if b['split']=='dev']
    parts={b['id']:[] for b in blocks}
    for receipt in cm['chunks']:
        path=cache_dir/receipt['file']
        if sha256(path)!=receipt['sha256']:raise RuntimeError('coordinate cache hash mismatch')
        a=np.load(path,mmap_mode='r')
        for b in blocks:
            offset=b['core_offset'];end=offset+b['core_size']
            sel=(a['y']>=b['y']+offset)&(a['y']<b['y']+end)&(a['x']>=b['x']+offset)&(a['x']<b['x']+end)
            parts[b['id']].append(a[sel].copy())
    for b in blocks:np.save(out/f"transcripts_{b['id']}.npy",np.concatenate(parts[b['id']]))
    save(out/'roi_coordinate_contract.json',dict(source_manifest=identity(cache_dir/'complete.json'),spec=sha256(out/'spec.json'),coordinate_contract='reused audited global native lookup; no ROI-local rescale',owners_reused=False))


def matrix_metrics(out,spec,block,pre,final,nucl,dest):
    begin=time.monotonic()
    a=np.load(out/f"transcripts_{block['id']}.npy")
    yy=a['y']-block['y'];xx=a['x']-block['x'];owner=final[yy,xx]
    ids=np.asarray(block['eligible_ids'],np.int64);ng=len(Path(MACHINE['genes']).read_text().splitlines())
    eligible_owner=np.where(np.isin(owner,ids),owner,0).astype(np.uint32)
    keys,n=aggregate(eligible_owner,a['gene'],ids,ng);counts=np.zeros((len(ids),ng),np.int64);counts.ravel()[keys]+=n
    assigned=int((owner>0).sum());inc=int((eligible_owner>0).sum());excluded=assigned-inc
    if int(counts.sum())!=inc or inc+excluded+int((owner==0).sum())!=len(a):raise RuntimeError('ROI count conservation failed')
    np.save(dest/'counts.npy',counts)
    np.savez_compressed(dest/'assignments.npz',transcript_id=a['transcript_id'],owner=owner,matrix_owner=eligible_owner,gene=a['gene'])
    genes=Path(MACHINE['genes']).read_text().splitlines()
    table=pd.DataFrame(counts,columns=genes);table.insert(0,'cell_id',ids);table.to_csv(dest/'expr_mat.csv')
    pd.DataFrame(dict(matrix_row=np.arange(len(ids)),canonical_cell_id=ids)).to_csv(dest/'cell_mapping.csv',index=False)
    start=block['core_offset'];stop=start+block['core_size'];core=final[start:stop,start:stop];areas=np.array([(final==i).sum() for i in ids])
    structure=ndi.generate_binary_structure(2,2)
    frag=lambda mask:sum(ndi.label(mask==i,structure)[1]>1 for i in ids)/max(1,len(ids))
    met=dict(block=block['id'],eligible_transcripts=len(a),assigned_transcripts=assigned,matrix_assigned=inc,excluded_cell_assigned=excluded,unassigned=int((owner==0).sum()),capture=assigned/max(1,len(a)),foreground=float((core>0).mean()),eligible_cells=len(ids),median_cell_area=float(np.median(areas)) if len(ids) else None,disconnected_pre=frag(pre),disconnected_final=frag(final),pre_post_changed_fraction=float((pre[start:stop,start:stop]!=core).mean()),purity_f1=None,purity_status='missing_definition')
    save(dest/'metrics.json',met);save(dest/'matrix_contract.json',dict(mask=identity(dest/'final.tif'),coordinates=identity(out/f"transcripts_{block['id']}.npy"),spec=sha256(out/'spec.json'),counts_sum=int(counts.sum()),conservation=True,raw_denominators=met))
    save(dest/'matrix_timing.json',dict(seconds=time.monotonic()-begin,core_transcripts=len(a),eligible_cells=len(ids),accounting='nested within export/acceptance command; includes structural metrics'))
    return met


def acceptance(out):
    spec=json.loads((out/'spec.json').read_text());rows=[];scale=[];es=[];updates=0;trial_scales=[]
    torch.manual_seed(SEED);model=ResidualAdapter('E05').cuda()
    opt=torch.optim.Adam(model.parameters(),lr=3e-4)
    for k,v in enumerate(spec['schedule'][:50]):
        p=load_view(out,v)
        if not p['gate'].any():continue
        if updates==0:
            for logical in ('E02','E04','E05'):
                fresh=ResidualAdapter(logical).cuda()
                zero,_=apply(fresh,p)
                if not torch.equal(zero.cpu(),p['z']):raise RuntimeError('zero initialized learned head differs from baseline')
        z,delta=apply(model,p);terms,total=losses(z,p['targets'])
        if k<10:
            for name,term in terms.items():
                grad=torch.autograd.grad(term,z,retain_graph=True)[0]
                mg=torch.autograd.grad(term,list(model.parameters()),retain_graph=True,allow_unused=True)
                gn=float(grad[:,1][p['gate'].cuda()].norm())
                rows.append(dict(view=k,loss=name,value=float(term),logit_gradient_norm=float(grad.norm()),G_foreground_gradient_norm=gn,adapter_gradient_norm=float(torch.sqrt(sum((g.square().sum() for g in mg if g is not None)))),status='usable_G_gradient' if gn>0 else 'zero_on_this_view',zero_reason='' if gn>0 else ('no_negative_marker_on_modifiable_support' if name=='neg' else 'native_loss_inactive_on_this_view')))
        if not torch.isfinite(total):raise RuntimeError('nonfinite original loss')
        scale.extend((p['absolute'][...,4]*4)[p['gate']].tolist());es.extend((p['z'][:,1]-p['z'][:,0])[p['gate']].abs().tolist())
        if len(trial_scales)<10:
            trial=p['z'].cuda().clone().requires_grad_(True)
            trial_z=torch.stack((trial[:,0],trial[:,1]+.1*p['gate'].cuda()),1)
            _,trial_base=losses(trial_z,p['targets'])
            bg=torch.autograd.grad(trial_base,trial)[0][:,1][p['gate'].cuda()]
            trial_scales.append(dict(view=k,N_G=int(p['gate'].sum()),candidate_count=len(p['ids']),trial_delta=.1,base_loss=float(trial_base),base_G_gradient_norm=float(bg.norm()),regularizer_G_gradient_norm_lambda_point1=float(.02/np.sqrt(int(p['gate'].sum()))),lambda_for_ten_percent_gradient=float(.5*bg.norm()*np.sqrt(int(p['gate'].sum())))))
        if updates==0:
            before={n:v.detach().clone() for n,v in model.state_dict().items()}
            opt.zero_grad();total.backward();opt.step()
            if not any(not torch.equal(before[n],v) for n,v in model.state_dict().items()):raise RuntimeError('adapter did not update')
            updates+=1
    if not rows or not any(r['G_foreground_gradient_norm']>0 for r in rows):raise RuntimeError('no modifiable-region gradient')
    if not all(np.isfinite(r['adapter_gradient_norm']) and np.isfinite(r['logit_gradient_norm']) and np.isfinite(r['value']) for r in rows):raise RuntimeError('invalid loss gradients')
    if not any(r['adapter_gradient_norm']>0 for r in rows):raise RuntimeError('no adapter gradient')
    beta=float(np.clip(np.median(es)/max(np.median(scale),.1),.25,1.))
    lam=float(np.median([r['lambda_for_ten_percent_gradient'] for r in trial_scales]))
    if not np.isfinite(lam) or lam<=0:raise RuntimeError('invalid train-only regularizer scale')
    frozen=dict(lambda_residual=lam,lambda_reason='before any pilot: median first10 nonemptyG train views .5*||dLnative/dfg onG atdelta+.1||2*sqrt(NG); reg gradient10percent native atdelta.1 (approximately native atbound1); same lambda all learned; native reductions unchanged',supersedes_draft_lambda=.1,regularizer_trial_scales=trial_scales,distance_beta=beta,distance_scale=float(beta*np.median(scale)),distance_rule='clip(-beta*dtilde_self+b,-1,1), b=beta*train median dtilde_self; one train-only configuration',train_views=50,source=sha256(out/'spec.json'),calibration_version='train-scale-overlay-1.0')
    save(out/'calibration.json',frozen);pd.DataFrame(rows).to_csv(out/'loss_gradients.csv',index=False)
    # Full real ROI E00/E01 native decoding, shifts, morphology, matrix.
    if not (out/'roi_coordinate_contract.json').exists():roi_transcripts(out,spec)
    b=next(b for b in spec['blocks'] if b['split']=='dev')
    artifacts=[]
    for method in ('E00','E01'):
        dest=out/'acceptance'/method/b['id'];m=ResidualAdapter(method).cuda()
        pre,final,nuc=merge_and_connect(out,spec,b,m,dest)
        met=matrix_metrics(out,spec,b,pre,final,nuc,dest)
        artifacts.append((pre,final,np.load(dest/'counts.npy')))
    for a,b_ in zip(artifacts[0],artifacts[1]):
        if not np.array_equal(a,b_):raise RuntimeError('zero-residual canonical mask/matrix mismatch')
    # Functional tests cover empty/single/multi candidate support, ties and
    # candidate reorder using actual geometry and nonzero learned weights.
    p=next(load_view(out,v) for v in spec['schedule'][:50] if load_view(out,v)['gate'].any())
    z,delta=apply(model,p)
    if torch.count_nonzero(delta[~p['gate'].cuda()]):raise RuntimeError('G scope leakage')
    if not torch.equal(z[:,0].cpu(),p['z'][:,0]):raise RuntimeError('background changed')
    for c in (0,1,2,4):
        zz=torch.zeros(c,2,3,3);ss=torch.ones(c,3,3,dtype=torch.bool);nn=torch.zeros_like(ss)
        dd=torch.ones(c,3,3);rr=torch.ones(c);den=torch.zeros(3,3);ids=torch.arange(c)+10
        ab,re,g,ad=features(zz,ss,nn,dd,rr,den,ids)
        if c<2 and g.any():raise RuntimeError('empty/single gate')
        perm=torch.arange(c-1,-1,-1)
        ap,rp,gp,adv=features(zz[perm],ss[perm],nn[perm],dd[perm],rr[perm],den,ids[perm])
        if not (torch.equal(ab[perm],ap) and torch.equal(re[perm],rp) and torch.equal(g[perm],gp)):raise RuntimeError('candidate order dependence')
    save(out/'acceptance.json',dict(status='PASS',scope=f"single dev{b['context_size']} ROI with{b['core_size']} core only",score_atol=1e-6,score_rtol=1e-6,zero_score_exact=True,pre_final_labels_exact=True,integer_matrix_exact=True,scope_and_permutation=True,adapter_updated=True,loss_gradient_rows=len(rows),backbone_evidence=str(out/'backbone_frozen.json'),calibration=identity(out/'calibration.json'),diagnostic_optimizer_updates=1,diagnostic_updates_reinitialized_for_pilots=True))
    print('ACCEPTANCE PASS',flush=True)


def train(out,method):
    spec=json.loads((out/'spec.json').read_text());cal=json.loads((out/'calibration.json').read_text())
    if json.loads((out/'acceptance.json').read_text())['status']!='PASS' or json.loads((out/'acceptance_geometry.json').read_text())['status']!='PASS':raise RuntimeError('acceptance gate')
    dest=out/method;dest.mkdir(exist_ok=True)
    if (dest/'checkpoint.pt').exists():raise RuntimeError('completed checkpoint protected')
    torch.manual_seed(SEED);model=ResidualAdapter(method,cal['distance_beta'],cal['distance_scale']).cuda()
    opt=torch.optim.Adam(model.parameters(),lr=3e-4,betas=(.9,.999),weight_decay=0.)
    log=[];t=time.monotonic()
    if method!='E03':
        for step,v in enumerate(spec['schedule'],1):
            p=load_view(out,v)
            if len(p['ids'])==0:raise RuntimeError('empty patch in frozen actual-update schedule')
            z,delta=apply(model,p);terms,base=losses(z,p['targets'])
            reg=delta.square().sum()/p['gate'].sum().clamp_min(1).cuda()
            total=base+cal['lambda_residual']*reg
            opt.zero_grad();total.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'))
            if not torch.isfinite(total) or not torch.isfinite(norm):raise RuntimeError('nonfinite training')
            opt.step()
            log.append(dict(update=step,loss=float(total),base_loss=float(base),regularization=float(reg),gradient_norm=float(norm),residual_max=float(delta.abs().max()),saturation_fraction=float((delta[p['gate'].cuda()].abs()>.99).float().mean()) if p['gate'].any() else 0.,seconds=time.monotonic()-t,key=p['key']))
            if step%50==0:
                pd.DataFrame(log).to_csv(dest/'training.csv',index=False)
                print('TRAIN',method,step,log[-1],flush=True)
                torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),actual_updates=step),dest/'progress.pt')
    torch.save(dict(model=model.state_dict(),method=method,actual_updates=len(log),calibration=cal,spec_hash=sha256(out/'spec.json')),dest/'checkpoint.pt')
    save(dest/'train_complete.json',dict(method=method,actual_updates=len(log),seed=SEED,seconds=time.monotonic()-t,parameters=sum(p.numel() for p in model.parameters()) if method!='E03' else 0,effectively_connected_parameters=1249 if method=='E02' else (0 if method=='E03' else 1569),gpu_peak_allocated=torch.cuda.max_memory_allocated(),gpu_peak_reserved=torch.cuda.max_memory_reserved(),provenance=provenance(),run_identity=run_identity(out,method),spec=identity(out/'spec.json'),configuration=identity(out/'calibration.json')))


def export(out,method):
    spec=json.loads((out/'spec.json').read_text());cal=json.loads((out/'calibration.json').read_text())
    model=ResidualAdapter(method,cal['distance_beta'],cal['distance_scale']).cuda()
    if method not in ('E00','E01'):
        cp=torch.load(out/method/'checkpoint.pt',map_location='cpu');model.load_state_dict(cp['model'],strict=True)
    rows=[]
    for b in spec['blocks']:
        if b['split']!='dev':continue
        dest=out/method/b['id']
        pre,final,nuc=merge_and_connect(out,spec,b,model,dest)
        met=matrix_metrics(out,spec,b,pre,final,nuc,dest)
        if method not in ('E00','E01'):
            base=tifffile.imread(out/'E00'/b['id']/'final.tif')
            start=b['core_offset'];stop=start+b['core_size']
            met['changed_final_fraction']=float((base[start:stop,start:stop]!=final[start:stop,start:stop]).mean())
            bm=json.loads((out/'E00'/b['id']/'metrics.json').read_text())
            met['capture_delta_pp']=100*(met['capture']-bm['capture'])
            met['foreground_relative_delta']=met['foreground']/bm['foreground']-1
            met['median_area_relative_delta']=met['median_cell_area']/bm['median_cell_area']-1
            met['disconnected_delta_pp']=100*(met['disconnected_final']-bm['disconnected_final'])
            met['screen_pass']=met['capture_delta_pp']>=-.5 and abs(met['foreground_relative_delta'])<=.05 and abs(met['median_area_relative_delta'])<=.05 and met['disconnected_delta_pp']<=.1
            save(dest/'metrics.json',met)
        rows.append(met)
    pd.DataFrame(rows).to_csv(out/method/'metrics.csv',index=False)
    save(out/method/'export_complete.json',dict(status='succeeded',method=method,blocks=len(rows),run_identity=run_identity(out,method),spec_hash=sha256(out/'spec.json'),purity_status='missing_definition',scope='held-out adapter ROI engineering pilot',provenance=provenance()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--stage',choices=['freeze','refreeze','probe','cache','geometry','acceptance','train','export','evaluate'],required=True);parser.add_argument('--method',choices=['E00','E01','E02','E03','E04','E05']);a=parser.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    t=time.monotonic()
    if a.stage=='freeze':freeze(a.out)
    elif a.stage=='refreeze':refreeze(a.out)
    elif a.stage in ('probe','cache'):cache(a.out,a.stage=='probe')
    elif a.stage=='acceptance':acceptance(a.out)
    elif a.stage=='geometry':
        from .checks import run
        run(a.out)
    elif a.stage=='evaluate':
        from .evaluation import run
        run(a.out,a.method)
    elif a.stage=='train':train(a.out,a.method)
    else:export(a.out,a.method)
    print('PHASE COMPLETE',a.stage,a.method,time.monotonic()-t,flush=True)


if __name__=='__main__':main()
