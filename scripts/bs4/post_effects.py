"""Track both P and native B owners through canonical downstream endpoints."""
import json
from pathlib import Path
import numpy as np,pandas as pd,tifffile
from scripts.crgf.common import atomic_json,identity,sha256
NAMES=('unchanged','retained_same_owner_pair','cancelled','reversed_owner_pair','retained_altered_owner_pair','new_difference')

def categories(p0,b0,p1,b1):
    before=p0!=b0;after=p1!=b1;code=np.zeros(p0.shape,np.uint8)
    code[before&after]=4;code[before&~after]=2;code[~before&after]=5
    code[before&after&(p1==p0)&(b1==b0)]=1
    code[before&after&(p1==b0)&(b1==p0)]=3
    return code

def write(methods,baseline_endpoints,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=[];sources={}
    base=json.loads(Path(baseline_endpoints).read_text())['endpoints']
    stages=('postconnect_canonical','tile_merge','split_cleanup','final')
    for arm in ('P1','P2'):
        root=Path(methods[arm]['cost_root']);own=json.loads((root/'predictions'/arm/'endpoints.json').read_text())['endpoints']
        masks=[]
        for entries in (own,base):
            d={}
            for stage in stages:
                obj=entries[stage]
                if sha256(obj['path'])!=obj['sha256']:raise ValueError('canonical endpoint drift')
                d[stage]=tifffile.memmap(obj['path']);sources[obj['path']]=obj
            masks.append(d)
        p,b=masks;initial='postconnect_canonical'
        manifest=json.loads((Path(methods[arm]['matrix'])/'complete.json').read_text())
        for stage in stages[1:]:
            pixels=np.zeros(6,np.int64);transcripts=pixels.copy()
            for y in range(0,p[initial].shape[0],256):
                c=categories(p[initial][y:y+256],b[initial][y:y+256],p[stage][y:y+256],b[stage][y:y+256]);pixels+=np.bincount(c.ravel(),minlength=6)
            for rec in manifest['receipts']:
                if sha256(rec['coordinate_file'])!=rec['coordinate_sha256']:raise ValueError('transcript coordinates drift')
                co=np.load(rec['coordinate_file'],mmap_mode='r');valid=(co['x']>=0)&(co['y']>=0)&(co['x']<p[initial].shape[1])&(co['y']<p[initial].shape[0]);ys=co['y'][valid];xs=co['x'][valid]
                c=categories(p[initial][ys,xs],b[initial][ys,xs],p[stage][ys,xs],b[stage][ys,xs]);transcripts+=np.bincount(c,minlength=6);transcripts[0]+=int((~valid).sum())
            if int(transcripts.sum())!=manifest['totals']['eligible']:raise ValueError('filtered transcript denominator drift')
            for unit,values in [('pixel',pixels),('filtered_transcript',transcripts)]:
                for cat,n in zip(NAMES,values):rows.append(dict(arm=arm,reference='B_native_replay',from_stage=initial,to_stage=stage,unit=unit,category=cat,n=int(n),denominator=int(values.sum()),initial_difference_denominator=int(values[1:5].sum())))
    pd.DataFrame(rows).to_csv(out/'downstream_difference_fate.csv',index=False)
    atomic_json(out/'contract.json',dict(status='succeeded',sources=sources,definition='Both method and B endpoints are followed. Retained categories require P != B after the downstream operation; reversed means the two initial owners exchange. Counts include each canonical pixel once.',outputs={'table':identity(out/'downstream_difference_fate.csv')}))
