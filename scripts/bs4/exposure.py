"""Frozen nuclear preannotation exposure, by nucleus/type/space in three explicit domains."""
import argparse,collections,json,re,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from scripts.crgf.common import atomic_json,identity,source_identity
from scripts.cos.ownership import compact_sample
from .config import read
from .data import dataset
from .method import teacher
from .tables import write_table
SOURCES=['scripts/bs4/exposure.py','scripts/bs4/tables.py','scripts/bs4/data.py','scripts/bs4/method.py','scripts/ssp/method.py','scripts/cos/ownership.py','bidcell/model/dataio/dataset_input.py','bidcell/model/dataio/candidate_geometry.py']

def run(root):
    root=Path(root);out=root/'prepared/exposure';out.mkdir(parents=True,exist_ok=True);execution=source_identity(SOURCES);torch.set_num_threads(2)
    cfg=read(root/'T1/configs/run.json');ds=dataset(cfg);views=json.loads((root/'configs/views.json').read_text())['views'];nuclei=collections.defaultdict(collections.Counter);spatial=[];cache={};types={};start=time.monotonic()
    def calculate(index,flip,rotation):
        ds.view_transform=(flip,rotation);s=compact_sample(ds[index]);ids=s['cell_ids']
        if not len(ids):return []
        t=teacher(s);p=s['p'][:,0];E=t['E_candidate'];p0=E&(p==0)&(t['pi']>0);p1=E&(p>0)&(t['pi']<1)
        winner=torch.zeros_like(p,dtype=torch.bool);winner.scatter_(0,t['winner'][None],t['H'][None]);rows=[]
        for i,cid in enumerate(ids):
            typ=ds.type_names[s['type_indices'][i]] if bool(s['type_valid'][i]) else '__unknown__'
            if int(cid) in types and types[int(cid)]!=typ:raise ValueError('frozen preannotation type drift')
            types[int(cid)]=typ
            rows.append(dict(cell_id=int(cid),type=typ,candidate=1,winner=int(winner[i].any()),affected_p0=int(p0[i].any()),affected_p1_T2=int(p1[i].any()),p0_pixels=int(p0[i].sum()),p1_pixels_T2=int(p1[i].sum()),p0_mass=float(t['pi'][i][p0[i]].sum()),p1_mass_T2=float((.25*(1-t['pi'][i]))[p1[i]].sum())))
        return rows
    def add(scope,index,rows,view_cursor,flip,rotation):
        path=ds.fp_patches[index];xy=list(map(int,re.findall(r'\d+',Path(path).name)[:2]))
        for row in rows:
            key=(scope,row['cell_id']);nuclei[key].update({k:v for k,v in row.items() if k not in ('cell_id','type')})
        spatial.append(dict(scope=scope,index=index,view_cursor=view_cursor,patch=path,global_y=xy[0],global_x=xy[1],flip=flip,rotation=rotation,**{k:sum(r[k] for r in rows) for k in ('candidate','winner','affected_p0','affected_p1_T2','p0_pixels','p1_pixels_T2','p0_mass','p1_mass_T2')}))
    first=set()
    for cursor,v in enumerate(views):
        if not v['k']:continue
        key=(v['index'],v['flip'],v['rotation'])
        if key not in cache:cache[key]=calculate(*key)
        add('formal_4000',v['index'],cache[key],cursor,v['flip'],v['rotation'])
        if v['index'] not in first:add('unique_crop_first_formal_view',v['index'],cache[key],cursor,v['flip'],v['rotation']);first.add(v['index'])
        if cursor%250==0:atomic_json(out/'progress.json',dict(scope='formal',completed=cursor,seconds=time.monotonic()-start))
    cache.clear()
    for index in range(len(ds)):
        add('full_training_pool_unaugmented',index,calculate(index,0,0),None,0,0)
        if index%250==0:atomic_json(out/'progress.json',dict(scope='full_pool',completed=index+1,total=len(ds),seconds=time.monotonic()-start))
    frame=pd.DataFrame([dict(scope=scope,cell_id=cid,type=types[cid],**dict(counts)) for (scope,cid),counts in sorted(nuclei.items())]);bytype=frame.groupby(['scope','type'],sort=True).sum(numeric_only=True).drop(columns='cell_id');bytype['unique_nuclei']=frame.groupby(['scope','type']).size()
    files={'nuclei':write_table(frame,out/'per_nucleus.csv',index=False),'type':write_table(bytype,out/'per_type.csv'),'spatial':write_table(pd.DataFrame(spatial),out/'spatial.csv',index=False)}
    if source_identity(SOURCES)!=execution:raise ValueError('exposure execution source changed')
    atomic_json(out/'complete.json',dict(status='passed',execution_source=execution,source_labels='frozen nuclear preannotation only; no evaluation labels',domains=['formal_4000','unique_crop_first_formal_view','full_training_pool_unaugmented'],arm_semantics={'T1':'p0 columns; p1 exactly zero','T2':'same frozen candidate/winner/p0 plus p1_T2 columns'},outputs={k:identity(p) for k,p in files.items()},seconds=time.monotonic()-start))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();run(a.root)
