"""Idempotent frozen input preparation and actual-view teacher exposure inventory."""
import argparse,json,time,collections
from pathlib import Path
import torch
from scripts.crgf.common import atomic_json,identity,sha256,source_identity
from scripts.cos.ownership import compact_sample
from .config import read
from .prototype import build
from .data import dataset
from .method import teacher,stats,target_hash

def coverage(root,cfg):
    root=Path(root);out=root/'prepared';dest=out/'coverage.json';source=source_identity(['scripts/ssp/method.py','scripts/ssp/data.py','bidcell/model/dataio/dataset_input.py','bidcell/model/dataio/candidate_geometry.py','scripts/ssp/prepare.py','scripts/ssp/prototype.py','scripts/cos/ownership.py'])
    if dest.exists():
        prev=json.loads(dest.read_text())
        if prev.get('prepared',{}).get('sha256')!=sha256(cfg.prepared_manifest) or prev['source']!=source or prev['views']['sha256']!=sha256(root/'configs/views.json'):raise ValueError('coverage source/views changed')
        return prev
    ds=dataset(cfg);torch.set_num_threads(2);views=json.loads((root/'configs/views.json').read_text());rows=[];start=time.monotonic();cache={};unique_crop={};exposure=collections.Counter();positive=collections.Counter();confident=collections.Counter()
    def run(v):
        ds.view_transform=(v['flip'],v['rotation']);s=compact_sample(ds[v['index']])
        if list(s['cell_ids'])!=v['cell_ids']:raise ValueError('coverage candidate alignment')
        t=teacher(s);st=stats(t,s);st['target_sha256']=target_hash(t)
        # Exact derivative of soft-vs-hard binary CE: nonzero independent of network for t != p or w != 1.
        st['delta_gradient_nonzero']=bool(((t['target']-s['p'][:,0]).abs()[t['replace']]>0).any())
        return st,s
    for cursor,v in enumerate(views['views']):
        if not v['k']:continue
        key=(v['index'],v['flip'],v['rotation'])
        if key not in cache:
            st,s=run(v);cache[key]=st
            for i,cid in enumerate(s['cell_ids']):
                positive[int(cid)]+=int(ds.prototype_cache.arrays['N'][ds.prototype_cache.arrays['ids'].searchsorted(cid)]>0);confident[int(cid)]+=int(s['prototype_confidence'][i]>=.25)
        for cid in v['cell_ids']:exposure[int(cid)]+=1
        rows.append(dict(view_cursor=cursor,patch=v['patch'],flip=v['flip'],rotation=v['rotation'],**cache[key]))
        unique_crop.setdefault(v['patch'],cache[key])
        if len(rows)%100==0:atomic_json(out/'coverage_progress.json',dict(updates=len(rows),total=4000,seconds=time.monotonic()-start))
    def aggregate(records):
        keys=('H','E','J','F','F_target_sum','F_weighted_target_sum','replaced_candidate_pixels','old_p1_modified','old_p0_modified','strong_negative_preserved','conflict_pixels','weights_sum','weight_reduction_sum')
        totals={k:sum(r[k] for r in records) for k in keys};totals.update(views=len(records),views_with_H=sum(r['H']>0 for r in records),views_with_F=sum(r['F']>0 for r in records),H_over_E=totals['H']/totals['E'] if totals['E'] else None,F_over_J=totals['F']/totals['J'] if totals['J'] else None,winner_ids=len(set(c for r in records for c in r['winner_ids'])),affected_ids=len(set(c for r in records for c in r['affected_ids'])))
        return totals
    formal=aggregate(rows)
    if formal['F']==0 or not any(r['delta_gradient_nonzero'] for r in rows):raise ValueError('zero actual new S supervision')
    # Pool descriptor distinct from augmented formal exposure: canonical unaugmented pool.
    pool=[]
    for index,path in enumerate(ds.fp_patches):
        ds.view_transform=(0,0);s=compact_sample(ds[index])
        if len(s['cell_ids']):pool.append(stats(teacher(s),s))
        if index%250==0:atomic_json(out/'pool_progress.json',dict(completed=index+1,total=len(ds)))
    ids=sorted(exposure);ix=ds.prototype_cache.arrays['ids'].searchsorted(ids);Ns=ds.prototype_cache.arrays['N'][ix];cs=ds.prototype_cache.arrays['confidence'][ix]
    warnings=[]
    if formal['H']<1000:warnings.append('H fewer than 1000 crop pixels')
    if (formal['H_over_E'] or 0)<.005:warnings.append('H/E below 0.5%')
    if formal['views_with_H']/sum(r['E']>0 for r in rows)<.1:warnings.append('H view coverage below 10%')
    if formal['winner_ids']<100:warnings.append('fewer than 100 winner nuclei')
    atomic_json(out/'coverage_views.json',dict(rows=rows))
    result=dict(status='passed',prepared=identity(cfg.prepared_manifest),source=source,views=identity(root/'configs/views.json'),formal_exposure=formal,unique_crop_first_formal_view=aggregate(list(unique_crop.values())),full_training_pool_unaugmented=aggregate(pool),marker_gene_intersections=int(ds.marker_intersection.sum()),
        prototype_coverage=dict(candidate_occurrences=sum(exposure.values()),weighted_N_positive_fraction=sum(exposure[c] for c,n in zip(ids,Ns) if n>0)/sum(exposure.values()),weighted_confidence_ge025_fraction=sum(exposure[c] for c,cval in zip(ids,cs) if cval>=.25)/sum(exposure.values()),unique_training_candidate_ids=len(ids),N_positive_fraction=float((Ns>0).mean()),confidence_ge025_fraction=float((cs>=.25).mean()),all_raster_inference_ids_lookup=True),warnings=warnings,wall_seconds=time.monotonic()-start,rows=identity(out/'coverage_views.json'))
    atomic_json(dest,result);return result

def prepare(root,inventory=True):
    root=Path(root);cfg=read(root/'S/configs/run.json');build(root,cfg)
    if inventory:coverage(root,cfg)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--prototype-only',action='store_true');a=p.parse_args();prepare(a.root,not a.prototype_only)
