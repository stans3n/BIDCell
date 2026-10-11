"""Idempotent frozen input preparation and actual-view teacher exposure inventory."""
import argparse,json,time,collections
from pathlib import Path
import torch
from scripts.crgf.common import atomic_json,identity,sha256,source_identity
from scripts.cos.ownership import compact_sample
from .config import read

from .data import dataset
from .method import teacher,stats,target_hash

def coverage(root,cfg):
    arm=cfg.arm_id
    root=Path(root);out=root/'prepared';dest=out/f'coverage_{arm}.json';source=source_identity(['scripts/bs4/method.py','scripts/bs4/data.py','bidcell/model/dataio/dataset_input.py','bidcell/model/dataio/candidate_geometry.py','scripts/bs4/prepare.py','scripts/ssp/method.py','scripts/cos/ownership.py'])
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

        for cid in v['cell_ids']:exposure[int(cid)]+=1
        rows.append(dict(view_cursor=cursor,patch=v['patch'],flip=v['flip'],rotation=v['rotation'],**cache[key]))
        unique_crop.setdefault(v['patch'],cache[key])
        if len(rows)%100==0:atomic_json(out/f'coverage_progress_{arm}.json',dict(updates=len(rows),total=4000,seconds=time.monotonic()-start))
    def aggregate(records):
        keys=('H','E','J','F','F_target_sum','F_weighted_target_sum','replaced_candidate_pixels','old_p1_modified','old_p0_modified','strong_negative_preserved','conflict_pixels','weights_sum','weight_reduction_sum','p1_softened_pixels','p1_softening_mass')
        totals={k:sum(r[k] for r in records) for k in keys};totals.update(views=len(records),views_with_H=sum(r['H']>0 for r in records),views_with_F=sum(r['F']>0 for r in records),H_over_E=totals['H']/totals['E'] if totals['E'] else None,F_over_J=totals['F']/totals['J'] if totals['J'] else None,winner_ids=len(set(c for r in records for c in r['winner_ids'])),affected_ids=len(set(c for r in records for c in r['affected_ids'])))
        return totals
    formal=aggregate(rows)
    if formal['F']==0 or not any(r['delta_gradient_nonzero'] for r in rows):raise ValueError('zero actual new S supervision')
    # Pool descriptor distinct from augmented formal exposure: canonical unaugmented pool.
    pool=[]
    for index,path in enumerate(ds.fp_patches):
        ds.view_transform=(0,0);s=compact_sample(ds[index])
        if len(s['cell_ids']):pool.append(stats(teacher(s),s))
        if index%250==0:atomic_json(out/f'pool_progress_{arm}.json',dict(completed=index+1,total=len(ds)))
    ids=sorted(exposure)
    if arm=='T2' and formal['p1_softened_pixels']==0:raise ValueError('T2 p1 softening empty')
    warnings=[]
    if formal['H']<1000:warnings.append('H fewer than 1000 crop pixels')
    if (formal['H_over_E'] or 0)<.005:warnings.append('H/E below 0.5%')
    if formal['views_with_H']/sum(r['E']>0 for r in rows)<.1:warnings.append('H view coverage below 10%')
    if formal['winner_ids']<100:warnings.append('fewer than 100 winner nuclei')
    atomic_json(out/f'coverage_views_{arm}.json',dict(rows=rows))
    result=dict(status='passed',prepared=identity(cfg.prepared_manifest),source=source,views=identity(root/'configs/views.json'),formal_exposure=formal,unique_crop_first_formal_view=aggregate(list(unique_crop.values())),full_training_pool_unaugmented=aggregate(pool),marker_gene_intersections=int(ds.marker_intersection.sum()),
        candidate_coverage=dict(candidate_occurrences=sum(exposure.values()),unique_training_candidate_ids=len(ids)),warnings=warnings,wall_seconds=time.monotonic()-start,rows=identity(out/f'coverage_views_{arm}.json'))
    atomic_json(dest,result);return result

def prepare(root,inventory=True):
    root=Path(root)
    for arm in ('T1','T2'):coverage(root,read(root/arm/'configs/run.json'))
    rows=[json.loads((root/f'prepared/coverage_views_{a}.json').read_text())['rows'] for a in ('T1','T2')]
    if len(rows[0])!=4000 or len(rows[1])!=4000:raise ValueError('formal exposure update count')
    for a,b in zip(*rows):
        if a['frozen_teacher_sha256']!=b['frozen_teacher_sha256']:raise ValueError('alpha affected frozen teacher')
    atomic_json(root/'prepared/coverage.json',dict(status='passed',formal_updates_each=4000,alpha_independent_teacher_verified=True,arms={a:identity(root/f'prepared/coverage_{a}.json') for a in ('T1','T2')}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();prepare(a.root)
