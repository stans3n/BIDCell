"""Resolve ABC inputs exclusively from accepted historical manifests."""
import json,shutil,sys
from pathlib import Path
from scripts.crgf.common import atomic_json,identity,sha256,verify_outputs
from .config import CosConfig
HIST=Path('/home/zwan0726/work/outputs/crgf_cos_20261009T000650+1100')
def select(items,quotas):
    selected=[];counts={};fallback=[]
    for name,pred,quota in [('K1',lambda k:k==1,quotas[0]),('K2_4',lambda k:2<=k<=4,quotas[1]),('K5plus',lambda k:k>=5,quotas[2])]:
        chunk=[v for v in items if pred(v['k'])][:quota];selected.extend(chunk);counts[name]=len(chunk)
    for pred in (lambda k:k==1,lambda k:2<=k<=4,lambda k:k>=5):
        for v in items:
            if len(selected)==sum(quotas):break
            if pred(v['k']) and v not in selected:selected.append(v);fallback.append(v['view_cursor'])
    assert len(selected)==sum(quotas)
    selected.sort(key=lambda v:v['view_cursor'])
    return dict(samples=selected,requested=quotas,selected_counts=counts,fallback_cursors=fallback)

def prepare(root):
    root=Path(root); repo=Path(__file__).resolve().parents[2]
    if (root/'configs/queue.json').exists(): raise FileExistsError('already prepared')
    old=json.loads((HIST/'B1_COS/configs/run.json').read_text())
    parent=Path(old['checkpoint']);complete_path=parent.parent/'complete.json';complete=json.loads(complete_path.read_text());verify_outputs(complete)
    assert complete['updates']==4000 and sha256(parent)==old['parent_sha256']=='02ce88f2007eaa7a4f4de534b951a4ea0c75243bccf2804e4ea17c50d4de3b39'
    assert json.loads((HIST/'final_acceptance.json').read_text())['accepted']
    for p in (HIST/'configs').glob('*.json'):
        if p.name!='queue.json':shutil.copyfile(p,root/'configs'/p.name)
    views=json.loads((root/'configs/views.json').read_text());unique=[];seen=set()
    for i,v in enumerate(views['views']):
        if v['k'] and v['patch'] not in seen: unique.append(dict(v,view_cursor=i));seen.add(v['patch'])
    selection=select(unique,[16,24,24]);selection.update(version='ABC-gradient-samples-v1',source_views=identity(root/'configs/views.json'),selection='first distinct nonempty patches in frozen view order, strata K1/K2-4/K5plus; fallback same order')
    for v in selection['samples']:v['patch_sha256']=sha256(v['patch'])
    atomic_json(root/'diagnostics/gradient_samples.json',selection)
    probes=select(selection['samples'],[2,3,3]);probes['source_samples']=identity(root/'diagnostics/gradient_samples.json');atomic_json(root/'configs/training_probes.json',probes)
    q=json.loads((HIST/'configs/queue.json').read_text());q.update(run_id=root.name.removeprefix('cos_loss_'),output_root=str(root),python=sys.executable,methods={},minimum_free_gb=15.,cpu_workers=4,admission=str(root/'tests/admission.json'),stage_timeouts=dict(train=21600,predict=43200,postprocess=28800,matrix=3600,evaluate=14400,report=1800))
    q['reference']=str(root/'configs/reference.json');q['engineering_protocol']=str(root/'configs/protocol.json')
    machine=json.loads((HIST/'B1_COS/configs/B1_COS_machine.json').read_text())
    for arm in 'ABC':
        ar=root/arm;(ar/'configs').mkdir(parents=True,exist_ok=True);(ar/'source').mkdir(exist_ok=True)
        for p in (root/'configs').glob('*.json'):shutil.copyfile(p,ar/'configs'/p.name)
        cfg=dict(old,arm_id=arm,base_variant='B1',variant='B1',cos_enabled=True,output_root=str(ar),run_id=q['run_id'],queue_config=str(root/'configs/queue.json'),method_version='COS-LOSS-ABC-v2.0',cpu_workers=2,checkpoint_every=1000,minimum_free_gb=15.,anchor_weight=0. if arm=='A' else 1.,pos_background_rho=.5 if arm=='C' else 1.,calibration_identity='PENDING')
        cfg=CosConfig.model_validate(cfg);atomic_json(ar/'configs/run.json',cfg.model_dump(mode='json'))
        m=dict(machine,final_mask=str(ar/'predictions'/arm/'final.tif'),pre_mask=str(ar/'predictions'/arm/'pre.tif'),final_matrix=str(ar/'matrix'/arm/'expr_mat.csv'));atomic_json(ar/f'configs/{arm}_machine.json',m)
        atomic_json(ar/'configs/frozen.json',dict(version='ABC-provisional',identities=dict(parent_complete=identity(complete_path),views=identity(ar/'configs/views.json'),config=identity(ar/'configs/run.json'))))
        q['methods'][arm]=dict(config=str(ar/'configs/run.json'),worktree=str(repo),commit='PENDING',machine=str(ar/f'configs/{arm}_machine.json'),matrix=str(ar/'matrix'/arm),evaluation=str(ar/'evaluation'/arm),cost_root=str(ar))
    atomic_json(root/'configs/queue.json',q)
    atomic_json(root/'source/prepare.json',dict(status='prepared_provisional',common_commit='263f91a6dcca7c597f83a38d646259762c2cc9d8',historical_acceptance=identity(HIST/'final_acceptance.json'),parent=identity(parent),parent_complete=identity(complete_path),historical_views=identity(HIST/'configs/views.json'),views=identity(root/'configs/views.json'),implementer=dict(model='gpt-6-astra',effort='high')))
if __name__=='__main__':prepare(sys.argv[1])
