"""Stage-specific semantic acceptance; receipts alone cannot assert success."""
import json
from pathlib import Path
import numpy as np
import tifffile
from scripts.crgf.common import sha256,verify_outputs,identity

def read(path):return json.loads(Path(path).read_text())
def validate_stage(stage,r,q,spec):
    if q is None or spec is None:raise ValueError('explicit queue/stage expectation required')
    if r.get('status')!='succeeded' or not r.get('outputs'):raise ValueError('missing successful hashed stage artifacts')
    verify_outputs(r)
    arm=spec.get('method')
    if stage=='report':
        if not r.get('ready') or not r.get('paper_protocol_complete') or set(r.get('arm_receipts',{}))!={'B','S','P1','P2','T1','T2'}:raise ValueError('six-method report incomplete')
        if r.get('fixed_population')!=q['population']:raise ValueError('report population identity')
        return
    if arm not in q['methods']:raise ValueError('unknown arm')
    item=q['methods'][arm];cfg=read(item['config']);root=Path(cfg['output_root'])
    if cfg['arm_id']!=arm or cfg['run_id']!=q['run_id']:raise ValueError('arm/run config mismatch')
    if stage in ('train','predict') and arm.startswith('P'):raise ValueError('P cannot have train/predict stage')
    if stage=='train':
        deps=r['dependencies']
        if deps['method']!=arm or deps['config']['run_id']!=q['run_id'] or deps['config']['arm_id']!=arm:raise ValueError('training arm/run')
        if r.get('updates')!=cfg['updates']:raise ValueError('training update budget')
        from .config import read as read_config
        from .training import dependencies
        if deps!=dependencies(read_config(item['config']),arm):raise ValueError('training dependencies drift')
        import torch
        checkpoint=torch.load(r['outputs']['checkpoint']['path'],map_location='cpu')
        if checkpoint['update_step']!=cfg['updates'] or checkpoint['dependencies']!=deps or checkpoint['sampler_cursor']<cfg['updates']:raise ValueError('checkpoint actual budget/dependencies')
        states=checkpoint['optimizer_state_dict']['state']
        if not states or any(int(v['step'])!=cfg['updates'] for v in states.values()):raise ValueError('Adam real update count')
        entries=[json.loads(l) for l in (root/'checkpoints'/arm/'updates.jsonl').read_text().splitlines() if l]
        if [v['update'] for v in entries]!=list(range(1,cfg['updates']+1)):raise ValueError('actual update log mismatch')
        if not cfg['mini_test']:
            for step in (100,1000,2000,4000):
                probe=read(root/'diagnostics'/f'probe_{step:06d}.json')
                if probe['status']!='succeeded' or probe['arm']!=arm or probe['update']!=step or len(probe['samples'])!=8:raise ValueError('milestone eight-probe missing')
    elif stage=='predict':
        if r['dependencies']['config']!=cfg:raise ValueError('prediction config drift')
        if {v['shift'] for v in r['shifts']}!={0,24} or len(r['shifts'])!=2:raise ValueError('dual shift missing')
        for shift in r['shifts']:
            inventory=read(root/f'configs/patches_shift{shift["shift"]}.json')
            if shift['patch_count']!=len(inventory):raise ValueError('prediction patch coverage')
            for obj in shift['chunks']:
                if sha256(obj['path'])!=obj['sha256']:raise ValueError('prediction receipt hash')
                verify_outputs(read(obj['path']))
    elif stage=='postprocess':
        if r.get('schema')!='BS4-postprocess-v2' or r.get('arm')!=arm or r.get('run')!=q['run_id'] or r.get('exit_code')!=0:raise ValueError('postprocess identity/schema')
        if r['dependencies']['config']!=cfg:raise ValueError('postprocess config drift')
        for name in ('pre','nuclei'):
            obj=r['dependencies'][name]
            if sha256(obj['path'])!=obj['sha256']:raise ValueError('post input drift')
        from .postprocess import validate_tile
        m=read(r['outputs']['manifest']['path']);coords=m['coords']
        shape=tifffile.memmap(r['dependencies']['pre']['path']).shape;size=min(cfg['postprocess_size'],min(shape))
        expected=[[y,x] for y in list(range(0,shape[0]-size,size))+[shape[0]-size] for x in list(range(0,shape[1]-size,size))+[shape[1]-size]]
        if coords!=expected or r['tiles']!=len(coords) or len(m['tiles'])!=len(coords):raise ValueError('tile coords/count coverage')
        for xy,obj in zip(coords,m['tiles']):
            if sha256(obj['path'])!=obj['sha256']:raise ValueError('tile receipt drift')
            tr=read(obj['path']);expected_dep=dict(base=r['dependencies'],coords=xy,shape=[size,size],dtype='uint32')
            validate_tile(obj['path'],expected_dep)
        endpoints=read(r['outputs']['endpoints']['path'])['endpoints']
        if set(endpoints)!={'raw_shift0','shift0','raw_shift24','shift24','pre','postconnect_canonical','tile_merge','split_cleanup','before_global_nuclear_restore','final'}:raise ValueError('native endpoint set')
        for name,obj in endpoints.items():
            if sha256(obj['path'])!=obj['sha256']:raise ValueError('endpoint hash')
            a=tifffile.memmap(obj['path'])
            if a.shape!=shape or a.dtype!=np.uint32:raise ValueError('endpoint shape/dtype')
    elif stage=='matrix':
        from scripts.crgf.evaluate_egr import validate_matrix
        validate_matrix(read(item['machine']),item['matrix'])
        totals=r['totals']
        if totals['assigned']+totals['background']!=totals['eligible']:raise ValueError('matrix conservation')
    elif stage=='evaluate':
        if r.get('version')!='BS4-OVERNIGHT-v2.0' or r.get('method')!=arm or not r.get('evaluation_executed') or not r.get('paper_method_complete'):raise ValueError('evaluation identity/completion')
        if r.get('profiles')!=['engineering_nuclear_v1','engineering_pearson_v1','paper_aligned_v1']:raise ValueError('three exact profiles missing')
        if r['dependencies']['arm_config']!=identity(item['config']) or r['dependencies']['queue_config']!=identity(cfg['queue_config']):raise ValueError('evaluation config drift')
        paper=Path(item['evaluation'])/'paper_aligned_v1'
        for name in ('scClassify_raw_predictions.rds','annotation.csv','R_sessionInfo.txt','paper_evaluation.R.log'):
            obj=r['outputs'].get('paper_aligned_v1/'+name)
            if not obj or obj['bytes']==0 or sha256(paper/name)!=obj['sha256']:raise ValueError('real scClassify artifact missing')
    else:raise ValueError('unknown stage')
