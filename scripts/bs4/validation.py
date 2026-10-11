from .tables import read_table
"""Stage-specific semantic acceptance; receipts alone cannot assert success."""
import json
from pathlib import Path
import numpy as np
import tifffile
from scripts.crgf.common import sha256,verify_outputs,identity,digest

def read(path):return json.loads(Path(path).read_text())
def validate_checkpoint_payload(checkpoint,deps,cfg,path):
    import torch
    from .training import finite_state
    from scripts.crgf.common import digest
    checkpoint_path=Path(path);sidecar_path=checkpoint_path.with_suffix('.json')
    if checkpoint_path.name=='final.pth':
        milestone=checkpoint_path.with_name(f"update_{checkpoint['update_step']:06d}.pth")
        if sha256(milestone)!=sha256(checkpoint_path):raise ValueError('final differs from published final milestone')
        sidecar_path=milestone.with_suffix('.json')
    sidecar=read(sidecar_path)
    if sidecar['checkpoint']['sha256']!=sha256(path) or sidecar['dependencies_hash']!=digest(deps):raise ValueError('checkpoint sidecar dependency/hash')
    step=checkpoint['update_step'];cursor=checkpoint['sampler_cursor']
    if sidecar['update']!=step or sidecar['cursor']!=cursor or checkpoint['visited_views']!=cursor:raise ValueError('checkpoint counters')
    views=read(Path(cfg['output_root'])/'configs/views.json')['views']
    if not 0<=cursor<=len(views) or sum(v['k']>0 for v in views[:cursor])!=step:raise ValueError('checkpoint real frozen-view prefix')
    if set(checkpoint['unique_patches'])!={v['index'] for v in views[:cursor] if v['k']>0}:raise ValueError('checkpoint seen prefix')
    weights=checkpoint['model_state_dict'];optim=checkpoint['optimizer_state_dict'];finite_state(weights);finite_state(optim)
    names=[n for n in weights if not n.endswith(('running_mean','running_var','num_batches_tracked'))]
    empty=[torch.nn.Parameter(torch.empty(0)) for n in names]
    expected=torch.optim.Adam(empty,lr=cfg['learning_rate'],betas=tuple(cfg['betas']),eps=cfg['eps'],weight_decay=cfg['weight_decay']).state_dict()['param_groups']
    if optim['param_groups']!=expected or set(optim['state'])!=set(range(len(names))):raise ValueError('checkpoint optimizer schema/parameter order')
    for i,n in enumerate(names):
        state=optim['state'][i]
        if float(state['step'])!=step or any(tuple(state[k].shape)!=tuple(weights[n].shape) for k in ('exp_avg','exp_avg_sq')) or (state['exp_avg_sq']<0).any():raise ValueError('checkpoint Adam state')
    if checkpoint['architecture']!={k:list(v.shape) for k,v in weights.items()}:raise ValueError('checkpoint architecture')
    if checkpoint['gene_order']!=deps['inputs']['gene_order'] or checkpoint['rng_version']!='python-numpy-torch-cuda-v1' or checkpoint['sampler_version']!='frozen-views-cursor-v1':raise ValueError('checkpoint contract')


def validate_stage(stage,r,q,spec):
    if q is None or spec is None:raise ValueError('explicit queue/stage expectation required')
    if r.get('status')!='succeeded' or not r.get('outputs'):raise ValueError('missing successful hashed stage artifacts')
    verify_outputs(r)
    arm=spec.get('method')
    if stage=='report':
        import pandas as pd
        if not r.get('ready') or not r.get('paper_protocol_complete') or set(r.get('arm_receipts',{}))!={'B','S','P1','P2','T1','T2'}:raise ValueError('six-method report incomplete')
        if r.get('fixed_population')!=q['population']:raise ValueError('report population identity')
        out=Path(q['output_root'])/'comparison';population=pd.read_csv(q['population']['path']).cell_id.to_numpy()
        required=['fixed_B1QC/paired_metrics.csv','fixed_B1QC/metric_summary.csv','fixed_B1QC/type_macro_metrics.csv','purity.json','direction_screen.json','tradeoffs.csv','metric_coverage.csv','raw_metrics.csv','diagnostics/stage_population_domains.csv','post_effects/downstream_difference_fate.csv','post_effects/contract.json']
        required += ['fixed_B1QC/'+a+'.csv.gz' for a in ('B','S','P1','P2','T1','T2')]
        for filename in required:
            if filename not in r['outputs']:raise ValueError('required report artifact missing '+filename)
        for a in ('B','S','P1','P2','T1','T2'):
            table=read_table(out/'fixed_B1QC'/f'{a}.csv')
            if not np.array_equal(table.cell_id.to_numpy(),population) or table.cell_id.duplicated().any():raise ValueError('report fixed population order')
            item={**q['historical_methods'],**q['methods']}[a]
            own_ids=pd.read_csv(Path(item['matrix'])/'cell_mapping.csv').canonical_cell_id.to_numpy()
            present=np.isin(population,own_ids)
            if not np.array_equal(table.stage_row_present.to_numpy(bool),present):raise ValueError('report own-matrix presence domain')
            for metric in ('positive_F1','negative_F1','positive_Precision','negative_Precision','positive_Recall','negative_Recall'):
                if not table.loc[~present,metric].isna().all() or not table.loc[~present,metric+'_reason'].eq('missing_stage_id').all():raise ValueError('report missing own IDs must remain explicit NA')
        pairs={'P1-B','P2-B','T1-B','T2-B','T2-T1','T1-S','T2-S'}
        table=pd.read_csv(out/'fixed_B1QC/paired_metrics.csv')
        if set(table.comparison)!=pairs or not {'positive_F1','negative_F1','positive_Precision','negative_Precision','positive_Recall','negative_Recall'}<=set(table.metric):raise ValueError('report contrasts/metrics incomplete')
        purity=read(out/'purity.json')
        if purity.get('method_set')!=['B','S','T1','T2','P1','P2']:raise ValueError('purity method set')
        coverage=pd.read_csv(out/'metric_coverage.csv')
        required_rows=coverage[coverage.required_this_run.astype(str).str.lower()=='true']
        if not len(required_rows) or not set(required_rows.status)<={'computed','computed_distribution','partial_domain_NA','not_identifiable_zero_range'}:raise ValueError('required metric coverage incomplete')
        if not {'paper_aligned_v1','engineering_nuclear_v1','engineering_pearson_v1'}<=set(coverage.profile):raise ValueError('profile coverage incomplete')
        for method in ('P1','P2','T1','T2'):
            for st in ('train','predict','postprocess','matrix','evaluate'):
                if method.startswith('P') and st in ('train','predict'):continue
                obj=r['arm_receipts'][method][st]
                if sha256(obj['path'])!=obj['sha256']:raise ValueError('report arm receipt drift')
                validate_stage(st,read(obj['path']),q,dict(method=method))
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
        if digest(deps)!=digest(dependencies(read_config(item['config']),arm)):raise ValueError('training dependencies drift')
        import torch
        checkpoint=torch.load(r['outputs']['checkpoint']['path'],map_location='cpu')
        if checkpoint['update_step']!=cfg['updates'] or digest(checkpoint['dependencies'])!=digest(deps) or checkpoint['sampler_cursor']<cfg['updates']:raise ValueError('checkpoint actual budget/dependencies')
        validate_checkpoint_payload(checkpoint,deps,cfg,r['outputs']['checkpoint']['path'])
        states=checkpoint['optimizer_state_dict']['state']
        if not states or any(int(v['step'])!=cfg['updates'] for v in states.values()):raise ValueError('Adam real update count')
        entries=[json.loads(l) for l in (root/'checkpoints'/arm/'updates.jsonl').read_text().splitlines() if l]
        if [v['update'] for v in entries]!=list(range(1,cfg['updates']+1)):raise ValueError('actual update log mismatch')
        if not cfg['mini_test']:
            for step in (100,1000,2000,4000):
                cp=root/'checkpoints'/arm/f'update_{step:06d}.pth';sidecar=read(cp.with_suffix('.json'))
                if sidecar['checkpoint']['sha256']!=sha256(cp) or sidecar['update']!=step:raise ValueError('milestone checkpoint hash/update')
                state=torch.load(cp,map_location='cpu')
                if state['update_step']!=step or digest(state['dependencies'])!=digest(deps) or state['sampler_cursor']<step or state['sampler_cursor']!=sidecar['cursor']:raise ValueError('milestone state/dependencies')
                if any(int(v['step'])!=step for v in state['optimizer_state_dict']['state'].values()):raise ValueError('milestone Adam update')
                validate_checkpoint_payload(state,deps,cfg,cp)
                del state
                probe=read(root/'diagnostics'/f'probe_{step:06d}.json')
                if probe['status']!='succeeded' or probe['arm']!=arm or probe['update']!=step or len(probe['samples'])!=8:raise ValueError('milestone eight-probe missing')
    elif stage=='predict':
        if r['dependencies']['config']!=cfg:raise ValueError('prediction config drift')
        if {v['shift'] for v in r['shifts']}!={0,24} or len(r['shifts'])!=2:raise ValueError('dual shift missing')
        for shift in r['shifts']:
            inventory=read(root/f'configs/patches_shift{shift["shift"]}.json')
            if shift['patch_count']!=len(inventory):raise ValueError('prediction patch coverage')
            covered=[]
            for obj in shift['chunks']:
                if sha256(obj['path'])!=obj['sha256']:raise ValueError('prediction receipt hash')
                chunk=read(obj['path']);verify_outputs(chunk)
                cd=chunk['dependencies'];start,stop=cd['start'],cd['stop']
                if cd['dependencies']!=r['dependencies'] or cd['patches']!=inventory[start:stop]:raise ValueError('prediction chunk dependency coverage')
                covered.extend(range(start,stop))
            if covered!=list(range(len(inventory))):raise ValueError('prediction chunk gaps/duplicates')
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
        machine=read(item['machine']);validate_matrix(machine,item['matrix'])
        deps=r.get('inputs',r.get('dependencies',{}))
        # Accepted validator binds final mask, nuclei, genes, transcripts and integer reconstruction.
        if machine['final_mask']!=str(root/'predictions'/arm/'final.tif') or machine['final_matrix']!=str(Path(item['matrix'])/'expr_mat.csv'):raise ValueError('matrix arm path mismatch')
        if identity(Path(item['matrix'])/'complete.json')['sha256']!=__import__('scripts.crgf.common',fromlist=['digest']).digest(r):
            actual=read(Path(item['matrix'])/'complete.json')
            if actual!=r:raise ValueError('matrix receipt belongs to another arm')
        totals=r['totals']
        if totals['assigned']+totals['unassigned']!=totals['eligible']:raise ValueError('matrix conservation')
    elif stage=='evaluate':
        if r.get('version')!='BS4-OVERNIGHT-v2.0' or r.get('method')!=arm or not r.get('evaluation_executed') or not r.get('paper_method_complete'):raise ValueError('evaluation identity/completion')
        if r.get('profiles')!=['engineering_nuclear_v1','engineering_pearson_v1','paper_aligned_v1']:raise ValueError('three exact profiles missing')
        if r['dependencies']['arm_config']!=identity(item['config']) or r['dependencies']['queue_config']!=identity(cfg['queue_config']):raise ValueError('evaluation config drift')
        from .evaluation import method_coverage
        coverage,_=method_coverage(q,arm,include_resources=False)
        if any(row['required_this_run'] and row['status']=='failed' for row in coverage):raise ValueError('required evaluation artifact missing')
        evaluation=Path(item['evaluation']);stage_validation=read(evaluation/'diagnostics/stages/validation.json')
        if not all(stage_validation.get(k) is True for k in ('integer_conservation','cell_gene_gain_loss_verified','all_required_endpoints')):raise ValueError('stage counts/owner conservation missing')
        for name in ('pre','postconnect_canonical','tile_merge','final'):
            for file in (f'counts_{name}.npz',f'cells_{name}.csv',f'fixed_B1_{name}.csv.gz'):
                if 'diagnostics/stages/'+file not in r['outputs']:raise ValueError('required diagnostic stage missing '+file)
        paper=Path(item['evaluation'])/'paper_aligned_v1'
        for name in ('scClassify_raw_predictions.rds','annotation.csv','R_sessionInfo.txt','paper_evaluation.R.log'):
            obj=r['outputs'].get('paper_aligned_v1/'+name)
            if not obj or obj['bytes']==0 or sha256(paper/name)!=obj['sha256']:raise ValueError('real scClassify artifact missing')
    else:raise ValueError('unknown stage')
