"""Read-only provenance resolution and content-addressed frozen replay preparation."""
import argparse,json,shutil,re,sys
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import numpy as np
import tifffile
from scripts.crgf.common import atomic_json,identity,sha256,digest,verify_outputs
from .config import CosConfig,ARMS

WORK=Path('/home/zwan0726/work')
HIST=WORK/'outputs/crgf_egr_20261008T121953+1100'
ENTRY=HIST/'configs/evaluation_methods.json'

def prepare(output=None):
    stamp=datetime.now(ZoneInfo('Australia/Sydney')).strftime('%Y%m%dT%H%M%S%z')
    root=Path(output) if output else WORK/'outputs'/('crgf_cos_'+stamp)
    if root.exists(): raise FileExistsError(root)
    (root/'configs').mkdir(parents=True)
    (root/'source').mkdir();(root/'tests').mkdir()
    entry=json.loads(ENTRY.read_text())['methods'];parents={};historical_views={}
    identities={'authority':identity(ENTRY)}
    for base in ('B1','G'):
        item=entry[base];cost=Path(item['cost_root']);name=item.get('historical_name',base)
        complete_path=cost/'checkpoints'/name/'complete.json'
        complete=json.loads(complete_path.read_text());verify_outputs(complete)
        if complete['updates']!=4000: raise ValueError('historical endpoint mismatch')
        parents[base]=complete['outputs']['checkpoint'];historical_views[base]=cost/'configs/views.json'
        identities[base+'_complete']=identity(complete_path)
        identities[base+'_parent']=parents[base]
        identities[base+'_machine']=identity(item['machine'])
        identities[base+'_views']=identity(historical_views[base])
    if sha256(historical_views['B1'])!=sha256(historical_views['G']):raise ValueError('historical views differ')
    source_cost=Path(entry['B1']['cost_root'])
    machine=json.loads(Path(entry['B1']['machine']).read_text())
    nuclei=tifffile.imread(machine['nuclei_mask'])
    original=json.loads(historical_views['B1'].read_text());views=json.loads(json.dumps(original))
    for v in views['views']:
        y,x=map(int,re.findall(r'\d+',Path(v['patch']).name)[:2]);ids=np.unique(nuclei[y:y+48,x:x+48]);ids=ids[ids>0]
        if len(ids)!=v['k']:raise ValueError('historical candidate count differs')
        v['cell_ids']=ids.tolist()
    if sum(v['k']>0 for v in views['views'])!=4000:raise ValueError('replay update count')
    views['provenance']=dict(B1=identities['B1_views'],G=identities['G_views'],normalization='added explicit ascending nuclear IDs; visit/order/augmentation unchanged')
    atomic_json(root/'configs/views.json',views)
    for filename in ('patches_shift0.json','patches_shift24.json','reference.json','protocol.json','resolved_native.json','B0_machine.json'):
        shutil.copyfile(source_cost/'configs'/filename,root/'configs'/filename)
    inputs=json.loads((source_cost/'configs/inputs.json').read_text())
    inputs['unknown_training_behavior']='invalid/missing/negative/out-of-range type: POS/NEG excluded with explicit mask and denominator total real K; all other losses unchanged'
    inputs['parents']=parents;inputs['historical_views']=identities['B1_views']
    inputs.pop('checkpoint',None)
    for key in ('baseline_config','nuclei_mask','preannotation','genes','transcripts','reference','positive_markers','negative_markers'):
        path=machine[key];obj=identity(path)
        if key in inputs and isinstance(inputs[key],dict) and 'sha256' in inputs[key] and inputs[key]['sha256']!=obj['sha256']:raise ValueError('input changed '+key)
        inputs[key]=obj
    atomic_json(root/'configs/inputs.json',inputs)
    rng=np.random.RandomState(20261008);probes=[];inventory_count={}
    for shift in (0,24):
        inv=json.loads((root/f'configs/patches_shift{shift}.json').read_text());inventory_count[str(shift)]=len(inv)
        groups={k:[] for k in ('K0','K1','K2_4','K5plus')}
        for i,item in enumerate(inv):
            y,x=map(int,re.findall(r'\d+',Path(item['path']).name)[:2]);ids=np.unique(nuclei[y:y+48,x:x+48]);k=int((ids>0).sum())
            group='K0' if k==0 else 'K1' if k==1 else 'K2_4' if k<=4 else 'K5plus'
            groups[group].append(dict(shift=shift,index=i,patch=item['path'],row=y,col=x,k=k,stratum=group))
        for group,items in groups.items():
            for i in rng.choice(len(items),min(4,len(items)),replace=False): probes.append(items[i])
    atomic_json(root/'configs/probes.json',dict(version='COS-fixed-probe-v1',seed=20261008,selection='four samples per shift/candidate density K stratum before training',probes=probes,denominator=inventory_count,estimated_uncompressed_bytes=sum((p['k']*14+25)*48*48*4 for p in probes),nuclear_owner_coverage_only=True))
    anchors={}
    for name,item in entry.items():
        path=Path(item['evaluation'])/'paper_aligned_v1/annotation.csv'
        if not path.exists():
            candidates=list(Path(item['evaluation']).rglob('*annotation*.csv'))
            if len(candidates)!=1: raise ValueError('ambiguous historical label path '+str(candidates))
            path=candidates[0]
        anchors[name]=identity(path)
    atomic_json(root/'configs/anchors.json',anchors)
    queue=dict(run_id=root.name.removeprefix('crgf_cos_'),output_root=str(root),python=sys.executable,methods={},
        paper_protocol=str(HIST/'configs/evaluation_protocol.json'),prepared=str(HIST/'shared/paper_reference'),
        reference=str(root/'configs/reference.json'),engineering_protocol=str(root/'configs/protocol.json'),
        anchors=anchors,diagnostics_inputs=str(WORK/'outputs/crgf_cos_diagnostic_20261008T115804Z/input_manifest.json'),
        gpu_lock=str(WORK/'outputs/cos_gpu.lock'),minimum_free_gb=8.,cpu_workers=4,
        stage_timeouts=dict(train=21600,predict=43200,postprocess=28800,matrix=7200,evaluate=43200,pair=21600,four=21600),retry_backoff_seconds=30,
        admission=str(root/'admission.json'))
    identities.update({key:identity(root/'configs'/filename) for key,filename in [('views','views.json'),('probe','probes.json'),('labels','anchors.json'),('inputs','inputs.json')]})
    identities['protocol']=identity(queue['paper_protocol']);identities['prepared']=identity(Path(queue['prepared'])/'complete.json')
    identities['plan']=identity(WORK/'docs/BIDCell_COS_FourArm_Overnight_Change_Plan_20261008.md')
    identities['prompt']=identity(WORK/'docs/BIDCell_COS_FourArm_Execution_Prompt_20261008.md')
    for k,v in anchors.items():identities['label_'+k]=v
    for arm in ARMS:
        base=arm.split('_')[0];armroot=root/arm;(armroot/'configs').mkdir(parents=True);(armroot/'source').mkdir()
        cfg=CosConfig(arm_id=arm,base_variant=base,variant=base,cos_enabled=arm.endswith('_COS'),queue_config=str(root/'configs/queue.json'),run_id=queue['run_id'],output_root=str(armroot),baseline_config=machine['baseline_config'],checkpoint=parents[base]['path'],parent_sha256=parents[base]['sha256'])
        for source in (root/'configs').glob('*.json'):shutil.copyfile(source,armroot/'configs'/source.name)
        atomic_json(armroot/'configs/run.json',cfg.model_dump(mode='json'))
        own=dict(machine);own.update(checkpoint=cfg.checkpoint,baseline_final_mask=machine['final_mask'],final_mask=str(armroot/'predictions'/arm/'final.tif'),pre_mask=str(armroot/'predictions'/arm/'pre.tif'),final_matrix=str(armroot/'matrix'/arm/'expr_mat.csv'))
        atomic_json(armroot/f'configs/{arm}_machine.json',own)
        frozen=dict(version='COS-frozen-input-v1',identities={**identities,'config':identity(armroot/'configs/run.json')})
        atomic_json(armroot/'configs/frozen.json',frozen)
        queue['methods'][arm]=dict(config=str(armroot/'configs/run.json'),worktree='',commit='',machine=str(armroot/f'configs/{arm}_machine.json'),matrix=str(armroot/'matrix'/arm),evaluation=str(armroot/'evaluation'/arm),cost_root=str(armroot))
    atomic_json(root/'configs/queue.json',queue)
    atomic_json(root/'source/prepare.json',dict(status='succeeded',identities=identities,parents=parents,views=len(views['views']),updates=4000,empty_views=sum(v['k']==0 for v in views['views']),pool=len(views['train_pool']),prediction_patches=inventory_count,probe_count=len(probes),postprocess_clarification='1024-pixel edge; 48 historical tiles. Model patches remain 48 pixels.',model_assignment=dict(controller='gpt-6.1-sol/high',implementer='gpt-6-astra/high',evaluation_runner='gpt-6-astra/high')))
    print(root,flush=True)
    return root
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output');a=p.parse_args();prepare(a.output)
