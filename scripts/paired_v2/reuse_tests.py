"""Historical full-chain provenance and fixed-ROI interface replay evidence."""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
from .common import *


def historical_provenance(root):
    root=Path(root);out=root/'tests/historical';out.mkdir(parents=True,exist_ok=True)
    arm=INPUT/'V2_MORPH';files={};producer_maps={}
    def verify_tree(value):
        if isinstance(value,dict):
            if 'path' in value and 'sha256' in value:
                p=Path(value['path'])
                if str(p) not in files:
                    if sha256(p)!=value['sha256']:raise ValueError('historical artifact changed: '+str(p))
                    files[str(p)]=identity(p)
            for key,item in value.items():
                if key in ('source','producer') and isinstance(item,dict) and item and all(isinstance(v,str) and len(v)==64 for v in item.values()):
                    for name,sha in item.items():
                        if sha256(HISTORICAL_SOURCE/name)!=sha:raise ValueError('historical producer changed: '+name)
                    producer_maps[digest(item)]=item
                verify_tree(item)
        elif isinstance(value,list):
            for item in value:verify_tree(item)
    paths={'train':arm/'checkpoints/complete.json','predict':arm/'predictions/V2_MORPH/complete.json',
        'postprocess':arm/'predictions/V2_MORPH/postprocess/complete.json','matrix':arm/'matrix/V2_MORPH/complete.json',
        'evaluate':arm/'evaluation/V2_MORPH/complete.json','authoritative_summary':AUTHORITY.parent/'methods/V2_MORPH/summary.json'}
    receipts={}
    for stage,path in paths.items():
        rec=read(path)
        if stage!='authoritative_summary' and rec['status']!='succeeded':raise ValueError('historical stage incomplete')
        verify_tree(rec);files[str(path)]=identity(path);receipts[stage]=rec
    state=torch.load(arm/'checkpoints/final.pth',map_location='cpu')
    if state['header']!=receipts['train']['dependencies'] or (state['actual_updates'],state['view_cursor'],state['empty_skips'])!=(3625,3998,373):raise ValueError('historical train state/header mismatch')
    expected=read(AUDIT)
    if sha256(arm/'checkpoints/final.pth')!=expected['arms']['V2_MORPH']['checkpoint_sha256']:raise ValueError('historical endpoint changed')
    logs=[json.loads(line) for line in (arm/'training.jsonl').read_text().splitlines()];logs=[r for r in logs if r['event']=='update']
    views=[v for v in read(INPUT/'configs/views.json')['views'] if v['K']]
    if len(logs)!=3625:raise ValueError('historical update log incomplete')
    for n,(v,r) in enumerate(zip(views,logs),1):
        if [r[k] for k in ('updates','cursor','nominal_index','K','flip','rot90','xy')]!=[n,v['cursor']+1,v['index'],v['K'],v['flip'],v['rot90'],[v['row'],v['col']]]:raise ValueError('historical visits diverge')
    if receipts['predict']['dependencies']['checkpoint']['sha256']!=sha256(arm/'checkpoints/final.pth') or receipts['predict']['shifts']!=[0,24]:raise ValueError('historical prediction endpoint/shift linkage differs')
    for obj in receipts['predict']['chunks']:
        chunk=read(obj['path']);verify_tree(chunk)
    coverage={}
    for shift in (0,24):
        pool=read(INPUT/f'configs/predict_shift{shift}.json');covered=0
        for start in range(0,len(pool),128):
            path=arm/'predictions/V2_MORPH'/f'shift{shift}_patches'/f'{start:06d}.json';rec=read(path)
            if rec['dependencies']['start']!=start or rec['dependencies']['stop']!=min(start+128,len(pool)):raise ValueError('historical prediction coverage mismatch')
            if rec['dependencies']['patches']!=[v['identity'] for v in pool[start:start+128]]:raise ValueError('historical prediction patch identity differs')
            covered+=min(128,len(pool)-start)
        coverage[str(shift)]=covered
    post=read(arm/'predictions/V2_MORPH/postprocess/manifest.json');verify_tree(post)
    for obj in post['tiles']:verify_tree(read(obj['path']))
    for item in post['split_chunks']:verify_tree(read(item['receipt']['path']))
    shape=read(INPUT/'prepared/manifest.json')['shape'];h,w=shape
    coords=[[y,x] for y in list(range(0,h-1024,1024))+[h-1024] for x in list(range(0,w-1024,1024))+[w-1024]]
    if post['dependencies']['seed']!=1234 or post['dependencies']['patch_size']!=1024 or post['coords']!=coords or post['split_ids']!=sorted(set(post['split_ids'])):raise ValueError('historical native postprocess config/order differs')
    if [r['ids'] for r in post['split_chunks']]!=[post['split_ids'][s:s+128] for s in range(0,len(post['split_ids']),128)]:raise ValueError('historical split order/chunks differ')
    matrix=receipts['matrix'];matrix_dir=arm/'matrix/V2_MORPH'
    if matrix['dependencies']['mask']['sha256']!=receipts['postprocess']['outputs']['final_mask']['sha256']:raise ValueError('historical own matrix mask linkage differs')
    if matrix['contract']['source_sha256']!=sha256(HISTORICAL_SOURCE/'scripts/crgf/matrix.py'):raise ValueError('historical matrix producer differs')
    for chunk in matrix['receipts']:
        for path,sha in ((matrix_dir/chunk['owner_file'],chunk['sha256']),(Path(chunk['coordinate_file']),chunk['coordinate_sha256'])):
            if sha256(path)!=sha:raise ValueError('historical matrix owner/coordinate differs')
            files[str(path)]=identity(path)
    counts=np.load(matrix_dir/'counts.npy',mmap_mode='r');ids=pd.read_csv(matrix_dir/'cell_mapping.csv').canonical_cell_id.to_numpy()
    if counts.shape!=(107955,313) or not np.issubdtype(counts.dtype,np.integer) or (counts<0).any() or int(counts.sum())!=matrix['totals']['assigned'] or matrix['totals']['eligible']!=34442716:raise ValueError('historical matrix semantic check failed')
    if not np.array_equal(ids,np.load(INPUT/'prepared/nucleus_metadata.npz')['ids']):raise ValueError('historical complete nucleus-ID set differs')
    from .evaluation import evaluation_contract
    _,_,science=evaluation_contract(root)
    if read(arm/'evaluation/V2_MORPH/summary.json')['model_sha256']!=science['reference_model']['sha256']:raise ValueError('historical sc model differs')
    result=dict(status='passed',scope='full existing endpoint provenance/hash and scientific configuration; no whole-slide prediction replay claim',
        stages={k:identity(v) for k,v in paths.items()},producer_maps=producer_maps,files=files,files_checked=len(files),
        sequence_updates_checked=3625,prediction_coverage=coverage,postprocess_tiles=len(coords),split_chunks=len(post['split_chunks']),
        matrix_shape=[107955,313],matrix_totals=matrix['totals'],evaluation_contract=science,
        inherited_limits=read(AUDIT)['limits'])
    atomic_json(out/'provenance.json',result);return result


def roi(root):
    from scripts.crgf.matrix import build
    from .evaluation import evaluation_contract,validate_evaluation
    from scripts.new_methods.accepted_profiles import run_profiles
    root=Path(root);out=root/'tests/roi';shared=out/'shared';shared.mkdir(parents=True,exist_ok=True)
    existing=read(INPUT/'admission/mini/complete.json');upstream={}
    for name in ('nuclei.tif','roi.json','transcripts_processed.csv'):
        obj=existing['outputs']['shared/'+name]
        if sha256(obj['path'])!=obj['sha256']:raise ValueError('frozen ROI source changed')
        shutil.copyfile(obj['path'],shared/name);upstream[name]=obj
    definition=read(shared/'roi.json');full=read(INPUT/'V2_MORPH/configs/machine.json');nuc=tifffile.imread(full['nuclei_mask'])
    y,x,size=definition['row'],definition['col'],definition['size'];local=tifffile.imread(shared/'nuclei.tif')
    if not np.array_equal(local,nuc[y:y+size,x:x+size]):raise ValueError('ROI nuclei differ from immutable global mask')
    del nuc
    executions=[('legacy','V2_MORPH',INPUT/'V2_MORPH/checkpoints/final.pth','historical_legacy'),
                ('paired','V2_MORPH',INPUT/'V2_MORPH/checkpoints/final.pth','historical_paired')]
    executions += [('paired',run,root/'tests/pilots'/run/'checkpoints/final.pth',run) for run in RUNS]
    for source,run,ckpt,folder in executions:
        dest=out/folder;dest.mkdir(exist_ok=True)
        with (dest/'worker.log').open('w') as log:
            subprocess.run([sys.executable,str(REPO/'scripts/paired_v2/roi_worker.py'),'--source',source,'--run-id',run,
                '--checkpoint',str(ckpt),'--shared',str(shared),'--out',str(dest)],stdout=log,stderr=subprocess.STDOUT,check=True)
    old=read(out/'historical_legacy/prediction_complete.json');new=read(out/'historical_paired/prediction_complete.json');comparisons=[]
    for os_,ns_ in zip(old['patches'],new['patches']):
        if os_['shift']!=ns_['shift'] or len(os_['records'])!=len(ns_['records']):raise ValueError('ROI shift coverage mismatch')
        for a,b in zip(os_['records'],ns_['records']):
            if a['view']!=b['view'] or a['candidate_ids']!=b['candidate_ids']:raise ValueError('ROI candidate order differs')
            az,bz=np.load(a['logits']['path']),np.load(b['logits']['path'])
            if not np.array_equal(az,bz):raise ValueError('same-checkpoint ROI logits differ')
            comparisons.append(dict(shift=os_['shift'],view=a['view'],candidate_ids=a['candidate_ids'],logits_exact=True))
    for name in ('shift0.tif','shift24.tif','pre.tif','fill.tif','final.tif'):
        if not np.array_equal(tifffile.imread(old['outputs'][name]['path']),tifffile.imread(new['outputs'][name]['path'])):raise ValueError('ROI discrete result differs: '+name)
    op=read(out/'historical_legacy/predictions/V2_MORPH/postprocess/manifest.json');np_=read(out/'historical_paired/predictions/V2_MORPH/postprocess/manifest.json')
    for key in ('coords','split_ids','split_order'):
        if op[key]!=np_[key]:raise ValueError('ROI postprocess split/order mismatch')
    queue,protocol,science=evaluation_contract(root);summaries={}
    for run in RUNS:
        dest=out/run;pred=dest/'predictions'/run
        machine=dict(full,nuclei_mask=str(shared/'nuclei.tif'),transcripts=str(shared/'transcripts_processed.csv'),
            pre_mask=str(pred/'pre.tif'),final_mask=str(pred/'final.tif'),baseline_final_mask=str(pred/'pre.tif'),
            final_matrix=str(dest/'matrix'/run/'expr_mat.csv'))
        atomic_json(dest/'machine.json',machine)
        matrix=build(machine,dest/'matrix'/run,cache=shared/'coordinate_cache')
        if matrix['totals']['eligible']!=18469:raise ValueError('ROI eligible molecule count changed')
        spec=dict(machine=str(dest/'machine.json'),matrix=str(dest/'matrix'/run),evaluation=str(dest/'evaluation'/run),cost_root=str(dest))
        Path(spec['evaluation']).mkdir(parents=True,exist_ok=True)
        summary=run_profiles(spec,queue,str(protocol),spec['evaluation'],run)
        validate_evaluation(spec);summaries[run]=dict(summary=summary,matrix=identity(dest/'matrix'/run/'complete.json'),
            independent_validation=identity(dest/'matrix'/run/'validation.json'),evaluation_contract=science)
    result=dict(status='passed',scope='fixed 144x144 ROI only; legacy/new E1 interface equality is not whole-slide replay',
        roi=definition,roi_sources=upstream,historical_same_checkpoint=identity(INPUT/'V2_MORPH/checkpoints/final.pth'),
        historical_candidates_logits=comparisons,discrete_shift_merge_pre_final_exact=True,postprocess_order_exact=True,
        new_arm_own_matrix_scclassify=summaries,outputs=file_tree(out))
    atomic_json(out/'complete.json',json_safe(result));return result


def finish(root):
    root=Path(root);provenance=read(root/'tests/historical/provenance.json');rr=read(root/'tests/roi/complete.json');scores=read(root/'tests/historical_scores/complete.json')
    if any(r['status']!='passed' for r in (provenance,rr,scores)):raise ValueError('historical reuse incomplete')
    result=dict(status='passed_implementation_gate_pending_independent_review',endpoint='V2_MORPH',
        provenance=identity(root/'tests/historical/provenance.json'),roi_interface_replay=identity(root/'tests/roi/complete.json'),
        authoritative_score_replay=identity(root/'tests/historical_scores/complete.json'),
        checkpoint=identity(INPUT/'V2_MORPH/checkpoints/final.pth'),
        whole_slide_replayed=False,full_existing_chain_provenance_verified=True,
        limits=read(AUDIT)['limits']+['The fixed ROI validates the new interface only, not a recomputed whole-slide endpoint.'])
    atomic_json(root/'reviews/historical_endpoint_reuse.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('test',choices=['provenance','roi','scores','finish']);p.add_argument('--root',type=Path,default=DEFAULT_ROOT);a=p.parse_args()
    if a.test=='provenance':r=historical_provenance(a.root)
    elif a.test=='roi':r=roi(a.root)
    elif a.test=='scores':
        from .evaluation import replay_historical_scores
        r=replay_historical_scores(a.root/'tests/historical_scores')
    else:r=finish(a.root)
    print(json.dumps(dict(status=r['status'],test=a.test),indent=2))


if __name__=='__main__':main()
