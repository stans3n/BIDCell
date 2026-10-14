"""Frozen Morph training ablations, strict snapshots, native full-chain endpoints."""
import copy
import gc
import json
import time
from pathlib import Path
from types import SimpleNamespace
import imgaug as ia
import numpy as np
import torch
import tifffile
from bidcell.model.model.morphology_variants import (
    VARIANTS, CANONICAL_SHA, canonical_full, MorphologyVariant, initialization_map, tensor_sha)
from bidcell.model.utils.utils import get_seg_mask
from scripts.new_methods.common import BaseSix, make_model as legacy_model, seed_all, optimizer
from scripts.new_methods.data import NativeViews
from scripts.new_methods.prediction import merge_native
from scripts.crgf.common import atomic_torch, atomic_npy, atomic_tiff, verify_outputs
from scripts.paired_v2.common import rng_state, restore_rng, state_equal
from scripts.paired_v2.training import sequence_identity, rewrite_sequence
from .common import (load_run, output_path, endpoint_spec, identity, check_identity, sha256,
    digest, read, atomic_json, progress, stop_requested, compute_lock, write_complete,
    validate_complete, receipt_dependencies, inventory)

NODES={'MORPH_NO_DAPI':'no_dapi','MORPH_SIMPLE_FUSION':'simple_fusion','MORPH_NO_GEO':'no_geo'}
REPO=Path(__file__).resolve().parents[2]
CHECKS=(2,32,64,100)


def scientific_source():
    paths=list((REPO/'bidcell').rglob('*.py'))+[REPO/p for p in (
        'scripts/boundary_three/morph.py','scripts/boundary_three/morph_mini.py',
        'scripts/new_methods/common.py','scripts/new_methods/data.py',
        'scripts/new_methods/prediction.py','scripts/crgf/postprocess.py',
        'scripts/paired_v2/common.py','scripts/paired_v2/training.py')]
    return {str(p.relative_to(REPO)):sha256(p) for p in paths if p.exists()}


def variant_spec(node):
    if node=='V2_MORPH':return dict(VARIANTS['full'])
    return dict(VARIANTS[NODES[node]])


def make_model(node,initial,device='cuda',canonical=None):
    model=legacy_model('BASE',initial,'cpu')
    original=canonical if canonical is not None else canonical_full()
    model.morphology=original if node=='V2_MORPH' else MorphologyVariant(NODES[node],original)
    model.morph_spec=variant_spec(node)
    return model.to(device)


def dataset(cfg,node,training=True,shift=0):
    spec=variant_spec(node)
    return NativeViews(cfg['inputs']['historical_input_root'],training=training,shift=shift,
        auxiliary=True,morphology=True,use_geometry=spec['use_geometry'],use_dapi=spec['use_dapi'])


def forward(model,batch,node):
    spec=variant_spec(node);aux=batch['aux']
    for key,enabled in [('geometry',spec['use_geometry']),('dapi',spec['use_dapi'])]:
        if (key in aux)!=enabled:raise ValueError('train/predict auxiliary variant mismatch: '+key)
    return model(batch['input'],aux)


def setup(cfg,node,device='cuda'):
    ia.random.get_global_rng()
    model=make_model(node,cfg['inputs']['shared_initial'],device)
    opt=optimizer(model);ia.seed(1234)
    data=dataset(cfg,node)
    return model,opt,data


def initial_header(cfg,node):
    return dict(version='MORPH-ABLATION-20261015-v1.0',node=node,**variant_spec(node),
        source=scientific_source(),initial=identity(cfg['inputs']['shared_initial']),
        views=identity(cfg['inputs']['views']),prepared=identity(Path(cfg['inputs']['historical_input_root'])/'prepared/manifest.json'),
        canonical_tensor_sha256=CANONICAL_SHA,seed=1234,canonical_seed=20261013,new_convolution_seed=2026101501,
        precision='FP32; no AMP; no TF32; full K; original BN',
        optimizer=dict(name='Adam',lr=1e-5,betas=[.9,.999],eps=1e-8,weight_decay=1e-4))


def snapshot(model,opt,header,updates=0,cursor=0,skips=0,sequence=None,pilot=False):
    sequence=[] if sequence is None else sequence
    return dict(header=header,model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state(),
        actual_updates=updates,view_cursor=cursor,empty_skips=skips,effective_sequence=sequence,
        sequence_digest=digest(sequence),bn_in_model_state=True,pilot=pilot)


def validate_cursor(cfg,state):
    views=read(cfg['inputs']['views'])['views'];cursor=state['view_cursor']
    if not 0<=cursor<=len(views):raise ValueError('cursor outside frozen visits')
    used=views[:cursor];seq=state['effective_sequence']
    if state['actual_updates']!=sum(v['K']>0 for v in used) or state['empty_skips']!=sum(v['K']==0 for v in used):
        raise ValueError('updates/visits/empty mismatch')
    if len(seq)!=state['actual_updates'] or state['sequence_digest']!=digest(seq):raise ValueError('sequence digest/length')
    hashes={v['path']:v['identity']['sha256'] for v in read(cfg['inputs']['predict_shift0'])}
    for n,(v,r) in enumerate(zip((v for v in used if v['K']),seq),1):
        if any(r.get(k)!=x for k,x in sequence_identity(v,n,hashes[v['path']]).items()):raise ValueError('frozen trajectory mismatch')


def restore(cfg,path,model,opt,header,pilot=False):
    state=torch.load(path,map_location='cpu')
    if state['header']!=header or state['pilot']!=pilot:raise ValueError('checkpoint variant/source/input/pilot mismatch')
    validate_cursor(cfg,state)
    model.load_state_dict(state['model'],strict=True);opt.load_state_dict(state['optimizer']);restore_rng(state['rng'])
    return state


def prepare_morph(config):
    """Call before final run source/config freeze; never load a trained endpoint."""
    cfg=load_run(config);dest=output_path(cfg,'prepared','morph');dest.mkdir(parents=True,exist_ok=True)
    original=canonical_full();path=dest/'canonical_full_seed20261013.pth'
    if path.exists():
        saved=torch.load(path,map_location='cpu')
        if tensor_sha(saved['model'])!=CANONICAL_SHA:raise ValueError('canonical full archive drift')
    else:atomic_torch(path,dict(model=original.state_dict(),tensor_sha256=CANONICAL_SHA,seed=20261013,
        limitation='frozen-code reconstruction; not independently archived historical step zero'))
    audit=read(cfg['inputs']['pairing_audit'])
    if audit['reconstructed_morphology_initial']['state_tensor_sha256']!=CANONICAL_SHA:raise ValueError('historical reconstruction identity')
    for key in ('prepared/shared_initial.pth','configs/views.json','prepared/manifest.json'):
        obj=audit['inputs'][key]
        if sha256(obj['path'])!=obj['sha256']:raise ValueError('historical paired input identity changed')
    for node in NODES:
        root=output_path(cfg,node);root.mkdir(parents=True,exist_ok=True)
        header=initial_header(cfg,node);model,opt,data=setup(cfg,node,'cpu');initial=root/'step_zero.pth'
        mapping=initialization_map(model.morphology,original)
        if initial.exists():
            saved=torch.load(initial,map_location='cpu')
            if saved['header']!=header or saved['actual_updates'] or saved['optimizer']['state']:raise ValueError('step-zero drift')
            if not state_equal(saved['model'],model.state_dict()):raise ValueError('step-zero tensor drift')
            del saved
        else:atomic_torch(initial,snapshot(model,opt,header))
        atomic_json(root/'initialization_map.json',dict(canonical=identity(path),tensors=mapping,
            total_parameters=sum(p.numel() for p in model.parameters()),
            added_parameters=sum(p.numel() for p in model.morphology.parameters()),
            actual_registered_trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            absent_modules=[n for n in ('image','geometry') if not hasattr(model.morphology,n)],
            historical_initial_limitation='reconstructed canonical full, checked against prior recorded hash'))
        atomic_json(root/'config.json',dict(node=node,**variant_spec(node),updates=3625,visits=3998,empty=373,
            initialization=identity(initial),postprocess='native',p2=False,a2=False))
        spec=endpoint_spec(cfg,node);machine=read(endpoint_spec(cfg,'BASE_MATCH')['machine'])
        machine.update(pre_mask=spec['pre'],final_mask=spec['final'],final_matrix=str(Path(spec['matrix'])/'expr_mat.csv'),checkpoint=spec['checkpoint'])
        atomic_json(spec['machine'],machine)
        del model,opt,data;gc.collect()
    rec=dict(canonical=identity(path),canonical_tensor_sha256=CANONICAL_SHA,nodes=list(NODES))
    atomic_json(dest/'complete.json',rec);return rec


def training_header(cfg,node):
    return dict(**initial_header(cfg,node),run_dependencies=receipt_dependencies(cfg,node),
        initial_snapshot=identity(output_path(cfg,node,'step_zero.pth')))


def gradient_groups(model):
    result={}
    for name,p in model.morphology.named_parameters():
        group=name.split('.')[0];r=result.setdefault(group,dict(parameters=0,gradient_parameters=0,nonzero_gradient_parameters=0,squared_norm=0.))
        r['parameters']+=p.numel()
        if p.grad is not None:
            r['gradient_parameters']+=p.numel();r['nonzero_gradient_parameters']+=int(torch.count_nonzero(p.grad))
            r['squared_norm']+=float(p.grad.double().square().sum())
    for r in result.values():r['norm']=r.pop('squared_norm')**.5
    return result


def effect_probe(model,batch,node):
    """Same weights/BN/RNG; temporarily omit only the learned residual."""
    before=tensor_sha(model.state_dict());saved=rng_state();mode=model.training;model.eval()
    morph=model.morphology;captured={}
    def hook(module,inputs,out):
        captured['delta_h3_norm']=float((out-inputs[0]).double().norm())
        captured['h3_norm']=float(inputs[0].double().norm())
    handle=morph.register_forward_hook(hook)
    try:
        with torch.no_grad():
            z=forward(model,batch,node);handle.remove();delattr(model,'morphology')
            baseline=model(batch['input']);model.morphology=morph
            changed=get_seg_mask(z.cpu().numpy(),batch['labels'].cpu().numpy())!=get_seg_mask(baseline.cpu().numpy(),batch['labels'].cpu().numpy())
        captured.update(logit_max_abs_delta=float((z-baseline).abs().max()),logit_l2_delta=float((z-baseline).double().norm()),
            owner_changed_pixels=int(changed.sum()),pixels=int(changed.size))
    finally:
        handle.remove();model.morphology=morph;model.train(mode);restore_rng(saved)
    if tensor_sha(model.state_dict())!=before:raise ValueError('diagnostic changed model/BN')
    return captured


def update(model,opt,data,base,view,node):
    start=time.monotonic();batch=data.get_view(view,'cuda')
    opt.zero_grad(set_to_none=True);z=forward(model,batch,node);loss,parts=base(z,batch)
    if not torch.isfinite(z).all() or not torch.isfinite(loss):raise FloatingPointError('nonfinite logits/loss')
    loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError('nonfinite gradient')
    groups=gradient_groups(model);opt.step();torch.cuda.synchronize()
    return batch,dict(base6=float(loss.detach()),base_terms={k:float(v.detach()) for k,v in parts.items()},
        seconds=time.monotonic()-start,gradient_groups=groups)


def train(config,node):
    cfg=load_run(config);root=output_path(cfg,node);out=root/'checkpoints';out.mkdir(exist_ok=True)
    header=training_header(cfg,node);model,opt,data=setup(cfg,node)
    initial=torch.load(root/'step_zero.pth',map_location='cpu')
    if initial['header']!=initial_header(cfg,node) or initial['actual_updates'] or initial['optimizer']['state']:raise ValueError('invalid formal initial')
    model.load_state_dict(initial['model']);opt.load_state_dict(initial['optimizer']);restore_rng(initial['rng']);del initial
    state=dict(actual_updates=0,view_cursor=0,empty_skips=0,effective_sequence=[])
    if (out/'latest.pth').exists():state=restore(cfg,out/'latest.pth',model,opt,header)
    updates,cursor,skips,seq=(state[k] for k in ('actual_updates','view_cursor','empty_skips','effective_sequence'));del state
    views=read(cfg['inputs']['views'])['views'];hashes={v['path']:v['identity']['sha256'] for v in read(cfg['inputs']['predict_shift0'])}
    log=root/'effective_sequence.jsonl';rewrite_sequence(log,seq);base=BaseSix('cuda');model.train();started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    while cursor<len(views):
        if stop_requested(cfg):
            atomic_torch(out/'latest.pth',snapshot(model,opt,header,updates,cursor,skips,seq))
            raise InterruptedError('safe Morph training stop; exact state preserved')
        view=views[cursor];cursor+=1
        if not view['K']:
            # Preserve native data visit and augmentation RNG, even for empty groups.
            data.get_view(view,'cuda');skips+=1;continue
        batch,row=update(model,opt,data,base,view,node);updates+=1
        row.update(sequence_identity(view,updates,hashes[view['path']]));row['empty_skips']=skips
        if updates in CHECKS:
            row['effect_probe']=effect_probe(model,batch,node)
            atomic_json(root/'train_receipts'/f'update_{updates:04d}.json',row)
        seq.append(row)
        with log.open('a') as f:f.write(json.dumps(row)+'\n')
        if updates%25==0 or updates==3625:atomic_torch(out/'latest.pth',snapshot(model,opt,header,updates,cursor,skips,seq))
        progress(cfg,node,'training',actual_updates=updates,view_cursor=cursor,empty_skips=skips,total_updates=3625)
        if updates%10==0:print(node,updates,'/3625',flush=True)
        del batch
    final=snapshot(model,opt,header,updates,cursor,skips,seq);validate_cursor(cfg,final)
    if (updates,cursor,skips)!=(3625,3998,373):raise ValueError('formal frozen budget differs')
    atomic_torch(out/'final.pth',final)
    rec=dict(status='succeeded',dependencies=header,actual_updates=updates,view_cursor=cursor,empty_skips=skips,
        wall_seconds=time.monotonic()-started,peak_vram_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),outputs={'checkpoint':identity(out/'final.pth'),'sequence':identity(log)})
    atomic_json(out/'complete.json',rec);return rec


def predict(config,node):
    cfg=load_run(config);spec=endpoint_spec(cfg,node);out=Path(spec['pre']).parent;out.mkdir(parents=True,exist_ok=True)
    trained=read(output_path(cfg,node,'checkpoints/complete.json'));verify_outputs(trained)
    state=torch.load(spec['checkpoint'],map_location='cpu')
    if state['header']!=training_header(cfg,node) or state['pilot']:raise ValueError('prediction checkpoint contract')
    validate_cursor(cfg,state)
    if (state['actual_updates'],state['view_cursor'],state['empty_skips'])!=(3625,3998,373):raise ValueError('prediction budget')
    model=make_model(node,cfg['inputs']['shared_initial']);model.load_state_dict(state['model']);model.eval();del state
    dep=dict(checkpoint=identity(spec['checkpoint']),source=scientific_source(),variant=variant_spec(node))
    maps=[];receipts=[];started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    for shift in (0,24):
        data=dataset(cfg,node,False,shift);pool=read(cfg['inputs'][f'predict_shift{shift}'])
        if list(map(str,data.fp_patches))!=[v['path'] for v in pool]:raise ValueError('frozen predict pool drift')
        whole=np.zeros(data.nuclei.shape,np.uint32);chunks=out/f'shift{shift}_patches';chunks.mkdir(exist_ok=True)
        for start in range(0,len(pool),128):
            if stop_requested(cfg):raise InterruptedError('safe Morph prediction chunk boundary')
            stop=min(start+128,len(pool));path=chunks/f'{start:06d}.npy';recpath=path.with_suffix('.json')
            chunkdep=dict(base=dep,start=start,stop=stop,patches=[identity(v['path']) for v in pool[start:stop]])
            if recpath.exists():
                rec=read(recpath)
                if rec['dependencies']!=chunkdep:raise ValueError('prediction chunk identity drift')
                verify_outputs(rec);masks=np.load(path)
            else:
                masks=np.zeros((stop-start,48,48),np.uint32)
                with torch.inference_mode():
                    for j,v in enumerate(pool[start:stop]):
                        b=data.get_view(dict(index=v['index']),'cuda')
                        if len(b['input']):
                            z=forward(model,b,node)
                            if not torch.isfinite(z).all():raise FloatingPointError('prediction nonfinite')
                            masks[j]=get_seg_mask(z.cpu().numpy(),b['labels'].cpu().numpy()).astype(np.uint32)
                atomic_npy(path,masks);atomic_json(recpath,dict(status='succeeded',dependencies=chunkdep,outputs={'masks':identity(path)}))
            for j,v in enumerate(pool[start:stop]):whole[v['row']:v['row']+48,v['col']:v['col']+48]=masks[j]
            receipts.append(identity(recpath));progress(cfg,node,'prediction',shift=shift,completed_tiles=stop,total_tiles=len(pool))
        atomic_tiff(out/f'shift{shift}.tif',whole);maps.append(whole)
        if shift==24:shifted=list(data.fp_patches)
        del data
    pre,fill=merge_native(*maps,shifted);atomic_tiff(out/'pre.tif',pre);atomic_tiff(out/'fill.tif',fill)
    rec=dict(status='succeeded',dependencies=dep,shifts=[0,24],chunks=receipts,wall_seconds=time.monotonic()-started,
        peak_vram_bytes=torch.cuda.max_memory_allocated(),outputs={p.name:identity(p) for p in [out/'pre.tif',out/'shift0.tif',out/'shift24.tif',out/'fill.tif']})
    atomic_json(out/'complete.json',rec);return rec


def native_postprocess(cfg,node,spec=None):
    from scripts.crgf.postprocess import postprocess
    spec=spec or endpoint_spec(cfg,node);machine=read(spec['machine']);root=Path(spec.get('cost_root',spec['root']))
    adapter=SimpleNamespace(output_root=str(root),seed=1234,postprocess_size=1024,cpu_workers=1,
        native=lambda:SimpleNamespace(files=SimpleNamespace(data_dir=str(Path(machine['nuclei_mask']).parent))))
    postprocess(adapter,node)
    final=tifffile.imread(machine['final_mask']);nuc=tifffile.imread(machine['nuclei_mask'])
    if not np.array_equal(final[nuc>0],nuc[nuc>0]):raise ValueError('native nuclear restoration failed')
    return read(Path(machine['pre_mask']).parent/'postprocess/complete.json')


def run_node(config,node):
    cfg=load_run(config)
    if node not in NODES:raise ValueError(node)
    root=output_path(cfg,node);started=time.monotonic()
    if (root/'complete.json').exists():return validate_node(cfg,node)
    if not (root/'step_zero.pth').exists():raise ValueError('prepare_morph must run before frozen admission')
    stages={}
    if (root/'checkpoints/complete.json').exists():
        stages['training']=read(root/'checkpoints/complete.json');verify_outputs(stages['training'])
        if stages['training']['dependencies']!=training_header(cfg,node):raise ValueError('training receipt identity')
    else:stages['training']=train(cfg,node)
    gc.collect();torch.cuda.empty_cache()
    if (Path(endpoint_spec(cfg,node)['pre']).parent/'complete.json').exists():
        stages['prediction']=read(Path(endpoint_spec(cfg,node)['pre']).parent/'complete.json');verify_outputs(stages['prediction'])
        expected=dict(checkpoint=identity(endpoint_spec(cfg,node)['checkpoint']),source=scientific_source(),variant=variant_spec(node))
        if stages['prediction']['dependencies']!=expected:raise ValueError('completed prediction identity drift')
    else:stages['prediction']=predict(cfg,node)
    gc.collect();torch.cuda.empty_cache();progress(cfg,node,'native_postprocess')
    stages['postprocess']=native_postprocess(cfg,node)
    from .evaluation import build_and_evaluate,validate_endpoint
    progress(cfg,node,'matrix_and_evaluation');stages['evaluation']=build_and_evaluate(cfg,node)
    validate_endpoint(cfg,node)
    from .reports import compare_endpoints
    for control in ('BASE_MATCH','V2_MORPH'):compare_endpoints(cfg,node,control,out=root/f'comparison_{control}')
    # Required top-level artifacts contain the actual pairwise tables, not empty markers.
    import pandas as pd
    strata=[];pairs=[];alias_sources={}
    for control in ('BASE_MATCH','V2_MORPH'):
        dest=root/f'comparison_{control}'
        for name,collection in [('strata_endpoint_values.csv',strata),('paired_deltas.csv',pairs)]:
            p=dest/name
            frame=pd.read_csv(p);frame.insert(0,'control_endpoint',control);frame.insert(0,'source_table',p.name)
            collection.append(frame);alias_sources[control+'/'+name]=identity(p)
    if not strata or not pairs:raise ValueError('shared report missing required pair/strata tables')
    pd.concat(strata,ignore_index=True).to_csv(root/'boundary_strata.csv',index=False)
    pd.concat(pairs,ignore_index=True).to_csv(root/'paired_comparisons.csv',index=False)
    atomic_json(root/'comparison_alias_provenance.json',alias_sources)
    atomic_json(root/'runtime_cost.json',dict(wall_seconds_this_invocation=time.monotonic()-started,stages=stages))
    write_complete(cfg,node,'MEASURED_AWAITING_INDEPENDENT_ANALYSIS',morph_variant=variant_spec(node))
    return validate_node(cfg,node)


def validate_node(config,node):
    cfg=load_run(config);rec=validate_complete(cfg,node)
    if rec['morph_variant']!=variant_spec(node):raise ValueError('completed Morph variant differs')
    root=output_path(cfg,node);trainrec=read(root/'checkpoints/complete.json')
    if trainrec['dependencies']!=training_header(cfg,node) or (trainrec['actual_updates'],trainrec['view_cursor'],trainrec['empty_skips'])!=(3625,3998,373):raise ValueError('formal train contract incomplete')
    from .evaluation import validate_endpoint
    validate_endpoint(cfg,node)
    from .reports import validate_comparison
    for control in ('BASE_MATCH','V2_MORPH'):validate_comparison(cfg,root/f'comparison_{control}')
    for name in ('step_zero.pth','initialization_map.json','boundary_strata.csv','paired_comparisons.csv','runtime_cost.json'):
        if not (root/name).is_file():raise ValueError('required Morph artifact absent '+name)
    return rec


def validate_mini(config,output):
    from .morph_mini import validate_mini as run
    return run(config,output)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','mini','run','validate']);p.add_argument('--config',required=True);p.add_argument('--node',choices=list(NODES));p.add_argument('--output')
    a=p.parse_args()
    if a.action=='prepare':result=prepare_morph(a.config)
    elif a.action=='mini':result=validate_mini(a.config,a.output)
    elif a.action=='run':result=run_node(a.config,a.node)
    else:result=validate_node(a.config,a.node)
    print(json.dumps(result,default=str))
