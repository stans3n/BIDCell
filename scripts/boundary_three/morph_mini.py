"""Real isolated Morph admission; never supplies a formal training initializer."""
import copy
import gc
import importlib.util
import resource
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import tifffile
import torch
from . import morph as m
from .common import load_run,read,identity,atomic_json,compute_lock,inventory,check_identity
from .evaluation import build_and_evaluate
from scripts.crgf.common import atomic_torch,atomic_tiff
from scripts.crgf.matrix import grid,lookup_axes
from scripts.new_methods.data import transform,transform_matrix
from scripts.new_methods.common import make_model as legacy_model
from tests.new_methods.contracts import compare_tensors,adam_formula

def historical_model(initial,device='cuda'):
    # Match the new constructor's frozen seed/thread/FP32 backend setup. Without
    # this, an OMP_NUM_THREADS=1 launcher compares a one-thread historical CPU
    # forward with the new constructor's explicit four-thread CPU forward.
    m.seed_all()
    source=Path('/home/zwan0726/work/projects/BIDCell_new_methods_20261013/bidcell/model/model')
    modules={}
    for name in ('model','morphology'):
        spec=importlib.util.spec_from_file_location('bidcell.model.model.historical_'+name,source/(name+'.py'))
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);modules[name]=module
    model=modules['model'].SegmentationModel(n_channels=313)
    model.morphology=modules['morphology'].MorphologyModule(chunk_size=0)
    return model.to(device)


def cpu_copy(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_copy(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_copy(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def augmentation_checks(data,view):
    plain=dict(view,flip=0,rot90=0);base=data.get_view(plain)
    evidence=data.evidence;records=[]
    for flip in (0,1):
        for rot in range(4):
            batch=data.get_view(dict(view,flip=flip,rot90=rot));aux=batch['aux'];R=transform_matrix(flip,rot)
            if 'dapi' in aux:
                expected=transform(base['aux']['dapi'][0].numpy().transpose(1,2,0),flip,rot).transpose(2,0,1)[None]
                assert np.array_equal(aux['dapi'].numpy(),expected)
            if 'geometry' in aux:
                for i,cid in enumerate(view['candidate_ids']):
                    row=evidence.id_to_row[cid];cov=R@evidence.meta['covariance'][row]@R.T;trace=np.trace(cov)
                    if trace>1e-12:np.testing.assert_allclose(aux['geometry'][i,[7,8,9],0,0],[cov[0,0]/trace,cov[0,1]/trace,cov[1,1]/trace],atol=1e-7)
                    for c in (0,1,2,3,6,10,15):
                        expected=transform(base['aux']['geometry'][i,c].numpy()[...,None],flip,rot)[...,0]
                        np.testing.assert_allclose(aux['geometry'][i,c],expected,atol=1e-7)
                    # Vector channels transform both spatial positions and components.
                    for pair in ((4,5),(13,14)):
                        fields=base['aux']['geometry'][i,list(pair)].numpy().transpose(1,2,0)
                        expected=transform(fields,flip,rot)@R.T
                        np.testing.assert_allclose(aux['geometry'][i,list(pair)].numpy().transpose(1,2,0),expected,atol=1e-7)
                # Native search areas are recomputed after augmentation with the
                # historical even-sized/ellipse kernel. They are not exactly
                # equivariant. Preserve that native contract; do not fix it in
                # one ablation. Only derived vectors/covariances obey R above.
                np.testing.assert_array_equal(aux['geometry'][:,11],aux['native_support'])
                multiplicity=np.minimum(np.log1p(aux['native_support'].numpy().sum(0))/np.log(33),1)
                np.testing.assert_allclose(aux['geometry'][:,12],np.broadcast_to(multiplicity,aux['geometry'][:,12].shape),atol=1e-7)
                expected=transform(base['aux']['native_support'].numpy().transpose(1,2,0),flip,rot).transpose(2,0,1)
                native_support_transform_differences=int(np.count_nonzero(expected!=aux['native_support'].numpy()))
            else:native_support_transform_differences=None
            records.append(dict(flip=flip,rot90=rot,dapi_aligned='dapi' in aux,geometry_vectors_covariance_equivariant='geometry' in aux,
                native_support_transform_differences=native_support_transform_differences,
                native_support_contract='recomputed after augmentation, historical kernel preserved'))
    return records


def resume_check(cfg,node,model,opt,data,base,state,view,out):
    """Actual uninterrupted/resumed next update, native five-repeat noise envelope."""
    before=cpu_copy(state);path=out/'resume_temporary.pth';atomic_torch(path,before)
    def trial(reload=True):
        nonlocal model,opt,data
        # Adam retains its CPU step tensors when loading a state dictionary.
        # Each trial needs independent counters; otherwise opt.step() mutates
        # the reference snapshot and later trials are no longer at update 32.
        if reload:
            model.load_state_dict(before['model']);opt.load_state_dict(copy.deepcopy(before['optimizer']));m.restore_rng(before['rng']);model.train()
        b=data.get_view(view,'cuda');opt.zero_grad(set_to_none=True);z=m.forward(model,b,node);loss,_=base(z,b);loss.backward()
        gradients={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}
        opt.step();torch.cuda.synchronize()
        return dict(logits=z.detach().cpu(),gradients=gradients,model=cpu_copy(model.state_dict()),optimizer=cpu_copy(opt.state_dict()),rng=m.rng_state())
    original=trial();floor=None;repeat_summaries=[]
    for _ in range(5):
        repeat=trial();assert torch.equal(original['logits'],repeat['logits'])
        rows=compare_tensors(original['gradients'],repeat['gradients'])
        repeat_summaries.append(max(r['rms'] for r in rows.values()))
        if floor is None:floor=copy.deepcopy(rows)
        else:
            for key,row in rows.items():
                if row['rms']>floor[key]['rms']:floor[key]=row
        del repeat;gc.collect()
    assert m.state_equal(before,torch.load(path,map_location='cpu'))
    # Construction must not replace the restored cursor, BN, optimizer or RNG.
    model,opt,data=m.setup(cfg,node)
    restored=m.restore(cfg,path,model,opt,before['header'],pilot=True)
    assert m.state_equal(before['model'],model.state_dict()) and m.state_equal(before['optimizer'],opt.state_dict())
    assert m.state_equal(before['rng'],m.rng_state())
    for key in ('view_cursor','actual_updates','empty_skips','effective_sequence'):assert m.state_equal(before[key],restored[key])
    # Execute directly from the actual disk-restored model/Adam/RNG above.
    resumed=trial(reload=False)
    assert torch.equal(original['logits'],resumed['logits'])
    bn=[n for n in original['model'] if n.endswith(('running_mean','running_var','num_batches_tracked'))]
    assert all(torch.equal(original['model'][n],resumed['model'][n]) for n in bn)
    assert m.state_equal(original['rng'],resumed['rng'])
    gradients=compare_tensors(original['gradients'],resumed['gradients'],floor,'gradients')
    state_rows=compare_tensors(original['model'],resumed['model'])
    for name,row in state_rows.items():
        if row['maxabs']>2e-5+1e-8:raise ValueError('next Adam delta exceeds original native contract '+name)
    formulas={}
    for name,trial_result in [('uninterrupted',original),('resumed',resumed)]:
        formula=adam_formula(before['model'],trial_result['model'],trial_result['gradients'],before['optimizer'],trial_result['optimizer'])
        formulas[name]=dict(max_parameter_formula_error=max(v['max_parameter_formula_error'] for v in formula['tensors'].values()),
            max_first_moment_formula_error=max(v['max_first_moment_formula_error'] for v in formula['tensors'].values()),
            max_second_moment_formula_error=max(v['max_second_moment_formula_error'] for v in formula['tensors'].values()),
            bound=formula['moment_rounding_bound'])
    rec=dict(restored_model_BN_optimizer_RNG_cursor_exact=True,actual_disk_restored_next_update=True,next_logits_BN_RNG_exact=True,
        next_model_exact=m.state_equal(original['model'],resumed['model']),
        next_optimizer_exact=m.state_equal(original['optimizer'],resumed['optimizer']),
        native_repeat_gradient_max_rms=repeat_summaries,
        next_gradient_max_rms=max(v['rms'] for v in gradients.values()),
        next_parameter_max_abs=max(v['maxabs'] for v in state_rows.values()),independent_adam=formulas,
        contract='original GPU gradient RMS <=2x five native repeats+1e-7; state max <=2lr+1e-8; both obey independent Adam; CPU resume bitwise separately tested')
    atomic_json(out/'resume.json',rec)
    # Continue the pilot from its original durable state, not any trial update.
    model.load_state_dict(before['model']);opt.load_state_dict(before['optimizer']);m.restore_rng(before['rng']);model.train()
    path.unlink();del before,original,resumed,restored;gc.collect();torch.cuda.empty_cache()
    return model,opt,data,rec


def pilot(cfg,node,out):
    out.mkdir(parents=True,exist_ok=True);started=time.monotonic();model,opt,data=m.setup(cfg,node)
    header=m.initial_header(cfg,node);views=read(cfg['inputs']['views'])['views']
    hashes={v['path']:v['identity']['sha256'] for v in read(cfg['inputs']['predict_shift0'])}
    initial=cpu_copy(m.snapshot(model,opt,header,pilot=True));atomic_torch(out/'step_zero.pth',initial)
    atomic_json(out/'initialization_map.json',m.initialization_map(model.morphology,m.canonical_full()))
    first=next(v for v in views if v['K']);batch=data.get_view(first,'cuda');model.eval()
    with torch.no_grad():
        actual=m.forward(model,batch,node);morph=model.morphology;delattr(model,'morphology');expected=model(batch['input']);model.morphology=morph
    assert torch.equal(actual,expected)
    augmentation=augmentation_checks(data,first)
    if node=='MORPH_NO_DAPI':
        old_load=np.load
        def guard(path,*args,**kwargs):
            if str(path).endswith('dapi_features.npy'):raise AssertionError('NO_DAPI attempted image file access')
            return old_load(path,*args,**kwargs)
        # A second construction with the image file made unavailable must still work.
        with patch('numpy.load',side_effect=guard):
            absent=m.dataset(cfg,node);b=absent.get_view(first,'cuda')
        assert not hasattr(absent.evidence,'dapi') and 'dapi' not in b['aux']
        with torch.no_grad():assert torch.equal(m.forward(model,b,node),actual)
        del absent,b
    # Restore genuine step zero after diagnostic dataset operations.
    model.load_state_dict(initial['model']);opt.load_state_dict(initial['optimizer']);m.restore_rng(initial['rng']);model.train();del initial,batch,actual,expected
    base=m.BaseSix('cuda');updates=cursor=skips=0;seq=[];checks={};torch.cuda.reset_peak_memory_stats()
    while updates<100:
        view=views[cursor];cursor+=1
        if not view['K']:data.get_view(view,'cuda');skips+=1;continue
        batch,row=m.update(model,opt,data,base,view,node);updates+=1
        row.update(m.sequence_identity(view,updates,hashes[view['path']]));seq.append(row)
        if updates in m.CHECKS:
            row['effect_probe']=m.effect_probe(model,batch,node)
            if row['effect_probe']['delta_h3_norm']<=0 or row['effect_probe']['logit_l2_delta']<=0:raise ValueError('trained residual ineffective')
            for name,group in row['gradient_groups'].items():
                if group['norm']<=0:raise ValueError('inactive morphology parameter group '+name)
            checks[str(updates)]=copy.deepcopy(row);atomic_json(out/f'update_{updates:04d}.json',row)
        del batch
        if updates==32:
            state=m.snapshot(model,opt,header,updates,cursor,skips,seq,pilot=True)
            nextview=next(v for v in views[cursor:] if v['K'])
            model,opt,data,resume=resume_check(cfg,node,model,opt,data,base,state,nextview,out)
            del state
        if updates%10==0:print(node,'isolated pilot',updates,'/100',flush=True)
    torch.cuda.synchronize();train_seconds=time.monotonic()-started;peak=torch.cuda.max_memory_allocated();maximum_started=time.monotonic()
    checkpoint=out/'pilot_final.pth'
    final=m.snapshot(model,opt,header,updates,cursor,skips,seq,pilot=True);m.validate_cursor(cfg,final)
    atomic_torch(checkpoint,final);del final
    # Complete maximum training candidate group enters one backbone forward.
    # It runs after the pilot endpoint is archived and cannot alter its BN/RNG.
    maxview=max(views,key=lambda v:v['K']);torch.cuda.reset_peak_memory_stats()
    batch=data.get_view(maxview,'cuda');opt.zero_grad(set_to_none=True);z=m.forward(model,batch,node);loss,_=base(z,batch);loss.backward();torch.cuda.synchronize()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('max K nonfinite')
    max_k=dict(K=maxview['K'],view=maxview,peak_vram_bytes=torch.cuda.max_memory_allocated(),full_backbone_forward=True)
    del batch,z,loss;opt.zero_grad(set_to_none=True)
    model.eval();prediction_maxima=[]
    # Candidate IDs derive from the unchanged nuclear crop, so enumerate all
    # prediction groups without loading their 313-channel expression tensors.
    for shift in (0,24):
        preddata=m.dataset(cfg,node,False,shift);pool=read(cfg['inputs'][f'predict_shift{shift}'])
        def candidate_count(v):
            ids=np.unique(preddata.nuclei[v['row']:v['row']+48,v['col']:v['col']+48])
            return int((ids>0).sum())
        largest=max(pool,key=candidate_count);k=candidate_count(largest);torch.cuda.reset_peak_memory_stats()
        batch=preddata.get_view(dict(index=largest['index']),'cuda')
        assert len(batch['input'])==k
        with torch.inference_mode():z=m.forward(model,batch,node)
        assert torch.isfinite(z).all();torch.cuda.synchronize()
        prediction_maxima.append(dict(shift=shift,K=k,index=largest['index'],peak_vram_bytes=torch.cuda.max_memory_allocated()))
        del preddata,batch,z;gc.collect();torch.cuda.empty_cache()
    rec=dict(node=node,variant=m.variant_spec(node),step_zero_logits_exact=True,augmentation=augmentation,
        no_dapi_file_access_absent=node=='MORPH_NO_DAPI',checks=checks,resume=resume,max_K=max_k,prediction_max_K=prediction_maxima,
        actual_updates=updates,view_cursor=cursor,empty_skips=skips,wall_seconds=time.monotonic()-started,
        training_and_resume_diagnostic_seconds=train_seconds,maximum_K_diagnostic_seconds=time.monotonic()-maximum_started,
        production_update_seconds=sum(row['seconds'] for row in seq),
        peak_vram_bytes=peak,parameter_count=sum(p.numel() for p in model.parameters()),
        added_parameter_count=sum(p.numel() for p in model.morphology.parameters()),checkpoint=identity(checkpoint))
    atomic_json(out/'pilot.json',rec);del model,opt,data;gc.collect();torch.cuda.empty_cache();return rec


def roi_inputs(cfg,dest):
    original=read(m.endpoint_spec(cfg,'V2_MORPH')['machine']);nuc=tifffile.imread(original['nuclei_mask'])
    view=next(v for v in read(cfg['inputs']['views'])['views'] if v['K']>=12);size=144;y0,x0=view['row'],view['col']
    if y0+size>nuc.shape[0] or x0+size>nuc.shape[1]:raise ValueError('fixed mini ROI exceeds image')
    dest.mkdir(parents=True,exist_ok=True);atomic_tiff(dest/'nuclei.tif',nuc[y0:y0+size,x0:x0+size].astype(np.uint32))
    xa,ya=lookup_axes((size,size),original['scale_x'],original['scale_y']);ix=np.searchsorted(xa,np.arange(size));iy=np.searchsorted(ya,np.arange(size))
    tables=[]
    for chunk in pd.read_csv(original['transcripts'],usecols=['transcript_id','x_location','y_location','feature_name','qv'],chunksize=500000):
        x,y,valid=grid(chunk,nuc.shape,original['scale_x'],original['scale_y']);keep=valid&(x>=x0)&(x<x0+size)&(y>=y0)&(y<y0+size)
        selected=chunk.loc[keep].copy()
        if len(selected):
            lx=x[keep]-x0;ly=y[keep]-y0;selected['x_location']=ix[lx]*original['scale_x'];selected['y_location']=iy[ly]*original['scale_y']
            cx,cy,ok=grid(selected,(size,size),original['scale_x'],original['scale_y'])
            if not ok.all() or not np.array_equal(cx,lx) or not np.array_equal(cy,ly):raise ValueError('ROI native coordinate rebase')
            tables.append(selected)
    table=pd.concat(tables,ignore_index=True);table.to_csv(dest/'transcripts_processed.csv',index=False)
    roi=dict(row=y0,col=x0,size=size,retained_transcripts=len(table),global_shape=list(nuc.shape),
        source_nuclei=identity(original['nuclei_mask']),source_transcripts=identity(original['transcripts']),
        coordinate_rebase='native global pixel to exact inverse local nearest-neighbour lookup; original IDs/genes')
    atomic_json(dest/'roi.json',roi);return original,roi


def roi_predict(cfg,node,checkpoint,root,roi,original,shared,legacy=False):
    started=time.monotonic();out=root/'predictions'/node;out.mkdir(parents=True,exist_ok=True)
    state=torch.load(checkpoint,map_location='cpu');model=(historical_model(cfg['inputs']['shared_initial']) if legacy else m.make_model(node,cfg['inputs']['shared_initial']))
    model.load_state_dict(state['model']);model.eval();del state
    y0,x0,size=roi['row'],roi['col'],roi['size'];maps=[];logit_hashes={};patches=[];patch_seconds=[]
    for shift in (0,24):
        data=m.dataset(cfg,node,False,shift);pool=read(cfg['inputs'][f'predict_shift{shift}'])
        chosen=[v for v in pool if y0<=v['row'] and v['row']+48<=y0+size and x0<=v['col'] and v['col']+48<=x0+size]
        if not chosen:raise ValueError('ROI lacks both shifts')
        whole=np.zeros((size,size),np.uint32)
        with torch.inference_mode():
            for v in chosen:
                begin=time.monotonic()
                b=data.get_view(dict(index=v['index']),'cuda')
                if len(b['input']):
                    z=model(b['input'],b['aux']) if legacy else m.forward(model,b,node)
                    if not torch.isfinite(z).all():raise ValueError('ROI nonfinite logits')
                    logit_hashes[f'{shift}:{v["index"]}']=m.tensor_sha({'logits':z})
                    patchmask=m.get_seg_mask(z.cpu().numpy(),b['labels'].cpu().numpy()).astype(np.uint32)
                else:patchmask=np.zeros((48,48),np.uint32)
                y,x=v['row']-y0,v['col']-x0;whole[y:y+48,x:x+48]=patchmask
                torch.cuda.synchronize();patch_seconds.append(time.monotonic()-begin)
        maps.append(whole);atomic_tiff(out/f'shift{shift}.tif',whole)
        if shift==24:shifted=[f"{v['row']-y0}_{v['col']-x0}.hdf5" for v in chosen]
        patches.append(dict(shift=shift,views=chosen));del data
    pre,fill=m.merge_native(*maps,shifted);atomic_tiff(out/'pre.tif',pre);atomic_tiff(out/'fill.tif',fill)
    machine=dict(original,nuclei_mask=str(shared/'nuclei.tif'),transcripts=str(shared/'transcripts_processed.csv'),pre_mask=str(out/'pre.tif'),final_mask=str(out/'final.tif'),baseline_final_mask=str(out/'pre.tif'),final_matrix=str(root/'matrix'/node/'expr_mat.csv'),checkpoint=str(checkpoint),checkpoint_scope='historical replay' if node=='V2_MORPH' else 'isolated temporary pilot; discarded after successful mini evaluation')
    atomic_json(root/'machine.json',machine)
    spec=dict(root=str(root),machine=str(root/'machine.json'),pre=str(out/'pre.tif'),final=str(out/'final.tif'),matrix=str(root/'matrix'/node),evaluation=str(root/'evaluation'/node),cost_root=str(root))
    rec=dict(checkpoint=identity(checkpoint),logits=logit_hashes,patches=patches,wall_seconds=time.monotonic()-started,
        production_patch_seconds=patch_seconds,production_prediction_seconds=sum(patch_seconds),shifts=[0,24])
    atomic_json(out/'prediction.json',rec);del model;gc.collect();torch.cuda.empty_cache();return spec,rec


def historical_replay(cfg,root,roi,original,shared):
    checkpoint=m.endpoint_spec(cfg,'V2_MORPH')['checkpoint']
    state=torch.load(checkpoint,map_location='cpu')['model'];data=m.dataset(cfg,'V2_MORPH')
    view=next(v for v in read(cfg['inputs']['views'])['views'] if v['K'])
    batch=data.get_view(view);cpu_results=[]
    for constructor in (lambda:historical_model(cfg['inputs']['shared_initial'],'cpu'),
                        lambda:m.make_model('V2_MORPH',cfg['inputs']['shared_initial'],'cpu')):
        model=constructor();model.load_state_dict(state);model.eval()
        with torch.no_grad():cpu_results.append(model(batch['input'],batch['aux']).clone())
        del model;gc.collect()
    assert torch.equal(*cpu_results)
    del state,data,batch,cpu_results;gc.collect()
    a,ar=roi_predict(cfg,'V2_MORPH',checkpoint,root/'new',roi,original,shared)
    b,br=roi_predict(cfg,'V2_MORPH',checkpoint,root/'legacy',roi,original,shared,legacy=True)
    assert ar['logits']==br['logits']
    for key in ('pre',):assert np.array_equal(tifffile.imread(a[key]),tifffile.imread(b[key]))
    pre=tifffile.imread(a['pre']);historical=tifffile.imread(m.endpoint_spec(cfg,'V2_MORPH')['pre'])
    y,x=roi['row'],roi['col'];interior=(slice(48,96),slice(48,96))
    assert np.array_equal(pre[interior],historical[y+48:y+96,x+48:x+96])
    m.native_postprocess(cfg,'V2_MORPH',a);m.native_postprocess(cfg,'V2_MORPH',b)
    assert np.array_equal(tifffile.imread(a['final']),tifffile.imread(b['final']))
    rec=dict(status='passed',checkpoint=identity(checkpoint),canonical_sha=m.CANONICAL_SHA,
        historical_source_cpu_logits_exact=True,new_vs_legacy_roi_logits_exact=True,dual_shift_pre_exact=True,native_final_exact=True,
        historical_global_pre_interior_exact=True,global_final_comparison_scope='ROI postprocess runs original rule on identical ROI, not a claim of global tile-halo equivalence',
        new_prediction=ar,legacy_prediction=br)
    atomic_json(root/'replay.json',rec);return rec


def validate_mini(config,output):
    cfg=load_run(config);out=Path(output).resolve();root=Path(cfg['output_root']).resolve()
    if out!=root and root not in out.parents:raise ValueError('mini output outside isolated config root')
    if 'mini' not in out.parts:raise ValueError('real pilot must be explicitly isolated under mini')
    out.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    if not torch.cuda.is_available():raise RuntimeError('real admission requires authorized host CUDA')
    print('Waiting for shared compute lock (mini-only timeout3600s; formal policy unchanged)',flush=True)
    with compute_lock(timeout=3600):
        acquired=time.monotonic()
        canonical=m.canonical_full();canonical_path=out/'canonical_full_seed20261013.pth'
        atomic_torch(canonical_path,dict(model=canonical.state_dict(),seed=20261013,tensor_sha256=m.CANONICAL_SHA,
            limitation='frozen-code reconstruction, not independently archived historical step zero'))
        del canonical
        original,roi=roi_inputs(cfg,out/'shared');replay=historical_replay(cfg,out/'full_replay',roi,original,out/'shared')
        results={};discarded=[]
        for node in m.NODES:
            dest=out/node;pilotrec=pilot(cfg,node,dest/'training')
            spec,pred=roi_predict(cfg,node,pilotrec['checkpoint']['path'],dest,roi,original,out/'shared')
            post=m.native_postprocess(cfg,node,spec)
            evaluation=build_and_evaluate(cfg,node,spec)
            matrix=read(Path(spec['matrix'])/'complete.json')
            if matrix['totals']['eligible']!=roi['retained_transcripts']:raise ValueError('ROI denominator drift')
            results[node]=dict(training=pilotrec,prediction=pred,postprocess=post,evaluation=evaluation)
            checkpoint=Path(pilotrec['checkpoint']['path']);discarded.append(identity(checkpoint));checkpoint.unlink()
            print(node,'mini full evaluator complete; temporary trained checkpoint discarded',flush=True)
        rec=dict(status='passed',source=m.scientific_source(),config_digest=m.digest(cfg),scope='real isolated ROI and 100 updates per variant; no scientific endpoint or formal initial',
            roi=roi,canonical_full_archive=identity(canonical_path),historical_full_replay=replay,variants=results,discarded_temporary_trained_checkpoints=discarded,
            wall_seconds=time.monotonic()-started,compute_wall_seconds=time.monotonic()-acquired,
            resource_wait_seconds=acquired-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            outputs={k:v for k,v in inventory(out).items() if k!='validation.log'})
        atomic_json(out/'complete.json',rec);return rec
