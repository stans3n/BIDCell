"""Measured preflight; development states are never formal initialisation."""
import copy
import fcntl
import gc
import json
import resource
import shutil
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from .common import *
from .data import NativeViews
from .training import *
from .calibration import calibrate,joint_stats,calibration_dependencies
from .diagnostics import coverage,fixed_probe_diagnostics,distribution


def equal_state(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal_state(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal_state(x,y) for x,y in zip(a,b))
    return a==b


def frozen_capture(model,data,views,arm):
    model.eval();rows=[]
    with torch.no_grad():
        for v in views:
            b=data.get_view(v,'cuda');z=model(b['input'],b['aux'] if arm=='V2_MORPH' else None)
            row=capture(z,b);row['view_cursor']=v['cursor'];row['owner']=native_owner(z).cpu().numpy()
            non=b['labels'].cpu().numpy()==0;mass=b['aux']['raw_counts'].sum(0).cpu().numpy()
            row['areas']=[int(((row['owner']==i+1)&non).sum()) for i in range(len(z))]
            row['cells']=[dict(cell_id=int(cid),type_index=int(b['aux']['type_index'][i]),
                         elongated=int(cid) in data.nuclei_ids_elong,area=row['areas'][i],
                         capture=float(mass[(row['owner']==i+1)&non].sum())) for i,cid in enumerate(b['candidate_ids'])]
            rows.append(row)
    return rows


def compare_capture(base,own):
    b=sum(r['capture'] for r in base);o=sum(r['capture'] for r in own)
    ba=np.median([a for r in base for a in r['areas']]);oa=np.median([a for r in own for a in r['areas']])
    by_type={};bc=[c for r in base for c in r['cells']];oc=[c for r in own for c in r['cells']]
    for group in [*sorted({c['type_index'] for c in bc}),'elongated']:
        idx=[i for i,c in enumerate(bc) if (c['elongated'] if group=='elongated' else c['type_index']==group)]
        bv=sum(bc[i]['capture'] for i in idx);ov=sum(oc[i]['capture'] for i in idx)
        ba_group=float(np.median([bc[i]['area'] for i in idx])) if idx else 0
        oa_group=float(np.median([oc[i]['area'] for i in idx])) if idx else 0
        by_type[str(group)]=dict(appearances=len(idx),unique_ids=len({bc[i]['cell_id'] for i in idx}),
             meets_20_sample_reporting=len(idx)>=20,base_capture=bv,own_capture=ov,relative_capture=ov/bv-1 if bv else None,
             base_area_median=ba_group,own_area_median=oa_group,relative_area=oa_group/ba_group-1 if ba_group else None)
    return dict(base_capture=b,own_capture=o,relative_capture=o/b-1 if b>0 else None,by_type=by_type,
                base_area_median=float(ba),own_area_median=float(oa),relative_area=float(oa/ba-1) if ba>0 else None,
                owner_transitions=sum(int((a['owner']!=b['owner']).sum()) for a,b in zip(base,own)),
                denominator_note='zero baseline denominators remain NA; no claimed pass from zero')


def developer_check(root,arm):
    """Bounded real updates, effective module changes and strict next-step resume."""
    root=Path(root);q=read(root/'configs/queue.json');cfg=read(q['methods'][arm]['config']);cal=read(cfg['calibration'])
    probes=read(root/'configs/probes.json');views=probes['development'];model=make_model(arm,cfg['initial']);opt=optimizer(model)
    data=NativeViews(root,morphology=arm=='V2_MORPH');base=BaseSix('cuda');extra=loss_module(cfg,'cuda')
    out=root/'admission'/arm;out.mkdir(exist_ok=True);groups=[];start=time.monotonic();stop=100 if arm=='V2_MORPH' else 64
    initial_logits=None
    probe_batch=data.get_view(views[0],'cuda');model.eval()
    with torch.no_grad():initial_logits=model(probe_batch['input'],probe_batch['aux'] if arm=='V2_MORPH' else None).cpu()
    model.train();resume_record=None;formal_views=read(cfg['views'])['views'];header=checkpoint_header(cfg)
    for step in range(stop):
        b=data.get_view(views[step],'cuda');opt.zero_grad(set_to_none=True)
        old_params={n:p.detach().clone() for n,p in model.named_parameters() if n.startswith('morphology.') and step+1 in (1,2,32,64,100)}
        z,l6,new,parts,diag=forward_losses(model,b,base,extra,arm);addition,lam=intervention(new,cal,step);loss=l6+addition
        if not torch.isfinite(loss):raise FloatingPointError('pilot loss nonfinite')
        loss.backward()
        if not finite_gradients(model):raise FloatingPointError('pilot gradient nonfinite')
        if step==1:expected_step_gradients={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters()}
        if old_params:
            norm={}
            for n,p in model.named_parameters():
                if n in old_params:
                    key='.'.join(n.split('.')[:3]) if n.startswith('morphology.blocks') else n.split('.')[1]
                    norm[key]=norm.get(key,0.)+(float(p.grad.double().square().sum()) if p.grad is not None else 0.)
        opt.step()
        if old_params:
            update={}
            for n,p in model.named_parameters():
                if n in old_params:
                    key='.'.join(n.split('.')[:3]) if n.startswith('morphology.blocks') else n.split('.')[1]
                    update[key]=update.get(key,0.)+float((p.detach()-old_params[n]).double().square().sum())
            groups.append(dict(step=step+1,gradient_norm={k:float(np.sqrt(v)) for k,v in norm.items()},
                               actual_update_norm={k:float(np.sqrt(v)) for k,v in update.items()},morphology=model.morphology.diagnostics))
        if step==0:
            cursor=views[0]['cursor']+1;skips=sum(v['K']==0 for v in formal_views[:cursor])
            assert sum(v['K']>0 for v in formal_views[:cursor])==1
            atomic_torch(out/'resume_step1.pth',snapshot(model,opt,header,1,cursor,skips,0))
        if step==1:
            # Production header, restore implementation and frozen cursor; the
            # resumed batch is loaded afresh from its next nonempty frozen view.
            expected={n:p.detach().cpu().clone() for n,p in model.state_dict().items()}
            saved_rng=rng_state();resumed=make_model(arm,cfg['initial']);ropt=optimizer(resumed)
            restored=restore(out/'resume_step1.pth',resumed,ropt,header)
            # Adam's CPU scalar step can alias load_state_dict input; preserve
            # the immutable pre-update moments/counter for independent formula.
            before_opt=copy.deepcopy(restored['optimizer'])
            assert all(torch.equal(v.detach().cpu(),restored['model'][n]) for n,v in resumed.state_dict().items())
            for key,state in ropt.state_dict()['state'].items():
                for field,value in state.items():
                    expected_value=restored['optimizer']['state'][key][field]
                    assert torch.equal(value.cpu(),expected_value.cpu()) if torch.is_tensor(value) else value==expected_value
            rng_restored=equal_state(rng_state(),restored['rng']);assert rng_restored
            drift_rejected=[]
            for key in header:
                altered=copy.deepcopy(header);altered[key]=dict(deliberate_drift=True,original=altered[key])
                try:restore(out/'resume_step1.pth',resumed,ropt,altered)
                except ValueError:drift_rejected.append(key)
                else:raise AssertionError('header drift accepted '+key)
            cursor=restored['view_cursor'];skips=restored['empty_skips']
            while formal_views[cursor]['K']==0:cursor+=1;skips+=1
            next_view=formal_views[cursor];rb=data.get_view(next_view,'cuda');cursor+=1
            assert next_view==views[1] and sum(v['K']>0 for v in formal_views[:cursor])==2
            resumed.train();ropt.zero_grad(set_to_none=True)
            rz,rl6,rnew,_,_=forward_losses(resumed,rb,base,extra,arm);ra,_=intervention(rnew,cal,1);(rl6+ra).backward()
            resumed_gradients={n:p.grad.detach().cpu().clone() for n,p in resumed.named_parameters()};ropt.step()
            next_rng_exact=equal_state(saved_rng,rng_state());assert next_rng_exact
            # Original CUDA interpolation/reductions are not bitwise backward
            # deterministic. Establish a second same-state native step's floor.
            control=make_model(arm,cfg['initial']);co=optimizer(control)
            from tests.new_methods.contracts import compare_tensors,adam_formula,adam_outlier_coordinates
            def opt_tensors(optim):return {f'{k}.{f}':v.detach().cpu() for k,s in optim.state_dict()['state'].items() for f,v in s.items() if torch.is_tensor(v)}
            expected_opt=opt_tensors(opt);resumed_opt=opt_tensors(ropt);floor=opt_floor=grad_floor=None;repeat_rows=[]
            for repetition in range(5):
                restore(out/'resume_step1.pth',control,co,header);control.train();co.zero_grad(set_to_none=True)
                cb=data.get_view(next_view,'cuda');cz,cl6,cn,_,_=forward_losses(control,cb,base,extra,arm)
                ca,_=intervention(cn,cal,1);(cl6+ca).backward()
                current_grad=compare_tensors(expected_step_gradients,{n:p.grad.detach().cpu() for n,p in control.named_parameters()});co.step()
                current=compare_tensors(expected,{n:v.detach().cpu() for n,v in control.state_dict().items()},kind='next_state')
                current_opt=compare_tensors(expected_opt,opt_tensors(co),kind='moments');repeat_rows.append(dict(state=current,Adam=current_opt,gradients=current_grad))
                if floor is None:floor=copy.deepcopy(current);opt_floor=copy.deepcopy(current_opt);grad_floor=copy.deepcopy(current_grad)
                else:
                    for target,values in ((floor,current),(opt_floor,current_opt),(grad_floor,current_grad)):
                        for name in target:
                            if values[name]['rms']>target[name]['rms']:target[name]=values[name]
            resumed_state={n:v.detach().cpu() for n,v in resumed.state_dict().items()}
            state_check=compare_tensors(expected,resumed_state,floor,kind='next_state')
            opt_check=compare_tensors(expected_opt,resumed_opt,opt_floor,kind='moments')
            grad_check=compare_tensors(expected_step_gradients,resumed_gradients,grad_floor)
            before_state=restored['model']
            formula_original=adam_formula(before_state,expected,expected_step_gradients,before_opt,opt.state_dict())
            formula_resumed=adam_formula(before_state,resumed_state,resumed_gradients,before_opt,ropt.state_dict())
            outlier_coordinates=adam_outlier_coordinates(dict(state=expected,gradients=expected_step_gradients),dict(state=resumed_state,gradients=resumed_gradients),state_check,before_state)
            bn=[n for n in expected if n.endswith(('running_mean','running_var','num_batches_tracked'))]
            assert all(torch.equal(expected[n],resumed_state[n]) for n in bn)
            maxdiff=max(v['maxabs'] for v in state_check.values())
            before_bn={n:p.detach().clone() for n,p in resumed.named_buffers()};before_rng=rng_state()
            probe(resumed,data,base,extra,cfg,[next_view],'resume_purity')
            bn_exact=all(torch.equal(v,dict(resumed.named_buffers())[n]) for n,v in before_bn.items())
            probe_rng_exact=equal_state(before_rng,rng_state());assert bn_exact and probe_rng_exact and resumed.training
            resume_record=dict(max_model_BN_difference=maxdiff,restored_model_optimizer_exact=True,next_BN_exact=True,
                               next_state_comparison=state_check,native_next_state_repeat=floor,next_Adam_comparison=opt_check,
                               next_gradient_comparison=grad_check,original_Adam_formula=formula_original,resumed_Adam_formula=formula_resumed,
                               Adam_RMS_outlier_coordinates=outlier_coordinates,
                               native_next_Adam_repeat=opt_floor,native_repeat_count=5,native_repeats=repeat_rows,cursor=cursor,empty_skips=skips,
                               full_header=header,drift_rejected=drift_rejected,next_view=next_view,rng_restored=rng_restored,
                               next_rng_exact=next_rng_exact,probe_BN_exact=bn_exact,probe_RNG_exact=probe_rng_exact)
            restore_rng(saved_rng);del resumed,ropt,expected,restored,rb,rz,rl6,rnew,ra,control,co,cb,cz,cl6,cn,ca,expected_step_gradients,resumed_gradients;gc.collect();torch.cuda.empty_cache()
        if (step+1)%16==0:print(arm,'isolated pilot',step+1,'/',stop,flush=True)
    if arm=='V2_MORPH':
        for rec in groups[1:]:
            missing=[k for k,v in rec['gradient_norm'].items() if v<=0]
            if missing:raise RuntimeError('morphology groups inactive '+str(missing))
    model.eval()
    with torch.no_grad():final_logits=model(probe_batch['input'],probe_batch['aux'] if arm=='V2_MORPH' else None).cpu()
    diagnostics=fixed_probe_diagnostics(model,data,base,extra,cfg,stop,cal,pilot=True)
    views_capture=probes['region'] if arm=='V1_REGION' else probes['boundary'] if arm=='V1_BOUNDARY' else probes['morphology']
    own_capture=frozen_capture(model,data,views_capture,arm)
    atomic_torch(out/'pilot_final.pth',dict(model=model.state_dict(),updates=stop,arm=arm,pilot=True,source=source_digest(),initial=identity(cfg['initial'])))
    del model,opt;gc.collect();torch.cuda.empty_cache()
    paired=make_model('BASE',cfg['initial']);pairedopt=optimizer(paired);basecfg=dict(cfg,arm='BASE')
    temporary_updates(paired,pairedopt,data,base,None,basecfg,views,0,stop)
    baseline_capture=frozen_capture(paired,data,views_capture,'BASE');guard=compare_capture(baseline_capture,own_capture)
    del paired,pairedopt;gc.collect();torch.cuda.empty_cache()
    if arm=='V1_REGION':
        if guard['relative_capture'] is None or guard['relative_area'] is None:raise RuntimeError('region guard remains unevaluable after64 development updates')
        if guard['relative_capture']<-.05 or guard['relative_area']<-.10:raise RuntimeError('region capture/area degeneration guard failed: '+str(guard))
        all_joint=[]
        # The gate uses all 32 prespecified region probes, beyond the routine
        # eight-view saved diagnostics. Recreate the exact pilot endpoint.
        endpoint=make_model(arm,cfg['initial']);endpoint.load_state_dict(torch.load(out/'pilot_final.pth',map_location='cpu')['model'],strict=True)
        full_probe=probe(endpoint,data,base,extra,cfg,probes['region'],'nominal64');all_joint=full_probe
        del endpoint;gc.collect();torch.cuda.empty_cache()
        joint=joint_stats(all_joint,cal['eta'],cal['lambda_max']);ratios=[r['ratio'] for r in joint if r['ratio'] is not None and r['norm7']>1e-12]
        cos=[r['cos_base'] for r in joint if r['cos_base'] is not None]
        atomic_json(out/'nominal64_gate.json',json_safe(dict(joint=joint,probe=full_probe,capture_guard=guard,resume=resume_record,
                    median=float(np.median(ratios)) if ratios else None,P95=float(np.quantile(ratios,.95)) if ratios else None,
                    negative_cos_fraction=float(np.mean(np.asarray(cos)<-.5)) if cos else None)))
        if not ratios or not .15<=np.median(ratios)<=.6 or np.quantile(ratios,.95)>1.5:raise RuntimeError('region nominal64 gradient ratio gate failed: '+str(ratios))
        if cos and np.mean(np.asarray(cos)<-.5)>.2:raise RuntimeError('region opposing-gradient review required')
    rec=dict(status='succeeded',arm=arm,updates=stop,wall_seconds=time.monotonic()-start,resume=resume_record,
             module_groups=groups,initial_final_logit_L2=float((final_logits-initial_logits).norm()),capture_guard=guard,
             pilot=identity(out/'pilot_final.pth'),source=source_digest())
    atomic_json(out/'development.json',rec);return rec


def sensitivity_region(root):
    root=Path(root);q=read(root/'configs/queue.json');cfg=read(q['methods']['V1_REGION']['config']);cal=read(cfg['calibration']);probes=read(root/'configs/probes.json')
    data=NativeViews(root);base=BaseSix('cuda');extra=loss_module(cfg,'cuda');rows={}
    for scale in (.5,1.,2.):
        model=make_model('V1_REGION',cfg['initial']);opt=optimizer(model);scaled=dict(cal,lambda_max=cal['lambda_max']*scale)
        updates=temporary_updates(model,opt,data,base,extra,cfg,probes['development'],0,32,scaled)
        p=probe(model,data,base,extra,cfg,probes['region'],f'scale{scale}_step32')
        rows[str(scale)]=dict(updates=updates,probe=p,joint=joint_stats(p,cal['eta'],cal['lambda_max']*scale))
        del model,opt;gc.collect();torch.cuda.empty_cache()
    atomic_json(root/'admission/V1_REGION/sensitivity.json',json_safe(rows));return rows


def performance(root):
    root=Path(root);q=read(root/'configs/queue.json');views=[v for v in read(root/'configs/views.json')['views'] if v['K']]
    ks=np.asarray([v['K'] for v in views]);selected=[]
    for quantile in (.5,.9,1.):
        target=int(np.quantile(ks,quantile,method='nearest'));selected.append(next(v for v in views if v['K']==target))
    results={}
    for arm in ('BASE',*ARMS):
        cfg=read(q['methods']['V1_REGION' if arm=='BASE' else arm]['config']);cfg=dict(cfg,arm=arm)
        model=make_model(arm,cfg['initial']);opt=optimizer(model);data=NativeViews(root,morphology=arm=='V2_MORPH')
        base=BaseSix('cuda');extra=loss_module(cfg,'cuda');cal={} if arm=='BASE' else read(cfg['calibration'])
        records=[];torch.cuda.reset_peak_memory_stats();model.train()
        # Distinct measured views prevent a three-view geometry/support cache
        # from understating the actual shuffled training-sidecar cost.
        used={v['cursor'] for v in views[:5]};measured=[]
        for i in range(15):
            target=selected[i%3]['K'];choice=min((v for v in views if v['cursor'] not in used),key=lambda v:(abs(v['K']-target),v['cursor']))
            used.add(choice['cursor']);measured.append(choice)
        timed=[*views[:5],*measured]
        for step,v in enumerate(timed):
            torch.cuda.synchronize();start=time.monotonic();b=data.get_view(v,'cuda');data_t=time.monotonic()-start;opt.zero_grad(set_to_none=True)
            t=time.monotonic();z=model(b['input'],b['aux'] if arm=='V2_MORPH' else None);l6,_=base(z,b);torch.cuda.synchronize();native_t=time.monotonic()-t
            t=time.monotonic()
            if arm=='V1_REGION':e,r,_=extra(z,b['aux'])
            elif arm=='V1_BOUNDARY':e,_=extra(z,b['aux']);r=z.sum()*0
            else:e=r=z.sum()*0
            add,_=intervention((e,r),cal,step);torch.cuda.synchronize();extra_t=time.monotonic()-t
            t=time.monotonic();(l6+add).backward();opt.step();torch.cuda.synchronize();back_t=time.monotonic()-t
            if not finite_gradients(model):raise FloatingPointError('performance high-K nonfinite')
            records.append(dict(step=step,view_cursor=v['cursor'],K=v['K'],sidecar_seconds=data_t,native_forward_six_seconds=native_t,
                                auxiliary_graph_gather_seconds=extra_t,backward_Adam_seconds=back_t,total_seconds=time.monotonic()-start))
        peak=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved();free,total=torch.cuda.mem_get_info()
        # Prediction worst group uses the same complete K and module policy.
        model.eval();torch.cuda.reset_peak_memory_stats()
        prediction=NativeViews(root,training=False,shift=0,auxiliary=arm=='V2_MORPH',morphology=arm=='V2_MORPH')
        # Independent full-pool inventory verified K=41 at this actual patch.
        predict_index=next(i for i,p in enumerate(prediction.fp_patches) if Path(p).stem=='4412_5776')
        with torch.inference_mode():
            worst=prediction.get_view(dict(index=predict_index),'cuda');z=model(worst['input'],worst.get('aux') if arm=='V2_MORPH' else None)
        if len(z)!=41:raise ValueError('expected verified prediction max K41')
        pred_peak=torch.cuda.max_memory_allocated();pred_reserved=torch.cuda.max_memory_reserved()
        headroom=min(free,torch.cuda.mem_get_info()[0],total-max(reserved,pred_reserved))
        if headroom<2**30:raise RuntimeError('less than1GiB VRAM headroom '+arm)
        checkpoint_path=root/'admission'/f'resource_checkpoint_{arm}.pth';t=time.monotonic()
        atomic_torch(checkpoint_path,snapshot(model,opt,dict(pilot_resource=True,arm=arm),20,20,0,0))
        checkpoint_seconds=time.monotonic()-t
        diagnostic_seconds=0.
        if arm!='BASE':
            t=time.monotonic();fixed_probe_diagnostics(model,data,base,extra,cfg,20,cal,pilot=True);diagnostic_seconds=time.monotonic()-t
        results[arm]=dict(records=records,timed=records[5:],peak_train_allocated_bytes=peak,peak_predict_allocated_bytes=pred_peak,
             checkpoint_write_fsync_seconds=checkpoint_seconds,checkpoint=identity(checkpoint_path),fixed_eight_probe_seconds=diagnostic_seconds,
             peak_reserved_bytes=max(reserved,pred_reserved),device_total_bytes=total,headroom_bytes=headroom,parameter_count=sum(p.numel() for p in model.parameters()),
             new_parameter_count=sum(p.numel() for n,p in model.named_parameters() if n.startswith('morphology.')),
             P50_seconds=float(np.median([r['total_seconds'] for r in records[5:]])),P95_seconds=float(np.quantile([r['total_seconds'] for r in records[5:]],.95)))
        del model,opt,data,b,z,prediction,worst;gc.collect();torch.cuda.empty_cache()
    for arm in ('V1_REGION','V1_BOUNDARY'):
        overhead=np.quantile([r['auxiliary_graph_gather_seconds'] for r in results[arm]['timed']],.95)
        results[arm]['auxiliary_overhead_ratio_P95']=float(overhead/results['BASE']['P95_seconds'])
        if overhead/results['BASE']['P95_seconds']>.5:raise RuntimeError('new loss overhead exceeds50% '+arm)
    results['system']=dict(peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,free_disk_bytes=shutil.disk_usage(root).free,
                           cuda_device=torch.cuda.get_device_name(),torch=torch.__version__,numpy=np.__version__)
    # Use measured K-conditioned training costs and explicitly historical CPU
    # endpoint costs; these are forecasts, never completion guarantees.
    history_path=WORK/'outputs/cos_bs4_20261012T013909+1100/run_status.json';history=read(history_path)
    historical={stage:float(np.median([sum(a['wall_seconds'] for a in history['stages'][f'{m}_{stage}']['attempts']) for m in ('T1','T2')]))
                for stage in ('predict','postprocess','matrix','evaluate')}
    estimates={}
    for arm in ARMS:
        timed=results[arm]['timed'];group={k:float(np.mean([r['total_seconds'] for r in timed if r['K']==k])) for k in sorted(set(r['K'] for r in timed))}
        update_seconds=float(np.interp(ks,list(group),list(group.values())).sum())
        checkpoint_seconds=146*results[arm]['checkpoint_write_fsync_seconds']
        diagnostic_seconds=4*results[arm]['fixed_eight_probe_seconds']
        train_seconds=update_seconds+checkpoint_seconds+diagnostic_seconds
        model_ratio=float(np.median([r['native_forward_six_seconds'] for r in timed])/np.median([r['native_forward_six_seconds'] for r in results['BASE']['timed']]))
        estimates[arm]=dict(train_seconds=train_seconds,predict_seconds=historical['predict']*model_ratio,
                           postprocess_seconds=historical['postprocess'],matrix_seconds=historical['matrix'],evaluate_seconds=historical['evaluate'])
        results[arm]['training_time_components']=dict(actual_updates=update_seconds,checkpoint_fsyncs_146=checkpoint_seconds,fixed_probe_diagnostics_4=diagnostic_seconds)
    train_total=1.2*sum(e['train_seconds'] for e in estimates.values())
    full_total=1.2*(sum(sum(e.values()) for e in estimates.values())+2*historical['evaluate'])
    from datetime import datetime,timedelta
    from zoneinfo import ZoneInfo
    now=datetime.now(ZoneInfo('Australia/Sydney'))
    results['eta']=dict(measured_at_sydney=now.isoformat(),training_seconds_with20pct=train_total,full_chain_seconds_with20pct=full_total,
                        train_if_started_now=(now+timedelta(seconds=train_total)).isoformat(),full_if_started_now=(now+timedelta(seconds=full_total)).isoformat(),
                        by_arm=estimates,historical_cpu_basis=identity(history_path),overhead_margin=.2,
                        caveat='single GPU serial training first; full-field prediction/CPU endpoint forecast partly historical; no morning guarantee')
    results['status']='succeeded';atomic_json(root/'admission/resources.json',results);return results


def admit(root):
    from .runner import lock_file
    root=Path(root);guard=lock_file(root/'admission.lock');q=read(root/'configs/queue.json');gpu=lock_file(q['gpu_lock'])
    if not torch.cuda.is_available():raise RuntimeError('host CUDA unavailable')
    atomic_json(root/'admission/status.json',dict(status='running',process=process_identity(),started=time.time()))
    try:
        from .inputs import verify_inputs
        atomic_json(root/'admission/input_validation.json',verify_inputs(root))
        from tests.new_methods.contracts import numerical,baseline_oracle
        atomic_json(root/'admission/numerical.json',dict(status='succeeded',results=numerical(),source=source_digest()))
        data=NativeViews(root,morphology=True);probeview=read(root/'configs/probes.json')['development'][0];batch=data.get_view(probeview,'cuda')
        atomic_json(root/'admission/baseline_equivalence.json',baseline_oracle(root,batch));del data,batch;gc.collect();torch.cuda.empty_cache()
        for arm in ARMS:
            cfg=read(q['methods'][arm]['config'])
            if Path(cfg['calibration']).exists() and read(cfg['calibration']).get('dependencies')!=calibration_dependencies(cfg):
                old=Path(cfg['calibration']);backup=old.with_name(old.stem+f'_superseded_{time.time_ns()}.json');old.rename(backup)
            if not Path(cfg['calibration']).exists():calibrate(cfg)
            cal=read(cfg['calibration'])
            if cal['status'] not in ('calibrated','admitted'):raise RuntimeError('calibration failed '+arm)
            from .admission_provenance import reusable_development,bind_development
            if not reusable_development(root,arm,cfg):
                developer_check(root,arm);bind_development(root/'admission'/arm/'development.json')
        sensitivity_region(root);performance(root)
        gpu.close();gpu=None
        coverage(root)
        from .mini import mini_chain
        mini_chain(root)
        for arm in ARMS:
            path=root/'admission'/f'{arm}_calibration.json';cal=read(path);cal['status']='admitted';cal['development_check']=identity(root/'admission'/arm/'development.json');atomic_json(path,cal)
        frozen=dict(files=source_manifest(),source_digest=source_digest());atomic_json(root/'source/manifest.json',frozen)
        atomic_json(root/'admission/status.json',dict(status='admitted',finished=time.time()))
        rec=dict(status='admitted',source_digest=source_digest(),queue_sha256=sha256(root/'configs/queue.json'),
                 outputs={str(p.relative_to(root/'admission')):identity(p) for p in sorted((root/'admission').rglob('*'))
                          if p.is_file() and p!=root/'admission/complete.json' and p.name not in ('admission.lock','independent_review.json') and not p.name.endswith('.tmp')},
                 scientific_review='awaiting_user_resume',formal_started=False)
        atomic_json(root/'admission/complete.json',rec);return rec
    except BaseException:
        atomic_json(root/'admission/status.json',dict(status='failed',finished=time.time(),failure=traceback.format_exc()))
        raise
    finally:
        if gpu:gpu.close()
        guard.close()
