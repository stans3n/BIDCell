"""Discarded development copies, frozen full-model gradient calibration."""
import copy
import gc
import time
from pathlib import Path
import numpy as np
import torch
from .common import *
from .data import NativeViews
from .training import *


def active(rows,a,b):return [r for r in rows if r['norms'][a]>1e-12 and r['norms'][b]>1e-12]


def calibration_dependencies(config):
    root=Path(config['runroot'])
    producers=['scripts/new_methods/calibration.py','scripts/new_methods/common.py','scripts/new_methods/training.py',
               'scripts/new_methods/data.py','bidcell/model/model/region_losses.py','bidcell/model/model/model.py',
               'bidcell/model/model/losses.py','bidcell/model/model/layers.py','bidcell/model/model/intialisation.py',
               'bidcell/model/dataio/dataset_input.py','bidcell/config.py']
    paths=dict(initial=config['initial'],reference=config['reference'],probes=root/'configs/probes.json',
               full_counts=root/'prepared/full_nuclear_counts.npy',nucleus_metadata=root/'prepared/nucleus_metadata.npz')
    prepared=read(config['prepared_manifest'])
    return dict(producers={p:sha256(REPO/p) for p in producers},inputs={k:sha256(p) for k,p in paths.items()},
                native_inputs={k:v['sha256'] for k,v in prepared['dependencies'].items()},
                raw_maps=[r['source']['sha256'] for r in prepared['raw_maps']],arm=config['arm'],
                optimizer=dict(lr=1e-5,betas=[.9,.999],weight_decay=1e-4),precision='FP32')


def joint_stats(rows,eta,lam):
    records=[]
    for row in rows:
        g=np.asarray(row['gram'],np.float64);v=np.array([0.,1.,eta]);n7=float(np.sqrt(max(0.,v@g@v)))
        n6,nE,nR=row['norms'];denom=nE+eta*nR
        records.append(dict(state=row['state'],view_cursor=row['view_cursor'],norm6=n6,norm7=n7,
                            ratio=lam*n7/n6 if n6>1e-12 else None,
                            cos_base=float(g[0]@v/(n6*n7)) if n6*n7>1e-24 else None,
                            cos_er=float(g[1,2]/(nE*nR)) if nE*nR>1e-24 else None,
                            cancellation=n7/denom if denom>1e-12 else None))
    return records


def prepare_dev(config):
    root=Path(config['runroot']);arm=config['arm'];seed_all()
    model=make_model(arm,config['initial']);opt=optimizer(model)
    data=NativeViews(root,morphology=arm=='V2_MORPH');base=BaseSix('cuda');extra=loss_module(config,'cuda')
    return model,opt,data,base,extra,read(root/'configs/probes.json')


def save_calibration(root,arm,record):
    q=read(Path(root)/'configs/queue.json');config=read(q['methods'][arm]['config'])
    record['dependencies']=calibration_dependencies(config)
    atomic_json(Path(root)/'admission'/f'{arm}_calibration.json',json_safe(record))
    return record


def calibrate_boundary(config):
    start=time.monotonic();model,opt,data,base,extra,probes=prepare_dev(config)
    rows=probe(model,data,base,extra,config,probes['boundary'],'initial')
    chosen=active(rows,0,1);development=[];state='initial'
    if len(chosen)<16:
        development=temporary_updates(model,opt,data,base,extra,config,probes['development'],0,32)
        later=probe(model,data,base,extra,config,probes['boundary'],'base32');rows+=later;chosen=active(later,0,1);state='base32'
    rec=dict(version=VERSION,arm=config['arm'],status='failed',probe_rows=rows,development=development,
             calibration_state=state,valid_views=len(chosen),eta=0.,source=source_digest(),
             fresh_initial=identity(config['initial']),probes=identity(Path(config['runroot'])/'configs/probes.json'))
    if len(chosen)<16:
        rec['reason']='INACTIVE_BOUNDARY: fewer than 16 finite nonzero full-model paired gradients after prescribed fallback'
        save_calibration(config['runroot'],config['arm'],rec);raise RuntimeError(rec['reason'])
    ratio=float(np.median([r['norms'][1]/r['norms'][0] for r in chosen]));lam=.30/ratio
    if not np.isfinite(lam):raise ValueError('nonfinite boundary calibration coefficient')
    rec.update(lambda_max=lam,median_unweighted_gradient_ratio=ratio,
               sensitivity={str(scale):joint_stats(rows,0,lam*scale) for scale in (.5,1.,2.)},
               status='calibrated',wall_seconds=time.monotonic()-start)
    save_calibration(config['runroot'],config['arm'],rec)
    del model,opt;gc.collect();torch.cuda.empty_cache();return rec


def calibrate_region(config):
    start=time.monotonic();model,opt,data,base,extra,probes=prepare_dev(config)
    rows=probe(model,data,base,extra,config,probes['region'],'initial');dev=[]
    for start_step,end in ((0,32),(32,64)):
        dev+=temporary_updates(model,opt,data,base,extra,config,probes['development'],start_step,end)
        rows+=probe(model,data,base,extra,config,probes['region'],f'base{end}')
    rec=dict(version=VERSION,arm=config['arm'],status='failed',source=source_digest(),
             fresh_initial=identity(config['initial']),probes=identity(Path(config['runroot'])/'configs/probes.json'),
             development=dev,probe_rows=rows)
    es=[r for r in rows if r['norms'][1]>1e-12];rs=[r for r in rows if r['norms'][2]>1e-12]
    if len(es)<8 or len([r for r in rows if r['norms'][0]>1e-12])<8:
        rec['reason']='INACTIVE_EXPRESSION_OR_BASE';save_calibration(config['runroot'],config['arm'],rec);raise RuntimeError(rec['reason'])
    if len(rs)<8:
        paired=active([r for r in rows if r['state']=='base64'],0,1)
        if not paired:raise RuntimeError('no base64 E pair for root fallback')
        lamE=.1*float(np.median([r['norms'][0]/r['norms'][1] for r in paired]))
        rec['fallback_lambdaE']=lamE
        # Prescribed no-warmup fallback at steps64..95; implemented explicitly.
        for a,b in ((64,80),(80,96)):
            model.train()
            for step in range(a,b):
                batch=data.get_view(probes['development'][step],'cuda');opt.zero_grad(set_to_none=True)
                z,l6,new,_,diag=forward_losses(model,batch,base,extra,config['arm'])
                loss=l6+lamE*new[0];loss.backward()
                if not torch.isfinite(loss) or not finite_gradients(model):raise FloatingPointError('region fallback nonfinite')
                opt.step()
            rows+=probe(model,data,base,extra,config,probes['region'],f'expression{b}')
        es=[r for r in rows if r['norms'][1]>1e-12];rs=[r for r in rows if r['norms'][2]>1e-12]
    if len(rs)<8:
        rec.update(calibration_status='ROOT_INACTIVE',eta=1.,reason='ROOT_INACTIVE: fewer than eight root gradient probes after base64 + expression32 fallback')
        save_calibration(config['runroot'],config['arm'],rec);raise RuntimeError(rec['reason'])
    paired=active(rows,1,2)
    raw_eta=float(np.median([r['norms'][1]/r['norms'][2] for r in paired])) if len(paired)>=8 else float(np.median([r['norms'][1] for r in es])/np.median([r['norms'][2] for r in rs]))
    eta=float(np.clip(raw_eta,1e-3,1e3));stats=joint_stats(rows,eta,1.)
    # RR-V1.2: eta retains its pooled rule. Joint strength is measured only
    # in the prespecified final development state; mixing fresh/base32/base64
    # activation regimes failed the real nominal64 ratio gate (archived).
    lambda_state='expression96' if 'fallback_lambdaE' in rec else 'base64'
    valid=[r for r in stats if r['state']==lambda_state and r['norm6']>1e-12 and r['norm7']>1e-12]
    if len(valid)<8:raise RuntimeError('RR-V1.2 final-state joint calibration has fewer than eight valid probes')
    lam=.3*float(np.median([r['norm6']/r['norm7'] for r in valid]))
    joint=joint_stats(rows,eta,lam)
    cancel=[r['cancellation'] for r in joint if r['cancellation'] is not None and r['cos_er'] is not None]
    fraction=float(np.mean(np.asarray(cancel)<.1)) if cancel else 0.
    if fraction>.2:raise RuntimeError('region severe vector cancellation')
    rec.update(status='calibrated',calibration_status='ACTIVE',calibration_protocol='RR-V1.2',lambda_state=lambda_state,
               lambda_rule='.3 * median(norm6 / norm(E+eta*R)) on final prespecified state only',lambda_valid_probes=len(valid),
               eta=eta,eta_raw=raw_eta,eta_clipped=eta!=raw_eta,
               eta_rule='paired_median' if len(paired)>=8 else 'unpaired_fallback',lambda_max=lam,joint=joint,
               cancellation_fraction=fraction,gram_note='FP64 exact full-vector inner products; ||gE+eta*gR||² = v^T Gram v, not sum of norms',
               wall_seconds=time.monotonic()-start)
    save_calibration(config['runroot'],config['arm'],rec)
    del model,opt;gc.collect();torch.cuda.empty_cache();return rec


def calibrate(config):
    config=read(config) if isinstance(config,(str,Path)) else config
    if not torch.cuda.is_available():raise RuntimeError('calibration requires host CUDA')
    before=calibration_dependencies(config)
    if config['arm']=='V1_REGION':result=calibrate_region(config)
    elif config['arm']=='V1_BOUNDARY':result=calibrate_boundary(config)
    else:result=dict(status='calibrated',version=VERSION,arm=config['arm'],lambda_max=0.,eta=0.,base_only=True,source=source_digest())
    if before!=calibration_dependencies(config):
        result['status']='obsolete';result['reason']='producer/input changed during calibration'
        save_calibration(config['runroot'],config['arm'],result);raise RuntimeError(result['reason'])
    result['dependencies']=before
    return save_calibration(config['runroot'],config['arm'],result)
