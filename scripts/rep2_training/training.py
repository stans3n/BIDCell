"""Exactly counted FP32 updates and two-slot, receipt-validated recovery."""
import gc
import os
import signal
import time
from pathlib import Path
import numpy as np
import torch
from .common import *
from .data import Views
from scripts.new_methods.common import BaseSix, make_model, optimizer
from scripts.paired_v2.common import rng_state, restore_rng, tensor_sha, state_equal, MORPHOLOGY_INITIAL_SHA

def manifest_for(arm): return ROOT/'shared/configs'/('views_b1.json' if arm==B1 else 'views_matched.json')

def setup(arm, device='cuda'):
    configure();configure_seed();kind,target=ARMS[arm]
    if arm==B1:
        # Canonical crgf_1008 B1 starts from model tensors only, with fresh Adam.
        model=make_model('BASE',read(ROOT/'configs/run.json')['shared_initial'],device)
        parent=ROOT/'runs'/BASE/'checkpoints/final.pth'
        state=torch.load(parent,map_location='cpu',weights_only=False)
        if state['actual_updates']!=3625 or state['arm']!=BASE:raise ValueError('B1 parent mismatch')
        model.load_state_dict(state['model'],strict=True);del state
    else:model=make_model(kind,read(ROOT/'configs/run.json')['shared_initial'],device)
    opt=optimizer(model)
    data=Views(kind,full_pool=arm==B1)
    return model,opt,data

def initial_evidence(model,opt,arm):
    if opt.state:raise ValueError('initial Adam must have no moments')
    state=model.state_dict();common={k:v for k,v in state.items() if not k.startswith('morphology.')}
    if arm==B1:
        path=ROOT/'runs'/BASE/'checkpoints/final.pth';old=torch.load(path,map_location='cpu',weights_only=False)['model']
        if not state_equal(common,old):raise ValueError('B1 parent tensor/BN mismatch')
        parent=identity(path)
    else:
        path=Path(read(ROOT/'configs/run.json')['shared_initial']);old=torch.load(path,map_location='cpu',weights_only=False)
        if old.get('trained') is not False or old.get('updates')!=0 or old.get('seed')!=1234:raise ValueError('initial is trained/unverified')
        if not state_equal(common,old['model']):raise ValueError('common trunk/BN mismatch')
        parent=identity(path)
    morph=tensor_sha(model.morphology.state_dict()) if hasattr(model,'morphology') else None
    if morph is not None and morph!=MORPHOLOGY_INITIAL_SHA:raise ValueError('morphology initialization drift')
    return dict(arm=arm,seed=1234,common_trunk_tensor_sha=tensor_sha(common),morphology_tensor_sha=morph,
                parent=parent,optimizer_state_count=0,bn_included=True,actual_updates=0)

def dependencies(arm):
    deps=base_dependencies()
    deps.update(arm=arm,views=identity(manifest_for(arm)),prepared=identity(ROOT/'receipts/prepare.json'),
                initial=identity(ROOT/'manifests/initialization_audit.json'))
    if arm==B1:deps['parent']=identity(ROOT/'runs'/BASE/'checkpoints/final.pth')
    return deps

def finite(value):
    if torch.is_tensor(value) and value.is_floating_point() and not torch.isfinite(value).all():raise FloatingPointError('nonfinite checkpoint tensor')
    if isinstance(value,dict):
        for child in value.values():finite(child)
    elif isinstance(value,(tuple,list)):
        for child in value:finite(child)

def validate_state(state, manifest, deps, target=None):
    if state['dependencies']!=deps:raise ValueError('checkpoint dependencies differ')
    updates=state['actual_updates'];cursor=state['view_cursor'];skips=state['empty_skips']
    if not 0<=cursor<=len(manifest['views']):raise ValueError('bad view cursor')
    views=manifest['views'][:cursor]
    if updates!=sum(v['K']>0 for v in views) or skips!=sum(v['K']==0 for v in views):raise ValueError('checkpoint counter disagreement')
    if target is not None and updates!=target:raise ValueError('wrong actual update endpoint')
    rows=state['rows']
    if len(rows)!=updates:raise ValueError('effective log count mismatch')
    for number,(v,row) in enumerate(zip((v for v in views if v['K']>0),rows),1):
        if row['update']!=number or row['view']!=v:raise ValueError('effective access sequence differs')
    steps=[int(v['step'].item()) for v in state['optimizer']['state'].values()]
    if updates and (not steps or set(steps)!={updates}):raise ValueError('actual Adam call counter differs')
    if not updates and steps:raise ValueError('initial optimizer is trained')
    if state['arm'] in ARMS:
        cumulative=updates+(3625 if state['arm']==B1 else 0)
        if state.get('stage_actual_updates')!=updates or state.get('cumulative_actual_updates')!=cumulative:raise ValueError('cumulative update count mismatch')
        counters=[int(v.item()) for k,v in state['model'].items() if k.endswith('num_batches_tracked')]
        if not counters or set(counters)!={cumulative}:raise ValueError('BatchNorm forward count differs from cumulative updates')
    finite(state['model']);finite(state['optimizer'])
    return state

def checkpoint_state(model,opt,arm,deps,updates,cursor,skips,rows):
    return dict(arm=arm,dependencies=deps,model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state(),
                actual_updates=updates,view_cursor=cursor,empty_skips=skips,rows=rows,
                stage_actual_updates=updates,cumulative_actual_updates=updates+(3625 if arm==B1 else 0),
                bn_in_model_state=True,precision='FP32; no AMP/TF32',scheduler=None,scaler=None)

def publish(out,state,manifest):
    if 'fixture' not in state['dependencies']:
        resource_check('checkpoint', minimum_extra_bytes=sum(t.numel()*t.element_size() for t in state['model'].values())*3 + (16<<20))
    validate_state(state,manifest,state['dependencies'])
    # Two generations only. The new payload and its receipt are committed
    # separately; a crash between them leaves the other validated slot usable.
    valid=[]
    for i in (0,1):
        rp=out/f'slot_{i}.json'
        if rp.exists():
            r=read(rp)
            if (out/f'slot_{i}.pth').exists() and sha(out/f'slot_{i}.pth')==r['checkpoint']['sha256']:valid.append((r['actual_updates'],i))
    index=1-max(valid)[1] if valid else 0
    path=out/f'slot_{index}.pth';atomic_torch(path,state)
    rec=dict(actual_updates=state['actual_updates'],view_cursor=state['view_cursor'],empty_skips=state['empty_skips'],
             checkpoint=identity(path),dependencies_digest=digest(state['dependencies']))
    atomic_json(path.with_suffix('.json'),rec)
    atomic_json(out/'latest.json',rec)
    return path

def recover(out,manifest,deps):
    candidates=[];invalid=[]
    for i in (0,1):
        path=out/f'slot_{i}.pth';rp=path.with_suffix('.json')
        if not path.exists() and not rp.exists():continue
        if not path.exists() or not rp.exists():invalid.append(str(path));continue
        rec=read(rp)
        if rec['dependencies_digest']!=digest(deps):raise ValueError('checkpoint slot belongs to different inputs/source')
        if sha(path)!=rec['checkpoint']['sha256']:invalid.append(str(path));continue
        state=torch.load(path,map_location='cpu',weights_only=False);validate_state(state,manifest,deps)
        if rec['actual_updates']!=state['actual_updates'] or rec['view_cursor']!=state['view_cursor']:raise ValueError('checkpoint receipt counter mismatch')
        candidates.append((state['actual_updates'],path));del state
    if invalid:
        atomic_json(out/f'recovery_audit_{time.time_ns()}.json',dict(uncommitted_slots=invalid,valid_slots=[str(x[1]) for x in candidates]))
        if not candidates:raise ValueError('no validated checkpoint remains; inspect corrupt/unpublished slots before restarting')
    return max(candidates)[1] if candidates else None

def verify_complete(arm,receipt_path=None):
    rec=validate_receipt(receipt_path or ROOT/'receipts'/f'{arm}.json',dependencies(arm))
    state=torch.load(rec['outputs']['checkpoint']['path'],map_location='cpu',weights_only=False)
    validate_state(state,read(manifest_for(arm)),dependencies(arm),ARMS[arm][1])
    if state['arm']!=arm:raise ValueError('checkpoint arm identity differs')
    initial=read(ROOT/'runs'/arm/'initialization.json')
    if initial['optimizer_state_count']!=0:raise ValueError('Adam initial moments were not empty')
    if rec['actual_updates']!=ARMS[arm][1]:raise ValueError('stage receipt update count differs')
    if arm=='R2_V2_MORPH_NATIVE':
        base_initial=read(ROOT/'runs'/BASE/'initialization.json')
        if initial['common_trunk_tensor_sha']!=base_initial['common_trunk_tensor_sha']:raise ValueError('paired initialization mismatch')
        base=torch.load(ROOT/'runs'/BASE/'checkpoints/final.pth',map_location='cpu',weights_only=False)
        if [r['view'] for r in state['rows']]!=[r['view'] for r in base['rows']]:raise ValueError('paired effective sequence differs')
    return rec

def train(arm):
    kind,target=ARMS[arm];manifest=read(manifest_for(arm));deps=dependencies(arm)
    model,opt,data=setup(arm);out=ROOT/'runs'/arm;ckpt=out/'checkpoints';ckpt.mkdir(parents=True,exist_ok=True)
    if list(map(str,data.fp_patches))!=manifest['train_pool']:raise ValueError('native training pool changed')
    init=initial_evidence(model,opt,arm)
    if (out/'initialization.json').exists() and read(out/'initialization.json')!=init:raise ValueError('initialization changed')
    atomic_json(out/'initialization.json',init)
    updates=cursor=skips=0;rows=[]
    resume=recover(ckpt,manifest,deps)
    if resume:
        state=torch.load(resume,map_location='cpu',weights_only=False)
        model.load_state_dict(state['model'],strict=True);opt.load_state_dict(state['optimizer']);restore_rng(state['rng'])
        updates,cursor,skips,rows=(state[k] for k in ('actual_updates','view_cursor','empty_skips','rows'));del state
    logpath=out/'effective_sequence.jsonl'
    if logpath.exists():
        # Preserve superseded attempt lines without replaying committed updates.
        os.replace(logpath,out/f'effective_sequence_attempt_{time.time_ns()}.jsonl')
    logpath.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    inventory={obj['path']:obj for obj in read(ROOT/'shared/configs/patch_inventory.json')}
    criterion=BaseSix('cuda');model.train();torch.cuda.reset_peak_memory_stats();stop=[];started=time.monotonic()
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda s,f:stop.append(s))
    interval=read(ROOT/'configs/run.json')['checkpoint_every']
    with logpath.open('a',buffering=1) as log:
        while updates<target:
            if cursor>=len(manifest['views']):raise ValueError('frozen views exhausted')
            view=manifest['views'][cursor];verify_identity(inventory[view['path']]);begin=time.monotonic()
            if view['K']==0:
                # No native forward/BN/optimizer call on an empty patch.
                cursor+=1;skips+=1;progress(arm,status='running',actual_updates=updates,cumulative_actual_updates=updates+(3625 if arm==B1 else 0),view_cursor=cursor,empty_skips=skips,total=target)
                if stop:
                    publish(ckpt,checkpoint_state(model,opt,arm,deps,updates,cursor,skips,rows),manifest);raise SystemExit(128+stop[-1])
                continue
            batch=data.get_view(view,'cuda');cursor+=1
            opt.zero_grad(set_to_none=True);z=model(batch['input'],batch.get('aux'));loss,parts=criterion(z,batch)
            if not torch.isfinite(z).all() or not torch.isfinite(loss):raise FloatingPointError('nonfinite training output')
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError('missing/nonfinite model gradient')
            opt.step();updates+=1;torch.cuda.synchronize()
            row=dict(update=updates,view=view,patch_sha256=inventory[view['path']]['sha256'],empty_skips=skips,
                     base6=float(loss.detach()),terms={k:float(v.detach()) for k,v in parts.items()},seconds=time.monotonic()-begin)
            rows.append(row);log.write(json.dumps(row)+'\n');del z,loss,parts,batch
            if updates%interval==0 or updates in (1,target) or stop:
                log.flush();os.fsync(log.fileno());publish(ckpt,checkpoint_state(model,opt,arm,deps,updates,cursor,skips,rows),manifest)
            progress(arm,status='running',actual_updates=updates,cumulative_actual_updates=updates+(3625 if arm==B1 else 0),view_cursor=cursor,empty_skips=skips,total=target,
                     elapsed_seconds=time.monotonic()-started,peak_vram_bytes=torch.cuda.max_memory_allocated())
            if updates%10==0 or updates==1:print(arm,updates,'/',target,'visits',cursor,'skips',skips,'loss',row['base6'],flush=True)
            if stop:raise SystemExit(128+stop[-1])
    latest=recover(ckpt,manifest,deps);final=ckpt/'final.pth'
    if final.exists():
        if sha(final)!=sha(latest):raise ValueError('different existing final checkpoint')
    else:os.link(latest,final)
    rec=dict(status='succeeded',stage=arm,dependencies=deps,outputs={k:identity(v) for k,v in dict(checkpoint=final,sequence=logpath,initialization=out/'initialization.json').items()},
             actual_updates=updates,stage_actual_updates=updates,cumulative_actual_updates=updates+(3625 if arm==B1 else 0),
             view_cursor=cursor,empty_skips=skips,peak_vram_bytes=torch.cuda.max_memory_allocated())
    pending=out/'completion.pending.json';atomic_json(pending,rec)
    verify_complete(arm,pending)
    os.replace(pending,ROOT/'receipts'/f'{arm}.json')
    progress(arm,status='succeeded',actual_updates=updates,cumulative_actual_updates=updates+(3625 if arm==B1 else 0),view_cursor=cursor,empty_skips=skips,total=target)
    return rec

def runtime_validation():
    """Disposable real largest-K updates; no formal model state is retained."""
    results={};common_rng=None
    for arm in (BASE,'R2_V2_MORPH_NATIVE'):
        model,opt,data=setup(arm)
        current_rng=rng_state()
        if common_rng is None:common_rng=current_rng
        elif not state_equal(common_rng,current_rng):raise ValueError('fresh-arm dataset-constructed RNG differs')
        manifest=read(manifest_for(arm));view=max(manifest['views'],key=lambda v:v['K'])
        before=initial_evidence(model,opt,arm);torch.cuda.reset_peak_memory_stats();model.train()
        batch=data.get_view(view,'cuda');z=model(batch['input'],batch.get('aux'));loss,parts=BaseSix('cuda')(z,batch)
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError('largest-K smoke gradient failed')
        opt.step();torch.cuda.synchronize()
        results[arm]=dict(status='passed',K=view['K'],path=view['path'],loss=float(loss.detach()),disposable_updates=1,
                          peak_vram_bytes=torch.cuda.max_memory_allocated(),initial=before,common_initial_RNG_exact=True)
        del model,opt,data,batch,z,loss,parts;gc.collect();torch.cuda.empty_cache()
    # B1's full pool may have a larger candidate group than the matched subset.
    manifest=read(manifest_for(B1));view=max(manifest['views'],key=lambda v:v['K'])
    configure_seed();model=make_model('BASE',read(ROOT/'configs/run.json')['shared_initial'],'cuda');opt=optimizer(model);data=Views(full_pool=True)
    torch.cuda.reset_peak_memory_stats();batch=data.get_view(view,'cuda');z=model(batch['input']);loss,_=BaseSix('cuda')(z,batch);loss.backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError('B1 max-K gradient failed')
    opt.step();torch.cuda.synchronize()
    if not torch.isfinite(loss):raise FloatingPointError('B1 max-K capacity failed')
    results[B1]=dict(status='passed',K=view['K'],path=view['path'],disposable_updates=1,capacity_only=True,peak_vram_bytes=torch.cuda.max_memory_allocated())
    path=ROOT/'tests/real_rep2_capacity.json';atomic_json(path,results)
    return dict(capacity=path)
