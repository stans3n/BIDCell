"""Real A/B/C 32-update finite admission, strict resume and probe isolation."""
import argparse,copy,hashlib,json,time,subprocess,sys
from pathlib import Path
import torch
from scripts.ssp.config import read
from scripts.ssp.training import forward,dataset,make_model,optimizer,losses,dependencies,save,load,rng_state,restore_rng
from scripts.ssp.loss_diagnostics import get,preserve,training_probe,groups,norm,record_update
from scripts.cos.tests.real_model import tensor_hashes
from scripts.crgf.common import atomic_json,identity,digest,sha256,source_identity
from bidcell.model.dataio.dataset_input import DataProcessing

def state(model,opt,ds):return dict(model=tensor_hashes(model.state_dict()),optimizer=tensor_hashes(opt.state_dict()),rng=tensor_hashes(rng_state()),grads={n:tensor_hashes(p.grad) for n,p in model.named_parameters()},modes={n:m.training for n,m in model.named_modules()},transform=list(ds.view_transform))
def update(model,opt,s,cfg):
    started=time.monotonic()
    opt.zero_grad(set_to_none=True);z=forward(model,s);value,parts=losses(z,s,True,cfg.anchor_weight,cfg.pos_background_rho)
    if not torch.isfinite(value):raise FloatingPointError('nonfinite loss')
    value.backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError('missing/nonfinite gradient')
    before={n:p.detach().clone() for n,p in model.named_parameters()};opt.step()
    decay=tuple(cfg.weight_decay*before[n] for n,p in model.named_parameters())
    delta=tuple(p.detach()-before[n] for n,p in model.named_parameters());norms={g:norm(delta,inds) for g,inds in groups(model).items()}
    assert norms['old_shared']>0
    if any(not torch.isfinite(p).all() for p in model.parameters()):raise FloatingPointError('nonfinite update')
    assert all(torch.isfinite(v).all() for values in opt.state.values() for v in values.values() if torch.is_tensor(v)), 'nonfinite Adam state'
    return dict(seconds=time.monotonic()-started,foreground_probability_mean=float(z.softmax(1)[:,1].mean()),weight_decay_norms={g:norm(decay,inds) for g,inds in groups(model).items()},gradient_norms={g:norm(tuple(p.grad for p in model.parameters()),inds) for g,inds in groups(model).items()},loss=float(value.detach()),parts={k:float(v.detach()) for k,v in parts.items()},actual_adam_update_norms=norms)

def main(root):
    from scripts.ssp.training import TRAIN_SOURCES
    execution_source=source_identity(TRAIN_SOURCES+['scripts/ssp/tests/numeric.py'])
    root=Path(root);out=root/'tests/numeric';out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);start=time.monotonic();evidence={};initial=None;initial_rng=None
    for arm in ('S','SP'):
        probe_root=out/arm;(probe_root/'configs').mkdir(parents=True,exist_ok=True);(probe_root/'configs/training_probes.json').write_text((root/'configs/training_probes.json').read_text())
        cfg=read(root/arm/'configs/run.json');ds=dataset(cfg);views=json.loads((root/'configs/views.json').read_text())['views'];used=[v for v in views if v['k']][:33]
        model=make_model(cfg,arm,len(ds.gene_names));opt=optimizer(model,cfg);model.train()
        ih=tensor_hashes(model.state_dict());initial_rng_hash=digest(tensor_hashes(rng_state()));assert len(opt.state)==0
        assert sum(p.numel() for p in model.cos_residual.parameters())==321 and all(torch.count_nonzero(p)==0 for p in model.cos_residual.parameters())
        assert sha256(root/arm/'configs/views.json')==sha256(root/'configs/views.json'), 'arm views differ'
        if initial is None:initial=ih;initial_rng=initial_rng_hash
        else:assert {k:v for k,v in ih.items() if not k.startswith('prototype_residual.')}==initial and initial_rng_hash==initial_rng
        deps=dependencies(cfg,arm);rows=[]
        for i,v in enumerate(used[:32]):
            rows.append(dict(update=i+1,**update(model,opt,get(ds,v),cfg)))
            if i==15:
                path=out/f'{arm}_resume.pth';cursor=views.index(v)+1;save(path,model,opt,deps,16,cursor,{x['index'] for x in used[:16]});atomic_json(path.with_suffix('.json'),dict(update=16,cursor=cursor,checkpoint=identity(path),dependencies_hash=digest(deps)))
                for key,badvalue in [('anchor_weight',cfg.anchor_weight+1),('pos_background_rho',.75),('pos_region_version','tampered'),('calibration_identity','tampered'),('arm_id','other')]:
                    bad=copy.deepcopy(deps);bad['config'][key]=badvalue
                    try:load(path,model,opt,bad)
                    except ValueError:pass
                    else:raise AssertionError('strict resume accepted '+key)
                for key in ('source','views','prepared','prepared_outputs','inputs'):
                    bad=copy.deepcopy(deps);bad[key]='tampered'
                    try:load(path,model,opt,bad)
                    except ValueError:pass
                    else:raise AssertionError('strict resume accepted '+key)
                for key in ('teacher_version','prototype_version','prepared_manifest'):
                    bad=copy.deepcopy(deps);bad['config'][key]='tampered'
                    try:load(path,model,opt,bad)
                    except ValueError:pass
                    else:raise AssertionError('strict resume accepted '+key)
                # Compute the same next view continuously and after a full disk restore.
                update(model,opt,get(ds,used[16]),cfg);expected=state(model,opt,ds)
                load(path,model,opt,deps);update(model,opt,get(ds,used[16]),cfg);actual=state(model,opt,ds)
                assert actual==expected,'next-step full state diverged after strict restore'
                atomic_json(out/f'{arm}_resume_expected.json',expected)
                if arm=='S':
                    subprocess.run([sys.executable,'-m','scripts.ssp.tests.numeric','--root',str(root),'--child',arm],check=True)
                    assert json.loads((out/'S_resume_child.json').read_text())==expected, 'new process strict resume diverged'
                load(path,model,opt,deps)
        # Real 8-probe wrapper restores all state including preexisting gradients.
        prior=state(model,opt,ds);training_probe(probe_root,arm,32,model,ds,cfg)
        assert state(model,opt,ds)==prior,'probe mutated model/optimizer/RNG/grad/mode/view'
        # Enabled vs disabled probe produces identical subsequent actual Adam step.
        path=out/f'{arm}_probe_state.pth';cursor=views.index(used[31])+1;save(path,model,opt,deps,32,cursor,{v['index'] for v in used[:32]});atomic_json(path.with_suffix('.json'),dict(update=32,cursor=cursor,checkpoint=identity(path),dependencies_hash=digest(deps)))
        update(model,opt,get(ds,used[32]),cfg);expected=state(model,opt,ds)
        load(path,model,opt,deps);training_probe(probe_root,arm,32,model,ds,cfg);update(model,opt,get(ds,used[32]),cfg);assert state(model,opt,ds)==expected,'probe next actual step diverged'
        if arm=='SP':
            assert rows[0]['gradient_norms']['prototype_first']==0
            assert any(r['gradient_norms']['prototype_final']>0 and r['actual_adam_update_norms']['prototype_final']>0 for r in rows)
            assert any(r['gradient_norms']['prototype_first']>0 and r['actual_adam_update_norms']['prototype_first']>0 for r in rows[1:])
        evidence[arm]=dict(peak_vram_bytes=torch.cuda.max_memory_allocated(),status='passed',updates=32,configuration=cfg.model_dump(mode='json'),initialization_hash=digest(ih),initial_rng_hash=initial_rng_hash,fresh_adam=True,added_zero_parameters=321,strict_resume_next_step_bitwise=True,rejected_dependency_changes=['anchor_weight','pos_background_rho','pos_region_version','calibration_identity','arm_id'],probe_full_state_and_next_update_bitwise=True,rows=rows)
        atomic_json(out/f'{arm}_complete.json',evidence[arm]);print(json.dumps(dict(arm=arm,status='passed',elapsed=time.monotonic()-start)),flush=True)
        del model,opt,ds;torch.cuda.empty_cache()
    assert evidence['S']['configuration']['anchor_weight']==evidence['SP']['configuration']['anchor_weight']
    assert evidence['S']['configuration']['calibration_identity']==evidence['SP']['configuration']['calibration_identity']=='c1f77e69c03e0f1768898989f6190e448140a0f805f8a8ed28d22de1e75cf64a'
    atomic_json(root/'diagnostics/numeric_admission.json',dict(status='passed',execution_source=execution_source,arms=evidence,initialization_all_tensors_and_bn_paired=True,views_hash=sha256(root/'configs/views.json'),temporary_models_discarded=True,peak_vram_bytes=torch.cuda.max_memory_allocated(),formal_checkpoints_absent=all(not (root/a/'checkpoints').exists() for a in ('S','SP')),torch_threads=torch.get_num_threads(),wall_seconds=time.monotonic()-start))
def child(root,arm):
    root=Path(root);out=root/'tests/numeric';cfg=read(root/arm/'configs/run.json');torch.set_num_threads(2)
    ds=dataset(cfg)
    model=make_model(cfg,arm,len(ds.gene_names));model.train();opt=optimizer(model,cfg);deps=dependencies(cfg,arm)
    step,cursor,seen=load(out/f'{arm}_resume.pth',model,opt,deps);assert step==16
    views=json.loads((root/'configs/views.json').read_text())['views'];view=next(v for v in views[cursor:] if v['k'])
    update(model,opt,get(ds,view),cfg);atomic_json(out/f'{arm}_resume_child.json',state(model,opt,ds))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--child');a=p.parse_args()
    if a.child:child(a.root,a.child)
    else:main(a.root)
