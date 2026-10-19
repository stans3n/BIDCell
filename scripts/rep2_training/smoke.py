"""Focused preprocessing equivalence, recovery, and disposable real-model tests."""
import argparse
import ast
import copy
import gc
import tempfile
from pathlib import Path
from types import SimpleNamespace
import h5py
import numpy as np
import pandas as pd
import tifffile
import torch
from .common import *

def cpu_tests():
    import cv2
    from .preprocess import patch_starts
    from .prepare import make_views
    from bidcell.processing.transcripts import process_gene_chunk
    from bidcell.processing.transcript_patches import generate_patches
    result={}
    with tempfile.TemporaryDirectory(dir=ROOT/'tests',prefix='cpu_fixture_') as td:
        tmp=Path(td)
        # Same round-to-even coordinates, same duplicate accumulation, same uint8.
        frame=pd.DataFrame(dict(gene=['A','A','B','A','B'],x=[.5,1.5,2.5,1.5,7.0],y=[.5,1.5,2.5,1.5,7.0]))
        process_gene_chunk(['A','B'],frame,8,8,str(tmp),0,0,'gene','x','y',None)
        got=np.zeros((8,8,2),np.uint16)
        np.add.at(got,(frame.y.round().to_numpy(int),frame.x.round().to_numpy(int),pd.Categorical(frame.gene,categories=['A','B']).codes),1)
        for i,g in enumerate(['A','B']):assert np.array_equal(got[...,i].astype(np.uint8),tifffile.imread(tmp/f'{g}_0_0.tif'))
        result['expression_round_duplicates_uint8']='exact'
        maps=tmp/'expr_maps';maps.mkdir();values=np.arange(112*110*3,dtype=np.uint8).reshape(112,110,3)
        with h5py.File(maps/'all_genes_0_103_0_107.hdf5','w') as f:f['data']=values
        cfg=SimpleNamespace(files=SimpleNamespace(data_dir=str(tmp),dir_out_maps='expr_maps',dir_patches='patches_'),model_params=SimpleNamespace(patch_size=48))
        generate_patches(cfg)
        expected={(y,x) for y in patch_starts(103) for x in patch_starts(107)}
        actual=set()
        for path in (maps/'patches_48x48_shift_0').glob('*.hdf5'):
            y,x=map(int,path.stem.split('_'));actual.add((y,x))
            with h5py.File(path) as f:assert np.array_equal(f['data'][:],values[y:y+48,x:x+48])
        assert actual==expected;result['all_shift0_boundary_patches']='exact native set and tensors'
        mask=np.arange(9*11,dtype=np.int32).reshape(9,11);scale=.2125
        enlarged=cv2.resize(mask,(int(np.round(11/scale)),int(np.round(9/scale))),interpolation=cv2.INTER_NEAREST)
        xx=pd.Series([0.,.10625,.31875,2.5,8.5]).div(scale).round().to_numpy(int)
        yy=pd.Series([0.,.10625,.31875,2.5,7.5]).div(scale).round().to_numpy(int)
        loop=np.asarray([enlarged[y,x] for x,y in zip(xx,yy)])
        assert np.array_equal(loop,enlarged[yy,xx]);result['nuclear_source_grid_rounding']='exact scalar/native indexing'
        class FakeDataset(SimpleNamespace):
            def __len__(self):return len(self.fp_patches)
        fake=FakeDataset(fp_patches=[str(tmp/f'{y}_0.hdf5') for y in range(0,240,48)],nuclei=np.zeros((240,48),np.int32))
        fake.nuclei[49:55,2:5]=1;fake.nuclei[99:104,2:5]=2;fake.nuclei[200:210,2:5]=3
        path=Path('/home/zwan0726/work/projects/BIDCell_crgf_1008_20261008T031528/scripts/crgf/training.py')
        tree=ast.parse(path.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='make_views')
        env={'np':np,'Path':Path};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),env)
        expected=env['make_views'](fake,SimpleNamespace(seed=1234,updates=8));actual=make_views(fake,8,b1=True)
        assert [(v['index'],v['flip'],v['rotation'],v['k'],v['epoch']) for v in expected['views']]==[(v['index'],v['flip'],v['rot90'],v['K'],v['epoch']) for v in actual['views']]
        result['B1_canonical_sequence_multiple_epochs']='exact'
        a=make_views(fake,8);b=make_views(fake,8);assert a==b and a['empty_skips']>0;result['matched_sequence_empty_counts']='exact'
    from .training import checkpoint_state,validate_state
    tiny=torch.nn.Linear(2,1);opt=torch.optim.Adam(tiny.parameters(),lr=1e-5)
    tiny(torch.ones(1,2)).sum().backward();opt.step()
    manifest={'views':[{'K':0,'index':0},{'K':1,'index':1}]}
    deps={'fixture':'counter_validation'}
    state=checkpoint_state(tiny,opt,'FIXTURE',deps,1,2,1,[{'update':1,'view':manifest['views'][1]}])
    validate_state(state,manifest,deps,1)
    for name,mutate in [('updates',lambda s:s.update(actual_updates=2)),('skips',lambda s:s.update(empty_skips=0)),('dependency',lambda s:s.update(dependencies={})),('Adam_counter',lambda s:s['optimizer']['state'][0]['step'].fill_(2))]:
        bad=copy.deepcopy(state);mutate(bad)
        try:validate_state(bad,manifest,deps,1)
        except ValueError:pass
        else:raise AssertionError('invalid state accepted '+name)
    result['forged_update_skip_Adam_dependency_rejected']=True
    atomic_json(ROOT/'tests/cpu_tests.json',dict(status='passed',tests=result))
    return result

def gpu_tests():
    from scripts.new_methods.common import make_model,optimizer,BaseSix
    from scripts.paired_v2.common import tensor_sha,state_equal,rng_state,restore_rng,MORPHOLOGY_INITIAL_SHA
    from .training import publish,recover,checkpoint_state
    from tests.new_methods.contracts import compare_tensors,adam_formula,adam_outlier_coordinates
    configure();results={};common_hash=None;global_initial_rng=None
    def cpu_state(state):
        if torch.is_tensor(state):return state.detach().cpu().clone()
        if isinstance(state,dict):return {k:cpu_state(v) for k,v in state.items()}
        if isinstance(state,list):return [cpu_state(v) for v in state]
        return copy.deepcopy(state)
    def opt_tensors(state):return {f'{k}.{f}':v for k,s in state['state'].items() for f,v in s.items() if torch.is_tensor(v)}
    for kind in ('BASE','V2_MORPH'):
        configure_seed();model=make_model(kind,read(ROOT/'configs/run.json')['shared_initial'],'cuda');opt=optimizer(model)
        shared={k:v for k,v in model.state_dict().items() if not k.startswith('morphology.')};h=tensor_sha(shared)
        if common_hash is None:common_hash=h
        else:assert h==common_hash
        initial_rng=rng_state()
        if global_initial_rng is None:global_initial_rng=initial_rng
        else:assert state_equal(initial_rng,global_initial_rng)
        if kind=='V2_MORPH':assert tensor_sha(model.morphology.state_dict())==MORPHOLOGY_INITIAL_SHA
        gen=torch.Generator().manual_seed(1234)
        x=torch.rand((2,313,48,48),generator=gen).cuda();n=torch.zeros((2,1,48,48),dtype=torch.long,device='cuda')
        n[0,0,10:15,10:15]=1;n[1,0,30:35,30:35]=1
        batch=dict(input=x,nuclei=n,support=torch.ones_like(n),positive=n,negative=1-n)
        aux=dict(geometry=torch.rand((2,16,48,48),generator=gen).cuda(),dapi=torch.rand((1,3,48,48),generator=gen).cuda()) if kind=='V2_MORPH' else None
        criterion=BaseSix('cuda');model.train();torch.cuda.reset_peak_memory_stats()
        def step():
            opt.zero_grad(set_to_none=True);z=model(x,aux);loss,parts=criterion(z,batch);loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            gradients={k:v.grad.detach().cpu().clone() for k,v in model.named_parameters()}
            opt.step();torch.cuda.synchronize()
            return dict(loss=float(loss.detach()),gradients=gradients,model=cpu_state(model.state_dict()),optimizer=cpu_state(opt.state_dict()),rng=rng_state())
        first=step();del first
        manifest=dict(views=[dict(K=2,index=i) for i in range(2)]);deps={'fixture':kind};rows=[dict(update=1,view=manifest['views'][0])]
        with tempfile.TemporaryDirectory(dir=ROOT/'tests',prefix='gpu_resume_') as td:
            out=Path(td);path=publish(out,checkpoint_state(model,opt,kind,deps,1,1,0,rows),manifest)
            expected=step()
            latest=recover(out,manifest,deps);assert latest==path
            def restore_exact():
                state=torch.load(latest,map_location='cpu',weights_only=False)
                model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);restore_rng(state['rng'])
                assert state_equal(state['model'],model.state_dict())
                assert state_equal(state['optimizer'],opt.state_dict())
                assert state_equal(state['rng'],rng_state())
                return state
            before=restore_exact();before_opt=copy.deepcopy(before['optimizer'])
            candidate=step()
            assert candidate['loss']==expected['loss'] and state_equal(candidate['rng'],expected['rng'])
            floor=None;repeat_summaries=[]
            for repetition in range(5):
                restore_exact();repeat=step()
                comp=dict(gradients=compare_tensors(expected['gradients'],repeat['gradients']),
                          model=compare_tensors(expected['model'],repeat['model'],kind='state'),
                          optimizer=compare_tensors(opt_tensors(expected['optimizer']),opt_tensors(repeat['optimizer']),kind='moments'))
                if floor is None:floor=copy.deepcopy(comp)
                else:
                    for field in floor:
                        for name,row in comp[field].items():
                            if row['rms']>floor[field][name]['rms']:floor[field][name]=row
                repeat_summaries.append({field:max(row['maxabs'] for row in values.values()) for field,values in comp.items()})
                del repeat;gc.collect()
            comparisons=dict(gradients=compare_tensors(expected['gradients'],candidate['gradients'],floor['gradients']),
                             model=compare_tensors(expected['model'],candidate['model'],floor['model'],kind='state'),
                             optimizer=compare_tensors(opt_tensors(expected['optimizer']),opt_tensors(candidate['optimizer']),floor['optimizer'],kind='moments'))
            formula_original=adam_formula(before['model'],expected['model'],expected['gradients'],before_opt,expected['optimizer'])
            formula_candidate=adam_formula(before['model'],candidate['model'],candidate['gradients'],before_opt,candidate['optimizer'])
            bn=[k for k in expected['model'] if k.endswith(('running_mean','running_var','num_batches_tracked'))]
            assert all(torch.equal(expected['model'][k],candidate['model'][k]) for k in bn)
            rows.append(dict(update=2,view=manifest['views'][1]));path2=publish(out,checkpoint_state(model,opt,kind,deps,2,2,0,rows),manifest)
            path2.write_bytes(b'interrupted publication fixture')
            assert recover(out,manifest,deps)==path
            detail=ROOT/'tests'/f'{kind}_resume_numerics.json'
            atomic_json(detail,dict(status='passed',comparisons=comparisons,native_repeat_floor=floor,repeats=repeat_summaries,
                                    original_adam_formula=formula_original,resumed_adam_formula=formula_candidate,
                                    outliers=adam_outlier_coordinates(dict(state=expected['model'],gradients=expected['gradients']),dict(state=candidate['model'],gradients=candidate['gradients']),comparisons['model'],before['model'])))
        results[kind]=dict(status='passed',initial_common_hash=h,restored_model_BN_Adam_RNG_exact=True,next_BN_RNG_loss_exact=True,
                           next_model_max_difference=max(row['maxabs'] for row in comparisons['model'].values()),
                           next_trajectory_bitwise=state_equal(expected['model'],candidate['model']),native_repeat_count=5,
                           native_repeat_model_max=max(r['model'] for r in repeat_summaries),independent_Adam_formula_passed=True,
                           numerical_details=identity(detail),corrupt_newest_falls_back=True,peak_vram_bytes=torch.cuda.max_memory_allocated(),disposable_update_calls=8)
        print(kind,json.dumps(results[kind]),flush=True)
        del model,opt,x,n,batch,aux,criterion,expected,candidate,before,before_opt,shared,step,restore_exact;gc.collect();torch.cuda.empty_cache()
    atomic_json(ROOT/'tests/gpu_training_smoke.json',dict(status='passed',tests=results))
    return results

def main():
    parser=argparse.ArgumentParser();parser.add_argument('kind',choices=['cpu','gpu']);args=parser.parse_args()
    print(json.dumps(cpu_tests() if args.kind=='cpu' else gpu_tests(),indent=2))

if __name__=='__main__':main()
