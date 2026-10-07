"""Required technical smoke tests, not a scientific pilot or selection rule."""
import argparse
import copy
import importlib.util
import json
import os
import random
import subprocess
import tempfile
import time
from pathlib import Path
import cv2
import imgaug.augmenters as iaa
import numpy as np
import pandas as pd
import tifffile
import torch
from scipy.spatial.distance import cdist
from bidcell.model.dataio.candidate_geometry import CandidateGeometry,transform
from bidcell.model.dataio.dataset_input import DataProcessing
from bidcell.model.model.relation_geometry import RelationGeometry
from bidcell.model.model.model import SegmentationModel
from bidcell.model.postprocess_predictions import postprocess_connect,remove_islands,process_check_splits
from scripts.crgf.common import atomic_json,atomic_tiff,identity,sha256,digest
from scripts.crgf.config import read
from scripts.crgf.training import losses,make_model,optimizer,to_device,save,load,seed_all
from scripts.crgf.prediction import decode,merge_native
from scripts.crgf.postprocess import split_patch
from scripts.crgf.matrix import build
from scripts.crgf.evaluate import validate_matrix
from scripts.crgf import structural


def cpu(out):
    torch.set_num_threads(4);torch.manual_seed(12);result={}
    labels=np.zeros((14,16),np.int32);labels[2:10,3:5]=7;labels[2:4,5:9]=7;labels[9:12,10:14]=19
    provider=CandidateGeometry(labels);crop=labels[5:13,2:10];ids=np.unique(crop);ids=ids[ids>0]
    raw=np.arange(64).reshape(8,8).astype(np.float32)
    for flip in (0,1):
        for rotation in range(4):
            aug=iaa.Rot90(rotation,keep_size=True)(images=iaa.Flipud(flip)(images=raw[None,:,:,None]))[0,:,:,0]
            assert np.array_equal(aug,transform(raw,flip,rotation))
            nuc=transform(crop,flip,rotation);n=np.stack([nuc==i for i in ids],-1);support=n.copy()
            geometry,areas=provider.maps(ids,5,2,8,flip,rotation,n,support)
            yy,xx=np.mgrid[5:13,2:10]
            for k,i in enumerate(ids):
                d=cdist(np.column_stack((yy.ravel(),xx.ravel())),np.argwhere(labels==i)).min(1).reshape(8,8)
                expect=transform(np.minimum(d/(np.sqrt(areas[k]/np.pi)+1e-6),4)/4,flip,rotation)
                np.testing.assert_allclose(geometry[k,1],expect,atol=4e-8)
                assert areas[k]>int((crop==i).sum())
                assert np.array_equal(geometry[k,0],n[:,:,k])
    result['geometry_8_transforms_bruteforce_full_nucleus']='passed'
    module=RelationGeometry();assert sum(p.numel() for p in module.parameters())==10688
    f=torch.randn(3,128,4,4,requires_grad=True);g=torch.randn(3,4,8,8);s=torch.ones(3,1,8,8)
    y=module(f,g,s);perm=torch.tensor([2,0,1]);torch.testing.assert_close(module(f[perm],g[perm],s[perm])[torch.argsort(perm)],y,atol=1e-6,rtol=1e-5)
    cross=torch.autograd.grad(y[0].sum(),f,retain_graph=True)[0][1].abs().sum().item();assert cross>0
    f2=f.detach().clone();f2[1]+=torch.randn_like(f2[1]);assert not torch.equal(module(f2,g,s)[0],y[0])
    z,r,c,a=module.relations(f[:1],g[:1],s[:1]);assert r.count_nonzero()==0 and c.count_nonzero()==0
    assert torch.equal(module(f,g,torch.zeros_like(s)),f)
    one=torch.zeros_like(s);one[0]=1;z,r,c,a=module.relations(f,g,one);assert r[0].count_nonzero()==0 and c[0].count_nonzero()==0
    result['module']=dict(parameters=10688,cross_gradient=cross,permutation='close',singleton='zero_relation',no_support='identity')
    # Independently calculate all six native reductions and their gradients, K>1.
    logits=torch.randn(3,2,5,6,requires_grad=True);n=torch.zeros(3,1,5,6,dtype=torch.long);n[0,0,0,0]=1;n[1,0,2,3]=1;n[2,0,4,5]=1
    sample=dict(n=n,s=torch.randint(0,2,n.shape),p=torch.randint(0,2,n.shape),q=torch.randint(0,2,n.shape))
    total,parts=losses(logits,sample);prob=logits.softmax(1)[:,1];N=n[:,0];B=1-N.sum(0)
    independent=dict(ne=torch.nn.functional.cross_entropy(logits,N,reduction='sum')/(3*30),
      cc=torch.nn.functional.cross_entropy(logits,sample['s'][:,0],reduction='sum')/3,
      os=(torch.sigmoid(prob*(1-N)-.5).sum()-torch.sigmoid(prob*N-.5).sum()).relu()/3,
      ov=(torch.sigmoid(prob*B-.5).sum(0)-B).relu().sum()/30,
      pos=torch.nn.functional.cross_entropy(logits,sample['p'][:,0],reduction='sum')/3,
      neg=(torch.sigmoid(prob-.5)*sample['q'][:,0]).sum()/3)
    for key in independent:
        torch.testing.assert_close(parts[key],independent[key],atol=5e-6,rtol=1e-6)
        x=torch.autograd.grad(parts[key],logits,retain_graph=True)[0];z=torch.autograd.grad(independent[key],logits,retain_graph=True)[0]
        torch.testing.assert_close(x,z,atol=1e-7,rtol=1e-5)
    torch.testing.assert_close(torch.autograd.grad(total,logits)[0],torch.autograd.grad(sum(independent.values()),logits)[0],atol=2e-7,rtol=1e-5)
    result['six_losses_values_and_gradients']={k:float(v) for k,v in parts.items()}
    nuc=np.zeros((4,4),np.int32);nuc[0,0]=2;nuc[3,3]=9
    tie=np.zeros((2,2,4,4));answer=decode(tie,nuc,np.array([2,9]));assert answer[1,1]==0
    tie[:,1]=1;answer=decode(tie,nuc,np.array([2,9]));assert answer[1,1]==2
    assert decode(None,np.zeros((4,4)),np.array([],np.int64)).sum()==0
    # Native split original independently writes full field; compare optimized crop.
    seg=np.zeros((20,22),np.uint32);seg[2:4,3:6]=2;seg[12:16,10:14]=2;seg[1:3,17:20]=9;seg[16:19,2:5]=9
    nuclei=np.zeros_like(seg);nuclei[13,11]=2 # ID9 no nucleus overlap tests first component tie.
    from scipy import ndimage as ndi
    boxes=ndi.find_objects(seg);sub=out/'split';sub.mkdir(exist_ok=True)
    process_check_splits(None,str(sub),nuclei,seg,np.array([2,9]))
    expected=tifffile.imread(sub/'2_checked_splits.tif');actual=np.zeros_like(seg)
    for i in (2,9):actual[boxes[i-1]][split_patch(seg,nuclei,i,boxes[i-1])]=i
    assert np.array_equal(expected,actual)
    # Native no-background/empty guards.
    assert postprocess_connect(np.full((8,8),5,np.uint32),np.full((8,8),5,np.uint32)).min()==5
    assert remove_islands(np.full((8,8),5,np.uint32),np.full((8,8),5,np.uint32)).min()==5
    process_check_splits(None,str(sub),nuclei,seg,np.array([],int))
    result['native_split_crop_equivalence_and_boundaries']='passed'
    # Actual native fill_grid wrapper vs extracted code, including a coordinate gap.
    from types import SimpleNamespace as NS
    from bidcell.model.predict import fill_grid
    data=out/'fill_fixture';patches=data/'maps/patch_48x48_shift_24';patches.mkdir(parents=True,exist_ok=True)
    paths=[]
    for y in (24,72,168):
        for x in (24,72,168):
            p=patches/f'{y}_{x}.hdf5';p.touch();paths.append(str(p))
    pred_dir=data/'model_outputs/test/test_output';pred_dir.mkdir(parents=True)
    a=np.arange(240*240,dtype=np.uint32).reshape(240,240);b=a[::-1].copy()
    atomic_tiff(pred_dir/'epoch_1_step_1_seg_shift0.tif',a);atomic_tiff(pred_dir/'epoch_1_step_1_seg_shift24.tif',b)
    config=NS(model_params=NS(patch_size=48),files=NS(data_dir=str(data),dir_out_maps='maps',dir_patches='patch_'),experiment_dirs=NS(test_output_dir='test_output'),testing_params=NS(test_epoch=1,test_step=1))
    fill_grid(config,'test');merged,fill=merge_native(a,b,paths)
    assert np.array_equal(merged,tifffile.imread(pred_dir/'epoch_1_step_1.tif'))
    result['native_fill_grid_gap_edge_equivalence']='passed'
    # Shape changes with equal ID/area must have distinct morphology/dependency.
    old=structural.legacy();import logging
    a=np.zeros((20,20),np.uint32);b=a.copy();a[2:6,2:6]=1;b[2:4,2:10]=1
    sa=old.compute_fast_shape_metrics(a,1,logging.getLogger());sb=old.compute_fast_shape_metrics(b,1,logging.getLogger())
    assert sa.area_um2.iloc[0]==sb.area_um2.iloc[0] and sa.perimeter_px.iloc[0]!=sb.perimeter_px.iloc[0]
    result['own_shape_equal_area_different_shape']='passed'
    # Matrix boundaries: empty, no background, large label (above signed int32).
    for name,mask in [('empty',np.zeros((9,11),np.uint32)),('no_background',np.full((9,11),2**31+3,np.uint32))]:
        d=out/name;d.mkdir();atomic_tiff(d/'mask.tif',mask);(d/'genes.txt').write_text('A\nB\n');(d/'baseline.json').write_text('{}')
        pd.DataFrame(dict(transcript_id=[1,2,3],x_location=[1.,-3.,2.],y_location=[1.,1.,2.],feature_name=['A','A','B'],qv=[30,30,30])).to_csv(d/'ts.csv',index=False)
        c=dict(final_mask=str(d/'mask.tif'),genes=str(d/'genes.txt'),transcripts=str(d/'ts.csv'),baseline_config=str(d/'baseline.json'),scale_x=.2125,scale_y=.2125,chunksize=2)
        manifest=build(c,d/'matrix',cache=d/'cache');validate_matrix(c,d/'matrix')
        assert manifest['totals']['out_of_bounds']==1
    result['matrix_empty_no_background_large_ids']='passed'
    # Confirm exact audited kernel AST migrated, not a reimplementation.
    import ast
    audited=Path('/home/zwan0726/work/outputs/matrix_reaudit_20261008T023051+1100/scripts/generate.py')
    here=Path(__file__).parents[1]/'matrix.py'
    def kernels(p):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('lookup_axes','grid','aggregate')}
    assert kernels(audited)==kernels(here);result['matrix_kernel_ast_identical']=True
    atomic_json(out/'cpu_receipt.json',dict(status='succeeded',checks=result));return result


def gpu(cfg,out):
    seed_all(cfg.seed);torch.set_num_threads(4);results={};start=time.monotonic()
    ds=DataProcessing(cfg.native(),isTraining=True,total_steps=None,structured=True)
    views=json.loads((Path(cfg.output_root)/'configs/views.json').read_text())
    chosen=[v for v in views['views'] if v['k']>1][:3]
    first=chosen[0];index=first['index'];ds.view_transform=(0,0);identity_sample=ds[index]
    infer=DataProcessing(cfg.native(),isTraining=False,structured=True,geometry_provider=ds.geometry_provider)
    inference_sample=infer[infer.fp_patches.index(identity_sample['patch'])]
    for key in ('x','n','s','p','q','geometry'):assert torch.equal(identity_sample[key],inference_sample[key])
    del infer
    # Eight real views compare the unchanged native tuple outputs and complete geometry.
    legacy=DataProcessing(cfg.native(),isTraining=True,total_steps=None)
    edge_seen=False
    for flip in (0,1):
        for rot in range(4):
            ds.view_transform=legacy.view_transform=(flip,rot);s=ds[index];old=legacy[index]
            for key,k in [('x',0),('n',1),('s',2),('p',3),('q',4)]:
                expected=old[k].permute(3,2,0,1) if k==0 else old[k].permute(2,0,1).unsqueeze(1)
                assert torch.equal(s[key],expected)
            assert np.array_equal(s['cell_ids'],identity_sample['cell_ids'])
            np.testing.assert_allclose(s['geometry'][:,1].numpy(),np.stack([transform(x,flip,rot) for x in identity_sample['geometry'][:,1].numpy()]),atol=1e-7)
            edge_seen |= bool(np.any(s['full_areas']>s['n'].sum((1,2,3)).numpy()))
    assert edge_seen;del legacy;results['real_8_augment_native_and_identity_inference']='passed'
    # Direct original class from the exact git baseline source; disabled keys/output equality.
    code=subprocess.check_output(['git','show','2fae67ad74eaa9ec77b5d6fed6178f7cd9feca9d:bidcell/model/model/model.py'],text=True)
    ns={'__package__':'bidcell.model.model'};exec(compile(code,'baseline_model.py','exec'),ns)
    disabled=make_model(cfg,'B1',len(ds.gene_names));original=ns['SegmentationModel'](n_channels=len(ds.gene_names)).cuda()
    original.load_state_dict(torch.load(cfg.checkpoint,map_location='cpu')['model_state_dict'],strict=True);original.eval();disabled.eval();sample=to_device(identity_sample)
    with torch.inference_mode():torch.testing.assert_close(disabled(sample['x']),original(sample['x']),atol=0,rtol=0)
    del original,disabled;torch.cuda.empty_cache();results['disabled_baseline_exact']='passed'
    model=make_model(cfg,'F1',len(ds.gene_names));opt=optimizer(model,cfg);model.train();captured={};handles=[]
    for name in ('maxpool2','h2_PT_hd4','h2_PT_hd3','h2_Cat_hd2_conv'):
        handles.append(getattr(model,name).register_forward_pre_hook(lambda m,args,n=name:captured.__setitem__(n,args[0].detach().clone())))
    handles.append(model.relation_geometry.register_forward_hook(lambda m,args,value:captured.update(h2_in=args[0].detach().clone(),h2_out=value.detach().clone())))
    named=dict(model.named_parameters());watch=[k for k in named if any(k.startswith(v) for v in ['conv1.conv1.0.weight','conv2.conv1.0.weight','conv5.conv1.0.weight','outconv_seg.weight','relation_geometry.'])]
    # Actual layer names are saved dynamically; include each major block's first weight.
    watch=list(dict.fromkeys(watch+[next(k for k in named if k.startswith(v) and k.endswith('weight')) for v in ['conv1.','conv2.','conv5.','outconv_seg.']]))
    before={k:named[k].detach().cpu().clone() for k in watch}
    opt.zero_grad(set_to_none=True);logits=model(sample['x'],sample['geometry'],sample['s']);total,parts=losses(logits,sample);total.backward()
    for name in ('maxpool2','h2_PT_hd4','h2_PT_hd3','h2_Cat_hd2_conv'):assert torch.equal(captured[name],captured['h2_out'])
    grad={k:float(named[k].grad.norm()) for k in watch};assert all(np.isfinite(v) and v>0 for v in grad.values())
    opt.step();updates={k:float((named[k].detach().cpu()-before[k]).abs().max()) for k in watch};assert all(v>0 for v in updates.values())
    ratio=float((captured['h2_out']-captured['h2_in']).norm()/captured['h2_in'].norm());assert ratio>0
    for h in handles:h.remove()
    results['h2_all_consumers_gradients_updates']=dict(gradients=grad,max_updates=updates,feature_delta_relative_norm=ratio)
    dep=dict(inputs={'gene_order':ds.gene_names},test='strict full state resume')
    checkpoint=out/'temporary_resume.pth';save(checkpoint,model,opt,dep,1,1,{index})
    second=chosen[1];ds.view_transform=(second['flip'],second['rotation']);next_sample=to_device(ds[second['index']])
    def step():
        opt.zero_grad(set_to_none=True);y=model(next_sample['x'],next_sample['geometry'],next_sample['s']);l,_=losses(y,next_sample);l.backward();opt.step();return y.detach().cpu(),float(l.detach())
    y1,l1=step();grad1={k:p.grad.detach().cpu().clone() for k,p in model.named_parameters()};state1={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};random1=(random.random(),np.random.rand(),torch.rand(1).item(),torch.rand(1,device='cuda').item())
    stepn,cursor,seen=load(checkpoint,model,opt,dep);assert (stepn,cursor,seen)==(1,1,{index})
    checkpoint_opt=torch.load(checkpoint,map_location='cpu')['optimizer_state_dict']
    loaded_opt=opt.state_dict()
    assert checkpoint_opt['param_groups']==loaded_opt['param_groups']
    for pid,record in checkpoint_opt['state'].items():
        for key,value in record.items():torch.testing.assert_close(value,loaded_opt['state'][pid][key].cpu(),atol=0,rtol=0)
    y2,l2=step();random2=(random.random(),np.random.rand(),torch.rand(1).item(),torch.rand(1,device='cuda').item())
    atomic_json(out/'resume_forward_diagnostics.json',dict(logits_max_abs=float((y1-y2).abs().max()),losses=[l1,l2],RNG=[random1,random2]))
    torch.testing.assert_close(y1,y2,atol=1e-6,rtol=1e-5)
    assert abs(l1-l2)<=1e-5*max(abs(l1),1) and random1==random2
    differences={k:float((state1[k]-v.cpu()).abs().max()) for k,v in model.state_dict().items()}
    gradient_differences={k:float((grad1[k]-p.grad.cpu()).abs().max()) for k,p in model.named_parameters()}
    atomic_json(out/'resume_numerical_diagnostics.json',dict(model_and_BN_max_abs_by_tensor=differences,gradient_max_abs_by_tensor=gradient_differences,optimizer_at_restore_exact=True,logits_max_diff=float((y1-y2).abs().max()),loss_diff=l1-l2,RNG_exact=True,explanation='Both methods use torch deterministic algorithms with warn_only=True (native NLL CUDA forward unsupported) and CUBLAS_WORKSPACE_CONFIG=:4096:8; installed torch dispatches bilinear interpolation to deterministic decomposition. Earlier non-deterministic attempts retained separately.'))
    maxdiff=max(differences.values())
    for k,v in model.state_dict().items():torch.testing.assert_close(state1[k],v.cpu(),atol=1e-6,rtol=1e-5)
    del grad1,checkpoint_opt,loaded_opt
    results['strict_resume']=dict(logits_max_diff=float((y1-y2).abs().max()),loss_difference=l1-l2,all_model_and_BN_max_diff=maxdiff,RNG_equal=True,optimizer_restored_exact=True,FP32_update_tolerance=dict(atol=1e-6,rtol=1e-5),gradient_max_abs_diff=max(gradient_differences.values()))
    del state1,y1,y2,before
    timings=[]
    # A few real updates including largest K in the fixed sequence; no truncation.
    for view in chosen+[max(views['views'],key=lambda v:v['k'])]:
        t=time.monotonic();ds.view_transform=(view['flip'],view['rotation']);s=to_device(ds[view['index']]);loaded=time.monotonic()
        opt.zero_grad(set_to_none=True);y=model(s['x'],s['geometry'],s['s']);l,_=losses(y,s);assert torch.isfinite(l);l.backward();opt.step();torch.cuda.synchronize()
        timings.append(dict(k=view['k'],geometry_and_load_seconds=loaded-t,total_update_seconds=time.monotonic()-t,loss=float(l.detach())))
    results['temporary_updates_timing']=timings;results['peak_vram_bytes']=torch.cuda.max_memory_allocated()
    # Own prediction -> native morphology -> full-coordinate sparse full-shape mask -> matrix.
    model.eval()
    with torch.inference_mode():logits=model(sample['x'],sample['geometry'],sample['s']).cpu().numpy()
    patch=decode(logits,sample['nuclei'],sample['cell_ids']);random.seed(cfg.seed)
    final=remove_islands(postprocess_connect(patch,sample['nuclei']),sample['nuclei']);final=np.where(sample['nuclei']>0,sample['nuclei'],final).astype(np.uint32)
    whole=np.zeros(ds.nuclei.shape,np.uint32);row,col=sample['row'],sample['col'];whole[row:row+48,col:col+48]=final
    d=out/'own_output_full_coordinate_fixture';d.mkdir(exist_ok=True);atomic_tiff(d/'final.tif',whole)
    genes=ds.gene_names[:3];(d/'genes.txt').write_text('\n'.join(genes)+'\n');(d/'config.json').write_text('{}')
    # Probe exact source/global coordinates and ties-to-even cases, including OOB/background.
    yy,xx=np.nonzero(final);yy=yy[:100];xx=xx[:100]
    xs=np.r_[col+xx+0.1,col+.2125*.5,-5];ys=np.r_[row+yy+.1,row+.2125*1.5,0]
    pd.DataFrame(dict(transcript_id=np.arange(1,len(xs)+1),x_location=xs,y_location=ys,feature_name=np.resize(genes,len(xs)),qv=30)).to_csv(d/'transcripts.csv',index=False)
    c=dict(final_mask=str(d/'final.tif'),transcripts=str(d/'transcripts.csv'),genes=str(d/'genes.txt'),baseline_config=str(d/'config.json'),scale_x=.2125,scale_y=.2125,chunksize=40)
    manifest=build(c,d/'matrix',cache=d/'cache');validate_matrix(c,d/'matrix')
    assert manifest['contract']['shape']==list(ds.nuclei.shape)
    results['own_mask_to_matrix_full_coordinate_contract']=dict(shape=list(ds.nuclei.shape),totals=manifest['totals'],validation=manifest['validation'])
    results['wall_seconds']=time.monotonic()-start
    atomic_json(out/'gpu_receipt.json',dict(status='succeeded',checks=results,model_import=__import__('bidcell.model.model.model',fromlist=['x']).__file__))
    return results


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--part',choices=['cpu','gpu','all'],default='all');a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if a.part in ('cpu','all'):cpu(out)
    if a.part in ('gpu','all'):gpu(read(a.config),out)
    print('VALIDATION PASSED',a.part,flush=True)
if __name__=='__main__':main()
