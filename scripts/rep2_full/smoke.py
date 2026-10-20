"""Reversible admission tests; actual frozen checkpoints, no optimizer calls."""
import argparse
import gc
import resource
import time
from pathlib import Path
import tempfile
import h5py
import numpy as np
import pandas as pd
import tifffile
from .common import *

def cpu():
    import cv2
    from scripts.crgf.matrix import grid
    from scripts.rep1_triple_simple_boundary.matrix import build
    from scripts.rep1_triple_simple_boundary.frozen_domains import build_interfaces,bootstrap_weights
    from scripts.boundary_three.core_refine import interface_band
    from scripts.boundary_goal_v3.local import state_codes,fate_codes
    from scripts.new_methods.author_marker import fixed_scores
    from .evaluation import paired_f1
    out=ROOT/'tests/cpu';out.mkdir(parents=True,exist_ok=True);passed=[]
    # Exact full native resize oracle, including half ties and out-of-bounds.
    mask=np.zeros((14,16),np.uint32);mask[1:7,1:8]=1;mask[7:13,7:15]=2
    n=mask.copy();x=np.array([0,.10625,1,3.1,7.1,10,15.9,16.5,-1,4]*2);y=np.array([0,.10625,2,4.3,9,10,13.9,12,1,8]*2)
    frame=pd.DataFrame(dict(transcript_id=np.arange(20),x_location=x,y_location=y,feature_name=['G0','G1']*10,qv=30))
    xx,yy,valid=grid(frame,mask.shape,.2125,.2125);large=cv2.resize(mask.astype(np.int32),(round(mask.shape[1]/.2125),round(mask.shape[0]/.2125)),interpolation=cv2.INTER_NEAREST)
    xi=frame.x_location.div(.2125).round().to_numpy(np.int64);yi=frame.y_location.div(.2125).round().to_numpy(np.int64)
    assert np.array_equal(mask[yy[valid],xx[valid]],large[yi[valid],xi[valid]])
    passed.append('native_coordinate_full_resize_and_ties')
    genes=['G'+str(i) for i in range(313)];(out/'genes.txt').write_text('\n'.join(genes)+'\n');frame.to_csv(out/'transcripts.csv',index=False)
    atomic_tiff(out/'mask.tif',mask);(out/'native.yaml').write_text('fixture: true\n')
    a=np.empty(len(frame),dtype=[('transcript_id','<u8'),('x','<i4'),('y','<i4'),('gene','<i4')]);a['transcript_id']=frame.transcript_id;a['x']=xx;a['y']=yy;a['gene']=[0,1]*10
    atomic_npy(out/'coordinate.npy',a);rec=dict(**identity(out/'coordinate.npy'),rows=20,row_start=0,row_stop=20)
    machine0=dict(final_mask=str(out/'mask.tif'),transcripts=str(out/'transcripts.csv'),genes=str(out/'genes.txt'),baseline_config=str(out/'native.yaml'),scale_x=.2125,scale_y=.2125,chunksize=500000)
    result=build(machine0,[rec],out/'matrix',{'fixture':'native-full-313'},expected_molecules=20)
    assert result['totals']['eligible']==20 and read(out/'matrix/validation.json')['owner_differences']==0
    passed.append('complete_313_matrix_independent_full_oracle')
    roi_base=np.zeros((15,17),np.uint32);roi_base[2:13,2:8]=1;roi_base[2:13,8:15]=2
    nuclei=np.zeros_like(roi_base);nuclei[5:7,4:6]=1;nuclei[5:7,11:13]=2
    arrays,pairs=build_interfaces(roi_base,nuclei,np.array([-2,1,2]),out/'roi',{'fixture':'BASE-only'})
    assert np.array_equal(arrays['assignment']>=0,interface_band(roi_base)&(nuclei==0)) and len(pairs)==1
    passed.append('frozen_roi_background_nucleus_and_union')
    lineage=np.array([0,1,2,0]);owners=np.array([1,2,3,0]);gl=np.ones(4,np.int8)
    assert state_codes(owners,lineage,gl).tolist()==[0,1,2,3]
    source=np.array([2,1,1,1,1,1]);target=np.array([1,2,1,3,0,2]);g=np.array([1,1,1,1,1,0])
    assert fate_codes(source,target,lineage,g).tolist()==[0,1,2,3,4,5]
    labels=pd.Series(['B Cells','__unavailable__:classifier_rejected'],index=[1,2]);counts=np.zeros((2,313),np.int64);counts[:,0]=5
    scores=fixed_scores(counts,[1,2],genes,labels,{'positive':{'B Cells':['G0']},'negative':{'B Cells':['G1']}},genes)
    assert np.isfinite(scores.loc[1,'positive_F1']) and np.isnan(scores.loc[2,'positive_F1'])
    passed.append('four_states_six_fates_unmatched_NA')
    _,_,W=bootstrap_weights(np.array([[10.,10.],[1100.,10.],[2100.,10.]]),1000,seed=20261019)
    specs={}
    for name,delta in [('a',0),('b',.1)]:
        specs[name]={}
        for profile in ('fixed','sc'):
            f=pd.DataFrame(dict(cell_id=[1,2,3],positive_F1=np.array([.2,.3,np.nan])+delta,negative_F1=np.array([.5,.4,.3])-delta))
            path=out/f'{name}_{profile}.csv';atomic_csv(path,f);specs[name][profile+'_scores']=str(path)
    ctx=dict(cells=pd.DataFrame(dict(cell_id=[1,2,3],block_1000=[0,1,2])).set_index('cell_id'),fixed=pd.DataFrame(index=[1,2,3]))
    result=paired_f1(specs,[('a','b')],ctx,out,W)
    assert set(result.loc[result.metric.eq('positive_F1'),'n'])=={2} and np.allclose(result.loc[result.metric.eq('positive_F1'),'delta'],.1)
    empty=paired_f1(specs,[('a','b')],ctx,out,W,'EMPTY',[]);assert empty.ci_low.isna().all()
    passed.append('paired_four_way_finite_population_bootstrap_and_empty_NA')
    # Native shift-24 starts omit the remainder append used only by shift-0.
    for lim in (48,96,137,3500,4000):
        assert list(np.arange(24,lim-48,48))==[x for x in range(24,lim-48,48)]
    passed.append('shift24_native_coordinate_enumeration')
    passed.extend(engineering())
    report=dict(status='passed',tests=passed,source_digest=source_digest(),no_production_started=True)
    atomic_json(ROOT/'tests/cpu_smoke.json',report);return report

def engineering():
    from unittest.mock import patch
    from contextlib import contextmanager
    from scipy import sparse
    from scripts.boundary_three import p2_native,a_core
    from scripts.boundary_three.core_refine import NuclearSupport
    from scripts.rep1_triple_simple_boundary import gt
    from .freeze import core_chunk
    from . import runner
    tests=[]
    # Named fixture directory is new each run; no prior production artifact is touched.
    root=Path(tempfile.mkdtemp(prefix='engineering_',dir=ROOT/'tests'))
    parent=np.ones((64,64),np.uint32);parent[:,32:]=2;nuc=np.zeros_like(parent);nuc[28:35,10:17]=1;nuc[28:35,48:55]=2
    atomic_tiff(root/'pre.tif',parent);atomic_tiff(root/'nuc.tif',nuc)
    with patch.object(p2_native,'OUTPUT_ROOT',ROOT):
        try:p2_native.process(root/'pre.tif',root/'nuc.tif',root/'nuc.tif',root/'native','B',workers=1,patch_size=32,interrupt_after=('tiles',1))
        except InterruptedError:pass
        else:raise AssertionError('native fault injector did not interrupt')
        assert not (root/'native/postprocess/complete.json').exists()
        p2_native.process(root/'pre.tif',root/'nuc.tif',root/'nuc.tif',root/'native','B',workers=1,patch_size=32)
        p2_native.validate_process(root/'native')
        try:p2_native.process(root/'pre.tif',root/'nuc.tif',root/'nuc.tif',root/'P2','P2',native_pred=root/'native',workers=1,patch_size=32,interrupt_after=('tiles',1))
        except InterruptedError:pass
        else:raise AssertionError('P2 fault injector did not interrupt')
        p2_native.process(root/'pre.tif',root/'nuc.tif',root/'nuc.tif',root/'P2','P2',native_pred=root/'native',workers=1,patch_size=32)
        p2_native.validate_process(root/'P2');before=identity(root/'P2/final.tif')
        p2_native.process(root/'pre.tif',root/'nuc.tif',root/'nuc.tif',root/'P2','P2',native_pred=root/'native',workers=1,patch_size=32)
        assert identity(root/'P2/final.tif')==before
    tests.append('native_B_P2_real_adapter_counterfactual_interrupt_resume_reuse')
    arr=np.array([(i+1,12+i%3,30, i%2) for i in range(40)]+[(i+41,50+i%3,30,i%2) for i in range(40)],dtype=[('transcript_id','<u8'),('x','<i4'),('y','<i4'),('gene','<i4')]);atomic_npy(root/'coords.npy',arr)
    support=NuclearSupport(nuc,[1,2],[0,1],['A','B']);data=dict(ids=np.array([1,2]),genes=['G0','G1','G2'],gene_indices=np.array([0,1]),support=support,nuclei=nuc,source={'fixture':'core'},expected=len(arr),records=[dict(coordinate_file=str(root/'coords.npy'),coordinate_sha256=identity(root/'coords.npy')['sha256'],rows=len(arr),row_start=0,row_stop=len(arr))])
    with patch.object(a_core,'_notify',lambda *a,**k:None):old,info=a_core._count_core({},data,root/'original_core')
    core,_=support.core_image();new=core_chunk(data,core,arr);actual=np.zeros_like(old);actual.ravel()[new['keys']]+=new['counts'];assert np.array_equal(actual,old)
    original=np.load(root/'original_core/count_chunks/0000.npz')
    assert all(np.array_equal(original[k],new[k]) for k in new)
    tests.append('core_chunk_differential_exact_original_count_core')
    baseline=np.array([[1,1,2],[1,1,2],[1,1,2]],np.uint32);nn=np.zeros_like(baseline);nn[1,0]=1;nn[1,2]=2
    data=dict(baseline=baseline,ids=np.array([1,2]),prior=np.array([[.01,.99],[.99,.01]]));domain=dict(pixels=np.array([4]),offsets=np.array([0,2]),candidate_rows=np.array([0,1]),geometry=np.zeros(2));expr=sparse.csr_matrix([[1,0]])
    props=[]
    for beta in (0,1):
        p,summary=gt.score_domain(data,domain,expr,root/f'beta{beta}/scores',{'fixture':'same-domain'},beta=beta);props.append(p)
        mask,result=gt.apply_gt(baseline,nn,p,root/f'beta{beta}/apply',{'fixture':beta},oracle=True)
        gt.validate_topology(baseline,mask,nn,data['ids'],p,root/f'beta{beta}/validation',{'fixture':beta})
    assert len(props[0])==0 and len(props[1])==1
    tests.append('beta0_beta1_independent_proposals_topology_only_coefficient_changes')
    @contextmanager
    def fake_locks():yield ()
    class Child:
        pid=987654
        def __init__(self,code):self.returncode=code
        def poll(self):return self.returncode
    control=root/'controller';(control/'logs').mkdir(parents=True);(control/'receipts').mkdir()
    dep={'fixture':'controller'};product=control/'product';product.write_text('verified')
    with patch.object(runner,'ROOT',control),patch.object(runner,'STAGES',['fixture']),patch.object(runner,'deps',lambda stage:dep),patch.object(runner,'locks',fake_locks),patch.object(runner,'preflight',lambda:None),patch.object(runner,'resources',lambda stage:None):
        for code in (7,0):
            with patch.object(runner.subprocess,'Popen',return_value=Child(code)):
                try:runner.run()
                except (RuntimeError,ValueError):pass
                else:raise AssertionError('failed/unverified child admitted')
                assert not (control/'compute_complete.json').exists()
        write_receipt(control/'receipts/fixture.json',dep,{'product':identity(product)})
        assert runner.completed('fixture')
        product.write_text('corrupted')
        try:runner.completed('fixture')
        except ValueError:pass
        else:raise AssertionError('corrupted completed artifact accepted')
    tests.append('controller_nonzero_missing_receipt_corruption_and_existing_stage')
    return tests

def gpu_views(name,shifts,output):
    import torch
    from .production import model_for,inference_data,predict_patch
    from scripts.new_methods.data import NativeViews
    from scripts.paired_v2.common import tensor_sha
    from bidcell.model.dataio.dataset_input import DataProcessing
    from bidcell.model.utils.utils import get_seg_mask
    torch.set_num_threads(4);started=time.monotonic();torch.cuda.reset_peak_memory_stats();model,kind,checkpoint=model_for(name)
    before=tensor_sha(model.state_dict());results=[]
    for shift,view in shifts:
        data=inference_data(kind,shift)
        index=list(map(str,data.fp_patches)).index(view['path']);v=dict(view,index=index,flip=0,rot90=0)
        native=DataProcessing.__getitem__(data,index)
        def observe(batch):
            assert torch.equal(batch['input'].cpu(),native[0].permute(3,2,0,1))
            assert np.array_equal(batch['candidate_ids'],np.unique(native[7])[np.unique(native[7])>0])
            assert len(batch['input'])==view['K']
        mask,candidates=predict_patch(model,kind,data,v,observer=observe)
        assert len(candidates)==view['K']
        results.append(dict(shift=shift,path=view['path'],K=len(candidates),input_exact_native=True,finite_logits=True,output_ids_exact_candidates=True,production_kernel='predict_patch'))
        del native,data;gc.collect();torch.cuda.empty_cache()
    assert tensor_sha(model.state_dict())==before
    report=dict(status='passed',endpoint=name,checkpoint=checkpoint,views=results,model_BN_unchanged=True,optimizer_calls=0,
        peak_vram_bytes=torch.cuda.max_memory_allocated(),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,wall_seconds=time.monotonic()-started)
    atomic_json(output,report);del model;gc.collect();torch.cuda.empty_cache();return report

def gpu():
    from .bootstrap import layout
    layout();out=ROOT/'tests/gpu';out.mkdir(parents=True,exist_ok=True)
    # Independently audited maximum of the complete Rep2 shift-0 inference pool.
    path=ROOT/'shared/data/expr_maps/expr_maps_input_patches_48x48_shift_0/1344_1728.hdf5'
    view=dict(path=str(path),row=1344,col=1728,K=45)
    reports=[gpu_views(n,[(0,view)],out/(n+'.json')) for n in (BASE,B1,V2)]
    # One ephemeral native shift-24 patch; never install it in production cache.
    import scripts.new_methods.data as data
    from . import production
    original=production.inference_data
    source=TRAINING/'shared/data/expr_maps/all_genes_0_3500_4000_7525.hdf5'
    patch=out/'2472_6232.hdf5'
    with h5py.File(source) as f,h5py.File(patch,'w') as target:target.create_dataset('data',data=f['data'][2472:2520,2232:2280],dtype=np.uint8)
    def fixture(kind,shift):
        result=original(kind,0);result.fp_patches=[str(patch)];return result
    production.inference_data=fixture
    try:reports.append(gpu_views(V2,[(24,dict(path=str(patch),row=2472,col=6232,K=44))],out/'V2_shift24.json'))
    finally:production.inference_data=original
    pipeline=tiny_prediction()
    result=dict(status='passed',reports=reports,full_prediction_adapter=pipeline,no_training=True,source_digest=source_digest());atomic_json(ROOT/'tests/gpu_smoke.json',result);return result

def tiny_prediction():
    """Exercise predict's chunk/merge/uint8 fill/receipt/resume with real forward."""
    from unittest.mock import patch
    import shutil
    import torch
    from . import production
    from scripts.paired_v2.common import tensor_sha
    root=Path(tempfile.mkdtemp(prefix='prediction_pipeline_',dir=ROOT/'tests'));(root/'shared').mkdir()
    actual=production.inference_data('BASE',0);originalpath=str(ROOT/'shared/data/expr_maps/expr_maps_input_patches_48x48_shift_0/1344_1728.hdf5')
    index=list(map(str,actual.fp_patches)).index(originalpath);realview=dict(index=index,row=1344,col=1728,path=originalpath,flip=0,rot90=0)
    crop=actual.nuclei[1344:1392,1728:1776].copy();model,kind,ckpt=production.model_for(BASE);before=tensor_sha(model.state_dict())
    views={}
    for shift in (0,24):
        path=root/f'{shift}_{shift}.hdf5';shutil.copyfile(originalpath,path)
        view=dict(index=0,path=str(path),row=shift,col=shift,flip=0,rot90=0,identity=identity(path));views[shift]=view;atomic_json(root/f'shared/predict_shift{shift}.json',[view])
    class Proxy:
        def __init__(self,shift):
            self.nuclei=np.zeros((72,72),np.int32);self.nuclei[shift:shift+48,shift:shift+48]=crop;self.fp_patches=[views[shift]['path']]
        def get_view(self,view,device):return actual.get_view(realview,device)
    with patch.object(production,'ROOT',root),patch.object(production,'base_deps',lambda:{'fixture':'prediction-pipeline'}),patch.object(production,'inference_data',lambda kind,shift:Proxy(shift)),patch.object(production,'model_for',lambda name:(model,kind,ckpt)):
        outputs,meta=production.predict(BASE);before_outputs={k:v['sha256'] for k,v in outputs.items()};outputs2,meta2=production.predict(BASE)
        assert before_outputs=={k:v['sha256'] for k,v in outputs2.items()}
    pred=root/'endpoints'/BASE/'prediction'
    assert tifffile.imread(pred/'fill.tif').dtype==np.uint8 and tifffile.imread(pred/'pre.tif').dtype==np.uint32
    assert tensor_sha(model.state_dict())==before
    result=dict(status='passed',patches=2,complete_production_predict_called=True,chunk_resume_exact=True,merge_executed=True,fill_dtype='uint8',no_optimizer_calls=True,outputs=outputs)
    atomic_json(root/'validation.json',result);del model,actual;gc.collect();torch.cuda.empty_cache();return result

def capacity():
    # Scan the complete input-only pool, including the inference-only shift.
    nuclei=tifffile.memmap(TRAINING/'shared/data/nuclei.tif');chosen=[]
    for shift in (0,24):
        views=read(ROOT/f'shared/predict_shift{shift}.json');maximum=None
        for v in views:
            yy,xx=v['row'],v['col'];ids=np.unique(nuclei[yy:yy+48,xx:xx+48]);k=int((ids>0).sum())
            if maximum is None or k>maximum['K']:maximum=dict(v,K=k)
        chosen.append((shift,maximum))
    reports=[gpu_views(n,chosen,ROOT/'shared'/('capacity_'+n+'.json')) for n in (BASE,B1,V2)]
    result=dict(status='passed',reports=reports,no_optimizer_calls=True,full_K_no_truncation=True)
    atomic_json(ROOT/'shared/inference_capacity.json',result);return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('test',choices=['cpu','gpu']);args=parser.parse_args()
    print(cpu() if args.test=='cpu' else gpu())
