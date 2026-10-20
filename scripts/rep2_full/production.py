"""Native both-shift inference, P2, and own complete matrices; no optimizer."""
import gc
import os
from pathlib import Path
import h5py
import natsort
import numpy as np
import pandas as pd
import tifffile
from .common import *

def shared():
    from scripts.new_methods.data import coords
    from scripts.crgf.matrix import grid
    c=config();maps=TRAINING/'shared/data/expr_maps';dest=ROOT/'shared/data/expr_maps/expr_maps_input_patches_48x48_shift_24'
    dest.mkdir(exist_ok=True);views=[]
    for source in natsort.natsorted(maps.glob('all_genes_*.hdf5')):
        y0,y1,x0,x1=coords(source)
        with h5py.File(source) as f:
            for y in np.arange(24,y1-y0-48,48):
                for x in np.arange(24,x1-x0-48,48):
                    path=dest/f'{y+y0}_{x+x0}.hdf5';rp=path.with_suffix('.json')
                    d=dict(source=identity(source),row=int(y),col=int(x),shape=[48,48,313],dtype='uint8') if not views or views[-1].get('_source')!=str(source) else None
                    # Source hash is computed once per map, never once per patch.
                    if d is not None:source_id=d['source']
                    cd=dict(source=source_id,row=int(y),col=int(x))
                    if rp.exists():check_receipt(rp,cd)
                    else:
                        a=f['data'][y:y+48,x:x+48]
                        if a.shape!=(48,48,313) or a.dtype!=np.uint8:raise ValueError('native shifted patch shape/dtype')
                        tmp=path.with_suffix('.tmp')
                        with h5py.File(tmp,'w') as out:out.create_dataset('data',data=a,dtype=np.uint8)
                        os.replace(tmp,path);write_receipt(rp,cd,{'patch':identity(path)})
                    views.append(dict(path=str(path),row=int(y+y0),col=int(x+x0),identity=identity(path),flip=0,rot=0,_source=str(source)))
        progress('shift24',len(views),map=source.name);resources('shared')
    for v in views:v.pop('_source',None)
    for shift in (0,24):
        if shift==0:
            views0=[dict(path=str(p),row=coords(p)[0],col=coords(p)[1],identity=identity(p),flip=0,rot=0)
                for p in natsort.natsorted((ROOT/'shared/data/expr_maps/expr_maps_input_patches_48x48_shift_0').glob('*.hdf5'))]
        else:views0=natsort.natsorted(views,key=lambda v:v['path'])
        for i,v in enumerate(views0):v['index']=i
        atomic_json(ROOT/f'shared/predict_shift{shift}.json',views0)
    genes=pd.Index((TRAINING/'shared/data/all_gene_names.txt').read_text().splitlines())
    out=ROOT/'shared/coordinates';records=[];cursor=0;allids=[]
    for i,chunk in enumerate(pd.read_csv(TRAINING/'shared/data/transcripts_processed.csv',usecols=['transcript_id','x_location','y_location','feature_name','qv'],chunksize=500000)):
        path=out/f'chunk_{i:04d}.npy';rp=path.with_suffix('.json');cd=dict(base=base_deps(),chunk=i,start=cursor,rows=len(chunk))
        if rp.exists():check_receipt(rp,cd);a=np.load(path,mmap_mode='r')
        else:
            if (chunk.qv<20).any():raise ValueError('processed transcript quality changed')
            g=genes.get_indexer(chunk.feature_name)
            if np.any(g<0):raise ValueError('unknown measured gene')
            x,y,valid=grid(chunk,c['shape'],.2125,.2125)
            a=np.empty(len(chunk),dtype=[('transcript_id','<u8'),('x','<i4'),('y','<i4'),('gene','<i4')])
            a['transcript_id']=chunk.transcript_id.to_numpy(np.uint64);a['x']=x;a['y']=y;a['gene']=g
            atomic_npy(path,a);write_receipt(rp,cd,{'coordinate':identity(path)})
        if len(a)!=len(chunk):raise ValueError('coordinate row coverage')
        records.append(dict(**identity(path),rows=len(a),row_start=cursor,row_stop=cursor+len(a)));cursor+=len(a);allids.append(a['transcript_id'].copy())
        progress('coordinates',i+1,molecules=cursor)
    ids=np.concatenate(allids);ids.sort()
    if np.any(ids[1:]==ids[:-1]) or cursor!=c['eligible']:raise ValueError('duplicate transcript IDs or denominator drift')
    atomic_json(out/'manifest.json',dict(records=records,rows=cursor,duplicate_transcript_ids=0,coordinate_contract='frozen scripts.crgf.matrix.grid; no refilter',source=identity(SOURCE/'scripts/crgf/matrix.py')))
    return stage_outputs(dest,out),{'shift24_patches':len(views),'rows':cursor}

def inference_data(kind,shift):
    configure()
    from scripts.new_methods.data import NativeViews
    return NativeViews(ROOT/'shared',training=False,shift=shift,auxiliary=kind=='V2_MORPH',morphology=kind=='V2_MORPH')

def model_for(name,device='cuda'):
    import torch
    from scripts.new_methods.common import make_model,seed_all
    configure();seed_all(1234)
    kind='V2_MORPH' if name==V2 else 'BASE'
    model=make_model(kind,TRAINING/'shared/prepared/shared_initial.pth',device)
    path=TRAINING/'runs'/ARMS[name]/'checkpoints/final.pth'
    s=torch.load(path,map_location='cpu',weights_only=False)
    target=4000 if name==B1 else 3625;cum=7625 if name==B1 else 3625
    if s['actual_updates']!=target or s['cumulative_actual_updates']!=cum or s['arm']!=ARMS[name]:raise ValueError('training receipt endpoint mismatch')
    if set(int(v['step']) for v in s['optimizer']['state'].values())!={target}:raise ValueError('Adam call counts differ')
    model.load_state_dict(s['model'],strict=True);model.eval()
    return model,kind,identity(path)

def predict_patch(model,kind,data,view,observer=None):
    """One scientific inference kernel shared by production and real-GPU smoke."""
    import torch
    from bidcell.model.utils.utils import get_seg_mask
    yy,xx=view['row'],view['col'];ids=np.unique(data.nuclei[yy:yy+48,xx:xx+48]);ids=ids[ids>0]
    if not len(ids):return np.zeros((48,48),np.uint32),[]
    with torch.inference_mode():
        batch=data.get_view(view,'cuda')
        if not np.array_equal(ids,batch['candidate_ids']):raise ValueError('complete native candidate IDs differ')
        if observer is not None:observer(batch)
        logits=model(batch['input'],batch['aux'] if kind=='V2_MORPH' else None)
        if not torch.isfinite(logits).all():raise FloatingPointError('nonfinite full native prediction')
        mask=get_seg_mask(logits.cpu().numpy(),batch['labels'].cpu().numpy()).astype(np.uint32)
    if mask.shape!=(48,48) or not set(np.unique(mask))<=set([0,*ids.tolist()]):raise ValueError('native prediction IDs/shape')
    return mask,ids.tolist()

def predict(name):
    import torch
    from bidcell.model.utils.utils import get_seg_mask
    from scripts.new_methods.prediction import merge_native
    out=ROOT/'endpoints'/name/'prediction';out.mkdir(parents=True,exist_ok=True)
    model,kind,checkpoint=model_for(name);torch.cuda.reset_peak_memory_stats();maps=[];maxk=0
    for shift in (0,24):
        data=inference_data(kind,shift);views=read(ROOT/f'shared/predict_shift{shift}.json')
        if list(map(str,data.fp_patches))!=[v['path'] for v in views]:raise ValueError('native inference order differs')
        whole=np.zeros(data.nuclei.shape,np.uint32);chunks=out/f'shift{shift}_patches';chunks.mkdir(exist_ok=True)
        for start in range(0,len(views),128):
            chosen=views[start:start+128];path=chunks/f'{start:06d}.npy';rp=path.with_suffix('.json')
            d=dict(base=base_deps(),checkpoint=checkpoint,shift=shift,start=start,patches=[v['identity'] for v in chosen])
            for v in chosen:check_identity(v['identity'])
            if rp.exists():rec=check_receipt(rp,d);masks=np.load(path)
            else:
                masks=np.zeros((len(chosen),48,48),np.uint32);ids=[]
                for j,v in enumerate(chosen):
                    masks[j],candidates=predict_patch(model,kind,data,v);ids.append(candidates)
                atomic_npy(path,masks);rec=write_receipt(rp,d,{'masks':identity(path)},candidate_ids=ids)
            if masks.shape!=(len(chosen),48,48) or masks.dtype!=np.uint32:raise ValueError('prediction cache shape/dtype')
            maxk=max(maxk,max(map(len,rec['candidate_ids']),default=0))
            for j,v in enumerate(chosen):whole[v['row']:v['row']+48,v['col']:v['col']+48]=masks[j]
            progress(name+'_predict',start+len(chosen),len(views),shift=shift,max_K=maxk,peak_vram=torch.cuda.max_memory_allocated())
        atomic_tiff(out/f'shift{shift}.tif',whole);maps.append(whole)
        shifted=list(data.fp_patches);del data;gc.collect()
    pre,fill=merge_native(*maps,shifted);atomic_tiff(out/'pre.tif',pre)
    from scripts.crgf.common import atomic_tiff as native_tiff
    native_tiff(out/'fill.tif',fill)
    return stage_outputs(out),dict(max_K=maxk,peak_vram=torch.cuda.max_memory_allocated(),checkpoint=checkpoint)

def cleanup(name):
    from scripts.boundary_three import p2_native
    p2_native.OUTPUT_ROOT=ROOT
    root=ROOT/'endpoints'/name;nuc=TRAINING/'shared/data/nuclei.tif'
    p2_native.process(root/'prediction/pre.tif',nuc,nuc,root,'B',workers=4,seed=1234,patch_size=1024)
    p2_native.validate_process(root);materialize_machine(name)
    return stage_outputs(root),dict(final=identity(root/'final.tif'),native_rule='B')

def p2():
    from scripts.boundary_three import p2_native
    p2_native.OUTPUT_ROOT=ROOT
    root=ROOT/'endpoints'/P2;nuc=TRAINING/'shared/data/nuclei.tif'
    p2_native.process(ROOT/'endpoints'/V2/'prediction/pre.tif',nuc,nuc,root,'P2',native_pred=ROOT/'endpoints'/V2,workers=4,seed=1234,patch_size=1024)
    p2_native.validate_process(root);materialize_machine(P2)
    return stage_outputs(root),dict(final=identity(root/'final.tif'),same_native_pre=identity(ROOT/'endpoints'/V2/'prediction/pre.tif'))

def matrix(name):
    from scripts.rep1_triple_simple_boundary.matrix import build,check_parent_conservation
    m=materialize_machine(name);out=Path(spec(name)['matrix'])
    result=build(m,coordinates(),out,dict(base_deps(),endpoint=name,mask=identity(m['final_mask'])),cfg_progress,expected_molecules=config()['eligible'])
    ids=pd.read_csv(out/'cell_mapping.csv').mask_label.to_numpy();meta=np.load(TRAINING/'shared/prepared/nucleus_metadata.npz')
    if not np.array_equal(ids,meta['ids']) or len(pd.read_csv(out/'gene_mapping.csv'))!=313:raise ValueError('full ID/gene universe differs')
    if name in (B0,BETA1):
        report=check_parent_conservation(spec(P2),result,out,len(ids),313,config()['eligible'])
        atomic_json(out/'parent_conservation.json',report)
    return stage_outputs(out),dict(totals=result['totals'],cells=len(ids),genes=313)
