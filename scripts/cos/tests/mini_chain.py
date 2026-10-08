"""Real regional input fixture; never confused with a production full-field run."""
import argparse,json,shutil,re,sys,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile,yaml
from scripts.crgf.common import atomic_json,identity
from scripts.crgf.matrix import lookup_axes
from scripts.cos.config import read,ARMS
from bidcell.model.dataio.dataset_input import DataProcessing

def prepare(queuepath,root):
    queue=json.loads(Path(queuepath).read_text());formal=Path(queue['output_root']);root=Path(root)
    if root.exists():raise FileExistsError(root)
    data=root/'data';data.mkdir(parents=True);(root/'configs').mkdir()
    cfg=read(queue['methods']['B1_CONT']['config']);native=cfg.native();olddata=Path(native.files.data_dir)
    source_views=json.loads((formal/'configs/views.json').read_text())['views']
    chosen=next(v for v in source_views if v['k']>=5 and all(int(n)<107900 for n in v['cell_ids']))
    y0,x0=map(int,re.findall(r'\d+',Path(chosen['patch']).name)[:2]);size=192
    nuclei=tifffile.imread(olddata/native.files.fp_nuclei)
    y0=min(y0,((nuclei.shape[0]-size)//48)*48);x0=min(x0,((nuclei.shape[1]-size)//48)*48)
    crop=nuclei[y0:y0+size,x0:x0+size];tifffile.imwrite(data/'nuclei.tif',crop)
    for name in (native.files.fp_nuclei_anno,native.files.fp_gene_names):shutil.copyfile(olddata/name,data/name)
    native_dict=yaml.safe_load(Path(cfg.baseline_config).read_text());native_dict['files']['data_dir']=str(data)
    baseline=data/'mini.yaml';baseline.write_text(yaml.safe_dump(native_dict))
    for shift in (0,24):
        src=olddata/native.files.dir_out_maps/(native.files.dir_patches+f'48x48_shift_{shift}')
        dest=data/native.files.dir_out_maps/src.name;dest.mkdir(parents=True)
        for path in src.glob('*.hdf5'):
            y,x=map(int,re.findall(r'\d+',path.name)[:2])
            if y0<=y<=y0+size-48 and x0<=x<=x0+size-48:
                name=re.sub(r'^\d+_\d+',f'{y-y0}_{x-x0}',path.name)
                if name==path.name and (y0 or x0):
                    nums=list(re.finditer(r'\d+',path.name));name=path.name[:nums[0].start()]+str(y-y0)+path.name[nums[0].end():nums[1].start()]+str(x-x0)+path.name[nums[1].end():]
                shutil.copyfile(path,dest/name)
    # Keep exact accepted uint64 row IDs and gene identity; express frozen raster
    # positions as local coordinates via the inverse of the same cv2 axes.
    cached=Path('/home/zwan0726/work/outputs/crgf_egr_20261008T121953+1100/shared/coordinates')
    manifests=list(cached.glob('*/complete.json'));assert len(manifests)==1
    cache=manifests[0].parent;cm=json.loads(manifests[0].read_text());pieces=[]
    genes=(data/native.files.fp_gene_names).read_text().splitlines()
    xa,ya=lookup_axes(crop.shape,.2125,.2125)
    ix=np.array([np.flatnonzero(xa==x)[0] for x in range(size)])*.2125;iy=np.array([np.flatnonzero(ya==y)[0] for y in range(size)])*.2125
    for entry in cm['chunks']:
        a=np.load(cache/entry['file']);keep=(a['x']>=x0)&(a['x']<x0+size)&(a['y']>=y0)&(a['y']<y0+size);a=a[keep]
        if len(a):pieces.append(pd.DataFrame(dict(transcript_id=a['transcript_id'],x_location=ix[a['x']-x0],y_location=iy[a['y']-y0],feature_name=np.asarray(genes)[a['gene']],qv=20)))
    pd.concat(pieces,ignore_index=True).to_csv(data/'transcripts_processed.csv',index=False)
    baseline_mask=json.loads(Path(queue['methods']['B1_CONT']['machine']).read_text())['baseline_final_mask']
    tifffile.imwrite(data/'historical_baseline_final.tif',tifffile.imread(baseline_mask)[y0:y0+size,x0:x0+size])
    mini=dict(queue);mini.update(run_id=root.name,output_root=str(root),admission=str(root/'admission.json'),mini_test=True,methods={})
    for filename in ('reference.json','protocol.json','anchors.json','probes.json'):
        shutil.copyfile(formal/'configs'/filename,root/'configs'/filename)
    for arm in ARMS:
        original=read(queue['methods'][arm]['config']);armroot=root/arm;(armroot/'configs').mkdir(parents=True);(armroot/'source').mkdir()
        values=original.model_dump(mode='json');values.update(output_root=str(armroot),run_id=root.name,baseline_config=str(baseline),queue_config=str(root/'configs/queue.json'),updates=2,early_checkpoint=2,checkpoint_every=2,mini_test=True)
        cfg=type(original).model_validate(values);atomic_json(armroot/'configs/run.json',cfg.model_dump(mode='json'))
        ds=DataProcessing(cfg.native(),isTraining=True,total_steps=None,structured=True,mask_unknown_markers=True)
        views=[]
        for i,p in enumerate(ds.fp_patches):
            y,x=map(int,re.findall(r'\d+',Path(p).name)[:2]);ids=np.unique(crop[y:y+48,x:x+48]);ids=ids[ids>0]
            views.append(dict(index=i,flip=i%2,rotation=i%4,k=len(ids),cell_ids=ids.tolist(),epoch=0,patch=p))
            if sum(v['k']>0 for v in views)==2:break
        atomic_json(armroot/'configs/views.json',dict(train_pool=ds.fp_patches,views=views,total_updates=2,mini_test=True))
        probes=[]
        for shift in (0,24):
            ds=DataProcessing(cfg.native(),isTraining=False,all_patches=True,shift_patches=shift,structured=True,mask_unknown_markers=True)
            atomic_json(armroot/f'configs/patches_shift{shift}.json',[identity(p) for p in ds.fp_patches])
            probes.extend(dict(shift=shift,index=i) for i in range(min(3,len(ds))))
        atomic_json(armroot/'configs/probes.json',dict(probes=probes))
        source_machine=json.loads(Path(queue['methods'][arm]['machine']).read_text());own=dict(source_machine)
        own.update(baseline_final_mask=str(data/'historical_baseline_final.tif'),baseline_config=str(baseline),nuclei_mask=str(data/'nuclei.tif'),preannotation=str(data/native.files.fp_nuclei_anno),genes=str(data/native.files.fp_gene_names),transcripts=str(data/'transcripts_processed.csv'),final_mask=str(armroot/'predictions'/arm/'final.tif'),pre_mask=str(armroot/'predictions'/arm/'pre.tif'),final_matrix=str(armroot/'matrix'/arm/'expr_mat.csv'))
        atomic_json(armroot/f'configs/{arm}_machine.json',own)
        inputs=json.loads((formal/'configs/inputs.json').read_text())
        for key in ('baseline_config','nuclei_mask','preannotation','genes','transcripts'):inputs[key]=identity(own[key])
        atomic_json(armroot/'configs/inputs.json',inputs)
        for filename in ('reference.json','protocol.json','anchors.json'):shutil.copyfile(formal/'configs'/filename,armroot/'configs'/filename)
        atomic_json(armroot/'configs/frozen.json',dict(identities={p.name:identity(p) for p in (armroot/'configs').glob('*.json')}))
        mini['methods'][arm]=dict(config=str(armroot/'configs/run.json'),worktree=str(Path(__file__).resolve().parents[3]),commit='mini-unfrozen',machine=str(armroot/f'configs/{arm}_machine.json'),matrix=str(armroot/'matrix'/arm),evaluation=str(armroot/'evaluation'/arm),cost_root=str(armroot))
    atomic_json(root/'configs/queue.json',mini);atomic_json(root/'fixture.json',dict(mini_test=True,origin=[y0,x0],size=size,cell_count=int((np.unique(crop)>0).sum()),transcript_count=sum(len(p) for p in pieces),source_coordinate_manifest=identity(manifests[0]),purpose='full real regional pipeline admission; not scientific whole-field evidence'))
    print(root/'configs/queue.json',flush=True)

def stages(queuepath):
    queue=json.loads(Path(queuepath).read_text())
    for arm,item in queue['methods'].items():
        for stage in ('train','predict','postprocess','matrix'):
            subprocess.run([sys.executable,'-u','-m','scripts.cos.stage','--config',item['config'],'--stage',stage],check=True)
    atomic_json(Path(queue['output_root'])/'first_four_stages.json',dict(status='succeeded',arms=list(queue['methods'])))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--queue',required=True);p.add_argument('--output');p.add_argument('--stages',action='store_true');a=p.parse_args()
    if a.stages:stages(a.queue)
    else:prepare(a.queue,a.output)
