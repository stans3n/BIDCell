"""Predefined 32 stratified ROI selections and deterministic evidence panels."""
from .continuation import frozen_root
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
from .common import (atomic_json,atomic_npz,check_receipt,dependencies,identity,inventory,
                     read,within,write_receipt,progress,chunk_receipt)
from .contracts import comparisons,comparison_id

def freeze_raw_transcripts(manifest,selection,out,deps,callback=None):
    """Original floating point coordinates and IDs; all genes, no point sampling."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return check_receipt(out/'complete.json',deps)
    chosen=[r for r in selection['rows'] if r['selected']]
    machine=read(manifest['endpoints']['BASE_MATCH']['machine']);genes=Path(machine['genes']).read_text().splitlines()
    dtype=np.dtype([('transcript_id','<u8'),('x','<f8'),('y','<f8'),('native_x','<i4'),('native_y','<i4'),('gene','<i4')])
    collected={r['id']:[] for r in chosen}
    rows=chunks=0
    for i,chunk in enumerate(pd.read_csv(machine['transcripts'],usecols=['transcript_id','x_location','y_location','feature_name'],chunksize=500000)):
        if i>=len(manifest['coordinates']):raise ValueError('raw ROI CSV has unexpected extra chunks')
        if not np.isfinite(chunk[['x_location','y_location']].to_numpy()).all():raise ValueError('raw ROI coordinate is nonfinite')
        coord=np.load(manifest['coordinates'][i]['path'],mmap_mode='r')
        if not np.array_equal(coord['transcript_id'],chunk.transcript_id.to_numpy(np.uint64)) or not np.array_equal(coord['gene'],pd.Index(genes).get_indexer(chunk.feature_name)):
            raise ValueError('raw ROI transcript ID/gene association differs from coordinate cache')
        for r in chosen:
            x0,y0,x1,y1=r['bounds_xyxy'];sel=(chunk.x_location>=x0)&(chunk.x_location<x1)&(chunk.y_location>=y0)&(chunk.y_location<y1)
            arr=np.empty(int(sel.sum()),dtype);arr['transcript_id']=coord['transcript_id'][sel];arr['x']=chunk.x_location[sel];arr['y']=chunk.y_location[sel]
            arr['native_x']=coord['x'][sel];arr['native_y']=coord['y'][sel];arr['gene']=coord['gene'][sel];collected[r['id']].append(arr)
        rows+=len(chunk);chunks+=1
    if chunks!=len(manifest['coordinates']) or rows!=manifest['expected_molecules']:raise ValueError('raw ROI full transcript coverage differs')
    for r in chosen:atomic_npz(out/(r['id']+'.npz'),transcripts=np.concatenate(collected[r['id']]))
    atomic_json(out/'semantics.json',dict(original_float_coordinates=True,all313_genes=True,sampling=False,rows=rows,chunks=chunks,selection_bounds='original native-micrometer x_location,y_location',source=identity(machine['transcripts'])))
    return write_receipt(out/'complete.json',deps,inventory(out),selected=len(chosen),original_coordinates=True)

def select_rois(regions,competition_type,interface_type,density_group,seed=20261017,size=128):
    rows=[];shape=next(iter(regions.values())).shape
    for ri,region in enumerate(('competition','interface','nuclear','noncompetitive')):
        classes=range(3) if ri<2 else [-1]
        types=interface_type if region=='interface' else competition_type
        for ti in classes:
            for di in range(4):
                use=regions[region]&(density_group==di)
                if ti>=0:use&=types==ti
                type_index=ti if ti>=0 else 3 # 0/1/2 known classes; 3 is the control-only sentinel.
                candidates=np.flatnonzero(use);rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed,ri,type_index,di])))
                item=dict(id=f'ROI_{len(rows)+1:02d}',region=region,type_class=['same','heterotypic','Unknown'][ti] if ti>=0 else 'all',
                          density_quartile=di,n_candidates=len(candidates),selected=int(len(candidates)>0),type_index=type_index,seed_sequence=[seed,ri,type_index,di],
                          reason='' if len(candidates) else 'empty_prespecified_stratum')
                if len(candidates):
                    flat=int(candidates[int(rng.integers(len(candidates)))]);y,x=divmod(flat,shape[1])
                    item.update(center_xy=[x,y],bounds_xyxy=[max(0,x-size//2),max(0,y-size//2),min(shape[1],x+size//2),min(shape[0],y+size//2)])
                rows.append(item)
    if len(rows)!=32:raise AssertionError('ROI selection contract requires 32 strata')
    return dict(rows=rows,summary=dict(target=32,selected=sum(r['selected'] for r in rows),empty=sum(not r['selected'] for r in rows),
                size_um=size,seed=seed,selection='uniform sorted flat pixel candidates within each prespecified layer',
                overlap='centers may coincide across strata; retain each selection identity'))

def colorize(labels):
    ids=np.asarray(labels,dtype=np.uint64)
    # A fixed arithmetic ID hash gives every panel exactly the same cell color.
    h=(ids*2654435761)&0xffffffff
    rgb=np.stack(((h>>16)&255,(h>>8)&255,h&255),axis=-1).astype(float)/255
    rgb[ids==0]=0
    return rgb

def boundary(mask):
    out=np.zeros(mask.shape,bool)
    out[1:]|=(mask[1:]!=mask[:-1]);out[:-1]|=(mask[1:]!=mask[:-1])
    out[:,1:]|=(mask[:,1:]!=mask[:,:-1]);out[:,:-1]|=(mask[:,1:]!=mask[:,:-1])
    return out&(mask>0)

def render(cfg,manifest,specs,out,callback=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    deps=dependencies(cfg,'ROI_'+str(len(specs)))
    if (out/'complete.json').exists():return check_receipt(out/'complete.json',deps,cfg['run_root'])
    frozen=frozen_root(cfg);selection=read(frozen/'roi_selection.json');info=read(frozen/'regions.json')
    chosen=[r for r in selection['rows'] if r['selected']];masks={}
    # Retain only small crops; twelve complete mask rasters are never co-resident.
    for name,spec in specs.items():
        full=tifffile.imread(spec['final']);masks[name]={r['id']:full[r['bounds_xyxy'][1]:r['bounds_xyxy'][3],r['bounds_xyxy'][0]:r['bounds_xyxy'][2]].copy() for r in chosen}
        del full
    nuc=tifffile.imread(manifest['endpoints']['BASE_MATCH']['nuclei']);dapi=tifffile.imread(manifest['dapi'])
    check_receipt(frozen/'roi_transcripts/complete.json')
    transcripts={r['id']:np.load(frozen/'roi_transcripts'/(r['id']+'.npz'))['transcripts'] for r in chosen}
    lo,hi=info['dapi_log1p_limits'];allpairs=comparisons(tuple(specs));products=[]
    for i,r in enumerate(chosen):
        root=out/r['id'];root.mkdir(exist_ok=True);dep=dict(deps,selection=r,methods=list(specs))
        if (root/'complete.json').exists():check_receipt(root/'complete.json',dep);continue
        x0,y0,x1,y1=r['bounds_xyxy'];nc=nuc[y0:y1,x0:x1];dc=np.log1p(dapi[y0:y1,x0:x1].astype(float))
        pts=transcripts[r['id']];atomic_npz(root/'data.npz',nuclei=nc,dapi=dapi[y0:y1,x0:x1],transcripts=pts,**{m:masks[m][r['id']] for m in specs})
        pdf=root/'panels.pdf'
        with PdfPages(pdf) as pages:
            cols=4;rows=int(np.ceil((3+len(specs))/cols));fig,axs=plt.subplots(rows,cols,figsize=(16,4*rows));axes=list(axs.ravel())
            axes[0].imshow(dc,cmap='gray',vmin=lo,vmax=hi);axes[0].set_title('DAPI (shared frozen log1p limits)')
            axes[1].imshow(colorize(nc));axes[1].set_title('Complete nuclei')
            axes[2].imshow(dc,cmap='gray',vmin=lo,vmax=hi)
            axes[2].scatter(pts['x']-x0,pts['y']-y0,c=pts['gene'],cmap='turbo',vmin=0,vmax=312,s=.25,alpha=.35,linewidths=0,rasterized=True)
            axes[2].set_title(f'All313 original molecules n={len(pts)}')
            for ax,name in zip(axes[3:],specs):
                ax.imshow(colorize(masks[name][r['id']]));overlay=np.zeros((*nc.shape,4));overlay[boundary(nc)]=[1,1,1,1];ax.imshow(overlay);ax.set_title(name)
            for ax in axes:ax.set_axis_off()
            fig.suptitle(r['id']+' '+r['region']+' / '+r['type_class']+' / density '+str(r['density_quartile'])+'; DAPI/markers are not owner ground truth')
            fig.tight_layout();fig.savefig(root/'masks.png',dpi=130);pages.savefig(fig);plt.close(fig)
            for start in range(0,len(allpairs),8):
                part=allpairs[start:start+8];fig,axs=plt.subplots(2,4,figsize=(16,8))
                for ax,(a,b) in zip(axs.ravel(),part):
                    ca,cb=masks[a][r['id']],masks[b][r['id']];view=np.zeros((*ca.shape,3));change=ca!=cb
                    view[change&(ca>0)&(cb>0)]=[.9,.6,.1];view[(cb>0)&(ca==0)]=[.9,.2,.3];view[(cb==0)&(ca>0)]=[.1,.7,.6]
                    ax.imshow(view);ax.set_title(a+' − '+b,fontsize=8);ax.set_axis_off()
                for ax in axs.ravel()[len(part):]:ax.set_axis_off()
                fig.suptitle('Gold: cell→cell; red: cell→background; teal: background→cell')
                fig.tight_layout();fig.savefig(root/f'changes_{start:02d}.png',dpi=130);pages.savefig(fig);plt.close(fig)
        atomic_json(root/'manifest.json',dict(selection=r,methods=list(specs),comparisons=[comparison_id(a,b) for a,b in allpairs],
                    molecule_points=len(pts),point_display='all eligible raw transcript IDs at original floating coordinates; gene color; alpha .35; no sampling',
                    shared_ID_color=True,dapi_limits=[lo,hi],ground_truth=False))
        write_receipt(root/'complete.json',dep,inventory(root),selected=True);products.append(identity(root/'complete.json'))
        progress(callback,'fixed_ROI_render',i+1,len(chosen),receipt=str(root/'complete.json'))
    atomic_json(out/'coverage.json',dict(selection=selection,methods=list(specs),target=32,rendered=len(chosen),empty_reasons=[r for r in selection['rows'] if not r['selected']]))
    return write_receipt(out/'complete.json',deps,inventory(out),target_strata=32,rendered=len(chosen),methods=list(specs))
