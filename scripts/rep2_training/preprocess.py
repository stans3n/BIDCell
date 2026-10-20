"""Bounded-memory native preparation; exact raster/rounding and complete shift-0 pool."""
import csv
import gc
import os
from pathlib import Path
import re
import h5py
import natsort
import numpy as np
import pandas as pd
import tifffile
from .common import *

def coords(path): return tuple(map(int, re.findall(r'\d+', Path(path).stem)))

def filtered_chunks(cfg):
    # The same pandas parser, filter, scale and nonnegative-coordinate rules as
    # generate_expression_maps. Only the iteration granularity changes.
    for frame in pd.read_csv(cfg.files.fp_transcripts, chunksize=250000):
        keep = ~frame[cfg.transcripts.gene_col].str.startswith(tuple(cfg.transcripts.transcripts_to_filter))
        if 'qv' in frame: keep &= frame.qv >= cfg.transcripts.min_qv
        if cfg.files.fp_selected_genes is not None:
            keep &= frame[cfg.transcripts.gene_col].isin(Path(cfg.files.fp_selected_genes).read_text().splitlines())
        frame = frame.loc[keep].copy()
        x,y = cfg.transcripts.x_col, cfg.transcripts.y_col
        frame[x] *= cfg.affine.scale_ts_x; frame[y] *= cfg.affine.scale_ts_y
        if cfg.transcripts.shift_to_origin: raise ValueError('shift-to-origin is outside frozen Rep2 configuration')
        frame = frame.loc[(frame[x]>=0)&(frame[y]>=0)]
        yield frame

def nuclei():
    cfg = configure(); configure_seed()
    from bidcell.processing.nuclei_segmentation import segment_nuclei
    segment_nuclei(cfg)
    data = Path(cfg.files.data_dir)
    mask = tifffile.imread(data/cfg.files.fp_nuclei)
    dapi = tifffile.imread(data/cfg.files.fp_rdapi)
    if list(mask.shape)!=read(ROOT/'manifests/resource_admission.json')['native_resized_shape']:raise ValueError('native resized shape differs from metadata-derived admission')
    ids = np.unique(mask)
    if mask.shape != dapi.shape or mask.dtype != np.uint32 or len(ids)<2 or ids[0]!=0: raise ValueError('invalid nuclei/DAPI output')
    if not np.array_equal(ids, np.arange(int(ids[-1])+1)): raise ValueError('noncontiguous native nuclear ID system')
    return dict(nuclei=data/cfg.files.fp_nuclei, dapi=data/cfg.files.fp_rdapi)

def expression():
    cfg = configure(); data = Path(cfg.files.data_dir); dest = data/cfg.files.dir_out_maps; dest.mkdir(exist_ok=True)
    mask = tifffile.memmap(data/cfg.files.fp_nuclei); H,W = mask.shape
    from bidcell.processing.utils import get_patches_coords
    ys,h = get_patches_coords(H,cfg.transcripts.max_height); xs,w = get_patches_coords(W,cfg.transcripts.max_width)
    processed = data/cfg.files.fp_transcripts_processed
    summary_path = data/'transcript_summary.json'; summary_receipt = data/'transcript_summary.receipt.json'
    deps = base_dependencies()
    if summary_receipt.exists(): validate_receipt(summary_receipt, deps)
    else:
        count=0; names=set(); lo=np.array([np.inf,np.inf]); hi=-lo; header=True
        tmp=processed.with_suffix('.tmp.csv')
        with tmp.open('w') as stream:
            for frame in filtered_chunks(cfg):
                if not len(frame): continue
                x,y=cfg.transcripts.x_col,cfg.transcripts.y_col
                xy=frame[[x,y]].to_numpy();lo=np.minimum(lo,xy.min(0));hi=np.maximum(hi,xy.max(0))
                names.update(frame[cfg.transcripts.gene_col].unique())
                frame.index=np.arange(count,count+len(frame));frame.to_csv(stream,header=header);header=False;count+=len(frame)
                progress('expression', part='filter_transcripts', rows=count)
            stream.flush();os.fsync(stream.fileno())
        if count==0: raise ValueError('no filtered Rep2 transcripts')
        if count!=read(ROOT/'manifests/raw_input_audit.json')['eligible_transcripts']:raise ValueError('preflight filtered transcript count differs')
        genes=natsort.natsorted(names)
        expected=Path(read(ROOT/'configs/run.json')['gene_panel']).read_text().splitlines()
        if genes!=expected or len(genes)!=313: raise ValueError('Rep2 measured gene set/order differs from frozen panel')
        if np.round(hi[0])>=W or np.round(hi[1])>=H: raise ValueError('transcripts exceed native nuclear raster')
        os.replace(tmp,processed)
        (data/cfg.files.fp_gene_names).write_text(''.join(g+'\n' for g in genes))
        keys=['scale_ts_x','scale_ts_y','min_x','min_y','size_x','size_y','global_shift_x','global_shift_y','origin']
        vals=[cfg.affine.scale_ts_x,cfg.affine.scale_ts_y,*lo,*(hi+1),cfg.affine.global_shift_x,cfg.affine.global_shift_y,False]
        with (data/cfg.files.fp_affine).open('w') as f: csv.writer(f,delimiter='\t').writerows(zip(keys,vals))
        atomic_json(summary_path,dict(rows=count,genes=genes,min_xy=lo.tolist(),max_xy=hi.tolist(),shape=[H,W]))
        atomic_json(summary_receipt,dict(status='succeeded',dependencies=deps,outputs={k:identity(p) for k,p in dict(processed=processed,genes=data/cfg.files.fp_gene_names,affine=data/cfg.files.fp_affine,summary=summary_path).items()}))
    summary=read(summary_path);genes=summary['genes'];outputs=dict(processed=processed,genes=data/cfg.files.fp_gene_names,affine=data/cfg.files.fp_affine,summary=summary_path)
    total=0
    for ys0,ye in ys:
        for xs0,xe in xs:
            out=dest/f'all_genes_{ys0}_{ye}_{xs0}_{xe}.hdf5'; rp=out.with_suffix('.receipt.json')
            if rp.exists():
                rec=validate_receipt(rp,deps);total+=rec['counts'];outputs[out.name]=out;continue
            resource_check('expression', minimum_extra_bytes=3500*4000*313)
            # uint16 staging only detects overflow. Frozen scientific maps stay
            # uint8 exactly as native; no precision reduction is introduced.
            values=np.zeros((h,w,313),np.uint16);seen=0
            for frame in filtered_chunks(cfg):
                xx=frame[cfg.transcripts.x_col].round().to_numpy(np.int64); yy=frame[cfg.transcripts.y_col].round().to_numpy(np.int64)
                inside=(xx>=xs0)&(xx<xe)&(yy>=ys0)&(yy<ye)
                gi=pd.Categorical(frame.loc[inside,cfg.transcripts.gene_col],categories=genes).codes
                if (gi<0).any():raise ValueError('unknown gene')
                np.add.at(values,(yy[inside]-ys0,xx[inside]-xs0,gi),1);seen+=int(inside.sum())
            if int(values.max())>255:raise ValueError('native uint8 expression overflow; do not silently wrap scientific counts')
            temp=out.with_suffix('.tmp.hdf5')
            with h5py.File(temp,'w') as f:
                ds=f.create_dataset('data',shape=values.shape,dtype=np.uint8)
                for row in range(0,h,64):ds[row:row+64]=values[row:row+64]
            if int(values.sum(dtype=np.int64))!=seen:raise ValueError('expression conservation failed')
            os.replace(temp,out);del values;gc.collect();total+=seen
            atomic_json(rp,dict(status='succeeded',dependencies=deps,counts=seen,outputs={'map':identity(out)}))
            progress('expression',part='map',completed=out.name,counts_so_far=total);outputs[out.name]=out
    if total!=summary['rows']:raise ValueError('global raw-map counts differ from filtered input')
    return outputs

def patch_starts(limit, size=48):
    return list(np.arange(0,limit-size,size))+[limit-size]

def patches():
    cfg=configure();data=Path(cfg.files.data_dir);maps=data/cfg.files.dir_out_maps
    dest=maps/f'{cfg.files.dir_patches}48x48_shift_0';dest.mkdir(exist_ok=True)
    inventory=[];dependencies=base_dependencies()
    for path in natsort.natsorted(maps.glob('all_genes_*.hdf5')):
        y0,y1,x0,x1=coords(path)
        with h5py.File(path,'r') as f:
            values=f['data']
            for y in patch_starts(y1-y0):
                for x in patch_starts(x1-x0):
                    out=dest/f'{y+y0}_{x+x0}.hdf5'
                    # Rerun this incomplete stage safely; only this stage's own
                    # incomplete files are overwritten. Valid content is reused.
                    expected=values[y:y+48,x:x+48]
                    valid=False
                    if out.exists():
                        try:
                            with h5py.File(out) as old: valid=old['data'].dtype==np.uint8 and np.array_equal(old['data'][:],expected)
                        except OSError: pass
                    if not valid:
                        temp=out.with_suffix('.tmp')
                        with h5py.File(temp,'w') as target:target.create_dataset('data',data=expected,dtype=np.uint8)
                        os.replace(temp,out)
                    inventory.append(identity(out))
        progress('patches',map=path.name,patches=len(inventory))
    inventory=natsort.natsorted(inventory,key=lambda x:x['path'])
    if len(inventory)!=read(ROOT/'manifests/resource_admission.json')['shift0_patch_files']:raise ValueError('complete shift0 patch count differs from native geometry')
    if {p['path'] for p in inventory}!={str(p.resolve()) for p in dest.glob('*.hdf5')}:raise ValueError('unexpected patch files')
    output=ROOT/'shared/configs/patch_inventory.json';atomic_json(output,inventory)
    return dict(inventory=output)

def nuclear_matrix():
    """Exactly native cv2 nearest-neighbour source-grid ownership and rounding."""
    import cv2
    cfg=configure(); data=Path(cfg.files.data_dir); nuclei=tifffile.imread(data/cfg.files.fp_nuclei)
    ids=np.unique(nuclei);ids=ids[ids>0];genes=(data/cfg.files.fp_gene_names).read_text().splitlines()
    H,W=nuclei.shape; hp=int(np.round(H/cfg.affine.scale_pix_y)); wp=int(np.round(W/cfg.affine.scale_pix_x))
    enlarged=cv2.resize(nuclei.astype(np.int32),(wp,hp),interpolation=cv2.INTER_NEAREST)
    id_to_row=np.full(int(ids.max())+1,-1,np.int64);id_to_row[ids]=np.arange(len(ids));counts=np.zeros((len(ids),len(genes)),np.int64)
    visited=assigned=excluded=0
    for frame in pd.read_csv(data/cfg.files.fp_transcripts_processed,chunksize=250000):
        xx=frame[cfg.transcripts.x_col].div(cfg.affine.scale_pix_x).round().to_numpy(np.int64)
        yy=frame[cfg.transcripts.y_col].div(cfg.affine.scale_pix_y).round().to_numpy(np.int64)
        valid=(xx>=0)&(xx<wp)&(yy>=0)&(yy<hp);excluded+=int((~valid).sum())
        owner=enlarged[yy[valid],xx[valid]];gi=pd.Categorical(frame.loc[valid,cfg.transcripts.gene_col],categories=genes).codes
        if (gi<0).any():raise ValueError('nuclear matrix unknown gene')
        nuclear=owner>0;np.add.at(counts,(id_to_row[owner[nuclear]],gi[nuclear]),1)
        visited+=len(frame);assigned+=int(nuclear.sum());progress('nuclear_matrix',rows=visited,assigned=assigned)
    if counts.sum()!=assigned:raise ValueError('nuclear matrix counts mismatch')
    out=data/cfg.files.dir_cgm/'nuclei'/cfg.files.fp_expr;out.parent.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame(counts,index=ids,columns=genes);frame.insert(0,'cell_id',ids)
    tmp=out.with_suffix('.tmp');frame.to_csv(tmp);os.replace(tmp,out)
    audit=data/'nuclear_matrix_audit.json';atomic_json(audit,dict(rows=visited,assigned=assigned,excluded_native_source_grid=excluded,nuclei=len(ids),source_shape=[hp,wp],coordinate='divide by scale_pix then pandas round; cv2.INTER_NEAREST resized native mask'))
    return dict(matrix=out,audit=audit)

def annotation():
    cfg=configure()
    from bidcell.processing.preannotate import preannotate
    preannotate(cfg)
    path=Path(cfg.files.data_dir)/cfg.files.fp_nuclei_anno
    with h5py.File(path) as f:
        ids=f['ids'][:];types=f['data'][:]
    nuclei=tifffile.imread(Path(cfg.files.data_dir)/cfg.files.fp_nuclei)
    expected=np.unique(nuclei);expected=expected[expected>0]
    if not np.array_equal(ids,expected) or len(ids)!=len(types):raise ValueError('preannotation ID mismatch')
    return dict(annotation=path)
