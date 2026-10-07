"""Exact native OpenCV nearest grid, chunked integer aggregation and receipts."""
import argparse
import json
import os
import time
import fcntl
import resource
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import tifffile
from .common import identity, sha256, atomic_json, atomic_npy, Monitor, VERSION

def lookup_axes(shape, scale_x, scale_y):
    h,w=shape; oh=int(np.round(h/scale_y)); ow=int(np.round(w/scale_x))
    # Invoke the SAME OpenCV INTER_NEAREST on separable integer axes. No assumed floor formula.
    x=cv2.resize(np.arange(w,dtype=np.int32)[None,:],(ow,1),interpolation=cv2.INTER_NEAREST)[0]
    y=cv2.resize(np.arange(h,dtype=np.int32)[:,None],(1,oh),interpolation=cv2.INTER_NEAREST)[:,0]
    return x,y

def grid(chunk, shape, sx, sy):
    xa,ya=lookup_axes(shape,sx,sy)
    x=chunk.x_location.to_numpy(float); y=chunk.y_location.to_numpy(float)
    if not (np.isfinite(x).all() and np.isfinite(y).all()): raise ValueError('nonfinite coordinates')
    xi=np.rint(x/sx).astype(np.int64); yi=np.rint(y/sy).astype(np.int64)
    valid=(xi>=0)&(xi<len(xa))&(yi>=0)&(yi<len(ya))
    mx=np.full(len(x),-1,np.int32); my=mx.copy()
    mx[valid]=xa[xi[valid]]; my[valid]=ya[yi[valid]]
    return mx,my,valid

def aggregate(labels, genes, ids, ng):
    keep=labels>0
    rows=np.searchsorted(ids,labels[keep])
    if np.any(rows>=len(ids)) or not np.array_equal(ids[rows],labels[keep]): raise ValueError('unknown mask ID')
    flat=rows.astype(np.int64)*ng+genes[keep]
    keys,n=np.unique(flat,return_counts=True)
    return keys,n

def build(cfg,out,kind='final',cache=None):
    setup_start=time.monotonic()
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    lock=(out/'run.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'complete.json').exists(): raise FileExistsError('completed output protected; verify and resume via runner')
    maskpath=cfg['final_mask' if kind=='final' else 'nuclei_mask']
    deps={k:identity(cfg[k]) for k in ['transcripts','genes','baseline_config']}
    deps['mask']=identity(maskpath)
    hashes_done=time.monotonic()
    contract=dict(version=VERSION,source_sha256=sha256(__file__),shape=None,scale_x=cfg['scale_x'],scale_y=cfg['scale_y'],coordinate='processed coords / scale_pix -> pandas-compatible ties-to-even round -> native cv2 INTER_NEAREST axes; OOB unassigned',filter='already processed qv>=20 + configured prefixes + nonnegative; no refilter; vendor cell_id ignored',data_scope='own-mask Rep1 full field engineering',fit_scope='none',kind=kind)
    mask=tifffile.imread(maskpath); contract['shape']=list(mask.shape)
    if mask.ndim!=2 or not np.issubdtype(mask.dtype,np.integer) or (mask<0).any() or int(mask.max(initial=0))>np.iinfo(np.uint32).max:
        raise ValueError("mask must be nonnegative uint32-compatible 2D labels")
    mask_loaded=time.monotonic()
    ids=np.unique(mask); ids=ids[ids>0].astype(np.int64)
    genes=Path(cfg['genes']).read_text().splitlines()
    if len(genes)!=len(set(genes)): raise ValueError('duplicate gene names')
    gi=pd.Index(genes); counts=np.zeros((len(ids),len(genes)),np.int64)
    cdeps=dict(transcripts=deps['transcripts']['sha256'],genes=deps['genes']['sha256'],shape=contract['shape'],scale_x=cfg['scale_x'],scale_y=cfg['scale_y'],builder=contract['source_sha256'], common=sha256(Path(__file__).with_name('common.py')),numpy=np.__version__,opencv=cv2.__version__,pandas=pd.__version__)
    key=__import__('hashlib').sha256(json.dumps(cdeps,sort_keys=True).encode()).hexdigest()
    cdir=Path(cache)/key if cache else out/'coordinate_cache'
    cdir.mkdir(parents=True,exist_ok=True)
    cached=(cdir/'complete.json').exists()
    if cfg.get('expected_cache_hit') is not None and cached != cfg['expected_cache_hit']:
        raise ValueError('unexpected cold/warm cache state')
    if cached:
        cm=json.loads((cdir/'complete.json').read_text())
        if cm['dependencies']!=cdeps: raise ValueError('cache dependencies mismatch')
        for r in cm['chunks']:
            if sha256(cdir/r['file'])!=r['sha256']: raise ValueError('cache corruption')
    preparation_done=time.monotonic()
    extract_start=time.monotonic()
    if cached:
        iterator=(np.load(cdir/r['file']) for r in cm['chunks'])
    else:
        iterator=pd.read_csv(cfg['transcripts'],usecols=['transcript_id','x_location','y_location','feature_name','qv'],chunksize=cfg.get('chunksize',500000))
    receipts=[]; cr=[]; tids=[]; totals=dict(eligible=0,assigned=0,unassigned=0,out_of_bounds=0)
    with Monitor(out/'resources.json'):
        for i,chunk in enumerate(iterator):
            t=time.monotonic()
            if cached: arr=chunk
            else:
                if (chunk.qv<20).any(): raise ValueError('processed qv violation')
                gc=gi.get_indexer(chunk.feature_name)
                if (gc<0).any(): raise ValueError('unknown genes')
                x,y,valid=grid(chunk,mask.shape,cfg['scale_x'],cfg['scale_y'])
                arr=np.empty(len(chunk),dtype=[('transcript_id','<u8'),('x','<i4'),('y','<i4'),('gene','<i4')])
                arr['transcript_id']=chunk.transcript_id.to_numpy(np.uint64);arr['x']=x;arr['y']=y;arr['gene']=gc
                cp=cdir/f'chunk_{i:04d}.npy'; atomic_npy(cp,arr); cr.append(dict(file=cp.name,sha256=sha256(cp),rows=len(arr)))
            valid=(arr['x']>=0)&(arr['y']>=0)
            ap=out/f'owners_{i:04d}.npy'; rp=out/f'owners_{i:04d}.json'
            coordinate_file=cdir/(cm['chunks'][i]['file'] if cached else cr[-1]['file'])
            ownerdeps=dict(mask=deps['mask']['sha256'],coordinates=sha256(coordinate_file),source=contract['source_sha256'])
            if rp.exists():
                saved=json.loads(rp.read_text())
                if saved['dependencies']!=ownerdeps or sha256(ap)!=saved['sha256']: raise ValueError('owner chunk dependency/corruption')
                owner=np.load(ap)
            else:
                owner=np.zeros(len(arr),np.uint32)
                owner[valid]=mask[arr['y'][valid],arr['x'][valid]]
                atomic_npy(ap,owner)
                atomic_json(rp,dict(dependencies=ownerdeps,sha256=sha256(ap)))
            k,n=aggregate(owner,arr['gene'],ids,len(genes)); counts.ravel()[k]+=n
            totals['eligible']+=len(arr);totals['assigned']+=int((owner>0).sum());totals['unassigned']+=int((owner==0).sum());totals['out_of_bounds']+=int((~valid).sum())

            tids.append(arr['transcript_id'].copy())
            receipts.append(dict(chunk=i,row_start=totals['eligible']-len(arr),row_stop=totals['eligible'],rows=len(arr),assigned=int((owner>0).sum()),owner_file=ap.name,coordinate_file=str(coordinate_file),coordinate_sha256=sha256(coordinate_file),transcript_association='coordinate record transcript_id in identical row order',sha256=sha256(ap),seconds=time.monotonic()-t))
            atomic_json(out/'progress.json',dict(totals=totals,completed_chunks=len(receipts)))
            print(kind,i,totals,flush=True)
        allids=np.concatenate(tids) if tids else np.array([],dtype=np.uint64);allids.sort(); duplicates=int(np.count_nonzero(allids[1:]==allids[:-1]));del allids,tids
        if duplicates: raise ValueError(f'duplicate transcript IDs: {duplicates}')
        if not cached: atomic_json(cdir/'complete.json',dict(dependencies=cdeps,chunks=cr,duplicate_ids=duplicates))
        if totals['eligible']!=totals['assigned']+totals['unassigned'] or int(counts.sum())!=totals['assigned']: raise ValueError('conservation failure')
        totals['background_valid']=totals['unassigned']-totals['out_of_bounds']
        atomic_npy(out/'counts.npy',counts);pd.DataFrame(dict(matrix_row=np.arange(len(ids)),mask_label=ids,canonical_cell_id=ids)).to_csv(out/'cell_mapping.csv',index=False)
        pd.DataFrame(dict(gene_column=np.arange(len(genes)),gene=genes)).to_csv(out/'gene_mapping.csv',index=False)
        before_csv=time.monotonic()
        frame=pd.DataFrame(counts,index=ids,columns=genes)
        frame.insert(0,'cell_id',ids)
        with (out/'expr_mat.csv.tmp').open('w') as csvfile:
            frame.to_csv(csvfile); csvfile.flush(); os.fsync(csvfile.fileno())
        os.replace(out/'expr_mat.csv.tmp',out/'expr_mat.csv')
        csv_closed_ns=time.monotonic_ns()
        extract_end=csv_closed_ns/1e9
        timings=dict(T_extract_seconds=extract_end-extract_start,
                     preparation_seconds=preparation_done-setup_start,
                     input_hash_seconds=hashes_done-setup_start,
                     mask_load_seconds=mask_loaded-hashes_done,
                     mapping_and_cache_precheck_seconds=preparation_done-mask_loaded,
                     CSV_export_seconds=extract_end-before_csv,
                     csv_closed_monotonic_ns=csv_closed_ns,
                     peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                     scope='continuous wall: transcript/cache opening and reads, coordinates, cold cache fsync/hash, owner query, aggregate, owner fsync/hash, progress receipts, duplicate-ID check, cache manifest, counts/maps, complete CSV export/fsync/close; excludes historical comparison and independent audit',
                     OS_cache='uncontrolled; cold/warm refers only to private coordinate cache')
        notify_fd=os.environ.get('MATRIX_CSV_NOTIFY_FD')
        if notify_fd is not None:
            os.write(int(notify_fd),(str(csv_closed_ns)+'\n').encode()); os.close(int(notify_fd))
        atomic_json(out/'timings.json',timings)
        atomic_json(out/'validation.json',dict(totals=totals,duplicate_transcript_ids=duplicates))
        manifest=dict(status='succeeded',kind=kind,dependencies=deps,contract=contract,coordinate_cache=str(cdir),cache_hit=cached,receipts=receipts,totals=totals,timings=timings,outputs={f:identity(out/f) for f in ['counts.npy','cell_mapping.csv','gene_mapping.csv','expr_mat.csv']})
        # Generation completion is distinct from independent acceptance.
        atomic_json(out/'generated.json',manifest)
        validate_production(cfg,out,manifest)
        atomic_json(out/'complete.json',manifest)
    return manifest

def validate_production(cfg,out,manifest):
    """Independent actual 2D OpenCV resize + pandas rounding + np.add.at.
    Re-read every transcript; validate every owner, valid/OOB flag and gene/ID.
    Does not call lookup_axes, grid or aggregate, nor read historical counts.
    """
    out=Path(out);mask=tifffile.imread(manifest['dependencies']['mask']['path'])
    h,w=mask.shape
    resized=cv2.resize(mask.astype(np.int32),
        (round(w/cfg['scale_x']),round(h/cfg['scale_y'])),interpolation=cv2.INTER_NEAREST)
    # OpenCV int32 preserves uint32 bit patterns on nearest-neighbour copies.
    resized=resized.view(np.uint32)
    ids=pd.read_csv(out/'cell_mapping.csv').mask_label.to_numpy(np.int64)
    genes=Path(cfg['genes']).read_text().splitlines();gi=pd.Index(genes)
    independent=np.zeros((len(ids),len(genes)),np.int64);rows=0;oob=0
    iterator=pd.read_csv(cfg['transcripts'],usecols=['transcript_id','x_location','y_location','feature_name'],chunksize=cfg.get('chunksize',500000))
    from itertools import zip_longest
    for rec,chunk in zip_longest(manifest['receipts'],iterator):
        if rec is None or chunk is None: raise ValueError('extra/missing transcript chunk')
        cached=np.load(rec['coordinate_file']);owner=np.load(out/rec['owner_file'])
        xi=(chunk.x_location/cfg['scale_x']).round().to_numpy(np.int64)
        yi=(chunk.y_location/cfg['scale_y']).round().to_numpy(np.int64)
        valid=(xi>=0)&(yi>=0)&(xi<resized.shape[1])&(yi<resized.shape[0])
        expected=np.zeros(len(chunk),np.uint32);expected[valid]=resized[yi[valid],xi[valid]]
        gc=gi.get_indexer(chunk.feature_name)
        if not np.array_equal(expected,owner): raise ValueError('independent native owner mismatch')
        if not np.array_equal(valid,(cached['x']>=0)&(cached['y']>=0)): raise ValueError('OOB classification mismatch')
        if not np.array_equal(cached['gene'],gc) or not np.array_equal(cached['transcript_id'],chunk.transcript_id.to_numpy(np.uint64)): raise ValueError('transcript ID/gene association mismatch')
        if rec['row_start']!=rows or rec['row_stop']!=rows+len(chunk) or rec['rows']!=len(chunk): raise ValueError('chunk row coverage mismatch')
        keep=expected>0;np.add.at(independent,(np.searchsorted(ids,expected[keep]),gc[keep]),1)
        rows+=len(chunk);oob+=int((~valid).sum())
    if rows!=manifest['totals']['eligible']: raise ValueError('incomplete transcript validation')
    counts=np.load(out/'counts.npy')
    if not np.array_equal(independent,counts): raise ValueError('independent integer matrix mismatch')
    frame=pd.read_csv(out/'expr_mat.csv',index_col=0)
    if list(frame.columns)!=['cell_id']+genes or not np.array_equal(frame.index.to_numpy(),ids) or not np.array_equal(frame.cell_id.to_numpy(),ids) or not np.array_equal(frame[genes].to_numpy(),counts):
        raise ValueError('CSV/NPY mapping or integer mismatch')
    receipt=dict(status='succeeded',validation_kind='independent_full_native_resize_lookup_and_conservation',
        rows=rows,owner_differences=0,integer_count_differences=0,csv_numpy_equal=True,out_of_bounds=oob,
        conservation=manifest['totals'],mask=manifest['dependencies']['mask'],
        outputs=manifest['outputs'],source=identity(__file__),numpy=np.__version__,opencv=cv2.__version__)
    for name,obj in manifest['dependencies'].items():
        if sha256(obj['path'])!=obj['sha256']: raise ValueError('input changed during extraction '+name)
    atomic_json(out/'validation.json',receipt)
    manifest['validation']=identity(out/'validation.json')
    manifest['validation_kind']=receipt['validation_kind']

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--kind',choices=['final','nuclear'],default='final');p.add_argument('--cache',required=True);a=p.parse_args()
    build(json.loads(Path(a.config).read_text()),a.out,a.kind,a.cache)
