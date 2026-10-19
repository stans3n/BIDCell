"""Own-matrix generation and a resumable full independent native 2D oracle.

The independent oracle is the algorithm in crgf.matrix.validate_production:
actual two-dimensional OpenCV nearest resize, pandas rounding, all original
transcripts, and independent np.add.at aggregation. Generation uses cached
native coordinates and crgf.matrix.aggregate instead. No parent counts are
copied into the new matrix.
"""
from __future__ import annotations
from itertools import zip_longest
from pathlib import Path
import time
import cv2
import numpy as np
import pandas as pd
import tifffile
from scripts.crgf.matrix import aggregate
from . import PARENTS, ASSIGNED, ELIGIBLE, N_CELLS, N_GENES
from .common import (atomic_csv, atomic_json, atomic_npy, atomic_npz, check_identity, check_receipt,
                     chunk_receipt, dependencies, identity, inventory, progress, read,
                     resume_chunk, within, write_receipt)
from .inputs import matrix_arrays


def build(machine, coordinates, out, deps, callback=None, expected_molecules=ELIGIBLE):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():
        return check_receipt(out/'complete.json',deps)
    start=time.monotonic()
    mask=tifffile.imread(machine['final_mask'])
    if mask.ndim!=2 or not np.issubdtype(mask.dtype,np.integer) or np.any(mask<0):
        raise ValueError('matrix mask is not a nonnegative 2D integer label image')
    ids=np.unique(mask);ids=ids[ids>0].astype(np.int64)
    genes=Path(machine['genes']).read_text().splitlines()
    if len(genes)!=len(set(genes)):raise ValueError('duplicate genes')
    counts=np.zeros((len(ids),len(genes)),np.int64)
    totals=dict(eligible=0,assigned=0,unassigned=0,out_of_bounds=0)
    receipts=[];tids=[]
    mask_id=identity(machine['final_mask'])
    for i,rec in enumerate(coordinates):
        check_identity(rec)
        arr=np.load(rec['path'],mmap_mode='r')
        if rec['row_start']!=totals['eligible'] or rec['rows']!=len(arr) or rec['row_stop']!=totals['eligible']+len(arr):
            raise ValueError('matrix coordinate coverage gap/overlap')
        if arr.dtype.names!=('transcript_id','x','y','gene') or np.any(arr['gene']<0) or np.any(arr['gene']>=len(genes)):
            raise ValueError('coordinate schema/gene mismatch')
        valid=(arr['x']>=0)&(arr['y']>=0)
        if np.any((arr['x']>=0)!=(arr['y']>=0)) or np.any(arr['x'][valid]>=mask.shape[1]) or np.any(arr['y'][valid]>=mask.shape[0]):
            raise ValueError('cached native coordinate bounds mismatch')
        ap=out/f'owners_{i:04d}.npy';part=out/f'aggregate_{i:04d}.npz';rp=out/f'owners_{i:04d}.json'
        cd=dict(deps,mask=mask_id,coordinate=rec,chunk=i)
        saved=resume_chunk(rp,cd)
        if saved is None:
            owner=np.zeros(len(arr),np.uint32)
            owner[valid]=mask[arr['y'][valid],arr['x'][valid]]
            keys,values=aggregate(owner,arr['gene'],ids,len(genes))
            atomic_npy(ap,owner);atomic_npz(part,keys=keys,counts=values)
            c=dict(eligible=len(arr),assigned=int((owner>0).sum()),unassigned=int((owner==0).sum()),out_of_bounds=int((~valid).sum()))
            saved=chunk_receipt(rp,cd,[ap,part],totals=c)
        z=np.load(part);counts.ravel()[z['keys']]+=z['counts']
        for k,v in saved['totals'].items():totals[k]+=v
        tids.append(arr['transcript_id'].copy())
        receipts.append(dict(chunk=i,row_start=rec['row_start'],row_stop=rec['row_stop'],rows=rec['rows'],
                             assigned=saved['totals']['assigned'],owner_file=ap.name,coordinate_file=rec['path'],
                             coordinate_sha256=rec['sha256'],sha256=identity(ap)['sha256']))
        progress(callback,'matrix_own_owner_and_counts',i+1,len(coordinates),molecules=totals['eligible'])
    tid=np.concatenate(tids) if tids else np.empty(0,np.uint64);tid.sort()
    duplicates=int(np.count_nonzero(tid[1:]==tid[:-1]));del tid,tids
    if duplicates:raise ValueError('duplicate transcript IDs')
    if totals['eligible']!=expected_molecules or totals['eligible']!=totals['assigned']+totals['unassigned'] or int(counts.sum())!=totals['assigned']:
        raise ValueError('matrix eligible denominator/conservation failure')
    totals['background_valid']=totals['unassigned']-totals['out_of_bounds']
    atomic_npy(out/'counts.npy',counts)
    atomic_csv(out/'cell_mapping.csv',pd.DataFrame(dict(matrix_row=np.arange(len(ids)),mask_label=ids,canonical_cell_id=ids)))
    atomic_csv(out/'gene_mapping.csv',pd.DataFrame(dict(gene_column=np.arange(len(genes)),gene=genes)))
    frame=pd.DataFrame(counts,index=ids,columns=genes);frame.insert(0,'cell_id',ids)
    atomic_csv(out/'expr_mat.csv',frame,index=True)
    outputs={p:identity(out/p) for p in ('counts.npy','cell_mapping.csv','gene_mapping.csv','expr_mat.csv')}
    mdeps={k:identity(machine[k]) for k in ('transcripts','genes','baseline_config')};mdeps['mask']=mask_id
    manifest=dict(status='succeeded',dependencies=deps,matrix_dependencies=mdeps,totals=totals,receipts=receipts,
                  outputs=outputs,coordinate_contract='processed/.2125 -> ties-to-even -> native OpenCV INTER_NEAREST inverse lookup',
                  generation_wall_seconds=time.monotonic()-start,duplicate_transcript_ids=0)
    atomic_json(out/'generated.json',manifest)
    oracle=validate_full_oracle(machine,out,manifest,callback)
    manifest['validation']=identity(out/'validation.json')
    manifest['validation_kind']='independent_full_native_resize_lookup_and_conservation'
    manifest['outputs'].update(validation=manifest['validation'])
    for rec in receipts:manifest['outputs'][rec['owner_file']]=identity(out/rec['owner_file'])
    # Also bind chunk recovery state, so corrupted completed chunks never become a silent retry.
    for p in sorted(out.glob('owners_*.json')):manifest['outputs'][p.name]=identity(p)
    for p in sorted(out.glob('aggregate_*.npz')):manifest['outputs'][p.name]=identity(p)
    for p in sorted((out/'oracle_chunks').glob('*')):
        if p.is_file():manifest['outputs'][str(p.relative_to(out))]=identity(p)
    atomic_json(out/'complete.json',manifest)
    return manifest


def validate_full_oracle(machine,out,manifest,callback=None):
    """Every molecule, every gene/ID, full 2D resized image; no sampling."""
    out=Path(out);mask=tifffile.imread(machine['final_mask']);h,w=mask.shape
    resized=cv2.resize(mask.astype(np.int32),(round(w/machine['scale_x']),round(h/machine['scale_y'])),interpolation=cv2.INTER_NEAREST).view(np.uint32)
    ids=pd.read_csv(out/'cell_mapping.csv').mask_label.to_numpy(np.int64)
    genes=Path(machine['genes']).read_text().splitlines();gi=pd.Index(genes)
    independent=np.zeros((len(ids),len(genes)),np.int64)
    rows=0;oob=0;folder=out/'oracle_chunks';folder.mkdir(exist_ok=True)
    chunksize=machine.get('chunksize',500000)
    iterator=pd.read_csv(machine['transcripts'],usecols=['transcript_id','x_location','y_location','feature_name','qv'],chunksize=chunksize)
    for i,(rec,chunk) in enumerate(zip_longest(manifest['receipts'],iterator)):
        if rec is None or chunk is None:raise ValueError('extra/missing oracle transcript chunk')
        path=folder/f'{i:04d}.npz';rp=path.with_suffix('.json')
        cd=dict(manifest['dependencies'],oracle='full_2D_native_resize_pandas_round_np_add_at',chunk=i,
                coordinate=rec['coordinate_sha256'],owner=rec['sha256'],inputs=manifest['matrix_dependencies'])
        saved=resume_chunk(rp,cd)
        if saved is None:
            cached=np.load(rec['coordinate_file'],mmap_mode='r');owner=np.load(out/rec['owner_file'])
            if not np.isfinite(chunk[['x_location','y_location']]).all().all() or (chunk.qv<20).any():
                raise ValueError('processed coordinate/qv contract violation')
            xi=(chunk.x_location/machine['scale_x']).round().to_numpy(np.int64)
            yi=(chunk.y_location/machine['scale_y']).round().to_numpy(np.int64)
            valid=(xi>=0)&(yi>=0)&(xi<resized.shape[1])&(yi<resized.shape[0])
            expected=np.zeros(len(chunk),np.uint32);expected[valid]=resized[yi[valid],xi[valid]]
            gc=gi.get_indexer(chunk.feature_name)
            if (gc<0).any():raise ValueError('independent oracle unknown gene')
            if not np.array_equal(expected,owner):raise ValueError('independent full native owner mismatch')
            if not np.array_equal(valid,(cached['x']>=0)&(cached['y']>=0)):raise ValueError('independent OOB classification mismatch')
            if not np.array_equal(cached['gene'],gc) or not np.array_equal(cached['transcript_id'],chunk.transcript_id.to_numpy(np.uint64)):
                raise ValueError('independent transcript ID/gene association mismatch')
            keep=expected>0;rr=np.searchsorted(ids,expected[keep])
            if np.any(rr>=len(ids)) or not np.array_equal(ids[rr],expected[keep]):raise ValueError('independent owner ID unknown')
            # Sparse unique keys only compress the np.add.at result; aggregation is independent.
            keys=rr.astype(np.int64)*len(genes)+gc[keep]
            unique,inverse=np.unique(keys,return_inverse=True)
            inc=np.zeros(len(unique),np.int64);np.add.at(inc,inverse,1)
            atomic_npz(path,keys=unique,counts=inc)
            saved=chunk_receipt(rp,cd,[path],rows=len(chunk),out_of_bounds=int((~valid).sum()))
        if rec['row_start']!=rows or rec['row_stop']!=rows+len(chunk) or rec['rows']!=len(chunk) or saved['rows']!=len(chunk):
            raise ValueError('independent oracle row coverage mismatch')
        z=np.load(path);independent.ravel()[z['keys']]+=z['counts']
        rows+=len(chunk);oob+=saved['out_of_bounds']
        progress(callback,'matrix_independent_full_oracle',i+1,len(manifest['receipts']),molecules=rows)
    counts=np.load(out/'counts.npy',mmap_mode='r')
    if rows!=manifest['totals']['eligible'] or not np.array_equal(independent,counts):
        raise ValueError('independent complete integer matrix mismatch')
    frame=pd.read_csv(out/'expr_mat.csv',index_col=0)
    if list(frame.columns)!=['cell_id']+genes or not np.array_equal(frame.index,ids) or not np.array_equal(frame.cell_id,ids) or not np.array_equal(frame[genes].to_numpy(),counts):
        raise ValueError('CSV/NPY/cell/gene association mismatch')
    for obj in manifest['matrix_dependencies'].values():check_identity(obj)
    result=dict(status='succeeded',validation_kind='independent_full_native_resize_lookup_and_conservation',rows=rows,
                owner_differences=0,integer_count_differences=0,csv_numpy_equal=True,out_of_bounds=oob,
                conservation=manifest['totals'],mask=manifest['matrix_dependencies']['mask'],outputs=manifest['outputs'].copy(),
                source=identity(__file__),reference_algorithm=identity(Path(__file__).parents[1]/'crgf/matrix.py'))
    atomic_json(out/'validation.json',result)
    return result


def check_parent_conservation(parent,child,matrix,expected_assigned):
    pids,pgenes,pc=matrix_arrays(parent)
    cids,cgenes,cc=matrix_arrays(dict(matrix=str(matrix)))
    if len(cids)!=N_CELLS or len(cgenes)!=N_GENES or not np.array_equal(pids,cids) or pgenes!=cgenes:
        raise ValueError('parent/GT complete ID or gene order differs')
    if child['totals']['eligible']!=ELIGIBLE or child['totals']['assigned']!=expected_assigned or child['totals']['assigned']!=int(pc.sum()):
        raise ValueError('GT capture does not equal its own parent')
    if child['totals']['unassigned']!=ELIGIBLE-expected_assigned or not np.array_equal(pc.sum(0,dtype=np.int64),cc.sum(0,dtype=np.int64)):
        raise ValueError('parent-relative unassigned or per-gene global conservation failed')
    return dict(eligible=ELIGIBLE,assigned=expected_assigned,unassigned=ELIGIBLE-expected_assigned,
                per_gene_global_totals_exact=True,full_ids_exact=True,full_313_genes_exact=True,
                gene_totals=cc.sum(0,dtype=np.int64).tolist())


def produce_matrix(cfg,manifest,endpoint,callback=None):
    parent=manifest['endpoints'][PARENTS[endpoint]]
    root=within(cfg['run_root'],'endpoints',endpoint)
    matrix=root/'matrix'
    machine=dict(read(parent['machine']))
    machine.update(final_mask=str(root/'final.tif'),pre_mask=parent['final'],final_matrix=str(matrix/'expr_mat.csv'),
                   parent_final_mask=parent['final'],parent_endpoint=PARENTS[endpoint],production_kind='Geometry+Type_postprocessing_only')
    mp=root/'machine.json'
    if mp.exists() and read(mp)!=machine:raise ValueError('machine recovery config drift')
    atomic_json(mp,machine)
    deps=dependencies(cfg,endpoint+'_MATRIX')
    result=build(machine,manifest['coordinates'],matrix,deps,callback)
    conservation=check_parent_conservation(parent,result,matrix,ASSIGNED[endpoint])
    atomic_json(root/'matrix_conservation.json',conservation)
    return result,conservation
