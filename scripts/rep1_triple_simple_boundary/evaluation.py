"""Four complete evaluations under admitted compute authorization, before resume."""
from __future__ import annotations
from pathlib import Path
import os
import subprocess
import time
import signal
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from scripts.new_methods.author_marker import fixed_scores, METRICS
from . import NEW, PARENTS, N_CELLS, N_GENES
from .common import (atomic_csv, atomic_json, check_identity, check_receipt, chunk_receipt,
                     dependencies, digest, identity, inventory, progress, read, require_compute_authorization,committed,
                     resume_chunk, within, write_receipt, fsync_dir)
from .inputs import load_inputs, matrix_arrays, new_spec
from .bt import checked_ids, mask_moments



def source_runtime(cfg,deps):
    snapshot=Path(cfg['source_snapshot']).resolve(strict=True)
    actual=Path(__file__).resolve()
    if snapshot not in actual.parents:
        raise RuntimeError('formal evaluation must execute the audited snapshot module')
    result=dict(source_snapshot=str(snapshot),dependencies_digest=digest(deps))
    mapping={'official_expression':'publish_expression_metrics.R','official_processing':'publish_processing.R',
             'official_utils':'publish_utils.R','official_spatial':'publish_spatialDiversity_metrics.R'}
    for key,name in mapping.items():
        path=snapshot/'external_sources'/name
        if not path.is_file() or path.is_symlink():raise RuntimeError('missing immutable snapshot R source: '+str(path))
        result[key]=str(path)
    result['allowed_sources']=[result[k] for k in mapping]
    return result


def commit_r_chunks(folder,deps_digest,callback=None):
    folder=Path(folder)
    for pending in sorted(folder.glob('*.pending.json')):
        meta=read(pending)
        target=Path(meta['file']).resolve()
        if target.parent!=folder.resolve() or meta.get('dependencies_digest')!=deps_digest:
            raise ValueError('R prediction batch checkpoint dependency/path mismatch')
        committed=target.with_name(target.name+'.committed.json')
        if committed.exists():
            saved=read(committed)
            if saved.get('dependencies_digest')!=deps_digest:raise ValueError('R committed dependency mismatch')
            check_identity(saved)
            continue
        fd=os.open(target,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
        obj=dict(**identity(target),dependencies_digest=deps_digest,cursor=meta['cursor'],total=meta['total'])
        atomic_json(committed,obj)
        progress(callback,'R_scClassify_prediction',meta['cursor'],meta['total'],file=str(target),receipt=str(committed))


def run_r(cfg,script,args,out,lock_fds,validate_execution,callback=None,checkpoint_digest=None,status_file='execution_status.txt'):
    """One explicit subprocess path; no legacy inherited_lock_fds hardcoded config."""
    if not lock_fds or validate_execution is None:
        raise PermissionError('R requires validated inherited run/workspace lock descriptors and snapshot validator')
    for fd in lock_fds:
        os.fstat(fd)
    validate_execution()
    snapshot=Path(cfg['source_snapshot']).resolve(strict=True)
    script=Path(script).resolve(strict=True)
    if snapshot not in script.parents:raise RuntimeError('R script is outside snapshot')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    temporary=out/'tmp';temporary.mkdir(exist_ok=True)
    threads=str(cfg.get('evaluation_threads',4))
    env=dict(os.environ,OMP_NUM_THREADS=threads,OPENBLAS_NUM_THREADS=threads,MKL_NUM_THREADS=threads,
             R_PROFILE_USER='/dev/null',R_ENVIRON_USER='/dev/null',R_PROFILE='/dev/null',R_ENVIRON='/dev/null',
             R_LIBS_USER=cfg.get('r_libs_user',''),R_LIBS_SITE=cfg.get('r_libs_site',''),
             REP1_TRIPLE_SIMPLE_LOCK_FDS=','.join(map(str,lock_fds)),TMPDIR=str(temporary.resolve()))
    cmd=[cfg['rscript'],'--vanilla',str(script),*map(str,args)]
    source_identity=identity(script)
    with (out/(script.stem+'.log')).open('a') as log:
        proc=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                              env=env,cwd=snapshot,close_fds=True,pass_fds=tuple(lock_fds),start_new_session=True)
        atomic_json(out/(script.stem+'_process.json'),dict(pid=proc.pid,argv=cmd,cwd=str(snapshot),
                    source=source_identity,lock_fds=list(lock_fds),process_group=proc.pid,execution='running'))
        failure=None
        try:
            while proc.poll() is None:
                validate_execution()
                if checkpoint_digest is not None:commit_r_chunks(out/'prediction_chunks',checkpoint_digest,callback)
                time.sleep(.5)
            if checkpoint_digest is not None:commit_r_chunks(out/'prediction_chunks',checkpoint_digest,callback)
        except BaseException as exc:
            # This is an engineering failure in this owned subprocess stage,
            # never a timer or Goal pause. Stop the entire owned R process group
            # and observe its real exit while inherited locks remain open.
            failure=exc
            if proc.poll() is None:
                try:os.killpg(proc.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    try:os.killpg(proc.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    proc.wait(timeout=10)
    atomic_json(out/(script.stem+'_exit.json'),dict(exit_code=proc.returncode,signal=-proc.returncode if proc.returncode<0 else None,
                source=source_identity,cwd=str(snapshot),lock_fds=list(lock_fds),
                engineering_failure=None if failure is None else repr(failure),actual_exit_observed=True))
    atomic_json(out/(script.stem+'_process.json'),dict(pid=proc.pid,argv=cmd,cwd=str(snapshot),
                source=source_identity,lock_fds=list(lock_fds),process_group=proc.pid,execution='exited',exit_code=proc.returncode))
    if failure is not None:raise failure
    if proc.returncode!=0:raise RuntimeError(f'R child {script.name} failed with exit code {proc.returncode}')
    if not (out/status_file).exists() or (out/status_file).read_text().strip()!='succeeded':
        raise ValueError('R successful exit lacks execution status')
    validate_execution()


def structural(spec,manifest,out,deps,callback=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return check_receipt(out/'complete.json',deps)
    old=Path(spec['evaluation'])/'structure'
    if old.resolve()!=out.resolve() and (old/'complete.json').exists():
        saved=check_receipt(old/'complete.json');frame=pd.read_csv(old/'cells_distribution.csv')
        ids,genes,counts=matrix_arrays(spec)
        required={'cell_id','area_um2','x_um','y_um','transcripts_per_cell','genes_per_cell','components_4','components_8','nuclei_overlapped'}
        if not required.issubset(frame) or not np.array_equal(frame.cell_id,ids) or not np.array_equal(frame.transcripts_per_cell,counts.sum(1)) or not np.array_equal(frame.genes_per_cell,(counts>0).sum(1)):
            raise ValueError('existing full structure no longer matches matrix/ID contract')
        summary=read(old/'structure.json')
        if summary['n_cells']!=len(ids) or summary['nuclear_preservation_fraction']!=1 or not frame.nuclei_overlapped.eq(1).all():raise ValueError('existing full structure nuclear semantics differ')
        atomic_csv(out/'cells_distribution.csv',frame);atomic_json(out/'structure.json',summary)
        return write_receipt(out/'complete.json',deps,inventory(out),summary=summary,reused_full_structure=identity(old/'complete.json'))
    machine=read(spec['machine']);mask=tifffile.imread(machine['final_mask']);nuc=tifffile.imread(machine['nuclei_mask'])
    ids,genes,counts=matrix_arrays(spec)
    props=mask_moments(mask,ids,out/'moments',dict(deps,mask=identity(machine['final_mask'])),callback)
    boxes=ndi.find_objects(mask);meta=np.load(manifest['metadata'])
    if not np.array_equal(ids,meta['ids']):raise ValueError('structure metadata ID mismatch')
    tables=[]
    for start in range(0,len(ids),1000):
        stop=min(start+1000,len(ids));p=out/f'components_{start:06d}.csv';r=p.with_suffix('.json');cd=dict(deps,start=start,stop=stop)
        if resume_chunk(r,cd) is None:
            rows=[]
            for cid in ids[start:stop]:
                box=boxes[int(cid)-1];cell=mask[box]==cid
                n4=ndi.label(cell,ndi.generate_binary_structure(2,1))[1]
                n8=ndi.label(cell,np.ones((3,3),np.uint8))[1]
                nuclear_ids=np.unique(nuc[box][cell]);nuclear_ids=nuclear_ids[nuclear_ids>0]
                rows.append(dict(cell_id=int(cid),components_4=n4,components_8=n8,nuclei_overlapped=len(nuclear_ids)))
            atomic_csv(p,pd.DataFrame(rows));chunk_receipt(r,cd,[p])
        tables.append(pd.read_csv(p));progress(callback,'evaluation_structure',stop,len(ids))
    comp=pd.concat(tables,ignore_index=True).set_index('cell_id')
    table=props.rename(columns={'area':'area_um2'}).copy()
    table['transcripts_per_cell']=counts.sum(1);table['genes_per_cell']=(counts>0).sum(1)
    table['nuclear_area_um2']=meta['area'];table['cell_nucleus_area_ratio']=table.area_um2/table.nuclear_area_um2
    table=table.join(comp,validate='one_to_one')
    exact=float(np.mean(mask[nuc>0]==nuc[nuc>0]))
    summary=dict(n_cells=len(ids),mean_transcripts_per_cell=float(table.transcripts_per_cell.mean()),
                 mean_genes_per_cell=float(table.genes_per_cell.mean()),mean_cell_area_um2=float(table.area_um2.mean()),
                 median_cell_area_um2=float(table.area_um2.median()),p95_cell_area_um2=float(table.area_um2.quantile(.95)),
                 disconnected_cells_4=int((table.components_4>1).sum()),disconnected_cells_8=int((table.components_8>1).sum()),
                 nuclear_preservation_fraction=exact,foreground_fraction=float((mask>0).mean()),
                 zero_nucleus_cells=int((table.nuclei_overlapped==0).sum()),multi_nucleus_cells=int((table.nuclei_overlapped>1).sum()),
                 true_boundary_accuracy=None,true_boundary_accuracy_reason='no independent membrane/manual owner ground truth')
    atomic_csv(out/'cells_distribution.csv',table.reset_index())
    atomic_json(out/'structure.json',summary)
    return write_receipt(out/'complete.json',deps,inventory(out),summary=summary)


def fixed_panel(spec,manifest,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    fixed=checked_ids(pd.read_csv(manifest['fixed_population']),'fixed panel')
    if len(fixed)!=103000:raise ValueError('fixed panel size drift')
    labels=fixed.evaluation_label.fillna('__unavailable__:unavailable_other').copy()
    reasons=fixed.unmatched_reason.fillna('')
    labels.loc[reasons!='']='__unavailable__:'+reasons.loc[reasons!='']
    ids,genes,counts=matrix_arrays(spec)
    prepared=Path(manifest['evaluation']['prepared'])
    scores=fixed_scores(counts,ids,genes,labels,read(prepared/'markers.json'),(prepared/'reference_genes.txt').read_text().splitlines())
    scores=scores.reindex(fixed.index);scores.index.name='cell_id'
    missing=~scores.index.isin(ids);scores.loc[missing,'label_availability']='missing_matrix_row';scores.loc[missing,METRICS]=np.nan
    atomic_csv(out/'fixed_population.csv',scores,index=True)
    rows=[]
    for group,frame in [('__all__',scores),*list(scores.groupby('anchor_label',dropna=False))]:
        for metric in METRICS:
            values=frame[metric].to_numpy(float);finite=np.isfinite(values)
            rows.append(dict(cell_type=str(group),metric=metric,n_population=len(values),n_finite=int(finite.sum()),
                             n_unavailable=int((~finite).sum()),mean=float(values[finite].mean()) if finite.any() else None))
    atomic_csv(out/'fixed_summary.csv',pd.DataFrame(rows))
    return dict(population=len(scores),missing_rows=int(missing.sum()),metrics=[r for r in rows if r['cell_type']=='__all__'])


def evaluate_endpoint(cfg,manifest,name,authorization,callback,lock_fds,validate_execution):
    from .contracts import EVALUATION_STAGES
    stage=next(k for k,v in EVALUATION_STAGES.items() if v==name)
    production=within(cfg['run_root'],'endpoints',name,'production_complete.json')
    require_compute_authorization(cfg,authorization,stage,production)
    if not lock_fds or validate_execution is None:
        raise PermissionError('new evaluation requires inherited lock descriptors and execution/snapshot validator')
    for fd in lock_fds:os.fstat(fd)
    validate_execution()
    spec=new_spec(cfg,name);out=Path(spec['evaluation']);deps=dependencies(cfg,stage)
    deps.update(production=identity(production))
    if (out/'complete.json').exists():return check_receipt(out/'complete.json',deps,cfg['run_root'])
    out.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    runtime=source_runtime(cfg,deps);rp=out/'r_runtime.json';atomic_json(rp,runtime)
    sr=structural(spec,manifest,out/'structure',dict(deps,part='structure'),callback)
    props=pd.read_csv(out/'structure/cells_distribution.csv')[['cell_id','x_um','y_um']]
    paper=out/'paper_aligned_v1';paper.mkdir(exist_ok=True)
    atomic_csv(paper/'own_cell_centroids.csv',props)
    machine=read(spec['machine'])
    pdeps=dict(deps,part='paper')
    if (paper/'complete.json').exists():check_receipt(paper/'complete.json',pdeps,cfg['run_root'])
    else:
        run_r(cfg,Path(__file__).with_name('paper_evaluation.R'),
              [manifest['evaluation']['protocol'],manifest['evaluation']['prepared'],machine['final_matrix'],paper,rp],
              paper,lock_fds,validate_execution,callback,digest(deps))
        ids,genes,counts=matrix_arrays(spec)
        total=counts.sum(1);detected=(counts>0).sum(1);q=(total>=20)&(total<=2000)&(detected>=20)
        qc=checked_ids(pd.read_csv(paper/'qc_all_cells.csv'),'new QC');ann=checked_ids(pd.read_csv(paper/'annotation.csv'),'new annotation')
        scores=checked_ids(pd.read_csv(paper/'markers_per_cell.csv'),'new sc scores')
        if not np.array_equal(qc.index,ids) or not np.array_equal(qc.qc_pass,q) or not np.array_equal(ann.index,ids[q]) or not np.array_equal(scores.index,ann.index):
            raise ValueError('new paper exact QC/annotation/score population mismatch')
        if not np.isfinite(scores[['positive_F1','negative_F1']]).all().all():raise ValueError('paper F1 has nonfinite values')
        write_receipt(paper/'complete.json',pdeps,inventory(paper),model=identity(manifest['evaluation']['reference_model']))
    fixed=fixed_panel(spec,manifest,out/'fixed')
    # Author spatial statistics use their own immutable helpers from the snapshot.
    spatial_files=['paper_aligned_v1/spatial_diversity_bins.csv','paper_aligned_v1/spatial_diversity_correlations.csv']
    spdeps=dict(deps,part='spatial')
    if (out/'spatial_complete.json').exists():check_receipt(out/'spatial_complete.json',spdeps,cfg['run_root'])
    else:
        run_r(cfg,Path(__file__).with_name('spatial_diversity.R'),[rp,out],out,lock_fds,validate_execution,callback,status_file='spatial_execution_status.txt')
        write_receipt(out/'spatial_complete.json',spdeps,{p:identity(out/p) for p in spatial_files})
        progress(callback,'author_spatial_diversity',1,1,endpoint=name)
    from .bt import run_endpoint
    bt=run_endpoint(cfg,manifest,spec,out/'bt',callback)
    from .evaluation_pairs import run as run_pairs
    paired=run_pairs(cfg,manifest,name,callback)
    scores=pd.read_csv(spec['sc_scores']);ann=pd.read_csv(paper/'annotation.csv')
    metrics={}
    for profile,path in [('scclassify',spec['sc_scores']),('fixed',spec['fixed_scores'])]:
        frame=pd.read_csv(path)
        for metric in METRICS:
            v=frame[metric].to_numpy(float);finite=np.isfinite(v)
            metrics[profile+'_'+metric]=dict(mean=float(v[finite].mean()) if finite.any() else None,n_finite=int(finite.sum()),n_population=len(v))
    summary=dict(endpoint=name,parent=PARENTS[name],metrics=metrics,structure=sr['summary'],fixed_population=fixed,
                 capture=read(Path(spec['matrix'])/'complete.json')['totals'],n_before_qc=N_CELLS,n_after_qc=len(scores),n_unassigned=int(ann.rejected.sum()),
                 n_missing_sc_labels=int(ann.evaluation_label.isna().sum()),n_unknown_sc_labels=int(ann.evaluation_label.eq('Unknown').sum()),
                 unassigned_fraction=float(ann.rejected.mean()),qc_retention=len(scores)/N_CELLS,
                 model_sha256=identity(manifest['evaluation']['reference_model'])['sha256'],
                 wall_seconds=time.monotonic()-started,profiles=['paper_scClassify','historical_B1_fixed_103000'],
                 no_true_boundary_accuracy=True)
    atomic_json(out/'summary.json',summary)
    from .coverage import validate_endpoint_products
    coverage=validate_endpoint_products(spec,manifest=manifest,cfg=cfg)
    atomic_json(out/'coverage.json',coverage)
    return write_receipt(out/'complete.json',deps,inventory(out),paper_method_complete=True,full_evaluate=True,summary=summary,coverage=coverage)
