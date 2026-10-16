"""Pooled spatial intervals using one frozen global W, and factorial contrasts."""
from pathlib import Path
import numpy as np
import pandas as pd
from .contracts import METRICS,SCALES,DAG,FACTORIAL8,METHODS,comparisons,comparison_id
from .common import atomic_csv,atomic_json,atomic_npz,dependencies,identity,read,id_hash,check_receipt,write_receipt,inventory,progress
from .populations import selected

def pooled_interval(sums,counts,W,*,minimum_units=20,minimum_blocks=10,minimum_valid=1900):
    """All supplied blocks remain in W, including zero-contribution blocks."""
    s=np.asarray(sums,float);n=np.asarray(counts,float);w=np.asarray(W)
    if s.shape!=n.shape or s.ndim!=1 or w.ndim!=2 or w.shape[1]!=len(s) or not np.isfinite(s).all() or not np.isfinite(n).all() or (n<0).any():raise ValueError('pooled block inputs invalid')
    total=float(n.sum());point=float(s.sum()/total) if total else None;nb=int(np.count_nonzero(n))
    dn=w@n;sn=w@s;samples=np.divide(sn,dn,out=np.full(len(w),np.nan),where=dn>0);valid=np.isfinite(samples)
    leave_n=total-n;leave=np.divide(s.sum()-s,leave_n,out=np.full(len(n),np.nan),where=leave_n>0)
    reason=('NA_zero_denominator' if not total else 'NA_sparse_units' if total<minimum_units else 'NA_insufficient_blocks' if nb<minimum_blocks else 'NA_insufficient_valid_replicates' if valid.sum()<minimum_valid else '')
    ci=np.quantile(samples[valid],[.025,.975],method='linear').tolist() if not reason else [None,None]
    return dict(point=point,ci_low=ci[0],ci_high=ci[1],n_units=int(total),contributing_blocks=nb,universe_blocks=len(n),
                valid_replicates=int(valid.sum()),replicates=len(w),sparse=bool(total<minimum_units),
                LOBO_min=float(np.nanmin(leave)) if np.isfinite(leave).any() else None,LOBO_max=float(np.nanmax(leave)) if np.isfinite(leave).any() else None,
                status='computed' if not reason else reason,NA_reason=reason)

def contrast_interval(values,coefficients,block_index,W):
    """Values share the identical finite population before applying coefficients."""
    arrays=[np.asarray(values[k],float) for k in coefficients]
    if not arrays or len({len(a) for a in arrays})!=1 or any(not np.isfinite(a).all() for a in arrays):raise ValueError('contrast must use one common finite population')
    delta=sum(coefficients[k]*np.asarray(values[k],float) for k in coefficients)
    n=np.bincount(block_index,minlength=W.shape[1]);s=np.bincount(block_index,weights=delta,minlength=W.shape[1])
    return pooled_interval(s,n,W)

def _intervals_for_groups(ids,values,cells,groups,weights,projector=None,checkpoint=None,callback=None):
    from .spatial_bootstrap import GroupProjector,block_intervals
    projector=projector or GroupProjector(cells,groups)
    sums,counts=projector.blocks(ids,values,int(weights['side']),weights['W'].shape[1])
    return block_intervals(sums,counts,weights['W'],projector.names,checkpoint=checkpoint,callback=callback)


def run(cfg,manifest,specs,tables,pop_root,out,callback=None,reuse_from=None):
    from .common import reuse_partition
    out=Path(out);out.mkdir(parents=True,exist_ok=True);methods=tuple(specs);deps=dependencies(cfg,'COMPARISONS_'+str(len(methods)))
    if (out/'complete.json').exists():return check_receipt(out/'complete.json',deps)
    frozen=Path(cfg['run_root'])/'frozen_domain';cells=pd.read_csv(frozen/'cells.csv').set_index('cell_id')
    groupsfile=np.load(frozen/'cell_groups.npz');groups={k:groupsfile[k] for k in groupsfile.files}
    weights={s:dict(np.load(frozen/f'bootstrap_weights_{s}.npz'),side=s) for s in SCALES}
    from .spatial_bootstrap import GroupProjector
    projector=GroupProjector(cells,groups)
    jobs=[(comparison_id(a,b),{a:1,b:-1},'pairwise') for a,b in comparisons(methods)]
    jobs += [('raw_'+m,{m:1},'raw_full_profile') for m in methods]
    if set(FACTORIAL8).issubset(methods):jobs += [(c['id'],c['coefficient'],'factorial8') for c in DAG['factorial_contrasts']]
    if set(METHODS).issubset(methods):jobs += [('all12_'+m,{m:1},'all12') for m in METHODS]
    summaries=[]
    for index,(key,coeff,kind) in enumerate(jobs):
        dest=out/key;dest.mkdir(exist_ok=True);dep=dict(deps,comparison=key,coefficient=coeff,population=kind)
        raw_method=next(iter(coeff)) if kind=='raw_full_profile' else None
        pophash=({p+'__'+metric:id_hash(tables[raw_method][p].index[np.isfinite(tables[raw_method][p][metric])]) for p in ('sc','fixed') for metric in METRICS}
                 if raw_method else {metric:id_hash(selected(pop_root,key if kind=='pairwise' else kind,metric)) for metric in METRICS})
        binding=dict(coefficient=coeff,population=kind,methods=list(coeff),populations=pophash,
            scores={m:{p:identity(specs[m][p+'_scores']) for p in ('sc','fixed')} for m in coeff},
            weights={str(s):identity(frozen/f'bootstrap_weights_{s}.npz') for s in SCALES},groups=identity(frozen/'cell_groups.npz'))
        if (dest/'complete.json').exists():check_receipt(dest/'complete.json',dep);continue
        source=Path(reuse_from)/key if reuse_from is not None else None
        if source is not None and (source/'complete.json').exists():reuse_partition(source,dest,dep,binding);continue
        results=[]
        for metric in METRICS:
            ids=None if raw_method else selected(pop_root,key if kind=='pairwise' else kind,metric)
            for profile in ('sc','fixed'):
                if raw_method:ids=tables[raw_method][profile].index[np.isfinite(tables[raw_method][profile][metric])].to_numpy(np.int64)
                population_hash=id_hash(ids)
                values=sum(c*tables[m][profile][metric].reindex(ids).to_numpy(float) for m,c in coeff.items())
                if not np.isfinite(values).all():raise ValueError('nonfinite values escaped common population')
                for scale in SCALES:
                    for row in _intervals_for_groups(ids,values,cells,groups,weights[scale],projector,(dest/'bootstrap'/f'{profile}_{metric}_{scale}',dict(dep,profile=profile,metric=metric,scale=scale,population_ids_sha256=population_hash)),callback):
                        row.update(comparison=key,profile=profile,metric=metric,scale_um=scale,population=kind,population_ids_sha256=population_hash,
                                   unit='cell_full_expression_score',coefficient=json_coefficient(coeff))
                        results.append(row)
        # Compression is explicit and deterministic; no millions-row in-memory
        # cross-comparison tensor and no top-N truncation.
        path=dest/'stratified_intervals.csv.gz';atomic_csv_gzip(path,pd.DataFrame(results))
        atomic_json(dest/'definition.json',dict(coefficient=coeff,population=kind,shared_global_weights=True,rows=len(results)))
        write_receipt(dest/'complete.json',dep,inventory(dest),rows=len(results),coefficient=coeff,population=kind,reuse_binding=binding)
        progress(callback,'prespecified_comparisons',index+1,len(jobs),receipt=str(dest/'complete.json'))
    # Original ownQC/fixed whole-panel comparisons remain distinct from paired diagnostics.
    raw=pd.read_csv(Path(pop_root)/'raw_full_population.csv');rows=[]
    for a,b in comparisons(methods):
        for profile in ('sc','fixed'):
            for metric in METRICS:
                ar=raw[(raw.endpoint==a)&(raw.profile==profile)&(raw.metric==metric)].iloc[0];br=raw[(raw.endpoint==b)&(raw.profile==profile)&(raw.metric==metric)].iloc[0]
                rows.append(dict(candidate=a,control=b,profile=profile,metric=metric,candidate_mean=ar['mean'],control_mean=br['mean'],delta=ar['mean']-br['mean'],
                                 candidate_finite=int(ar.n_finite),control_finite=int(br.n_finite),population='original_full_profile_not_paired'))
    atomic_csv(out/'raw_differences.csv',pd.DataFrame(rows))
    b0=manifest['background']['B0'];background=[]
    for profile,key in [('sc','sc_scores'),('fixed','fixed_scores')]:
        f=pd.read_csv(b0[key])
        for metric in METRICS:
            v=f[metric].to_numpy(float);v=v[np.isfinite(v)];mean=v.mean() if len(v) else np.nan
            for m in methods:
                ar=raw[(raw.endpoint==m)&(raw.profile==profile)&(raw.metric==metric)].iloc[0]
                background.append(dict(candidate=m,control='B0',profile=profile,metric=metric,delta=ar['mean']-mean,
                                       paired=False,interpretation='historical training-unpaired performance background only'))
    atomic_csv(out/'B0_historical_background.csv',pd.DataFrame(background))
    atomic_json(out/'coverage.json',dict(jobs=[j[0] for j in jobs],comparisons=[comparison_id(a,b) for a,b in comparisons(methods)],
                 factorial=19 if set(FACTORIAL8).issubset(methods) else 0,metrics=list(METRICS),profiles=['sc','fixed'],scales=list(SCALES),strata=list(groups)))
    return write_receipt(out/'complete.json',deps,inventory(out),methods=list(methods),jobs=[j[0] for j in jobs])

def json_coefficient(coeff):
    import json
    return json.dumps(coeff,sort_keys=True,separators=(',',':'))

def atomic_csv_gzip(path,frame):
    import gzip
    from .common import atomic
    def writer(f):
        with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as zipped:
            import io
            text=io.TextIOWrapper(zipped,encoding='utf-8',newline='');frame.to_csv(text,index=False,na_rep='NA');text.flush();text.detach()
    atomic(path,writer)
