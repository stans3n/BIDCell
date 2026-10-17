"""Separate raw, pairwise four-way, factorial8 and all12 finite populations."""
from pathlib import Path
import numpy as np
import pandas as pd
from .contracts import METRICS,FACTORIAL8,METHODS,comparisons,comparison_id
from .common import (atomic_csv,atomic_json,atomic_npz,dependencies,identity,id_hash,read,
                     check_receipt,write_receipt,inventory,progress,reuse_partition,partition_source)
from .bt import checked_ids
from .inputs import matrix_arrays
from .continuation import read_data

def endpoint_scores(spec,fixed_ids):
    ids,genes,counts=matrix_arrays(spec);total=counts.sum(1,dtype=np.int64);ng=(counts>0).sum(1)
    qc=(total>=20)&(total<=2000)&(ng>=20)
    sc=checked_ids(pd.read_csv(spec['sc_scores']),spec['name']+' sc')
    fixed=checked_ids(pd.read_csv(spec['fixed_scores']),spec['name']+' fixed')
    sourceqc=checked_ids(pd.read_csv(Path(spec['evaluation'])/'paper_aligned_v1/qc_all_cells.csv'),spec['name']+' qc')
    if not np.array_equal(sc.index,ids[qc]) or not np.array_equal(sourceqc.index,ids) or not np.array_equal(sourceqc.qc_pass,qc):raise ValueError('score/QC/313 matrix population mismatch')
    if not np.array_equal(fixed.index,np.sort(fixed_ids)):raise ValueError('fixed full panel lost IDs')
    if not set(METRICS).issubset(sc) or not set(METRICS).issubset(fixed):raise ValueError('six metrics missing')
    return dict(sc=sc,fixed=fixed,qc_ids=ids[qc],all_ids=ids,counts=total,genes=ng)

def common_finite(tables,methods,fixed_ids,metric):
    """Never infer missing rows as zero; legal exact zero remains finite."""
    full=np.sort(np.asarray(fixed_ids,np.int64));eligible=np.ones(len(full),bool);reasons={}
    for name in methods:
        q=np.isin(full,tables[name]['qc_ids']);reasons[name+'__QC_excluded']=~q;eligible&=q
    initial=full[eligible];finite=np.ones(len(initial),bool);exclusions={}
    for name in methods:
        for profile in ('sc','fixed'):
            source=tables[name][profile];present=np.isin(initial,source.index)
            values=source[metric].reindex(initial).to_numpy(float);ok=np.isfinite(values)
            exclusions[name+'__'+profile+'__missing_row']=~present
            exclusions[name+'__'+profile+'__nonfinite']=~ok
            finite&=ok
    return initial,initial[finite],dict(initial_ids=initial,four_way_or_all_finite=finite,**exclusions),dict(fixed_ids=full,**reasons)

def prepare(cfg,manifest,specs,out,callback=None,reuse_from=None,pair_list=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);methods=tuple(specs)
    deps=dependencies(cfg,'POPULATIONS_'+str(len(methods)))
    fixed=pd.read_csv(manifest['fixed_population']).cell_id.to_numpy(np.int64)
    tables={name:endpoint_scores(spec,fixed) for name,spec in specs.items()}
    if (out/'complete.json').exists():check_receipt(out/'complete.json',deps);return tables
    pairs=comparisons(methods) if pair_list is None else pair_list
    if any(a not in specs or b not in specs for a,b in pairs):raise ValueError('paired population endpoint unavailable')
    rows=[];raw=[];sets=[(comparison_id(a,b),(a,b),'pairwise') for a,b in pairs]
    if pair_list is None and set(FACTORIAL8).issubset(methods):sets.append(('factorial8',FACTORIAL8,'factorial8'))
    if pair_list is None and set(METHODS).issubset(methods):sets.append(('all12',METHODS,'all12'))
    for name,t in tables.items():
        for profile in ('sc','fixed'):
            for metric in METRICS:
                v=t[profile][metric].to_numpy(float);ok=np.isfinite(v)
                raw.append(dict(endpoint=name,profile=profile,metric=metric,n_population=len(v),n_finite=int(ok.sum()),mean=float(v[ok].mean()) if ok.any() else None,NA_reason='' if ok.any() else 'no_finite_scores',population='raw_ownQC' if profile=='sc' else 'raw_fixed_full103000'))
    atomic_csv(out/'raw_full_population.csv',pd.DataFrame(raw))
    for index,(key,names,kind) in enumerate(sets):
        dest=out/key;dest.mkdir(exist_ok=True);manifest_rows=[]
        binding=dict(methods=list(names),kind=kind,fixed=identity(manifest['fixed_population']),
            metrics=list(METRICS),scores={m:{p:identity(specs[m][p+'_scores']) for p in ('sc','fixed')} for m in names},
            QC={m:id_hash(tables[m]['qc_ids']) for m in names})
        cd=dict(deps,key=key)
        source=partition_source(reuse_from,key)
        if (dest/'complete.json').exists():
            rec=check_receipt(dest/'complete.json',cd)
            if rec.get('reuse_binding')!=binding:raise ValueError('population binding drift')
            rows.extend(read(read_data(dest/'populations.json')));continue
        if source is not None and (source/'complete.json').exists():
            reuse_partition(source,dest,cd,binding,cfg=cfg);rows.extend(read(read_data(dest/'populations.json')));continue
        for metric in METRICS:
            initial,selected,excl,qexcl=common_finite(tables,names,fixed,metric)
            p=dest/(metric+'.npz');atomic_npz(p,ids=selected,**excl)
            if metric==METRICS[0]:atomic_npz(dest/'QC_exclusions.npz',**qexcl)
            row=dict(population=kind,key=key,metric=metric,n_initial=len(initial),n_finite=len(selected),n_excluded=len(initial)-len(selected),ids_sha256=id_hash(selected),file=identity(p),NA_reason='' if len(selected) else 'no_common_finite_IDs')
            rows.append(row);manifest_rows.append(row)
        if len(names)==2:
            a,b=names;common=np.intersect1d(tables[a]['qc_ids'],tables[b]['qc_ids'])
            la=tables[a]['sc']['cell_type'].reindex(common).fillna('__missing__');lb=tables[b]['sc']['cell_type'].reindex(common).fillna('__missing__')
            transition=pd.crosstab(lb,la,dropna=False);atomic_csv(dest/'sc_label_migration.csv',transition,index=True)
        atomic_json(dest/'populations.json',manifest_rows);write_receipt(dest/'complete.json',cd,inventory(dest),methods=list(names),kind=kind,reuse_binding=binding)
        progress(callback,'finite_populations',index+1,len(sets),receipt=str(dest/'complete.json'))
    atomic_json(out/'populations.json',rows)
    write_receipt(out/'complete.json',deps,inventory(out),methods=list(methods),sets=[x[0] for x in sets],raw_profiles_retained=True)
    return tables

def selected(pop_root,key,metric):
    z=np.load(read_data(Path(pop_root)/key/(metric+'.npz')));return z['ids']
