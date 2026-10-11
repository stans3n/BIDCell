from .tables import read_table,write_table,table_path
"""Stage comparisons on an immutable population, with explicit missing domains."""
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.cos.diagnostics import METRICS,load_anchor
from scripts.crgf.common import atomic_json,identity,sha256

def load_population(spec):
    if sha256(spec['path'])!=spec['sha256']:raise ValueError('frozen population drift')
    return population_index(pd.read_csv(spec['path']).cell_id)

def population_index(values):
    raw=pd.Series(list(values))
    numeric=pd.to_numeric(raw,errors='raise')
    if numeric.isna().any() or not np.isfinite(numeric).all() or (numeric<=0).any() or (numeric!=np.floor(numeric)).any():
        raise ValueError('population IDs must be finite positive integers')
    ids=pd.Index(numeric.to_numpy(np.int64),name='cell_id')
    if ids.has_duplicates:raise ValueError('duplicate population ID')
    return ids

def align_stage(table,population,labels):
    """Reindex; never intersect or replace an unavailable score with zero."""
    ids=population_index(population)
    if not {'cell_id','anchor_label','label_availability',*METRICS}<=set(table):raise ValueError('stage table schema incomplete')
    own_ids=population_index(table.cell_id)
    if labels.index.has_duplicates:raise ValueError('duplicate frozen label ID')
    original=table.copy();original.index=own_ids
    expected=labels.reindex(ids).fillna('missing_historical_label').astype(str)
    aligned=original.reindex(ids);present=ids.isin(own_ids)
    if not np.array_equal(aligned.loc[present,'anchor_label'].astype(str).to_numpy(),expected.loc[present].to_numpy()):
        raise ValueError('stage label differs from frozen historical label')
    aligned=aligned.drop(columns='cell_id');aligned['anchor_label']=expected
    aligned['stage_row_present']=present
    aligned['label_availability']=aligned.label_availability.fillna('missing_stage_id')
    known=present & aligned.label_availability.eq('known').to_numpy()
    for metric in METRICS:
        values=pd.to_numeric(aligned[metric],errors='raise').to_numpy(float)
        reason=np.full(len(ids),'available',dtype=object)
        reason[~np.isfinite(values)]='undefined_original_metric'
        reason[np.isinf(values)]='nonfinite_original_metric'
        if 'zero_reference_counts' in aligned:reason[present & aligned.zero_reference_counts.fillna(False).astype(bool).to_numpy()]='zero_reference_counts'
        reason[present & ~known]='label_ineligible'
        reason[~present]='missing_stage_id'
        values[reason!='available']=np.nan
        aligned[metric]=values;aligned[metric+'_reason']=reason
    return aligned,dict(n_population=len(ids),n_stage_rows=len(own_ids),n_present=int(present.sum()),n_missing_stage_rows=int((~present).sum()),n_extra_stage_ids=int((~own_ids.isin(ids)).sum()))

def postprocess_increment(methods,anchors,pairs,population,out,stages=('pre','final')):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);ids=population_index(population)
    rows=[];domains=[];inputs={};aligned={};inventory={}
    used=sorted(set(x for pair in pairs for x in pair))
    for anchor,spec in anchors.items():
        labels=load_anchor(spec)
        for arm in used:
            for stage in stages:
                path=table_path(Path(methods[arm]['evaluation'])/'diagnostics/stages'/f'fixed_{anchor}_{stage}.csv')
                table,counts=align_stage(pd.read_csv(path),ids,labels)
                key=(anchor,arm,stage);aligned[key]=table;inventory['/'.join(key)]=counts;inputs['/'.join(key)]=identity(path)
                dest=out/'stage_population'/f'{arm}_{anchor}_{stage}.csv';dest.parent.mkdir(exist_ok=True);write_table(table,dest,na_rep='NA')
                for metric in METRICS:
                    reason=table[metric+'_reason'];n=int(reason.eq('available').sum())
                    domains.append(dict(anchor=anchor,method=arm,stage=stage,metric=metric,**counts,n_evaluable=n,n_unavailable=len(ids)-n,**{'n_'+k:int(reason.eq(k).sum()) for k in ('available','missing_stage_id','label_ineligible','zero_reference_counts','undefined_original_metric','nonfinite_original_metric')}))
        for a,b in pairs:
            values={f'{arm}_{stage}':aligned[(anchor,arm,stage)] for arm in (a,b) for stage in stages}
            cell=pd.DataFrame(index=ids);cell['anchor_label']=labels.reindex(ids).fillna('missing_historical_label')
            for metric in METRICS:
                pre=values[a+'_'+stages[0]][metric]-values[b+'_'+stages[0]][metric]
                final=values[a+'_'+stages[1]][metric]-values[b+'_'+stages[1]][metric]
                finite=np.isfinite(pre)&np.isfinite(final);n=int(finite.sum())
                reason=np.full(len(ids),'available',dtype=object)
                # Mutually exclusive reason priority; detailed endpoint reasons stay in aligned tables.
                for category in ('zero_reference_counts','undefined_original_metric','nonfinite_original_metric','label_ineligible','missing_stage_id'):
                    any_bad=np.logical_or.reduce([t[metric+'_reason'].eq(category).to_numpy() for t in values.values()]);reason[any_bad]=category
                if not np.array_equal(finite.to_numpy(),reason=='available'):raise ValueError('paired validity/reason mismatch')
                cell[metric+'_pre_increment']=pre.where(finite)
                cell[metric+'_final_increment']=final.where(finite)
                cell[metric+'_postprocess_rewrite']=(final-pre).where(finite)
                cell[metric+'_joint_evaluable']=finite;cell[metric+'_reason']=reason
                rows.append(dict(from_stage=stages[0],to_stage=stages[1],comparison=a+'-'+b,anchor=anchor,metric=metric,population='frozen_historical_B1QC',n_population=len(ids),n_evaluable=n,n_unavailable=len(ids)-n,pre_cos_increment=float(pre[finite].mean()) if n else None,final_cos_increment=float(final[finite].mean()) if n else None,postprocess_rewrite=float((final-pre)[finite].mean()) if n else None,**{'n_'+category:int((reason==category).sum()) for category in ('missing_stage_id','label_ineligible','zero_reference_counts','undefined_original_metric','nonfinite_original_metric')},status='computed' if n==len(ids) else 'partial_domain_NA' if n else 'empty_evaluable_domain'))
            write_table(cell,out/f'postprocess_{a}_minus_{b}_{anchor}_per_cell.csv',na_rep='NA')
    pd.DataFrame(rows).to_csv(out/'postprocess_cos_increment.csv',index=False,na_rep='NA')
    pd.DataFrame(domains).to_csv(out/'stage_population_domains.csv',index=False)
    atomic_json(out/'stage_population_contract.json',dict(version='SSP-fixed-stage-population-v1',population_count=len(ids),population_order=ids.tolist(),inputs=inputs,inventory=inventory,missing_score_policy='NA; no intersection and no score imputation',paired_domain='all four method/pre/final metric values finite and label-eligible',reason_priority=['missing_stage_id','label_ineligible','nonfinite_original_metric','undefined_original_metric']))
