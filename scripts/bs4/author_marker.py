"""Author marker arithmetic; explicit profile eligibility stays outside this helper."""
import numpy as np
import pandas as pd

METRICS=[s+'_'+m for s in ('positive','negative') for m in ('F1','Precision','Recall')]

def marker_metrics(expression,genes,labels,marker_list):
    x=np.asarray(expression);genes=pd.Index(genes);labels=pd.Series(list(labels),dtype=object)
    if x.ndim!=2 or x.shape!=(len(labels),len(genes)) or not len(genes):raise ValueError('complete expression row/gene domain required')
    if genes.has_duplicates or genes.isna().any() or labels.isna().any():raise ValueError('duplicate genes or unmapped missing label')
    if not np.isfinite(x).all() or np.any(x<0):raise ValueError('damaged expression values')
    if not marker_list:raise ValueError('marker type inventory absent')
    values=np.zeros((len(labels),3),float)
    for label,markers in marker_list.items():
        rows=np.flatnonzero(labels.to_numpy()==label)
        if not len(rows):continue
        actual=np.isin(genes,markers);xx=x[rows]
        predicted=xx>np.quantile(xx,1-actual.mean(),axis=1,method='linear')[:,None]
        tp=predicted[:,actual].sum(axis=1)
        with np.errstate(divide='ignore',invalid='ignore'):
            precision=tp/predicted.sum(axis=1);recall=tp/actual.sum();f1=2*precision*recall/(precision+recall)
        f1[np.isnan(f1)]=0 # R is.na: F1 NA/NaN only; P/R and infinity untouched.
        values[rows]=np.column_stack((f1,precision,recall))
    return values

def fixed_scores(counts,ids,genes,labels,markers,reference_genes):
    """Preserve the fixed historical label eligibility; author scores for eligible rows."""
    counts=np.asarray(counts);ids=pd.Index(ids,name='cell_id');genes=pd.Index(genes);refs=pd.Index(reference_genes)
    if ids.has_duplicates or ids.isna().any() or genes.has_duplicates or refs.has_duplicates:raise ValueError('duplicate or missing input identifier')
    if counts.shape!=(len(ids),len(genes)):raise ValueError('counts/ID mapping mismatch')
    columns=genes.get_indexer(refs)
    if not len(refs) or np.any(columns<0):raise ValueError('missing reference gene inventory')
    if not np.isfinite(counts).all() or np.any(counts<0):raise ValueError('damaged count rows')
    frame=pd.DataFrame(index=ids);frame['anchor_label']=labels.reindex(ids).fillna('missing_historical_label')
    known_types=set(markers['positive'])&set(markers['negative'])
    frame['label_availability']=np.where(frame.anchor_label.eq('missing_historical_label'),'missing_historical_label',np.where(frame.anchor_label.isin(known_types),'known',frame.anchor_label.str.replace('__unavailable__:','',regex=False)))
    eligible=frame.label_availability.eq('known').to_numpy();x=counts[:,columns]
    for sign in ('positive','negative'):
        values=np.full((len(ids),3),np.nan)
        values[eligible]=marker_metrics(x[eligible],refs,frame.anchor_label.to_numpy()[eligible],markers[sign])
        for j,name in enumerate(('F1','Precision','Recall')):frame[sign+'_'+name]=values[:,j]
    frame['zero_reference_counts']=np.all(x==0,axis=1)
    return frame
