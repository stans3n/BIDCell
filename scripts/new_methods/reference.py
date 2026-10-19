"""Distinct, versioned atlas tolerance pools: R excludes singleton types; B includes."""
import numpy as np
import pandas as pd
from .common import identity


def js(p,q):
    m=(p+q)/2
    return float(.5*np.sum(p*np.log(p/m))+.5*np.sum(q*np.log(q/m)))


def build_reference(path, genes, arm):
    df=pd.read_csv(path,index_col=0)
    if df.columns.duplicated().any() or len(genes)!=len(set(genes)):raise ValueError('duplicate genes')
    if df.groupby('ct_idx').cell_type.nunique().max()!=1 or df.groupby('cell_type').ct_idx.nunique().max()!=1:raise ValueError('type ID conflict')
    names=df.sort_values('ct_idx').cell_type.drop_duplicates().tolist()
    common=[g for g in genes if g in df.columns]
    if len(common)<32:raise ValueError('fewer than 32 measured reference genes')
    profiles={};deviations={};rejected=[]
    for typ,group in df.groupby('cell_type',sort=False):
        atlas_profiles={}
        for atlas,rows in group.groupby('atlas',sort=True):
            vals=rows[common].to_numpy(float)
            valid=np.isfinite(vals).all(1)&(vals>=0).all(1)&(vals.sum(1)>0)
            if not valid.all():
                rejected.extend(rows.index[~valid].astype(str).tolist())
                if arm=='V1_BOUNDARY':raise ValueError('invalid boundary reference scores')
            vals=vals[valid]
            if len(vals):atlas_profiles[str(atlas)]=(.99*vals/vals.sum(1)[:,None]+.01/len(common)).mean(0)
        if atlas_profiles:
            theta=np.mean(list(atlas_profiles.values()),axis=0)
            profiles[typ]=(theta,atlas_profiles)
            deviations[typ]=[js(v,theta) for v in atlas_profiles.values()]
    pool=[d for typ,dev in deviations.items() if arm=='V1_BOUNDARY' or len(dev)>=2 for d in dev]
    q90=float(np.quantile(pool,.9,method='linear')) if pool else None
    theta=[];delta=[];valid=[];details={}
    for typ in names:
        if typ not in profiles:
            theta.append([1/len(common)]*len(common));delta.append(float(np.log(2)));valid.append(False);continue
        profile,atlases=profiles[typ];dev=deviations[typ]
        radius=.02+float(np.quantile(dev,.9,method='linear')) if len(dev)>=3 else (.02+q90 if q90 is not None else .12)
        theta.append(profile.tolist());delta.append(radius);valid.append(True)
        details[typ]=dict(atlases={k:v.tolist() for k,v in atlases.items()},deviations=dev,delta=radius,n_atlases=len(dev))
    return dict(version='RR-V1.1' if arm=='V1_REGION' else 'B-V1.3',arm=arm,source=identity(path),
                gene_indices=[genes.index(g) for g in common],gene_names=common,type_names=names,
                theta=theta,delta=delta,valid=valid,pooled_q90=q90,
                pool_rule='exclude_singletons' if arm=='V1_REGION' else 'include_singletons',details=details,rejected_rows=rejected)
