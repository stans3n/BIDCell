"""Frozen A_CORE statistics. No masks, expression labels, or evaluation teachers."""
from __future__ import annotations
import hashlib
import math
import numpy as np

ALPHA = 20.0
BETA = 1.0
EPSILON = 1e-8
CORE_EDT = 2.0
MIN_FOLD = 10
SHUFFLE_SEEDS = tuple(range(2026101500, 2026101520))
BOOTSTRAP_SEED = 20261015
BOOTSTRAP_REPLICATES = 2000
MARGIN = math.log(2.0)
PARAMETERS = dict(alpha=ALPHA,beta=BETA,epsilon=EPSILON,core_edt_native=CORE_EDT,
    minimum_per_fold=MIN_FOLD,shuffle_seeds=list(SHUFFLE_SEEDS),
    bootstrap_seed=BOOTSTRAP_SEED,bootstrap_replicates=BOOTSTRAP_REPLICATES,
    bootstrap_scales_um=[500,1000,2000],bootstrap_origin_um=[0,0],margin=MARGIN,
    a1_minimum_cells=1000,a1_minimum_blocks=10,a1_minimum_valid_bootstraps=1900,
    a1_minimum_gain=.01)


def molecule_fold(ids):
    """Numeric transcript IDs have one canonical decimal spelling; never hash()."""
    return np.fromiter((hashlib.sha256(('20261015|'+str(int(i))).encode('ascii')).digest()[0]&1
                        for i in ids),dtype=np.uint8,count=len(ids))


def reference_arrays(reference, genes):
    genes=list(genes); names=list(reference['gene_names']); indices=np.asarray(reference['gene_indices'],int)
    if len(set(genes))!=len(genes) or len(set(names))!=len(names):raise ValueError('duplicate reference/spatial genes')
    if len(names)!=len(indices) or np.any(indices<0) or np.any(indices>=len(genes)) or [genes[i] for i in indices]!=names:
        raise ValueError('training reference gene name/order mismatch')
    theta=np.asarray(reference['theta'],float); valid=np.asarray(reference['valid'],bool)
    if theta.shape!=(len(reference['type_names']),len(names)) or valid.shape!=(len(theta),):raise ValueError('reference shape')
    if not len(names) or not valid.any():raise ValueError('empty valid training reference')
    if not np.isfinite(theta[valid]).all() or np.any(theta[valid]<0) or np.any(theta[valid].sum(1)<=0):raise ValueError('invalid legal theta')
    # Invalid rows stay unusable. Never map Unknown (-1) to the last row.
    safe=np.full_like(theta,1/len(names));safe[valid]=np.maximum(theta[valid],EPSILON)
    safe/=safe.sum(1,keepdims=True)
    return indices,safe,valid


def legal_types(types, valid):
    types=np.asarray(types,int);good=(types>=0)&(types<len(valid))
    result=np.zeros(len(types),bool);result[good]=valid[types[good]]
    return result


def profile(counts, theta):
    counts=np.asarray(counts);theta=np.asarray(theta,float)
    if np.any(counts<0) or not np.isfinite(counts).all():raise ValueError('illegal Core counts')
    if counts.shape!=theta.shape or not np.allclose(theta.sum(-1),1,atol=1e-12):raise ValueError('profile shape/normalization')
    q=(counts+ALPHA*theta)/(counts.sum(-1,keepdims=True)+ALPHA)
    if np.any(q<=0) or not np.allclose(q.sum(-1),1,atol=1e-12):raise ValueError('q normalization')
    return q


def matched_donors(counts,types,eligible,seed):
    """Uniform rejection-sampled derangement within type and floor(log2 n)."""
    n=np.asarray(counts).sum(1,dtype=np.int64);types=np.asarray(types,int)
    eligible=np.asarray(eligible,bool)&(n>0)&(types>=0)
    donor=np.full(len(n),-1,np.int32);bins=np.full(len(n),-1,np.int16)
    bins[n>0]=np.floor(np.log2(n[n>0])).astype(np.int16)
    rows=np.flatnonzero(eligible);rng=np.random.Generator(np.random.PCG64(seed))
    if len(rows):
        keys=np.stack((types[rows],bins[rows]),1)
        for key in np.unique(keys,axis=0):
            group=rows[np.all(keys==key,axis=1)]
            if len(group)<2:continue
            while True:
                perm=rng.permutation(group)
                if np.all(perm!=group):break
            donor[group]=perm
    good=donor>=0
    if np.any(donor[good]==np.flatnonzero(good)) or np.any(types[donor[good]]!=types[good]) or np.any(bins[donor[good]]!=bins[good]):raise ValueError('donor contract')
    return donor,bins


def cross_scores(counts,types,theta,valid):
    """Two folds combined by held-out n within cell; cells remain independent units."""
    counts=np.asarray(counts);types=np.asarray(types,int)
    if counts.ndim!=3 or counts.shape[0]!=2:raise ValueError('expected F0/F1 counts')
    n=counts.sum(2,dtype=np.int64);good=legal_types(types,valid)&np.all(n>=MIN_FOLD,axis=0)
    safe_types=np.where(legal_types(types,valid),types,0);prior=theta[safe_types]
    total=n.sum(0);den=np.maximum(total,1)
    obs=np.zeros(len(types));typ=np.zeros(len(types));shuffle=np.zeros((20,len(types)))
    donors=np.full((20,2,len(types)),-1,np.int32);bins=np.full((2,len(types)),-1,np.int16)
    for fold in (0,1):
        held=counts[1-fold];q=profile(counts[fold],prior)
        obs+=(held*np.log(q)).sum(1)/den
        typ+=(held*np.log(prior)).sum(1)/den
        for s,seed in enumerate(SHUFFLE_SEEDS):
            d,b=matched_donors(counts[fold],types,good,seed);donors[s,fold]=d;bins[fold]=b
            rr=np.flatnonzero(d>=0)
            # Chunking avoids 20 materialized N x G profile arrays.
            for start in range(0,len(rr),4096):
                ix=rr[start:start+4096];qs=profile(counts[fold,d[ix]],prior[ix])
                shuffle[s,ix]+=(held[ix]*np.log(qs)).sum(1)/den[ix]
    paired=good&np.all(donors>=0,axis=(0,1))
    obs[~good]=np.nan;typ[~good]=np.nan;shuffle[:,~paired]=np.nan
    p0=profile(counts[0],prior);p1=profile(counts[1],prior)
    cosine=np.sum(p0*p1,1)/np.sqrt(np.sum(p0*p0,1)*np.sum(p1*p1,1))
    e0=counts[0]/np.maximum(n[0,:,None],1);e1=counts[1]/np.maximum(n[1,:,None],1)
    residual0=e0-prior;residual1=e1-prior
    norms=np.sqrt(np.sum(residual0**2,1)*np.sum(residual1**2,1))
    residual_cos=np.divide(np.sum(residual0*residual1,1),norms,out=np.full(len(types),np.nan),where=norms>0)
    # Additional empirical within-type centering reports stability beyond a type mean.
    for t in np.unique(types[good]):
        use=good&(types==t);e0[use]-=e0[use].mean(0);e1[use]-=e1[use].mean(0)
    norms=np.sqrt(np.sum(e0**2,1)*np.sum(e1**2,1))
    centered=np.divide(np.sum(e0*e1,1),norms,out=np.full(len(types),np.nan),where=norms>0)
    return dict(eligible=good,paired=paired,core=obs,type=typ,shuffle=shuffle,
                donors=donors,bins=bins,fold_n=n,total_n=total,
                split_cosine=cosine,type_residual_cosine=residual_cos,within_type_centered_cosine=centered)


def paired_bootstrap(values,centroids,side,notify=None):
    """Identical resampled spatial-block multiplicities for all three arms."""
    values=np.asarray(values,float);xy=np.asarray(centroids,float)
    if values.shape!=(len(xy),3) or not np.isfinite(values).all() or not np.isfinite(xy).all():raise ValueError('bootstrap paired population')
    blocks,idx=np.unique(np.floor(xy/side).astype(np.int64),axis=0,return_inverse=True)
    n=np.bincount(idx,minlength=len(blocks));sums=np.zeros((len(blocks),3));np.add.at(sums,idx,values)
    rng=np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED));samples=[]
    for iteration in range(BOOTSTRAP_REPLICATES):
        if notify is not None and iteration%100==0:notify(iteration,BOOTSTRAP_REPLICATES)
        if not len(blocks):continue
        sampled=rng.integers(0,len(blocks),size=len(blocks));weights=np.bincount(sampled,minlength=len(blocks))
        denominator=int(weights@n)
        if denominator:
            means=weights@sums/denominator
            samples.append([means[0]-means[1],means[0]-means[2]])
    samples=np.asarray(samples).reshape(-1,2);mean=values.mean(0) if len(values) else np.full(3,np.nan)
    result=dict(side_um=side,origin_um=[0,0],n_cells=len(values),n_blocks=len(blocks),valid_replicates=len(samples),
                requested_replicates=BOOTSTRAP_REPLICATES,seed=BOOTSTRAP_SEED,
                block_xy=blocks.tolist(),block_n=n.tolist(),block_sums=sums.tolist(),
                paired_weights=True,conditional_on='fixed molecule split and 20 prespecified donor permutations')
    for k,col in [('core_minus_type',0),('core_minus_shuffle',1)]:
        result[k]=dict(mean=float(mean[0]-mean[col+1]) if len(values) else None,
                       ci95=np.quantile(samples[:,col],[.025,.975]).tolist() if len(samples) else [None,None])
    return result,samples


def decide(main):
    a,b=main['core_minus_type'],main['core_minus_shuffle']
    estimated=all(v is not None and np.isfinite(v) for v in [a['mean'],b['mean'],*a['ci95'],*b['ci95']])
    coverage=main['n_cells']>=1000 and main['n_blocks']>=10 and main['valid_replicates']>=1900 and estimated
    if coverage and a['mean']>=.01 and a['ci95'][0]>0 and b['ci95'][0]>0:status='GO_A2'
    elif coverage and (a['ci95'][1]<.01 or b['ci95'][1]<=0):status='NO_USEFUL_INCREMENT'
    else:status='INCONCLUSIVE'
    return dict(execution_status='COMPLETED',science_status=status,coverage_ok=coverage,
                gate=PARAMETERS,primary=main,owner_accuracy_claim=False)
