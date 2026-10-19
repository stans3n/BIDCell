"""Prespecified pooled spatial statistics and decision rules."""
import numpy as np

EPS=1e-8


def quartiles(values):
    a=np.asarray(values,float)
    if not len(a) or not np.isfinite(a).all():raise ValueError('quartile input population absent/nonfinite')
    edges=np.unique(np.quantile(a,[.25,.5,.75],method='linear'))
    return edges,np.searchsorted(edges,a,side='left')


def finite_mean(a):
    a=np.asarray(a,float);ok=np.isfinite(a)
    return dict(mean=float(a[ok].mean()) if ok.any() else None,n_finite=int(ok.sum()),
                n_population=len(a),n_unavailable=int((~ok).sum()),
                status='computed' if ok.any() else 'not_estimable',reason=None if ok.any() else 'no finite score')


def pooled_deltas(sums_a,counts_a,sums_b,counts_b,draws):
    # Repeated block indices apply identical weights to both endpoint sums/counts.
    sa=np.asarray(sums_a)[draws].sum(1);na=np.asarray(counts_a)[draws].sum(1)
    sb=np.asarray(sums_b)[draws].sum(1);nb=np.asarray(counts_b)[draws].sum(1)
    result=np.full(len(draws),np.nan);ok=(na>0)&(nb>0)
    result[ok]=sa[ok]/na[ok]-sb[ok]/nb[ok]
    return result


def spatial(values_a,values_b,block_indices,nblocks,seed=20261013,repetitions=2000):
    result={};draws=np.random.Generator(np.random.PCG64(seed)).integers(0,nblocks,size=(repetitions,nblocks))
    for metric in ('positive_F1','negative_F1'):
        a=np.asarray(values_a[metric],float);b=np.asarray(values_b[metric],float)
        fa=np.isfinite(a);fb=np.isfinite(b)
        sa=np.bincount(block_indices,weights=np.where(fa,a,0),minlength=nblocks)
        sb=np.bincount(block_indices,weights=np.where(fb,b,0),minlength=nblocks)
        na=np.bincount(block_indices,weights=fa,minlength=nblocks).astype(int)
        nb=np.bincount(block_indices,weights=fb,minlength=nblocks).astype(int)
        boot=pooled_deltas(sa,na,sb,nb,draws);valid=np.isfinite(boot)
        loa=na.sum()-na;lob=nb.sum()-nb;loo=np.full(nblocks,np.nan);ok=(loa>0)&(lob>0)
        loo[ok]=(sa.sum()-sa[ok])/loa[ok]-(sb.sum()-sb[ok])/lob[ok]
        ci=np.quantile(boot[valid],[.025,.975],method='linear').tolist() if valid.any() else [None,None]
        result[metric]=dict(block_sums_candidate=sa,block_sums_control=sb,block_n_candidate=na,block_n_control=nb,
                            bootstrap=boot,leave_one_out=loo,ci=ci,n_valid=int(valid.sum()),n_invalid=int((~valid).sum()))
    return result,draws


def spatial_flags(result,delta_p,delta_n,nblocks):
    flags=[]
    if nblocks<2 or any(result[m]['n_valid']<1900 for m in result):flags.append('spatial_insufficient')
    p=result['positive_F1'];n=result['negative_F1']
    if ((delta_p>=-EPS and np.any(p['leave_one_out']<-EPS)) or
        (delta_n<=EPS and np.any(n['leave_one_out']>EPS))):flags.append('spatial_sensitive')
    favorable=((delta_p>EPS and p['ci'][0] is not None and p['ci'][0]>0) or
               (delta_n<-EPS and n['ci'][1] is not None and n['ci'][1]<0))
    if not favorable:flags.append('spatial_interval_ambiguous')
    return flags


def guard_values(candidate,control):
    result={}
    for name,threshold,kind in [('capture',-.005,'difference'),('area_mean',.05,'relative_abs'),
        ('area_median',.10,'relative_abs'),('area_p95',.10,'relative_abs'),('qc_retention',-.005,'difference'),
        ('unassigned',.01,'difference'),('disconnected8',0,'difference')]:
        a,b=candidate.get(name),control.get(name)
        reason=None
        if a is None or b is None:reason='empty population or undefined denominator'
        elif kind=='relative_abs' and b==0:reason='zero control denominator'
        value=None if reason else (abs(a/b-1) if kind=='relative_abs' else a-b)
        passed=None if reason else (value>=threshold if threshold<0 else value<=threshold)
        result[name]=dict(candidate=a,control=b,change=value,threshold=threshold,passed=passed,
                          status='not_estimable' if reason else 'computed',reason=reason,kind=kind)
    return result


def decide(delta_p,delta_n,guards,flags,correct=True,complete=True):
    missing=delta_p is None or delta_n is None
    direction=False if missing else delta_p>=-EPS and delta_n<=EPS and (delta_p>EPS or delta_n<-EPS)
    guards_pass=all(v['passed'] is True for v in guards.values())
    magnitude=False if missing else direction and guards_pass and (delta_p>=.001 or delta_n<=-.0001)
    if not correct:status='implementation_invalid'
    elif not complete or missing or any(v['passed'] is None for v in guards.values()):status='comparison_incomplete'
    elif not guards_pass or delta_p<-EPS or delta_n>EPS:status='tradeoff_or_negative'
    elif abs(delta_p)<=EPS and abs(delta_n)<=EPS:status='no_expression_gain'
    elif flags:status='exploratory_signal'
    elif magnitude:status='priority_signal'
    else:status='small_signal'
    return dict(delta_P=delta_p,delta_N=delta_n,direction_pass=direction,guards_pass=guards_pass,
                reliability_flags=sorted(set(flags)),magnitude_priority=magnitude,status=status,
                capture_delta_pp=None if guards['capture']['change'] is None else 100*guards['capture']['change'],
                all_guard_values_and_denominators=guards)
