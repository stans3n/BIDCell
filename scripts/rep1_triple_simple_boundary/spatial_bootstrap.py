"""Exact pooled block statistics with bounded, recoverable 100-replicate batches."""
from pathlib import Path
import hashlib
import numpy as np
from .common import (atomic_json,atomic_json_gzip,read_json_gzip,atomic_npz,check_receipt,chunk_receipt,identity,
                     read,write_receipt,progress,fsync_dir)

def array_hash(value):
    a=np.ascontiguousarray(value)
    return hashlib.sha256(a.dtype.str.encode()+str(a.shape).encode()+a.tobytes()).hexdigest()

class GroupProjector:
    """One bounded sparse group membership index; no repeated np.isin scans."""
    def __init__(self,cells,groups):
        self.ids=cells.index.to_numpy(np.int64);self.names=list(groups);self.G=len(groups);self.cells=cells
        lengths=np.asarray([len(groups[k]) for k in self.names]);self.rows=np.repeat(np.arange(self.G,dtype=np.int32),lengths)
        values=np.concatenate([groups[k] for k in self.names]) if lengths.sum() else np.empty(0,np.int64)
        self.positions=np.searchsorted(self.ids,values)
        if np.any(self.positions>=len(self.ids)) or not np.array_equal(self.ids[self.positions],values):raise ValueError('frozen group contains unknown cell')
        self.codes={};self.count_cache={}
    def blocks(self,ids,values,side,B,denominator=None):
        ix=np.searchsorted(self.ids,ids)
        if np.any(ix>=len(self.ids)) or not np.array_equal(self.ids[ix],ids):raise ValueError('statistical population contains unknown cell')
        v=np.zeros(len(self.ids));d=np.zeros(len(self.ids));v[ix]=values;d[ix]=1 if denominator is None else denominator
        if side not in self.codes:
            bi=self.cells['block_'+str(side)].to_numpy(np.int64)
            if np.any(bi<0) or np.any(bi>=B):raise ValueError('frozen block index invalid')
            self.codes[side]=(self.rows.astype(np.int64)*B+bi[self.positions]).astype(np.int32)
        code=self.codes[side]
        s=np.bincount(code,weights=v[self.positions],minlength=self.G*B).reshape(self.G,B).T
        key=(side,array_hash(ix)) if denominator is None else None
        if key in self.count_cache:n=self.count_cache[key]
        else:
            n=np.bincount(code,weights=d[self.positions],minlength=self.G*B).reshape(self.G,B).T
            if key is not None:
                if len(self.count_cache)>=18:self.count_cache.clear()
                self.count_cache[key]=n
        return s,n

def block_intervals(s,n,W,names,minimum_units=20,observations=None,checkpoint=None,callback=None):
    s=np.asarray(s,float);n=np.asarray(n,float);W=np.asarray(W)
    if s.shape!=n.shape or s.ndim!=2 or W.shape[1]!=s.shape[0] or len(names)!=s.shape[1] or not np.isfinite(s).all() or not np.isfinite(n).all() or np.any(n<0):raise ValueError('invalid pooled block arrays')
    B,G=s.shape;R=len(W);path=dep=None
    if checkpoint is not None:
        path,base=checkpoint;path=Path(path);path.mkdir(parents=True,exist_ok=True)
        dep=dict(base,sums=array_hash(s),denominators=array_hash(n),weights=array_hash(W),names=names,
                 minimum_units=minimum_units,observations=None if observations is None else np.asarray(observations).tolist())
        if (path/'complete.json').exists():
            rec=check_receipt(path/'complete.json',dep)
            retire_batches(path,rec)
            return read_json_gzip(path/'intervals.json.gz')
    samples=np.full((R,G),np.nan);batches=[]
    for start in range(0,R,100):
        end=min(start+100,R);rp=path/f'batch_{start:04d}.json' if path is not None else None
        cd=dict(dep,start=start,stop=end) if dep is not None else None
        if rp is not None and rp.exists():
            check_receipt(rp,cd);part=np.load(rp.with_suffix('.npz'))['samples']
        else:
            dn=W[start:end]@n;num=W[start:end]@s;part=np.divide(num,dn,out=np.full(num.shape,np.nan),where=dn>0)
            if rp is not None:
                atomic_npz(rp.with_suffix('.npz'),samples=part);chunk_receipt(rp,cd,[rp.with_suffix('.npz')],replicate_start=start,replicate_stop=end)
        samples[start:end]=part
        if rp is not None:
            batches.append(dict(receipt=identity(rp),data=identity(rp.with_suffix('.npz'))));progress(callback,'spatial_bootstrap',end,R,receipt=str(rp))
    total=n.sum(0);summ=s.sum(0);nb=(n>0).sum(0);result=[]
    observed=total if observations is None else np.asarray(observations)
    for j,name in enumerate(names):
        point=float(summ[j]/total[j]) if total[j] else None;valid=np.isfinite(samples[:,j]);nv=int(valid.sum());ln=total[j]-n[:,j]
        leave=np.divide(summ[j]-s[:,j],ln,out=np.full(B,np.nan),where=ln>0)
        reason='NA_zero_denominator' if total[j]==0 else 'NA_sparse_units' if observed[j]<minimum_units else 'NA_insufficient_blocks' if nb[j]<10 else 'NA_insufficient_valid_replicates' if nv<1900 else ''
        ci=np.quantile(samples[valid,j],[.025,.975],method='linear').tolist() if not reason else [None,None]
        result.append(dict(stratum=name,point=point,n_units=float(total[j]),n_observations=float(observed[j]),contributing_blocks=int(nb[j]),universe_blocks=B,
            valid_replicates=nv,replicates=R,ci_low=ci[0],ci_high=ci[1],LOBO_min=float(np.nanmin(leave)) if np.isfinite(leave).any() else None,
            LOBO_max=float(np.nanmax(leave)) if np.isfinite(leave).any() else None,NA_reason=reason,status=reason or 'computed'))
    if path is not None:
        atomic_json_gzip(path/'intervals.json.gz',result);atomic_npz(path/'block_sufficient_statistics.npz',sums=s,denominators=n)
        atomic_json(path/'retired_batches.json',dict(policy='verified committed temporary batches retired only after exact interval and block statistics commit; failed/uncommitted temporaries are retained',batches=batches))
        rec=write_receipt(path/'complete.json',dep,{k:identity(path/k) for k in ('intervals.json.gz','block_sufficient_statistics.npz','retired_batches.json')},replicates=R,batches=len(batches),temporary_batches_retired=True)
        retire_batches(path,rec);progress(callback,'spatial_bootstrap_complete',R,R,receipt=str(path/'complete.json'))
    return result

def retire_batches(path,receipt):
    """Auditable retirement of good temporary batches; never delete corruption."""
    from .common import check_identity
    path=Path(path);items=read(path/'retired_batches.json')['batches']
    remaining=[obj for row in items for obj in (row['receipt'],row['data']) if Path(obj['path']).exists()]
    for obj in remaining:check_identity(obj)
    for obj in remaining:Path(obj['path']).unlink()
    fsync_dir(path)
