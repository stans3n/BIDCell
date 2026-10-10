"""Disjoint global input-map aggregation, fixed prior and exact keyed lookup."""
import json,re,time
from pathlib import Path
import h5py
import numpy as np
import tifffile
from scripts.crgf.common import atomic_json,atomic_npy,identity,sha256,verify_outputs,source_identity
INPUT_KEYS=('nuclei_mask','genes','preannotation','positive_markers','negative_markers','reference')

def build(root,cfg):
    executed_source=source_identity(['scripts/ssp/prototype.py'])
    root=Path(root);out=root/'prepared';out.mkdir(exist_ok=True)
    manifest=out/'manifest.json'
    if manifest.exists():
        m=json.loads(manifest.read_text());verify_outputs(m)
        for info in [*m['inputs'].values(),*m['global_blocks']]:
            if sha256(info['path'])!=info['sha256']:raise ValueError('prepared input drift')
        if m['views']['sha256']!=sha256(root/'configs/views.json'):raise ValueError('prepared views drift')
        return m
    inputs=json.loads((root/'configs/inputs.json').read_text());native=cfg.native()
    # Explicit allowlist excludes every evaluation/final-count field of historical manifest.
    provenance={k:identity(inputs[k]['path']) for k in INPUT_KEYS}
    for k in INPUT_KEYS:
        if provenance[k]['sha256']!=inputs[k]['sha256']:raise ValueError('input hash mismatch '+k)
    genes=Path(provenance['genes']['path']).read_text().splitlines()
    if genes!=inputs['gene_order'] or len(set(genes))!=len(genes):raise ValueError('gene order')
    nuclei=tifffile.imread(provenance['nuclei_mask']['path']);ids,areas=np.unique(nuclei[nuclei>0],return_counts=True)
    counts=np.zeros((len(ids),len(genes)),np.float64);prior=np.zeros(len(genes),np.float64)
    union=np.zeros(nuclei.shape,bool);views=json.loads((root/'configs/views.json').read_text())
    for v in views['views']:
        if v['k']:
            y,x=map(int,re.findall(r'\d+',Path(v['patch']).name)[:2]);union[y:y+48,x:x+48]=True
    blocks=sorted((Path(native.files.data_dir)/native.files.dir_out_maps).glob('all_genes_[0-9]*.hdf5'))
    if not blocks:raise ValueError('global expression blocks absent')
    coverage=np.zeros(nuclei.shape,np.uint8);sources=[];start=time.monotonic();nuclear_sum=0.
    for path in blocks:
        y0,y1,x0,x1=map(int,re.findall(r'\d+',path.stem));sources.append(identity(path))
        if coverage[y0:y1,x0:x1].any():raise ValueError('overlapping global input blocks')
        coverage[y0:y1,x0:x1]=1
        with h5py.File(path,'r') as f:
            if f['data'].shape[0]<y1-y0 or f['data'].shape[1]<x1-x0 or f['data'].shape[2]!=len(genes):raise ValueError('global block shape')
            for row in range(0,y1-y0,16):
                x=np.asarray(f['data'][row:min(row+16,y1-y0),:x1-x0],dtype=np.float64);nu=nuclei[y0+row:min(y0+row+16,y1),x0:x1]
                if not np.isfinite(x).all() or (x<0).any() or not np.equal(x,np.floor(x)).all():raise ValueError('noninteger input counts')
                prior+=x[union[y0+row:min(y0+row+16,y1),x0:x1]].sum(0,dtype=np.float64)
                mask=nu>0;vals=x[mask];ix=np.searchsorted(ids,nu[mask]);np.add.at(counts,ix,vals)
                nuclear_sum+=vals.sum(dtype=np.float64)
        atomic_json(out/'progress.json',dict(blocks=len(sources),total_blocks=len(blocks),elapsed=time.monotonic()-start))
    if not coverage.all():raise ValueError('input map coverage incomplete')
    if counts.sum()!=nuclear_sum:raise AssertionError('whole nucleus integer conservation')
    mu=(prior+1)/(prior.sum()+len(genes));N=counts.sum(1);U=(counts>0).sum(1)
    r=((counts+20*mu)/(N[:,None]+20)).astype(np.float32);confidence=(N/(N+20)*np.minimum(U/8,1)).astype(np.float32)
    projection=(np.random.RandomState(20261011).choice([-1.,1.],size=(len(genes),16))/4).astype(np.float32)
    arrays=dict(ids=ids.astype(np.int64),areas=areas.astype(np.int64),counts=counts,N=N,U=U,confidence=confidence,r=r,mu=mu,projection=projection,train_union_gene_counts=prior)
    for name,a in arrays.items():atomic_npy(out/(name+'.npy'),a)
    quant=lambda a:np.quantile(a,[0,.1,.25,.5,.75,.9,1]).tolist()
    m=dict(execution_source=executed_source,version=cfg.prototype_version,teacher_version=cfg.teacher_version,status='succeeded',inputs=provenance,views=identity(root/'configs/views.json'),gene_order=genes,global_blocks=sources,
        rules=dict(aggregation='float64 once over disjoint global expression blocks; raster ID only',prior='unique pixel union of nonempty formal views',smoothing=20,projection_seed=20261011,projection_dimension=16),
        nuclear_count_conservation=nuclear_sum,train_union_pixels=int(union.sum()),nucleus_count=len(ids),zero_count_fraction=float((N==0).mean()),quantiles=dict(N=quant(N),U=quant(U),confidence=quant(confidence)),
        outputs={name:identity(out/(name+'.npy')) for name in arrays},wall_seconds=time.monotonic()-start)
    atomic_json(manifest,m);return m

class PrototypeCache:
    def __init__(self,path,genes):
        self.path=Path(path);self.manifest=json.loads(self.path.read_text());verify_outputs(self.manifest)
        if list(genes)!=self.manifest['gene_order']:raise ValueError('prototype gene order mismatch')
        self.arrays={name:np.load(info['path'],mmap_mode='r') for name,info in self.manifest['outputs'].items()}
        ids=self.arrays['ids']
        if np.any(np.diff(ids)<=0):raise ValueError('invalid cache nuclear IDs')
        if self.arrays['r'].shape!=(len(ids),len(genes)) or self.arrays['confidence'].shape!=(len(ids),):raise ValueError('prototype dimensions')
        if (self.arrays['r']<0).any() or (self.arrays['confidence']<0).any() or (self.arrays['confidence']>1).any():raise ValueError('prototype range')
        for name in ('r','confidence','projection'):
            if not np.isfinite(self.arrays[name]).all():raise ValueError('nonfinite prototype cache')
        if not np.allclose(self.arrays['r'].sum(1),1,rtol=2e-6):raise ValueError('prototype normalization')
    def lookup(self,ids):
        ids=np.asarray(ids);keys=self.arrays['ids'];ix=np.searchsorted(keys,ids)
        if (ix>=len(keys)).any() or not np.array_equal(keys[ix],ids):raise ValueError('missing true nuclear ID in prototype cache')
        return self.arrays['r'][ix].copy(),self.arrays['confidence'][ix].copy()
