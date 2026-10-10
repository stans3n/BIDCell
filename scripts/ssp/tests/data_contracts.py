"""Synthetic boundary/zero-count fixtures plus actual dataset alignment."""
import argparse,json,tempfile,re
from pathlib import Path
from types import SimpleNamespace
import h5py,numpy as np,torch,tifffile
from scripts.crgf.common import atomic_json,identity,source_identity
from scripts.ssp.prototype import build,PrototypeCache,INPUT_KEYS
from scripts.ssp.config import read
from scripts.ssp.data import dataset
from scripts.ssp.method import teacher
from scripts.cos.ownership import compact_sample
from bidcell.model.dataio.candidate_geometry import transform

def synthetic(out):
    root=out/'synthetic';root.mkdir(parents=True,exist_ok=True);(root/'configs').mkdir(exist_ok=True);data=root/'data';data.mkdir(exist_ok=True);(data/'maps').mkdir(exist_ok=True)
    nu=np.zeros((7,7),np.int32);nu[3,3:5]=1;nu[1,1]=2;tifffile.imwrite(data/'nuclei.tif',nu);(data/'genes').write_text('a\nb\n');(data/'allowed').write_text('synthetic training input only')
    inputs={k:identity(data/('nuclei.tif' if k=='nuclei_mask' else 'genes' if k=='genes' else 'allowed')) for k in INPUT_KEYS};inputs['gene_order']=['a','b'];atomic_json(root/'configs/inputs.json',inputs)
    atomic_json(root/'configs/views.json',dict(views=[dict(k=1,patch='0_0.hdf5'),dict(k=1,patch='0_2.hdf5')]))
    X=np.zeros((7,7,2),np.uint8);X[3,3,0]=2;X[3,4,1]=3;X[6,6,0]=5
    for x0,x1 in ((0,4),(4,7)):
        with h5py.File(data/'maps'/f'all_genes_0_7_{x0}_{x1}.hdf5','w') as f:f.create_dataset('data',data=X[:,x0:x1])
    cfg=SimpleNamespace(native=lambda:SimpleNamespace(files=SimpleNamespace(data_dir=str(data),dir_out_maps='maps')),prototype_version='synthetic',teacher_version='synthetic')
    build(root,cfg);c=PrototypeCache(root/'prepared/manifest.json',['a','b']);assert np.array_equal(c.arrays['counts'],[[2,3],[0,0]])
    assert np.array_equal(c.arrays['train_union_gene_counts'],X.sum((0,1)));assert np.array_equal(c.lookup([2])[0][0],c.arrays['mu'].astype(np.float32));assert c.lookup([2])[1][0]==0
    try:c.lookup([3])
    except ValueError:pass
    else:raise AssertionError('missing ID accepted')
    try:PrototypeCache(root/'prepared/manifest.json',['b','a'])
    except ValueError:pass
    else:raise AssertionError('gene permutation accepted')
    assert np.array_equal(c.lookup([2,1])[0],c.arrays['r'][[1,0]])
    return dict(overlapping_view_union_once=True,full_nucleus_cross_storage_boundary_once=True,zero_count_prior_confidence=True,missing_id_error=True,gene_permutation_error=True,keyed_ID_reorder=True)

def main(root):
    root=Path(root);out=root/'tests/data_contracts';out.mkdir(parents=True,exist_ok=True);result=synthetic(out);cfg=read(root/'S/configs/run.json');ds=dataset(cfg);v=next(v for v in json.loads((root/'configs/views.json').read_text())['views'] if v['k']>=2)
    with h5py.File(v['patch']) as f:raw=f['data'][:]
    for flip in (0,1):
        for rot in range(4):
            ds.view_transform=(flip,rot);s=compact_sample(ds[v['index']]);assert np.array_equal(s['local_expression'].numpy(),transform(raw,flip,rot).astype(np.float32))
            assert np.array_equal(s['prototype_r'].numpy(),ds.prototype_cache.lookup(s['cell_ids'])[0])
    # Real patch agrees with once-only global map even at storage boundary.
    y,x=map(int,re.findall(r'\d+',Path(v['patch']).name)[:2]);assembled=np.zeros_like(raw)
    native=cfg.native()
    for block in (Path(native.files.data_dir)/native.files.dir_out_maps).glob('all_genes_[0-9]*.hdf5'):
        y0,y1,x0,x1=map(int,re.findall(r'\d+',block.stem));ay,by=max(y,y0),min(y+48,y1);ax,bx=max(x,x0),min(x+48,x1)
        if ay<by and ax<bx:
            with h5py.File(block) as f:assembled[ay-y:by-y,ax-x:bx-x]=f['data'][ay-y0:by-y0,ax-x0:bx-x0]
    assert np.array_equal(assembled,raw)
    t=teacher(s)
    # Tensor-aligned equivariance, without asserting native SearchArea reconstruction equivariance.
    for flip in (0,1):
        for rot in range(4):
            q=dict(s)
            for key in ('p','q','s','geometry','n','x'):
                q[key]=torch.from_numpy(np.ascontiguousarray(transform(s[key].numpy().transpose(2,3,0,1),flip,rot).transpose(2,3,0,1)))
            q['nuclei']=transform(s['nuclei'],flip,rot);q['local_expression']=torch.from_numpy(transform(s['local_expression'].numpy(),flip,rot))
            u=teacher(q)
            for key in ('target','weight','pi'):
                expected=transform(t[key].numpy().transpose(1,2,0),flip,rot).transpose(2,0,1);assert np.allclose(u[key].numpy(),expected,rtol=2e-6,atol=1e-7),key
            assert np.array_equal(u['H'].numpy(),transform(t['H'].numpy(),flip,rot))
    # Public compaction reorders all candidate-aligned feature tensors; raw expression remains fixed.
    k=len(s['cell_ids']);order=np.arange(k)[::-1].copy();q=dict(s)
    for key in ('p','q','s','geometry','n','x','type_valid','candidate_valid','prototype_r','prototype_confidence','exclusive_positive'):q[key]=s[key][torch.tensor(order)]
    for key in ('cell_ids','full_areas'):q[key]=s[key][order]
    for key in ('type_reasons','type_indices'):q[key]=[s[key][i] for i in order]
    q=compact_sample(q)
    for key in ('prototype_r','prototype_confidence','exclusive_positive','local_expression','x'):assert torch.equal(q[key],s[key])
    padded=dict(s)
    for key in ('p','q','s','geometry','n','x','type_valid','candidate_valid','prototype_r','prototype_confidence','exclusive_positive'):
        padded[key]=torch.cat((s[key],torch.zeros_like(s[key][:1])),0)
    padded['cell_ids']=np.r_[s['cell_ids'],0];padded['full_areas']=np.r_[s['full_areas'],0]
    for key in ('type_reasons','type_indices'):padded[key]=s[key]+[None]
    padded=compact_sample(padded)
    assert torch.equal(padded['prototype_r'],s['prototype_r']) and torch.equal(teacher(padded)['target'],t['target'])
    empty=dict(s,nuclei=np.zeros_like(s['nuclei']),cell_ids=np.array([],np.int64))
    for key in ('p','q','s','geometry','n','x','type_valid','candidate_valid','prototype_r','prototype_confidence','exclusive_positive'):empty[key]=s[key][:0]
    et=teacher(empty);assert not et['H'].any() and torch.equal(et['bg'],torch.ones_like(et['bg']))
    result.update(padding_compaction=True,K0_background_one=True,actual_patch_global_map_equal=True,actual_all_8_augmentations_raw_expression_aligned=True,prototype_invariant_across_augmentations=True,tensor_teacher_flip_rotation_equivariance=True,compact_all_candidate_fields=True)
    atomic_json(root/'tests/data_contracts.json',dict(status='passed',checks=result,source=source_identity(['scripts/ssp/tests/data_contracts.py','scripts/ssp/data.py','scripts/ssp/prototype.py','scripts/ssp/method.py','scripts/cos/ownership.py'])))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();main(a.root)
