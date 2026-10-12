"""Thin wrapper around audited remote QC; formulas retain explicit raster labels."""
import importlib.util
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
from skimage.measure import regionprops_table
from .common import atomic_json, identity, sha256

LEGACY=Path(__file__).with_name('legacy_benchmark.py')
def legacy():
    spec=importlib.util.spec_from_file_location('remote_legacy_benchmark',LEGACY)
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod);return mod

def evaluate(c,matrixdir,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);old=legacy();log=logging.getLogger('qc')
    deps={k:sha256(c[k]) for k in ['final_mask','nuclei_mask','pre_mask','reference','preannotation']}
    deps.update(pixel_um=c['pixel_um'],skimage=__import__('skimage').__version__,matrix=sha256(Path(matrixdir)/'complete.json'),source=sha256(__file__),legacy=sha256(LEGACY))
    receipt=out/'complete.json'
    if receipt.exists():
        saved=__import__('json').loads(receipt.read_text())
        if saved['dependencies']!=deps:raise ValueError('structural cache dependency mismatch')
        for name,h in saved['outputs'].items():
            if sha256(out/name)!=h:raise ValueError('structural output modified')
        return pd.read_csv(out/'cells_distribution.csv'),__import__('json').loads((out/'structure.json').read_text())
    mask=tifffile.imread(c['final_mask']);nuc=tifffile.imread(c['nuclei_mask']);pre=tifffile.imread(c['pre_mask'])
    counts=np.load(Path(matrixdir)/'counts.npy',mmap_mode='r');ids=pd.read_csv(Path(matrixdir)/'cell_mapping.csv').mask_label.to_numpy()
    u,areas=np.unique(mask,return_counts=True);positive=u>0
    if not np.array_equal(ids,u[positive]):raise ValueError('mask/matrix mapping mismatch')
    table=pd.DataFrame(dict(cell_id=ids,transcripts_per_cell=np.asarray(counts.sum(axis=1)),genes_per_cell=np.asarray((counts>0).sum(axis=1)),area_um2=areas[positive]*c['pixel_um']**2))
    summary,nt,ct=old.nucleus_cell_structure(mask,nuc,log)
    ns=pd.DataFrame(regionprops_table(nuc,properties=['label','area','centroid','axis_major_length','axis_minor_length','eccentricity']))
    ns=ns.rename(columns={'label':'cell_id','area':'nucleus_area_px','centroid-0':'nucleus_y','centroid-1':'nucleus_x','axis_major_length':'nucleus_major','axis_minor_length':'nucleus_minor','eccentricity':'nucleus_eccentricity'})
    table=table.merge(ns,on='cell_id',how='left',validate='one_to_one')
    table['cell_nucleus_area_ratio']=table.area_um2/(table.nucleus_area_px*c['pixel_um']**2)
    dis,components=old.disconnected_cells(mask,log);pdis,pcomponents=old.disconnected_cells(pre,log)
    _,boundary=old.boundary_masks(mask)
    summary.update(dis);summary.update(foreground_fraction=float((mask>0).mean()),cell_cell_boundary_pixel_fraction=float(boundary.mean()),pre_post_changed_pixels=int(np.count_nonzero(pre!=mask)),pre_post_changed_fraction=float((pre!=mask).mean()),cell_nucleus_area_ratio_mean=float(table.cell_nucleus_area_ratio.mean()))
    atomic_json(out/'structure.json',summary);atomic_json(out/'preconnect_structure.json',pdis)
    table.to_csv(out/'cells_distribution.csv',index=False);nt.to_csv(out/'nuclei_distribution.csv',index=False);ct.to_csv(out/'cell_nuclei_associations.csv',index=False);components.to_csv(out/'components_8.csv',index=False);pcomponents.to_csv(out/'pre_components_8.csv',index=False)
    table[['transcripts_per_cell','genes_per_cell','area_um2','cell_nucleus_area_ratio']].describe(percentiles=[.05,.25,.5,.75,.95,.99]).to_csv(out/'distribution_summary.csv')
    genes=Path(c['genes']).read_text().splitlines();pd.DataFrame(dict(gene=genes,fraction_expressing=(counts>0).mean(axis=0))).to_csv(out/'gene_detection.csv',index=False)
    _,types=old.load_cell_types(Path(c['preannotation']),Path(c['reference']),ids,log)
    pd.DataFrame(dict(cell_id=ids,cell_type=types,x_um=table.nucleus_x*c['pixel_um'],y_um=table.nucleus_y*c['pixel_um'])).to_csv(out/'annotation_nuclear.csv',index=False)
    shape=old.compute_fast_shape_metrics(mask,c['pixel_um'],log).sort_values('cell_id')
    if not np.array_equal(shape.cell_id,ids): raise ValueError('own-mask shape ID mismatch')
    shape=shape.drop(columns=[k for k in ['circularity_raster','convexity_area_proxy'] if k in shape])
    # Their old values duplicate compactness and solidity; don't relabel them as official circularity/convexity.
    shape.to_csv(out/'shape_raster_distribution.csv',index=False)
    atomic_json(out/'implementation_sources.json',dict(legacy=identity(LEGACY),own_mask=identity(c['final_mask']),pixel_um=c['pixel_um'],shape_source='own final mask regionprops; polygon metrics not claimed'))
    atomic_json(receipt,dict(status='succeeded',dependencies=deps,outputs={p.name:sha256(p) for p in out.iterdir() if p.is_file()}))
    return table,summary
