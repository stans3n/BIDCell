"""Method-native ROI summaries and unchanged official engineering evaluation.

Full-field legacy structural tables and counts are never reused. CellSPA runs
once per method on the union of complete eligible cells. Two profiles remain
separate. A missing B/T domain can leave marker metrics computed and the curve
explicitly unavailable, without fabricating a scalar or primary Purity F1.
"""
import json
from pathlib import Path
import subprocess
import h5py
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import tifffile


def run(out,method):
    from .pilot import MACHINE,TOOLS,save,provenance
    spec=json.loads((out/'spec.json').read_text());root=out/method
    reference=json.loads((TOOLS/'configs/reference.json').read_text())
    ref=pd.read_csv(MACHINE['reference'],index_col=0)
    mapping=dict(zip(ref.ct_idx,ref.cell_type));names=[mapping[k] for k in sorted(mapping)]
    with h5py.File(MACHINE['preannotation'],'r') as f:anno=dict(zip(f['ids'][:].astype(int),f['data'][:].astype(int)))
    matrices=[];annotations=[];cells=[];rows=[]
    for b in spec['blocks']:
        if b['split']!='dev':continue
        dest=root/b['id'];final=tifffile.imread(dest/'final.tif');pre=tifffile.imread(dest/'pre_connect.tif');nuc=tifffile.imread(dest/'nuclei.tif')
        mat=pd.read_csv(dest/'expr_mat.csv',index_col=0);matrices.append(mat)
        met=json.loads((dest/'metrics.json').read_text());rows.append(met)
        start=b['core_offset'];stop=start+b['core_size']
        for cid in b['eligible_ids']:
            mask=final==cid;pos=np.argwhere(mask)
            if len(pos) and ((pos.min(0)<start).any() or (pos.max(0)>=stop).any()):raise RuntimeError('eligible cell reaches outside core')
            coords=np.argwhere(nuc==cid);center=coords.mean(0)+[b['y'],b['x']]
            code=anno.get(cid,-1);ct=names[code] if 0<=code<len(names) else 'Unknown'
            annotations.append(dict(cell_id=cid,cell_type=ct,x_um=float(center[1]),y_um=float(center[0])))
            vals=mat.loc[mat.cell_id==cid].iloc[0,1:].to_numpy()
            cells.append(dict(cell_id=cid,block=b['id'],area=int(mask.sum()),nucleus_area=int((nuc==cid).sum()),nucleus_coverage=float((mask & (nuc==cid)).sum()/max(1,(nuc==cid).sum())),transcripts=int(vals.sum()),genes=int((vals>0).sum()),disconnected_pre=int(ndi.label(pre==cid,ndi.generate_binary_structure(2,2))[1]>1),disconnected_final=int(ndi.label(mask,ndi.generate_binary_structure(2,2))[1]>1)))
        if method!='E00':
            base=tifffile.imread(out/'E00'/b['id']/'pre_connect.tif')
            met['changed_pre_fraction']=float((base[start:stop,start:stop]!=pre[start:stop,start:stop]).mean());save(dest/'metrics.json',met)
    matrix=pd.concat(matrices,ignore_index=True);annotations=pd.DataFrame(annotations);cells=pd.DataFrame(cells)
    if matrix.cell_id.duplicated().any():raise RuntimeError('duplicate complete cells across dev blocks')
    matrix.to_csv(root/'eligible_expr_mat.csv');annotations.to_csv(root/'annotation_nuclear.csv',index=False);cells.to_csv(root/'cell_metrics.csv',index=False)
    pooled=dict(eligible_transcripts=sum(r['eligible_transcripts'] for r in rows),assigned_transcripts=sum(r['assigned_transcripts'] for r in rows),eligible_cells=len(cells),median_cell_area=float(cells.area.median()),foreground=float(np.mean([r['foreground'] for r in rows])),disconnected_final=float(cells.disconnected_final.mean()),disconnected_pre=float(cells.disconnected_pre.mean()))
    pooled['capture']=pooled['assigned_transcripts']/pooled['eligible_transcripts']
    if method!='E00':
        base=json.loads((out/'E00'/'pooled_metrics.json').read_text())
        pooled.update(capture_delta_pp=100*(pooled['capture']-base['capture']),foreground_relative_delta=pooled['foreground']/base['foreground']-1,median_area_relative_delta=pooled['median_cell_area']/base['median_cell_area']-1,disconnected_delta_pp=100*(pooled['disconnected_final']-base['disconnected_final']))
        pooled['screen_pass']=pooled['capture_delta_pp']>=-.5 and abs(pooled['foreground_relative_delta'])<=.05 and abs(pooled['median_area_relative_delta'])<=.05 and pooled['disconnected_delta_pp']<=.1
    save(root/'pooled_metrics.json',pooled)
    cmd=[reference['rscript'],str(Path(__file__).with_name('cellspa_roi.R')),reference['official_sources'],str(root/'eligible_expr_mat.csv'),MACHINE['reference'],MACHINE['positive_markers'],MACHINE['negative_markers'],str(root/'annotation_nuclear.csv'),str(root/'cellspa')]
    with (root/'cellspa.log').open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    profiles={}
    for profile in reference['annotation_profiles']:
        path=root/'cellspa'/profile/'official_marker_summary.csv'
        if path.exists():
            df=pd.read_csv(path);profiles[profile]=json.loads(df.to_json(orient='records'))
        else:profiles[profile]=dict(status='not_computed',reason='official execution failed; see cellspa.log')
    save(root/'evaluation.json',dict(status='succeeded' if r.returncode==0 else 'partial_official_execution',returncode=r.returncode,official_primary_purity_f1=dict(status='missing_definition',value=None),annotation_profiles=profiles,pooled=pooled,provenance=provenance(),command=cmd,eligibility='fixed complete cells only, all zeros retained; global nuclear centroids; both profiles raw pooled training reference; historical exposure disclosed'))
    print('EVALUATION',method,'official_returncode',r.returncode,pooled,flush=True)
    if r.returncode:raise RuntimeError('official engineering evaluator failed; see preserved logs')
