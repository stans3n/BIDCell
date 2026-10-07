"""Own-mask engineering metrics; independent components preserve explicit NA."""
import json
import subprocess
import time
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from .common import atomic_json,identity,sha256,Monitor,verify_outputs
from . import structural


def validate_matrix(machine,matrix):
    matrix=Path(matrix);m=json.loads((matrix/'complete.json').read_text())
    if m['status']!='succeeded' or m.get('validation_kind')!='independent_full_native_resize_lookup_and_conservation':
        raise ValueError('matrix lacks production acceptance')
    verify_outputs(m)
    for key,path in [('mask',machine['final_mask']),('transcripts',machine['transcripts']),('genes',machine['genes'])]:
        if sha256(path)!=m['dependencies'][key]['sha256']: raise ValueError('stale own-mask matrix '+key)
    if sha256(matrix/'validation.json')!=m['validation']['sha256']:raise ValueError('validation changed')
    return m


def changes(c,out):
    own=tifffile.imread(c['final_mask']);base=tifffile.imread(c['baseline_final_mask'])
    if own.shape!=base.shape: raise ValueError('B0/current shape mismatch')
    changed=own!=base;ids=np.union1d(np.unique(base),np.unique(own));ids=ids[ids>0]
    maxid=int(ids.max(initial=0));aa=np.bincount(own.ravel(),minlength=maxid+1);bb=np.bincount(base.ravel(),minlength=maxid+1)
    same=np.where(own==base,own,0);intersection=np.bincount(same.ravel(),minlength=maxid+1)[ids]
    union=aa[ids]+bb[ids]-intersection
    pd.DataFrame(dict(cell_id=ids,intersection=intersection,union=union,iou=np.divide(intersection,union,out=np.full(len(ids),np.nan),where=union>0))).to_csv(out/'same_id_iou_vs_B0.csv',index=False)
    boundary=np.zeros(base.shape,bool)
    edge=(base[1:,:]!=base[:-1,:])&((base[1:,:]>0)|(base[:-1,:]>0));boundary[1:,:]|=edge;boundary[:-1,:]|=edge
    edge=(base[:,1:]!=base[:,:-1])&((base[:,1:]>0)|(base[:,:-1]>0));boundary[:,1:]|=edge;boundary[:,:-1]|=edge
    distance=ndi.distance_transform_edt(~boundary,sampling=c['pixel_um'])
    bands=[]
    for width in (1,2,3,5):
        band=distance<=width;n=int(band.sum());hit=int((band&changed).sum())
        bands.append(dict(width_um=width,band_pixels=n,changed_pixels=hit,within_band_change_fraction=hit/n if n else None,share_of_all_changes=hit/int(changed.sum()) if changed.any() else None))
    result=dict(comparison='B0 own ID change diagnostic, not accuracy',B0=identity(c['baseline_final_mask']),own=identity(c['final_mask']),changed_pixels=int(changed.sum()),changed_fraction=float(changed.mean()),bands=bands)
    atomic_json(out/'change_vs_B0.json',result);return result


def evaluate(machine,reference,protocol,matrix,out,profile='EP-CORE'):
    c=json.loads(Path(machine).read_text());r=json.loads(Path(reference).read_text());p=json.loads(Path(protocol).read_text());out=Path(out);out.mkdir(parents=True,exist_ok=True)
    m=validate_matrix(c,matrix);start=time.monotonic();result=[];failures={}
    def metric(i,name,value,status='computed_engineering',definition=''):
        result.append(dict(metric_id=str(i),metric_name=name,value=value,status=status,definition=definition))
    counts=np.load(Path(matrix)/'counts.npy',mmap_mode='r');mask=tifffile.imread(c['final_mask']);ids,areas=np.unique(mask,return_counts=True);areas=areas[ids>0]
    metric(1,'Number of segmented cells',len(areas));metric(2,'Assigned transcript fraction',m['totals']['assigned']/m['totals']['eligible'] if m['totals']['eligible'] else None)
    metric(3,'Transcripts per cell',float(counts.sum(1).mean()) if len(areas) else None)
    metric(4,'Genes per cell',float((counts>0).sum(1).mean()) if len(areas) else None)
    metric(5,'Cell area',float(areas.mean()*c['pixel_um']**2) if len(areas) else None)
    with Monitor(out/'resources_python.json'):
        try:
            table,s=structural.evaluate(c,matrix,out/'structure')
            for i,name,key in [(12,'Nucleus containment rate','exact_same_id_nucleus_containment_fraction'),(14,'Zero-nucleus fraction','zero_nucleus_cell_fraction'),(15,'Multi-nucleus fraction','multi_nucleus_cell_fraction'),(16,'Split-nucleus fraction','split_nucleus_fraction'),(17,'Disconnected-cell fraction','disconnected_cell_fraction_8'),(18,'Cell/nucleus area ratio','cell_nucleus_area_ratio_mean'),(20,'Foreground fraction','foreground_fraction'),(21,'Cell-cell boundary fraction','cell_cell_boundary_pixel_fraction')]:metric(i,name,s[key])
            metric('engineering_pre_post','Pre to post changed pixel fraction',s['pre_post_changed_fraction'],definition='native connectivity diagnostic; no frozen metric number')
            metric(13,'Nucleus coverage distribution',None,'computed_distribution','structure/nuclei_distribution.csv')
        except Exception:
            failures['structure']=traceback.format_exc();metric('structure','Structural metrics',None,'failed_component',failures['structure'])
        annotation=out/'structure/annotation_nuclear.csv'
        if annotation.exists():
            cmd=[r['rscript'],str(Path(__file__).with_name('cellspa.R')),r['official_sources'],c['final_matrix'],r['reference'],r['positive_markers'],r['negative_markers'],str(annotation),str(out/'cellspa')]
            with (out/'cellspa.log').open('w') as log:
                proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=Path(__file__).resolve().parents[2],stdin=subprocess.DEVNULL)
            if proc.returncode: failures['cellspa']=f'R exit code {proc.returncode}; see cellspa.log'
            cor=out/'cellspa/official_expression_max_pearson.csv'
            if cor.exists():
                frame=pd.read_csv(cor);metric(6,'Expression-reference max Pearson',float(frame.cor_correlation.mean()),definition='official raw pooled training reference engineering value; not paper normalized profile')
            else:metric(6,'Expression-reference max Pearson',None,'failed_component','see cellspa.log')
            for ann in r['annotation_profiles']:
                for name in ['profile_failed.txt','curve_failed.txt']:
                    error_file=out/'cellspa'/ann/name
                    if error_file.exists():failures[ann+':'+name]=error_file.read_text()
                summary=out/'cellspa'/ann/'official_marker_summary.csv'
                if summary.exists():
                    for row in pd.read_csv(summary).to_dict('records'):
                        v=row['all_cell_mean'];metric(8 if row['metric'].startswith('positive') else 9,ann+':'+row['metric'],float(v) if np.isfinite(v) else None,'computed_engineering' if np.isfinite(v) else 'partial_domain_NA',f"known-type domain; {int(row['n_finite'])}/{int(row['n_total'])} finite; all-cell NA retained")
                else:metric('8/9',ann+' markers',None,'failed_component','see cellspa log/profile failure')
                curve=out/'cellspa'/ann/'official_B_to_T_curve.csv'
                metric(11,ann+' B-to-T CD4/CD8A/CD8B curves',None,'computed_curve' if curve.exists() else 'failed_component','nuclear centroid distance; no scalar; full field')
        else:
            failures['cellspa']='own-nuclear annotation unavailable';metric('6/8/9/11','CellSPA',None,'blocked_component',failures['cellspa'])
        try: changes(c,out)
        except Exception: failures['changes']=traceback.format_exc()
        metric(30,'V3 coarse to refined',None,'not_applicable','CRGF has no V3 coarse/refined endpoint')
        for i,reason in p['blocked_metrics'].items():
            if i=='38':continue
            metric(i,reason['name'],None,reason['status'],reason['reason'])
        metric(38,'Inference overhead vs BIDCell',None,'pending_paired_timing','resolved by queue report after both inference stages')
        pd.DataFrame(result).to_csv(out/'metrics.csv',index=False,na_rep='NA')
        atomic_json(out/'component_failures.json',failures)
        atomic_json(out/'complete.json',dict(status='succeeded',evaluation_executed=True,paper_protocol_complete=False,
            partial_components=failures,profile=profile,annotation_profiles=r['annotation_profiles'],wall_seconds=time.monotonic()-start,
            configuration={k:identity(v) for k,v in [('machine',machine),('reference',reference),('protocol',protocol)]},
            matrix_manifest=identity(Path(matrix)/'complete.json'),
            outputs={str(f.relative_to(out)):identity(f) for f in out.rglob('*') if f.is_file() and f.name not in ['complete.json','resources_python.json']},
            implementations={f:identity(Path(__file__).with_name(f)) for f in ['evaluate.py','structural.py','legacy_benchmark.py','cellspa.R']}))
