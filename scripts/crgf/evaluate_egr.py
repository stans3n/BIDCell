"""Five-method EGR evaluation. Scientific profiles remain separate and receipts strict."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.measure import regionprops_table
from .common import atomic_json, identity, sha256, verify_outputs, Monitor
from .evaluate import evaluate as engineering_evaluate, validate_matrix

HERE = Path(__file__).resolve().parent
METHODS = ['B1', 'E', 'G', 'R', 'GR']
PROFILES = {'remote_nuclear_annotation_raw': 'engineering_nuclear_v1',
            'windows_style_maxpearson_raw': 'engineering_pearson_v1'}


def read(path):
    return json.loads(Path(path).read_text())


def receipt(out, **fields):
    out = Path(out)
    atomic_json(out/'complete.json', dict(status='succeeded', **fields,
        outputs={str(f.relative_to(out)): identity(f) for f in sorted(out.rglob('*'))
                 if f.is_file() and f != out/'complete.json'}))


def checked_r(protocol, script, args, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
    with (out/(script+'.log')).open('w') as log:
        p = subprocess.run([protocol['rscript'], str(HERE/script), *map(str,args)],
                           stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env)
    if p.returncode:
        raise RuntimeError(f'{script} failed ({p.returncode}); {out/(script+".log")}')
    if (out/'execution_status.txt').read_text().strip() != 'succeeded':
        raise ValueError('R execution receipt missing')


def prepare(protocol, out):
    p=read(protocol); out=Path(out);out.mkdir(parents=True,exist_ok=True)
    deps={k:identity(p[k]) for k in ['reference','official_processing','official_expression','rscript']}
    deps['protocol']=identity(protocol);deps['implementation']=identity(HERE/'prepare_paper.R')
    if p.get('reference_sha256') and p['reference_sha256'] != deps['reference']['sha256']:
        raise ValueError('fixed reference hash mismatch')
    if (out/'complete.json').exists():
        saved=read(out/'complete.json')
        if saved['dependencies'] != deps: raise ValueError('prepared reference dependencies changed')
        verify_outputs(saved);return saved
    start=time.monotonic()
    checked_r(p,'prepare_paper.R',[protocol,out],out)
    atomic_json(out/'preparation_timing.json',dict(wall_seconds=time.monotonic()-start,scope='R reference training plus real smoke prediction and artifact export'))
    required=['scClassify_model.rds','markers.rds','reference_genes.txt','type_mapping.csv','reference_smoke_predictions.csv','R_packages.csv']
    for name in required:
        if not (out/name).is_file():raise ValueError('missing prepared '+name)
    receipt(out,dependencies=deps,protocol_version=p['version'],model=identity(out/'scClassify_model.rds'),
            reference_cells=30365,reference_genes=307,actual_scClassify_prediction_validated=True)
    return read(out/'complete.json')


def purity(raw, methods=METHODS):
    if set(raw)!=set(methods):return dict(status='incomplete_comparison_set',comparison_set=methods,available=sorted(raw))
    vals=np.asarray([[raw[m]['positive_F1'],raw[m]['negative_F1']] for m in methods],float)
    if not np.isfinite(vals).all():return dict(status='incomplete_comparison_set',reason='nonfinite_raw_means',comparison_set=methods)
    lo=vals.min(0);hi=vals.max(0)
    base=dict(comparison_set=methods,raw=raw,positive_min=float(lo[0]),positive_max=float(hi[0]),negative_min=float(lo[1]),negative_max=float(hi[1]))
    if np.any(hi==lo):return dict(base,status='not_identifiable_zero_range')
    z=(vals-lo)/(hi-lo);a=z[:,0];b=1-z[:,1]
    score=np.divide(2*a*b,a+b,out=np.zeros(len(a)),where=a+b!=0)
    return dict(base,status='computed',methods={m:dict(positive_scaled=float(a[i]),negative_scaled=float(z[i,1]),purity=float(score[i])) for i,m in enumerate(methods)})


def neighbour_curve(coords, labels, marker_values, settings):
    """KD-tree equivalent nearest opposite-type distances, right-closed bins."""
    labels=np.asarray(labels); coords=np.asarray(coords,float)
    src=labels==settings['source_type']; dst=np.isin(labels,settings['target_types'])
    markers=settings['markers']; rows=[]
    if not src.any() or not dst.any():
        reason='empty_source_type' if not src.any() else 'empty_target_type'
        for marker in markers:rows.append(dict(marker=marker,bin='all',n_eligible=int(src.sum()),n_positive=0,fraction=None,status=reason))
        return pd.DataFrame(rows), np.full(int(src.sum()),np.nan)
    distances=cKDTree(coords[dst]).query(coords[src],k=1,workers=1)[0]
    breaks=list(settings['breaks_um'])
    # Official function appends max and then cut sorts breaks. At max <=100 it
    # produces irregular/duplicate edges; retain fixed prespecified bins and
    # document the last interval as empty when max<=100 rather than silently fail.
    endpoint=max(float(distances.max()),100.)
    edges=np.asarray(breaks+[endpoint] if endpoint>100 else breaks,float)
    bins=np.searchsorted(edges[1:],distances,side='left')
    for i in range(7):
        ix=bins==i if i<len(edges)-1 else np.zeros(len(distances),bool)
        low=breaks[i] if i<len(breaks) else 100
        high=breaks[i+1] if i+1<len(breaks) else endpoint
        label=f'{"[" if i==0 else "("}{low},{high}]'
        for marker in markers:
            v=np.asarray(marker_values[marker])[src]
            n=int(ix.sum()); hit=int(np.count_nonzero(v[ix]>settings['threshold']))
            rows.append(dict(marker=marker,bin=label,lower_um=low,upper_um=high,n_eligible=n,n_positive=hit,
                fraction=hit/n if n else None,status='computed' if n else 'empty_bin'))
    return pd.DataFrame(rows),distances


def paper_spatial(c, out, p):
    out=Path(out); ann=pd.read_csv(out/'annotation.csv'); counts=pd.read_csv(out/'neighbour_marker_counts.csv')
    mask=tifffile.imread(c['final_mask'])
    props=pd.DataFrame(regionprops_table(mask,properties=['label','centroid']))
    props.columns=['cell_id','y_pixel','x_pixel'];props['x_um']=props.x_pixel*c['pixel_um'];props['y_um']=props.y_pixel*c['pixel_um']
    props.to_csv(out/'own_cell_centroids.csv',index=False)
    frame=ann.merge(props,on='cell_id',validate='one_to_one').merge(counts,on='cell_id',validate='one_to_one')
    if len(frame)!=len(ann):raise ValueError('annotation/own-centroid missing ID')
    curve,dist=neighbour_curve(frame[['x_um','y_um']],frame.evaluation_label,frame,p['neighbour'])
    curve['centroid']='own_final_cell';curve.to_csv(out/'neighbour_curves.csv',index=False)
    src=frame.evaluation_label==p['neighbour']['source_type']
    pd.DataFrame(dict(cell_id=frame.loc[src,'cell_id'],distance_um=dist)).to_csv(out/'nearest_opposite_distances.csv',index=False)
    nuclear=pd.read_csv(out.parent/'engineering/structure/annotation_nuclear.csv')
    joined=props.merge(nuclear,on='cell_id',suffixes=('_cell','_nucleus'),validate='one_to_one')
    joined['centroid_shift_um']=np.hypot(joined.x_um_cell-joined.x_um_nucleus,joined.y_um_cell-joined.y_um_nucleus)
    joined.to_csv(out/'cell_vs_nuclear_centroid.csv',index=False)
    frame=ann.merge(nuclear,on='cell_id',validate='one_to_one').merge(counts,on='cell_id',validate='one_to_one')
    curve,dist=neighbour_curve(frame[['x_um','y_um']],frame.evaluation_label,frame,p['neighbour'])
    curve['centroid']='nuclear_supplement';curve.to_csv(out/'neighbour_curves_nuclear_centroid.csv',index=False)
    pd.DataFrame(dict(cell_id=frame.loc[frame.evaluation_label==p['neighbour']['source_type'],'cell_id'],distance_um=dist)).to_csv(out/'nearest_opposite_distances_nuclear.csv',index=False)


def method(machine, reference, engineering_protocol, protocol, prepared, matrix, out, method):
    if method not in METHODS:raise ValueError(method)
    c=read(machine);p=read(protocol);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    prep=read(Path(prepared)/'complete.json');verify_outputs(prep)
    for key in ['reference','official_processing','official_expression','rscript']:
        if sha256(p[key])!=prep['dependencies'][key]['sha256']:raise ValueError('prepared source changed: '+key)
    if prep['protocol_version'] != p['version'] or prep['dependencies']['protocol']['sha256'] != sha256(protocol):raise ValueError('reference protocol mismatch')
    validate_matrix(c,matrix)
    if sha256(c['final_matrix'])!=sha256(Path(matrix)/'expr_mat.csv'):raise ValueError('native CSV is not the accepted own matrix')
    atomic_json(out/'complete.json',dict(status='running',paper_protocol_complete=False))
    start=time.monotonic()
    deps={k:identity(v) for k,v in [('machine',machine),('reference',reference),('engineering_protocol',engineering_protocol),('protocol',protocol),('prepared',Path(prepared)/'complete.json'),('matrix',Path(matrix)/'complete.json')]}
    with Monitor(out/'resources_python.json'):
        engineering_evaluate(machine,reference,engineering_protocol,matrix,out/'engineering',profile='EP-CORE')
        eng=read(out/'engineering/complete.json');verify_outputs(eng)
        if eng.get('partial_components'):raise RuntimeError('required engineering components failed: '+str(eng['partial_components']))
        for profile in PROFILES:
            for name in ['official_markers_per_cell.csv','official_marker_summary.csv','official_B_to_T_curve.csv']:
                if not (out/'engineering/cellspa'/profile/name).exists():raise ValueError('missing engineering output '+name)
        paper=out/'paper_aligned_v1'
        checked_r(p,'paper_evaluation.R',[protocol,prepared,c['final_matrix'],paper],paper)
        paper_spatial(c,paper,p)
        qc=pd.read_csv(paper/'qc_all_cells.csv');markers=pd.read_csv(paper/'markers_per_cell.csv');ann=pd.read_csv(paper/'annotation.csv')
        if set(qc.loc[qc.qc_pass,'cell_id']) != set(markers.cell_id) or set(markers.cell_id)!=set(ann.cell_id):raise ValueError('QC/marker/annotation IDs mismatch')
        if not np.isfinite(markers[['positive_F1','negative_F1']]).all().all():raise ValueError('nonfinite F1')
        summary=dict(method=method,n_before_qc=len(qc),n_after_qc=len(markers),qc_retention=len(markers)/len(qc),
                     n_unassigned=int(ann.rejected.sum()),unassigned_fraction=float(ann.rejected.mean()),
                     positive_F1=float(markers.positive_F1.mean()),negative_F1=float(markers.negative_F1.mean()),
                     protocol_version=p['version'],model_sha256=prep['model']['sha256'])
        atomic_json(out/'summary.json',summary)
    receipt(out,method=method,dependencies=deps,evaluation_executed=True,paper_method_complete=True,
            paper_protocol_complete=False,purity_status='awaiting_complete_comparison_set',protocol_version=p['version'],wall_seconds=time.monotonic()-start,
            implementations={n:identity(HERE/n) for n in ['evaluate_egr.py','prepare_paper.R','paper_evaluation.R','evaluate.py','cellspa.R','structural.py']})
    return summary


def changes(own_path, reference_path, out, comparison, pixel_um=1.):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    own=tifffile.imread(own_path);base=tifffile.imread(reference_path)
    if own.shape!=base.shape:raise ValueError('pair mask shape mismatch')
    changed=own!=base;ids=np.union1d(np.unique(base),np.unique(own));ids=ids[ids>0]
    maxid=int(ids.max(initial=0));aa=np.bincount(own.ravel(),minlength=maxid+1);bb=np.bincount(base.ravel(),minlength=maxid+1)
    intersection=np.bincount(np.where(own==base,own,0).ravel(),minlength=maxid+1)[ids];union=aa[ids]+bb[ids]-intersection
    cells=pd.DataFrame(dict(cell_id=ids,own_area_px=aa[ids],reference_area_px=bb[ids],intersection=intersection,union=union,iou=intersection/union,changed=intersection!=union))
    cells.to_csv(out/'same_id_iou.csv',index=False)
    boundary=np.zeros(base.shape,bool)
    edge=(base[1:]!=base[:-1])&((base[1:]>0)|(base[:-1]>0));boundary[1:]|=edge;boundary[:-1]|=edge
    edge=(base[:,1:]!=base[:,:-1])&((base[:,1:]>0)|(base[:,:-1]>0));boundary[:,1:]|=edge;boundary[:,:-1]|=edge
    distance=ndi.distance_transform_edt(~boundary,sampling=pixel_um)
    bands=[]
    for width in (1,2,3,5):
        band=distance<=width;n=int(band.sum());hit=int((band&changed).sum())
        bands.append(dict(width_um=width,band_pixels=n,changed_pixels=hit,within_band_change_fraction=hit/n if n else None,share_of_all_changes=hit/int(changed.sum()) if changed.any() else None))
    interior=(base>0)&(distance>5);n=int(interior.sum())
    result=dict(comparison=comparison,direction='own minus reference',reference=identity(reference_path),own=identity(own_path),
        changed_pixels=int(changed.sum()),n_pixels=int(changed.size),changed_fraction=float(changed.mean()),
        changed_cell_fraction=float(cells.changed.mean()),n_cells=len(ids),bands=bands,
        interior_pixels=n,interior_preservation=1-int((changed&interior).sum())/n if n else None,
        definition='reference foreground/background and cell/cell 4-neighbour interface pixels, both sides; Euclidean distance in um; interior reference foreground >5um; same-ID IoU is change not accuracy')
    atomic_json(out/'change.json',result);return result


def resource_rows(method, spec):
    root=Path(spec['cost_root']); old=spec.get('historical_name',method)
    status_path=Path(spec.get('run_status',root/'run_status.json'))
    rows=[]
    if status_path.exists():
        status=read(status_path)
        for name,stage in status.get('stages',{}).items():
            if stage.get('method') not in [old,method]:continue
            for i,a in enumerate(stage.get('attempts',[])):
                rows.append(dict(method=method,stage=stage.get('stage',name),attempt=i+1,
                    wall_seconds=a.get('wall_seconds'),sampled_peak_rss_bytes_sum=a.get('sampled_peak_rss_bytes_sum'),
                    sampled_peak_pss_bytes_sum=a.get('sampled_peak_pss_bytes_sum'),exit_code=a.get('exit_code'),
                    source=str(status_path),scope=a.get('memory_sampling_scope','monitor scope not recorded')))
    return rows


def paired_diagnostics(methods,out,comparisons):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    tables={}; ids=None
    for m,spec in methods.items():
        root=Path(spec['evaluation']);p=root/'paper_aligned_v1'
        table=pd.read_csv(p/'markers_per_cell.csv').set_index('cell_id')
        tables[m]=table;ids=set(table.index) if ids is None else ids&set(table.index)
    ids=sorted(ids);pd.DataFrame({'cell_id':ids}).to_csv(out/'common_qc_cell_ids.csv',index=False)
    summaries=[]
    for a,b in comparisons:
        ta=tables[a].loc[ids];tb=tables[b].loc[ids]
        trans=pd.crosstab(tb.cell_type,ta.cell_type,dropna=False)
        trans.to_csv(out/f'{a}_minus_{b}_type_transition_common_qc.csv')
        nuclear={}
        for m in [a,b]:
            root=Path(methods[m]['evaluation'])/'engineering/cellspa/remote_nuclear_annotation_raw'
            nuclear[m]=pd.read_csv(root/'official_markers_per_cell.csv').set_index('cell_id').loc[ids]
        if not nuclear[a].cell_type.equals(nuclear[b].cell_type):raise ValueError('fixed nuclear types changed between methods')
        frame=pd.DataFrame(index=ids);frame.index.name='cell_id';frame['fixed_nuclear_type']=nuclear[a].cell_type
        for metric in ['positive_F1','positive_Precision','positive_Recall','negative_F1','negative_Precision','negative_Recall']:
            frame['paper_delta_'+metric]=ta[metric]-tb[metric]
            frame['fixed_nuclear_delta_'+metric]=nuclear[a][metric]-nuclear[b][metric]
        frame.to_csv(out/f'{a}_minus_{b}_paired_cells.csv')
        for group,df in [('__all__',frame),*list(frame.groupby('fixed_nuclear_type',dropna=False))]:
            for metric in frame.columns[1:]:
                v=df[metric];finite=v[np.isfinite(v)]
                summaries.append(dict(comparison=f'{a}-{b}',fixed_nuclear_type=group,metric=metric,n_cells=len(v),n_finite=len(finite),mean_delta=float(finite.mean()) if len(finite) else None,reason='' if len(finite) else 'empty_or_nonfinite_group'))
        # Fixed B1 nuclear-centroid density quartiles; descriptive shared strata.
        shape=pd.read_csv(Path(methods[b]['evaluation'])/'engineering/structure/annotation_nuclear.csv').set_index('cell_id').loc[ids]
        if len(shape)>1:
            d=cKDTree(shape[['x_um','y_um']]).query(shape[['x_um','y_um']],k=min(6,len(shape)),workers=1)[0]
            radius=d[:,-1] if d.ndim==2 else d
            frame['density_neighbour_radius_um']=radius
            frame['density_quartile']=pd.qcut(pd.Series(radius,index=ids),q=4,labels=False,duplicates='drop')
            frame.groupby('density_quartile').agg(n_cells=('paper_delta_positive_F1','size'),positive_delta=('paper_delta_positive_F1','mean'),negative_delta=('paper_delta_negative_F1','mean'),radius_mean=('density_neighbour_radius_um','mean')).to_csv(out/f'{a}_minus_{b}_density_strata.csv')
    pd.DataFrame(summaries).to_csv(out/'paired_summary.csv',index=False)
    return len(ids)


def aggregate(manifest,protocol,out):
    p=read(protocol);entries=read(manifest);methods=entries.get('methods',entries)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'complete.json',dict(status='running',paper_protocol_complete=False))
    valid={};failures={};raw={}
    for m in METHODS:
        try:
            spec=methods[m];root=Path(spec['evaluation']);r=read(root/'complete.json')
            if r['status']!='succeeded' or not r.get('paper_method_complete') or r['protocol_version']!=p['version']:raise ValueError('method incomplete or protocol mismatch')
            verify_outputs(r);s=read(root/'summary.json');valid[m]=spec;raw[m]={k:s[k] for k in ['positive_F1','negative_F1']}
        except (KeyError,OSError,ValueError) as exc:failures[m]=str(exc)
    if len(valid)!=5 and (out/'final_comparison.csv').exists():(out/'final_comparison.csv').unlink()
    pur=purity(raw);atomic_json(out/'purity.json',pur);atomic_json(out/'incomplete_methods.json',failures)
    coverage=[];metrics=[]
    def cover(i,profile,comparison,required,status,reason,n,artifact):
        coverage.append(dict(metric_id=str(i),profile=profile,comparison=comparison,required_this_run=required,status=status,reason=reason,n_eligible=n,definition_version=p['version'],implementation='scripts/crgf/evaluate_egr.py',artifact_path=str(artifact)))
    costs=[]
    for m,spec in valid.items():
        root=Path(spec['evaluation']);paper=root/'paper_aligned_v1';s=read(root/'summary.json')
        eng=pd.read_csv(root/'engineering/metrics.csv',dtype={'metric_id':str})
        for row in eng.to_dict('records'):
            i=row['metric_id']
            if i in [str(x) for x in list(range(1,6))+list(range(12,19))+[20,21]]:
                metrics.append(dict(method=m,profile='structural_own_mask',**row))
        for i in list(range(1,6))+list(range(12,19))+[20,21]:
            cover(i,'structural_own_mask',m,True,'computed_distribution' if i in [3,4,5,13,18] else 'computed','own final mask; all cells; capture denominator all eligible transcripts',len(pd.read_csv(root/'engineering/structure/cells_distribution.csv')),root/'engineering/structure')
        for old,profile in PROFILES.items():
            for i in [8,9,11]:cover(i,profile,m,True,'computed','historical raw/Unknown semantics retained',None,root/'engineering/cellspa'/old)
            summary=pd.read_csv(root/'engineering/cellspa'/old/'official_marker_summary.csv')
            for row in summary.to_dict('records'):metrics.append(dict(method=m,profile=profile,metric_id='8' if row['metric'].startswith('positive') else '9',metric_name=row['metric'],value=row['all_cell_mean'],n_eligible=row['n_total'],n_finite=row['n_finite'],status='computed' if np.isfinite(row['all_cell_mean']) else 'partial_domain_NA',finite_mean=row['finite_mean']))
        for i,artifact in [(6,'type_correlations.csv'),(7,'type_correlations.csv'),(8,'markers_per_cell.csv'),(9,'markers_per_cell.csv'),(11,'neighbour_curves.csv')]:cover(i,'paper_aligned_v1',m,True,'computed','QC-pass complete population; per-type/empty-bin status in artifact',s['n_after_qc'],paper/artifact)
        for k in ['positive_F1','negative_F1','qc_retention','unassigned_fraction']:
            metrics.append(dict(method=m,profile='paper_aligned_v1',metric_id='8' if k=='positive_F1' else '9' if k=='negative_F1' else k,metric_name=k,value=s[k],n_eligible=s['n_after_qc'],status='computed'))
        for row in pd.read_csv(paper/'marker_summary.csv').query('cell_type == "__all__"').to_dict('records'):
            if row['metric'].endswith('F1'):continue
            metrics.append(dict(method=m,profile='paper_aligned_v1',metric_id='8' if row['metric'].startswith('positive') else '9',metric_name=row['metric'],value=row['all_cell_mean'],n_eligible=row['n_total'],n_finite=row['n_finite'],status='computed' if np.isfinite(row['all_cell_mean']) else 'partial_domain_NA'))
        corr=pd.read_csv(paper/'type_correlations.csv')
        for row in corr.to_dict('records'):metrics.append(dict(method=m,profile='paper_aligned_v1',metric_id='7' if row['definition']=='detection' else '6',metric_name=row['definition']+':'+row['cell_type'],value=row['pearson'],n_eligible=row['n_cells'],status=row['status']))
        single=pd.read_csv(paper/'single_cell_max_pearson.csv');metrics.append(dict(method=m,profile='paper_aligned_v1',metric_id='6',metric_name='single_cell_max_reference_Pearson',value=single.max_pearson.mean(),n_eligible=len(single),status='computed'))
        if pur['status']=='computed':metrics.append(dict(method=m,profile='paper_aligned_v1',metric_id='10',metric_name='Purity',value=pur['methods'][m]['purity'],status='computed'))
        cover(10,'paper_aligned_v1',m,True,pur['status'],'fixed comparison set B1/E/G/R/GR',s['n_after_qc'],out/'purity.json')
        for i,status in p['scope'].items():cover(i,'all',m,False,status,'prespecified scope/protocol limitation',None,protocol)
        for item,artifact in [('gene_detection',paper/'gene_detection_313.csv'),('type_composition',paper/'type_composition.csv'),('raster_morphology',root/'engineering/structure/shape_raster_distribution.csv'),('nuclear_matching',root/'engineering/structure/cells_distribution.csv'),('spatial_distribution',paper/'own_cell_centroids.csv')]:cover('Tier1B:'+item,'paper_aligned_v1' if item in ['gene_detection','type_composition'] else 'structural_own_mask',m,True,'computed','descriptive raster/own-cell outputs',None,artifact)
        for item,reason in [('polygon_morphology','official polygon/alpha-shape morphology not implemented; raster descriptors separately provided'),('local_Shannon','spatial neighbourhood parameter not frozen'),('local_CV','spatial neighbourhood parameter not frozen')]:cover('Tier1B:'+item,'supplementary',m,False,'not_run_scope',reason,None,'')
        cr=resource_rows(m,spec);costs+=cr
        training=Path(spec['cost_root'])/'checkpoints'/spec.get('historical_name',m)/'complete.json'
        tr=read(training) if training.exists() else {}
        param_source=training
        if not any(k in tr for k in ['parameters','trainable_parameters','parameter_count']):
            init=training.with_name('initialization.json')
            historical=Path(spec['cost_root'])/'analysis/analyst_training_verification.json'
            if init.exists():
                tr['parameters']=read(init)['parameters'];param_source=init
            elif historical.exists():
                obj=read(historical);old=spec.get('historical_name',m)
                # This historical audit records actual tensors and optimizer identity.
                for section in obj.values():
                    if isinstance(section,dict) and isinstance(section.get(old),dict) and 'parameter_count' in section[old]:
                        tr['parameters']=section[old]['parameter_count'];param_source=historical;break
        stage_sums={stage:sum(r['wall_seconds'] or 0 for r in cr if r['stage']==stage) for stage in ['train','predict','postprocess']}
        for i,stage in [(31,'train'),(32,'predict'),(33,'postprocess')]:
            found=any(r['stage']==stage for r in cr)
            cover(i,'resources',m,True,'computed' if found else 'missing_historical_monitor','all attempts wall seconds; stage process startup included',None,out/'resource_costs.csv')
            metrics.append(dict(method=m,profile='resources',metric_id=str(i),metric_name=stage+'_wall_seconds',value=stage_sums[stage] if found else None,status='computed' if found else 'missing_historical_monitor'))
        prediction_receipt=Path(spec['cost_root'])/'predictions'/spec.get('historical_name',m)/'complete.json'
        pr=read(prediction_receipt) if prediction_receipt.exists() else {}
        gpu_peaks=[v for v in [tr.get('peak_vram_bytes'),pr.get('peak_vram_bytes')] if v is not None]
        peak=max(gpu_peaks) if gpu_peaks else None;rss=[r['sampled_peak_rss_bytes_sum'] for r in cr if r['sampled_peak_rss_bytes_sum'] is not None]
        for i,name,val in [(34,'max_train_prediction_torch_allocated_bytes',peak),(35,'sampled_peak_cpu_RSS_process_group_sum_bytes',max(rss) if rss else None),(36,'trainable_parameters',tr.get('parameters',tr.get('trainable_parameters',tr.get('parameter_count')))),(37,'training_updates_per_second',tr.get('updates',0)/stage_sums['train'] if stage_sums['train']>0 else None)]:
            status='computed' if val is not None else 'missing_historical_monitor'
            cover(i,'resources',m,True,status,'single run; CUDA allocator peak not total device usage; sampled CPU RSS sum double-counts shared pages; transients may be missed',None,param_source if i==36 else training if i==34 else out/'resource_costs.csv')
            metrics.append(dict(method=m,profile='resources',metric_id=str(i),metric_name=name,value=val,status=status))
    changes_results={}
    if len(valid)==5:
        model_hashes={read(Path(spec['evaluation'])/'summary.json')['model_sha256'] for spec in valid.values()}
        if len(model_hashes)!=1:raise ValueError('methods use different fixed reference models')
        ncommon=paired_diagnostics(valid,out/'paired',p['comparisons'])
        for a,b in p['comparisons']:
            ac=read(valid[a]['machine']);bc=read(valid[b]['machine'])
            if ac['pixel_um']!=bc['pixel_um']:raise ValueError('pair spacing differs')
            dest=out/'changes'/f'{a}_minus_{b}'
            ch=changes(ac['final_mask'],bc['final_mask'],dest,f'{a}-{b}',ac['pixel_um']);changes_results[f'{a}-{b}']=ch
            cells=pd.read_csv(dest/'same_id_iou.csv')
            coords=pd.read_csv(Path(valid['B1']['evaluation'])/'engineering/structure/annotation_nuclear.csv')
            radius=cKDTree(coords[['x_um','y_um']]).query(coords[['x_um','y_um']],k=min(6,len(coords)),workers=1)[0]
            coords['density_radius_um']=radius[:,-1] if radius.ndim==2 else radius
            coords['density_quartile']=pd.qcut(coords.density_radius_um,4,labels=False,duplicates='drop')
            cells=cells.merge(coords[['cell_id','density_quartile','density_radius_um']],on='cell_id',validate='one_to_one')
            cells['symmetric_difference_pixels']=cells.union-cells.intersection
            cells.groupby('density_quartile').agg(n_cells=('cell_id','size'),changed_cell_fraction=('changed','mean'),mean_iou=('iou','mean'),median_iou=('iou','median'),symmetric_difference_pixels=('symmetric_difference_pixels','sum'),mean_density_radius_um=('density_radius_um','mean')).to_csv(dest/'change_density_strata.csv')
            for i in range(22,30):cover(i,'own_mask_change',f'{a}-{b}',True,'computed','same-ID diagnostic; reference boundary Euclidean bands; not accuracy',ch['n_cells'] if i in [23,24] else ch['n_pixels'],dest)
        frame=pd.DataFrame(metrics)
        for m in METHODS:
            own=frame[(frame.method==m)&frame.metric_id.isin(['32','33'])].value.sum(min_count=2)
            base=frame[(frame.method=='B1')&frame.metric_id.isin(['32','33'])].value.sum(min_count=2)
            value=own/base if np.isfinite(own) and np.isfinite(base) and base>0 else None
            metrics.append(dict(method=m,profile='resources',metric_id='38',metric_name='prediction_plus_postprocessing_ratio_to_B1',value=value,status='computed' if value is not None else 'missing_historical_monitor'))
            cover(38,'resources',m,True,'computed' if value is not None else 'missing_historical_monitor','single full inference includes prediction + postprocessing; no repeated speed inference',None,out/'raw_metrics.csv')
    else:
        ncommon=None
        for a,b in p['comparisons']:
            for i in range(22,30):cover(i,'own_mask_change',f'{a}-{b}',True,'blocked_by_dependency','full set incomplete; no full comparison/rank publication',None,out/'incomplete_methods.json')
        for m in METHODS:
            if m not in valid:
                for i in range(1,44):cover(i,'all',m,str(i) not in p['scope'],'blocked_by_dependency' if str(i) not in p['scope'] else p['scope'][str(i)],failures.get(m,''),None,'')
            cover(38,'resources',m,True,'blocked_by_dependency','full set incomplete',None,'')
    pd.DataFrame(costs).to_csv(out/'resource_costs.csv',index=False)
    pd.DataFrame(metrics).to_csv(out/'raw_metrics.csv',index=False,na_rep='NA')
    pd.DataFrame(coverage).to_csv(out/'metric_coverage.csv',index=False,na_rep='NA')
    frame=pd.DataFrame(metrics)
    if len(valid)==5:
        wide=frame.pivot_table(index=['profile','metric_id','metric_name'],columns='method',values='value',aggfunc='first',dropna=True)
        for a,b in p['comparisons']:wide[a+'-'+b]=wide[a]-wide[b]
        wide['interaction_(GR-G)-(R-E)']=(wide['GR']-wide['G'])-(wide['R']-wide['E'])
        wide.to_csv(out/'final_comparison.csv',na_rep='NA')
    allowed={'computed','computed_distribution','not_identifiable_zero_range','partial_domain_NA','missing_historical_monitor'}
    coverage_failures=[r for r in coverage if r['required_this_run'] and (r['status'] not in allowed or (r['status']=='missing_historical_monitor' and r['metric_id'] not in ['34','35']))]
    atomic_json(out/'coverage_failures.json',coverage_failures)
    complete=len(valid)==5 and not coverage_failures and pur['status'] in ['computed','not_identifiable_zero_range']
    if complete:
        receipt(out,evaluation_executed=True,paper_protocol_complete=True,protocol_version=p['version'],comparison_set=METHODS,common_qc_n=ncommon,
            dependencies={'manifest':identity(manifest),'protocol':identity(protocol)},method_receipts={m:identity(Path(s['evaluation'])/'complete.json') for m,s in valid.items()})
    else:
        atomic_json(out/'complete.json',dict(status='incomplete_comparison_set',evaluation_executed=True,paper_protocol_complete=False,failures=failures,coverage_failures=coverage_failures,purity_status=pur['status']))
    return complete


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare',help='train fixed real scClassify reference and validate prediction');a.add_argument('--protocol',required=True);a.add_argument('--out',required=True)
    a=sub.add_parser('method',help='own-matrix engineering plus paper-aligned evaluation')
    for name in ['machine','reference','engineering-protocol','protocol','prepared','matrix','out','method']:a.add_argument('--'+name,required=True)
    a=sub.add_parser('aggregate',help='strict full-five-method purity, pairs and frozen coverage');a.add_argument('--manifest',required=True);a.add_argument('--protocol',required=True);a.add_argument('--out',required=True)
    args=vars(parser.parse_args());command=args.pop('command')
    if command=='prepare':prepare(**args)
    elif command=='method':method(**args)
    elif not aggregate(**args):raise SystemExit(2)


if __name__=='__main__':main()
