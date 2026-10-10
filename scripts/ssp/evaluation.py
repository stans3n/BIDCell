"""Accepted three-profile evaluator plus strictly scoped COS comparisons."""
import argparse
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scripts.crgf import evaluate_egr as accepted
from scripts.crgf.common import atomic_json, identity, sha256, verify_outputs, Monitor, digest
from scripts.cos import diagnostics

ARMS=['S','SP']
ALL_METHODS=['A','B','S','SP']
PAIRS=[('S','B'),('SP','S'),('SP','B'),('S','A'),('SP','A')]
PROFILES=['engineering_nuclear_v1','engineering_pearson_v1','paper_aligned_v1']
VERSION='SSP-OVERNIGHT-v1.0'
SCOPE={19:'pending_project_threshold',30:'not_applicable',39:'not_applicable',40:'not_run_scope',41:'not_run_scope',42:'pending_uncertainty_design',43:'not_run_scope'}


def read(path):return json.loads(Path(path).read_text())


def times():
    now=datetime.now(timezone.utc)
    return dict(time_utc=now.isoformat(),time_sydney=now.astimezone(ZoneInfo('Australia/Sydney')).isoformat())


def publish(out, **fields):
    out=Path(out)
    atomic_json(out/'complete.json',dict(status='succeeded',version=VERSION,**times(),**fields,
        outputs={str(p.relative_to(out)):identity(p) for p in sorted(out.rglob('*')) if p.is_file() and p!=out/'complete.json'}))


def _evaluate(config):
    armcfg=read(config) if isinstance(config,(str,Path)) else (config.model_dump(mode='json') if hasattr(config,'model_dump') else dict(config))
    queue=read(armcfg['queue_config']);arm=armcfg['arm_id'];spec=queue['methods'][arm]
    if arm not in ARMS or armcfg['base_variant']!='B1':raise ValueError('arm/base identity mismatch')
    out=Path(spec['evaluation']);out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'complete.json',dict(status='running',paper_method_complete=False,**times()))
    # The accepted method argument is a provenance label only. All scientific
    # inputs are the new arm's own machine/counts; replace the label immediately.
    new_protocol=read(queue['paper_protocol']);prep=read(Path(queue['prepared'])/'complete.json');original_protocol=prep['dependencies']['protocol']['path']
    original=read(original_protocol)
    allowed={'version','comparison_set','comparisons'}
    if {k:v for k,v in original.items() if k not in allowed}!={k:v for k,v in new_protocol.items() if k not in allowed}:raise ValueError('scientific evaluator protocol differs from accepted reference preparation')
    if new_protocol['comparison_set']!=ALL_METHODS or new_protocol['comparisons']!=[list(x) for x in PAIRS]:raise ValueError('SSP comparison protocol identity mismatch')
    import inspect
    profile_dependencies={k:identity(v) for k,v in {'machine':spec['machine'],'matrix':Path(spec['matrix'])/'complete.json','protocol':original_protocol,'prepared':Path(queue['prepared'])/'complete.json','engineering_reference':queue['reference'],'engineering_protocol':queue['engineering_protocol']}.items()}
    profile_dependencies['adapter_source']=digest(inspect.getsource(run_profiles))
    profile_dependencies['accepted_source']={name:sha256(accepted.HERE/name) for name in ['evaluate.py','evaluate_egr.py','structural.py','cellspa.R','paper_evaluation.R']}
    cached=out/'profiles_complete.json'
    if cached.exists():
        previous=read(cached)
        if previous['dependencies']!=profile_dependencies:raise ValueError('completed profile dependencies changed; explicit new attempt/version required')
        verify_outputs(previous);result=previous['summary']
    else:
        result=run_profiles(spec,queue,original_protocol,out,arm)
        atomic_json(cached,dict(status='succeeded',dependencies=profile_dependencies,summary=result,outputs={str(file.relative_to(out)):identity(file) for folder in ['engineering','paper_aligned_v1'] for file in (out/folder).rglob('*') if file.is_file()}))
    result['method']=arm;result['protocol_version']=new_protocol['version'];result['intervention']={k:armcfg[k] for k in ('anchor_weight','pos_background_rho','pos_region_version','calibration_identity','cos_enabled','teacher_version','prototype_version')};result['intervention']['prototype_enabled']=arm=='SP';result['intervention']['prepared_manifest']=identity(armcfg['prepared_manifest']);result['arm_config']=identity(spec['config']);atomic_json(out/'summary.json',result)
    atomic_json(out/'reference_reuse.json',dict(status='succeeded',allowed_identity_only_changes=sorted(allowed),scientific_settings_exactly_equal=True,prepared=identity(Path(queue['prepared'])/'complete.json'),original_protocol=identity(original_protocol),cos_protocol=identity(queue['paper_protocol']),model=prep['model']))
    diagnostics.stage_diagnostics(dict(spec,base_variant='B1'),queue['anchors'],queue['prepared'],out/'diagnostics/stages')
    probe_manifest=read(Path(read(spec['machine'])['final_mask']).parent/'endpoints.json').get('probes')
    if not probe_manifest:raise ValueError('required frozen probe output absent')
    diagnostics.probe_diagnostics(probe_manifest,out/'diagnostics/probes')
    atomic_json(out/'probe_manifest.json',probe_manifest)
    coverage,raw=method_coverage(queue,arm,include_resources=False)
    pd.DataFrame(coverage).to_csv(out/'metric_coverage.csv',index=False,na_rep='NA')
    pd.DataFrame(raw).to_csv(out/'raw_metrics.csv',index=False,na_rep='NA')
    failures=[r for r in coverage if r['required_this_run'] and r['status']=='failed']
    if failures:raise ValueError('required method coverage failed: '+str(failures))
    publish(out,method=arm,evaluation_executed=True,paper_method_complete=True,paper_protocol_complete=False,
        profiles=PROFILES,purity_status='incomplete_comparison_set',protocol_version=read(queue['paper_protocol'])['version'],
        dependencies={'arm_config':identity(spec['config']),'intervention':result['intervention'],'queue_config':identity(armcfg['queue_config']),'matrix':identity(Path(spec['matrix'])/'complete.json'),'accepted_protocol':identity(queue['paper_protocol'])})
    return result


def evaluate(config):
    armcfg=read(config) if isinstance(config,(str,Path)) else (config.model_dump(mode='json') if hasattr(config,'model_dump') else dict(config))
    queue=read(armcfg['queue_config']);out=Path(queue['methods'][armcfg['arm_id']]['evaluation']);out.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    try:
        with Monitor(out/'resources_evaluation_python.json'):result=_evaluate(config)
    except Exception:
        import traceback
        atomic_json(out/'complete.json',dict(status='failed',paper_method_complete=False,failure=traceback.format_exc(),wall_seconds=time.monotonic()-start,**times()))
        raise
    complete=read(out/'complete.json');complete['wall_seconds']=time.monotonic()-start;complete['outputs']['resources_evaluation_python.json']=identity(out/'resources_evaluation_python.json');atomic_json(out/'complete.json',complete)
    return result


def row(metric,profile,comparison,artifact,status='computed',n=None,reason='',required=True,subitem=None,n_evaluable=None):
    artifact=Path(artifact) if artifact else None
    if required and (artifact is None or not artifact.exists()):status='failed';reason='required artifact missing'
    return dict(metric_id=str(metric),subitem=str(subitem if subitem is not None else metric),n_evaluable=n_evaluable,n_unavailable=(n-n_evaluable) if n is not None and n_evaluable is not None else None,profile=profile,comparison=comparison,required_this_run=required,status=status,reason=reason,n_eligible=n,definition_version=VERSION,implementation='scripts/ssp/evaluation.py',artifact_path=str(artifact) if artifact else '')


def validate_table(path,columns,nonnegative=()):
    table=pd.read_csv(path)
    if not set(columns)<=set(table.columns):raise ValueError('missing required table columns '+str(path))
    for name in nonnegative:
        if table[name].isna().any() or (table[name]<0).any():raise ValueError('invalid denominator '+str(path)+':'+name)
    if 'cell_id' in table and table.cell_id.duplicated().any():raise ValueError('duplicate per-cell output '+str(path))
    return table


def method_coverage(queue,arm,include_resources=True,cost_snapshot=None):
    spec=queue['methods'][arm];root=Path(spec['evaluation']);paper=root/'paper_aligned_v1';coverage=[];raw=[]
    summary=read(root/'summary.json');eng=pd.read_csv(root/'engineering/metrics.csv',dtype={'metric_id':str})
    validate_table(paper/'markers_per_cell.csv',['cell_id',*diagnostics.METRICS])
    validate_table(paper/'neighbour_curves.csv',['marker','bin','n_eligible','n_positive','fraction','status'],['n_eligible','n_positive'])
    validate_table(paper/'type_correlations.csv',['cell_type','definition','n_cells','pearson','status'],['n_cells'])
    correlation=validate_table(root/'engineering/cellspa/official_expression_max_pearson.csv',['cell_id','cor_correlation'])
    coverage.append(row(6,'engineering_pearson_v1',arm,root/'engineering/cellspa/official_expression_max_pearson.csv',n=len(correlation),n_evaluable=int(np.isfinite(correlation.cor_correlation).sum()),subitem='single_cell_max_reference_Pearson_raw'))
    raw.append(dict(method=arm,profile='engineering_pearson_v1',metric_id='6',metric_name='single_cell_max_reference_Pearson_raw',value=float(correlation.cor_correlation.mean()),n_eligible=len(correlation),n_finite=int(np.isfinite(correlation.cor_correlation).sum()),status='computed'))
    structural=list(range(1,6))+list(range(12,19))+[20,21]
    structure=root/'engineering/structure';stats=read(structure/'structure.json');matrix_receipt=read(Path(spec['matrix'])/'complete.json')
    cells=pd.read_csv(structure/'cells_distribution.csv');nuclei=pd.read_csv(structure/'nuclei_distribution.csv')
    denominator={i:len(cells) for i in structural}
    denominator.update({2:int(matrix_receipt['totals']['eligible']),12:len(nuclei),13:len(nuclei),16:len(nuclei),20:int(np.prod(matrix_receipt['contract']['shape'])),21:int(np.prod(matrix_receipt['contract']['shape']))})
    artifacts={i:structure/'cells_distribution.csv' for i in structural}
    artifacts.update({2:Path(spec['matrix'])/'complete.json',12:structure/'nuclei_distribution.csv',13:structure/'nuclei_distribution.csv',14:structure/'cell_nuclei_associations.csv',15:structure/'cell_nuclei_associations.csv',16:structure/'nuclei_distribution.csv',17:structure/'components_8.csv',20:structure/'structure.json',21:structure/'structure.json'})
    for i in structural:
        found=eng[eng.metric_id==str(i)]
        if found.empty:coverage.append(row(i,'structural_own_mask',arm,artifacts[i],status='failed',subitem='missing_metric',n=denominator[i],n_evaluable=0));continue
        for record in found.to_dict('records'):
            status='computed_distribution' if i in (3,4,5,13,18) else 'computed'
            if any(x in str(record['status']) for x in ['failed','blocked']):status='failed'
            finite=int(np.isfinite(cells.cell_nucleus_area_ratio).sum()) if i==18 else denominator[i]
            coverage.append(row(i,'structural_own_mask',arm,artifacts[i],status=status,subitem=record['metric_name'],n=denominator[i],n_evaluable=finite,reason='all eligible transcripts' if i==2 else 'all fixed nuclei' if i in (12,13,16) else 'entire aligned raster pixels' if i in (20,21) else 'own final cell-ID population'))
            raw.append(dict(method=arm,profile='structural_own_mask',**record))
    for old,profile in accepted.PROFILES.items():
        dest=root/'engineering/cellspa'/old
        validate_table(dest/'official_markers_per_cell.csv',['cell_id',*diagnostics.METRICS])
        validate_table(dest/'curve_with_denominators.csv',['marker','bin','n_eligible','n_positive','fraction','status'],['n_eligible','n_positive'])
        marker_table=pd.read_csv(dest/'official_markers_per_cell.csv')
        for metric in diagnostics.METRICS:
            n_finite=int(np.isfinite(marker_table[metric]).sum())
            coverage.append(row(8 if metric.startswith('positive') else 9,profile,arm,dest/'official_markers_per_cell.csv',subitem=metric,n=len(marker_table),n_evaluable=n_finite,status='computed' if n_finite==len(marker_table) else 'partial_domain_NA',reason='historical engineering known-type domain; unavailable stays NA'))
        for curve_row in pd.read_csv(dest/'curve_with_denominators.csv').to_dict('records'):
            coverage.append(row(11,profile,arm,dest/'curve_with_denominators.csv',subitem=curve_row['marker']+':'+str(curve_row['bin']),n=int(curve_row['n_eligible']),n_evaluable=int(curve_row['n_eligible']),status='computed' if curve_row['status']=='computed' else 'partial_domain_NA',reason=curve_row['status']))
        for rec in pd.read_csv(dest/'official_marker_summary.csv').to_dict('records'):
            raw.append(dict(method=arm,profile=profile,metric_id='8' if rec['metric'].startswith('positive') else '9',metric_name=rec['metric'],value=rec['all_cell_mean'],finite_mean=rec['finite_mean'],n_eligible=rec['n_total'],n_finite=rec['n_finite'],status='computed' if np.isfinite(rec['all_cell_mean']) else 'partial_domain_NA'))
    single=pd.read_csv(paper/'single_cell_max_pearson.csv')
    coverage.append(row(6,'paper_aligned_v1',arm,paper/'single_cell_max_pearson.csv',subitem='single_cell_max_reference_Pearson',n=len(single),n_evaluable=int(np.isfinite(single.max_pearson).sum())))
    raw.append(dict(method=arm,profile='paper_aligned_v1',metric_id='6',metric_name='single_cell_max_reference_Pearson',value=float(single.max_pearson.mean()),n_eligible=len(single),n_finite=int(np.isfinite(single.max_pearson).sum()),status='computed'))
    for corr_row in pd.read_csv(paper/'type_correlations.csv').to_dict('records'):
        coverage.append(row(6 if corr_row['definition']=='mean_logcounts' else 7,'paper_aligned_v1',arm,paper/'type_correlations.csv',subitem=corr_row['definition']+':'+corr_row['cell_type'],n=int(corr_row['n_cells']),n_evaluable=int(corr_row['n_cells']) if np.isfinite(corr_row['pearson']) else 0,status='computed' if corr_row['status']=='computed' else 'partial_domain_NA',reason=corr_row['status']))
    marker_table=pd.read_csv(paper/'markers_per_cell.csv')
    for metric in diagnostics.METRICS:
        n_finite=int(np.isfinite(marker_table[metric]).sum())
        coverage.append(row(8 if metric.startswith('positive') else 9,'paper_aligned_v1',arm,paper/'markers_per_cell.csv',subitem=metric,n=len(marker_table),n_evaluable=n_finite,status='computed' if n_finite==len(marker_table) else 'partial_domain_NA',reason='accepted paper rejection/unmatched zero rule; nonfinite Precision/Recall preserved'))
    for curve_row in pd.read_csv(paper/'neighbour_curves.csv').to_dict('records'):
        coverage.append(row(11,'paper_aligned_v1',arm,paper/'neighbour_curves.csv',subitem=curve_row['marker']+':'+str(curve_row['bin']),n=int(curve_row['n_eligible']),n_evaluable=int(curve_row['n_eligible']),status='computed' if curve_row['status']=='computed' else 'partial_domain_NA',reason=curve_row['status']))
    for rec in pd.read_csv(paper/'marker_summary.csv').query('cell_type == "__all__"').to_dict('records'):
        raw.append(dict(method=arm,profile='paper_aligned_v1',metric_id='8' if rec['metric'].startswith('positive') else '9',metric_name=rec['metric'],value=rec['all_cell_mean'],n_eligible=rec['n_total'],n_finite=rec['n_finite'],status='computed'))
    for rec in pd.read_csv(paper/'type_correlations.csv').to_dict('records'):raw.append(dict(method=arm,profile='paper_aligned_v1',metric_id='6' if rec['definition']=='mean_logcounts' else '7',metric_name=rec['definition']+':'+rec['cell_type'],value=rec['pearson'],n_eligible=rec['n_cells'],status=rec['status']))
    for name in ['qc_retention','unassigned_fraction']:raw.append(dict(method=arm,profile='paper_aligned_v1',metric_id=name,metric_name=name,value=summary[name],status='computed'))
    for i,status in SCOPE.items():coverage.append(row(i,'all',arm,queue['paper_protocol'],status=status,required=False,reason='prespecified scope; area raw distribution supplied for #19'))
    for name,file in [('gene_detection',paper/'gene_detection_313.csv'),('type_composition',paper/'type_composition.csv'),('raster_morphology',root/'engineering/structure/shape_raster_distribution.csv'),('nuclear_matching',root/'engineering/structure/cells_distribution.csv'),('spatial_distribution',paper/'own_cell_centroids.csv')]:coverage.append(row('Tier1B:'+name,'supplementary',arm,file))
    for name in ['polygon_morphology','local_Shannon','local_CV']:coverage.append(row('Tier1B:'+name,'supplementary',arm,None,status='not_run_scope',required=False,reason='not frozen/implemented in accepted protocol'))
    for metric,file in [('COS:stage_ownership','stage_effects.csv'),('COS:stage_fixed_anchor','stage_fixed_anchor_summary.csv'),('COS:stage_connectivity','stage_connectivity.csv'),('COS:stage_marker_type_density','stage_marker_type_density_transitions.csv'),('COS:stage_boundary','stage_boundary_transitions.csv'),('COS:stage_gene','stage_gene_composition.csv')]:coverage.append(row(metric,diagnostics.VERSION,arm,root/'diagnostics/stages'/file))
    coverage.append(row('COS:frozen_probe',diagnostics.VERSION,arm,root/'diagnostics/probes/probe_strata.csv'))
    if include_resources:
        cost_source=Path(cost_snapshot) if cost_snapshot else Path(queue['output_root'])/'run_status.json'
        run=read(cost_source);records=run['stages'];costs=[]
        for record in records.values():
            if record.get('method')==arm:
                for attempt in record.get('attempts',[]):costs.append(dict(method=arm,stage=record['stage'],**attempt))
        tr=read(Path(spec['cost_root'])/'checkpoints'/arm/'complete.json');pr=read(Path(spec['cost_root'])/'predictions'/arm/'complete.json')
        timing={stage:(sum(float(a['wall_seconds']) for a in costs if a['stage']==stage) if any(a['stage']==stage and a.get('wall_seconds') is not None for a in costs) else None) for stage in ('train','predict','postprocess')}
        values={31:timing['train'],32:timing['predict'],33:timing['postprocess'],34:max([v for v in [tr.get('peak_vram_bytes'),pr.get('peak_vram_bytes')] if v is not None],default=None),35:max([a['sampled_peak_rss_bytes_sum'] for a in costs if a.get('sampled_peak_rss_bytes_sum') is not None],default=None),36:tr.get('parameters',tr.get('trainable_parameters',tr.get('parameter_count'))),37:tr.get('actual_updates',tr.get('updates',0))/timing['train'] if timing['train'] is not None and timing['train']>0 else None}
        for i,value in values.items():
            coverage.append(row(i,'resources',arm,cost_source,status='computed' if value is not None else 'failed',reason='population denominator not applicable to measured resource cost; '+ '; '.join(sorted(set(a.get('memory_sampling_scope','scope not recorded') for a in costs)))))
            raw.append(dict(method=arm,profile='resources',metric_id=str(i),metric_name='resource_'+str(i),value=value,status='computed' if value is not None else 'failed'))
    return coverage,raw


def validate_arm(queue,arm):
    spec=queue['methods'][arm];root=Path(spec['cost_root']);receipts={}
    paths={'train':root/'checkpoints'/arm/'complete.json','predict':root/'predictions'/arm/'complete.json','postprocess':root/'predictions'/arm/'postprocess/complete.json','matrix':Path(spec['matrix'])/'complete.json','evaluate':Path(spec['evaluation'])/'complete.json'}
    for stage,path in paths.items():
        rec=read(path)
        if rec.get('status')!='succeeded':raise ValueError(arm+' '+stage+' incomplete')
        verify_outputs(rec);receipts[stage]=identity(path)
        if stage=='train':receipts['checkpoint']=rec.get('outputs',{}).get('checkpoint')
        if stage=='train':
            updates=rec.get('actual_updates',rec.get('updates'))
            expected=read(spec['config']).get('actual_updates',read(spec['config']).get('updates',4000))
            if updates!=expected:raise ValueError('actual update budget mismatch')
        if stage=='evaluate' and (not rec.get('paper_method_complete') or rec.get('profiles')!=PROFILES):raise ValueError('three profiles incomplete')
    machine=read(spec['machine']);receipts.update(config=identity(spec['config']),final_mask=identity(machine['final_mask']),matrix_counts=identity(Path(spec['matrix'])/'counts.npy'),commit=spec['commit'],actual_updates=updates)
    return receipts


def aggregate(config,comparison):
    from .comparison import aggregate as build
    return build(config)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--comparison',choices=['SSP']);args=parser.parse_args()
    if args.comparison:
        if not aggregate(args.config,args.comparison):raise SystemExit(2)
    else:evaluate(args.config)



def run_profiles(spec,queue,original_protocol,out,arm):
    """Scientific computation uses accepted functions; empty engineering groups are explicit."""
    out=Path(out);machine=read(spec['machine']);p=read(original_protocol);prep=read(Path(queue['prepared'])/'complete.json')
    verify_outputs(prep)
    for key in ['reference','official_processing','official_expression','rscript']:
        if sha256(p[key])!=prep['dependencies'][key]['sha256']:raise ValueError('prepared source changed '+key)
    if prep['protocol_version']!=p['version'] or prep['dependencies']['protocol']['sha256']!=sha256(original_protocol):raise ValueError('accepted reference protocol mismatch')
    accepted.validate_matrix(machine,spec['matrix'])
    if sha256(machine['final_matrix'])!=sha256(Path(spec['matrix'])/'expr_mat.csv'):raise ValueError('native CSV differs from own matrix')
    atomic_json(out/'complete.json',dict(status='running',paper_method_complete=False))
    accepted.engineering_evaluate(spec['machine'],queue['reference'],queue['engineering_protocol'],spec['matrix'],out/'engineering',profile='EP-CORE')
    eng=read(out/'engineering/complete.json');verify_outputs(eng);failures=dict(eng.get('partial_components',{}))
    coords=pd.read_csv(out/'engineering/structure/annotation_nuclear.csv').set_index('cell_id')
    ids=pd.read_csv(Path(spec['matrix'])/'cell_mapping.csv').canonical_cell_id.to_numpy(int);genes=pd.read_csv(Path(spec['matrix'])/'gene_mapping.csv').gene.tolist();counts=np.load(Path(spec['matrix'])/'counts.npy',mmap_mode='r')
    for old,profile in accepted.PROFILES.items():
        dest=out/'engineering/cellspa'/old;labels=pd.read_csv(dest/'official_markers_per_cell.csv').set_index('cell_id')
        if not np.array_equal(labels.index.to_numpy(),ids):raise ValueError('engineering own-cell order mismatch')
        settings=dict(p['neighbour'],source_type='B',target_types=['CD4Tconv/Treg','CD8T/CD8Tex'])
        values={g:counts[:,genes.index(g)] for g in settings['markers']}
        curve,_=accepted.neighbour_curve(coords.loc[ids,['x_um','y_um']],labels.cell_type,values,settings)
        curve.to_csv(dest/'curve_with_denominators.csv',index=False,na_rep='NA')
        failure_key=old+':curve_failed.txt'
        if failure_key in failures:
            if not set(curve.status)<= {'empty_source_type','empty_target_type'}:raise ValueError('nonempty engineering curve failed: '+failures[failure_key])
            atomic_json(dest/'empty_curve_domain.json',dict(status='empty_evaluable_domain',reason=failures.pop(failure_key),n_source=int((labels.cell_type=='B').sum()),n_target=int(labels.cell_type.isin(settings['target_types']).sum())))
            # Retain original R error file as evidence. The full denominator-aware
            # table is the accepted output for this explicitly empty population.
            if not (dest/'official_B_to_T_curve.csv').exists():pd.DataFrame({g:[np.nan] for g in settings['markers']}).to_csv(dest/'official_B_to_T_curve.csv',index=False)
    if failures:raise ValueError('required engineering components failed: '+str(failures))
    eng['partial_components']={};eng['empty_domain_handling']='explicit empty source/target bins; preserved original R errors';eng['outputs']={str(f.relative_to(out/'engineering')):identity(f) for f in (out/'engineering').rglob('*') if f.is_file() and f!=out/'engineering/complete.json'};atomic_json(out/'engineering/complete.json',eng)
    paper=out/'paper_aligned_v1';accepted.checked_r(p,'paper_evaluation.R',[original_protocol,queue['prepared'],machine['final_matrix'],paper],paper);accepted.paper_spatial(machine,paper,p)
    qc=pd.read_csv(paper/'qc_all_cells.csv');markers=pd.read_csv(paper/'markers_per_cell.csv');ann=pd.read_csv(paper/'annotation.csv')
    if set(qc.loc[qc.qc_pass,'cell_id'])!=set(markers.cell_id) or set(markers.cell_id)!=set(ann.cell_id):raise ValueError('paper QC/marker/annotation ID mismatch')
    if not np.isfinite(markers[['positive_F1','negative_F1']]).all().all():raise ValueError('nonfinite paper F1')
    return dict(method=arm,n_before_qc=len(qc),n_after_qc=len(markers),qc_retention=len(markers)/len(qc) if len(qc) else None,n_unassigned=int(ann.rejected.sum()),unassigned_fraction=float(ann.rejected.mean()) if len(ann) else None,positive_F1=float(markers.positive_F1.mean()) if len(markers) else None,negative_F1=float(markers.negative_F1.mean()) if len(markers) else None,protocol_version=p['version'],model_sha256=prep['model']['sha256'])


if __name__=='__main__':main()
