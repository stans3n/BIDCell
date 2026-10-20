"""Verified accepted three-profile computation, copied without scientific edits."""
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.crgf import evaluate_egr as accepted
from .common import read,verify_outputs,sha256,atomic_json,identity

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
