"""Reuse historical implementations; beta is the only scientific code change."""
from pathlib import Path
import os,time
import numpy as np
import pandas as pd
import tifffile
from scipy import sparse
from scripts.rep1_triple_simple_boundary import gt,matrix,evaluation
from scripts.rep1_triple_simple_boundary.common import atomic_json,atomic_csv,atomic_tiff,identity,read,digest,check_identity
from scripts.rep1_triple_simple_boundary.inputs import matrix_arrays
from scripts.boundary_three.core_refine import score_proposals
from .config import beta_spec,NAME

def replay(cfg,out,beta,progress,smoke=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    old=cfg['old_config'];inputs=cfg['inputs'];data=gt.load_data(old,inputs,'P2_MORPH_GT')
    legacy=Path(cfg['specs']['P2_MORPH_GT']['root'])/'gt'
    domain=dict(np.load(legacy/'domain/domain.npz'));expression=sparse.load_npz(legacy/'domain/expression.npz')
    deps=dict(run_id=cfg['run_id'],source=cfg['source_digest'],beta=float(beta),domain=identity(legacy/'domain/complete.json'),parent=identity(data['spec']['final']))
    if smoke:
        size=min(256,len(domain['pixels']));end=domain['offsets'][size]
        p,s,n=score_proposals(data['baseline'],domain['pixels'][:size],domain['offsets'][:size+1],domain['candidate_rows'][:end],domain['geometry'][:end],expression[:size],data['ids'],data['prior'],beta=1.0)
        oldscore=np.load(legacy/'scores/candidate_scores.npy',mmap_mode='r')[:end]
        if not np.array_equal(s,oldscore):raise ValueError('small patch beta1 score regression failed')
        oldproposals=np.load(legacy/'scores/proposals.npy',mmap_mode='r');first=oldproposals[:64].copy()
        mask,summary=gt.apply_gt(data['baseline'],data['nuclei'],first,out/'topology',dict(deps,kind='smoke'),progress)
        decisions=np.load(read(out/'topology/state.json')['decisions']['path'])
        historical=np.load(read(legacy/'recovery/state.json')['decisions']['path'],mmap_mode='r')[:len(first)]
        if not np.array_equal(decisions,historical):raise ValueError('small global-prefix topology decisions differ')
        atomic_json(out/'smoke.json',dict(status='PASS',score_pixels=size,candidate_edges=int(end),topology_global_prefix=len(first),beta1_discrete_equal=True))
        return
    proposals,score=gt.score_domain(data,domain,expression,out/'scores',deps,progress,beta=beta)
    mask,application=gt.apply_gt(data['baseline'],data['nuclei'],proposals,out/'recovery',dict(deps,proposals=identity(out/'scores/proposals.npy')),progress)
    validation=gt.validate_topology(data['baseline'],mask,data['nuclei'],data['ids'],proposals,out/'validation',deps,progress)
    atomic_tiff(out/'final.tif',mask.astype(np.uint32))
    decisions=np.load(read(out/'recovery/state.json')['decisions']['path'])
    trans=pd.DataFrame.from_records(proposals);trans['decision']=decisions;trans['accepted']=decisions==1
    atomic_csv(out/'pixel_changes.csv',trans)
    if beta==1:
        comparisons=dict(mask_equal=np.array_equal(mask,tifffile.imread(cfg['specs']['P2_MORPH_GT']['final'])),proposals_equal=np.array_equal(proposals,np.load(legacy/'scores/proposals.npy')),candidate_scores_equal=np.array_equal(np.load(out/'scores/candidate_scores.npy'),np.load(legacy/'scores/candidate_scores.npy')),decisions_equal=np.array_equal(decisions,np.load(read(legacy/'recovery/state.json')['decisions']['path'])))
        # Owners are a deterministic lookup from discrete exact mask on locked coordinates.
        for i,rec in enumerate(inputs['coordinates']):
            co=np.load(rec['path'],mmap_mode='r');good=co['x']>=0;owner=np.zeros(len(co),np.uint32);owner[good]=mask[co['y'][good],co['x'][good]]
            historical=np.load(Path(cfg['specs']['P2_MORPH_GT']['matrix'])/f'owners_{i:04d}.npy',mmap_mode='r')
            if not np.array_equal(owner,historical):comparisons['RNA_owners_equal']=False;break
        else:comparisons['RNA_owners_equal']=True
        atomic_json(out/'REGRESSION.json',dict(status='PASS' if all(comparisons.values()) else 'FAIL',**comparisons,scoring=score,application=application))
        if not all(comparisons.values()):raise ValueError('full beta1 exact regression failed')
    atomic_json(out/'complete.json',dict(status='succeeded',beta=beta,scoring=score,application=application,validation=validation,final=identity(out/'final.tif'),reference_gates_unchanged=True))

def build_matrix(cfg,out,progress):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);spec=beta_spec(cfg);parent=cfg['specs']['P2_MORPH']
    machine=dict(read(parent['machine']));machine.update(final_mask=spec['final'],final_matrix=str(out/'matrix/expr_mat.csv'),pre_mask=parent['final'],production_kind='reference-gated geometry-only scoring',parent_endpoint='P2_MORPH')
    atomic_json(out/'machine.json',machine)
    deps=dict(run_id=cfg['run_id'],source=cfg['source_digest'],beta=0.0,final=identity(spec['final']))
    result=matrix.build(machine,cfg['inputs']['coordinates'],out/'matrix',deps,progress)
    conservation=matrix.check_parent_conservation(parent,result,out/'matrix')
    atomic_json(out/'conservation.json',conservation)
    atomic_json(out/'complete.json',dict(status='succeeded',matrix=identity(out/'matrix/complete.json'),conservation=conservation))

def evaluate(cfg,out,progress,lock_fds,validator):
    out=Path(out);spec=beta_spec(cfg);dest=Path(spec['evaluation']);dest.mkdir(parents=True,exist_ok=True)
    ecfg=dict(cfg['old_config'],source_snapshot=cfg['source_snapshot'],evaluation_threads=1)
    deps=dict(run_id=cfg['run_id'],source=cfg['source_digest'],matrix=identity(Path(spec['matrix'])/'complete.json'))
    runtime=evaluation.source_runtime(ecfg,deps);atomic_json(dest/'r_runtime.json',runtime)
    sr=evaluation.structural(spec,cfg['inputs'],dest/'structure',dict(deps,part='structure'),progress)
    props=pd.read_csv(dest/'structure/cells_distribution.csv')[['cell_id','x_um','y_um']]
    paper=dest/'paper_aligned_v1';paper.mkdir(exist_ok=True);atomic_csv(paper/'own_cell_centroids.csv',props)
    if not (paper/'execution_status.txt').exists() or (paper/'execution_status.txt').read_text().strip()!='succeeded':
        evaluation.run_r(ecfg,Path(cfg['source_snapshot'])/'scripts/rep1_triple_simple_boundary/paper_evaluation.R',[cfg['inputs']['evaluation']['protocol'],cfg['inputs']['evaluation']['prepared'],read(spec['machine'])['final_matrix'],paper,dest/'r_runtime.json'],paper,lock_fds,validator,progress,digest(deps))
    fixed=evaluation.fixed_panel(spec,cfg['inputs'],dest/'fixed')
    ids,genes,counts=matrix_arrays(spec);qc=(counts.sum(1)>=20)&(counts.sum(1)<=2000)&((counts>0).sum(1)>=20)
    sc=pd.read_csv(spec['sc_scores']);qcframe=pd.read_csv(paper/'qc_all_cells.csv')
    if not np.array_equal(qcframe.cell_id,ids) or not np.array_equal(qcframe.qc_pass,qc) or not np.array_equal(sc.cell_id,ids[qc]):raise ValueError('new complete evaluation QC population mismatch')
    summary=dict(status='succeeded',method=NAME,structure=sr['summary'],fixed=fixed,n_cells=len(ids),n_qc=int(qc.sum()),n_qc_rejected=int((~qc).sum()),capture=read(Path(spec['matrix'])/'complete.json')['totals'])
    for profile,path in [('fixed',spec['fixed_scores']),('scclassify',spec['sc_scores'])]:
        scores=pd.read_csv(path)
        for metric in ('positive_F1','negative_F1'):
            v=scores[metric].to_numpy(float);ok=np.isfinite(v);summary[profile+'_'+metric]=float(v[ok].mean()) if ok.any() else None;summary[profile+'_'+metric+'_n']=int(ok.sum())
    atomic_json(out/'complete.json',summary)
