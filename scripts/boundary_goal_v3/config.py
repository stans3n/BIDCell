"""Single run manifest; all historical paths are read-only sources."""
from pathlib import Path
import os, shutil, subprocess
import numpy as np
import pandas as pd
from scripts.rep1_triple_simple_boundary.common import atomic_json, atomic_csv, read, identity, digest, check_identity
from scripts.rep1_triple_simple_boundary.inputs import new_spec
from scripts.rep1_triple_simple_boundary.continuation import read_data

WORK=Path('/home/zwan0726/work')
R05=WORK/'outputs/bidcell_rep1_triple_simple_boundary_20261017/rep1-triple-simple-boundary-20261017-r05'
BASE=WORK/'outputs/isbi2027_boundary_goal_v3'
NAME='G_SCORE_ONLY_REF_GATED'
REGISTRY=WORK/'outputs/bidcell_historical_dual_bt_gt_fusion_20261017'
STAGES=('preflight_and_lock','reuse_global_BIDCell_metrics','local_BIDCell_style','marker_aware_reassignment','local_pair_CI','beta1_full_regression','beta0_new_mask','beta0_RNA_matrix','beta0_complete_evaluate','beta0_local_matched','aggregate_and_validate','package_and_complete')
MARKERS={'CD4':2,'CD8A':2,'CD8B':2,'CD79A':1,'CD79B':1,'MS4A1':1}
STATES=('CONSISTENT','OPPOSITE','OTHER_OR_UNMATCHED','UNASSIGNED')
FATES=('FAVORABLE_PROXY','ADVERSE_PROXY','SAME_LINEAGE','THIRD_OR_UNKNOWN','UNASSIGNED','OTHER_GENE/INDETERMINATE')

def artifact(path):
    return Path(read_data(path))

def specs_and_inputs():
    old=read(R05/'configs/run.json'); inputs=read(old['input_manifest'])
    specs=dict(inputs['endpoints'])
    for name in inputs['new_order']:specs[name]=new_spec(old,name)
    specs['B0']=inputs['background']['B0']
    m=read(WORK/'outputs/bidcell_new_methods_20261013/configs/B1_machine.json')
    specs['B1']=dict(name='B1',machine=str(WORK/'outputs/bidcell_new_methods_20261013/configs/B1_machine.json'),final=m['final_mask'],nuclei=m['nuclei_mask'],matrix=str(Path(m['final_matrix']).parent),evaluation=str(WORK/'outputs/crgf_egr_20261008T121953+1100/reevaluation/B1'),fixed_scores=str(WORK/'outputs/bidcell_historical_dual_20261013/methods/B1/fixed/fixed_population.csv'),sc_scores=str(WORK/'outputs/bidcell_historical_dual_20261013/methods/B1/paper/markers_per_cell.csv'))
    return old,inputs,specs

def root_check(run):
    run=Path(run).absolute()
    if run.parent!=BASE or run.is_symlink() or any(p.is_symlink() for p in run.parents):raise ValueError('output must be a new direct nonsymlink child of '+str(BASE))
    return run

def prepare(run, repo):
    run=root_check(run);run.mkdir(parents=True,exist_ok=False)
    repo=Path(repo).resolve();old,inputs,specs=specs_and_inputs()
    for d in ('logs','receipts','artifacts','stages'): (run/d).mkdir()
    snapshot=run/'CODE_SNAPSHOT';snapshot.mkdir()
    names=subprocess.check_output(['git','ls-files'],cwd=repo,text=True).splitlines()
    names += [str(p.relative_to(repo)) for p in (repo/'scripts/boundary_goal_v3').glob('*') if p.is_file()]
    # Only source, never git metadata, neural checkpoints or experiment datasets.
    for rel in sorted(set(names)):
        p=repo/rel
        if p.suffix not in ('.py','.R','.sh','.json','.yaml','.yml','.toml','.cfg') or not p.is_file():continue
        q=snapshot/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    shutil.copytree(R05/'source_snapshot/external_sources',snapshot/'external_sources')
    code=[identity(p) for p in sorted(snapshot.rglob('*')) if p.is_file()]
    for p in snapshot.rglob('*'):
        if p.is_file():p.chmod(0o444)
    for p in sorted(snapshot.rglob('*'),reverse=True):
        if p.is_dir():p.chmod(0o555)
    snapshot.chmod(0o555)
    cfg=dict(run_id=run.name,run_dir=str(run),source_snapshot=str(snapshot),source_files=code,source_digest=digest(code),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),branch=subprocess.check_output(['git','branch','--show-current'],cwd=repo,text=True).strip(),worktree=str(repo),old_config=old,inputs=inputs,specs=specs,stage_order=list(STAGES),python=old['python'],rscript=old['rscript'],evaluation_threads=1,eligible=34442716,interface_expected=7541976,bootstrap_replicates=2000,seed=20261019,spatial_block_um=1000,disk_reserve_bytes=10*1024**3,ram_reserve_bytes=6*1024**3,phase_II_manual_resume_required=True)
    atomic_json(run/'RUN_MANIFEST.json',cfg)
    shutil.copy2(repo/'scripts/boundary_goal_v3/status.py',run/'status.py')
    (run/'check_status.sh').write_text('#!/usr/bin/env bash\nset -Eeuo pipefail\nexec /usr/bin/python3 "$(dirname "$(readlink -f "$0")")/status.py" "$@"\n')
    (run/'check_status.sh').chmod(0o755)
    wrapper='#!/usr/bin/env bash\nset -Eeuo pipefail\nexport PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1\ntrap \'task_rc=$?; /usr/bin/python3 SNAPSHOT/scripts/boundary_goal_v3/exit_receipt.py --run-dir RUN_DIR --exit-code "$task_rc"; exit "$task_rc"\' EXIT\ntrap \'printf "pipeline shell error at line %s (exit %s)\\n" "$LINENO" "$?" >&2\' ERR\ncd SNAPSHOT\nPYTHON -B -m scripts.boundary_goal_v3.pipeline --run-dir RUN_DIR "$@"\n'.replace('SNAPSHOT',str(snapshot)).replace('RUN_DIR',str(run)).replace('PYTHON -B',old['python']+' -B')
    (run/'run_boundary_goal_v3.sh').write_text(wrapper)
    (run/'run_boundary_goal_v3.sh').chmod(0o755)
    atomic_json(run/'STATUS.json',dict(run_id=run.name,state='QUEUED',stage=None,completed_stages=[],exit_code=None,pid=None))
    return cfg

def beta_spec(cfg):
    r=Path(cfg['run_dir'])/'stages/07/matrix';ep=r.parent
    return dict(name=NAME,root=str(ep),matrix=str(r),machine=str(ep/'machine.json'),final=str(Path(cfg['run_dir'])/'stages/06/final.tif'),evaluation=str(Path(cfg['run_dir'])/'stages/08/evaluation'),fixed_scores=str(Path(cfg['run_dir'])/'stages/08/evaluation/fixed/fixed_population.csv'),sc_scores=str(Path(cfg['run_dir'])/'stages/08/evaluation/paper_aligned_v1/markers_per_cell.csv'))
