"""Read-only old-snapshot semantic validation, launched with five inherited locks."""
from pathlib import Path
import json
import runpy
import sys


def main():
    root=Path(sys.argv[1]).resolve(strict=True);fds=tuple(map(int,sys.argv[2].split(',')))
    expected=Path('/home/zwan0726/work/outputs/bidcell_rep1_bt_gt_fusion_20261016/rep1-bt-gt-fusion-20261016-r01')
    if root!=expected:raise ValueError('wrong legacy run')
    runpy.run_path(str(root/'source_snapshot/scripts/rep1_bt_gt_fusion/bootstrap.py'),run_name='legacy_recovery_bootstrap')
    from scripts.rep1_bt_gt_fusion import execution as ex, runtime as rt
    from scripts.rep1_bt_gt_fusion.common import read,identity,digest,dependencies,check_receipt
    from scripts.rep1_bt_gt_fusion.science import validate_stage
    from scripts.rep1_bt_gt_fusion.inputs import load_inputs
    from scripts.rep1_bt_gt_fusion.gt_refine import REASONS
    import numpy as np
    import tifffile
    cfg=rt.load_config(root/'configs/run.json');ex.install_import_guard(cfg)
    rt.check_fds(cfg,fds);rt.verify_execution(cfg,fds,True);manifest=load_inputs(cfg)
    validate_stage(cfg,'BT_EXISTING6',manifest)
    state=read(root/'state.json')
    if state['status']!='FAILED' or rt.alive(state.get('supervisor')) or rt.alive(state.get('worker')):
        raise ValueError('legacy queue is not an inactive failed producer')
    if identity(root/'bt_existing6/complete.json')!=state['nodes']['BT_EXISTING6']['receipt']:
        raise ValueError('legacy BT receipt/state mismatch')
    gt=root/'endpoints/P2_BASE_GT/gt';base=dependencies(cfg,'P2_BASE_GT_GT');parent=manifest['endpoints']['P2_BASE']
    check_receipt(gt/'domain/complete.json',dict(base,parent=manifest['files'][str(Path(parent['final']).resolve())]),root)
    check_receipt(gt/'scores/complete.json',dict(base,domain=identity(gt/'domain/complete.json')),root)
    r=check_receipt(gt/'recovery/state.json',dict(base,proposals=identity(gt/'scores/proposals.npy')),root)
    if identity(gt/'recovery/state.json')!=state['committed_witness']:raise ValueError('legacy GT witness drift')
    for k in ('mask','decisions'):
        if r[k]!=r['outputs'][k]:raise ValueError('legacy recovery output drift')
    p=np.load(gt/'scores/proposals.npy',mmap_mode='r');d=np.load(r['decisions']['path'],mmap_mode='r');m=np.load(r['mask']['path'],mmap_mode='r')
    with tifffile.TiffFile(parent['final']) as t:shape=t.series[0].shape
    with tifffile.TiffFile(parent['nuclei']) as t:nshape=t.series[0].shape
    if r['cursor']!=196771 or len(p)!=196771 or len(d)!=196771 or m.shape!=shape or shape!=nshape or m.dtype!=np.dtype('uint32'):
        raise ValueError('legacy dimensions/cursor mismatch')
    y,x=p['y'],p['x']
    if not np.all((d>0)&(d<len(REASONS))) or not np.all((y>=0)&(y<shape[0])&(x>=0)&(x<shape[1])):
        raise ValueError('invalid legacy proposal/decision')
    if len(np.unique(y.astype(np.int64)*shape[1]+x))!=len(p) or not np.array_equal(np.lexsort((p['target'],x,y,-p['margin'])),np.arange(len(p))):
        raise ValueError('legacy proposal order drift')
    if not np.all(p['margin']>=np.log(2)) or not np.array_equal(m[y,x],np.where(d==1,p['target'],p['source'])):
        raise ValueError('legacy proposals disagree with recovered mask')
    print(json.dumps(dict(status='PASS_LEGACY_CHECKPOINTS',source_commit=cfg['source_commit'],source_digest=cfg['source_digest'],
        config_digest=digest(cfg),input_digest=cfg['input_digest'],environment_digest=cfg['environment_digest'],
        old_snapshot_admission_environment_inputs_verified=True,inherited_locks_verified=True,
        cursor=r['cursor'],proposals_n=len(p),decision_counts={n:int(np.count_nonzero(d==i)) for i,n in enumerate(REASONS)},
        read_only=True,new_computation_claim=False,full_topology_validation_still_required=True)))


if __name__=='__main__':main()
