"""Small real evaluator and process-lifecycle rehearsals."""
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from scripts.crgf.common import atomic_json,atomic_tiff,identity
from scripts.crgf.matrix import build
from scripts.crgf.evaluate import evaluate
from scripts.crgf import structural
from scripts.crgf.config import read


def output_fixture(cfg,out):
    out.mkdir(parents=True,exist_ok=True);native=cfg.native();base=Path(native.files.data_dir)
    with h5py.File(base/'nuclei_cell_type.h5') as f:
        ids=f['ids'][:];types=f['data'][:]
    chosen=[int(ids[np.where(types==t)[0][0]]) for t in (0,1,-1)]
    mask=np.zeros((32,36),np.uint32);nuc=np.zeros_like(mask)
    locations=[(3,3),(3,20),(20,3)]
    for i,(y,x) in zip(chosen,locations):mask[y:y+6,x:x+6]=i;nuc[y+2:y+4,x+2:x+4]=i
    atomic_tiff(out/'final.tif',mask);atomic_tiff(out/'pre.tif',mask);atomic_tiff(out/'nuclei.tif',nuc)
    genes=(base/'all_gene_names.txt').read_text().splitlines();(out/'genes.txt').write_text('\n'.join(genes)+'\n')
    rows=[];rng=np.random.RandomState(7)
    for j,(y,x) in enumerate(locations):
        for k in range(500):rows.append((len(rows)+1,x+2.1,y+2.1,genes[rng.randint(len(genes))],30))
    pd.DataFrame(rows,columns=['transcript_id','x_location','y_location','feature_name','qv']).to_csv(out/'transcripts.csv',index=False)
    c=json.loads((Path(cfg.output_root)/'configs/B0_machine.json').read_text())
    c.update(final_mask=str(out/'final.tif'),pre_mask=str(out/'pre.tif'),nuclei_mask=str(out/'nuclei.tif'),transcripts=str(out/'transcripts.csv'),genes=str(out/'genes.txt'),final_matrix=str(out/'matrix/expr_mat.csv'),baseline_final_mask=str(out/'pre.tif'),chunksize=400)
    atomic_json(out/'machine.json',c);build(c,out/'matrix',cache=out/'cache')
    evaluate(out/'machine.json',Path(cfg.output_root)/'configs/reference.json',Path(cfg.output_root)/'configs/protocol.json',out/'matrix',out/'evaluation')
    manifest=json.loads((out/'evaluation/complete.json').read_text())
    assert not manifest['paper_protocol_complete']
    for profile in ('remote_nuclear_annotation_raw','windows_style_maxpearson_raw'):
        p=out/'evaluation/cellspa'/profile/'official_marker_summary.csv'
        assert p.exists(),f'missing profile {profile}'
    remote=pd.read_csv(out/'evaluation/cellspa/remote_nuclear_annotation_raw/official_markers_per_cell.csv')
    unknown=remote[remote.cell_id==chosen[-1]]
    assert unknown.filter(regex='positive_|negative_').isna().all().all()
    original=pd.read_csv(out/'evaluation/structure/shape_raster_distribution.csv')
    # Same cell ID and area, different shape; change actual mask and check cache rejection.
    mask[3:9,3:9]=0;mask[3:6,3:15]=chosen[0];atomic_tiff(out/'final.tif',mask)
    rejected=False
    try:structural.evaluate(c,out/'matrix',out/'evaluation/structure')
    except ValueError as error:
        assert 'dependency mismatch' in str(error);rejected=True
    assert rejected
    # Rebuild counts because the mask changed, then independently recompute morphology.
    build(c,out/'matrix_changed',cache=out/'cache')
    structural.evaluate(c,out/'matrix_changed',out/'shape_changed')
    changed=pd.read_csv(out/'shape_changed/shape_raster_distribution.csv')
    a=original.set_index('cell_id').loc[chosen[0]];b=changed.set_index('cell_id').loc[chosen[0]]
    assert a.area_um2==b.area_um2 and a.perimeter_px!=b.perimeter_px
    atomic_json(out/'receipt.json',dict(status='succeeded',two_profiles=True,unknown_NA=True,
                independent_matrix_accepted=True,own_shape_cache_rejected_changed_mask=True,
                equal_area_changed_perimeter=[float(a.perimeter_px),float(b.perimeter_px)],evaluation=identity(out/'evaluation/complete.json')))


def lifecycle(out):
    out.mkdir(parents=True,exist_ok=True);lockpath=out/'test.lock';pidpath=out/'child.pid'
    code="""import fcntl,subprocess,sys,time
from pathlib import Path
lock=open(sys.argv[1],'a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)'],pass_fds=(lock.fileno(),),start_new_session=True)
Path(sys.argv[2]).write_text(str(p.pid))
time.sleep(30)
"""
    parent=subprocess.Popen([sys.executable,'-c',code,str(lockpath),str(pidpath)])
    for _ in range(100):
        if pidpath.exists():break
        time.sleep(.05)
    child=int(pidpath.read_text());os.kill(parent.pid,signal.SIGKILL);parent.wait()
    lock=lockpath.open('a');held=False
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:held=True
    assert held,'orphan stage must retain runner flock'
    os.kill(child,signal.SIGTERM)
    for _ in range(100):
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
        except BlockingIOError:time.sleep(.05)
    else:raise AssertionError('lock did not release after child exit')
    # Execute actual runner failure path against a controlled stage subprocess.
    from scripts.crgf import run_stage1 as runner
    from scripts.crgf.config import RunConfig
    runroot=out/'runner_rehearsal';runroot.mkdir(exist_ok=True)
    cfg=RunConfig(run_id='temporary_lifecycle',output_root=str(runroot),baseline_config='/unused',checkpoint='/unused',stage_timeouts={'report':30},minimum_free_gb=0)
    config=runroot/'config.json';atomic_json(config,cfg.model_dump())
    runner.verify_preflight=lambda cfg:None
    runner.signature=lambda *args:'fixture-signature'
    runner.stage_specs=lambda root:[('fixture_failure',None,'report',[],root/'never.json')]
    original_popen=runner.subprocess.Popen
    def failing_popen(cmd,**kwargs):return original_popen([sys.executable,'-c','raise RuntimeError("injected stage failure")'],**kwargs)
    runner.subprocess.Popen=failing_popen
    try:rc=runner.run(str(config))
    finally:runner.subprocess.Popen=original_popen
    state=json.loads((runroot/'run_status.json').read_text())
    assert rc==1 and state['status']=='partial_failed' and state['stages']['fixture_failure']['exit_code']!=0
    assert len(state['stages']['fixture_failure']['attempts'])==1
    # A duplicate lock fails before any status write.
    heldlock=(runroot/'runner.lock').open('a');fcntl.flock(heldlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    before=(runroot/'run_status.json').read_bytes()
    try:runner.run(str(config))
    except BlockingIOError:pass
    else:raise AssertionError('duplicate writer accepted')
    assert (runroot/'run_status.json').read_bytes()==before
    atomic_json(out/'receipt.json',dict(status='succeeded',inherited_lock_survives_parent_kill=True,
                duplicate_writer_status_unchanged=True,subprocess_exception_propagated=True,no_retry_for_code_error=True))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--part',choices=['output','lifecycle'],required=True);a=p.parse_args();out=Path(a.out)
    if a.part=='output':output_fixture(read(a.config),out)
    else:lifecycle(out)
    print('PASSED',a.part,flush=True)
if __name__=='__main__':main()
