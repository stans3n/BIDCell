"""Serial persistent controller. Successful receipts precede completion only after validation."""
import argparse
import os
import subprocess
import sys
import time
import traceback
from .common import *

def completed(stage):
    path=ROOT/'receipts'/(stage+'.json')
    if not path.exists():return False
    rec=check_receipt(path,deps(stage))
    # Shared inference manifests refer to read-only old patches as well as new ones.
    if stage=='shared':
        for shift in (0,24):
            for v in read(ROOT/f'shared/predict_shift{shift}.json'):check_identity(v['identity'])
        for r in coordinates():check_identity(r)
    return rec

def acceptance():
    from scripts.rep1_triple_simple_boundary.inputs import matrix_arrays
    import numpy as np
    import tifffile
    for stage in STAGES[:-1]:
        if not completed(stage):raise ValueError('missing accepted stage '+stage)
    nucleus=tifffile.imread(TRAINING/'shared/data/nuclei.tif');meta=np.load(TRAINING/'shared/prepared/nucleus_metadata.npz');expected=meta['ids']
    parent=tifffile.imread(spec(P2)['final']);pids,pgenes,pc=matrix_arrays(spec(P2))
    for name in ENDPOINTS:
        mask=tifffile.imread(spec(name)['final']);ids,genes,counts=matrix_arrays(spec(name))
        if mask.shape!=nucleus.shape or mask.dtype!=np.uint32 or not np.array_equal(mask[nucleus>0],nucleus[nucleus>0]) or not np.array_equal(ids,expected) or len(genes)!=313:raise ValueError('final endpoint ID/nuclear/panel invariant '+name)
        r=read(Path(spec(name)['matrix'])/'complete.json')
        if r['totals']['eligible']!=config()['eligible'] or counts.sum()!=r['totals']['assigned']:raise ValueError('complete molecule ledger conservation '+name)
        if name in (B0,BETA1) and (not np.array_equal(mask>0,parent>0) or not np.array_equal(pc.sum(0),counts.sum(0))):raise ValueError('same-parent foreground/per-gene conservation '+name)
    summary=dict(status='succeeded',endpoints=NAMES,training_reused=True,training=identity(TRAINING/'training_complete.json'),
        statistics=identity(ROOT/'statistics/validation.json'),labels=identity(ROOT/'receipts/labels.json'),ROI=identity(ROOT/'receipts/roi.json'),
        method_reference=identity(ROOT/'method_reference/decision.json'),source_digest=config()['source_digest'],interpretation='WAITING_FOR_MANUAL_RESUME',context_or_manuscript_modified=False)
    out=ROOT/'acceptance.json';atomic_json(out,summary)
    return {'acceptance':identity(out)},summary

def stage_run(stage):
    inherited_fds();review_binding();resources(stage)
    if completed(stage):return
    from . import production,freeze,evaluation
    if stage=='shared':outputs,meta=production.shared();outputs.update({f'views{shift}':identity(ROOT/f'shared/predict_shift{shift}.json') for shift in (0,24)})
    elif stage=='inference_capacity':
        from .smoke import capacity
        result=capacity();outputs={'capacity':identity(ROOT/'shared/inference_capacity.json')};meta=result
    elif stage.endswith('_predict'):outputs,meta=production.predict(stage[:-8])
    elif stage.endswith('_cleanup'):outputs,meta=production.cleanup(stage[:-8])
    elif stage.endswith('_matrix'):outputs,meta=production.matrix(stage[:-7])
    elif stage=='B1_annotate':outputs,meta=evaluation.annotation(B1)
    elif stage=='labels':outputs,meta=freeze.labels()
    elif stage=='roi':outputs,meta=freeze.roi()
    elif stage=='P2':outputs,meta=production.p2()
    elif stage=='reference':outputs,meta=freeze.reference()
    elif stage=='domain':outputs,meta=freeze.domain()
    elif stage in ('beta0','beta1'):outputs,meta=freeze.refine(int(stage[-1]))
    elif stage.endswith('_evaluate'):outputs,meta=evaluation.evaluate(stage[:-9])
    elif stage=='statistics':outputs,meta=evaluation.statistics()
    elif stage=='acceptance':outputs,meta=acceptance()
    else:raise ValueError('unsupported stage; no train/evaluate expansion '+stage)
    if not outputs:raise ValueError('stage has no accepted artifacts')
    validate_outputs(outputs)
    write_receipt(ROOT/'receipts'/(stage+'.json'),deps(stage),outputs,acceptance=meta)
    progress(stage,status='succeeded',receipt=str(ROOT/'receipts'/(stage+'.json')))

def status():
    path=ROOT/'status.json';print(__import__('json').dumps(read(path) if path.exists() else {'status':'not_started'},indent=2))

def run():
    with locks() as fds:
        preflight();atomic_json(ROOT/'status.json',dict(status='running',pid=os.getpid(),child_pid=None,started=time.time(),completed=0,total=len(STAGES)))
        for i,stage in enumerate(STAGES):
            if completed(stage):
                progress(stage,status='validated_existing',completed=i+1,total=len(STAGES));continue
            if (ROOT/'STOP_AFTER_STAGE').exists():
                atomic_json(ROOT/'status.json',dict(status='stopped_at_validated_stage_boundary',pid=os.getpid(),completed=i,total=len(STAGES)));return
            resources(stage);env=dict(os.environ,REP2_FULL_LOCK_FDS=','.join(map(str,fds)))
            with (ROOT/'logs'/(stage+'.log')).open('a',buffering=1) as log:
                proc=subprocess.Popen([sys.executable,'-B','-m','scripts.rep2_full.runner','stage',stage],cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,pass_fds=fds)
                while proc.poll() is None:
                    current=dict(status='running',pid=os.getpid(),child_pid=proc.pid,stage=stage,completed=i,total=len(STAGES),heartbeat=time.time())
                    atomic_json(ROOT/'status.json',current)
                    print(f'heartbeat stage={stage} child={proc.pid} time={time.time():.0f}',flush=True)
                    time.sleep(10)
                if proc.returncode!=0:raise RuntimeError(f'{stage} failed exit={proc.returncode}; inspect logs/{stage}.log; no dependent stage launched')
            if not completed(stage):raise ValueError('child exit without verified stage receipt')
        receipt=dict(status='succeeded',scope='remaining Rep2 full compute only; interpretation deferred',completed=len(STAGES),total=len(STAGES),
            endpoints=NAMES,config=identity(ROOT/'configs/run.json'),source_digest=config()['source_digest'],
            receipts={s:identity(ROOT/'receipts'/(s+'.json')) for s in STAGES},acceptance=identity(ROOT/'acceptance.json'))
        atomic_json(ROOT/'compute_complete.json',receipt)
        atomic_json(ROOT/'status.json',dict(status='compute_complete',pid=os.getpid(),child_pid=None,completed=len(STAGES),total=len(STAGES),time=time.time()))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['run','stage','status','preflight','verify']);parser.add_argument('stage',nargs='?');args=parser.parse_args()
    try:
        if args.command=='run':run()
        elif args.command=='stage':stage_run(args.stage)
        elif args.command=='status':status()
        elif args.command=='preflight':print(preflight())
        else:
            for s in STAGES:
                if not completed(s):print('not_completed',s);break
                print('verified',s)
    except BaseException as exc:
        path=ROOT/'status.json'
        if args.command=='run' and path.exists() and read(path).get('pid')==os.getpid():
            state=read(path);state.update(status='failed',error=repr(exc),time=time.time());atomic_json(path,state)
        traceback.print_exc();raise
if __name__=='__main__':main()
