"""One resolved deterministic stage; no model API dependency."""
import argparse
import json
import time
from pathlib import Path
from .config import read
from .common import atomic_json,identity,verify_outputs,sha256


def report(cfg):
    root=Path(cfg.output_root);results={};timings={}
    for method in ('F1','B1'):
        results[method]={}
        for stage,path in [('train',root/'checkpoints'/method/'complete.json'),('predict',root/'predictions'/method/'complete.json'),('postprocess',root/'predictions'/method/'postprocess/complete.json'),('matrix',root/'matrix'/method/'complete.json'),('evaluate',root/'evaluation'/method/'complete.json')]:
            results[method][stage]=json.loads(path.read_text()) if path.exists() else {'status':'not_completed'}
        results[method]['metrics_csv']=str(root/'evaluation'/method/'metrics.csv')
        pred=results[method]['predict'];timings[method]=pred.get('wall_seconds')
    ratio=timings['F1']/timings['B1'] if all(v is not None and v>0 for v in timings.values()) else None
    all_complete=all(v['status']=='succeeded' for stages in results.values() for k,v in stages.items() if isinstance(v,dict))
    result=dict(status='awaiting_user_analysis' if all_complete else 'partial_failed',paper_protocol_complete=False,methods=results,
        B0=json.loads((root/'configs/B0_machine.json').read_text()),paired_inference=dict(wall_seconds=timings,F1_over_B1=ratio,scope='measured stage execution including loading, geometry, patch receipts and stitching; retries retained in run_status'),
        analysis='Not performed; awaiting explicit user return. F1 vs B1 is the matched comparison.',
        limitations=['Rep1 within-tissue adaptation','same training reference','Unknown marker legacy indexing retained','no true owner labels','geometry/relation/capacity not separately ablated','historical B0 postprocess RNG not recorded'])
    atomic_json(root/'results.json',result)
    import pandas as pd
    frames=[]
    for method in ('F1','B1'):
        path=root/'evaluation'/method/'metrics.csv'
        if path.exists():
            frame=pd.read_csv(path);frame['method']=method
            idx=frame.metric_id.astype(str)=='38'
            frame.loc[idx,'value']=ratio if ratio is not None else float('nan')
            frame.loc[idx,'status']='computed_paired_timing' if ratio is not None else 'missing_input'
            frames.append(frame)
    b0=Path('/home/zwan0726/work/outputs/remote_prep_20261007T122955/evaluation')
    path=b0/'metrics.csv'
    if path.exists():
        receipt=json.loads((b0/'complete.json').read_text())
        baseline_machine_identity=receipt['configuration']['machine']
        if sha256(baseline_machine_identity['path'])!=baseline_machine_identity['sha256']:raise ValueError('B0 machine identity changed')
        baseline_machine=json.loads(Path(baseline_machine_identity['path']).read_text())
        if baseline_machine['final_mask']!=result['B0']['final_mask']:raise ValueError('B0 evaluated different final mask')
        recorded=receipt['outputs']['metrics.csv']
        if sha256(path)!=recorded['sha256']:raise ValueError('B0 metrics identity changed')
        frame=pd.read_csv(path);frame['method']='B0_historical_engineering';frame['source']=str(path)
        frame['original_metric_id']=frame.metric_id
        frame.loc[frame.metric_id.astype(str)=='30','metric_id']='engineering_pre_post'
        frames.append(frame)
        result['B0_metrics_evidence']=dict(metrics=identity(path),receipt=identity(b0/'complete.json'),configuration=receipt['configuration'])
    status_path=root/'run_status.json'
    result['execution_costs_and_failures']=json.loads(status_path.read_text()) if status_path.exists() else None
    result['output_bytes_at_report']=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    atomic_json(root/'results.json',result)
    if frames:pd.concat(frames,ignore_index=True).to_csv(root/'raw_comparison.csv',index=False,na_rep='NA')
    lines=['# CRGF stage 1 computation package','',f"Computation status: {result['status']}.",'',
           'Raw metrics and explicit unavailable components are in results.json and raw_comparison.csv. Final scientific analysis awaits the user.',
           '', 'Configured: same B0 weights and BN, fresh Adam, matching views, target 4000 actual updates per method. Actual completion:']
    for method in ('F1','B1'):
        train=results[method]['train'];lines.append(f"- {method}: training {train['status']}, actual updates {train.get('updates','not verified')}; stage states "+str({k:v['status'] for k,v in results[method].items() if isinstance(v,dict)}))
    lines+=['','F1 versus B1 is the matched comparison; B0 is a historical anchor. Historical postprocessing RNG was not recorded. Unknown legacy training indexing and evaluation NA rules are retained. No owner accuracy or independent generalization claim is made.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    atomic_json(root/'report_complete.json',dict(status='succeeded',outputs={name:identity(root/name) for name in ['results.json','raw_comparison.csv','report.md'] if (root/name).exists()}))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--method',choices=['F1','B1']);p.add_argument('--stage',choices=['train','predict','postprocess','matrix','evaluate','report'],required=True);a=p.parse_args();cfg=read(a.config)
    root=Path(cfg.output_root)
    if a.stage=='train':
        from .training import train;train(cfg,a.method)
    elif a.stage=='predict':
        from .prediction import predict;predict(cfg,a.method)
    elif a.stage=='postprocess':
        from .postprocess import postprocess;postprocess(cfg,a.method)
    elif a.stage=='matrix':
        from .matrix import build
        c=json.loads((root/f'configs/{a.method}_machine.json').read_text());build(c,root/'matrix'/a.method,cache=root/'shared/coordinate_cache')
        # Assignment files are an explicit part of the matrix manifest; expose an index.
        manifest=json.loads((root/'matrix'/a.method/'complete.json').read_text())
        atomic_json(root/'assignment'/a.method/'manifest.json',dict(mask=manifest['dependencies']['mask'],chunks=manifest['receipts'],coordinate_cache=manifest['coordinate_cache'],matrix_manifest=identity(root/'matrix'/a.method/'complete.json')))
    elif a.stage=='evaluate':
        from .evaluate import evaluate
        evaluate(root/f'configs/{a.method}_machine.json',root/'configs/reference.json',root/'configs/protocol.json',root/'matrix'/a.method,root/'evaluation'/a.method)
    else:report(cfg)

if __name__=='__main__':main()
