"""Host-only real mini evaluator orchestration and measured process-group costs.

Front stages run separately; their internal clocks and scope are kept explicit.
This harness cannot launch a formal queue and is not a substitute for runner tests.
"""
import argparse
import os
from pathlib import Path
import subprocess
import time
from scripts.cos.evaluation import ARMS,read,times
from scripts.cos.runner import process_group_memory,accept,complete_receipt
from scripts.crgf.common import atomic_json,identity,process_identity


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--rerun-comparisons',action='store_true');args=p.parse_args();cfg=read(args.config)
    if cfg.get('mini_test') is not True or any(read(s['config']).get('mini_test') is not True for s in cfg['methods'].values()):raise ValueError('this harness only runs explicit mini test fixtures')
    root=Path(cfg['output_root']);path=root/'run_status.json'
    if path.exists():state=read(path)
    else:state=dict(run_id=cfg['run_id'],runner=process_identity(),status='mini_evaluation_running',stages={},scope='test harness; front stages completed separately; actual evaluation process group sampling')
    for arm,spec in cfg['methods'].items():
        base=Path(spec['cost_root']);sources={'train':base/'checkpoints'/arm/'complete.json','predict':base/'predictions'/arm/'complete.json','postprocess':base/'predictions'/arm/'postprocess/complete.json','matrix':Path(spec['matrix'])/'complete.json'}
        for stage,receipt in sources.items():
            name=arm+'_'+stage
            if name in state['stages']:continue
            rec=read(receipt)
            if rec['status']!='succeeded':raise ValueError('front mini stage incomplete')
            wall=rec.get('wall_seconds');peak=None;scope='internal stage elapsed; parent process startup excluded; CPU memory not sampled for this front stage'
            if stage=='matrix':
                resource=read(Path(spec['matrix'])/'resources.json');wall=resource['wall_seconds'];peak=max([s['rss'] for s in resource['samples']],default=None);scope='matrix-only single-process sampled RSS; not full mini-chain peak'
            if wall is None:raise ValueError('front stage measured time missing '+name)
            attempt=dict(wall_seconds=wall,sampled_peak_rss_bytes_sum=peak,exit_code=0,memory_sampling_scope=scope,timing_source=identity(receipt))
            state['stages'][name]=dict(method=arm,stage=stage,status='succeeded',receipt=str(receipt),receipt_sha256=identity(receipt)['sha256'],attempts=[attempt])
    state['runner']=process_identity();atomic_json(path,state)
    checkout=cfg['methods']['B1_CONT']['worktree'];logs=root/'logs';logs.mkdir(exist_ok=True)
    tasks=[]
    for base in ['B1','G']:
        for suffix in ['CONT','COS']:
            arm=base+'_'+suffix;tasks.append((arm+'_evaluate',arm,'evaluate',[cfg['python'],'-u','-m','scripts.cos.evaluation','--config',cfg['methods'][arm]['config']],Path(cfg['methods'][arm]['evaluation'])/'complete.json'))
        tasks.append(('pair_'+base,None,'pair',[cfg['python'],'-u','-m','scripts.cos.evaluation','--config',args.config,'--comparison',base],root/('pair_'+base)/'complete.json'))
    tasks.append(('four',None,'four',[cfg['python'],'-u','-m','scripts.cos.evaluation','--config',args.config,'--comparison','four'],root/'four_arm/complete.json'))
    for name,arm,stage,command,receipt in tasks:
        existing=state['stages'].get(name)
        if existing and existing['status']=='succeeded' and not (args.rerun_comparisons and stage in ['pair','four']):accept(stage,read(receipt));continue
        number=len(existing['attempts'])+1 if existing else 1;logpath=logs/f'{name}_attempt{number}.log'
        attempt=dict(start=times(),command=command,log=str(logpath),sampled_peak_rss_bytes_sum=0,sampled_peak_pss_bytes_sum=0,memory_sampling_scope='actual mini stage process-group including R subprocesses sampled every 0.5 seconds; sampled lower bound')
        record=dict(method=arm,stage=stage,status='running',receipt=str(receipt),attempts=(existing['attempts'] if existing else [])+[attempt]);state['stages'][name]=record;state['current_stage']=name;atomic_json(path,state)
        start=time.monotonic()
        with logpath.open('w') as log:
            child=subprocess.Popen(command,cwd=checkout,env=dict(os.environ,PYTHONPATH=checkout,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            record['child_identity']=process_identity(child.pid)
            while child.poll() is None:
                mem=process_group_memory(child.pid)
                for k in ['rss','pss']:attempt['sampled_peak_'+k+'_bytes_sum']=max(attempt['sampled_peak_'+k+'_bytes_sum'],mem[k+'_bytes_sum'])
                state['heartbeat']=times();record['wall_seconds_current_attempt']=time.monotonic()-start;atomic_json(path,state);time.sleep(.5)
            code=child.wait()
        attempt.update(exit_code=code,wall_seconds=time.monotonic()-start,end=times());record['child_identity']=None
        if code==0:
            try:accept(stage,read(receipt))
            except Exception as exc:code=2;attempt['artifact_failure']=str(exc)
        record['status']='succeeded' if code==0 else 'failed'
        if code==0:record['receipt_sha256']=identity(receipt)['sha256'];attempt['published_receipt']=identity(receipt)
        else:record['failure']=logpath.read_text()[-12000:]
        atomic_json(path,state);print(name,record['status'],attempt['wall_seconds'],flush=True)
        if code:raise SystemExit(code)
    complete_receipt(cfg,state);state.update(status='mini_compute_succeeded',current_stage=None);atomic_json(path,state)
    atomic_json(root/'mini_evaluation_receipt.json',dict(status='succeeded',scope='real four-arm mini three profiles, both pairs and four-arm comparison; not whole-slide scientific result',stages={k:identity(v['receipt']) for k,v in state['stages'].items()},sources={str(p):identity(p) for p in Path(checkout,'scripts/cos').glob('*.py')},**times()))

if __name__=='__main__':main()
