"""Concrete local prepare/admit/launch/resume and separate read-only checks."""
import argparse
import json
import os
from pathlib import Path
import shutil
from .common import *
from .runtime import initial_state


def prepare(root,bundle):
    from scripts.paired_v2.finalize import verify_bundle
    root=Path(root).resolve();verify_bundle(bundle)
    frozen=root/'prepared';snapshot=frozen/'historical_main_table18.csv'
    source=Path('/home/zwan0726/work/outputs/bidcell_historical_dual_20261013/main_table.csv')
    if snapshot.exists():
        if sha256(snapshot)!=read(root/'handoff/formal_outputs_before.json')[str(source)]['sha256']:raise ValueError('historical18 snapshot changed')
    else:shutil.copyfile(source,snapshot)
    authority=read(frozen/'authoritative_protocol.json')
    bindings={
        'objective':Path('/home/zwan0726/.codex/attachments/d6e3286e-0794-4d67-8b20-ff13d1f78d75/goal-objective.md'),
        'plan':Path('/home/zwan0726/work/docs/BIDCell_Tonight_Paired_V2_Experiment_Plan_20261014.md'),
        'contract':frozen/'experiment_contract.json','authority':frozen/'authoritative_protocol.json',
        'historical_audit':Path('/home/zwan0726/work/outputs/bidcell_v2_fuse_decision_20261014/historical_pairing_audit.json'),
        'historical_reuse':root/'reviews/historical_endpoint_reuse.json','implementation_review':root/'reviews/r1_code_review.json',
        'initial_review':root/'reviews/initial_code_review.json','implementation_report':root/'reviews/implementation_report.json',
        'revision_report':root/'reviews/reviewer_r1_implementation.json','environment_lock':root/'reviews/environment_lock.json',
        'paper_R_environment':root/'tests/paper_R_runtime_lock.json','model_configuration':root/'handoff/model_configuration.json',
        'resource_pilot':root/'tests/resources.json','disk_forecast':root/'tests/disk_forecast_preflight.json',
        'ETA_forecast':root/'tests/combined_eta_preflight.json','hosting_proof':root/'tests/hosting_validation.json',
        'pipeline_verification':root/'tests/pipeline/verification.json',
        'historical_main_table':snapshot,'spatial_wrapper':Path('/home/zwan0726/work/scripts/bidcell_benchmark/recompute_historical_spatial_diversity.R'),
        'spatial_author':Path(authority['spatial_diversity']['source']),
        'spatial_utils':Path('/home/zwan0726/work/outputs/bidcell_metric_protocol_review_20261008/publish_utils.R')}
    # The pinned entropy package is read through the historical author library; bind every package byte.
    lib=Path(authority['spatial_diversity']['source']).parent/'R_library/entropy'
    for p in sorted(lib.rglob('*')):
        if p.is_file():bindings['entropy_'+str(p.relative_to(lib)).replace('/','_')]=p
    forecast=read(bindings['disk_forecast']);eta=read(bindings['ETA_forecast'])
    if forecast['status']!='passed' or eta['estimated_remaining_with_buffer_hours']>9:raise ValueError('two-arm resource/window forecast not admitted')
    q=dict(version=1,root=str(root),repo=str(REPO),python=PYTHON,rscript=RSCRIPT,
        source_bundle=identity(bundle),bindings={k:identity(p) for k,p in bindings.items()},tasks=task_graph(),
        comparisons={k:list(v) for k,v in COMPARISONS.items()},max_retries=1,
        retry_policy='only parent timeout for training with strictly verified durable state; all other failures stop for manual repair',
        stage_timeouts=dict(freeze_inputs=3600,train=3600,paired_training=1800,predict=10800,postprocess=10800,matrix=1800,
            freeze_counts=600,evaluate=10800,reports=7200,package=1800),
        disk_forecast_remaining_bytes=forecast['forecast_remaining_bytes'],disk_reserve_bytes=10<<30,atomic_peak_bytes=2<<30,
        minimum_ram_available_bytes=6<<30,gpu_lock='/home/zwan0726/work/outputs/cos_gpu.lock',
        tmux_socket='bidcell-paired-v2-20261014',tmux_session='paired-v2-night',
        review_receipt=str(root/'reviews/final_integration_review.json'),nightly_publication=False,auto_ai_wakeup=False,
        available_compute_window_hours=9,estimated_compute_with_buffer_hours=eta['estimated_remaining_with_buffer_hours'],
        bootstrap_rng='NumPy PCG64 seed20261013, fixed 2000 paired occupied-block draws',
        count_freeze='after E0 own matrix, before both new formal evaluation stages',
        score_access_scope='prior ROI pilots are interface/admission tests; no full formal endpoint scores exist',
        scientific_scope='two new arms only; historical E1 reuse; no extra modules, sweeps or final AI research interpretation overnight')
    dest=root/'configs/pipeline.json'
    if dest.exists() and read(dest)!=q:raise ValueError('existing pipeline config immutable; version and independently review any migration')
    if not dest.exists():atomic_json(dest,q)
    state=root/'pipeline_status.json'
    if not state.exists():atomic_json(state,initial_state(q))
    verify_execution(q);verify_history(q);environment_check(q);require_space(q,launch=True)
    return dict(status='prepared_not_admitted_not_launched',pipeline_config=identity(dest),tasks=len(q['tasks']),source_bundle=q['source_bundle'])


def admit(root):
    q=queue(root);verify_execution(q)
    r=read(q['review_receipt'])
    if r.get('status')!='approved_for_launch' or not r.get('formal_admission') or r.get('blocking_findings'):
        raise ValueError('final independent review has not approved admission')
    if r['source_bundle']!=q['source_bundle'] or r['pipeline_config']!=identity(Path(root)/'configs/pipeline.json'):
        raise ValueError('review identity does not match actual execution')
    environment_check(q);require_space(q,launch=True)
    for arm in RUNS:
        cfg=load_config(config_path(arm))
        gate=dict(status='admitted',run_id=arm,model_kind=RUNS[arm],base_only=True,lambda_max=0,eta=0,
            train_producer=producer('train'),initial_snapshot=identity(cfg['initial_snapshot']),
            pipeline_config=identity(Path(root)/'configs/pipeline.json'),review=identity(q['review_receipt']),source_bundle=q['source_bundle'])
        p=Path(root)/'admission'/f'{arm}.json'
        if p.exists() and read(p)!=gate:raise ValueError('previous admission identity differs')
        if not p.exists():atomic_json(p,gate)
    verify_execution(q,admitted=True);return dict(status='both_arms_admitted_not_launched',review=identity(q['review_receipt']))


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','admit','launch','resume','run','worker','status','train-status','tail','trains-complete-check','complete-check'])
    p.add_argument('--root',default=str(DEFAULT_ROOT));p.add_argument('--bundle');p.add_argument('--arm',choices=list(RUNS));p.add_argument('--task');p.add_argument('--attempt',type=int,default=0);p.add_argument('--lines',type=int,default=30)
    a=p.parse_args();code=0
    from . import runtime
    if a.command=='prepare':result=prepare(a.root,a.bundle)
    elif a.command=='admit':result=admit(a.root)
    elif a.command in ('launch','resume'):result=runtime.launch(a.root)
    elif a.command=='run':runtime.run(a.root);result={'status':'completed_awaiting_ai_resume'}
    elif a.command=='worker':runtime.worker(a.root,a.task,a.attempt);return
    elif a.command=='status':result=runtime.status(a.root)
    elif a.command=='train-status':result=runtime.train_status(a.root,a.arm)
    elif a.command=='tail':result=runtime.tail(a.root,a.lines)
    else:
        result=runtime.completion_check(a.root,trains_only=a.command=='trains-complete-check');code=0 if result['complete'] else 3
    print(json.dumps(result,indent=2,allow_nan=False));raise SystemExit(code)


if __name__=='__main__':main()
