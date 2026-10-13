"""Stage entry points and semantic acceptance for non-model computations."""
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from .common import *
from .reports import write_json


def paired_dependencies(q):
    return dict(source_bundle=q['source_bundle'],views=identity(INPUT/'configs/views.json'),
        historical_log=identity(INPUT/'V2_MORPH/training.jsonl'),
        new_training={a:identity(endpoint(a)['root']/'checkpoints/complete.json') for a in RUNS},
        sequences={a:identity(endpoint(a)['root']/'effective_sequence.jsonl') for a in RUNS})


def compare_training(q):
    from scripts.paired_v2.stages import validate_stage
    from scripts.paired_v2.training import sequence_identity
    views=read(INPUT/'configs/views.json')['views'];used=[v for v in views if v['K']]
    hashes={v['path']:v['identity']['sha256'] for v in read(INPUT/'configs/predict_shift0.json')}
    expected=[sequence_identity(v,i,hashes[v['path']]) for i,v in enumerate(used,1)]
    actual={a:[json.loads(s) for s in (endpoint(a)['root']/'effective_sequence.jsonl').read_text().splitlines()] for a in RUNS}
    for a in RUNS:
        validate_stage(config_path(a),'train',require_bound=True)
        if len(actual[a])!=3625 or [{k:r[k] for k in expected[0]} for r in actual[a]]!=expected:raise ValueError('formal actual sequence differs: '+a)
    historical=[r for line in (INPUT/'V2_MORPH/training.jsonl').read_text().splitlines() if (r:=json.loads(line)).get('event')=='update']
    if len(historical)!=3625:raise ValueError('historical update sequence incomplete')
    for row,v in zip(historical,expected):
        if (row['updates'],row['cursor'],row['nominal_index'],row['xy'],row['K'],row['flip'],row['rot90'])!=(v['update'],v['cursor'],v['index'],[v['row'],v['col']],v['K'],v['flip'],v['rot90']):
            raise ValueError('historical/current paired order differs')
    return dict(status='passed',updates=3625,visits=3998,empty_skips=373,effective_identity_digest=digest(expected),
        historical_scope='runtime order/coordinates/K/augmentation checked; candidate order from frozen views and historical producer assertions, no all-historical-input recomputation')


def paired_training(q):
    out=Path(q['root'])/'paired_training';out.mkdir(exist_ok=True)
    result=compare_training(q)
    if (out/'complete.json').exists():return validate_custom(q,'paired_training')
    atomic_json(out/'comparison.json',result)
    atomic_json(out/'complete.json',dict(status='succeeded',dependencies=paired_dependencies(q),outputs=inventory(out)))
    return validate_custom(q,'paired_training')


def package_dependencies(q):
    return dict(source_bundle=q['source_bundle'],pipeline_config=identity(Path(q['root'])/'configs/pipeline.json'),
        final_review=identity(q['review_receipt']),reports=identity(Path(q['root'])/'results/complete.json'),
        upstream_receipts={t['id']:identity(own_receipt(q,t['id'])) for t in q['tasks'] if t['stage']!='package'})


def result_package(q):
    root=Path(q['root']);out=root/'result_pack';out.mkdir(exist_ok=True)
    if (out/'complete.json').exists():return validate_custom(q,'package')
    for task in q['tasks']:
        if task['stage']!='package':validate_task(q,task)
    verify_execution(q,admitted=True);verify_history(q)
    # Durable reference manifest for every formal artifact, not a second matrix copy.
    trees={arm:inventory(root/arm) for arm in RUNS}
    for folder in ('prepared','frozen_strata','frozen_counts','paired_training','results','pipeline_receipts','admission'):
        trees[folder]=inventory(root/folder)
    e1=read(root/'reviews/historical_endpoint_reuse.json')
    manifest=dict(status='all_computation_and_evaluation_accepted_awaiting_manual_AI_resume',created_utc=utc(),
        branch='feature/v2-paired-20261014',repo=str(REPO),runroot=str(root),source_bundle=q['source_bundle'],
        final_review=identity(q['review_receipt']),pipeline_config=identity(root/'configs/pipeline.json'),
        bindings=q['bindings'],formal_artifact_trees=trees,historical_endpoint_reuse=e1,
        historical_provenance=e1['provenance'],nightly_formal_sync=False,auto_AI_wakeup=False,
        required_manual_resume=['Verify all stages and result manifest','Repair/resume only affected stages if needed',
            'Back up and idempotently publish BASE_MATCH/V2_INTERACT to main tables/context; preserve old18 and three-endpoint cohorts; recalculate 20-method Purity',
            'Start independent Astra/Max analyst C with original plan, reviews, raw results, manifest and published table',
            'Verify C report, sync final analysis, only then complete Goal'])
    write_json(out/'manifest.json',manifest)
    refs={**q['bindings'],'source_bundle':q['source_bundle'],'final_review':identity(q['review_receipt'])}
    for name,obj in refs.items():
        src=Path(obj['path']);target=out/'review_and_protocol'/f'{name}{src.suffix}'
        target.parent.mkdir(exist_ok=True);shutil.copyfile(src,target)
    write_json(out/'result_index.json',dict(main_table=str(root/'results/main_table.csv'),paired_deltas=str(root/'results/paired_deltas.csv'),
        decision=str(root/'results/decision.json'),population_files={a:str(root/'results'/f'{a}_all_ID_population.csv') for a in ENDPOINTS},
        strata=str(root/'results/strata_comparisons.json'),spatial=str(root/'results/spatial_summary.json'),
        stage_cost=str(root/'results/stage_cost.json'),manifest=str(out/'manifest.json'),handoff=str(root/'handoff/launch_handoff.json')))
    atomic_json(out/'complete.json',dict(status='succeeded',dependencies=package_dependencies(q),outputs=inventory(out)))
    return validate_custom(q,'package')


def validate_custom(q,stage):
    if stage=='freeze_inputs':
        from .freeze import validate_inputs
        return validate_inputs(q)
    if stage=='freeze_counts':
        from .freeze import validate_counts
        return validate_counts(q)
    if stage=='reports':
        from .reports import validate_reports
        return validate_reports(q)
    folder='paired_training' if stage=='paired_training' else 'result_pack'
    out=Path(q['root'])/folder;rec=read(out/'complete.json')
    deps=paired_dependencies(q) if stage=='paired_training' else package_dependencies(q)
    if rec['status']!='succeeded' or rec['dependencies']!=deps:raise ValueError('custom completion lineage changed: '+stage)
    verify_outputs(rec)
    if stage=='paired_training':
        if read(out/'comparison.json')!=compare_training(q):raise ValueError('paired training comparison invalid')
        return dict(status='passed',receipt=identity(out/'complete.json'),paired_updates=3625)
    if stage!='package':raise ValueError(stage)
    manifest=read(out/'manifest.json')
    if manifest['source_bundle']!=q['source_bundle'] or manifest['pipeline_config']!=identity(Path(q['root'])/'configs/pipeline.json'):
        raise ValueError('result manifest identity changed')
    for tree in manifest['formal_artifact_trees'].values():
        for obj in tree.values():check_obj(obj)
    return dict(status='passed',receipt=identity(out/'complete.json'),manifest=identity(out/'manifest.json'),awaiting_manual_AI_resume=True)


def execute(q,task):
    for key in task['depends_on']:
        previous=next(t for t in q['tasks'] if t['id']==key);validate_task(q,previous)
    if task['arm']:
        if task['stage']=='evaluate':
            from .freeze import validate_counts
            validate_counts(q)
        from scripts.paired_v2.stages import run_stage
        return run_stage(config_path(task['arm']),task['stage'])
    if task['stage']=='freeze_inputs':
        from .freeze import freeze_inputs
        return freeze_inputs(q)
    if task['stage']=='freeze_counts':
        from .freeze import freeze_counts
        return freeze_counts(q)
    if task['stage']=='paired_training':return paired_training(q)
    if task['stage']=='reports':
        from .reports import reports
        return reports(q)
    if task['stage']=='package':return result_package(q)
    raise ValueError(task)
