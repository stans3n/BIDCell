"""Versioned review bundles; validate original step zero without rebinding it."""
import argparse
import importlib.metadata
import re
import subprocess
from .common import *


def full_source():
    from scripts.new_methods.common import source_manifest
    return source_manifest()


def verify_bundle(path):
    bundle=read(path)
    if bundle['source_files']!=full_source() or bundle['source_digest']!=digest(full_source()):
        raise ValueError('reviewed complete source tree changed')
    for stage,files in bundle['component_producers'].items():
        if producer(stage)!=files:raise ValueError('scientific component changed: '+stage)
    for obj in bundle['immutable_bindings'].values():
        if identity(obj['path'])!=obj:raise ValueError('review bundle immutable input/config/evidence changed: '+obj['path'])
    return dict(status='passed',bundle=identity(path),source_digest=bundle['source_digest'])


def freeze_bundle(root=DEFAULT_ROOT,label='implementation'):
    root=Path(root)
    if not re.fullmatch(r'[a-z0-9_-]+',label):raise ValueError('invalid bundle label')
    if not (root/'prepared/step_zero_receipt.json').exists():
        raise ValueError('prepare and all pre-pilot step-zero receipts must already exist')
    # The idempotent preparation path reconstructs BOTH original untrained
    # models and compares all tensors/BN/Morph/RNG/Adam/header fields. Any
    # scientific change is fatal; existing snapshots/receipts are never edited.
    from .prepare import prepare
    prepare(root)
    frozen=read(INPUT/'source/manifest.json')['files']
    unexpected=[name for name,sha in frozen.items() if name!='bidcell/model/model/model.py' and sha256(REPO/name)!=sha]
    if unexpected:raise ValueError('unexpected modification of frozen mature modules: '+str(unexpected))
    files=full_source();bindings={}
    for name in ('input_audit.json','step_zero_receipt.json','authoritative_protocol.json','experiment_contract.json'):
        bindings['prepared/'+name]=identity(root/'prepared'/name)
    for run in RUNS:
        for name in ('run.json','machine.json'):bindings[run+'/'+name]=identity(root/run/'configs'/name)
        bindings[run+'/step_zero']=identity(root/'prepared'/f'{run}_step_zero.pth')
    for name in ('implementation_report.json','historical_endpoint_reuse.json'):
        if (root/'reviews'/name).exists():bindings['reviews/'+name]=identity(root/'reviews'/name)
    packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata.get('Name')}
    env=dict(numerical_environment=read(root/'prepared/step_zero_receipt.json')['environment'],python_packages=dict(sorted(packages.items())),
             paper_rscript=read(read(INPUT/'configs/queue.json')['prepared']+'/complete.json')['dependencies']['rscript'])
    env_path=root/'reviews/environment_lock.json'
    if env_path.exists() and read(env_path)!=env:raise ValueError('environment lock changed')
    if not env_path.exists():atomic_json(env_path,env)
    bindings['environment_lock']=identity(env_path)
    result=dict(status='ready_for_independent_review_not_admission',label=label,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        source_files=files,source_digest=digest(files),component_producers={s:producer(s) for s in ('initial','train','predict','postprocess','matrix','evaluate')},
        immutable_bindings=bindings,original_step_zero_reconstructed_and_equal=True,
        source_note='Complete tree includes paired_v2 and any paired_v2_pipeline orchestration present at freeze time. Scientific headers remain the original immutable objects.',
        original_mature_sources_unchanged_except_h3_interaction_insertion=True)
    dest=root/'reviews/source_bundles'/f'{label}.json'
    if dest.exists() and read(dest)!=result:raise ValueError('bundle label already frozen; choose a new label and obtain new independent review')
    if not dest.exists():atomic_json(dest,result)
    verify_bundle(dest);return identity(dest)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=DEFAULT_ROOT);p.add_argument('--label',default='implementation');p.add_argument('--verify',type=Path);a=p.parse_args()
    print(json.dumps(verify_bundle(a.verify) if a.verify else freeze_bundle(a.root,a.label),indent=2))
