"""Revalidate real mini artifacts and bind a control-only follow-up revision.

The expensive mini ran the exact a4ff61c scientific producers. A later change
only inserts formal-run stop boundaries. Compare ASTs of every transfer-module
helper actually used by mini, then validate its real artifacts with current
code. This records the distinction rather than claiming the formal full node
ran, or repeating the long unchanged numerical replay.
"""
import ast
from pathlib import Path
import subprocess
from scripts.boundary_three.common import read,identity,sha256,check_identity,atomic_json
from scripts.boundary_three.p2_native import validate_process,STAGES
from scripts.boundary_three.evaluation import validate_endpoint


def definitions(source):
    return {node.name:ast.dump(node,include_attributes=False) for node in ast.parse(source).body
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}


def validate(root,output):
    root = Path(root)
    rec = read(root/'complete.json')
    if rec['status'] != 'passed':
        raise ValueError('mini incomplete')
    for obj in rec['outputs'].values():
        check_identity(obj)
    repo = Path(__file__).resolve().parents[2]
    sources = rec['dependencies']['sources']
    source_checks = {}
    for name in ('p2_native.py','p2_rules.py','p2_diagnostics.py','p2_mini.py'):
        path = repo/'scripts/boundary_three'/name
        if sha256(path) != sources[name]['sha256']:
            raise ValueError('mini scientific source changed: '+name)
        source_checks[name] = identity(path)
    old = subprocess.check_output(['git','show','a4ff61c:scripts/boundary_three/p2_transfer.py'],cwd=repo,text=True)
    current_path = repo/'scripts/boundary_three/p2_transfer.py'
    current = current_path.read_text()
    if __import__('hashlib').sha256(old.encode()).hexdigest() != sources['p2_transfer.py']['sha256']:
        raise ValueError('mini transfer source not the recorded implementation commit')
    a,b = definitions(old),definitions(current)
    helpers = ('copy_exact','matrix_only','assert_matrix_equal','validate_mini')
    if any(a[name] != b[name] for name in helpers):
        raise ValueError('mini-called transfer helper changed')
    cfg = read(root/'config.json')
    results = {}
    for node,base in (('P2_BASE','BASE_MATCH'),('P2_MORPH','V2_MORPH')):
        endpoint = root/node
        pred = endpoint/'predictions'/node
        for directory in (pred,endpoint/'native_replay/predictions'/base):
            stage = validate_process(directory)
            for obj in stage['dependencies']['source'].values():
                check_identity(obj)
        plain = read(endpoint/'plain_native/predictions'/base/'postprocess/complete.json')
        for relative,expected_hash in plain['dependencies']['source'].items():
            if sha256(repo/relative) != expected_hash:
                raise ValueError('uninstrumented native oracle producer changed: '+relative)
        spec = dict(root=str(endpoint),machine=str(endpoint/'configs/machine.json'),
                    final=str(pred/'final.tif'),matrix=str(endpoint/'matrix'/node),
                    evaluation=str(endpoint/'evaluation'/node),cost_root=str(endpoint))
        result = validate_endpoint(cfg,node,spec)
        for stage in ('tiles','splits'):
            target = endpoint/'recovery'/('interrupted_'+stage)
            validate_process(target)
            for name in STAGES:
                if sha256(target/(name+'.tif')) != sha256(endpoint/'recovery/uninterrupted'/(name+'.tif')):
                    raise ValueError('recovery hash mismatch')
        results[node] = result
    evidence = dict(status='passed',real_mini=identity(root/'complete.json'),scientific_producers=source_checks,
        mini_transfer_producer_commit='a4ff61c',current_transfer=identity(current_path),
        mini_called_helper_ASTs_unchanged=list(helpers),
        followup_scope='formal run_node stop boundaries only; real stop-file unit test separately passed',
        unit_test_log=identity(root.parent/'unit_tests.log'),endpoints=results,
        native_split_source=identity(repo/'scripts/crgf/postprocess.py'),
        full_formal_node_executed=False,independent_code_audit=False)
    atomic_json(output,evidence)
    return evidence


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    print(__import__('json').dumps(validate(args.root,args.output),indent=2))
