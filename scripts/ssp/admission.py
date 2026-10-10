"""Collect real required evidence; independent approval is a separate authority."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from scripts.crgf.common import atomic_json,identity,sha256,verify_outputs
from .evaluation import read
REQUIRED=['tests/math_edges.json','tests/checkpoint_faults.json','tests/cpu_equivalence.json','tests/data_contracts.json','prepared/coverage.json','tests/prototype_audit.json','tests/equivalence.json','diagnostics/numeric_admission.json','diagnostics/gradient_diagnostics.json','tests/independent_review/independent_method_cpu.json','tests/runner_engineering.json','tests/mini/mini_complete.json','b_cost/complete.json','tests/resources.json']

def collect(root):
    root=Path(root);qpath=root/'configs/queue.json';q=read(qpath);failures=[];evidence={}
    for name in REQUIRED:
        path=root/name
        if not path.exists():failures.append(name+': missing');continue
        r=read(path);evidence[name]=identity(path)
        if name=='tests/resources.json':
            if not r.get('sufficient'):failures.append(name+': resource insufficient')
        elif r.get('status') not in ('passed','succeeded'):failures.append(name+': did not pass')
        if r.get('outputs'):verify_outputs(r)
    coverage=read(root/'prepared/coverage.json') if (root/'prepared/coverage.json').exists() else {}
    if coverage.get('formal_exposure',{}).get('F',0)<=0:failures.append('zero genuine p0 new supervision')
    numeric=read(root/'diagnostics/numeric_admission.json') if (root/'diagnostics/numeric_admission.json').exists() else {}
    for arm in ('S','SP'):
        nr=numeric.get('arms',{}).get(arm,{})
        for k in ('strict_resume_next_step_bitwise','probe_full_state_and_next_update_bitwise'):
            if nr.get(k) is not True:failures.append(arm+': '+k)
        if nr.get('updates')!=32:failures.append(arm+': real 32 updates missing')
    frozen=root/'source/frozen.json'
    if not frozen.exists():failures.append('source not frozen')
    resource=read(root/'tests/resources.json') if (root/'tests/resources.json').exists() else {}
    result=dict(version='SSP-OVERNIGHT-v1.0',run_id=q['run_id'],status='passed' if not failures else 'failed',approved_to_launch=not failures,required_failures=failures,evidence=evidence,resources=resource,queue=identity(qpath),frozen_source=identity(frozen) if frozen.exists() else None,resolved_configs={a:read(s['config']) for a,s in q['methods'].items()},arm_frozen={a:identity(Path(s['cost_root'])/'configs/frozen.json') for a,s in q['methods'].items()},independent_review_required=True,scientific_acceptance=False)
    atomic_json(root/'tests/admission.json',result);return result

def execute_missing(root):
    root=Path(root);qpath=root/'configs/queue.json';q=read(qpath);python=q['python'];cwd=q['methods']['S']['worktree']
    from .prepare import prepare
    from .bootstrap import provisional_freeze
    if not (root/'source/frozen.json').exists():
        prepare(root,inventory=False);provisional_freeze(root)
    commands=[('tests/math_edges.json',[python,'-m','scripts.ssp.tests.math_edges','--root',str(root)]),('tests/data_contracts.json',[python,'-m','scripts.ssp.tests.data_contracts','--root',str(root)]),('prepared/coverage.json',[python,'-m','scripts.ssp.prepare','--root',str(root)]),('tests/prototype_audit.json',[python,'-m','scripts.ssp.tests.prototype_audit','--root',str(root)]),('tests/equivalence.json',[python,'-m','scripts.ssp.tests.equivalence','--root',str(root)]),('tests/cpu_equivalence.json',[python,'-m','scripts.ssp.tests.cpu_equivalence','--root',str(root)]),('diagnostics/numeric_admission.json',[python,'-m','scripts.ssp.tests.numeric','--root',str(root)]),('tests/checkpoint_faults.json',[python,'-m','scripts.ssp.tests.checkpoint_faults','--root',str(root)]),('diagnostics/gradient_diagnostics.json',[python,'-c','from scripts.ssp.loss_diagnostics import calibration_diagnostics; import sys; calibration_diagnostics(sys.argv[1])',str(root)]),('tests/runner_engineering.json',[python,'-m','scripts.ssp.test_engineering',str(root/'tests/runner_engineering.json')])]
    for result,command in commands:
        p=root/result
        if p.exists() and read(p).get('status') in ('passed','succeeded'):continue
        subprocess.run(command,cwd=cwd,check=True)
    mini=root/'tests/mini'
    if not (mini/'configs/queue.json').exists():subprocess.run([python,'-m','scripts.ssp.mini','--queue',str(qpath),'--output',str(mini)],cwd=cwd,check=True)
    if not (mini/'mini_complete.json').exists():subprocess.run([python,'-m','scripts.ssp.mini','--queue',str(mini/'configs/queue.json'),'--stages'],cwd=cwd,check=True)
    subprocess.run([python,'-m','scripts.ssp.b_cost','--config',str(qpath)],cwd=cwd,check=True)
    from .resources import measure
    measure(qpath)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--run-tests',action='store_true');a=p.parse_args()
    if a.run_tests:execute_missing(a.root)
    r=collect(a.root);print(json.dumps(r,indent=2));raise SystemExit(0 if r['status']=='passed' else 2)
