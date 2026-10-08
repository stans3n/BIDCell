"""Assemble machine-readable admission from completed evidence, never a plan."""
import argparse,json,subprocess
from pathlib import Path
import xml.etree.ElementTree as ET
from scripts.crgf.common import atomic_json,identity,sha256
from .evaluation import ARMS,times


def read(p):return json.loads(Path(p).read_text())
def assemble(config,independent=None,fault_receipt=None):
    q=read(config);root=Path(q['output_root']);failures=[];evidence={}
    required={
        'real_models_and_resume':root/'tests/real_model_frozen/complete.json',
        'mini_front_four_stages':root/'tests/mini_chain_v2/first_four_stages.json',
        'mini_three_profiles_pair_four':root/'tests/mini_chain_v2/mini_evaluation_receipt.json',
        'mini_compute':root/'tests/mini_chain_v2/compute_complete.json',
        'storage_resources':root/'tests/resources.json',
        'historical_inputs':Path('/home/zwan0726/work/outputs/cos_admission_audit_20261009/historical_patch_identity_audit.json'),
        'failure_injection':Path(fault_receipt) if fault_receipt else root/'tests/fault_receipt_required.json',
    }
    for name,path in required.items():
        if not path.exists():failures.append(name+':missing');continue
        data=read(path);evidence[name]=identity(path)
        if name=='storage_resources':
            if not data.get('sufficient'):failures.append(name+':insufficient')
        elif name=='failure_injection':
            if not data.get('all_passed') or len(data.get('results',[]))<24 or not all(r.get('passed') for r in data['results']):failures.append(name+':not_passed')
        elif name not in ('historical_inputs',) and data.get('status')!='succeeded':failures.append(name+':not_succeeded')
    unit=root/'tests/final_pytest.xml'
    if not unit.exists():failures.append('final_pytest:missing')
    else:
        xml=ET.parse(unit).getroot();suites=list(xml.iter('testsuite'))
        if not suites or sum(int(s.get('tests',0)) for s in suites)<17 or any(int(s.get('errors',0))+int(s.get('failures',0)) for s in suites):failures.append('final_pytest:failure_or_missing_cases')
        evidence['final_pytest']=identity(unit)
    branches=read(root/'source/branches.json');frozen=read(root/'source/frozen.json')
    for arm,item in q['methods'].items():
        if subprocess.check_output(['git','rev-parse','HEAD'],cwd=item['worktree'],text=True).strip()!=item['commit']:failures.append(arm+':commit_mismatch')
        cfg=read(item['config'])
        if cfg['updates']!=4000 or cfg['mini_test']:failures.append(arm+':formal_budget')
    if independent is None:failures.append('independent_final_review:pending')
    else:
        review=read(independent);evidence['independent_final_review']=identity(independent)
        if not review.get('approved_to_launch'):failures.append('independent_final_review:not_approved')
    result=dict(schema='COS-admission-v1',version='COS-4ARM-PLAN-1.0',run_id=q['run_id'],timezone='Australia/Sydney',**times(),resolved_configs={a:read(v['config']) for a,v in q['methods'].items()},commits={a:v['commit'] for a,v in q['methods'].items()},common_commit=branches['common_commit'],frozen_source=identity(root/'source/frozen.json'),queue=identity(config),inputs={a:identity(Path(v['cost_root'])/'configs/frozen.json') for a,v in q['methods'].items()},parent_view_label_probe_provenance=identity(root/'source/prepare.json'),tests=evidence,resources=read(root/'tests/resources.json') if (root/'tests/resources.json').exists() else None,required_failures=failures,approved_to_launch=not failures,model_assignment=dict(controller='gpt-6.1-sol/high',implementer='gpt-6-astra/high',evaluation_runner='gpt-6-astra/high',independent_admission_auditor='gpt-6-astra/high'),scientific_final_analysis='pending user resume; separate analyst required',initial_checkpoint_load_validation=evidence.get('real_models_and_resume'))
    atomic_json(root/'admission.json',result);print(json.dumps(dict(approved_to_launch=result['approved_to_launch'],required_failures=failures),indent=2));return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--independent-review');p.add_argument('--fault-receipt',required=True);a=p.parse_args();assemble(a.config,a.independent_review,a.fault_receipt)
