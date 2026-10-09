"""Fail-closed ABC launch admission assembled only from executable evidence."""
import argparse,json,subprocess
from pathlib import Path
import xml.etree.ElementTree as ET
from scripts.crgf.common import atomic_json,identity,sha256,verify_outputs
from .evaluation import ARMS,times,read

def assemble(config,independent=None):
    q=read(config);root=Path(q['output_root']);failures=[];evidence={}
    required={'calibration':root/'diagnostics/calibration.json','numeric_admission':root/'diagnostics/numeric_admission.json','real_mini_front':root/'tests/mini_chain_abc/first_four_stages.json','real_mini_profiles':root/'tests/mini_chain_abc/mini_evaluation_receipt.json','real_mini_compute':root/'tests/mini_chain_abc/compute_complete.json','storage_resources':root/'tests/resources.json','pipeline_test_execution':root/'tests/pipeline_pytest_receipt.json','loss_test_execution':root/'tests/loss_pytest_receipt.json'}
    for name,path in required.items():
        if not path.exists():failures.append(name+':missing');continue
        data=read(path);evidence[name]=identity(path)
        good=data.get('sufficient') if name=='storage_resources' else data.get('status') in ('succeeded','passed')
        if not good:failures.append(name+':not_passed')
        if name.endswith('_test_execution'):
            checkout=Path(__file__).resolve().parents[2]
            for file,expected in data.get('source',{}).items():
                if sha256(checkout/file)!=expected:failures.append(name+':tested_source_changed:'+file)
            xml=root/'tests'/('pipeline_pytest.xml' if name.startswith('pipeline') else 'loss_pytest.xml')
            if data.get('xml_sha256')!=sha256(xml):failures.append(name+':xml_changed')
    for name in ('pipeline_pytest.xml','loss_pytest.xml'):
        path=root/'tests'/name
        if not path.exists():failures.append(name+':missing');continue
        suites=list(ET.parse(path).getroot().iter('testsuite'))
        if not suites or sum(int(s.get('tests',0)) for s in suites)<(19 if name.startswith('pipeline') else 9) or any(int(s.get('errors',0))+int(s.get('failures',0)) for s in suites):failures.append(name+':not_passed')
        evidence[name]=identity(path)
    configs={a:read(q['methods'][a]['config']) for a in ARMS}
    for a,c in configs.items():
        if c['arm_id']!=a or c['updates']!=4000 or c['mini_test'] or not c['cos_enabled'] or c['base_variant']!='B1':failures.append(a+':invalid_formal_config')
    if configs['A']['anchor_weight']!=0 or configs['A']['pos_background_rho']!=1 or configs['B']['pos_background_rho']!=1 or configs['C']['pos_background_rho']!=.5:failures.append('arm_intervention_mismatch')
    if configs['B']['anchor_weight']!=configs['C']['anchor_weight'] or configs['B']['calibration_identity']!=configs['C']['calibration_identity']:failures.append('BC_calibration_mismatch')
    if required['calibration'].exists() and configs['B']['anchor_weight']!=read(required['calibration'])['lambda_cal']:failures.append('calibrated_weight_mismatch')
    if required['calibration'].exists() and any(c['calibration_identity']!=sha256(required['calibration']) for c in configs.values()):failures.append('calibration_identity_mismatch')
    branch=read(root/'source/branches.json');frozen=read(root/'source/frozen.json')
    for a,s in q['methods'].items():
        if subprocess.check_output(['git','rev-parse','HEAD'],cwd=s['worktree'],text=True).strip()!=s['commit']:failures.append(a+':source_commit_mismatch')
    if independent is None:failures.append('independent_admission_review:missing')
    else:
        review=read(independent);evidence['independent_admission_review']=identity(independent)
        if not review.get('approved_to_launch'):failures.append('independent_admission_review:not_approved')
    result=dict(schema='COS-LOSS-ABC-admission-v2.0',run_id=q['run_id'],**times(),resolved_configs=configs,common_commit=branch['common_commit'],frozen_source=identity(root/'source/frozen.json'),queue=identity(config),tests=evidence,resources=read(required['storage_resources']) if required['storage_resources'].exists() else None,required_failures=failures,approved_to_launch=not failures,model_assignment=dict(requested_controller_setting='gpt-6.1-sol/high',actual_controller_setting='not_independently_observable',implementer_actual_override='gpt-6-astra/high',evaluation_runner_actual_override='gpt-6-astra/high',independent_admission_auditor_actual_override='gpt-6-astra/high',subagent_setting_evidence='controller_journal records confirmed spawn-tool overrides'),scientific_final_analysis='pending manual user resume and separate analyst')
    atomic_json(root/'tests/admission.json',result);print(json.dumps(dict(approved_to_launch=result['approved_to_launch'],required_failures=failures),indent=2));return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--independent-review');a=p.parse_args();assemble(a.config,a.independent_review)
