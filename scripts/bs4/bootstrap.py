"""Manifest-driven four-arm configuration; all historical inputs remain read-only."""
import json,sys,shutil,subprocess
from pathlib import Path
from scripts.crgf.common import atomic_json,identity,sha256,verify_outputs
from .config import BS4Config,ARMS
HIST=Path('/home/zwan0726/work/outputs/cos_loss_20261009T160351+1100')
SSP=Path('/home/zwan0726/work/outputs/cos_ssp_20261011T005741+1100')
REPO=Path(__file__).resolve().parents[2]

def put(path,obj):
    path=Path(path)
    if path.exists():
        if json.loads(path.read_text())!=obj:raise ValueError('existing identity changed '+str(path))
    else:atomic_json(path,obj)

def prepare(root):
    root=Path(root).resolve()
    oldq=json.loads((SSP/'configs/queue.json').read_text());bq=json.loads((HIST/'configs/queue.json').read_text())
    old=json.loads((SSP/'S/configs/run.json').read_text());old.pop('prototype_version',None)
    if sha256(old['checkpoint'])!='02ce88f2007eaa7a4f4de534b951a4ea0c75243bccf2804e4ea17c50d4de3b39':raise ValueError('B1 checkpoint mismatch')
    for d in ['configs','source','prepared','diagnostics','tests']:(root/d).mkdir(parents=True,exist_ok=True)
    pop=SSP/'configs/evaluation_population.csv'
    if sha256(pop)!='6c3505d5b0a70d3977bd88d4cda61281ecec3fa924fdf4cfa63cf2b8720abd0b':raise ValueError('population hash')
    dest=root/'configs/evaluation_population.csv'
    if dest.exists():
        if sha256(dest)!=sha256(pop):raise ValueError('population drift')
    else:shutil.copyfile(pop,dest)
    for name in ['reference.json','protocol.json','anchors.json','inputs.json','views.json','probes.json','training_probes.json','patches_shift0.json','patches_shift24.json']:
        put(root/'configs'/name,json.loads((SSP/'S/configs'/name).read_text()))
    for name in ['calibration.json','gradient_samples.json']:put(root/'diagnostics'/name,json.loads((HIST/'diagnostics'/name).read_text()))
    prepared=dict(version='BS4-frozen-teacher-inputs-v2',status='succeeded',gene_order=json.loads((root/'configs/inputs.json').read_text())['gene_order'],teacher_source=identity(REPO/'scripts/ssp/method.py'),outputs={k:v for k,v in json.loads((root/'configs/inputs.json').read_text()).items() if isinstance(v,dict) and 'sha256' in v and 'path' in v})
    verify_outputs(prepared);put(root/'prepared/manifest.json',prepared)
    q={k:v for k,v in oldq.items() if k not in ('methods','historical_methods','population','review_receipt','resource_estimate','admission')}
    q.update(version='BS4-OVERNIGHT-v2.0',run_id=root.name.removeprefix('cos_bs4_'),output_root=str(root),python=sys.executable,methods={},historical_methods={'B':bq['methods']['B'],'S':oldq['methods']['S']},historical_root=str(HIST),historical_ssp_root=str(SSP),population=identity(dest),arm_order=list(ARMS),review_receipt=str(root/'tests/independent_review/approval.json'),resource_estimate=str(root/'tests/resources.json'),admission=str(root/'tests/admission.json'),stage_timeouts=dict(train=21600,predict=43200,postprocess=28800,matrix=3600,evaluate=14400,report=14400))
    for key,name in [('reference','reference.json'),('engineering_protocol','protocol.json'),('paper_protocol','evaluation_protocol.json')]:q[key]=str(root/'configs'/name)
    protocol=json.loads(Path(oldq['paper_protocol']).read_text());protocol.update(version='BS4-OVERNIGHT-v2.0',comparison_set=['B','S','T1','T2','P1','P2'],comparisons=[['P1','B'],['P2','B'],['T1','B'],['T2','B'],['T2','T1'],['T1','S'],['T2','S']]);put(q['paper_protocol'],protocol)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True,timeout=30).strip()
    for arm in ARMS:
        ar=root/arm;(ar/'configs').mkdir(parents=True,exist_ok=True);(ar/'source').mkdir(exist_ok=True)
        for path in (root/'configs').glob('*.json'):
            if path.name!='queue.json':put(ar/'configs'/path.name,json.loads(path.read_text()))
        cfg=BS4Config.model_validate(dict(old,arm_id=arm,method_version='BS4-OVERNIGHT-v2.0',run_id=q['run_id'],output_root=str(ar),queue_config=str(root/'configs/queue.json'),prepared_manifest=str(root/'prepared/manifest.json'),alpha=.25 if arm=='T2' else 0.,beta=1.,post_rule=arm if arm.startswith('P') else 'B',reused_prediction_root=str(HIST/'B/predictions/B') if arm.startswith('P') else None))
        put(ar/'configs/run.json',cfg.model_dump(mode='json'))
        machine=json.loads((HIST/'B/configs/B_machine.json').read_text());machine.update(final_mask=str(ar/'predictions'/arm/'final.tif'),pre_mask=str(ar/'predictions'/arm/'pre.tif'),final_matrix=str(ar/'matrix'/arm/'expr_mat.csv'));put(ar/f'configs/{arm}_machine.json',machine)
        q['methods'][arm]=dict(config=str(ar/'configs/run.json'),worktree=str(REPO),commit=commit,machine=str(ar/f'configs/{arm}_machine.json'),matrix=str(ar/'matrix'/arm),evaluation=str(ar/'evaluation'/arm),cost_root=str(ar))
        if not (ar/'configs/frozen.json').exists():put(ar/'configs/frozen.json',dict(version='BS4-inputs-v2',identities={p.name:identity(p) for p in (ar/'configs').glob('*.json') if p.name!='frozen.json'}))
    queue=root/'configs/queue.json'
    if queue.exists():
        prior=json.loads(queue.read_text())
        # Code commit binding is finalized separately, before source/admission freeze.
        for arm in ARMS:q['methods'][arm]['commit']=prior['methods'][arm]['commit']
    put(queue,q)
    return root/'configs/queue.json'

if __name__=='__main__':print(prepare(sys.argv[1]))
