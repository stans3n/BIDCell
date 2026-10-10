"""Idempotent manifest-driven setup; historical artifacts are read-only."""
import json,shutil,sys,subprocess
from pathlib import Path
import pandas as pd
from scripts.crgf.common import atomic_json,identity,sha256
from .config import SSPConfig
HIST=Path('/home/zwan0726/work/outputs/cos_loss_20261009T160351+1100')
BASE='541eeae61a53d84ec5543716cf4cb46bf6a292b2'

def put(path,obj):
    path=Path(path)
    if path.exists():
        if json.loads(path.read_text())!=obj:raise ValueError('bootstrap identity changed: '+str(path))
    else:atomic_json(path,obj)

def prepare(root):
    root=Path(root).resolve();repo=Path(__file__).resolve().parents[2]
    for folder in ('configs','source','tests','prepared','diagnostics'): (root/folder).mkdir(parents=True,exist_ok=True)
    if (root/'configs/queue.json').exists():
        q=json.loads((root/'configs/queue.json').read_text())
        if q.get('historical_root')!=str(HIST) or set(q['methods'])!={'S','SP'}:raise ValueError('existing bootstrap identity mismatch')
        return root/'configs/queue.json'
    oldq=json.loads((HIST/'configs/queue.json').read_text());old=json.loads((HIST/'B/configs/run.json').read_text())
    if sha256(old['checkpoint'])!='02ce88f2007eaa7a4f4de534b951a4ea0c75243bccf2804e4ea17c50d4de3b39':raise ValueError('wrong B1 parent')
    for path in (HIST/'configs').glob('*.json'):
        if path.name in ('queue.json','evaluation_protocol.json'):continue
        put(root/'configs'/path.name,json.loads(path.read_text()))
    for name in ('calibration.json','gradient_samples.json'):put(root/'diagnostics'/name,json.loads((HIST/'diagnostics'/name).read_text()))
    # This frozen evaluation population is never passed to target/prototype preparation.
    anchor=oldq['anchors']['B1'];ann=pd.read_csv(anchor['path']);ids=ann.cell_id.astype(int).tolist()
    if len(set(ids))!=len(ids):raise ValueError('duplicate B1 QC population')
    pop=root/'configs/evaluation_population.csv'
    fields=['cell_id']+[k for k in ('evaluation_label','cell_type','unmatched_reason','rejected') if k in ann]
    text=ann[fields].to_csv(index=False)
    if pop.exists() and pop.read_text()!=text:raise ValueError('population changed')
    if not pop.exists():pop.write_text(text)
    q={k:v for k,v in oldq.items() if k not in ('methods','anchors')}
    q.update(run_id=root.name.removeprefix('cos_ssp_'),output_root=str(root),python=sys.executable,methods={},anchors={'B1':anchor},historical_root=str(HIST),historical_methods={a:oldq['methods'][a] for a in ('A','B')},population=identity(pop),review_receipt=str(root/'tests/independent_review/approval.json'),resource_estimate=str(root/'tests/resources.json'),admission=str(root/'tests/admission.json'),stage_timeouts=dict(train=21600,predict=43200,postprocess=28800,matrix=3600,evaluate=14400,report=3600,b_cost=3600))
    q['reference']=str(root/'configs/reference.json');q['engineering_protocol']=str(root/'configs/protocol.json');q['paper_protocol']=str(root/'configs/evaluation_protocol.json')
    protocol=json.loads(Path(oldq['paper_protocol']).read_text());protocol.update(version='SSP-OVERNIGHT-v1.0',comparison_set=['A','B','S','SP'],comparisons=[['S','B'],['SP','S'],['SP','B'],['S','A'],['SP','A']]);put(q['paper_protocol'],protocol)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    for arm in ('S','SP'):
        ar=root/arm;(ar/'configs').mkdir(parents=True,exist_ok=True);(ar/'source').mkdir(exist_ok=True)
        for path in (root/'configs').glob('*.json'):
            if path.name!='queue.json':put(ar/'configs'/path.name,json.loads(path.read_text()))
        cfg=SSPConfig.model_validate(dict(old,arm_id=arm,method_version='SSP-OVERNIGHT-v1.0',run_id=q['run_id'],output_root=str(ar),queue_config=str(root/'configs/queue.json'),prepared_manifest=str(root/'prepared/manifest.json')))
        put(ar/'configs/run.json',cfg.model_dump(mode='json'))
        machine=json.loads((HIST/'B/configs/B_machine.json').read_text());machine.update(final_mask=str(ar/'predictions'/arm/'final.tif'),pre_mask=str(ar/'predictions'/arm/'pre.tif'),final_matrix=str(ar/'matrix'/arm/'expr_mat.csv'));put(ar/f'configs/{arm}_machine.json',machine)
        q['methods'][arm]=dict(config=str(ar/'configs/run.json'),worktree=str(repo),commit=commit,machine=str(ar/f'configs/{arm}_machine.json'),matrix=str(ar/'matrix'/arm),evaluation=str(ar/'evaluation'/arm),cost_root=str(ar))
    put(root/'configs/queue.json',q)
    put(root/'source/bootstrap.json',dict(version='SSP-OVERNIGHT-v1.0',base_commit=BASE,historical_acceptance=identity(HIST/'final_acceptance.json'),parent=identity(old['checkpoint']),population=identity(pop)))
    return root/'configs/queue.json'

if __name__=='__main__':print(prepare(sys.argv[1]))

def provisional_freeze(root):
    """Stable test-only dependencies; final reviewed freeze is separately required."""
    root=Path(root)
    if (root/'source/frozen.json').exists():raise ValueError('cannot replace dependencies after final freeze')
    for arm in ('S','SP'):
        ar=root/arm
        paths={p.name:identity(p) for p in (ar/'configs').glob('*.json') if p.name not in ('frozen.json','queue.json')}
        prepared=json.loads((root/'prepared/manifest.json').read_text())
        paths['prepared_manifest']=identity(root/'prepared/manifest.json')
        paths.update({'prepared_'+k:v for k,v in prepared['outputs'].items()})
        put(ar/'configs/frozen.json',dict(version='SSP-provisional-test-inputs-v1',reviewed_for_formal_launch=False,identities=paths))
