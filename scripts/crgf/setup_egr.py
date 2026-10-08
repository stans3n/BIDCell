"""Freeze explicit method inputs, code bindings and the five-method manifest."""
import argparse,json,shutil,subprocess,sys
from pathlib import Path
from .common import atomic_json,identity,sha256
from .config import RunConfig
from .prepare_egr import prepare_method,HIST
from .egr_config import QueueConfig

def setup(root,run_id,worktrees):
 root=Path(root);root.mkdir(parents=True,exist_ok=True)
 if (root/'run_status.json').exists():raise ValueError('setup cannot overwrite a launched queue')
 checkout=Path(__file__).resolve().parents[2]
 base=json.loads((HIST/'configs/run.json').read_text());methods={};manifest={};artifacts={}
 for method,tree in worktrees.items():
  tree=Path(tree).resolve();contract=json.loads((tree/'scripts/crgf/configs/variant_contract.json').read_text())
  if contract['variant']!=method:raise ValueError('wrong branch variant')
  cfg=RunConfig.model_validate_json((tree/f'scripts/crgf/configs/egr_{method}.json').read_text())
  if cfg.variant!=method or cfg.run_id!=run_id or Path(cfg.output_root)!=root/method:raise ValueError('committed run binding mismatch')
  prepare_method(cfg,check_views=True)
  commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=tree,text=True).strip();branch=subprocess.check_output(['git','branch','--show-current'],cwd=tree,text=True).strip()
  binding=dict(variant=method,worktree=str(tree),branch=branch,commit=commit);atomic_json(root/method/'source/binding.json',binding)
  methods[method]=dict(worktree=str(tree),config=str(root/method/'configs/run.json'),commit=commit)
  manifest[method]=dict(evaluation=str(root/method/'evaluation'/method),machine=str(root/method/f'configs/{method}_machine.json'),matrix=str(root/method/'matrix'/method),cost_root=str(root/method),run_status=str(root/'run_status.json'))
  subprocess.run([sys.executable,'-m','scripts.crgf.run_stage1','--config',methods[method]['config'],'--freeze'],cwd=tree,env=dict(__import__('os').environ,PYTHONPATH=str(tree)),check=True)
 for name in ('reference.json','protocol.json','B0_machine.json'):
  shutil.copyfile(HIST/'configs'/name,root/'configs'/name)
 for method,oldname in [('B1','B1'),('GR','F1')]:
  machine=json.loads((HIST/f'configs/{oldname}_machine.json').read_text());atomic_json(root/f'configs/{method}_machine.json',machine)
  matrix=HIST/'matrix'/oldname
  validation=json.loads((matrix/'validation.json').read_text())
  for key,obj in dict(validation['outputs'],mask=validation['mask']).items():
   if sha256(obj['path'])!=obj['sha256']:raise ValueError('historical identity mismatch '+obj['path'])
   artifacts[f'{method}:{key}']=obj
  manifest[method]=dict(evaluation=str(root/'reevaluation'/method),machine=str(root/f'configs/{method}_machine.json'),matrix=str(matrix),cost_root=str(HIST),historical_name=oldname,run_status=str(HIST/'run_status.json'))
 atomic_json(root/'input_manifest.json',dict(historical_artifacts=artifacts,B0=identity(base['checkpoint']),views=identity(HIST/'configs/views.json'),plan=identity('/home/zwan0726/work/docs/BIDCell_CRGF_EGR_Ablation_Execution_Plan_20261008.md')))
 atomic_json(root/'configs/evaluation_methods.json',dict(methods=manifest,comparison_set=['B1','E','G','R','GR']))
 protocol=checkout/'scripts/crgf/configs/egr_evaluation_protocol.json'
 cfg=QueueConfig(run_id=run_id,output_root=str(root),methods=methods,paper_protocol=str(protocol),prepared=str(root/'shared/paper_reference'))
 atomic_json(root/'configs/queue.json',cfg.model_dump())
 # Immutable scientific protocol snapshot is an additional artifact; runtime
 # dependency still names frozen checkout protocol to preserve preparation hash.
 shutil.copyfile(protocol,root/'configs/evaluation_protocol.json')
 atomic_json(root/'source/worktrees.json',dict(common=str(checkout),methods=methods))
 print(root/'configs/queue.json')

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--run-id',required=True);p.add_argument('--worktrees',required=True,help='JSON object E/G/R to absolute fixed worktree');a=p.parse_args();setup(a.root,a.run_id,json.loads(Path(a.worktrees).read_text()))
if __name__=='__main__':main()
