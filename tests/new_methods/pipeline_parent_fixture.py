"""Real supervisor-parent death after successful child, confined to tests."""
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.paired_v2_pipeline.common import DEFAULT_ROOT,REPO,PYTHON,atomic_json,identity,own_dependencies,process_identity,read,numeric_env
from scripts.paired_v2_pipeline.runtime import supervise

root=Path(sys.argv[1]).resolve();source=Path(sys.argv[2])
if DEFAULT_ROOT/'tests' not in root.parents:raise ValueError('fixture outside tests')
q=read(root/'fixture_contract.json');task=read(root/'fixture_task.json')
parent=process_identity();atomic_json(root/'parent_started.json',parent)
atomic_json(root/'prelaunch_dependencies.json',own_dependencies(q,task))
command=[PYTHON,'-B','-c','import sys,shutil; shutil.copyfile(sys.argv[1],sys.argv[2])',str(source),str(root/'worker_output.json')]
code,attempt=supervise(command,REPO,numeric_env(),root/'actual_worker.log',10)
if code!=0:raise RuntimeError('fixture child did not succeed')
atomic_json(root/'worker_succeeded.json',dict(parent=parent,attempt=attempt,output=identity(root/'worker_output.json')))
os._exit(23)  # Deliberately before production commit_completed; no parent receipt.
