"""Actual host tmux lifecycle check; engineering-only serial cursor fixtures."""
import json
from pathlib import Path
import subprocess
import sys
import time
from scripts.boundary_three.common import atomic_json,identity,read,source_manifest,digest,utc
from scripts.boundary_three.contracts import NODES,PYTHON


def main(root):
    root=Path(root).resolve();evidence=[]
    def command(*args,success=True):
        cmd=[PYTHON,'-m','tests.boundary_three.fixture_runtime',*args]
        proc=subprocess.run(cmd,text=True,capture_output=True,timeout=40)
        evidence.append(dict(command=cmd,exit_code=proc.returncode,stdout=proc.stdout,stderr=proc.stderr,time=utc()))
        if success and proc.returncode:raise AssertionError(proc.stderr)
        if not success and not proc.returncode:raise AssertionError('duplicate launch unexpectedly succeeded')
        return proc
    command('init',str(root));cfg=str(root/'configs/run.json')
    command('launch','--config',cfg)
    command('launch','--config',cfg,success=False)
    deadline=time.monotonic()+35
    while time.monotonic()<deadline:
        value=json.loads(command('status','--config',cfg,'--json').stdout)
        if value['current_node']=='P2_BASE' and (value.get('progress') or {}).get('tile_cursor',0)>=2:break
        time.sleep(.5)
    else:raise AssertionError('second serial node never made actual progress')
    command('status','--config',cfg);command('tail','--config',cfg,'--lines','10')
    command('safe-stop','--config',cfg,'--reason','engineering lifecycle safe-stop')
    deadline=time.monotonic()+20
    while time.monotonic()<deadline:
        value=json.loads(command('status','--config',cfg,'--json').stdout)
        if not value['process_alive']:break
        time.sleep(.5)
    else:raise AssertionError('safe stop did not finish')
    assert value['execution_status']=='PAUSED_USER',value
    a1=identity(root/'A1_CORE/complete.json');a1visits=(root/'A1_CORE/visits.log').read_text()
    p2cursor=read(root/'P2_BASE/fixture_checkpoint.json')['cursor']
    command('resume','--config',cfg,'--budget-hours',str(120/3600))
    deadline=time.monotonic()+55
    while time.monotonic()<deadline:
        value=json.loads(command('status','--config',cfg,'--json').stdout)
        if not value['process_alive']:break
        time.sleep(.5)
    else:raise AssertionError('resumed queue did not exit')
    assert value['execution_status']=='FINISHED',value
    assert identity(root/'A1_CORE/complete.json')==a1
    assert (root/'A1_CORE/visits.log').read_text()==a1visits
    for n in NODES:
        if n=='A2_CORE':
            assert not (root/n/'complete.json').exists();continue
        assert (root/n/'visits.log').read_text().splitlines()==list(map(str,range(40 if n=='P2_BASE' else 12))),n
    result=json.loads(command('complete-check','--config',cfg).stdout)
    assert result['minimum_three_complete'] and result['all_enabled_applicable_complete']
    command('status','--config',cfg);command('tail','--config',cfg,'--node','MORPH_NO_GEO','--lines','6')
    events=[json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
    started=[r['node'] for r in events if r['event']=='NODE_PROCESS_STARTED']
    expected=['A1_CORE','P2_BASE','P2_BASE','MORPH_NO_DAPI','P2_MORPH','MORPH_SIMPLE_FUSION','MORPH_NO_GEO']
    assert started==expected,started
    receipt=dict(schema='boundary-three-runtime-engineering-mini-v1',status='passed',scope='engineering-only; no scientific validation/efficacy or GPU',
        created_utc=utc(),host=value['host'],config=identity(cfg),source_digest=digest(source_manifest()),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        real_tmux_detached_launch=True,duplicate_launch_refused=True,safe_stop_status='PAUSED_USER',
        resumed_p2_cursor=p2cursor,completed_a1_receipt_unchanged=True,no_duplicate_fixture_visits=True,
        actual_start_order=started,negative_a1_continued=True,negative_p2_base_did_not_cancel_p2_morph=True,
        scientific_a2_skip_verified=True,complete_check=result,commands=evidence)
    atomic_json(root/'lifecycle_receipt.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!='commands'},indent=2))


if __name__=='__main__':main(sys.argv[1])
