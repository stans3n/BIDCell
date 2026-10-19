"""Failure-safe stage gateway and incomplete JSONL recovery fixtures."""
import argparse
import subprocess
import sys
from unittest import mock
from .common import *
from .stages import repair_uncommitted_training_log,validate_stage,validate_layout,STAGES
from .training import rewrite_sequence,train


def check(root=DEFAULT_ROOT):
    root=Path(root);records={}
    for run in RUNS:
        cfg=load_config(config_path(run,root));out=root/'tests/gateway'/run
        (out/'checkpoints').mkdir(parents=True,exist_ok=True)
        src=root/'tests/pilots'/run/'checkpoints/final.pth';dst=out/'checkpoints/latest.pth'
        if not dst.exists():os.link(src,dst)
        state=torch.load(dst,map_location='cpu');rows=state['effective_sequence'];log=out/'effective_sequence.jsonl'
        rewrite_sequence(log,rows)
        with log.open('a') as f:f.write('{"partial_uncommitted_write":')
        recovered=repair_uncommitted_training_log(cfg,out)
        assert recovered['status']=='recovered_uncommitted_tail'
        assert [json.loads(line) for line in log.read_text().splitlines()]==rows
        assert sha256(dst)==sha256(src)
        bad=copy.deepcopy(rows);bad[0]['K']+=1;rewrite_sequence(log,bad)
        try:repair_uncommitted_training_log(cfg,out)
        except ValueError as e:prefix_error=str(e)
        else:raise AssertionError('changed durable log prefix accepted')
        rewrite_sequence(log,rows[:-1])
        try:repair_uncommitted_training_log(cfg,out)
        except ValueError as e:truncated_error=str(e)
        else:raise AssertionError('truncated durable log prefix accepted')
        rewrite_sequence(log,rows)
        # Exercise the admission stop before model construction/file creation.
        # Environment/input checks are independently tested on the real host.
        with mock.patch('scripts.paired_v2.training.check_environment'),mock.patch('scripts.paired_v2.prepare.verify_inputs'):
            try:train(cfg)
            except FileNotFoundError as e:
                assert str(root/'admission'/f'{run}.json') in str(e)
                no_admission=True
            else:raise AssertionError('unadmitted formal train accepted')
        missing=[]
        for stage in STAGES:
            try:validate_stage(cfg,stage,require_bound=True)
            except FileNotFoundError:missing.append(stage)
            else:raise AssertionError('missing formal stage marked completed')
        assert not (Path(cfg['output_root'])/'checkpoints').exists()
        redirects=[]
        for key,value in [('output_root',str(INPUT)),('output_root',str(root/'unrelated')),('config_path',str(INPUT/'configs/queue.json')),('machine',str(INPUT/'V2_MORPH/configs/machine.json'))]:
            changed=dict(cfg);changed[key]=value
            try:validate_layout(changed)
            except ValueError:redirects.append(dict(key=key,value=value))
            else:raise AssertionError('unsafe output/config redirect accepted')
        records[run]=dict(partial_uncommitted_line_recovered=recovered,
            corrupted_committed_prefix_rejected=prefix_error,truncated_committed_prefix_rejected=truncated_error,
            exact_checkpoint_unchanged=True,unadmitted_formal_training_rejected=no_admission,
            incomplete_formal_stages_rejected=missing,unsafe_layout_redirects_rejected=redirects,formal_train_never_started=True)
    cli={}
    for command in (['--help'],['train','--help'],['validate','--help']):
        p=subprocess.run([sys.executable,'-m','scripts.paired_v2.cli',*command],capture_output=True,text=True,check=True)
        cli[' '.join(command)]=p.stdout
    result=dict(status='passed',arms=records,cli_help=cli,test_source=identity(__file__),
        scope='real saved pilot checkpoint and filesystem log recovery fixtures; admission-stop unit path mocks only separately checked environment/input preflight; no formal training')
    atomic_json(root/'tests/gateway.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=DEFAULT_ROOT);a=p.parse_args();print(check(a.root)['status'])
