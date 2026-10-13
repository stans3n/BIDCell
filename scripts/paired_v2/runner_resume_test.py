"""Production train-entry resume with a durable 25-update checkpoint and log tail."""
import argparse
import json
from pathlib import Path
from .common import *
from .training import train,validate_cursor,sequence_identity


def check(root=DEFAULT_ROOT):
    root=Path(root);results={}
    for run in RUNS:
        out=root/'tests/runner_resume'/run;cfg=config_path(run,root)
        if out.exists():raise FileExistsError('preserve existing runner resume evidence')
        first=train(cfg,25,out);cp=out/'checkpoints/latest.pth';checkpoint25=identity(cp)
        # Model a process that wrote another attempted row before its next
        # durable checkpoint. Resume must discard that uncommitted suffix.
        with (out/'effective_sequence.jsonl').open('a') as f:
            f.write(json.dumps({'uncommitted_tail_fixture':True})+'\n')
        second=train(cfg,100,out);state=torch.load(out/'checkpoints/final.pth',map_location='cpu');validate_cursor(state)
        rows=[json.loads(x) for x in (out/'effective_sequence.jsonl').read_text().splitlines()]
        if len(rows)!=100 or any('uncommitted_tail_fixture' in x for x in rows) or rows!=state['effective_sequence']:
            raise ValueError('production training runner did not reconstruct its committed sequence')
        baseline=read(root/'tests/pilots'/run/'checkpoints/complete.json')
        expected=[json.loads(x) for x in Path(baseline['outputs']['sequence']['path']).read_text().splitlines()]
        keys=('update','cursor','index','path','row','col','candidate_ids','K','flip','rot90','patch_sha256')
        if [{k:r[k] for k in keys} for r in rows]!=[{k:r[k] for k in keys} for r in expected]:
            raise ValueError('resumed production runner external sequence differs from uninterrupted pilot')
        results[run]=dict(status='passed',checkpoint25=checkpoint25,checkpoint100=identity(out/'checkpoints/final.pth'),
            restored_at_update=25,final_updates=100,final_cursor=state['view_cursor'],empty_skips=state['empty_skips'],
            attempted_log_tail_discarded=True,no_duplicate_effective_rows=True,external_sequence_exact=True,
            scope='actual train entry called for 25 then resumed to100; crash-window log tail simulated, no OS process termination claimed')
    result=dict(status='passed',arms=results,producer=producer('train'),test_source=identity(__file__))
    atomic_json(root/'tests/runner_resume.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=DEFAULT_ROOT);a=p.parse_args();print(check(a.root)['status'])
