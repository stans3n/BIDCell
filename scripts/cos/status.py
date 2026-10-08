"""Read-only COS status, with real process identity and progress."""
import argparse,json
from pathlib import Path
from .evaluation import read,times
from scripts.crgf.run_egr import is_alive


def status(config):
    cfg=read(config);root=Path(cfg['output_root']);state=read(root/'run_status.json')
    return dict(run_id=cfg['run_id'],status=state['status'],runner=state.get('runner'),runner_alive=is_alive(state['runner']) if state.get('runner') else False,current_stage=state.get('current_stage'),heartbeat=state.get('heartbeat'),stages={k:{f:v.get(f) for f in ['status','progress','log','receipt','failure','reason','wall_seconds_current_attempt']} for k,v in state['stages'].items()},pair_B1=str(root/'pair_B1/pair_ready.json'),pair_G=str(root/'pair_G/pair_ready.json'),compute_complete=str(root/'compute_complete.json'),**times())


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();print(json.dumps(status(a.config),indent=2))

if __name__=='__main__':main()
