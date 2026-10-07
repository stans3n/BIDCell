#!/usr/bin/env python3
"""Read-only queue status, heartbeat and actual progress reported separately."""
import argparse
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',required=True);a=p.parse_args();root=Path(a.run_root)
    state=json.loads((root/'run_status.json').read_text())
    print(json.dumps({k:state.get(k) for k in ['run_id','status','runner','current_stage','heartbeat','computation_finished','unfinished']},indent=2))
    for name,r in state['stages'].items():
        print(name,r['status'],'attempts',len(r.get('attempts',[])))
        if r.get('progress'):print('  progress:',json.dumps(r['progress']))
        if r.get('log'):print('  log:',r['log'])
        if r.get('failure'):print('  failure:',r['failure'][-2000:])
if __name__=='__main__':main()
