#!/usr/bin/env python3
"""Read-only status uses /proc identity, terminal receipt and output hashes."""
import argparse,hashlib,json,os,time
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def inspect(run,verify=False):
 run=Path(run);s=json.loads((run/'STATUS.json').read_text());pid=s.get('pid');alive=False
 if pid:
  try:
   stat=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
   alive=stat[0]!='Z' and str(stat[19])==str(s.get('process_start_ticks'))
  except (OSError,IndexError):pass
 if s['state']=='RUNNING' and pid and not alive and s.get('pid_namespace')==os.readlink('/proc/self/ns/pid'):
  s['state']='FAILED';s['error']='process disappeared without terminal success receipt'
 s['process_alive']=alive;s['elapsed_seconds']=round(time.time()-s.get('started_epoch',time.time()),1)
 s['heartbeat_age_seconds']=round(time.time()-s.get('heartbeat_epoch',time.time()),1)
 s['status_note']='LIVE' if alive else 'EXITED' if s.get('exit_code') is not None else 'PROCESS_NOT_VISIBLE_OR_DEAD'
 required=['EXISTING_METHODS_MASTER_METRICS.csv','BIDCELL_STYLE_DISTANCE_CURVE.csv','BOUNDARY_LOCAL_MARKER_RESULTS.csv','MARKER_AWARE_TRANSFERS.csv','G_ONLY_MATCHED_RESULTS.csv','BOUNDARY_SPATIAL_INTERVALS.csv','RESULTS_SUMMARY.json','SHA256SUMS','boundary_evidence.zip']
 s['key_outputs']={x:(run/x).is_file() for x in required};s['zip_path']=str(run/'boundary_evidence.zip')
 if s['state'] in ('COMPLETED','COMPLETED_WITH_LIMITATIONS'):
  f=run/'receipts/final.json';ok=False
  if f.exists():
   final=json.loads(f.read_text());ok=final.get('exit_code')==0 and all(Path(x['path']).is_file() and Path(x['path']).stat().st_size==x['bytes'] for x in final['outputs'])
   if ok:ok=all(sha(x['path'])==x['sha256'] for x in final['outputs'])
   ex=run/'OS_EXIT.json';observed=json.loads(ex.read_text()) if ex.exists() else {}
   ok=ok and observed.get('exit_code')==0 and observed.get('shell_pid')==s.get('ppid') and observed.get('finished_epoch',0)>=s.get('started_epoch',0)
   if verify and ok:
    for line in (run/'SHA256SUMS').read_text().splitlines():
     expected,relative=line.split('  ',1);p=run/relative
     if not p.is_file() or sha(p)!=expected:ok=False;break
  s['final_receipt_valid']=ok
  if not ok:s['state']='INVALID_TERMINAL_RECEIPT'
 print(json.dumps(s,indent=2,ensure_ascii=False))
 return s
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run-dir',default=str(Path(__file__).resolve().parent));p.add_argument('--verify',action='store_true');a=p.parse_args();inspect(a.run_dir,a.verify)
