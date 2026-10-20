#!/usr/bin/env python3
"""Shell calls this only after observing the Python child's real exit status."""
import argparse,json,os,tempfile,time
from pathlib import Path

def write(path,obj):
 fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
 with os.fdopen(fd,'w') as f:json.dump(obj,f,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--exit-code',type=int,required=True);a=p.parse_args();r=Path(a.run_dir);write(r/'OS_EXIT.json',dict(exit_code=a.exit_code,actual_child_exit_observed=True,finished_epoch=time.time(),shell_pid=os.getppid()))
 s=json.loads((r/'STATUS.json').read_text())
 if a.exit_code!=0 and s['state']!='PARTIAL_BLOCKED':s.update(state='FAILED',exit_code=a.exit_code,error=s.get('error','shell observed nonzero child exit'));write(r/'STATUS.json',s)
