"""One serial, detached orchestrator; receipts control idempotent resume."""
import argparse,fcntl,json,os,signal,sys,threading,time,traceback
from pathlib import Path
from .config import STAGES,prepare,root_check,specs_and_inputs
from scripts.rep1_triple_simple_boundary.common import atomic_json,identity,check_identity,read,digest
from . import stages,compute

class Runner:
 def __init__(self,cfg):
  self.cfg=cfg;self.root=Path(cfg['run_dir']);self.mutex=threading.Lock();self.stop=threading.Event();self.last_validation=0
  self.state=dict(run_id=cfg['run_id'],state='RUNNING',stage=None,completed_stages=[],pid=os.getpid(),ppid=os.getppid(),pgid=os.getpgid(0),sid=os.getsid(0),pid_namespace=os.readlink('/proc/self/ns/pid'),process_start_ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19],started_epoch=time.time(),heartbeat_epoch=time.time(),exit_code=None,phase_II_manual_resume_required=True)
  self.source_stats={x['path']:(Path(x['path']).stat().st_size,Path(x['path']).stat().st_mtime_ns) for x in cfg['source_files']}
  self.lock=open(self.root/'pipeline.lock','a+');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  self.lock_fds=[self.lock.fileno()];self.fatal=None
 def save(self,**updates):
  with self.mutex:self.state.update(updates);atomic_json(self.root/'STATUS.json',self.state)
 def validate(self,force=False):
  if self.fatal:raise RuntimeError(self.fatal)
  if not force and time.time()-self.last_validation<10:return
  for p,expected in self.source_stats.items():
   st=Path(p).stat()
   if (st.st_size,st.st_mtime_ns)!=expected:raise RuntimeError('immutable code snapshot changed '+p)
  if os.statvfs(self.root).f_bavail*os.statvfs(self.root).f_frsize<self.cfg['disk_reserve_bytes']:raise RuntimeError('disk reserve below 10GiB')
  mem={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if len(line.split())>=3 and line.split()[1].isdigit()}
  if mem['MemAvailable']<self.cfg['ram_reserve_bytes']:raise RuntimeError('RAM reserve below 6GiB')
  os.fstat(self.lock.fileno());self.last_validation=time.time()
 def heartbeat(self):
  while not self.stop.wait(10):
   try:self.save(heartbeat_epoch=time.time());self.validate()
   except BaseException as e:self.fatal=repr(e)
 def progress(self,name,cursor=None,total=None,**extra):
  if isinstance(name,dict):
   event=name;name=event['stage'];cursor=event['cursor'];total=event['total'];extra=dict(event.get('detail',{}),**extra)
  self.validate();self.save(progress=dict(name=name,cursor=cursor,total=total,**extra),heartbeat_epoch=time.time())
  print(json.dumps(dict(time=time.time(),stage=self.state['stage'],name=name,cursor=cursor,total=total,**extra)),flush=True)
 def stage(self,n):
  sid=f'{n:02d}';self.validate(force=True);dest=stages.stage(self.cfg,n);dest.mkdir(exist_ok=True)
  rp=self.root/'receipts'/f'{sid}.json';binding=dict(manifest=identity(self.root/'RUN_MANIFEST.json'),source_digest=self.cfg['source_digest'],stage=sid)
  if rp.exists():
   r=read(rp)
   if r.get('status')!='succeeded' or r.get('binding')!=binding:raise ValueError('stage resume binding/status mismatch '+sid)
   for x in r['outputs']:check_identity(x)
   self.state['completed_stages'].append(sid);self.save();return
  self.save(stage=sid,stage_name=STAGES[n],stage_started_epoch=time.time(),progress=None)
  start=time.time();atomic_json(self.root/'receipts'/f'{sid}_start.json',dict(started_epoch=start,binding=binding,command=[self.cfg['python'],'-B','-m','scripts.boundary_goal_v3.pipeline','--run-dir',str(self.root),'--resume'],stage_action=STAGES[n]))
  funcs={0:stages.preflight,1:stages.global_metrics,2:stages.local_curves,3:stages.transfers,4:stages.initial_ci,9:stages.beta_local,10:stages.aggregate,11:stages.package}
  if n in funcs:outputs=funcs[n](self.cfg,self.progress)
  elif n in (5,6):compute.replay(self.cfg,dest,1.0 if n==5 else 0.0,self.progress);outputs=[p for p in dest.rglob('*') if p.is_file()]
  elif n==7:compute.build_matrix(self.cfg,dest,self.progress);outputs=[p for p in dest.rglob('*') if p.is_file()]
  elif n==8:compute.evaluate(self.cfg,dest,self.progress,self.lock_fds,self.validate);outputs=[p for p in dest.rglob('*') if p.is_file() and 'tmp' not in p.relative_to(dest).parts]
  else:raise ValueError(n)
  ids=[identity(p) for p in outputs if Path(p).is_file()]
  if not ids:raise ValueError('stage has no committed artifacts')
  atomic_json(rp,dict(status='succeeded',stage=sid,name=STAGES[n],binding=binding,started_epoch=start,finished_epoch=time.time(),wall_seconds=time.time()-start,exit_code=0,outputs=ids))
  self.state['completed_stages'].append(sid);self.save(heartbeat_epoch=time.time())
 def run(self,resume):
  self.save();worker=threading.Thread(target=self.heartbeat,daemon=True);worker.start()
  result=1
  try:
   for x in self.cfg['source_files']:check_identity(x)
   for n in range(len(STAGES)):
    try:self.stage(n)
    except Exception as e:
     if n!=5:raise
     # Beta1 regression blocks beta0 only; package valid stages00–04.
     atomic_json(self.root/'receipts/05_failed.json',dict(status='BLOCKED',exit_code=2,error=repr(e),traceback=traceback.format_exc()))
     stages.aggregate(self.cfg,self.progress,partial=True);outputs=stages.package(self.cfg,self.progress,partial=True)
     atomic_json(self.root/'receipts/final.json',dict(state='PARTIAL_BLOCKED',exit_code=2,outputs=[identity(p) for p in outputs]));self.save(state='PARTIAL_BLOCKED',exit_code=2,error=repr(e));result=2;break
   else:
    outputs=read(self.root/'receipts/11.json')['outputs']
    for x in outputs:check_identity(x)
    atomic_json(self.root/'receipts/final.json',dict(state='COMPLETED',exit_code=0,finished_epoch=time.time(),completed_stages=self.state['completed_stages'],outputs=outputs))
    self.save(state='COMPLETED',exit_code=0,finished_epoch=time.time(),phase='WAITING_FOR_USER_RESUME',evidence_attribute='REFERENCE-CONSISTENCY PROXY',BLCR_status='NOT_DEFINED / NOT_APPLICABLE')
    result=0
  except BaseException as e:
   tb=traceback.format_exc();print(tb,file=sys.stderr,flush=True)
   atomic_json(self.root/'receipts/failure.json',dict(state='FAILED',stage=self.state['stage'],exit_code=1,error=repr(e),traceback=tb,finished_epoch=time.time()))
   self.save(state='FAILED',exit_code=1,error=repr(e),finished_epoch=time.time());result=1
  finally:
   self.stop.set();worker.join(timeout=2)
   atomic_json(self.root/'process_exit.json',dict(pid=os.getpid(),exit_code=result,finished_epoch=time.time()))
  return result

def dry_run(run):
 root=root_check(run)
 cfg=read(root/'RUN_MANIFEST.json') if (root/'RUN_MANIFEST.json').exists() else None
 if cfg is None:old,inputs,specs=specs_and_inputs()
 else:old,inputs,specs=cfg['old_config'],cfg['inputs'],cfg['specs']
 for p in (old['python'],old['rscript'],inputs['evaluation']['protocol'],inputs['evaluation']['reference_model'],specs['P2_MORPH_GT']['final'],specs['P2_MORPH']['final']):
  if not Path(p).is_file():raise FileNotFoundError(p)
 print(json.dumps(dict(mode='COMMAND_DRY_RUN',run_dir=str(root),python=old['python'],rscript=old['rscript'],stages=[dict(id=f'{i:02d}',action=x,output=str(root/'stages'/f'{i:02d}')) for i,x in enumerate(STAGES)],command=['bash',str(root/'run_boundary_goal_v3.sh')],beta1_regression_blocks_beta0=True,manual_resume_before_Phase_II=True),indent=2))
 return 0

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--init',action='store_true');p.add_argument('--dry-run',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--smoke',action='store_true');a=p.parse_args();root=root_check(a.run_dir)
 if a.dry_run:return dry_run(root)
 if a.init:prepare(root,Path(__file__).resolve().parents[2]);return 0
 if a.smoke:
  old,inputs,specs=specs_and_inputs();cfg=dict(run_id='SMOKE',old_config=old,inputs=inputs,specs=specs,source_digest='SMOKE');compute.replay(cfg,root,1.0,lambda *a,**k:None,smoke=True);return 0
 cfg=read(root/'RUN_MANIFEST.json')
 if Path(__file__).resolve().parents[2]!=Path(cfg['source_snapshot']):raise RuntimeError('production must run immutable CODE_SNAPSHOT')
 if not (root/'CODE_REVIEW.md').is_file() or 'CODE_REVIEW_PASS' not in (root/'CODE_REVIEW.md').read_text():raise RuntimeError('independent review pass required')
 if (root/'STATUS.json').exists() and read(root/'STATUS.json')['state']!='QUEUED' and not a.resume:raise RuntimeError('use --resume for existing run')
 return Runner(cfg).run(a.resume)

if __name__=='__main__':sys.exit(main())
