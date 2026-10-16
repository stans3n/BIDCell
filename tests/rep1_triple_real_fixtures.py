"""Fixed real small-ROI and one real maximum prediction batch, never full slide."""
import argparse,fcntl,os,random,shutil,threading,time,resource
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
from scripts.rep1_triple_simple_boundary.common import *
from scripts.rep1_triple_simple_boundary import p2
from scripts.rep1_triple_simple_boundary.evaluation import run_r
from scripts.rep1_triple_simple_boundary.inputs import matrix_arrays
ROOT=Path(__file__).resolve().parents[1]

def process_memory_tree(pid,lock_objects=()):
    children=[pid];total=private=0;seen=set();locks=[]
    for parent in children:
        if parent in seen:continue
        seen.add(parent)
        try:
            for task in Path('/proc',str(parent),'task').iterdir():
                children.extend(int(x) for x in (task/'children').read_text().split())
            stats={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc',str(parent),'smaps_rollup').read_text().splitlines()[1:] if ':' in line}
            total+=stats.get('Pss',0);private+=stats.get('Private_Clean',0)+stats.get('Private_Dirty',0)
            start=Path('/proc',str(parent),'stat').read_text().rsplit(')',1)[1].split()[19];found=[]
            for fd in Path('/proc',str(parent),'fd').iterdir():
                try:
                    st=fd.stat()
                    for obj in lock_objects:
                        if (st.st_dev,st.st_ino)==(obj['device'],obj['inode']):found.append(dict(fd=int(fd.name),path=str(fd.resolve()),lock=obj))
                except (FileNotFoundError,ProcessLookupError):pass
            locks.append(dict(pid=parent,start_ticks=start,lock_fds=found))
        except (FileNotFoundError,ProcessLookupError):pass
    return dict(processes=len(seen),pss_bytes=total,private_bytes=private,locks=locks)

class MemorySampler:
    def __init__(self,lock_fds=()):
        self.lock_objects=[dict(device=os.fstat(fd).st_dev,inode=os.fstat(fd).st_ino,path=str(Path('/proc/self/fd',str(fd)).resolve())) for fd in lock_fds]
    def __enter__(self):
        self.done=False;self.samples=[];self.error=None;self.thread=threading.Thread(target=self.sample,daemon=True);self.thread.start();return self
    def sample(self):
        try:
            while not self.done:self.samples.append(dict(time=time.monotonic(),**process_memory_tree(os.getpid(),self.lock_objects)));time.sleep(.5)
        except BaseException as error:self.error=error
    def __exit__(self,*args):
        self.done=True;self.thread.join()
        if self.error is not None:raise self.error
    def result(self):
        if self.error is not None:raise RuntimeError('resource sampler failed; measurement incomplete') from self.error
        if not self.samples:raise RuntimeError('resource sampler produced no observations')
        evidence={}
        for sample in self.samples:
            for row in sample['locks']:
                key=(row['pid'],row['start_ticks']);old=evidence.get(key)
                if old is None or len(row['lock_fds'])>len(old['lock_fds']):evidence[key]=row
        return dict(peak_pss_bytes=max(x['pss_bytes'] for x in self.samples),peak_private_bytes=max(x['private_bytes'] for x in self.samples),peak_processes=max(x['processes'] for x in self.samples),samples=len(self.samples),inherited_lock_evidence=list(evidence.values()))

def mini(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);ins=read('/home/zwan0726/work/docs/BIDCell_Rep1_Triple_Simple_Inputs_20261017.json');f=ins['endpoints']['MORPH_SIMPLE_FUSION']['files']
    for k in ('pre_mask','nuclei_mask'):check_identity(f[k])
    pre=tifffile.imread(f['pre_mask']['path']);nuc=tifffile.imread(f['nuclei_mask']['path']);y,x,size=4096,0,128
    atomic_tiff(out/'pre.tif',pre[y:y+size,x:x+size]);atomic_tiff(out/'nuc.tif',nuc[y:y+size,x:x+size])
    finals=[];lock=Path('/home/zwan0726/work/outputs/cos_gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with MemorySampler() as memory:
        for workers in (1,4):
            native=out/f'native_w{workers}';candidate=out/f'P2_w{workers}'
            kw=dict(workers=workers,origin=(y,x),patch_size=96,allowed_root=out,dependency_context={'fixture':'real_Simple_ROI'},lock_fds=(lock.fileno(),))
            p2.process(out/'pre.tif',out/'nuc.tif',f['nuclei_mask']['path'],native,'B',**kw)
            if workers==4:
                try:p2.process(out/'pre.tif',out/'nuc.tif',f['nuclei_mask']['path'],candidate,'P2',native_pred=native,interrupt_after=('tiles',1),**kw)
                except InterruptedError:pass
                else:raise AssertionError('fault injector did not interrupt')
                before={str(p):identity(p) for p in (candidate/'postprocess').glob('tile_*.json')}
            p2.process(out/'pre.tif',out/'nuc.tif',f['nuclei_mask']['path'],candidate,'P2',native_pred=native,**kw)
            if workers==4:
                for path,obj in before.items():
                    if identity(path)!=obj:raise AssertionError('committed P2 tile replaced on recovery')
            finals.append(tifffile.imread(candidate/'final.tif'))
        np.testing.assert_array_equal(finals[0],finals[1])
    for k in ('pre_mask','nuclei_mask'):check_identity(f[k])
    atomic_json(out/'fixture_receipt.json',dict(status='passed',scope='fixed 128x128 Simple crop with complete global nuclei; mini is not full Simple replay',coords=[y,x],
        workers1_4_final_exact=True,interrupted_tiles_reused=True,inputs=f,resource=memory.result(),outputs=inventory(out)))
    print('F01 real mini PASS',memory.result(),flush=True)

def r_batch(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);ins=read('/home/zwan0726/work/docs/BIDCell_Rep1_Triple_Simple_Inputs_20261017.json');spec=ins['endpoints']['MORPH_SIMPLE_FUSION'];protocol=read(ins['shared']['evaluation']['protocol'])
    matrix=spec['matrix_root'];ids,genes,counts=matrix_arrays(dict(matrix=matrix));total=counts.sum(1);q=(total>=20)&(total<=2000)&((counts>0).sum(1)>=20)
    chosen=np.flatnonzero(q)[:protocol['prediction_batch_size']];frame=pd.DataFrame(counts[chosen],index=ids[chosen],columns=genes);frame.insert(0,'cell_id',ids[chosen]);csv=out/'max_batch313.csv';atomic_csv(csv,frame,index=True)
    snapshot=out/'source_snapshot';snapshot.mkdir();wrapper=snapshot/'paper_evaluation.R';shutil.copyfile(ROOT/'scripts/rep1_triple_simple_boundary/paper_evaluation.R',wrapper)
    expression=snapshot/'publish_expression_metrics.R';shutil.copyfile('/home/zwan0726/work/outputs/bidcell_metric_protocol_review_20261008/publish_expression_metrics.R',expression)
    sources=[identity(wrapper),identity(expression)];runtime=dict(source_snapshot=str(snapshot),allowed_sources=[str(expression)],official_expression=str(expression),dependencies_digest=digest(dict(fixture='max_batch',matrix=identity(csv))))
    rp=out/'runtime.json';atomic_json(rp,runtime)
    cfg=dict(source_snapshot=str(snapshot),rscript='/home/zwan0726/work/envs/bidcell_egr_r/bin/Rscript',r_libs_user='/home/zwan0726/work/outputs/bidcell_historical_dual_20261013/sources/R_library',r_libs_site='')
    def validate():
        for obj in sources:check_identity(obj)
    with (out/'run.lock').open('a') as a,Path('/home/zwan0726/work/outputs/cos_gpu.lock').open('a') as b:
        fcntl.flock(a,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(b,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with MemorySampler((a.fileno(),b.fileno())) as memory:
            dest=out/'evaluation';run_r(cfg,wrapper,[ins['shared']['evaluation']['protocol'],ins['shared']['evaluation']['prepared'],csv,dest,rp],dest,(a.fileno(),b.fileno()),validate,checkpoint_digest=runtime['dependencies_digest'])
        annotation=identity(dest/'annotation.csv');batch=identity(dest/'prediction_chunks/batch_0001.rds');commit=identity(dest/'prediction_chunks/batch_0001.rds.committed.json')
        # Actual batch recovery restores .Random.seed and skips prediction.
        run_r(cfg,wrapper,[ins['shared']['evaluation']['protocol'],ins['shared']['evaluation']['prepared'],csv,dest,rp],dest,(a.fileno(),b.fileno()),validate,checkpoint_digest=runtime['dependencies_digest'])
        if annotation!=identity(dest/'annotation.csv') or batch!=identity(batch['path']) or commit!=identity(commit['path']):raise AssertionError('actual R batch recovery drift')
    atomic_json(out/'fixture_receipt.json',dict(status='passed',scope='exact actual maximum256 prediction batch; not full-slide evaluation',batch_size=len(chosen),source=sources,
        selected_original_ids=ids[chosen].tolist(),original_input=identity(Path(matrix)/'counts.npy'),resource=memory.result(),R_committed_batch_reused=True,
        annotation_rows=len(pd.read_csv(dest/'annotation.csv')),complete313_gene_detection_rows=len(pd.read_csv(dest/'gene_detection_313.csv')),outputs=inventory(out)))
    print('F14 real R max batch PASS',memory.result(),flush=True)

def r_failure(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);snapshot=out/'snapshot';snapshot.mkdir();script=snapshot/'fault.R'
    script.write_text('args<-commandArgs(trailingOnly=TRUE);out<-args[1];dir.create(file.path(out,"prediction_chunks"),recursive=TRUE);p<-file.path(out,"prediction_chunks/batch.rds");writeLines("committed fixture bytes",p);jsonlite::write_json(list(file=p,cursor=1,total=2,dependencies_digest="fixture"),paste0(p,".pending.json"),auto_unbox=TRUE);Sys.sleep(30);writeLines("succeeded",file.path(out,"execution_status.txt"))\n')
    cfg=dict(source_snapshot=str(snapshot),rscript='/home/zwan0726/work/envs/bidcell_egr_r/bin/Rscript',r_libs_user='/home/zwan0726/work/outputs/bidcell_historical_dual_20261013/sources/R_library',r_libs_site='')
    def fail(event):raise RuntimeError('injected engineering callback failure')
    exits={}
    with (out/'run.lock').open('a') as lock,Path('/home/zwan0726/work/outputs/cos_gpu.lock').open('a') as shared:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(shared,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for mode in ('callback','validator'):
            dest=out/mode
            def validate():
                if mode=='validator' and (dest/'prediction_chunks/batch.rds.pending.json').exists():raise RuntimeError('injected engineering validator failure')
            try:run_r(cfg,script,[dest],dest,(lock.fileno(),shared.fileno()),validate,fail if mode=='callback' else None,checkpoint_digest='fixture')
            except RuntimeError as error:
                if 'injected engineering' not in str(error):raise
            else:raise AssertionError('R engineering failure hidden')
            exited=read(dest/'fault_exit.json')
            if exited['exit_code']>=0 or not exited['actual_exit_observed'] or not exited['engineering_failure']:raise AssertionError('R failure lifecycle not recorded')
            exits[mode]=exited
    atomic_json(out/'fixture_receipt.json',dict(status='passed',owned_R_callback_and_validator_failures_reaped=True,actual_exits=exits,outputs=inventory(out)))

def old_bt(out):
    from scripts.rep1_triple_simple_boundary.bt import endpoint_frame,validate_official
    out=Path(out);out.mkdir(parents=True,exist_ok=False);ins=read('/home/zwan0726/work/docs/BIDCell_Rep1_Triple_Simple_Inputs_20261017.json')
    trace=read('/home/zwan0726/work/docs/BIDCell_Rep1_Triple_Simple_RootOld7BTTrace_20261017.json');records=[]
    with Path('/home/zwan0726/work/outputs/cos_gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name,legacy in trace['endpoints'].items():
            spec=ins['endpoints'][name];f=endpoint_frame({},ins['shared'],dict(matrix=spec['matrix_root'],evaluation=spec['evaluation_root']))
            table=validate_official(f,f.index[f.own_qc].to_numpy(np.int64),legacy['profile_root'],recompute_nearest=True)
            records.append(dict(endpoint=name,rows=len(table),all_seven_bins_all_six_markers_exact=True,full_nearest_minID_compatibility=True,producer=identity(Path(legacy['profile_root'])/'complete.json')))
            print(name,'old six-marker all-bin compatibility PASS',flush=True)
    atomic_json(out/'fixture_receipt.json',dict(status='passed',scope='read-only real seven legacy official profiles, no new classification or formal results',endpoints=records))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case',choices=['mini','r_batch','r_failure','old_bt']);p.add_argument('--output',required=True);a=p.parse_args();globals()[a.case](a.output)
