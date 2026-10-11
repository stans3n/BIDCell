"""Real P CLI interruption after a published tile, then strict resume and control replay."""
import argparse,json,os,signal,subprocess,sys,time,shutil
from pathlib import Path
import numpy as np,tifffile,yaml
from scripts.crgf.common import atomic_json,identity,source_identity
SOURCES=['scripts/bs4/postprocess.py','scripts/bs4/post_rules.py','scripts/bs4/stage.py','scripts/bs4/reuse.py','scripts/bs4/tests/post_resume.py']

def run(root):
    root=Path(root);source=source_identity(SOURCES);out=root/'tests/post_resume_v2';out.mkdir(exist_ok=True);results={}
    # Rectangular real ROI yields three native min-dimension tiles while retaining
    # the immutable literal postprocess_size=1024 production configuration.
    original=json.loads((root/'tests/mini_v5/P1/configs/run.json').read_text());y,x=original['mini_origin'];height,width=96,288
    data=out/'data';data.mkdir(exist_ok=True);native=yaml.safe_load(Path(original['baseline_config']).read_text());native['files']['data_dir']=str(data)
    baseline=data/'mini.yaml';baseline.write_text(yaml.safe_dump(native));nuclei=tifffile.memmap(original['mini_full_nuclei']);tifffile.imwrite(data/'nuclei.tif',nuclei[y:y+height,x:x+width])
    historical=Path('/home/zwan0726/work/outputs/cos_loss_20261009T160351+1100/B/predictions/B');native_endpoints=json.loads((historical/'endpoints.json').read_text())['endpoints'];reuse=data/'B_prediction';(reuse/'postprocess').mkdir(parents=True,exist_ok=True);(reuse/'probes').mkdir(exist_ok=True);entries={}
    for name in ('raw_shift0','shift0','raw_shift24','shift24','pre'):
        obj=native_endpoints[name]
        if identity(obj['path'])!=obj:raise ValueError('historical ROI source drift')
        target=reuse/(name+'.tif');tifffile.imwrite(target,tifffile.memmap(obj['path'])[y:y+height,x:x+width]);entries[name]=identity(target)
    atomic_json(reuse/'endpoints.json',dict(endpoints=entries,global_origin=[y,x],shape=[height,width],historical={name:native_endpoints[name] for name in entries}))
    atomic_json(reuse/'postprocess/complete.json',dict(status='succeeded',scope='real historical ROI input provenance only',outputs=entries))
    for arm in ('P1','P2'):
        endpoints={}
        for attempt in ('interrupted','control'):
            ar=out/arm/attempt;(ar/'configs').mkdir(parents=True,exist_ok=True);(ar/'source').mkdir(exist_ok=True)
            old=root/'tests/mini_v5'/arm/'configs';cfg=json.loads((old/'run.json').read_text());cfg.update(output_root=str(ar),baseline_config=str(baseline),reused_prediction_root=str(reuse),queue_config=str(ar/'configs/queue.json'))
            atomic_json(ar/'configs/queue.json',dict(cpu_workers=1));atomic_json(ar/'configs/run.json',cfg)
            shutil.copyfile(old/'probes.json',ar/'configs/probes.json')
            inputs=json.loads((old/'inputs.json').read_text());inputs.update(baseline_config=identity(baseline),nuclei_mask=identity(data/'nuclei.tif'));atomic_json(ar/'configs/inputs.json',inputs)
            atomic_json(ar/'configs/frozen.json',dict(identities={p.name:identity(p) for p in (ar/'configs').glob('*.json') if p.name!='frozen.json'}))
            cmd=[sys.executable,'-m','scripts.bs4.stage','--config',str(ar/'configs/run.json'),'--stage','postprocess'];folder=ar/'predictions'/arm/'postprocess'
            with (ar/'attempt1.log').open('w') as log:
                child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                if attempt=='interrupted':
                    deadline=time.monotonic()+180;published=[]
                    while child.poll() is None and time.monotonic()<deadline:
                        published=list(folder.glob('tile_*_*.json'))
                        if published and len(published)<3:break
                        time.sleep(.002)
                    if not published or len(published)>=3:
                        if child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=30)
                        raise ValueError('did not observe actual partial tile boundary; inspect retained attempt log, exit='+str(child.returncode))
                    os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=30)
                    if (folder/'complete.json').exists():raise ValueError('interruption occurred after completion')
                    before={str(p):dict(identity(p),mtime_ns=p.stat().st_mtime_ns) for rec in published for p in [rec,*[Path(x['path']) for x in json.loads(rec.read_text())['outputs'].values()]]}
                    shutil.copytree(folder,ar/'interrupted_snapshot')
                    with (ar/'resume.log').open('w') as resumed:subprocess.run(cmd,stdout=resumed,stderr=subprocess.STDOUT,check=True,timeout=300)
                    if before!={p:dict(identity(p),mtime_ns=Path(p).stat().st_mtime_ns) for p in before}:raise ValueError('published tile rewritten on resume')
                    results[arm]=dict(interrupted_exit=child.returncode,published_tiles=len(published),completed_tiles_preserved=before)
                else:
                    try:
                        if child.wait(timeout=300):raise ValueError('uninterrupted real control failed')
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=30);raise
            ep=json.loads((ar/'predictions'/arm/'endpoints.json').read_text())['endpoints'];endpoints[attempt]=ep
        for name,obj in endpoints['control'].items():
            if not np.array_equal(tifffile.memmap(obj['path']),tifffile.memmap(endpoints['interrupted'][name]['path'])):raise ValueError('interrupted/control endpoint diverged '+name)
        results[arm]['all_endpoints_bitwise']=True
    if source_identity(SOURCES)!=source:raise ValueError('post resume source drift')
    atomic_json(out/'complete.json',dict(status='passed',execution_source=source,arms=results,actual_process_interruption=True,outputs={str(p.relative_to(out)):identity(p) for p in out.rglob('*') if p.is_file() and p!=out/'complete.json'}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();run(a.root)
