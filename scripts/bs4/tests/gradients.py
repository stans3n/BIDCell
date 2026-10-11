"""64 historical frozen patches: paired alpha effect on identical model logits."""
import argparse,json,time
from pathlib import Path
import torch
from scripts.bs4.config import read
from scripts.bs4.training import dataset,make_model,forward,losses,TRAIN_SOURCES,to_device
from scripts.bs4.loss_diagnostics import get,preserve,groups,norm,write_csv
from scripts.bs4.method import teacher,replace_pos,stats,frozen_teacher_hash
from scripts.crgf.common import atomic_json,identity,source_identity

def run(root):
    execution_source=source_identity(TRAIN_SOURCES+['scripts/bs4/tests/gradients.py'])
    root=Path(root);cfg=read(root/'T1/configs/run.json');torch.set_num_threads(2)
    ds=dataset(cfg);model=make_model(cfg,'T1',len(ds.gene_names));model.train();params=list(model.parameters());group=groups(model)
    views=json.loads((root/'diagnostics/gradient_samples.json').read_text())['samples']
    if len(views)!=64:raise ValueError('fixed gradient sample count')
    rows=[];exposures=[];start=time.monotonic();nonzero={k:False for k in ('T1-B','T2-B','T2-T1')}
    for i,v in enumerate(views):
        with preserve(model,ds):
            sample=get(ds,v);z=forward(model,sample);_,old=losses(z,sample,True,cfg.anchor_weight,s_enabled=False)
            targets={};pos={}
            for arm,alpha in [('T1',0.),('T2',.25)]:
                s={k:(val.detach().cpu() if torch.is_tensor(val) else val) for k,val in sample.items() if k!='_teacher'};s.update(alpha=alpha,beta=1.)
                s=to_device(s);targets[arm]=teacher(s)
                if frozen_teacher_hash(targets[arm])!=frozen_teacher_hash(sample['_teacher']):raise ValueError('diagnostic differs from formal CPU frozen teacher')
                pos[arm]=replace_pos(old['pos'],z,s,targets[arm])
                exposures.append(dict(sample=i,arm=arm,**stats(targets[arm],s)))
            assert frozen_teacher_hash(targets['T1'])==frozen_teacher_hash(targets['T2'])
            values={'B_POS':old['pos'],'T1_POS':pos['T1'],'T2_POS':pos['T2'],'T1-B':pos['T1']-old['pos'],'T2-B':pos['T2']-old['pos'],'T2-T1':pos['T2']-pos['T1'],'anchor':old['anchor'],'B_six':sum(old[k] for k in ('ne','os','cc','ov','pos','neg'))}
            for label,value in values.items():
                gs=torch.autograd.grad(value,params,retain_graph=True,allow_unused=True)
                lg=torch.autograd.grad(value,z,retain_graph=True)[0]
                for name,indices in group.items():
                    rows.append(dict(sample=i,view_cursor=v['view_cursor'],contrast=label,group=name,gradient_norm=norm(gs,indices),scalar=float(value),logit_gradient_norm=float(lg.double().norm())))
                if label in nonzero:nonzero[label]|=bool(torch.count_nonzero(lg))
            atomic_json(root/'diagnostics/gradient_progress.json',dict(completed=i+1,total=64,seconds=time.monotonic()-start))
    if not all(nonzero.values()) or not any(e['arm']=='T2' and e['p1_softened_pixels']>0 for e in exposures):raise ValueError('method supervision/alpha effect absent')
    if execution_source!=source_identity(TRAIN_SOURCES+['scripts/bs4/tests/gradients.py']):raise ValueError('gradient execution source drift')
    write_csv(root/'diagnostics/gradient_norms_shared_v1.csv',rows)
    atomic_json(root/'diagnostics/gradient_diagnostics_shared_v1.json',dict(status='passed',samples=64,fixed_sample_manifest=identity(root/'diagnostics/gradient_samples.json'),same_logits=True,all_weights_one=True,nonzero=nonzero,exposures=exposures,source=execution_source,outputs={'norms':identity(root/'diagnostics/gradient_norms_shared_v1.csv')},wall_seconds=time.monotonic()-start))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();run(a.root)
