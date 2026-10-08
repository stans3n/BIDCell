"""Targeted EGR ablation, historical initialization, and resumability tests."""
import argparse,copy,json,subprocess,hashlib,random
from pathlib import Path
import numpy as np
import torch
from scripts.crgf.common import atomic_json,identity
from scripts.crgf.config import read
from scripts.crgf.training import make_model,seed_all,optimizer,losses,to_device,save,load
from bidcell.model.model.relation_geometry import RelationGeometry
OLD='f840a8cf92e7ef987f8ee92f14a2e0ee628d9952'
COUNTS={'E':9472,'G':9632,'R':10528,'GR':10688}

def frozen_class():
    code=subprocess.check_output(['git','show',OLD+':bidcell/model/model/relation_geometry.py'],text=True)
    ns={};exec(compile(code,'frozen_relation_geometry.py','exec'),ns);return ns['RelationGeometry']

def cpu(out):
    torch.set_num_threads(4);legacy=frozen_class();checks={}
    for v in COUNTS:
        torch.manual_seed(20261008);old=legacy();torch.manual_seed(20261008);m=RelationGeometry(v)
        assert sum(p.numel() for p in m.parameters())==COUNTS[v]
        for k,x in m.state_dict().items():
            expected=old.state_dict()[k]
            if k=='wm.weight' and v in ('E','G'):expected=expected[:,:32]
            torch.testing.assert_close(x,expected,rtol=0,atol=0)
        f=torch.randn(3,128,4,4,requires_grad=True);g=torch.randn(3,4,8,8);a=torch.ones(3,1,8,8)
        y=m(f,g,a);newg=g+torch.randn_like(g)
        gd=float((m(f,newg,a)-y).abs().max());assert (gd==0)==(v in ('E','R'))
        newf=f.detach().clone();newf[1]+=torch.randn_like(newf[1]);fd=float((m(newf,g,a)[0]-y[0]).abs().max())
        cross=torch.autograd.grad(y[0].sum(),f,retain_graph=True)[0][1].abs().sum().item()
        assert (fd==0)==(v in ('E','G')) and (cross==0)==(v in ('E','G'))
        noother=a.clone();noother[1:]=0
        assert torch.equal(m(f,g,noother)[0],m(newf,g,noother)[0])
        assert torch.equal(m(f,g,a*0),f)
        perm=torch.tensor([2,0,1]);torch.testing.assert_close(m(f[perm],g[perm],a[perm])[torch.argsort(perm)],y)
        m(f[:1],g[:1],a[:1])
        if v in ('R','GR'):
            _,r,c,_=m.relations(f[:1],g[:1],a[:1]);assert r.count_nonzero()==0 and c.count_nonzero()==0
        try:m(f[:0],g[:0],a[:0]);raise AssertionError('K0 accepted')
        except ValueError:pass
        diff=None
        if v=='GR':
            yold=old(f,g,a);torch.testing.assert_close(y,yold,atol=0,rtol=0)
            grad=torch.autograd.grad(y.sum(),f)[0];gradold=torch.autograd.grad(yold.sum(),f)[0]
            torch.testing.assert_close(grad,gradold,atol=0,rtol=0);diff=float((y-yold).abs().max())
        checks[v]=dict(parameters=COUNTS[v],template_exact=True,geometry_response=gd,other_candidate_response=fd,cross_gradient=cross,GR_forward_max_difference=diff)
    try:RelationGeometry('gr');raise AssertionError('typo accepted')
    except ValueError:pass
    atomic_json(out/'module_contract.json',dict(status='succeeded',checks=checks))

def gpu(cfg,out):
    from bidcell.model.dataio.dataset_input import DataProcessing
    from bidcell.model.model.model import SegmentationModel
    torch.set_num_threads(4);ds=DataProcessing(cfg.native(),isTraining=True,total_steps=None,structured=True)
    views=json.loads((Path(cfg.output_root)/'configs/views.json').read_text());chosen=[v for v in views['views'] if v['k']>1][:2]
    samples=[]
    for v in chosen:
        ds.view_transform=(v['flip'],v['rotation']);samples.append(to_device(ds[v['index']]))
    result={};basecfg=cfg.model_copy(update={'variant':None});oldmod=frozen_class()
    for method in ('GR','B1','E','G','R'):
        c=basecfg.model_copy(update={'variant':method});model=make_model(c,method,len(ds.gene_names));model.train();opt=optimizer(model,c)
        before={k:p.detach().clone() for k,p in model.named_parameters() if k.startswith(('relation_geometry.','conv1.','conv2.','conv5.','outconv_seg.'))}
        s=samples[0];opt.zero_grad();y=model(s['x'],s['geometry'],s['s']);total,parts=losses(y,s);total.backward()
        grad={k:float(p.grad.norm()) for k,p in model.named_parameters()}
        assert all(np.isfinite(x) for x in grad.values())
        assert all(grad[k]>0 for k in before)
        olddiff=None
        if method in ('GR','B1'):
            # Whole model including BN, original head and six loss gradients.
            model.zero_grad();model.eval();reference=copy.deepcopy(model)
            if method=='GR':
                reference.relation_geometry=oldmod().cuda();reference.relation_geometry.load_state_dict(model.relation_geometry.state_dict())
            else:
                code=subprocess.check_output(['git','show','2fae67ad74eaa9ec77b5d6fed6178f7cd9feca9d:bidcell/model/model/model.py'],text=True)
                ns={'__package__':'bidcell.model.model'};exec(compile(code,'baseline_model.py','exec'),ns)
                reference=ns['SegmentationModel'](n_channels=len(ds.gene_names)).cuda();reference.load_state_dict(model.state_dict())
            reference.eval();y=model(s['x'],s['geometry'],s['s']);yo=reference(s['x'],s['geometry'],s['s']) if method=='GR' else reference(s['x'])
            torch.testing.assert_close(y,yo,atol=0,rtol=0);oldtotal,oldparts=losses(yo,s);total,parts=losses(y,s)
            total.backward();oldtotal.backward()
            for k in parts:torch.testing.assert_close(parts[k],oldparts[k],rtol=0,atol=0)
            for (k,p),(ko,po) in zip(model.named_parameters(),reference.named_parameters()):
                assert k==ko;torch.testing.assert_close(p.grad,po.grad,rtol=0,atol=0)
            olddiff=float((y-yo).abs().max());del reference,yo,oldtotal,oldparts
            model.train()
        opt.step();delta={k:float((p-before[k]).abs().max()) for k,p in model.named_parameters() if k in before};assert all(x>0 for x in delta.values())
        # Save all state and require the next real update to match after restore.
        dep={'inputs':{'gene_order':ds.gene_names},'test':'EGR exact resume'};path=out/f'{method}_resume.pth';save(path,model,opt,dep,1,1,{chosen[0]['index']})
        def step():
            s=samples[1];opt.zero_grad();y=model(s['x'],s['geometry'],s['s']);l,_=losses(y,s);l.backward();opt.step();return y.detach().cpu(),float(l)
        y1,l1=step();state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};rng1=(random.random(),np.random.rand(),torch.rand(1).item(),torch.rand(1,device='cuda').item())
        assert load(path,model,opt,dep)==(1,1,{chosen[0]['index']})
        y2,l2=step();rng2=(random.random(),np.random.rand(),torch.rand(1).item(),torch.rand(1,device='cuda').item());assert rng1==rng2
        assert abs(l1-l2)<=1e-5*max(abs(l1),1)
        torch.testing.assert_close(y1,y2,atol=1e-6,rtol=1e-5)
        for k,v in model.state_dict().items():torch.testing.assert_close(state[k],v.cpu(),atol=1e-6,rtol=1e-5)
        result[method]=dict(parameters=sum(p.numel() for p in model.parameters()),historical_forward_loss_gradient_max_difference=olddiff,gradients=grad,updates=delta,resume_logits_max_difference=float((y1-y2).abs().max()),resume_losses=[l1,l2],resume_loss_difference=l1-l2,RNG_exact=True)
        if method in ('E','G','R'):mini_chain(c,model,samples[0],out/method)
        del model,opt,before,state,y1,y2,y,total,parts
        torch.cuda.empty_cache()
    atomic_json(out/'gpu_contract.json',dict(status='succeeded',checks=result,tolerance={'atol':1e-6,'rtol':1e-5}))

def mini_chain(cfg,model,sample,out):
    import tifffile,pandas as pd
    from scripts.crgf.prediction import decode,merge_native
    from bidcell.model.postprocess_predictions import postprocess_connect,remove_islands
    from scripts.crgf.common import atomic_tiff
    from scripts.crgf.matrix import build
    from scripts.crgf.evaluate import validate_matrix
    out.mkdir(parents=True,exist_ok=True);model.eval();shiftmaps=[]
    with torch.inference_mode():
        for shift in (0,24):
            # Two deterministic genuine model invocations; small synthetic spatial
            # fixture explicitly distinct from full-field production prediction.
            logits=model(sample['x'],sample['geometry'],sample['s']).cpu().numpy()
            patch=decode(logits,sample['nuclei'],sample['cell_ids']);mask=np.zeros((96,96),np.uint32);mask[shift:shift+48,shift:shift+48]=patch;shiftmaps.append(mask)
            atomic_tiff(out/f'shift{shift}.tif',mask)
    merged,fill=merge_native(*shiftmaps,['24_24.hdf5']);nuc=np.zeros((96,96),np.uint32);nuc[24:72,24:72]=sample['nuclei']
    random.seed(cfg.seed);final=remove_islands(postprocess_connect(merged,nuc),nuc);final=np.where(nuc>0,nuc,final).astype(np.uint32);atomic_tiff(out/'final.tif',final)
    genes=Path(cfg.native().files.data_dir,'all_gene_names.txt').read_text().splitlines();(out/'genes.txt').write_text('\n'.join(genes)+'\n');(out/'baseline.json').write_text('{}')
    yy,xx=np.nonzero(final);idx=np.linspace(0,len(xx)-1,min(4000,len(xx)),dtype=int)
    pd.DataFrame(dict(transcript_id=np.arange(len(idx)),x_location=xx[idx]+.1,y_location=yy[idx]+.1,feature_name=np.resize(genes,len(idx)),qv=30)).to_csv(out/'transcripts.csv',index=False)
    c=dict(final_mask=str(out/'final.tif'),genes=str(out/'genes.txt'),transcripts=str(out/'transcripts.csv'),baseline_config=str(out/'baseline.json'),scale_x=.2125,scale_y=.2125,chunksize=1000)
    receipt=build(c,out/'matrix',cache=out/'cache');validated=validate_matrix(c,out/'matrix')
    counts=np.load(out/'matrix/counts.npy');evaluation=dict(cells=len(counts),capture=receipt['totals']['assigned']/receipt['totals']['eligible'],counts=int(counts.sum()),genes_per_cell=(counts>0).sum(1).tolist())
    atomic_json(out/'chain_receipt.json',dict(status='succeeded',scope='small synthetic coordinate fixture with real trained model outputs; not full-field or scientific evaluation',variant=cfg.variant,stages=['train','shift0','shift24','merge','native_postprocess','own_matrix','independent_validation','fixture_evaluation'],matrix=identity(out/'matrix/complete.json'),evaluation=evaluation))

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--part',choices=['cpu','gpu'],required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    cpu(out) if a.part=='cpu' else gpu(read(a.config),out)
if __name__=='__main__':main()
