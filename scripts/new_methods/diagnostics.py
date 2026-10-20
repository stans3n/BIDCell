"""Frozen probe diagnostics and input-only coverage with explicit denominators."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi
from .common import *
from .data import NativeViews
from .training import forward_losses,gradient_stats,capture,scalar_diag
from bidcell.model.model.region_losses import native_owner,probabilities


def distribution(values):
    x=np.asarray(values,float);x=x[np.isfinite(x)]
    return dict(n=int(len(x)),quantiles={str(q):float(np.quantile(x,q)) for q in (0,.1,.5,.9,.95,1)}) if len(x) else dict(n=0,quantiles={})


def soft_hard_diagnostics(z,batch):
    f,b=probabilities(z);owner=native_owner(z);k=len(f)
    hard=(owner[None]==torch.arange(1,k+1,device=z.device)[:,None,None]).to(z.dtype)
    non=batch['labels']==0;captured=(owner>0)&non
    raw=batch['aux']['raw_counts'].sum(0)
    scores=torch.cat((b[None],f));top=scores.sort(dim=0,descending=True,stable=True).values
    margins=(top[0]-top[1])[non]
    result=dict(winner_margin=distribution(margins.detach().cpu().numpy()),exact_ties=int((margins==0).sum()),temperatures={})
    counts=batch['aux']['raw_counts'].flatten(1).T
    for t in (.05,.1,.2):
        pi=(f/t).softmax(0);tv=.5*(pi-hard).abs().sum(0)
        mass=raw[captured].sum()
        hard_counts=(hard*non[None]).flatten(1)@counts
        soft_counts=(pi*captured[None]).flatten(1)@counts
        a=hard_counts/hard_counts.sum(1).clamp_min(1e-12)[:,None];q=soft_counts/soft_counts.sum(1).clamp_min(1e-12)[:,None];mid=(a+q)/2
        js_each=.5*((a*(a.clamp_min(1e-12).log()-mid.clamp_min(1e-12).log())).sum(1)+(q*(q.clamp_min(1e-12).log()-mid.clamp_min(1e-12).log())).sum(1))
        valid=(hard_counts.sum(1)>0)&(soft_counts.sum(1)>0)
        result['temperatures'][str(t)]=dict(molecule_weighted_TV=float((tv*raw*captured).sum()/mass) if mass>0 else None,
            aggregate_L1=float((soft_counts-hard_counts).abs().sum()),jacobian_scale_mean=float((pi*(1-pi)/t).mean()),
            per_cell_aggregate_L1=(soft_counts-hard_counts).abs().sum(1).detach().cpu().tolist(),
            per_cell_aggregate_JS=[float(v) if bool(ok) else None for v,ok in zip(js_each.detach().cpu(),valid.cpu())],
            saturation_fraction=float(((pi<1e-6)|(pi>1-1e-6)).float().mean()))
    return result


def structure_diagnostics(z,batch,loss_diag=None):
    owner=native_owner(z).cpu().numpy();labels=batch['labels'].cpu().numpy();ids=batch['candidate_ids'];rows=[]
    for i,cid in enumerate(ids):
        region=(owner==i+1)&(labels==0);own=labels==cid;mask=region|own
        lab,n=ndi.label(mask,structure=ndi.generate_binary_structure(2,2))
        root_components=np.unique(lab[own]);root_components=root_components[root_components>0]
        border_components=np.unique(np.r_[lab[0],lab[-1],lab[:,0],lab[:,-1]]);border_components=border_components[border_components>0]
        anchored=np.isin(lab,root_components);boundary=np.isin(lab,border_components)
        _,n4=ndi.label(mask,structure=ndi.generate_binary_structure(2,1))
        row=dict(cell_id=int(cid),components8=n,components4=n4,hard_anchored=int((region&anchored).sum()),
                 hard_internal_disconnected=int((region&~anchored&~boundary).sum()),hard_boundary_only=int((region&~anchored&boundary).sum()))
        if loss_diag and 'root_penalty_mask' in loss_diag:
            indices=loss_diag['root_candidate_indices'].cpu().tolist()
            if i in indices:
                j=indices.index(i);active=loss_diag['root_penalty_mask'][j].cpu().numpy()
                r=loss_diag['root_r'][j].cpu().numpy();bd=loss_diag['root_boundary'][j].cpu().numpy()
                row.update(penalized_hard_connected_weak_margin=int((active&anchored).sum()),
                           penalized_hard_internal_disconnected=int((active&~anchored&~boundary).sum()),
                           confident_boundary_only=int((region&(bd>=.55)&(r<.55)).sum()),confident_nucleus_anchored=int((region&(r>=.55)).sum()))
        rows.append(row)
    return rows


def fixed_probe_diagnostics(model,data,base,extra,config,updates,calibration,pilot=False):
    root=Path(config['runroot']);probes=read(root/'configs/probes.json');arm=config['arm']
    views=probes['boundary_fixed'] if arm=='V1_BOUNDARY' else probes['region'][:8] if arm=='V1_REGION' else probes['morphology']
    saved_rng=rng_state();old=model.training;model.eval();records=[]
    for view in views:
        batch=data.get_view(view,'cuda');z,l6,new,parts,diag=forward_losses(model,batch,base,extra,arm)
        gram,layers=gradient_stats(model,[l6,*new])
        direct=torch.autograd.grad(new[0],z,retain_graph=True,allow_unused=True)[0]
        leakage=0.
        if direct is not None and arm=='V1_BOUNDARY':leakage=float(direct.permute(1,0,2,3)[:,~diag['direct_domain']].abs().max()) if (~diag['direct_domain']).any() else 0.
        elif direct is not None and arm=='V1_REGION':
            blocked=(native_owner(z)==0)|(batch['labels']>0)
            leakage=float(direct[:,:,blocked].abs().max()) if blocked.any() else 0.
        record=dict(view=view,losses=[float(l6.detach()),*[float(v.detach()) for v in new]],gram=gram.tolist(),layer_gram=layers,
                    diagnostics=scalar_diag(diag),direct_gradient_leakage=leakage,capture=capture(z,batch),soft_hard=soft_hard_diagnostics(z,batch),
                    structure=structure_diagnostics(z,batch,diag))
        if arm=='V2_MORPH':
            with torch.no_grad():
                aux=dict(batch['aux']);aux['geometry']=aux['geometry'].flip(0)
                permuted=model(batch['input'],aux);aux=dict(batch['aux']);aux['dapi']=torch.zeros_like(aux['dapi']);zeroed=model(batch['input'],aux)
                record['morphology_inference_diagnostics']=dict(geometry_permutation_logits_L2=float((permuted-z).norm()),
                      zero_DAPI_logits_L2=float((zeroed-z).norm()),permutation_owner_changes=int((native_owner(permuted)!=native_owner(z)).sum()),
                      zero_DAPI_owner_changes=int((native_owner(zeroed)!=native_owner(z)).sum()))
        records.append(record)
    restore_rng(saved_rng);model.train(old)
    out=(root/'admission'/arm/'diagnostics') if pilot else Path(config['output_root'])/'diagnostics';out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/f'probe_{updates:04d}.json',json_safe(dict(updates=updates,pilot=pilot,rows=records,calibration=calibration,source=source_digest())))
    return records


def coverage(root):
    root=Path(root);data=NativeViews(root);views=read(root/'configs/views.json')['views'];rows=[];ref=read(root/'configs/reference_region.json')
    from bidcell.model.model.region_losses import BoundaryRegionLoss
    # Eligibility is independent of logits. Use a neutral forward only to expose
    # Omega0; record no activation claim from its artificial owner.
    boundary=BoundaryRegionLoss(read(root/'configs/reference_boundary.json'))
    for j,v in enumerate(views):
        if not v['K']:continue
        b=data.get_view(v);aux=b['aux'];non=aux['nuclear_labels']==0;counts=aux['raw_counts'][boundary.gene_indices].sum(0)
        _,bd=boundary(torch.zeros((v['K'],2,48,48)),aux)
        eligible_b=(aux['native_support']&bd['omega0_mask'][None]).flatten(1).any(1)
        for i,cid in enumerate(v['candidate_ids']):
            row=data.evidence.id_to_row[cid];typ=int(aux['type_index'][i]);name=ref['type_names'][typ] if typ>=0 else 'Unknown'
            static=float(aux['full_nuclear_counts'][i,boundary.gene_indices].sum()+(counts*aux['global_support_visible'][i]*non).sum())
            c=float(aux['support_area_visible'][i]/aux['support_area_global'][i]);native=aux['native_support'][i];glob=aux['global_support_visible'][i]
            union=(native|glob).sum();area=int(aux['full_nucleus_pixels'][i]);cov=data.evidence.meta['covariance'][row]
            eigen=np.linalg.eigvalsh(cov);major=float(4*np.sqrt(max(eigen[-1],0)))
            positions=torch.nonzero(aux['nuclear_labels']==cid);edge=int(torch.minimum(positions,47-positions).min())
            rows.append(dict(view_cursor=v['cursor'],cell_id=cid,type=name,elongated=name in data.evidence.elongated,K=v['K'],
                 K_bin='1' if v['K']==1 else '2' if v['K']==2 else '3-5' if v['K']<=5 else '>=6',
                 nucleus_area=area,major_axis=major,edge_distance=edge,support_visibility=c,
                 visibility_bin='<.25' if c<.25 else '.25-.5' if c<.5 else '.5-.75' if c<.75 else '.75-1' if c<1 else '1',
                 static_count=static,count_bin='0' if static==0 else '1-7' if static<8 else '8-31' if static<32 else '>=32',
                 region_eligible=bool(aux['ref_valid'][i] and static>0),boundary_eligible=bool(eligible_b[i]),
                 weight=c*min(1,static/32),native_global_iou=float((native&glob).sum()/union) if union else 1.,
                 nucleus_complete=int(aux['visible_nucleus_pixels'][i])==area,support_complete=c==1,
                 specimen_clipped=bool(aux['specimen_clipped'][i])))
        if (j+1)%256==0:print('coverage',j+1,'/',len(views),flush=True)
    frame=pd.DataFrame(rows);frame['area_quintile']=pd.qcut(frame.nucleus_area,5,labels=False,duplicates='drop')
    frame['major_quintile']=pd.qcut(frame.major_axis,5,labels=False,duplicates='drop')
    out=root/'admission';frame.to_csv(out/'coverage_candidates.csv',index=False)
    def summary(df):return dict(appearances=len(df),unique_ids=int(df.cell_id.nunique()),region_eligible=int(df.region_eligible.sum()),
        boundary_eligible=int(df.boundary_eligible.sum()),weight_sum=float(df.weight.sum()),visibility=distribution(df.support_visibility),
        native_global_iou=distribution(df.native_global_iou))
    result=dict(all=summary(frame),elongated=summary(frame[frame.elongated]),strata={key:{str(k):summary(g) for k,g in frame.groupby(key)}
        for key in ['type','elongated','K_bin','visibility_bin','count_bin','area_quintile','major_quintile','edge_distance']})
    result['status']='succeeded'
    for group in ('all','elongated'):
        if result[group]['appearances'] and result[group]['region_eligible']/result[group]['appearances']<.1:raise RuntimeError('region eligibility below10% '+group)
    atomic_json(out/'coverage_summary.json',result);return result
