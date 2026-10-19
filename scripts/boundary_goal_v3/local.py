"""Fixed raw denominators, lineage mass balance, exits/fates and spatial sums."""
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
from scipy.stats import norm
from scripts.rep1_triple_simple_boundary import bt
from scripts.rep1_triple_simple_boundary.common import atomic_csv,atomic_json,atomic_npy,atomic_npz,read,identity
from scripts.rep1_triple_simple_boundary.inputs import matrix_arrays
from scripts.rep1_triple_simple_boundary.frozen_domains import in_region
from .config import MARKERS,STATES,FATES
from scripts.rep1_triple_simple_boundary.comparisons import atomic_csv_gzip

REGIONS=('GLOBAL','INTERFACE_0_2','INTERFACE_0_1','INTERFACE_1_2')

def state_codes(owner,lineage,gene_lineage):
    o=np.asarray(owner);typ=lineage[o]
    s=np.full(len(o),2,np.int8)
    s[(typ>0)&(typ==gene_lineage)]=0
    s[(typ>0)&(gene_lineage>0)&(typ!=gene_lineage)]=1
    s[o==0]=3
    return s

def fate_codes(source,target,lineage,gene_lineage):
    a,b=lineage[source],lineage[target];marker=gene_lineage>0
    f=np.full(len(source),5,np.int8)
    f[marker]=3
    f[marker&(a>0)&(a==b)]=2
    f[marker&(a>0)&(a!=gene_lineage)&(b==gene_lineage)]=0
    f[marker&(a==gene_lineage)&(b>0)&(b!=gene_lineage)]=1
    f[marker&(target==0)]=4
    return f

def wilson(hit,n):
    if n==0:return np.nan,np.nan
    z=norm.ppf(.975);p=hit/n;c=(p+z*z/(2*n))/(1+z*z/n);h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return c-h,c+h

def context(cfg,out):
    inputs=cfg['inputs'];frozen=Path(cfg['old_config']['domain_manifest']).parent
    cells=pd.read_csv(frozen/'cells.csv').set_index('cell_id')
    fixed=pd.read_csv(inputs['fixed_population']).set_index('cell_id')
    labels=fixed.evaluation_label.where(fixed.unmatched_reason.fillna('').eq(''))
    lineage=np.zeros(int(cells.index.max())+1,np.int8)
    lin=labels.map({'B Cells':1,'CD4 T':2,'CD8 T':2}).fillna(0).astype(np.int8)
    lineage[lin.index]=lin
    genes=Path(read(inputs['endpoints']['BASE_MATCH']['machine'])['genes']).read_text().splitlines()
    gl=np.array([MARKERS.get(g,0) for g in genes],np.int8)
    assignment=np.load(frozen/'interfaces/assignment.npy',mmap_mode='r')
    cache=Path(out)/'unique_interface_distance_squared.npy'
    distance=np.full(assignment.size,255,np.uint8)
    # Each RNA receives exactly one interface through historical unique assignment.
    for p in ([] if cache.exists() else sorted((frozen/'interfaces').glob('exposure_*.npz'))):
        with np.load(p) as z:
            pixel=z['pixel'];keep=assignment.ravel()[pixel]==z['interface']
            distance[pixel[keep]]=z['distance_squared'][keep]
    if cache.exists():distance=np.load(cache,mmap_mode='r').ravel()
    if np.any(distance[assignment.ravel()>=0]>4):raise ValueError('missing unique interface distance')
    if not cache.exists():atomic_npy(cache,distance.reshape(assignment.shape))
    return dict(frozen=frozen,cells=cells,fixed=fixed,lineage=lineage,genes=genes,gene_lineage=gl,assignment=assignment,distance=distance,shape=assignment.shape)

def distance_curves(cfg,specs,out,ctx):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=[];cellblocks=[]
    cells=ctx['cells'];fixed=ctx['fixed'];reference=cells[['x_um','y_um']]
    for name,spec in specs.items():
        ids,genes,counts=matrix_arrays(spec)
        frame=pd.DataFrame({g:np.asarray(counts[:,genes.index(g)]) for g in MARKERS},index=pd.Index(ids,name='cell_id'))
        available=fixed.unmatched_reason.fillna('').eq('');pool=np.intersect1d(ids,fixed.index[available])
        annpath=Path(spec['sc_scores']);ann=pd.read_csv(annpath).set_index('cell_id')
        clcol='cell_type' if 'cell_type' in ann else 'evaluation_label'
        ownlabels=ann[clcol]
        # Method own-QC descriptive curves use method centroids, fixed curves use nuclei.
        centroidpath=Path(spec['evaluation'])/'paper_aligned_v1/own_cell_centroids.csv'
        if centroidpath.exists():ownref=pd.read_csv(centroidpath).set_index('cell_id')[['x_um','y_um']]
        else:
            mask=tifffile.imread(spec['final']);yy,xx=np.indices(mask.shape);nn=np.bincount(mask.ravel(),minlength=len(ctx['lineage']))
            sx=np.bincount(mask.ravel(),weights=xx.ravel(),minlength=len(nn));sy=np.bincount(mask.ravel(),weights=yy.ravel(),minlength=len(nn));ownref=pd.DataFrame(dict(x_um=sx[ids]/nn[ids],y_um=sy[ids]/nn[ids]),index=ids);del mask,yy,xx
        for profile,pop,labels,ref in [('fixed_nuclear_reference',pool,fixed.evaluation_label,reference),('own_scQC_own_centroid',np.intersect1d(ids,ann.index),ownlabels,ownref)]:
            for panel in bt.PANELS:
                lab=labels.reindex(pop);rids=lab.index[lab.isin(panel['recipient_types'])].to_numpy(np.int64);tids=lab.index[lab.isin(panel['target_types'])].to_numpy(np.int64)
                curve,dist=bt.curve(ref,rids,tids,frame,panel)
                curve['method']=name;curve['population']=profile;curve['geometry']='nuclear centroid Euclidean' if profile.startswith('fixed') else 'method cell centroid Euclidean'
                ci=[wilson(int(h),int(n)) for h,n in zip(curve.n_positive,curve.n_eligible)];curve['ci_low']=[a for a,b in ci];curve['ci_high']=[b for a,b in ci];curve['ci_method']='Wilson 95% descriptive cell prevalence'
                rows.append(curve)
                if profile.startswith('fixed'):
                    for g in panel['markers']:
                        a=dist.reset_index();a['marker']=g;a['method']=name;a['positive']=(frame.loc[rids,g].to_numpy()>0).astype(np.int8);a['block']=cells.loc[rids,'block_1000'].to_numpy();a['direction']=panel['direction'];cellblocks.append(a)
    table=pd.concat(rows,ignore_index=True);atomic_csv(out/'distance.csv',table)
    atomic_csv_gzip(out/'distance_cells.csv.gz',pd.concat(cellblocks,ignore_index=True))
    return table

def _acc(group,weights,n):
    return np.bincount(group,weights=np.asarray(weights,dtype=float),minlength=n).astype(np.int64)

def stream(cfg,specs,pairs,out,ctx,progress=lambda *a,**k:None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);inputs=cfg['inputs'];nblocks=int(ctx['cells'].block_1000.max())+1
    nregions=len(REGIONS);mnames=list(specs);pnames=[a+'__TO__'+b for a,b in pairs]
    # Per block sufficient statistics; no 34M row table is retained.
    mass=np.zeros((len(mnames),nregions,nblocks,4),np.int64);capture=np.zeros((len(mnames),nregions,nblocks,2),np.int64)
    exits=np.zeros((len(pairs),nregions,3,nblocks,4),np.int64) # onN,onExit,offN,offExit; all/eligible/ineligible
    fates=np.zeros((len(pairs),nregions,nblocks,6),np.int64);transfers=np.zeros((len(pairs),nregions,nblocks,6),np.int64) # all,marker,cc,c0,0c,unchanged
    per_gene=np.zeros((len(pairs),nregions,len(ctx['genes']),4),np.int64)
    delta={p:np.zeros((len(ctx['cells']),len(ctx['genes'])),np.int64) for p in pnames}
    masks={m:tifffile.imread(s['final']) for m,s in specs.items()}
    nuclei=tifffile.imread(inputs['endpoints']['BASE_MATCH']['nuclei']);correspondence=[]
    for m,mask in masks.items():
        equal=mask.shape==nuclei.shape and np.array_equal(mask[nuclei>0],nuclei[nuclei>0])
        if not equal:raise ValueError('cross-method stable nuclear mapping failed for '+m)
        correspondence.append(dict(method=m,matching='exact same-ID complete nuclear raster',reference_cells=len(ctx['cells']),matched_cells=len(ctx['cells']),coverage=1.0,nuclear_pixels=int((nuclei>0).sum()),unmatched_cells=0,status='HIGH_CONFIDENCE_ONE_TO_ONE'))
    del nuclei
    legacy_gt=Path(cfg['specs']['P2_MORPH_GT']['root'])/'gt';domain=np.load(legacy_gt/'domain/domain.npz');eligible=np.zeros(np.prod(ctx['shape']),bool);eligible[domain['pixels']]=True
    boundary_pairs=set();bt_pairs=set();total=0;last_tid=None
    for i,rec in enumerate(inputs['coordinates']):
        co=np.load(rec['path'],mmap_mode='r');valid=co['x']>=0;flat=np.zeros(len(co),np.int64);flat[valid]=co['y'][valid].astype(np.int64)*ctx['shape'][1]+co['x'][valid]
        st=np.load(ctx['frozen']/'molecule_strata'/f'chunk_{i:04d}.npz');block=st['block_1000'].astype(int)
        if np.any(block<0) or np.any(block>=nblocks):raise ValueError('RNA block outside frozen universe')
        isroi=in_region(st['region_bits'],'interface');d2=np.full(len(co),255,np.uint8);d2[valid]=ctx['distance'][flat[valid]]
        if np.any(isroi&(d2>4)):raise ValueError('ROI distance does not cover frozen RNA')
        regions=[np.ones(len(co),bool),isroi,isroi&(d2<=1),isroi&(d2>1)]
        if np.any(regions[2]&regions[3]) or not np.array_equal(regions[2]|regions[3],regions[1]):raise ValueError('distance bins overlap/gap')
        gene=co['gene'];gl=ctx['gene_lineage'][gene];marker=gl>0;elig=np.zeros(len(co),bool);elig[valid]=eligible[flat[valid]]
        owners={m:np.where(valid,masks[m].ravel()[flat],0).astype(np.uint32) for m in mnames}
        codes={m:state_codes(owners[m],ctx['lineage'],gl) for m in mnames}
        for mi,m in enumerate(mnames):
            for ri,use in enumerate(regions):
                capture[mi,ri,:,0]+=_acc(block,use,nblocks);capture[mi,ri,:,1]+=_acc(block,use&(owners[m]>0),nblocks)
                for sc in range(4):mass[mi,ri,:,sc]+=_acc(block,use&marker&(codes[m]==sc),nblocks)
        pairids=st['interface_id'][isroi];boundary_pairs.update(np.unique(pairids[pairids>=0]).tolist())
        itab=pd.read_csv(ctx['frozen']/'interfaces/pairs.csv') if i==0 else itab
        pairlookup=np.zeros(len(itab),bool);a=ctx['lineage'][itab.cell_i.to_numpy()];b=ctx['lineage'][itab.cell_j.to_numpy()];pairlookup=((a==1)&(b==2))|((a==2)&(b==1))
        chosen=pairids[pairids>=0];bt_pairs.update(np.unique(chosen[pairlookup[chosen]]).tolist())
        for pi,(source,target) in enumerate(pairs):
            so,ta=owners[source],owners[target];change=so!=ta;on=marker&(codes[source]==0);off=marker&(codes[source]==1)
            fc=fate_codes(so,ta,ctx['lineage'],gl)
            # Full per-gene/cell matrix delta, including unassigned transitions.
            de=delta[pnames[pi]]
            for o,sign in ((so,-1),(ta,1)):
                keep=change&(o>0);rr=np.searchsorted(ctx['cells'].index.to_numpy(),o[keep]);keys=rr*len(ctx['genes'])+gene[keep]
                kk,vv=np.unique(keys,return_counts=True);de.ravel()[kk]+=sign*vv
            for ri,use in enumerate(regions):
                for ei,es in enumerate((np.ones(len(co),bool),elig,~elig)):
                    for j,v in enumerate((on,on&change,off,off&change)):exits[pi,ri,ei,:,j]+=_acc(block,use&es&v,nblocks)
                for j in range(6):fates[pi,ri,:,j]+=_acc(block,use&change&(fc==j),nblocks)
                for j,v in enumerate((change,change&marker,change&(so>0)&(ta>0),change&(so>0)&(ta==0),change&(so==0)&(ta>0),~change)):
                    transfers[pi,ri,:,j]+=_acc(block,use&v,nblocks)
                for j,v in enumerate((on,on&change,off,off&change)):per_gene[pi,ri,:,j]+=np.bincount(gene[use&v],minlength=len(ctx['genes']))
        total+=len(co);progress('local_stream',i+1,len(inputs['coordinates']),molecules=total)
    if total!=cfg['eligible']:raise ValueError('fixed global denominator changed')
    expected=cfg['interface_expected']
    if not np.all(capture[:,1,:,0].sum(1)==expected):raise ValueError('frozen interface denominator changed')
    assertions=[]
    for pi,(source,target) in enumerate(pairs):
        si,sg,sc=matrix_arrays(specs[source]);ti,tg,tc=matrix_arrays(specs[target])
        if not np.array_equal(si,ti) or sg!=tg or not np.array_equal(tc-sc,delta[pnames[pi]]):raise ValueError('RNA ledger does not reconstruct per-gene/cell matrix delta '+pnames[pi])
        assertions.append(dict(comparison=pnames[pi],cell_gene_delta_exact=True,all_raw_rows=total))
    for mi,m in enumerate(mnames):
        ids,genes,counts=matrix_arrays(specs[m])
        if int(counts.sum())!=int(capture[mi,0,:,1].sum()):raise ValueError('capture does not equal matrix '+m)
    atomic_npz(out/'block_sums.npz',mass=mass,capture=capture,exits=exits,fates=fates,transfers=transfers,per_gene=per_gene)
    atomic_json(out/'schema.json',dict(methods=mnames,pairs=pairs,genes=ctx['genes'],regions=list(REGIONS),states=list(STATES),fates=list(FATES),blocks=nblocks,all_interfaces=len(boundary_pairs),bt_interfaces=len(bt_pairs),matrix_assertions=assertions,eligibility='historical P2_MORPH GT legal candidate pixel domain',raw_rows=total))
    atomic_csv(out/'correspondence.csv',pd.DataFrame(correspondence));return out

def summaries(out):
    out=Path(out);j=read(out/'schema.json');z=np.load(out/'block_sums.npz');mass=z['mass'].sum(2);cap=z['capture'].sum(2);ex=z['exits'].sum(3);f=z['fates'].sum(2);tr=z['transfers'].sum(2);rows=[];flow=[];risk=[]
    for mi,m in enumerate(j['methods']):
        for ri,r in enumerate(REGIONS):
            den=int(mass[mi,ri].sum())
            for si,s in enumerate(STATES):rows.append(dict(method=m,region=r,state=s,count=int(mass[mi,ri,si]),fixed_raw_marker_D=den,fixed_roi_off_target_fraction=mass[mi,ri,1]/den if den else np.nan,metric_provenance='literature-inspired local adaptation; not an original-paper standard metric',fraction=mass[mi,ri,si]/den if den else np.nan,captured_all_gene=int(cap[mi,ri,1]),fixed_raw_all_gene_D=int(cap[mi,ri,0]),capture=cap[mi,ri,1]/cap[mi,ri,0] if cap[mi,ri,0] else np.nan,BLCR=np.nan,BLCR_status='NOT_DEFINED / NOT_APPLICABLE',evidence='REFERENCE-CONSISTENCY PROXY'))
            risk.append(dict(method=m,region=r,raw_marker_D=den,raw_all_gene_D=int(cap[mi,ri,0]),blocks=j['blocks'],all_interfaces=j['all_interfaces'],bt_interfaces=j['bt_interfaces']))
    for pi,(a,b) in enumerate(j['pairs']):
        for ri,r in enumerate(REGIONS):
            mi=j['methods'].index(a);raw=int(mass[mi,ri].sum());alltrans=int(tr[pi,ri,0])
            for ei,subset in enumerate(('ALL_FIXED_ROI','ELIGIBLE','INELIGIBLE')):
                on,oe,off,fe=map(int,ex[pi,ri,ei]);ron=oe/on if on else np.nan;roff=fe/off if off else np.nan
                flow.append(dict(source=a,target=b,region=r,kind='HOST_EXIT',subset=subset,on_N=on,on_exit=oe,off_N=off,off_exit=fe,R_on=ron,R_off=roff,on_target_retention=1-ron,selectivity=roff-ron,status='COMPUTED' if on and off else 'INSUFFICIENT_SUPPORT',raw_marker_D=raw,all_transferred=alltrans))
            for fi,fate in enumerate(FATES):
                count=int(f[pi,ri,fi]);flow.append(dict(source=a,target=b,region=r,kind='RECIPIENT_FATE',subset='ALL_FIXED_ROI',fate=fate,count=count,all_transferred=alltrans,raw_marker_D=raw,fraction_all_transferred=count/alltrans if alltrans else np.nan,fraction_fixed_raw_marker=count/raw if raw else np.nan))
            for key,col in [('cell_to_cell',2),('cell_to_background',3),('background_to_cell',4),('unchanged',5)]:flow.append(dict(source=a,target=b,region=r,kind='OWNER_TRANSITION',transition=key,count=int(tr[pi,ri,col]),all_transferred=alltrans,raw_marker_D=raw))
    genes=[]
    for pi,(a,b) in enumerate(j['pairs']):
        for ri,r in enumerate(REGIONS):
            for gi,vals in enumerate(z['per_gene'][pi,ri]):
                if j['genes'][gi] in MARKERS:genes.append(dict(source=a,target=b,region=r,gene_index=gi,gene=j['genes'][gi],marker_lineage=MARKERS.get(j['genes'][gi],0),on_N=int(vals[0]),on_exit=int(vals[1]),off_N=int(vals[2]),off_exit=int(vals[3]),status='COMPUTED' if vals[0] and vals[2] else 'INSUFFICIENT_SUPPORT'))
    atomic_csv(out/'mass.csv',pd.DataFrame(rows));atomic_csv(out/'flows.csv',pd.DataFrame(flow));atomic_csv(out/'risk.csv',pd.DataFrame(risk));atomic_csv(out/'per_gene.csv',pd.DataFrame(genes))
    # Every gene-direction denominator is kept; numeric gene indices refer to the frozen mapping.
    return pd.DataFrame(rows),pd.DataFrame(flow)

def intervals(cfg,out,ctx,curves_path=None):
    out=Path(out);j=read(out/'schema.json');z=np.load(out/'block_sums.npz');n=j['blocks'];rng=np.random.default_rng(cfg['seed']);W=np.vstack([np.bincount(rng.integers(n,size=n),minlength=n) for _ in range(cfg['bootstrap_replicates'])]);rows=[]
    def ratio(a,b):return np.divide(a,b,out=np.full(np.broadcast_shapes(np.shape(a),np.shape(b)),np.nan),where=np.asarray(b)>0)
    def record(a,b,r,metric,point,draw,den):
        good=np.isfinite(draw);lo,hi=np.quantile(draw[good],[.025,.975]) if good.any() else (np.nan,np.nan)
        rows.append(dict(source=a,target=b,region=r,metric=metric,estimate=point,ci_low=lo,ci_high=hi,valid_bootstrap=int(good.sum()),replicates=len(W),block_um=1000,blocks=n,bt_interfaces=j['bt_interfaces'],raw_marker_D=int(den),ci='paired fixed 1000um spatial block percentile',status='COMPUTED' if good.any() else 'INSUFFICIENT_SUPPORT'))
    for pi,(a,b) in enumerate(j['pairs']):
        ai=j['methods'].index(a);bi=j['methods'].index(b)
        for ri,r in enumerate(REGIONS):
            ma,mb=z['mass'][ai,ri],z['mass'][bi,ri];da=ma.sum(1);db=mb.sum(1)
            if not np.array_equal(da,db):raise ValueError('method-specific marker raw denominator')
            record(a,b,r,'fixed_roi_off_target_fraction_delta',ratio(mb[:,1].sum(),db.sum())-ratio(ma[:,1].sum(),da.sum()),ratio(W@mb[:,1],W@db)-ratio(W@ma[:,1],W@da),da.sum())
            record(a,b,r,'fixed_roi_consistent_fraction_delta',ratio(mb[:,0].sum(),db.sum())-ratio(ma[:,0].sum(),da.sum()),ratio(W@mb[:,0],W@db)-ratio(W@ma[:,0],W@da),da.sum())
            for sc,metric in ((2,'fixed_roi_other_unmatched_fraction_delta'),(3,'fixed_roi_unassigned_fraction_delta')):
                record(a,b,r,metric,ratio(mb[:,sc].sum(),db.sum())-ratio(ma[:,sc].sum(),da.sum()),ratio(W@mb[:,sc],W@db)-ratio(W@ma[:,sc],W@da),da.sum())
            ex=z['exits'][pi,ri,0];sel=ratio(ex[:,3].sum(),ex[:,2].sum())-ratio(ex[:,1].sum(),ex[:,0].sum());draw=ratio(W@ex[:,3],W@ex[:,2])-ratio(W@ex[:,1],W@ex[:,0]);record(a,b,r,'R_off_minus_R_on',sel,draw,da.sum())
            f=z['fates'][pi,ri];tr=z['transfers'][pi,ri,:,0];bal=f[:,0]-f[:,1];record(a,b,r,'favorability_balance_per_all_transfer',ratio(bal.sum(),tr.sum()),ratio(W@bal,W@tr),da.sum())
    pairlist=[tuple(p) for p in j['pairs']]
    parent='P2_MORPH';gonly='G_SCORE_ONLY_REF_GATED';full='P2_MORPH_GT'
    if (parent,gonly) in pairlist and (parent,full) in pairlist:
        gi=pairlist.index((parent,gonly));fi=pairlist.index((parent,full));mi=j['methods'].index(parent)
        for ri,r in enumerate(REGIONS):
            eg,ef=z['exits'][gi,ri,0],z['exits'][fi,ri,0]
            if not np.array_equal(eg[:,[0,2]],ef[:,[0,2]]):raise ValueError('ablation must retain identical parent on/off origin risk sets')
            def select(e,w=None):
                q=e.sum(0) if w is None else w@e
                return ratio(q[...,3],q[...,2])-ratio(q[...,1],q[...,0])
            den=z['mass'][mi,ri].sum(1)
            record(gonly,full,r,'type_score_selectivity_increment_same_parent_origins',select(ef)-select(eg),select(ef,W)-select(eg,W),den.sum())
            bg=z['fates'][gi,ri,:,0]-z['fates'][gi,ri,:,1];bf=z['fates'][fi,ri,:,0]-z['fates'][fi,ri,:,1]
            record(gonly,full,r,'type_score_favorability_increment_per_fixed_raw_marker',ratio((bf-bg).sum(),den.sum()),ratio(W@(bf-bg),W@den),den.sum())
    atomic_csv(out/'intervals.csv',pd.DataFrame(rows));return pd.DataFrame(rows)
