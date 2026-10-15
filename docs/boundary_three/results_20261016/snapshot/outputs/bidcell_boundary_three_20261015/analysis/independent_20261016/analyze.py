"""Independent read-only result analysis. No imports from the scientific repository.

Run using the frozen bidcell_local Python. Every write stays beside this script.
--matrices adds chunked own-count/QC verification and A1 fold recomputation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RUN = OUT.parents[1] / 'boundary-three-20261015-r02'
SRC = Path('/home/zwan0726/work/projects/BIDCell_boundary_three_20261015')
METS = ['positive_F1','negative_F1','positive_Precision','positive_Recall','negative_Precision','negative_Recall']
N = 107955
DEN = 34442716
identities = {}
checks = []

def source(p):
    p=Path(p)
    if str(p) not in identities:
        h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
        identities[str(p)]={'bytes':p.stat().st_size,'sha256':h.hexdigest()}
    return p

def readj(p):return json.loads(source(p).read_text())
def readc(p,**kw):return pd.read_csv(source(p),**kw)
def check(name,v,**details):
    checks.append(dict(check=name,passed=bool(v),**details))
    if not v:raise AssertionError(name+': '+str(details))
def clean(v):
    if isinstance(v,dict):return {str(k):clean(a) for k,a in v.items()}
    if isinstance(v,(list,tuple,np.ndarray)):return [clean(a) for a in v]
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,float) and not np.isfinite(v):return None
    return v
def savej(name,v):(OUT/name).write_text(json.dumps(clean(v),indent=2,allow_nan=False)+'\n')
def frame(p,cols=None):
    x=readc(p,usecols=cols)
    check('unique cell IDs: '+str(p),not x.cell_id.duplicated().any())
    check('original cell IDs: '+str(p),set(x.cell_id)<=set(range(1,N+1)))
    return x.set_index('cell_id').reindex(range(1,N+1))
def stats(d,ix=None):
    a=d if ix is None else d.iloc[ix]
    q=a.qc_pass.to_numpy(bool)
    return dict(capture=int(a.total_transcripts.sum())/DEN,
        rna=int(a.total_transcripts.sum()),area_mean=float(a.area_um2.mean()),
        area_median=float(a.area_um2.median()),area_p95=float(a.area_um2.quantile(.95)),
        qc_retention=float(q.mean()),unassigned=float((a.sc_label.eq('unassigned')&q).sum()/q.sum()) if q.any() else None,
        disconnected8=int((a.components_8>1).sum()),n_ids=len(a),n_qc=int(q.sum()))
def guards(a,b,include_capture=True):
    out={}
    for k,t,relative in [('capture',-.005,False),('area_mean',.05,True),('area_median',.1,True),('area_p95',.1,True),('qc_retention',-.005,False),('unassigned',.01,False),('disconnected8',0,False)]:
        if k=='capture' and not include_capture:continue
        av,bv=a[k],b[k]
        v=None if av is None or bv is None or relative and bv==0 else abs(av/bv-1) if relative else av-bv
        passed=None if v is None else v>=t if t<0 else v<=t
        out[k]=dict(candidate=av,control=bv,change=v,threshold=t,passed=passed)
    return out
def spatial(a,b,blocks,nb,seed=20261013):
    draw=np.random.Generator(np.random.PCG64(seed)).integers(nb,size=(2000,nb))
    aa=np.isfinite(a);bb=np.isfinite(b)
    sa=np.bincount(blocks,weights=np.where(aa,a,0),minlength=nb)
    sb=np.bincount(blocks,weights=np.where(bb,b,0),minlength=nb)
    na=np.bincount(blocks,weights=aa,minlength=nb);nn=np.bincount(blocks,weights=bb,minlength=nb)
    wa=na[draw].sum(1);wb=nn[draw].sum(1);ok=(wa>0)&(wb>0)
    boot=np.full(2000,np.nan);boot[ok]=sa[draw].sum(1)[ok]/wa[ok]-sb[draw].sum(1)[ok]/wb[ok]
    loa=na.sum()-na;lob=nn.sum()-nn;lo=np.full(nb,np.nan);valid=(loa>0)&(lob>0)
    lo[valid]=(sa.sum()-sa[valid])/loa[valid]-(sb.sum()-sb[valid])/lob[valid]
    return dict(ci95=np.quantile(boot[ok],[.025,.975]).tolist(),valid=int(ok.sum()),n_blocks=nb,
                LOBO_min=float(np.nanmin(lo)),LOBO_max=float(np.nanmax(lo))),boot,sa,sb,na,nn

def main(matrices=False):
    table=readc(RUN/'reports/endpoint_table.csv')
    cfg=readj(RUN/'configs/run.json')
    pop=readc(RUN/'frozen_strata/fixed_cell_groups.csv').merge(readc(RUN/'frozen_counts/fixed_count_groups.csv'),on='cell_id',validate='one_to_one')
    check('frozen original IDs ordered',np.array_equal(pop.cell_id,np.arange(1,N+1)))
    strata=readj(RUN/'frozen_strata/strata.json')
    endpoints={};frames={};aggs={};metricrows=[]
    for name,rows in table.groupby('method',sort=False):
        if name=='A2_NATIVE':continue
        row=rows.iloc[0];ev=Path(row.evaluation_path);mat=Path(row.matrix_path)
        print('Raw endpoint',name,flush=True)
        qc=frame(ev/'paper_aligned_v1/qc_all_cells.csv')
        sc=frame(ev/'paper_aligned_v1/markers_per_cell.csv',['cell_id']+METS)
        ann=frame(ev/'paper_aligned_v1/annotation.csv',['cell_id','evaluation_label','raw_label','rejected'])
        fixed=frame(ev/'fixed/fixed_population.csv',['cell_id','anchor_label','label_availability']+METS)
        geom=frame(ev/'engineering/structure/cells_distribution.csv',['cell_id','area_um2'])
        cc=frame(ev/'engineering/structure/components_8.csv')
        expected=(qc.total_transcripts.between(20,2000)&qc.total_genes.ge(20)).to_numpy()
        check(name+' own raw QC predicate',np.array_equal(expected,qc.qc_pass))
        check(name+' scores exactly own QC',np.array_equal(sc.positive_F1.notna(),expected) and np.array_equal(sc.negative_F1.notna(),expected))
        check(name+' annotation exactly own QC',np.array_equal(ann.evaluation_label.notna(),expected))
        d=qc.copy();d['sc_label']=ann.evaluation_label.fillna('not_QC');d['raw_label']=ann.raw_label
        d['area_um2']=geom.area_um2;d['components_8']=cc.n_components_8
        for m in METS:d['sc_'+m]=sc[m];d['fixed_'+m]=fixed[m]
        receipt=readj(mat/'complete.json')
        check(name+' raw QC totals match matrix receipt',int(qc.total_transcripts.sum())==receipt['totals']['assigned'])
        check(name+' eligible fixed',receipt['totals']['eligible']==DEN)
        if matrices:
            arr=np.load(source(mat/'counts.npy'),mmap_mode='r')
            mapping=readc(mat/'cell_mapping.csv')
            check(name+' own integer matrix IDs/dimensions',arr.shape==(N,313) and np.issubdtype(arr.dtype,np.integer) and np.array_equal(mapping.canonical_cell_id,pop.cell_id))
            ts=[];gs=[];nonnegative=True
            for k in range(0,N,8192):
                c=arr[k:k+8192];nonnegative &= bool((c>=0).all());ts.extend(c.sum(1).tolist());gs.extend((c>0).sum(1).tolist())
            check(name+' actual matrix counts/QC totals/genes',nonnegative and np.array_equal(ts,qc.total_transcripts) and np.array_equal(gs,qc.total_genes))
            del arr
        agg=stats(d);aggs[name]=agg;frames[name]=d
        out=dict(method=name,**agg,capture_pct=100*agg['capture'],qc_cells=int(expected.sum()),unassigned_pct=100*agg['unassigned'],matrix_path=str(mat),evaluation_path=str(ev),matrix_sha256=row.matrix_sha256,checkpoint_sha256=row.checkpoint_sha256,metrics={})
        for m in METS:
            for profile,x,prefix in [('scclassify',sc[m],'scclassify'),('fixed',fixed[m],'fixed')]:
                v=x.to_numpy(float);good=np.isfinite(v);mean=float(v[good].mean()) if good.any() else None
                check(name+' raw '+profile+' '+m+' mean/finite',np.isclose(mean,float(row[prefix+'_'+m]),rtol=0,atol=2e-14) and good.sum()==int(row[prefix+'_'+m+'_n']))
                z=dict(mean=mean,n_finite=int(good.sum()),n_original_ids=N,n_unavailable_all_original=int((~good).sum()),n_zero=int((v[good]==0).sum()))
                out['metrics'][profile+'_'+m]=z;metricrows.append(dict(method=name,profile=profile,metric=m,**z))
        out['fixed_panel_ids']=int(fixed.anchor_label.notna().sum())
        check(name+' fixed panel size and NA retained',out['fixed_panel_ids']==103000 and fixed.positive_F1.notna().sum()==85421)
        for col,actual in [('scclassify_qc_cells',expected.sum()),('capture_pct',100*agg['capture']),('disconnected_cells_8',agg['disconnected8']),('mean_cell_area_um2',agg['area_mean'])]:
            check(name+' endpoint table '+col,np.isclose(float(row[col]),actual,atol=1e-12,rtol=0))
        endpoints[name]=out
    # NATIVE is an alias, not an additional unique scientific endpoint.
    frames['A2_NATIVE']=frames['BASE_MATCH'];aggs['A2_NATIVE']=aggs['BASE_MATCH']
    pairs={};origins={}
    for path in sorted(RUN.glob('**/decision.json')):
        if path.parent.name=='A1_CORE':continue
        raw=readj(path)
        for k,v in raw.get('comparisons',{}).items():
            a,b=v['candidate'],v['control'];key=a+'_minus_'+b
            pairs[key]=(a,b);origins[key]=(path.parent,k,v)
    # Additional descriptive comparisons between already tested endpoints; no new methods.
    for a,b in [('A2_GT','A2_NATIVE'),('A2_SHUFFLED','A2_NATIVE'),('V2_MORPH','MORPH_NO_DAPI'),('V2_MORPH','MORPH_SIMPLE_FUSION'),('V2_MORPH','MORPH_NO_GEO')]:pairs[a+'_minus_'+b]=(a,b)
    groupdefs={}
    for family in ['original_type','density_quartile','count_quartile','block_500','block_1000','block_2000']:
        for level,idx in pop.groupby(family,sort=False).indices.items():groupdefs[(family,str(level))]=idx
    groupstats={name:{g:stats(d,ix) for g,ix in groupdefs.items()} for name,d in frames.items()}
    comparisons={};comprows=[];spatialrows=[];localrows=[];boots={}
    for key,(an,bn) in pairs.items():
        a,b=frames[an],frames[bn];aq=a.qc_pass.to_numpy();bq=b.qc_pass.to_numpy();both=aq&bq
        result=dict(candidate=an,control=bn,origin='prespecified' if key in origins else 'additional descriptive comparison of already-tested endpoints',
            qc_both=int(both.sum()),qc_entered=int((aq&~bq).sum()),qc_exited=int((~aq&bq).sum()),qc_neither=int((~aq&~bq).sum()),
            same_ID_label_changes=int((a.sc_label[both]!=b.sc_label[both]).sum()),guards=guards(aggs[an],aggs[bn]),metrics={},spatial={})
        local=[]
        for g in groupdefs:
            aa,bb=groupstats[an][g],groupstats[bn][g];gg=guards(aa,bb,False)
            rna=aa['rna']/bb['rna'] if bb['rna'] else None
            violations=[k for k,v in gg.items() if v['passed'] is False]+(['fixed_ID_RNA_retention'] if rna is not None and rna<.995 else [])
            if violations:local.append(dict(family=g[0],group=g[1],n_ids=aa['n_ids'],violations=violations,RNA_retention=rna))
            localrows.append(dict(comparison=key,family=g[0],group=g[1],n_ids=aa['n_ids'],RNA_retention=rna,candidate_RNA=aa['rna'],control_RNA=bb['rna'],violations=';'.join(violations)))
        result['local_fixed_ID_adverse']=local
        for m in METS:
            av=a['sc_'+m].to_numpy();bv=b['sc_'+m].to_numpy();common=both&np.isfinite(av)&np.isfinite(bv)
            am=float(np.nanmean(av));bm=float(np.nanmean(bv));delta=am-bm
            cm=float(np.mean(av[common]-bv[common]))
            result['metrics'][m]=dict(candidate_mean=am,control_mean=bm,delta=delta,common_ID_delta=cm,common_ID_n=int(common.sum()),cohort_migration_component=delta-cm,
                fixed_auxiliary_delta=float(np.nanmean(a['fixed_'+m])-np.nanmean(b['fixed_'+m])))
        for side in [500,1000,2000]:
            result['spatial'][str(side)]={};nb=strata['grids'][str(side)]['n_blocks'];block=pop['block_'+str(side)].to_numpy()
            for m in METS[:2]:
                z,boot,*_=spatial(a['sc_'+m].to_numpy(),b['sc_'+m].to_numpy(),block,nb)
                result['spatial'][str(side)][m]=z;boots[(key,side,m)]=boot
                spatialrows.append(dict(comparison=key,side_um=side,metric=m,**{k:v for k,v in z.items() if k!='ci95'},ci_lower=z['ci95'][0],ci_upper=z['ci95'][1]))
                if key in origins:
                    parent,oldkey,dec=origins[key];p=parent/'spatial'/f'{oldkey}_{side}_{m}_bootstrap.npy'
                    old=np.load(source(p));check(key+' independent bootstrap '+str(side)+' '+m,np.allclose(boot,old,rtol=0,atol=2e-14,equal_nan=True))
        dp=result['metrics']['positive_F1']['delta'];dn=result['metrics']['negative_F1']['delta']
        flags=[]
        for side in [500,1000,2000]:
            sp=result['spatial'][str(side)];p=sp['positive_F1'];n=sp['negative_F1'];f=[]
            if p['valid']<1900 or n['valid']<1900:f.append('spatial_insufficient')
            if dp>=-1e-8 and p['LOBO_min']<-1e-8 or dn<=1e-8 and n['LOBO_max']>1e-8:f.append('spatial_sensitive')
            if not (dp>1e-8 and p['ci95'][0]>0 or dn<-1e-8 and n['ci95'][1]<0):f.append('spatial_interval_ambiguous')
            if side==1000:flags+=f
            elif f:flags.append('scale_sensitive')
        if result['metrics']['positive_F1']['common_ID_delta']<-1e-8 or result['metrics']['negative_F1']['common_ID_delta']>1e-8:flags.append('population_sensitive')
        if local:flags.append('localized_adverse')
        if key in origins:
            parent,oldkey,dec=origins[key]
            result['producer_decision_source']=str(parent/'decision.json');result['producer_status']=dec['status']
            result['producer_localized_adverse_n']=len(dec['localized_adverse_groups'])
            # Include fixed spatial molecule denominators/capture alongside fixed-ID checks.
            dc=readc(parent/'domain_capture.csv');bc=readc(parent/'block_capture.csv')
            result['domain_capture']=dc[dc.comparison==oldkey].to_dict('records')
            adverse_domain=result['domain_capture'] and any(pd.notna(x['capture_delta_pp']) and x['capture_delta_pp']<-.5 for x in result['domain_capture'])
            adverse_block=bool(((bc.comparison==oldkey)&(bc.capture_delta_pp<-.5)).any())
            if adverse_domain or adverse_block:flags.append('localized_adverse')
            cat=readc(parent/(oldkey+'_owner_categories.csv'))
            check(key+' owner categories conserve denominator',cat.molecules.sum()==DEN)
            result['owner_categories']=cat.to_dict('records')
            changed=cat[~cat.category.str.startswith('unchanged')]
            result['changed_molecules']=int(changed.molecules.sum())
            target=changed.domain.isin(['same_type_competition','heterotypic_competition','unknown_competition'])
            result['changed_molecules_in_competition']=int(changed.loc[target,'molecules'].sum())
            result['changed_competition_share']=float(changed.loc[target,'molecules'].sum()/changed.molecules.sum()) if changed.molecules.sum() else None
        flags=sorted(set(flags));direction=dp>=-1e-8 and dn<=1e-8 and (dp>1e-8 or dn<-1e-8)
        gp=all(g['passed'] is True for g in result['guards'].values())
        status='tradeoff_or_negative' if not gp or dp<-1e-8 or dn>1e-8 else 'no_expression_gain' if abs(dp)<=1e-8 and abs(dn)<=1e-8 else 'exploratory_signal' if flags else 'priority_signal' if dp>=.001 or dn<=-.0001 else 'small_signal'
        result.update(status=status,direction_pass=direction,guards_pass=gp,reliability_flags=flags)
        if key in origins:check(key+' decision status independently reproduced',status==origins[key][2]['status'])
        comparisons[key]=result
        row=dict(comparison=key,candidate=an,control=bn,delta_P=dp,delta_N=dn,capture_delta_pp=100*(aggs[an]['capture']-aggs[bn]['capture']),
            RNA_retention=aggs[an]['rna']/aggs[bn]['rna'],qc_both=result['qc_both'],qc_entered=result['qc_entered'],qc_exited=result['qc_exited'],
            same_ID_label_changes=result['same_ID_label_changes'],common_ID_delta_P=result['metrics']['positive_F1']['common_ID_delta'],common_ID_delta_N=result['metrics']['negative_F1']['common_ID_delta'],
            status=status,guards_pass=gp,flags=';'.join(flags),local_fixed_ID_adverse_n=len(local),
            P_CI_low=result['spatial']['1000']['positive_F1']['ci95'][0],P_CI_high=result['spatial']['1000']['positive_F1']['ci95'][1],
            N_CI_low=result['spatial']['1000']['negative_F1']['ci95'][0],N_CI_high=result['spatial']['1000']['negative_F1']['ci95'][1])
        comprows.append(row)
    interaction={}
    for m in METS[:2]:
        dk=comparisons['P2_MORPH_minus_V2_MORPH']['metrics'][m]['delta']-comparisons['P2_BASE_minus_BASE_MATCH']['metrics'][m]['delta']
        interaction[m]={'delta_difference_in_differences':dk,'spatial':{}}
        for side in [500,1000,2000]:
            bb=boots[('P2_MORPH_minus_V2_MORPH',side,m)]-boots[('P2_BASE_minus_BASE_MATCH',side,m)]
            interaction[m]['spatial'][str(side)]=dict(ci95=np.quantile(bb,[.025,.975]).tolist(),valid=int(np.isfinite(bb).sum()))
    a1=analyze_a1(cfg,matrices)
    timing={}
    for n in ['MORPH_NO_DAPI','MORPH_SIMPLE_FUSION','MORPH_NO_GEO']:
        init=readj(RUN/n/'initialization_map.json');train=readj(RUN/n/'checkpoints/complete.json');pred=readj(RUN/n/'predictions'/n/'complete.json');cost=readj(RUN/n/'runtime_cost.json')
        timing[n]=dict(parameters={k:init[k] for k in ['total_parameters','added_parameters','actual_registered_trainable_parameters','absent_modules']},
            training={k:train[k] for k in ['actual_updates','view_cursor','empty_skips','wall_seconds','peak_vram_bytes']},prediction={k:pred[k] for k in ['wall_seconds','peak_vram_bytes','shifts']},node_wall=cost)
    for n in ['P2_BASE','P2_MORPH']:timing[n]=readj(RUN/n/'runtime_cost.json')
    evidence=dict(analysis_role='new independent post-resume result analyst; no producer/source edits',protocol='SC-PRIMARY-FIXED-v1.2-20methods-20261014',
        run_root=str(RUN),source_root=str(SRC),source_commit='ccb846cbf344bfbe965178cd8f9b738e3654c8b0',
        unique_mask_endpoints=10,A2_NATIVE_alias_of='BASE_MATCH',matrix_QC_independently_checked=matrices,
        endpoints=endpoints,comparisons=comparisons,p2_interaction=interaction,a1=a1,
        a2_domain=readj(RUN/'A2_CORE/domain.json'),a2_proposals=readc(RUN/'A2_CORE/proposal_and_rejection_counts.csv').to_dict('records'),timing=timing,
        limitations=['Single training seed and developed Rep1 field; spatial bootstrap is conditional field uncertainty, not seed/specimen uncertainty.',
        'No independent true owner reference; marker F1 and RNA changes do not establish owner accuracy.',
        'A1 heldout nuclear prediction is conditional on upstream full-nuclear type labels, and differs from nuclear-exterior application.',
        'Historical full Morph uses a canonical initial state reconstructed from frozen code/seed; new ablations use archived mappings.',
        'No A2+Morph/P2 stack was tested; no new loss, new architecture, additional seeds or sweeps were performed.'])
    for rel in ['scripts/paired_v2_pipeline/math.py','scripts/boundary_three/reports.py','scripts/boundary_three/core_scores.py','scripts/boundary_three/core_refine.py','scripts/boundary_three/a_core.py','bidcell/model/model/morphology_variants.py','scripts/boundary_three/morph.py']:source(SRC/rel)
    savej('main_evidence.json',evidence);savej('checks.json',checks);savej('source_identities.json',identities)
    pd.DataFrame(comprows).to_csv(OUT/'comparisons.csv',index=False,na_rep='NA')
    pd.DataFrame(spatialrows).to_csv(OUT/'spatial_uncertainty.csv',index=False,na_rep='NA')
    pd.DataFrame(localrows).to_csv(OUT/'local_guardrails.csv',index=False,na_rep='NA')
    pd.DataFrame(metricrows).to_csv(OUT/'raw_metric_denominators.csv',index=False,na_rep='NA')
    pd.DataFrame([{k:v for k,v in e.items() if k not in ['metrics']} for e in endpoints.values()]).to_csv(OUT/'endpoints.csv',index=False)
    print('Saved',OUT,'checks',len(checks),flush=True)

def analyze_a1(cfg,matrices):
    scores=readc(RUN/'A1_CORE/per_cell_scores.csv');cover=readc(RUN/'A1_CORE/cell_coverage.csv');dec=readj(RUN/'A1_CORE/decision.json')
    check('A1 ordered full original IDs',np.array_equal(scores.cell_id,cover.cell_id) and len(scores)==N and not scores.cell_id.duplicated().any())
    pair=scores.paired.to_numpy();ev=scores.evaluable.to_numpy();check('A1 paired population',int(pair.sum())==88102 and int(ev.sum())==88106)
    out=dict(n_original=N,n_evaluable=int(ev.sum()),n_paired=int(pair.sum()),coverage_reason_counts=cover.reason.value_counts().to_dict(),
        core_minus_type=float(scores.loc[pair,'core_minus_type'].mean()),core_minus_shuffle=float(scores.loc[pair,'core_minus_shuffle'].mean()),
        molecule_weighted_gain=float(np.average(scores.loc[pair,'core_minus_type'],weights=scores.loc[pair,'heldout_molecules'])),
        split_diagnostics={k:float(scores.loc[pair,k].mean()) for k in ['split_cosine','type_residual_cosine','within_type_centered_cosine']},spatial={},type_scope={})
    for side in [500,1000,2000]:
        xy=np.floor(cover.loc[pair,['x_um','y_um']].to_numpy()/side).astype(int);_,block=np.unique(xy,axis=0,return_inverse=True);nb=int(block.max()+1)
        vals=scores.loc[pair,['core','type','shuffle_mean']].to_numpy();boot=[]
        rng=np.random.Generator(np.random.PCG64(20261015));sums=np.zeros((nb,3));np.add.at(sums,block,vals);nn=np.bincount(block,minlength=nb)
        for _ in range(2000):
            ds=rng.integers(nb,size=nb);means=sums[ds].sum(0)/nn[ds].sum();boot.append([means[0]-means[1],means[0]-means[2]])
        boot=np.array(boot);old=np.load(source(RUN/f'A1_CORE/bootstrap_{side}.npy'))
        check('A1 independently recomputed bootstrap '+str(side),np.allclose(boot,old,rtol=0,atol=2e-14))
        out['spatial'][str(side)]=dict(n_blocks=nb,valid=2000,core_minus_type_ci95=np.quantile(boot[:,0],[.025,.975]).tolist(),core_minus_shuffle_ci95=np.quantile(boot[:,1],[.025,.975]).tolist())
    scopes=readj(RUN/'A1_CORE/type_scope.json')
    for t,x in scopes.items():
        use=pair&cover.type.eq(t).to_numpy();xy=np.floor(cover.loc[use,['x_um','y_um']].to_numpy()/1000).astype(int)
        _,bi=np.unique(xy,axis=0,return_inverse=True);nb=int(bi.max()+1)
        vv=scores.loc[use,['core','type','shuffle_mean']].to_numpy();ss=np.zeros((nb,3));np.add.at(ss,bi,vv);nn=np.bincount(bi,minlength=nb)
        rng=np.random.Generator(np.random.PCG64(20261015));bt=[]
        for _ in range(2000):
            dr=rng.integers(nb,size=nb);mm=ss[dr].sum(0)/nn[dr].sum();bt.append([mm[0]-mm[1],mm[0]-mm[2]])
        bt=np.array(bt);old=np.load(source(RUN/f"A1_CORE/bootstrap_type_{x['type_index']:02d}_1000.npy"))
        check('A1 independently recomputed type bootstrap '+t,np.allclose(bt,old,rtol=0,atol=2e-14))
        ct=float(scores.loc[use,'core_minus_type'].mean());cs=float(scores.loc[use,'core_minus_shuffle'].mean());ci=np.quantile(bt,[.025,.975],axis=0)
        coverage=int(use.sum())>=1000 and nb>=10
        status='GO_A2' if coverage and ct>=.01 and ci[0,0]>0 and ci[0,1]>0 else 'NO_USEFUL_INCREMENT' if coverage and (ci[1,0]<.01 or ci[1,1]<=0) else 'INCONCLUSIVE'
        check('A1 per-type original gate '+t,status==x['science_status'])
        out['type_scope'][t]={k:x.get(k) for k in ['type_index','science_status','coverage_ok']}|dict(n_cells=int(use.sum()),n_blocks=nb,core_minus_type=dict(mean=ct,ci95=ci[:,0].tolist()),core_minus_shuffle=dict(mean=cs,ci95=ci[:,1].tolist()),valid_replicates=2000)
    out['support']=readj(RUN/'A1_CORE/support_diagnostics.json')
    out['strata']=readc(RUN/'A1_CORE/strata_scores.csv').to_dict('records')
    if matrices:
        saved=np.load(source(RUN/'A1_CORE/core_counts.npz'));counts=saved['counts'];don=np.load(source(RUN/'A1_CORE/donor_tables.npz'))['donors'];ref=readj(cfg['inputs']['reference_region'])
        raw=np.asarray(ref['theta'],float);valid=np.asarray(ref['valid'],bool);theta=np.full_like(raw,1/raw.shape[1]);theta[valid]=np.maximum(raw[valid],1e-8);theta/=theta.sum(1,keepdims=True)
        types=cover.type_index.to_numpy();nn=counts.sum(2);tot=nn.sum(0);obs=np.zeros(N);typ=np.zeros(N);sh=np.zeros((20,N));fold=[]
        for f in [0,1]:
            for k in range(0,N,4096):
                ix=np.arange(k,min(k+4096,N));prior=theta[np.maximum(types[ix],0)];c=counts[f,ix];held=counts[1-f,ix]
                q=(c+20*prior)/(c.sum(1)[:,None]+20);den=np.maximum(tot[ix],1)
                obs[ix]+=(held*np.log(q)).sum(1)/den;typ[ix]+=(held*np.log(prior)).sum(1)/den
                for s in range(20):
                    jj=ix[pair[ix]];dd=don[s,f,jj]
                    checkok=np.all(dd!=jj) and np.all(types[dd]==types[jj]) and np.all(np.floor(np.log2(nn[f,dd]))==np.floor(np.log2(nn[f,jj])))
                    if not checkok:raise AssertionError('donor matching')
                    pr=theta[types[jj]];cs=counts[f,dd];qs=(cs+20*pr)/(cs.sum(1)[:,None]+20)
                    sh[s,jj]+=(counts[1-f,jj]*np.log(qs)).sum(1)/np.maximum(tot[jj],1)
            # Compute a separately normalized gain in each cross direction.
            ix=np.flatnonzero(ev);sdir=[]
            for k in range(0,len(ix),4096):
                rr=ix[k:k+4096];pr=theta[types[rr]];c=counts[f,rr];held=counts[1-f,rr];q=(c+20*pr)/(c.sum(1)[:,None]+20)
                sdir.extend(((held*(np.log(q)-np.log(pr))).sum(1)/held.sum(1)).tolist())
            fold.append(dict(train_fold=f,heldout_fold=1-f,n_cells=len(ix),cell_equal_core_minus_type=float(np.mean(sdir))))
        check('A1 observed fold recomputation',np.allclose(obs[ev],scores.loc[ev,'core'],atol=2e-14,rtol=0) and np.allclose(typ[ev],scores.loc[ev,'type'],atol=2e-14,rtol=0))
        check('A1 all 20 matched shuffle recomputation',np.allclose(sh[:,pair].mean(0),scores.loc[pair,'shuffle_mean'],atol=2e-14,rtol=0))
        out['cross_directions']=fold;out['shuffle_seed_effects']=[dict(seed=2026101500+s,core_minus_shuffle=float((obs[pair]-sh[s,pair]).mean())) for s in range(20)]
    return out

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--matrices',action='store_true');args=parser.parse_args();main(args.matrices)
