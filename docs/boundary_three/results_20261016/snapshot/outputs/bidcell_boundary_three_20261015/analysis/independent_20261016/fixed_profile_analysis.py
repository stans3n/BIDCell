"""Latest authorized dual-profile assessment, preserving original frozen sc gates.

Run after analyze.py --matrices. Reads raw fixed/sc score CSVs only. All writes
stay in this independent-analysis directory. No source/gate/config modification.
"""
import json
import numpy as np
import pandas as pd
import analyze as a

d=json.loads((a.OUT/'main_evidence.json').read_text())
ident=json.loads((a.OUT/'source_identities.json').read_text());a.identities.update(ident)
checks=json.loads((a.OUT/'checks.json').read_text());a.checks.extend(checks)
pop=a.readc(a.RUN/'frozen_strata/fixed_cell_groups.csv');ss=a.readj(a.RUN/'frozen_strata/strata.json')
fixed={};sc={}
for n,e in d['endpoints'].items():
    root=a.Path(e['evaluation_path'])
    fixed[n]=a.frame(root/'fixed/fixed_population.csv',['cell_id','positive_F1','negative_F1'])
    sc[n]=a.frame(root/'paper_aligned_v1/markers_per_cell.csv',['cell_id','positive_F1','negative_F1'])
fixed['A2_NATIVE']=fixed['BASE_MATCH'];sc['A2_NATIVE']=sc['BASE_MATCH']
rows=[];spatialrows=[];boot={}
for key,c in d['comparisons'].items():
    an,bn=c['candidate'],c['control'];fx,fy=fixed[an],fixed[bn];sx,sy=sc[an],sc[bn]
    dual={};c['fixed_spatial']={str(side):{} for side in [500,1000,2000]}
    for m in ['positive_F1','negative_F1']:
        av=fx[m].to_numpy();bv=fy[m].to_numpy();cm=np.isfinite(av)&np.isfinite(bv)
        am=float(np.nanmean(av));bm=float(np.nanmean(bv));delta=am-bm
        a.check(key+' fixed identical ID cohort '+m,np.array_equal(np.isfinite(av),np.isfinite(bv)) and cm.sum()==85421)
        four=cm&np.isfinite(sx[m])&np.isfinite(sy[m]);v=c['metrics'][m]
        v['fixed_delta']=v.pop('fixed_auxiliary_delta',delta)
        v['fixed_candidate_mean']=am;v['fixed_control_mean']=bm;v['fixed_candidate_n']=int(np.isfinite(av).sum());v['fixed_control_n']=int(np.isfinite(bv).sum())
        v['four_profile_common_ID_n']=int(four.sum())
        v['four_profile_common_ID_sc_delta']=float((sx.loc[four,m]-sy.loc[four,m]).mean())
        v['four_profile_common_ID_fixed_delta']=float((fx.loc[four,m]-fy.loc[four,m]).mean())
        for side in [500,1000,2000]:
            nb=ss['grids'][str(side)]['n_blocks'];z,b,*_=a.spatial(av,bv,pop['block_'+str(side)].to_numpy(),nb)
            c['fixed_spatial'][str(side)][m]=z;boot[(key,side,m)]=b
            spatialrows.append(dict(comparison=key,side_um=side,metric=m,profile='fixed',ci_lower=z['ci95'][0],ci_upper=z['ci95'][1],valid=z['valid'],n_blocks=nb,LOBO_min=z['LOBO_min'],LOBO_max=z['LOBO_max']))
    p=c['metrics']['positive_F1']['fixed_delta'];n=c['metrics']['negative_F1']['fixed_delta']
    c['fixed_point_direction_favorable']=p>=-1e-8 and n<=1e-8 and (p>1e-8 or n<-1e-8)
    c['fixed_both_primary_CIs_favorable']=c['fixed_spatial']['1000']['positive_F1']['ci95'][0]>0 and c['fixed_spatial']['1000']['negative_F1']['ci95'][1]<0
    c['current_assessment_policy']='scClassify and fixed P/N assessed in parallel by latest explicit user instruction; stored status/flags remain frozen original sc-primary program output'
    row=dict(comparison=key,candidate=an,control=bn,sc_delta_P=c['metrics']['positive_F1']['delta'],sc_delta_N=c['metrics']['negative_F1']['delta'],fixed_delta_P=p,fixed_delta_N=n,
        fixed_n_P=c['metrics']['positive_F1']['fixed_candidate_n'],fixed_n_N=c['metrics']['negative_F1']['fixed_candidate_n'],
        fixed_P_CI_low=c['fixed_spatial']['1000']['positive_F1']['ci95'][0],fixed_P_CI_high=c['fixed_spatial']['1000']['positive_F1']['ci95'][1],
        fixed_N_CI_low=c['fixed_spatial']['1000']['negative_F1']['ci95'][0],fixed_N_CI_high=c['fixed_spatial']['1000']['negative_F1']['ci95'][1],
        frozen_sc_status=c['status'],fixed_point_direction_favorable=c['fixed_point_direction_favorable'],fixed_both_primary_CIs_favorable=c['fixed_both_primary_CIs_favorable'],
        global_guards_pass=c['guards_pass'],fixed_ID_local_adverse_n=len(c['local_fixed_ID_adverse']),
        four_profile_common_ID_n=c['metrics']['positive_F1']['four_profile_common_ID_n'],
        four_profile_common_ID_sc_delta_P=c['metrics']['positive_F1']['four_profile_common_ID_sc_delta'],four_profile_common_ID_sc_delta_N=c['metrics']['negative_F1']['four_profile_common_ID_sc_delta'],
        four_profile_common_ID_fixed_delta_P=c['metrics']['positive_F1']['four_profile_common_ID_fixed_delta'],four_profile_common_ID_fixed_delta_N=c['metrics']['negative_F1']['four_profile_common_ID_fixed_delta'])
    rows.append(row)
for m in ['positive_F1','negative_F1']:
    c=d['p2_interaction'][m];c['fixed_delta_difference_in_differences']=d['comparisons']['P2_MORPH_minus_V2_MORPH']['metrics'][m]['fixed_delta']-d['comparisons']['P2_BASE_minus_BASE_MATCH']['metrics'][m]['fixed_delta'];c['fixed_spatial']={}
    for side in [500,1000,2000]:
        v=boot[('P2_MORPH_minus_V2_MORPH',side,m)]-boot[('P2_BASE_minus_BASE_MATCH',side,m)]
        c['fixed_spatial'][str(side)]=dict(ci95=np.quantile(v,[.025,.975]).tolist(),valid=int(np.isfinite(v).sum()))
d['current_scientific_assessment']='Dual-profile F1 assessment with equal visibility, requested after completion. Original SC-PRIMARY program decisions are preserved as historical frozen computation, not silently rerun or redefined.'
d['fixed_bootstrap_scope']='New read-only analysis of existing raw fixed scores: paired pooled sums/finite counts, original nucleus blocks 500/1000/2000um, 2000 PCG64 replicates seed20261013; field uncertainty only; no new experiment.'
frontiers={}
for profile in ['scclassify','fixed']:
    frontier=[]
    for n,e in d['endpoints'].items():
        p=e['metrics'][profile+'_positive_F1']['mean'];nn=e['metrics'][profile+'_negative_F1']['mean']
        dominant=[]
        for other,f in d['endpoints'].items():
            fp=f['metrics'][profile+'_positive_F1']['mean'];fn=f['metrics'][profile+'_negative_F1']['mean']
            if fp>=p and fn<=nn and (fp>p or fn<nn):dominant.append(other)
        if not dominant:frontier.append(n)
    frontiers[profile]=frontier
d['descriptive_point_PN_frontiers_among_10_compared_masks_only']=frontiers
a.savej('main_evidence.json',d);a.savej('checks.json',a.checks);a.savej('source_identities.json',a.identities)
pd.DataFrame(rows).to_csv(a.OUT/'dual_profile_comparisons.csv',index=False,na_rep='NA')
pd.DataFrame(spatialrows).to_csv(a.OUT/'fixed_spatial_uncertainty.csv',index=False,na_rep='NA')
print('Dual-profile results saved',len(rows),'comparisons; frontiers',frontiers)
