"""Historical B0 references requested after completion; no new experiment/gate."""
import json
import numpy as np
import pandas as pd
import analyze as a

d=json.loads((a.OUT/'main_evidence.json').read_text())
a.identities.update(json.loads((a.OUT/'source_identities.json').read_text()));a.checks.extend(json.loads((a.OUT/'checks.json').read_text()))
hist=a.Path('/home/zwan0726/work/outputs/bidcell_historical_dual_20261013')
h=a.readc(hist/'main_table.csv',encoding='utf-8-sig');row=h[h.method.eq('B0')].iloc[0]
root=hist/'methods/B0';inp=a.readj(root/'input.json');receipt=a.readj(root/'scoring_complete.json')
ev=a.Path(inp['source_evaluation']);er=a.readj(ev/'complete.json')
a.check('B0 re-scoring source evaluation receipt hash',a.identities[str(ev/'complete.json')]['sha256']==receipt['source']['sha256'])
pop=a.readc(a.RUN/'frozen_strata/fixed_cell_groups.csv');spec=a.readj(a.RUN/'frozen_strata/strata.json')
geom=a.frame(ev/'engineering/structure/cells_distribution.csv',['cell_id','area_um2','nucleus_x','nucleus_y'])
a.check('B0 same original nuclear coordinates support paired field CIs',np.allclose(geom.nucleus_x,pop.x_um,rtol=0,atol=1e-10) and np.allclose(geom.nucleus_y,pop.y_um,rtol=0,atol=1e-10))
qc=a.frame(ev/'paper_aligned_v1/qc_all_cells.csv');cc=a.frame(ev/'engineering/structure/components_8.csv')
ann=a.frame(ev/'paper_aligned_v1/annotation.csv',['cell_id','evaluation_label'])
f0=a.frame(root/'fixed/fixed_population.csv',['cell_id']+a.METS);s0=a.frame(root/'paper/markers_per_cell.csv',['cell_id']+a.METS)
a.check('B0 raw own QC threshold',np.array_equal(qc.qc_pass,qc.total_transcripts.between(20,2000)&qc.total_genes.ge(20)))
a.check('B0 raw own scores match QC',np.array_equal(s0.positive_F1.notna(),qc.qc_pass))
b=qc.copy();b['area_um2']=geom.area_um2;b['components_8']=cc.n_components_8;b['sc_label']=ann.evaluation_label.fillna('not_QC')
agg=a.stats(b);b0=dict(aggregate=agg,reported_scalar_row=row.to_dict(),metrics={},input=inp,evaluation_dependencies=er['dependencies'],
    inference_scope='Historical B0 is a useful external paper reference, but initialization/training trajectory/order are unmatched; endpoint differences cannot be attributed solely to the new mechanism. BASE_MATCH remains the causal/mechanistic paired control.')
for profile,values in [('scclassify',s0),('fixed',f0)]:
    for m in a.METS:
        v=values[m].to_numpy();v=v[np.isfinite(v)];mean=float(v.mean())
        a.check('B0 published raw '+profile+' '+m,np.isclose(mean,float(row[profile+'_'+m]),rtol=0,atol=2e-14) and len(v)==int(row[profile+'_'+m+'_n']))
        b0['metrics'][profile+'_'+m]=dict(mean=mean,n_finite=len(v))
for key,val in [('capture_pct',100*agg['capture']),('scclassify_qc_cells',agg['n_qc']),('disconnected_cells_8',agg['disconnected8']),('mean_cell_area_um2',agg['area_mean'])]:
    a.check('B0 published '+key,np.isclose(float(row[key]),val,rtol=0,atol=1e-10))
refs={};rows=[];spatialrows=[]
for name,e in d['endpoints'].items():
    evn=a.Path(e['evaluation_path']);sc=a.frame(evn/'paper_aligned_v1/markers_per_cell.csv',['cell_id','positive_F1','negative_F1']);fx=a.frame(evn/'fixed/fixed_population.csv',['cell_id','positive_F1','negative_F1'])
    aqc=a.frame(evn/'paper_aligned_v1/qc_all_cells.csv');aq=aqc.qc_pass.to_numpy();bq=qc.qc_pass.to_numpy()
    # Existing independent endpoint aggregates use the same all-ID denominators.
    ca={k:e[k] for k in agg if k in e};ca['n_qc']=e['qc_cells'];gg=a.guards(ca,agg)
    r=dict(candidate=name,control='B0',inference_scope=b0['inference_scope'],metrics={},spatial={},guards=gg,
        qc_both=int((aq&bq).sum()),qc_entered=int((aq&~bq).sum()),qc_exited=int((~aq&bq).sum()),capture_delta_pp=e['capture_pct']-float(row.capture_pct),
        area_mean_relative_change=e['area_mean']/agg['area_mean']-1,disconnected8_change=e['disconnected8']-agg['disconnected8'],qc_cells_change=e['qc_cells']-agg['n_qc'])
    for profile,aa,bb in [('scclassify',sc,s0),('fixed',fx,f0)]:
        r['spatial'][profile]={str(s):{} for s in [500,1000,2000]}
        for m in ['positive_F1','negative_F1']:
            av=aa[m].to_numpy();bv=bb[m].to_numpy();common=np.isfinite(av)&np.isfinite(bv)
            r['metrics'][profile+'_'+m]=dict(delta=float(np.nanmean(av)-np.nanmean(bv)),candidate_mean=float(np.nanmean(av)),control_mean=float(np.nanmean(bv)),candidate_n=int(np.isfinite(av).sum()),control_n=int(np.isfinite(bv).sum()),common_ID_n=int(common.sum()),common_ID_delta=float((av[common]-bv[common]).mean()))
            for side in [500,1000,2000]:
                z,_,*_=a.spatial(av,bv,pop['block_'+str(side)].to_numpy(),spec['grids'][str(side)]['n_blocks'])
                r['spatial'][profile][str(side)][m]=z
                spatialrows.append(dict(candidate=name,control='B0',profile=profile,metric=m,side_um=side,n_blocks=z['n_blocks'],valid=z['valid'],ci_lower=z['ci95'][0],ci_upper=z['ci95'][1],LOBO_min=z['LOBO_min'],LOBO_max=z['LOBO_max']))
    refs[name+'_minus_B0']=r
    rr={k:r[k] for k in ['candidate','control','capture_delta_pp','area_mean_relative_change','disconnected8_change','qc_cells_change','qc_both','qc_entered','qc_exited']}
    rr['global_guards_pass']=all(v['passed'] is True for v in gg.values())
    for profile in ['scclassify','fixed']:
        for m in ['positive_F1','negative_F1']:
            k=profile+'_'+m;rr[k+'_delta']=r['metrics'][k]['delta'];rr[k+'_candidate_n']=r['metrics'][k]['candidate_n'];rr[k+'_control_n']=r['metrics'][k]['control_n']
            rr[k+'_CI_low']=r['spatial'][profile]['1000'][m]['ci95'][0];rr[k+'_CI_high']=r['spatial'][profile]['1000'][m]['ci95'][1]
    rows.append(rr)
d['B0_historical_reference']=b0;d['B0_reference_comparisons']=refs
# Compact the evidence; full local rows remain in local_guardrails.csv and source receipts.
for c in d['comparisons'].values():
    local=c['local_fixed_ID_adverse'];c['local_fixed_ID_adverse_n']=len(local)
    c['local_fixed_ID_adverse']=[x for x in local if x['family'] in ['original_type','count_quartile','density_quartile']]
    c['full_local_guardrails_artifact']=str(a.OUT/'local_guardrails.csv')
for value in d['timing'].values():
    if 'node_wall' in value:value['node_wall']={k:v for k,v in value['node_wall'].items() if k!='stages'}
a.savej('main_evidence.json',d);a.savej('checks.json',a.checks)
for name in ['analyze.py','fixed_profile_analysis.py','b0_references.py']:a.source(a.OUT/name)
a.savej('source_identities.json',a.identities)
pd.DataFrame(rows).to_csv(a.OUT/'B0_reference_comparisons.csv',index=False,na_rep='NA')
pd.DataFrame(spatialrows).to_csv(a.OUT/'B0_spatial_uncertainty.csv',index=False,na_rep='NA')
print('B0 references saved',len(rows),'comparisons; all raw means verified; same original coordinates verified; checks',len(a.checks))
