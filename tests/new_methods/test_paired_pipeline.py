"""Independent arithmetic oracles, real process locks, and an actual E1 report fixture."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.paired_v2_pipeline.common import DEFAULT_ROOT,INPUT,REPO,PYTHON,identity,lock,alive,process_identity,task_graph
from scripts.paired_v2_pipeline.math import pooled_deltas,spatial,spatial_flags,quartiles,guard_values,decide
from scripts.paired_v2_pipeline.freeze import classify_domains
from scripts.paired_v2_pipeline.reports import load_endpoint,local_summary,purity,write_json,spatial_diversity
from scripts.paired_v2_pipeline.runtime import terminate_owned


def main():
    out=DEFAULT_ROOT/'tests/pipeline';out.mkdir(exist_ok=True);checks={}
    sums_a=np.array([8.,0,0]);sums_b=np.array([5.,1,0]);counts=np.array([10,1,0]);draws=np.array([[0,0,0],[1,1,1],[2,2,2],[0,1,2],[0,0,1]])
    got=pooled_deltas(sums_a,counts,sums_b,counts,draws);want=np.array([.3,-1,np.nan,2/11,5/21])
    assert np.allclose(got,want,atol=1e-15,rtol=0,equal_nan=True)
    vals_a={'positive_F1':np.r_[np.full(10,.8),0,np.nan],'negative_F1':np.r_[np.full(10,.1),0,np.nan]}
    vals_b={'positive_F1':np.r_[np.full(10,.5),1,np.nan],'negative_F1':np.r_[np.full(10,.2),0,np.nan]}
    block=np.r_[np.zeros(10,int),1,2];result,sampled=spatial(vals_a,vals_b,block,3)
    assert np.allclose(result['positive_F1']['leave_one_out'],[-1,.3,2/11],atol=1e-15,rtol=0)
    assert result['positive_F1']['block_n_candidate'].tolist()==[10,1,0]
    assert 0<result['positive_F1']['n_invalid']<100 and len(sampled)==2000
    flags=spatial_flags(result,2/11,-1/11,3)
    assert {'spatial_sensitive','spatial_interval_ambiguous'}<=set(flags)
    checks['pooled_bootstrap']=dict(explicit_draws=draws,expected=want,actual=got,LOBO=result['positive_F1']['leave_one_out'],valid=result['positive_F1']['n_valid'],invalid=result['positive_F1']['n_invalid'],flags=flags,zero_metric_block_retained=True,no_redraw=True)
    edges,g=quartiles([0,1,1,2]);assert edges.tolist()==[.75,1,1.25] and g.tolist()==[0,1,1,3]
    e,z=quartiles([0,0,0,0]);assert e.tolist()==[0] and z.tolist()==[0]*4
    checks['quartiles']='linear/<=lower/duplicate edges and constant population passed'
    nuc=np.array([[0,0,0,0,0,7]],np.int32);n=np.array([[0,1,3,2,4,1]],np.uint32)
    u=np.array([[0,1,1,0,0,0]],bool);h=np.array([[0,0,1,0,1,0]],bool)
    assert classify_domains(nuc,n,u,h).tolist()==[[0,1,2,3,4,5]]
    checks['support_domains']='all five exclusive exterior domains, Unknown priority, separate nuclei passed'
    base=dict(capture=.7,area_mean=100.,area_median=90.,area_p95=250.,qc_retention=.95,unassigned=.1,disconnected8=10)
    guards=guard_values(base,base)
    for dp,dn,ff,expected in [(0,0,['localized_adverse'],'no_expression_gain'),(-.001,-.001,[],'tradeoff_or_negative'),(.001,0,[],'priority_signal'),(.00001,0,[],'small_signal'),(.001,0,['population_sensitive'],'exploratory_signal')]:
        assert decide(dp,dn,guards,ff)['status']==expected
    assert decide(None,0,guards,[])['status']=='comparison_incomplete'
    assert decide(.001,0,guards,[],correct=False)['status']=='implementation_invalid'
    adverse=guard_values({**base,'capture':.69},base);assert decide(.001,0,adverse,[])['status']=='tradeoff_or_negative'
    assert not decide(.001,0,adverse,[])['magnitude_priority'] and not decide(-.001,-.001,guards,[])['magnitude_priority']
    checks['decision_precedence']='all seven statuses and adverse-global-guard precedence passed; fixed never input'
    # Real OS processes/locks and identity-based termination; no model, fake weights or formal task.
    p=out/'exclusive_fixture.lock';held=lock(p)
    try:
        try:duplicate=lock(p)
        except BlockingIOError:pass
        else:duplicate.close();raise AssertionError('duplicate lock accepted')
    finally:held.close()
    log=(out/'owned_process.log').open('w')
    child=subprocess.Popen([PYTHON,'-B','-c','import time; print("owned fixture started",flush=True); time.sleep(30)'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    ident=process_identity(child.pid);assert alive(ident)
    bad={**ident,'start_ticks':'invalid'};assert not alive(bad)
    terminate_owned(child,ident);log.close();assert child.poll() is not None and not alive(ident)
    checks['actual_process_controls']=dict(duplicate_lock_rejected=True,wrong_identity_not_alive=True,owned_group_terminated=True,exit_code=child.returncode)
    graph=task_graph();assert len(graph)==15 and [t['arm'] for t in graph if t['stage']=='train']==['BASE_MATCH','V2_INTERACT']
    count=next(i for i,t in enumerate(graph) if t['stage']=='freeze_counts');assert all(count<i for i,t in enumerate(graph) if t['stage']=='evaluate')
    assert graph[-1]['stage']=='package' and all(t['depends_on']==[graph[i-1]['id']] for i,t in enumerate(graph) if i)
    checks['whole_graph']='both serial trains registered; count freeze before both evaluate stages; full dependency chain passed'
    meta=np.load(INPUT/'prepared/nucleus_metadata.npz');frame,aggregate,row=load_endpoint('V2_MORPH',meta['ids'].astype(int),{})
    old=pd.read_csv('/home/zwan0726/work/outputs/bidcell_historical_dual_20261013/main_table.csv').set_index('method').loc['V2_MORPH']
    for m in ('positive_F1','negative_F1','positive_Precision','positive_Recall','negative_Precision','negative_Recall'):
        for profile in ('scclassify','fixed'):
            assert np.isclose(row[profile+'_'+m],old[profile+'_'+m],atol=1e-10,rtol=1e-8)
            assert row[profile+'_'+m+'_n']==old[profile+'_'+m+'_n']
    for c in ('capture_pct','mean_cell_area_um2','disconnected_cells_8','scclassify_qc_cells','scclassify_unassigned_pct','scclassify_expression_pearson_type_mean','scclassify_detection_pearson_type_mean'):
        assert np.isclose(row[c],old[c],atol=1e-10,rtol=1e-8)
    empty=local_summary(frame,np.zeros(len(frame),bool));assert empty['n_ids']==0 and empty['area_mean'] is None and empty['unassigned'] is None
    checks['real_E1_report']=dict(status='passed',all_ids=len(frame),row=row,empty_fixed_group=empty,scope='read-only actual full E1 matrix/evaluation/mask fixture, no new arm scores or formal computation')
    table=pd.read_csv('/home/zwan0726/work/outputs/bidcell_historical_dual_20261013/main_table.csv');spec=purity(table)
    for profile in ('scclassify','fixed'):
        assert np.allclose(table[profile+'_Purity_F1'],pd.read_csv('/home/zwan0726/work/outputs/bidcell_historical_dual_20261013/main_table.csv')[profile+'_Purity_F1'],atol=1e-12,rtol=0)
    checks['purity']='actual old18 derived values replayed; old table read-only; missing/zero policy preserved'
    q={'bindings':{'authority':identity(DEFAULT_ROOT/'prepared/authoritative_protocol.json'),
        'spatial_wrapper':identity('/home/zwan0726/work/scripts/bidcell_benchmark/recompute_historical_spatial_diversity.R')}}
    actual=spatial_diversity(q,out,arms=('V2_MORPH',))
    for _,r in actual.iterrows():
        assert np.isclose(r.pearson,old['scclassify_spatial_entropy_vs_'+r.metric],atol=1e-10,rtol=1e-8)
    checks['real_author_spatial_R']=dict(status='passed',rows=actual.to_dict('records'),scope='actual E1 own-QC author R computation in private test output; old outputs read-only')
    components={str(p.relative_to(REPO)):identity(p) for p in (REPO/'scripts/paired_v2_pipeline').glob('*') if p.suffix in ('.py','.sh')}
    write_json(out/'verification.json',dict(status='passed',checks=checks,producer=identity(__file__),tested_components=components,no_formal_training=True,no_formal_table_writes=True))
    print('passed pipeline math, controls, complete graph and real E1 report fixture',flush=True)


if __name__=='__main__':main()
