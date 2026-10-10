"""CPU-only factual B−A stage/cell/transfer diagnostics from accepted artifacts."""
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.crgf.common import atomic_json,identity,sha256,verify_outputs
from .evaluation import read,publish

def run(config):
    q=read(config);hist=Path(q['historical_root']);out=Path(q['output_root'])/'b_cost';out.mkdir(exist_ok=True);start=time.monotonic()
    paths=[hist/'analysis/science'/x for x in ('pre_final_metrics.csv','owner_marker_transition_summary.csv','sensitivity_and_type_metrics.csv')]
    diag=hist/'comparison/diagnostics/B_minus_A'
    paths += [diag/x for x in ('boundary_transitions.csv','density_transitions.csv','type_ownership_transitions.csv','marker_consistency.csv','cell_ledger.csv','fixed_nuclear_density.csv','validation.json')]
    paths += [hist/a/'evaluation'/a/'diagnostics/stages'/f for a in ('A','B') for f in ('stage_marker_type_density_transitions.csv','stage_boundary_transitions.csv','fixed_B1_final.csv')]
    stage_names=('raw_shift0','shift0','raw_shift24','shift24','pre','tile_merge','split_cleanup','before_global_nuclear_restore','final')
    paths += [hist/a/'evaluation'/a/'diagnostics/stages'/('fixed_B1_'+stage+'.csv') for a in ('A','B') for stage in stage_names]
    common_path=hist/'comparison/fixed_anchor/common_qc_cell_ids.csv';paths.append(common_path)
    paths.append(Path(__file__).resolve())
    deps={str(p):identity(p) for p in paths}
    receipt=out/'complete.json'
    if receipt.exists():
        old=read(receipt)
        if old.get('dependencies')!=deps:raise ValueError('B cost inputs changed')
        verify_outputs(old);return old
    pre=pd.read_csv(paths[0]);rows=[];facts={}
    for metric in ('positive_F1','negative_F1','positive_Precision','negative_Precision','positive_Recall','negative_Recall'):
        delta={}
        for stage in ('pre','final'):
            values={a:float(pre[(pre.arm==a)&(pre.stage==stage)&(pre.metric==metric)]['mean'].iloc[0]) for a in ('A','B')}
            delta[stage]=values['B']-values['A'];rows.append(dict(metric=metric,stage=stage,A=values['A'],B=values['B'],B_minus_A=delta[stage],statistic='difference_of_available_cell_means',n_evaluable_A=int(pre[(pre.arm=='A')&(pre.stage==stage)&(pre.metric==metric)].n_evaluable.iloc[0]),n_evaluable_B=int(pre[(pre.arm=='B')&(pre.stage==stage)&(pre.metric==metric)].n_evaluable.iloc[0])))
        facts[metric]=dict(pre_delta=delta['pre'],final_delta=delta['final'],stage_amplification_delta=delta['final']-delta['pre'],final_over_pre=delta['final']/delta['pre'] if delta['pre'] else None,pre_to_final_fraction_of_final=(delta['final']-delta['pre'])/delta['final'] if delta['final'] else None)
    pd.DataFrame(rows).to_csv(out/'stage_deltas.csv',index=False)
    common_ids=pd.read_csv(common_path).cell_id.to_numpy();native=[]
    for stage in stage_names:
        fs={a:pd.read_csv(hist/a/'evaluation'/a/'diagnostics/stages'/('fixed_B1_'+stage+'.csv')).set_index('cell_id').reindex(common_ids) for a in ('A','B')}
        for metric in ('positive_F1','negative_F1','positive_Precision','negative_Precision','positive_Recall','negative_Recall'):
            d=fs['B'][metric]-fs['A'][metric];native.append(dict(stage=stage,metric=metric,population='historical_ABC_commonQC',n_eligible=len(d),n_evaluable=int(d.notna().sum()),n_evaluable_A=int(fs['A'][metric].notna().sum()),n_evaluable_B=int(fs['B'][metric].notna().sum()),statistic='paired_delta_on_joint_finite_domain',A_mean=fs['A'][metric].mean(),B_mean=fs['B'][metric].mean(),paired_B_minus_A=d.mean()))
    pd.DataFrame(native).to_csv(out/'native_stage_deltas.csv',index=False,na_rep='NA')
    breakdown=[]
    for name in ('boundary_transitions','density_transitions','type_ownership_transitions'):
        f=pd.read_csv(diag/(name+'.csv'));f.insert(0,'dimension',name);f.insert(0,'scope','B-A final owner transitions');breakdown.append(f)
    # Preserve detailed native operator strata, including marker source/destination.
    for arm in ('A','B'):
        stages=hist/arm/'evaluation'/arm/'diagnostics/stages'
        for name in ('stage_marker_type_density_transitions','stage_boundary_transitions'):
            p=stages/(name+'.csv');deps[str(p)]=identity(p);f=pd.read_csv(p);f.insert(0,'dimension',name);f.insert(0,'scope',arm+' native stages');breakdown.append(f)
    pd.concat(breakdown,ignore_index=True).to_csv(out/'type_region_breakdown.csv',index=False,na_rep='NA')
    m=pd.read_csv(diag/'marker_consistency.csv');known={'neutral','positive','negative','both'}
    m['both_evaluable']=m.source.isin(known)&m.destination.isin(known)
    m['source_unavailable']=~m.source.isin(known|{'background'});m['destination_unavailable']=~m.destination.isin(known|{'background'})
    m['source_positive']=m.source.isin({'positive','both'});m['destination_positive']=m.destination.isin({'positive','both'});m['source_negative']=m.source.isin({'negative','both'});m['destination_negative']=m.destination.isin({'negative','both'})
    m.to_csv(out/'paired_marker_transfers.csv',index=False)
    # Pair per-cell fixed-label scores to frozen nuclear density and label strata.
    density=pd.read_csv(diag/'fixed_nuclear_density.csv').set_index('cell_id')
    frames={a:pd.read_csv(hist/a/'evaluation'/a/'diagnostics/stages/fixed_B1_final.csv').set_index('cell_id') for a in ('A','B')}
    common=frames['A'].index.intersection(frames['B'].index);delta=frames['B'].loc[common,['positive_F1','negative_F1']]-frames['A'].loc[common,['positive_F1','negative_F1']]
    delta['type']=frames['B'].loc[common,'anchor_label'];delta=delta.join(density[['density_quartile']]);delta.to_csv(out/'paired_cell_costs.csv',na_rep='NA')
    by=[]
    for keys,part in delta.groupby(['type','density_quartile'],dropna=False):
        for metric in ('positive_F1','negative_F1'):
            v=part[metric];by.append(dict(type=keys[0],density_quartile=keys[1],metric=metric,n_eligible=len(v),n_evaluable=int(v.notna().sum()),mean_delta=v.mean(),n_increase=int((v>0).sum()),n_decrease=int((v<0).sum()),n_equal=int((v==0).sum())))
    pd.DataFrame(by).to_csv(out/'paired_type_density_scores.csv',index=False,na_rep='NA')
    atomic_json(out/'facts.json',dict(status='succeeded',stage_arithmetic=facts,positive_increment_retained_fraction=facts['positive_F1']['final_over_pre'],negative_final_difference_formed_pre_to_final_fraction=facts['negative_F1']['pre_to_final_fraction_of_final'],interpretation='Stage difference decomposition, not causal contamination attribution; positive/negative markers overlap and are not additive exclusive categories. Stage deltas subtract available-cell means; native paired deltas use the joint finite domain, so Precision can differ when zero prediction denominators produce NA.',annotation_reruns=0,training_runs=0,unlocated_operator_effects='No causal operator isolation: native stage arithmetic only',wall_seconds=time.monotonic()-start))
    atomic_json(out/'coverage.json',dict(status='succeeded',required={k:True for k in ('stage_deltas','type_region_breakdown','paired_marker_transfers','facts','paired_type_density_scores')},reused_existing_accepted_derivatives=True,unknown_source_destination_separate=True,source_inputs=deps))
    publish(out,dependencies=deps,cpu_only=True,wall_seconds=time.monotonic()-start)
    return read(receipt)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();run(a.config)
