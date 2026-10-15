"""Root checks the independent analyst against distinct raw-data reductions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from boundary_three_root_review import BASE, RUN, OUT, read, identity, close, write


def main():
    own=read(OUT/'raw_validation.json')
    independent=BASE/'analysis/independent_20261016'
    evidence=read(independent/'main_evidence.json')
    checks=read(independent/'checks.json')
    assert len(checks)>=680 and all(x['passed'] for x in checks)
    comparisons=evidence['comparisons']
    def lookup(a,b):
        for value in comparisons.values():
            ca='BASE_MATCH' if value['candidate']=='A2_NATIVE' else value['candidate']
            cb='BASE_MATCH' if value['control']=='A2_NATIVE' else value['control']
            if (ca,cb)==(a,b):return value
        raise ValueError((a,b))
    point_checks=0
    for x in own['comparisons']:
        value=lookup(x['candidate'],x['control'])['metrics'][x['metric']]
        close(x['own_QC_delta'],value['delta'])
        close(x['common_QC_delta'],value['common_ID_delta'])
        close(x['fixed_delta'],value['fixed_delta'] if x['metric'].endswith('_F1') else value['fixed_auxiliary_delta'])
        assert x['common_QC_n']==value['common_ID_n']
        point_checks+=4
    for x in own['dual_profile_1000um']:
        value=lookup(x['candidate'],x['control'])
        if x['profile']=='scclassify':value=value['spatial']['1000'][x['metric']]
        else:value=value['fixed_spatial']['1000'][x['metric']]
        assert np.allclose(x['ci95'],value['ci95'],atol=2e-14,rtol=0),(x,value)
    # Independent B0 field-interval check for consequential paper-reference claims.
    old=Path('/home/zwan0726/work/outputs/bidcell_historical_dual_20261013')
    groups=pd.read_csv('/home/zwan0726/work/outputs/bidcell_v2_paired_20261014/frozen_strata/fixed_cell_groups.csv').set_index('cell_id')
    ids=groups.index
    blocks=groups.block_1000.to_numpy(int);nb=int(blocks.max()+1)
    draws=np.random.Generator(np.random.PCG64(20261013)).integers(0,nb,size=(2000,nb))
    b0={}
    for profile,rel in [('scclassify','paper/markers_per_cell.csv'),('fixed','fixed/fixed_population.csv')]:
        b0[profile]=pd.read_csv(old/'methods/B0'/rel).set_index('cell_id').reindex(ids)
    b0_checks=[]
    for method in ('BASE_MATCH','V2_MORPH','A2_GT'):
        values=pd.read_csv(own['endpoints'][method]['population']['path']).set_index('cell_id').reindex(ids)
        other=evidence['B0_reference_comparisons'][method+'_minus_B0']
        for profile,prefix in [('scclassify','sc'),('fixed','fixed')]:
            for metric in ('positive_F1','negative_F1'):
                a=values[prefix+'_'+metric].to_numpy(float);b=b0[profile][metric].to_numpy(float)
                fa,fb=np.isfinite(a),np.isfinite(b)
                sa=np.bincount(blocks,weights=np.where(fa,a,0),minlength=nb)
                sb=np.bincount(blocks,weights=np.where(fb,b,0),minlength=nb)
                na=np.bincount(blocks,weights=fa,minlength=nb);nn=np.bincount(blocks,weights=fb,minlength=nb)
                delta=float(np.nanmean(a)-np.nanmean(b))
                ci=np.quantile(sa[draws].sum(1)/na[draws].sum(1)-sb[draws].sum(1)/nn[draws].sum(1),[.025,.975])
                observed=other['metrics'][profile+'_'+metric]
                close(delta,observed['delta'])
                assert np.allclose(ci,other['spatial'][profile]['1000'][metric]['ci95'],rtol=0,atol=2e-14)
                b0_checks.append(dict(candidate=method,control='B0',profile=profile,metric=metric,delta=delta,ci95=ci.tolist()))
    for col in ('core_minus_type','core_minus_shuffle'):
        close(own['A1'][col],evidence['a1'][col])
    for metric in ('positive_F1','negative_F1'):
        for profile,prefix in [('scclassify','delta_difference_in_differences'),('fixed','fixed_delta_difference_in_differences')]:
            arms=own['endpoints']
            val=(arms['P2_MORPH']['metrics'][profile+'_'+metric]['mean']-arms['V2_MORPH']['metrics'][profile+'_'+metric]['mean'])-(arms['P2_BASE']['metrics'][profile+'_'+metric]['mean']-arms['BASE_MATCH']['metrics'][profile+'_'+metric]['mean'])
            close(val,evidence['p2_interaction'][metric][prefix])
    write('independent_crosscheck.json',dict(status='passed',
        scope='Root distinct raw reductions agree with independent analyst: all shared point/commonQC/fixed deltas, 1000um dual-profile intervals, A1, four-grid interactions; B0 intervals freshly checked on BASE/fullMorph/A2_GT. Analyst additionally verifies all B0 pairs/all three scales and A1 folds/shuffles.',
        independent_checks=len(checks),point_checks=point_checks,primary_interval_checks=len(own['dual_profile_1000um']),
        B0_primary_interval_checks=b0_checks,analyst_evidence=identity(independent/'main_evidence.json'),
        analyst_checks=identity(independent/'checks.json'),root_raw=identity(OUT/'raw_validation.json'),
        producer=identity(__file__)))
    print('crosscheck passed',point_checks,len(own['dual_profile_1000um']),len(b0_checks))


if __name__=='__main__':main()
