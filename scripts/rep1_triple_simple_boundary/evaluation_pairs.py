"""Parent/BASE paired evidence required before a new FULL_EVALUATE receipt."""
from pathlib import Path
import importlib
import numpy as np
import pandas as pd
from .contracts import NEW,METRICS,SCALES,evaluation_pairs,comparison_id
from .common import (dependencies,identity,id_hash,read,check_receipt,inventory,write_receipt)
from .inputs import new_spec,resolve_parent
from . import populations,bt


def specs_for(cfg,manifest,name):
    pairs=evaluation_pairs(name);parent=resolve_parent(cfg,manifest,name)
    if parent['name']!=pairs[0][1]:raise ValueError('full evaluation true parent differs')
    return {name:new_spec(cfg,name),parent['name']:parent,'BASE_MATCH':manifest['endpoints']['BASE_MATCH']}


def run(cfg,manifest,name,callback=None):
    specs=specs_for(cfg,manifest,name);pairs=evaluation_pairs(name)
    out=Path(specs[name]['evaluation'])/'paired';dep=dependencies(cfg,'EVALUATION_PAIRS_'+name)
    if (out/'complete.json').exists():
        validate(cfg,manifest,name,specs=specs)
        return check_receipt(out/'complete.json',dep)
    tables=populations.prepare(cfg,manifest,specs,out/'populations',callback,pair_list=pairs)
    compare=importlib.import_module(__package__+'.comparisons')
    compare.run(cfg,manifest,specs,tables,out/'populations',out/'comparisons',callback,pair_list=pairs)
    bt.paired(cfg,manifest,specs,out/'bt',callback,pair_list=pairs)
    coverage=validate(cfg,manifest,name,specs=specs,require_complete=False)
    return write_receipt(out/'complete.json',dep,inventory(out),endpoint=name,pairs=[comparison_id(a,b) for a,b in pairs],
                         direction='candidate minus control',coverage=coverage)


def _same(actual,expected,description):
    if isinstance(expected,str) and expected=='' and pd.isna(actual):return
    if expected is None or (isinstance(expected,(float,np.floating)) and not np.isfinite(expected)):
        if not pd.isna(actual):raise ValueError(description+' expected explicit NA')
    elif isinstance(expected,(float,int,np.number)):
        if pd.isna(actual) or not np.isclose(actual,expected,rtol=0,atol=1e-12):raise ValueError(description+' numerical result differs')
    elif actual!=expected:raise ValueError(description+' value differs')


def validate(cfg,manifest,name,specs=None,require_complete=True):
    """Check actual arrays/tables against source populations, counts and fixed W."""
    from .coverage import exact_keys,comparison_rows,paired_bt_coverage
    from .comparisons import pooled_interval
    specs=specs_for(cfg,manifest,name) if specs is None else specs;pairs=evaluation_pairs(name)
    out=Path(specs[name]['evaluation'])/'paired';expected=[comparison_id(a,b) for a,b in pairs]
    if require_complete:
        rec=check_receipt(out/'complete.json',dependencies(cfg,'EVALUATION_PAIRS_'+name))
        if rec.get('endpoint')!=name or rec.get('pairs')!=expected or rec.get('direction')!='candidate minus control':raise ValueError('endpoint paired completion identity differs')
    for part,field in (('populations','sets'),('comparisons','jobs'),('bt','sets')):
        rec=check_receipt(out/part/'complete.json')
        if rec.get(field)!=expected:raise ValueError('full evaluation parent/BASE '+part+' coverage differs')
    fixed=pd.read_csv(manifest['fixed_population']).cell_id.to_numpy(np.int64)
    tables={m:populations.endpoint_scores(s,fixed) for m,s in specs.items()}
    frames={m:bt.endpoint_frame(cfg,manifest,s) for m,s in specs.items()}
    frozen=Path(cfg['run_root'])/'frozen_domain';gf=np.load(frozen/'cell_groups.npz');groups={k:gf[k] for k in gf.files}
    reference=bt.checked_ids(pd.read_csv(frozen/'cells.csv'),'paired fixed nuclear reference');target=frozen/'bt_targets'
    check_receipt(target/'complete.json');records=[]
    for candidate,control in pairs:
        key=comparison_id(candidate,control);folder=out/'populations'/key
        check_receipt(folder/'complete.json');declared=pd.DataFrame(read(folder/'populations.json'))
        exact_keys(declared,('metric',),[(m,) for m in METRICS],'paired finite population records')
        hashes={}
        for metric in METRICS:
            initial,selected,excl,qexcl=populations.common_finite(tables,(candidate,control),fixed,metric)
            z=np.load(folder/(metric+'.npz'));correct=dict(ids=selected,**excl)
            if set(z.files)!=set(correct) or any(not np.array_equal(z[k],v) for k,v in correct.items()):raise ValueError('paired actual finite IDs/exclusions differ')
            qc=np.load(folder/'QC_exclusions.npz')
            if set(qc.files)!=set(qexcl) or any(not np.array_equal(qc[k],v) for k,v in qexcl.items()):raise ValueError('paired actual QC exclusions differ')
            row=declared.set_index('metric').loc[metric]
            if row.n_initial!=len(initial) or row.n_finite!=len(selected) or row.n_excluded!=len(initial)-len(selected) or row.ids_sha256!=id_hash(selected) or row.key!=key or row.population!='pairwise':raise ValueError('paired finite population count/hash differs')
            if not len(selected) and row.NA_reason!='no_common_finite_IDs':raise ValueError('empty paired score population lacks reason')
            for profile in ('sc','fixed'):
                hashes[(metric,profile)]=id_hash(selected)
                records.append(dict(endpoint=name,candidate=candidate,control=control,profile=profile,metric=metric,population='four_way_finite',n_initial=len(initial),n_finite=len(selected),ids_sha256=id_hash(selected),NA_reason='' if len(selected) else 'no_common_finite_IDs'))
        check_receipt(out/'comparisons'/key/'complete.json')
        comparison_rows(pd.read_csv(out/'comparisons'/key/'stratified_intervals.csv.gz'),groups,{candidate:1,control:-1},'pairwise',hashes)
        # B/T intentionally uses commonQC without an F1-finite restriction.
        q=np.asarray(sorted(set(fixed)&set(tables[candidate]['qc_ids'])&set(tables[control]['qc_ids'])),np.int64)
        paired_bt_coverage(out/'bt'/key,key,(candidate,control),True,frozen,q)
        for m in (candidate,control):
            frame=frames[m];curves=pd.read_csv(out/'bt'/key/m/'full_curves.csv')
            for panel in bt.PANELS:
                direction=panel['direction'];dist=bt.checked_ids(pd.read_csv(target/(direction+'_frozen_distances.csv')),'paired frozen distances')
                rids=np.intersect1d(q,pd.read_csv(target/(direction+'_recipients.csv')).cell_id);tids=pd.read_csv(target/(direction+'_targets.csv')).cell_id.to_numpy(np.int64)
                for role,pool in (('recipients',rids),('targets',tids)):
                    if not np.array_equal(pd.read_csv(out/'bt'/key/m/(direction+'_'+role+'.csv')).cell_id,pool):raise ValueError('paired actual B/T pool differs')
                expected_curves,_=bt.curve(reference,rids,tids,frame,panel,dist)
                observed=curves[curves.direction==direction].set_index(['marker','bin_index'])
                for _,r in expected_curves.iterrows():
                    actual=observed.loc[(r.marker,r.bin_index)]
                    for field in ('n_eligible','n_positive','fraction','status','lower_um','upper_um','recipient_ids_sha256','target_ids_sha256'):_same(actual[field],r[field],'paired actual B/T curve '+field)
        ci=pd.read_csv(out/'bt'/key/'paired_0_40um.csv').set_index(['direction','marker','scale_um'])
        for panel in bt.PANELS:
            dist=bt.checked_ids(pd.read_csv(target/(panel['direction']+'_frozen_distances.csv')),'paired frozen distances');ids=np.intersect1d(q,dist.index[dist.distance_um.le(40)])
            for marker in panel['markers']:
                values=(frames[candidate].loc[ids,marker].to_numpy()>0).astype(float)-(frames[control].loc[ids,marker].to_numpy()>0).astype(float)
                for side in SCALES:
                    W=np.load(frozen/f'bootstrap_weights_{side}.npz')['W'];bi=reference.loc[ids,'block_'+str(side)].to_numpy(int)
                    result=pooled_interval(np.bincount(bi,weights=values,minlength=W.shape[1]),np.bincount(bi,minlength=W.shape[1]),W)
                    actual=ci.loc[(panel['direction'],marker,side)]
                    for field,value in result.items():_same(actual[field] if pd.notna(actual[field]) else None,value,'paired actual B/T CI '+field)
    return dict(endpoint=name,pairs=expected,direction='candidate minus control',score_records=records,six_markers=True,full_seven_bins=True,paired_window=[0,40],scales=list(SCALES),validated_actual_tables=True)


def reuse_sources(cfg,methods,part,initial=None):
    """Prefer already verified INITIAL, otherwise endpoint evaluation producers."""
    if part not in ('populations','comparisons','bt'):raise ValueError('unsupported endpoint paired reuse part')
    result={}
    for name in NEW:
        if name not in methods:continue
        for a,b in evaluation_pairs(name):
            if b not in methods:continue
            key=comparison_id(a,b);source=Path(new_spec(cfg,name)['evaluation'])/'paired'/part/key
            if (source/'complete.json').exists():result[key]=source
    if initial is not None:
        for source in (Path(initial)/part).iterdir():
            if source.is_dir() and (source/'complete.json').exists():result[source.name]=source
    return result
