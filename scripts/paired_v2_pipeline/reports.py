"""Complete three-endpoint computation; leaves publication to manual AI resume."""
from collections import Counter
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from scripts.new_methods.common import json_safe
from scripts.crgf.common import atomic_npy
from .common import *
from .freeze import validate_inputs,validate_counts,stream_coordinates
from .math import finite_mean,spatial,spatial_flags,guard_values,decide,EPS


def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list,np.ndarray)):return [clean(v) for v in x]
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x


def write_json(path,x):atomic_json(path,clean(x))


def aligned(path,ids):
    d=pd.read_csv(path)
    if d.cell_id.duplicated().any() or not set(d.cell_id)<=set(ids):raise ValueError('duplicate or foreign population IDs: '+str(path))
    return d.set_index('cell_id').reindex(ids)


def load_endpoint(arm,ids,q):
    e=endpoint(arm);machine=read(e['machine']);matrix=read(e['matrix']/'complete.json')
    counts=np.load(e['matrix']/'counts.npy',mmap_mode='r');mapping=pd.read_csv(e['matrix']/'cell_mapping.csv')
    if counts.shape!=(107955,313) or not np.issubdtype(counts.dtype,np.integer) or (counts<0).any() or not np.array_equal(mapping.canonical_cell_id,ids):
        raise ValueError('full own integer matrix identity or dimensions differ')
    if matrix['totals']['eligible']!=34442716 or int(counts.sum())!=matrix['totals']['assigned']:raise ValueError('matrix molecule conservation differs')
    verify_outputs(matrix);totals=counts.sum(1);ng=(counts>0).sum(1)
    qc=aligned(e['evaluation']/'paper_aligned_v1/qc_all_cells.csv',ids)
    expected=(totals>=20)&(totals<=2000)&(ng>=20)
    if not np.array_equal(qc.qc_pass,expected):raise ValueError('own QC differs from integer matrix')
    ann=aligned(e['evaluation']/'paper_aligned_v1/annotation.csv',ids)
    scores=aligned(e['evaluation']/'paper_aligned_v1/markers_per_cell.csv',ids)
    fixed=aligned(e['evaluation']/'fixed/fixed_population.csv',ids)
    if not np.array_equal(ann.evaluation_label.notna(),expected) or not np.array_equal(scores.positive_F1.notna(),expected):raise ValueError('own sc annotation/primary population differs')
    if not set(METRICS)<=set(scores) or not set(METRICS)<=set(fixed):raise ValueError('raw six-metric profiles missing')
    mask=tifffile.imread(machine['final_mask']);nuc=tifffile.imread(machine['nuclei_mask'])
    if not np.array_equal(mask[nuc>0],nuc[nuc>0]) or not set(np.unique(mask))<=set(np.r_[0,ids]):raise ValueError('nuclear restoration or new-ID rule failed')
    area=np.bincount(mask.ravel().astype(np.int64),minlength=int(ids.max())+1)[ids]*machine['pixel_um']**2
    structural=aligned(e['evaluation']/'engineering/structure/cells_distribution.csv',ids)
    if not np.array_equal(structural.area_um2,area):raise ValueError('area report does not match own native mask')
    c8=aligned(e['evaluation']/'engineering/structure/components_8.csv',ids).n_components_8.to_numpy(int)
    c4=np.empty(len(ids),int);boxes=ndi.find_objects(mask)
    for i,cid in enumerate(ids):
        box=boxes[int(cid)-1]
        if box is None:raise ValueError('original nucleus ID vanished')
        c4[i]=ndi.label(mask[box]==cid,structure=ndi.generate_binary_structure(2,1))[1]
    if (c4<c8).any():raise ValueError('4/8 adjacency component ordering invalid')
    frame=pd.DataFrame(dict(cell_id=ids,method=arm,qc_pass=expected,qc_reason=qc.qc_reason.to_numpy(),
        total_transcripts=totals,total_genes=ng,area_um2=area,components_8=c8,components_4=c4,
        sc_label=ann.evaluation_label.fillna('not_QC').to_numpy(),raw_label=ann.raw_label.to_numpy(),
        rejected=ann.rejected.to_numpy(),unmatched_reason=ann.unmatched_reason.to_numpy(),
        fixed_label=fixed.anchor_label.to_numpy(),fixed_availability=fixed.label_availability.to_numpy()))
    for metric in METRICS:
        frame['sc_'+metric]=scores[metric].to_numpy(float);frame['sc_finite_'+metric]=np.isfinite(frame['sc_'+metric])
        frame['fixed_'+metric]=fixed[metric].to_numpy(float);frame['fixed_finite_'+metric]=np.isfinite(frame['fixed_'+metric])
    qc_n=int(expected.sum());unassigned=(frame.sc_label=='unassigned')&expected
    aggregate=dict(capture=matrix['totals']['assigned']/34442716,area_mean=float(area.mean()),area_median=float(np.median(area)),
        area_p95=float(np.quantile(area,.95,method='linear')),qc_retention=qc_n/len(ids),
        unassigned=float(unassigned.sum()/qc_n) if qc_n else None,disconnected8=int((c8>1).sum()))
    row=dict(method=arm,experiment_group='PAIRED_V2_20261014',n_segmented_cells=len(ids),capture_pct=100*aggregate['capture'],
        mean_transcripts_per_cell=float(totals.mean()),mean_genes_per_cell=float(ng.mean()),mean_cell_area_um2=aggregate['area_mean'],
        median_cell_area_um2=aggregate['area_median'],p95_cell_area_um2=aggregate['area_p95'],disconnected_cells_8=aggregate['disconnected8'],
        disconnected_cells_4=int((c4>1).sum()),scclassify_qc_cells=qc_n,scclassify_qc_retention_pct=100*aggregate['qc_retention'],
        scclassify_unassigned_pct=None if aggregate['unassigned'] is None else 100*aggregate['unassigned'],
        fixed_population_cells=int(fixed.anchor_label.notna().sum()),geometry_source='own_final_native_mask',
        matrix_path=str(e['matrix']),matrix_sha256=sha256(e['matrix']/'counts.npy'),evaluation_path=str(e['evaluation']),
        evaluation_sha256=sha256(e['evaluation']/'complete.json'),checkpoint_sha256=sha256(e['root']/'checkpoints/final.pth'),
        nuclear_preservation_fraction=1.0,new_IDs=0)
    for prefix,columns in [('scclassify','sc'),('fixed','fixed')]:
        for metric in METRICS:
            values=frame[f'{columns}_{metric}'].to_numpy()
            if columns=='sc':values=values[expected]
            stats=finite_mean(values);row[f'{prefix}_{metric}']=stats['mean'];row[f'{prefix}_{metric}_n']=stats['n_finite']
    cor=pd.read_csv(e['evaluation']/'paper_aligned_v1/type_correlations.csv')
    for label,definition in [('expression','mean_logcounts'),('detection','detection')]:
        values=cor.loc[cor.definition==definition,'pearson'].to_numpy();s=finite_mean(values)
        row[f'scclassify_{label}_pearson_type_mean']=s['mean'];row[f'scclassify_{label}_pearson_types_n']=s['n_finite']
    row['evaluation_wall_seconds']=read(e['evaluation']/'summary.json').get('wall_seconds')
    return frame,aggregate,row


def spatial_diversity(q,out,arms=ENDPOINTS):
    work=out/'author_spatial';(work/'sources').mkdir(parents=True,exist_ok=True)
    authority=read(q['bindings']['authority']['path'])
    source=Path(authority['spatial_diversity']['source'])
    shutil.copyfile(source,work/'sources/publish_spatialDiversity_metrics.R')
    lib=work/'sources/R_library'
    if not lib.exists():lib.symlink_to(Path(source).parent/'R_library',target_is_directory=True)
    atomic_json(work/'protocol.json',dict(methods=list(arms)))
    for arm in arms:
        e=endpoint(arm);d=work/'methods'/arm;(d/'paper').mkdir(parents=True,exist_ok=True)
        for name in ('annotation.csv','qc_all_cells.csv'):
            shutil.copyfile(e['evaluation']/'paper_aligned_v1'/name,d/'paper'/name)
        atomic_json(d/'input.json',dict(annotation=str(e['evaluation']/'paper_aligned_v1/annotation.csv')))
        atomic_json(d/'scoring_complete.json',dict(status='identity_verified_own_QC_inputs'))
    with (work/'execution.log').open('w') as log:
        subprocess.run([RSCRIPT,q['bindings']['spatial_wrapper']['path'],str(work)],env=numeric_env(),stdout=log,stderr=subprocess.STDOUT,check=True)
    results=pd.read_csv(work/'spatial_diversity_all_methods.csv')
    if set(results.method)!=set(arms) or len(results)!=6*len(arms):raise ValueError('author spatial diversity outputs incomplete')
    return results


def local_summary(frame,mask):
    f=frame.loc[mask];qc=f.qc_pass.to_numpy(bool);n=len(f);qn=int(qc.sum())
    return dict(n_ids=n,n_qc=qn,rna=int(f.total_transcripts.sum()),capture=None,
        area_mean=float(f.area_um2.mean()) if n else None,area_median=float(f.area_um2.median()) if n else None,
        area_p95=float(f.area_um2.quantile(.95,interpolation='linear')) if n else None,
        qc_retention=qn/n if n else None,unassigned=int((f.sc_label=='unassigned').sum())/qn if qn else None,
        disconnected8=int((f.components_8>1).sum()) if n else None,
        **{m:finite_mean(f['sc_'+m].to_numpy()) for m in METRICS})


def strata_reports(q,out,pop,frames):
    rows=[];comparisons=[];localized={key:[] for key in COMPARISONS}
    families={'original_type':pop.original_type,'density_quartile':pop.density_quartile,'count_quartile':pop.count_quartile,
              'block_500':pop.block_500,'block_1000':pop.block_1000,'block_2000':pop.block_2000}
    for family,values in families.items():
        # Include all prescribed original types even if empty; zero denominators remain explicit.
        levels=sorted(values.unique(),key=str)
        if family=='original_type':levels=read(Path(q['root'])/'frozen_strata/strata.json')['original_type_names']+['Unknown']
        if family=='density_quartile':levels=list(range(len(read(Path(q['root'])/'frozen_strata/strata.json')['density_edges'])+1))
        if family=='count_quartile':levels=list(range(len(read(Path(q['root'])/'frozen_counts/strata.json')['edges'])+1))
        for level in levels:
            mask=(values==level).to_numpy();summaries={arm:local_summary(frames[arm],mask) for arm in ENDPOINTS}
            for arm,s in summaries.items():
                rows.append(dict(family=family,group=level,method=arm,**{k:v for k,v in s.items() if k not in METRICS},
                    **{m:s[m]['mean'] for m in METRICS},**{m+'_n':s[m]['n_finite'] for m in METRICS},
                    status='computed' if int(mask.sum()) else 'not_estimable',reason=None if mask.any() else 'empty fixed-ID group'))
            for key,(candidate,control) in COMPARISONS.items():
                a,b=summaries[candidate],summaries[control];guards=guard_values(a,b);guards.pop('capture')
                retention=a['rna']/b['rna'] if b['rna']>0 else None
                violations=[k for k,v in guards.items() if v['passed'] is False]
                if retention is not None and retention<.995:violations.append('fixed_ID_RNA_retention')
                if violations:localized[key].append(dict(family=family,group=level,violations=violations,n_ids=int(mask.sum())))
                aa=frames[candidate].loc[mask];bb=frames[control].loc[mask]
                aq=aa.qc_pass.to_numpy(bool);bq=bb.qc_pass.to_numpy(bool)
                comparisons.append(dict(comparison=key,family=family,group=level,n_ids=int(mask.sum()),candidate=candidate,control=control,
                    RNA_retention=retention,RNA_retention_status='computed' if retention is not None else 'not_estimable',
                    RNA_retention_reason=None if retention is not None else 'zero control counts',candidate_RNA=a['rna'],control_RNA=b['rna'],
                    capture_status='not_estimable',capture_reason='fixed cell-ID group has no spatial molecular denominator; RNA retention used',
                    qc_both=int((aq&bq).sum()),qc_entered=int((aq&~bq).sum()),qc_exited=int((~aq&bq).sum()),qc_neither=int((~aq&~bq).sum()),
                    guard_values=guards,violations=violations))
    pd.DataFrame(rows).to_csv(out/'strata_endpoint_values.csv',index=False)
    write_json(out/'strata_comparisons.json',comparisons)
    return localized


def owner_reports(q,out,pop):
    domain=np.load(Path(q['root'])/'frozen_strata/domains.npy',mmap_mode='r')
    single=np.load(Path(q['root'])/'frozen_strata/single_unknown.npy',mmap_mode='r')
    spec=read(Path(q['root'])/'frozen_strata/strata.json')
    den=np.array(spec['domain_molecule_denominators']);assigned={a:np.zeros(7,np.int64) for a in ENDPOINTS}
    single_assigned={a:0 for a in ENDPOINTS};block_assigned={a:{str(s):Counter() for s in (500,1000,2000)} for a in ENDPOINTS}
    matrices={a:read(endpoint(a)['matrix']/'complete.json') for a in ENDPOINTS}
    types=np.full(int(pop.cell_id.max())+1,-2,int);types[pop.cell_id]=pop.original_type_index
    sparse={key:Counter() for key in COMPARISONS};category={key:Counter() for key in COMPARISONS}
    for index,(reference,coord) in enumerate(stream_coordinates(q)):
        valid=(coord['x']>=0)&(coord['y']>=0);d=np.full(len(coord),6,np.uint8);d[valid]=domain[coord['y'][valid],coord['x'][valid]]
        us=np.zeros(len(coord),bool);us[valid]=single[coord['y'][valid],coord['x'][valid]];owners={}
        for arm in ENDPOINTS:
            rec=matrices[arm]['receipts'][index];e=endpoint(arm);p=e['matrix']/rec['owner_file']
            if sha256(p)!=rec['sha256']:raise ValueError('actual owner chunk changed')
            cp=np.load(rec['coordinate_file'],mmap_mode='r')
            if not np.array_equal(cp,coord):raise ValueError('cross-endpoint transcript/gene/native coordinate association differs')
            owner=np.load(p);owners[arm]=owner
            if len(owner)!=len(coord):raise ValueError('owner row coverage mismatch')
            assigned[arm]+=np.bincount(d[owner>0],minlength=7);single_assigned[arm]+=int(((owner>0)&us).sum())
            for side in (500,1000,2000):
                keep=valid&(owner>0);xy=np.stack((coord['x'][keep]//side,coord['y'][keep]//side),1)
                pairs,nn=np.unique(xy,axis=0,return_counts=True)
                for pair,n in zip(pairs,nn):block_assigned[arm][str(side)][f'{pair[0]},{pair[1]}']+=int(n)
        for key,(candidate,control) in COMPARISONS.items():
            a,b=owners[candidate],owners[control];pairs=np.stack((d,b,a),1).astype(np.int64)
            uniq,nn=np.unique(pairs,axis=0,return_counts=True)
            for row,n in zip(uniq,nn):sparse[key][tuple(row)]+=int(n)
            changed=a!=b
            same=changed&(a>0)&(b>0)&(types[a]>=0)&(types[b]>=0)&(types[a]==types[b])
            different=changed&(a>0)&(b>0)&(types[a]>=0)&(types[b]>=0)&(types[a]!=types[b])
            unk=changed&(a>0)&(b>0)&((types[a]==-1)|(types[b]==-1))
            masks={'unchanged_background':(a==0)&(b==0),'unchanged_cell':(a==b)&(a>0),
                'background_to_cell':(b==0)&(a>0),'cell_to_background':(b>0)&(a==0),
                'cell_to_cell_same_known_type':same,'cell_to_cell_different_known_type':different,'cell_to_cell_involving_Unknown':unk}
            if sum(int(m.sum()) for m in masks.values())!=len(a):raise ValueError('owner transfer categories not exhaustive')
            for label,m in masks.items():
                for dd,n in enumerate(np.bincount(d[m],minlength=7)):category[key][(dd,label)]+=int(n)
        print('owner comparison chunk',index,flush=True)
    rows=[];localized={key:[] for key in COMPARISONS}
    for key,(candidate,control) in COMPARISONS.items():
        pd.DataFrame([dict(domain=DOMAINS[d],control_owner=int(b),candidate_owner=int(a),molecules=int(n)) for (d,b,a),n in sorted(sparse[key].items())]).to_csv(out/(key+'_owner_ID_transfers.csv'),index=False)
        pd.DataFrame([dict(domain=DOMAINS[d],category=label,molecules=int(n)) for (d,label),n in sorted(category[key].items())]).to_csv(out/(key+'_owner_categories.csv'),index=False)
        for d,name in enumerate(DOMAINS):
            n=int(den[d]);av=int(assigned[candidate][d]);bv=int(assigned[control][d]);delta=(av-bv)/n if n else None
            rows.append(dict(comparison=key,domain=name,eligible=n,candidate_assigned=av,control_assigned=bv,
                candidate_capture=av/n if n else None,control_capture=bv/n if n else None,capture_delta_pp=100*delta if n else None,
                status='computed' if n else 'not_estimable',reason=None if n else 'zero native eligible molecules'))
            if delta is not None and delta<-.005:localized[key].append(dict(domain=name,capture_delta_pp=100*delta,eligible=n))
        n=spec['single_unknown_molecule_denominator'];a=single_assigned[candidate];b=single_assigned[control]
        rows.append(dict(comparison=key,domain='single_candidate_Unknown_subflag',eligible=n,candidate_assigned=a,control_assigned=b,
            candidate_capture=a/n if n else None,control_capture=b/n if n else None,capture_delta_pp=100*(a-b)/n if n else None,
            status='computed' if n else 'not_estimable',reason=None if n else 'zero eligible molecules'))
        if n and (a-b)/n<-.005:localized[key].append(dict(domain='single_candidate_Unknown_subflag',capture_delta_pp=100*(a-b)/n,eligible=n))
    pd.DataFrame(rows).to_csv(out/'domain_capture.csv',index=False)
    block_rows=[]
    for key,(candidate,control) in COMPARISONS.items():
        for side in (500,1000,2000):
            for block,n in spec['block_molecule_denominators'][str(side)].items():
                a=block_assigned[candidate][str(side)][block];b=block_assigned[control][str(side)][block]
                delta=(a-b)/n
                block_rows.append(dict(comparison=key,side_um=side,block=block,eligible=n,candidate_assigned=a,control_assigned=b,capture_delta_pp=100*delta))
                if delta<-.005:localized[key].append(dict(side_um=side,block=block,capture_delta_pp=100*delta,eligible=n))
    pd.DataFrame(block_rows).to_csv(out/'block_capture.csv',index=False)
    for arm in ENDPOINTS:
        if int(assigned[arm].sum())!=matrices[arm]['totals']['assigned']:raise ValueError('domain owner aggregation not conserved')
    return localized


def report_dependencies(q):
    return dict(source_bundle=q['source_bundle'],pipeline_config=identity(Path(q['root'])/'configs/pipeline.json'),
        frozen_inputs=identity(Path(q['root'])/'frozen_strata/complete.json'),frozen_counts=identity(Path(q['root'])/'frozen_counts/complete.json'),
        paired_training=identity(Path(q['root'])/'paired_training/complete.json'),
        endpoints={a:{s:identity(endpoint(a)[s]/'complete.json') for s in ('matrix','evaluation')} for a in ENDPOINTS},
        historical_review=q['bindings']['implementation_review'])


def paired_reports(q,out,pop,frames,aggregates,local_flags):
    deltas=[];decisions={};common_rows=[];qc_rows=[];migrations=[];spatial_summary=[]
    bootstrap_dir=out/'spatial';bootstrap_dir.mkdir(exist_ok=True)
    for key,(candidate,control) in COMPARISONS.items():
        a,b=frames[candidate],frames[control];aq=a.qc_pass.to_numpy(bool);bq=b.qc_pass.to_numpy(bool)
        statuses=np.select([aq&bq,aq&~bq,~aq&bq],['both','entered_candidate','exited_candidate'],default='neither')
        pd.DataFrame(dict(cell_id=pop.cell_id,qc_state=statuses,candidate_label=a.sc_label,control_label=b.sc_label)).to_csv(out/(key+'_QC_population_and_labels.csv'),index=False)
        for state in ('both','entered_candidate','exited_candidate','neither'):
            qc_rows.append(dict(comparison=key,state=state,n_ids=int((statuses==state).sum())))
        labels=pd.DataFrame(dict(control_label=b.sc_label,candidate_label=a.sc_label,qc_state=statuses)).value_counts().reset_index(name='n_ids')
        labels.insert(0,'comparison',key);migrations.extend(labels.to_dict('records'))
        pair={};flags=[]
        for metric in METRICS:
            av=a['sc_'+metric].to_numpy();bv=b['sc_'+metric].to_numpy()
            am=finite_mean(av[aq]);bm=finite_mean(bv[bq]);delta=None if am['mean'] is None or bm['mean'] is None else am['mean']-bm['mean'];pair[metric]=delta
            deltas.append(dict(comparison=key,candidate=candidate,control=control,metric=metric,profile='own_QC_scClassify',candidate_mean=am['mean'],control_mean=bm['mean'],delta=delta,candidate_n=am['n_finite'],control_n=bm['n_finite']))
            common=aq&bq&np.isfinite(av)&np.isfinite(bv)
            for name,mask in [('sc_only_common_QC_finite',common),('sc_common_conditioned_on_fixed_finite_auxiliary',common&np.isfinite(a['fixed_'+metric])&np.isfinite(b['fixed_'+metric]))]:
                n=int(mask.sum());cv=float((av[mask]-bv[mask]).mean()) if n else None
                common_rows.append(dict(comparison=key,profile=name,metric=metric,n_ids=n,delta=cv,
                    candidate_mean=float(av[mask].mean()) if n else None,control_mean=float(bv[mask].mean()) if n else None,
                    status='computed' if n else 'not_estimable',reason=None if n else 'no common finite/QC IDs'))
                if name=='sc_only_common_QC_finite' and n and ((metric=='positive_F1' and cv<-EPS) or (metric=='negative_F1' and cv>EPS)):
                    flags.append('population_sensitive')
            # Fixed labels stay auxiliary and never affect flags, ranking or gates.
            fa=a['fixed_'+metric].to_numpy();fb=b['fixed_'+metric].to_numpy();af=finite_mean(fa);bf=finite_mean(fb)
            deltas.append(dict(comparison=key,candidate=candidate,control=control,metric=metric,profile='fixed_auxiliary',candidate_mean=af['mean'],control_mean=bf['mean'],delta=None if af['mean'] is None or bf['mean'] is None else af['mean']-bf['mean'],candidate_n=af['n_finite'],control_n=bf['n_finite']))
        scale_results={}
        for side in (500,1000,2000):
            blocks=pop[f'block_{side}'].to_numpy(int);nb=read(Path(q['root'])/'frozen_strata/strata.json')['grids'][str(side)]['n_blocks']
            vals_a={m:a['sc_'+m].to_numpy() for m in ('positive_F1','negative_F1')}
            vals_b={m:b['sc_'+m].to_numpy() for m in ('positive_F1','negative_F1')}
            result,draws=spatial(vals_a,vals_b,blocks,nb)
            atomic_npy(bootstrap_dir/f'{key}_{side}_paired_draws.npy',draws)
            scale_flags=spatial_flags(result,pair['positive_F1'] or 0,pair['negative_F1'] or 0,nb)
            scale_results[str(side)]=dict(n_blocks=nb,flags=scale_flags)
            for metric,r in result.items():
                atomic_npy(bootstrap_dir/f'{key}_{side}_{metric}_bootstrap.npy',r['bootstrap'])
                pd.DataFrame(dict(block=np.arange(nb),sum_candidate=r['block_sums_candidate'],sum_control=r['block_sums_control'],n_candidate=r['block_n_candidate'],n_control=r['block_n_control'],leave_one_out=r['leave_one_out'])).to_csv(bootstrap_dir/f'{key}_{side}_{metric}_blocks_LOBO.csv',index=False)
                spatial_summary.append(dict(comparison=key,side_um=side,metric=metric,n_blocks=nb,n_valid=r['n_valid'],n_invalid=r['n_invalid'],ci_lower=r['ci'][0],ci_upper=r['ci'][1],flags=scale_flags,seed=20261013,repetitions=2000,RNG='NumPy PCG64',statistic='difference of pooled sums/finite counts; occupied original-nucleus blocks including n=0'))
            if side==1000:flags.extend(scale_flags)
            elif scale_flags:flags.append('scale_sensitive')
        if local_flags[key]:flags.append('localized_adverse')
        guards=guard_values(aggregates[candidate],aggregates[control])
        decision=decide(pair['positive_F1'],pair['negative_F1'],guards,flags)
        decision.update(candidate=candidate,control=control,global_denominators=dict(original_ids=107955,eligible_molecules=34442716,
            candidate_QC=int(aq.sum()),control_QC=int(bq.sum())),localized_adverse_groups=local_flags[key],spatial_scales=scale_results,
            fixed_role='auxiliary only; never rank/veto/flags',scope='single seed on developed Rep1; spatial intervals describe this field, not seeds/specimens or owner accuracy')
        decisions[key]=decision
    # Also supply a genuinely common three-endpoint table, both sc-only and fixed-conditioned.
    for metric in METRICS:
        mask=np.ones(len(pop),bool)
        for a in ENDPOINTS:mask &= frames[a].qc_pass.to_numpy(bool)&np.isfinite(frames[a]['sc_'+metric])
        for name,condition in [('three_endpoint_sc_only_common',mask),('three_endpoint_fixed_conditioned_auxiliary',mask & np.logical_and.reduce([np.isfinite(frames[a]['fixed_'+metric]) for a in ENDPOINTS]))]:
            for a in ENDPOINTS:
                stats=finite_mean(frames[a].loc[condition,'sc_'+metric].to_numpy())
                common_rows.append(dict(comparison='all_three',profile=name,method=a,metric=metric,n_ids=int(condition.sum()),candidate_mean=stats['mean'],status=stats['status'],reason=stats['reason']))
    pd.DataFrame(deltas).to_csv(out/'paired_deltas.csv',index=False);pd.DataFrame(common_rows).to_csv(out/'common_population_deltas.csv',index=False)
    pd.DataFrame(qc_rows).to_csv(out/'QC_entry_exit_counts.csv',index=False);pd.DataFrame(migrations).to_csv(out/'label_migrations.csv',index=False)
    write_json(out/'spatial_summary.json',spatial_summary)
    write_json(out/'decision.json',dict(status='computational_rules_applied_awaiting_manual_AI_analysis',comparisons=decisions,
        expand_old_v2=decisions['old_morph_increment']['status']=='priority_signal',
        expand_interaction=all(decisions[k]['status']=='priority_signal' for k in ('interaction_increment','package_adoption')),
        correctness='passed_runtime_acceptance',claim_scope='single seed Rep1; neither bootstrap repetitions nor cell numbers are independent biological or seed replicates'))


def purity(table):
    spec=dict(comparison_set=table.method.tolist(),formula='2*p*(1-n)/(p+1-n)',role='auxiliary; never decision/rank/veto')
    for profile in ('scclassify','fixed'):
        endpoints={};scaled=[]
        for metric in ('positive_F1','negative_F1'):
            v=table[f'{profile}_{metric}'].to_numpy(float);ok=np.isfinite(v)
            lo=float(v[ok].min()) if ok.any() else None;hi=float(v[ok].max()) if ok.any() else None
            endpoints[metric]=dict(min=lo,max=hi,status='computed' if lo is not None and hi>lo else 'not_estimable',reason=None if lo is not None and hi>lo else 'degenerate/no finite scaling endpoints')
            scaled.append((v-lo)/(hi-lo) if lo is not None and hi>lo else np.full(len(v),np.nan))
        p,n=scaled;den=p+1-n;v=np.full(len(p),np.nan);ok=np.isfinite(den)&(den>0);v[ok]=2*p[ok]*(1-n[ok])/den[ok]
        v[np.isfinite(den)&(den==0)]=0.0  # Protocol's harmonic-mean zero limit, not missing-value imputation.
        table[profile+'_Purity_F1']=v;spec[profile]=endpoints
    spec['zero_policy']='finite scaled p=0 and clean=0 -> harmonic mean 0; missing/degenerate endpoints remain NA'
    return spec


def reports(q):
    out=Path(q['root'])/'results';out.mkdir(exist_ok=True)
    if (out/'complete.json').exists():return validate_reports(q)
    validate_inputs(q);validate_counts(q);verify_history(q)
    from scripts.paired_v2.stages import validate_stage
    for arm in RUNS:
        for stage in ('train','predict','postprocess','matrix','evaluate'):validate_stage(config_path(arm),stage,require_bound=True)
    pop=pd.read_csv(Path(q['root'])/'frozen_strata/fixed_cell_groups.csv').merge(pd.read_csv(Path(q['root'])/'frozen_counts/fixed_count_groups.csv'),on='cell_id',validate='one_to_one')
    ids=pop.cell_id.to_numpy();frames={};aggregates={};rows=[]
    for arm in ENDPOINTS:
        frames[arm],aggregates[arm],row=load_endpoint(arm,ids,q);rows.append(row)
        frames[arm].to_csv(out/(arm+'_all_ID_population.csv'),index=False)
        # Keep all original author scores, correlation vectors, centroid and neighbor curve tables inspectable.
        e=endpoint(arm);dest=out/'endpoint_evaluation'/arm;dest.mkdir(parents=True,exist_ok=True)
        for folder in ('paper_aligned_v1','fixed'):
            for p in (e['evaluation']/folder).glob('*.csv'):
                target=dest/folder/p.name;target.parent.mkdir(exist_ok=True);shutil.copyfile(p,target)
    diversity=spatial_diversity(q,out)
    for row in rows:
        subset=diversity[diversity.method==row['method']]
        for _,r in subset.iterrows():row['scclassify_spatial_entropy_vs_'+r.metric]=r.pearson
        row['scclassify_spatial_bins_n']=int(subset.n_bins.iloc[0])
    table=pd.DataFrame(rows);write_json(out/'purity_scaling_three_endpoint.json',purity(table));table.to_csv(out/'main_table.csv',index=False)
    # Preserve the unmodified 18-method values and cohort separately from the new three-endpoint display.
    shutil.copyfile(q['bindings']['historical_main_table']['path'],out/'historical_main_table18.csv')
    local=strata_reports(q,out,pop,frames);owner_local=owner_reports(q,out,pop)
    for key in local:local[key].extend(owner_local[key])
    paired_reports(q,out,pop,frames,aggregates,local)
    cost=[]
    for task in q['tasks']:
        p=own_receipt(q,task['id'])
        if p.exists():
            rec=read(p);cost.append(dict(task=task['id'],attempts=rec.get('attempts',[]),origin='new paired execution'))
    write_json(out/'stage_cost.json',dict(stages=cost,historical_E1_scope='reused historical cost; not timed again or an exact speed comparison',
        historical_stage_cost=identity(INPUT/'final_analysis/stage_cost.csv'),resource_pilot=q['bindings']['resource_pilot']))
    write_json(out/'scope_and_missingness.json',dict(scope='single seed, previously developed Rep1; no owner accuracy or generalized improvement claim',
        population_files='all 107955 IDs, raw six metrics and each finite flag; missing values kept as NA',
        empty_strata='not_estimable with reason; no substitution of zero; fixed labels auxiliary only',
        scientific_failures_retained=['extra whole-network FP32 1e-6 permutation gate failed','exploratory whole-network FP64 harness not_verified'],
        original_reviews=q['bindings']['implementation_review'],official_publication='awaiting manual AI resume'))
    write_json(out/'complete.json',dict(status='succeeded',dependencies=report_dependencies(q),outputs=inventory(out)))
    return validate_reports(q)


def validate_reports(q):
    out=Path(q['root'])/'results';rec=read(out/'complete.json')
    if rec['status']!='succeeded' or rec['dependencies']!=report_dependencies(q):raise ValueError('report dependency changed')
    verify_outputs(rec);table=pd.read_csv(out/'main_table.csv');decision=read(out/'decision.json')
    if set(table.method)!=set(ENDPOINTS) or len(table)!=3:raise ValueError('three endpoint table incomplete')
    if set(decision['comparisons'])!=set(COMPARISONS):raise ValueError('three comparisons missing')
    for arm in ENDPOINTS:
        f=pd.read_csv(out/(arm+'_all_ID_population.csv'))
        if len(f)!=107955 or f.cell_id.duplicated().any():raise ValueError('all-ID report incomplete')
        for m in METRICS:
            if not {'sc_'+m,'sc_finite_'+m,'fixed_'+m,'fixed_finite_'+m}<=set(f):raise ValueError('raw metric/finite report missing')
    for key in COMPARISONS:
        d=decision['comparisons'][key]
        if not {'delta_P','delta_N','direction_pass','guards_pass','reliability_flags','magnitude_priority','status','all_guard_values_and_denominators'}<=set(d):raise ValueError('decision contract fields incomplete')
        if d['status']!=decide(d['delta_P'],d['delta_N'],d['all_guard_values_and_denominators'],d['reliability_flags'])['status']:raise ValueError('decision precedence changed')
        transfers=pd.read_csv(out/(key+'_owner_ID_transfers.csv'))
        if int(transfers.molecules.sum())!=34442716:raise ValueError('owner transition population incomplete')
        for side in (500,1000,2000):
            draws=np.load(out/'spatial'/f'{key}_{side}_paired_draws.npy',mmap_mode='r')
            if draws.shape[0]!=2000:raise ValueError('paired bootstrap repetitions incomplete')
    required=('strata_endpoint_values.csv','strata_comparisons.json','domain_capture.csv','block_capture.csv','common_population_deltas.csv','QC_entry_exit_counts.csv','label_migrations.csv','spatial_summary.json','stage_cost.json','scope_and_missingness.json','author_spatial/spatial_diversity_all_methods.csv')
    if any(not (out/name).is_file() for name in required):raise ValueError('prespecified report absent')
    return dict(status='passed',receipt=identity(out/'complete.json'),endpoints=list(ENDPOINTS),comparisons=list(COMPARISONS),all_ids=107955)
