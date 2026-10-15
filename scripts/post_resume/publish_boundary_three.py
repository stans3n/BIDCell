"""Versioned local publication after explicit goal resume and scientific review.

No training, evaluator, thresholds, frozen inputs, or historical20 file is edited.
The requested context is backed up byte-for-byte; its historical text is retained.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import pandas as pd

WORK = Path('/home/zwan0726/work')
ROOT = WORK / 'outputs/bidcell_boundary_three_20261015'
RUN = ROOT / 'boundary-three-20261015-r02'
OLD = WORK / 'outputs/bidcell_historical_dual_20261013'
OFFICIAL = WORK / 'outputs/bidcell_historical_dual_boundary_three_20261016'
CONTEXT = WORK / 'context'
POST = ROOT / 'post_resume_20261016'
STAGE = POST / 'staging'
BACKUP = POST / 'before_publication'
REPO = Path(__file__).resolve().parents[2]
TRACKED = REPO / 'docs/boundary_three/results_20261016'
VERSION = 'DUAL-F1-v1.0-28methods-20261016'
PURITY_VERSION = 'PURITY-28methods-v1.0-20261016'
SOURCE_COMMIT = 'ccb846cbf344bfbe965178cd8f9b738e3654c8b0'
NEW = ('P2_BASE','P2_MORPH','MORPH_NO_DAPI','MORPH_SIMPLE_FUSION','MORPH_NO_GEO',
       'A2_GT','A2_GTC','A2_SHUFFLED')
METRICS = ('positive_F1','negative_F1','positive_Precision','positive_Recall',
           'negative_Precision','negative_Recall')
MUTABLE = {'scclassify_Purity_F1','fixed_Purity_F1','purity_scaling_version','purity_scaling_path'}
MARKER = 'BOUNDARY_THREE_20261016'


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()


def ident(p):
    p=Path(p)
    return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))


def js(d):
    return (json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()


def atomic(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists() and p.read_bytes()==data:return
    temp=p.with_name(p.name+'.publish_tmp')
    with temp.open('wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(temp,p)


def table(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:
        rd=csv.DictReader(f);return list(rd.fieldnames),list(rd)


def csv_bytes(columns,rows):
    f=io.StringIO(newline='');w=csv.DictWriter(f,fieldnames=columns)
    w.writeheader();w.writerows({k:r.get(k,'') for k in columns} for r in rows)
    return ('\ufeff'+f.getvalue()).encode()


def close(a,b):
    assert math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-12),(a,b)


def baseline():
    manifest=BACKUP/'manifest.json'
    if not manifest.exists():
        files=list(CONTEXT.glob('*.md'))+[CONTEXT/'experiment_registry.csv']
        files+=list(OLD.glob('*.json'))+list(OLD.glob('*.csv'))
        records={}
        for p in files:
            dest=BACKUP/p.relative_to(WORK);dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,dest);assert sha(p)==sha(dest)
            records[str(p)]=ident(dest)
        atomic(manifest,js(dict(status='verified_backup',files=records)))
    data=read(manifest)
    for p,old in data['files'].items():
        assert sha(old['path'])==old['sha256']
        if str(p).startswith(str(OLD)+'/'):assert sha(p)==old['sha256'],p
    return data


def role_record():
    return dict(implementation_roles=ident(ROOT/'handoff/model_configuration.json'),
        post_resume_independent_analyst=dict(task='/root/results_analysis',
        model='gpt-6.1-sol',reasoning_effort='xhigh',fork_turns='none',
        actual_spawn_status='accepted',new_after_resume=True,
        independent_of_implementation_and_code_audits=True),
        parent=dict(requested_model='gpt-6.1-sol',requested_reasoning_effort='xhigh',
        runtime_override_exposure='not exposed; configuration recorded in prior role receipt'),
        scientific_source_commit=SOURCE_COMMIT,results_branch='results/boundary-three-20261016')


def prepare():
    baseline()
    raw=read(ROOT/'analysis/root_review_20261016/raw_validation.json')
    assert raw['status']=='passed' and raw['A2_gene_totals_exactly_conserved']
    completion=read(RUN/'complete-check.json')
    assert completion['all_enabled_applicable_complete'] and not completion['failures'] and not completion['deferred']
    assert read(ROOT/'continuation_morph_first_20261015/state.json')['status']=='COMPLETED'
    oldcols,oldrows=table(OLD/'main_table.csv')
    assert len(oldrows)==20 and len({r['method'] for r in oldrows})==20
    _,packed=table(RUN/'reports/endpoint_table.csv'); by={}
    for row in packed:by.setdefault(row['method'],row)
    excluded={'node','execution_status','source_table','source_table_sha256','comparison_set',
              'purity_scaling_version','purity_scaling_sha256','comparison_receipt_sha256',
              'scclassify_Purity_F1','fixed_Purity_F1'}
    rows=[dict(r) for r in oldrows]
    for method in NEW:
        row={k:v for k,v in by[method].items() if k not in excluded}
        ev=raw['endpoints'][method]
        row.update(execution_status='COMPLETED',boundary_run_id=RUN.name,
            boundary_node='A2_CORE' if method.startswith('A2_') else method,
            scientific_source_commit=SOURCE_COMMIT,
            scientific_config_digest=completion['config_digest'],
            boundary_source_table=by[method]['source_table'],
            boundary_source_table_sha256=by[method]['source_table_sha256'],
            boundary_full_receipt=str(RUN/('A2_CORE' if method.startswith('A2_') else method)/'complete.json'),
            matrix_counts_path=ev['matrix']['path'],matrix_counts_sha256=ev['matrix']['sha256'],
            matrix_receipt_path=ev['matrix_receipt']['path'],matrix_receipt_sha256=ev['matrix_receipt']['sha256'],
            matrix_mapping_sha256=ev['cell_mapping']['sha256'],matrix_gene_mapping_sha256=ev['gene_mapping']['sha256'],
            evaluation_receipt_path=ev['evaluation_receipt']['path'],evaluation_receipt_sha256=ev['evaluation_receipt']['sha256'],
            annotation_path=ev['annotation']['path'],annotation_sha256=ev['annotation']['sha256'],
            canonical_artifact_registry_path=str(OFFICIAL/'artifact_registry.json'))
        if method.startswith('MORPH_'):
            row.update(model_kind=method,training_seed='1234',actual_updates='3625',view_cursor='3998',empty_skips='373',
                training_identity_scope='fresh_frozen_step_zero_variant')
        else:
            row.update(model_kind='A2_FIXED_REFINEMENT' if method.startswith('A2_') else 'P2_POSTPROCESS',
                training_identity_scope='no_new_training; verified inherited checkpoint',
                checkpoint_parent='V2_MORPH' if method=='P2_MORPH' else 'BASE_MATCH')
        rows.append(row)
    assert len(rows)==len({r['method'] for r in rows})==28
    assert 'A1_CORE' not in {r['method'] for r in rows} and 'A2_NATIVE' not in {r['method'] for r in rows}
    rowmap={r['method']:r for r in rows};methods=list(rowmap)
    # Check the B0 reference directly against the published re-scored per-cell inputs.
    b0=OLD/'methods/B0';b0checks={}
    for profile,relative in [('scclassify','paper/markers_per_cell.csv'),('fixed','fixed/fixed_population.csv')]:
        data=pd.read_csv(b0/relative).set_index('cell_id')
        assert data.index.is_unique
        for metric in METRICS:
            values=data[metric].dropna()
            close(values.mean(),rowmap['B0'][profile+'_'+metric])
            assert len(values)==int(rowmap['B0'][profile+'_'+metric+'_n'])
            b0checks[profile+'_'+metric]=dict(mean=float(values.mean()),n_finite=len(values))
    scaling=dict(version=PURITY_VERSION,comparison_set=methods,
        formula='2*p*(1-n)/(p+1-n); independent profile min-max; finite harmonic 0/0 limit=0; missing/degenerate inputs stay NA',
        role='Auxiliary within this declared set; raw dual-profile P/N F1 remains visible.',
        previous_scaling=ident(OLD/'purity_scaling.json'))
    for profile in ('scclassify','fixed'):
        pv=[float(r[profile+'_positive_F1']) for r in rows];nv=[float(r[profile+'_negative_F1']) for r in rows]
        pmin,pmax,nmin,nmax=min(pv),max(pv),min(nv),max(nv)
        assert pmax>pmin and nmax>nmin
        scaling[profile]=dict(positive_F1=dict(min=pmin,max=pmax),negative_F1=dict(min=nmin,max=nmax))
        for r,p,n in zip(rows,pv,nv):
            sp=(p-pmin)/(pmax-pmin);clean=1-(n-nmin)/(nmax-nmin)
            r[profile+'_Purity_F1']=repr(0.0 if sp+clean==0 else 2*sp*clean/(sp+clean))
    for r in rows:
        r.update(purity_scaling_version=PURITY_VERSION,purity_scaling_path=str(OFFICIAL/'purity_scaling.json'),
            reporting_protocol=VERSION)
    for old in oldrows:
        assert all(rowmap[old['method']][k]==old[k] for k in oldcols if k not in MUTABLE),old['method']
    columns=list(oldcols)
    for row in rows:
        for k in row:
            if k not in columns:columns.append(k)
    compactcols,_=table(OLD/'main_table_compact.csv')
    compactcols+=['scclassify_positive_F1_n','scclassify_negative_F1_n','fixed_negative_F1_n',
                  'mean_transcripts_per_cell','mean_genes_per_cell','reporting_protocol','purity_scaling_version']
    outputs={}
    def stage(target,data):
        target=Path(target);p=STAGE/target.relative_to(WORK);atomic(p,data)
        outputs[str(target)]=ident(p)
    stage(OFFICIAL/'main_table.csv',csv_bytes(columns,rows))
    stage(OFFICIAL/'main_table_compact.csv',csv_bytes(compactcols,rows))
    stage(OFFICIAL/'purity_scaling.json',js(scaling))
    stage(OFFICIAL/'role_configuration.json',js(role_record()))
    policy=dict(version=VERSION,user_instruction_date='2026-10-16 Australia/Sydney',
        F1_instruction='在评价时，将固定指标的F1和Scclassify F1同样作为评测指标，如果固定指标上有明显突出也需要注明',
        comparator_instruction='同时将需要与Base match进行比较的实验对照组中添加B0的指标；论文可能主要以B0为主要对照组',
        current_assessment='Parallel fixed-label and scClassify positive/negative F1; highlight either profile standout and conflicts, with original populations. No post-hoc scalar winner score.',
        paper_reference='B0 explicitly included for all BASE_MATCH-related contrasts; may serve as paper principal reference.',
        mechanistic_control='Retain BASE_MATCH and A2 GT/SHUFFLED controlled contrasts; B0 has unmatched training initialization/history/order.',
        frozen_gate='Original sc-primary/fixed-auxiliary software decisions are immutable historical calculations, not silently relabeled as dual-profile gates.',
        evaluations='Reuse verified actual existing scores; no mask, expression matrix, annotation, reference model, NA rule or scientific thresholds changed.')
    stage(OFFICIAL/'assessment_policy.json',js(policy))
    protocol=read(OLD/'protocol.json')
    protocol.update(version=VERSION,methods=methods,primary=['fixed-label P/N F1','scClassify P/N F1'],
        secondary='Purity and other diagnostics; capture, RNA retention, structure, QC and spatial uncertainty accompany both profiles',
        purity=scaling['formula'],assessment_policy=str(OFFICIAL/'assessment_policy.json'))
    protocol['boundary_extension']=dict(scientific_source_commit=SOURCE_COMMIT,
        predecessor_protocol=ident(OLD/'protocol.json'),complete_check=ident(RUN/'complete-check.json'),
        root_raw_review=ident(ROOT/'analysis/root_review_20261016/raw_validation.json'),
        new_methods=list(NEW),not_new_methods=dict(A2_NATIVE='BASE_MATCH alias',A1_CORE='information-only no mask endpoint'),
        frozen_scientific_core_unchanged=True,actual_continuation=ident(ROOT/'continuation_morph_first_20261015/state.json'))
    stage(OFFICIAL/'protocol.json',js(protocol))
    aliases=dict(A2_NATIVE=dict(canonical_method='BASE_MATCH',new_independent_endpoint=False,
        evidence=str(RUN/'A2_CORE/comparison.json')),A1_CORE=dict(kind='information_study',mask_endpoint=False,
        evidence=str(RUN/'A1_CORE/decision.json')))
    stage(OFFICIAL/'aliases_and_nonmask_studies.json',js(aliases))
    contrasts=[]
    pairs=[(m,b) for m in ('BASE_MATCH','V2_MORPH',*NEW) for b in ('B0','BASE_MATCH') if m!=b]
    pairs += [('P2_MORPH','V2_MORPH'),('A2_GTC','A2_GT'),('A2_GTC','A2_SHUFFLED')]
    pairs += [(m,'V2_MORPH') for m in ('MORPH_NO_DAPI','MORPH_SIMPLE_FUSION','MORPH_NO_GEO')]
    for m,b in dict.fromkeys(pairs):
        a,c=rowmap[m],rowmap[b]
        for profile in ('fixed','scclassify'):
            contrasts.append(dict(candidate=m,control=b,profile=profile,
                candidate_positive_F1=a[profile+'_positive_F1'],control_positive_F1=c[profile+'_positive_F1'],
                delta_positive_F1=repr(float(a[profile+'_positive_F1'])-float(c[profile+'_positive_F1'])),
                candidate_negative_F1=a[profile+'_negative_F1'],control_negative_F1=c[profile+'_negative_F1'],
                delta_negative_F1=repr(float(a[profile+'_negative_F1'])-float(c[profile+'_negative_F1'])),
                candidate_P_n=a[profile+'_positive_F1_n'],control_P_n=c[profile+'_positive_F1_n'],
                candidate_N_n=a[profile+'_negative_F1_n'],control_N_n=c[profile+'_negative_F1_n'],
                candidate_capture_pct=a['capture_pct'],control_capture_pct=c['capture_pct'],
                capture_delta_pp=repr(float(a['capture_pct'])-float(c['capture_pct'])),
                RNA_retention_vs_control=repr(float(a['mean_transcripts_per_cell'])/float(c['mean_transcripts_per_cell'])),
                candidate_area_um2=a['mean_cell_area_um2'],control_area_um2=c['mean_cell_area_um2'],
                candidate_disconnected8=a['disconnected_cells_8'],control_disconnected8=c['disconnected_cells_8'],
                comparison_scope='historical unmatched reference; no causal attribution' if b=='B0' else 'tested controlled contrast; see endpoint identity and limits'))
    stage(OFFICIAL/'dual_profile_comparisons.csv',csv_bytes(list(contrasts[0]),contrasts))
    show=['B0','BASE_MATCH','V2_MORPH',*NEW]
    stage(OFFICIAL/'boundary_endpoints_with_B0.csv',csv_bytes(compactcols+[k for k in ['n_segmented_cells','fixed_positive_F1_n'] if k not in compactcols],[rowmap[m] for m in show]))
    spatialcols=['method']+[k for k in columns if k.startswith('scclassify_spatial_')]
    stage(OFFICIAL/'spatial_diversity_all_methods.csv',csv_bytes(spatialcols,rows))
    registry=dict(status='verified_boundary_extension',methods=methods,new_methods=list(NEW),
        historical20_registry=ident(OLD/'artifact_registry_paired_20261014.json'),
        historical20_complete=ident(OLD/'complete.json'),new_and_control_endpoints=raw['endpoints'],
        B0_reference=dict(raw_metric_check=b0checks,input=ident(b0/'input.json'),
            sc_scores=ident(b0/'paper/markers_per_cell.csv'),fixed_scores=ident(b0/'fixed/fixed_population.csv'),
            scoring_receipt=ident(b0/'scoring_complete.json'),
            comparison_scope='historical unmatched reference; full native B0 matrix and evaluation trace through source_evaluation/input.json'),
        nonmask_or_alias=aliases,source=ident(RUN/'reports/endpoint_table_provenance.json'),
        full_completion=ident(RUN/'complete-check.json'))
    stage(OFFICIAL/'artifact_registry.json',js(registry))
    coverage=read(OLD/'metric_coverage.json')
    coverage.update(version=VERSION,comparison_set=methods,
        boundary_extension='Eight new unique endpoints: own final mask/matrix, ownQC, both raw six metrics/finite denominators, reference correlations, author spatial diversity, fixed-ID strata/owners and spatial bootstrap. Independent analysis and root raw reductions linked in complete.json.',
        current_assessment_policy=policy,old20_unchanged=ident(OLD/'main_table.csv'),
        replication='Single specimen/seed; spatial intervals conditional within slide, not training/specimen replication or true owner accuracy.')
    stage(OFFICIAL/'metric_coverage.json',js(coverage))
    validation=dict(status='passed',n_methods=28,unique_new_methods=list(NEW),
        historical_non_Purity_strings_exact=True,historical_modified_fields_in_new_copy=sorted(MUTABLE),
        old20_on_disk_unchanged=True,all_enabled_applicable_complete=True,raw_review=raw['status'],
        B0_raw_metric_reductions=b0checks,source_table=ident(RUN/'reports/endpoint_table.csv'),
        producer=ident(__file__),backup=ident(BACKUP/'manifest.json'),
        NA_policy='Historical blanks/NA retained; no fabricated values for nonmask A1 or duplicate NATIVE.')
    stage(OFFICIAL/'validation.json',js(validation))
    atomic(POST/'prepared.json',js(dict(status='prepared',version=VERSION,outputs=outputs,
        original20=ident(OLD/'main_table.csv'),producer=ident(__file__))))
    print('prepared',len(outputs),'files',OFFICIAL)


def publish():
    plan=read(POST/'prepared.json')
    assert plan['producer']['sha256']==sha(__file__)
    review=read(ROOT/'analysis/root_review_20261016/final_review.json')
    assert review['status']=='passed' and review['latest_user_instructions_incorporated']
    independent=ROOT/'analysis/independent_20261016'
    checks=read(independent/'checks.json')
    assert len(checks)>=680 and all(c['passed'] for c in checks)
    assert (independent/'independent_analysis.md').stat().st_size>4000
    for record in review['evidence'].values():
        assert sha(record['path'])==record['sha256'],record['path']
    backup=baseline(); outputs=dict(plan['outputs'])
    prior=POST/'publication_journal.json'
    prior_hashes={x['path']:x['sha256'] for x in read(prior)['files']} if prior.exists() else {}
    journal=[]
    for target,source in outputs.items():
        assert sha(source['path'])==source['sha256']
        if Path(target).exists():
            assert sha(target) in (source['sha256'],prior_hashes.get(target)),target
        atomic(target,Path(source['path']).read_bytes());assert sha(target)==source['sha256']
        journal.append(ident(target))
    # Current overlay explicitly supersedes preserved historical selection prose.
    overlay=(ROOT/'analysis/root_review_20261016/context_overlay.md').read_text()
    assert overlay.startswith('<!-- '+MARKER+'_BEGIN -->') and overlay.endswith('<!-- '+MARKER+'_END -->\n')
    for name in ['README.md','01_project_context.md','02_paths_and_environment.md','03_experiment_matrix.md',
                 '04_metrics_framework.md','05_decisions_and_protocols.md','06_paper_evaluation_protocol_20261008.md',
                 '07_evaluation_dual_annotation_20261013.md']:
        target=CONTEXT/name;original=Path(backup['files'][str(target)]['path']).read_text()
        data=(overlay+'\n'+original).encode()
        if target.exists():
            assert target.read_bytes() in (data,Path(backup['files'][str(target)]['path']).read_bytes()),target
        atomic(target,data);journal.append(ident(target))
    exp=ROOT/'analysis/root_review_20261016/experiment_record.md'
    atomic(CONTEXT/'experiments/boundary_three_20261015.md',exp.read_bytes())
    journal.append(ident(CONTEXT/'experiments/boundary_three_20261015.md'))
    cols,rows=table(backup['files'][str(CONTEXT/'experiment_registry.csv')]['path'])
    additional=[dict(experiment_id='BOUNDARY-20261015-'+n,logical_method=n,status='COMPLETED',
        scope_reason=('information_only;GO_A2;not_mask_or_F1_endpoint' if n=='A1_CORE' else
            'four_arms_evaluated;A2_NATIVE_alias_BASE_MATCH' if n=='A2_CORE' else
            'full_own_mask_matrix_dual_F1_evaluated;single_seed_Rep1'),
        evidence_root=str(RUN/('A2_CORE' if n.startswith('A2_') else n)))
        for n in ('A1_CORE','A2_CORE',*NEW)]
    assert not ({r['experiment_id'] for r in rows}&{r['experiment_id'] for r in additional})
    data=csv_bytes(cols,rows+additional);target=CONTEXT/'experiment_registry.csv'
    assert target.read_bytes() in (data,Path(backup['files'][str(target)]['path']).read_bytes())
    atomic(target,data);journal.append(ident(target))
    published_rows=table(OFFICIAL/'main_table.csv')[1]
    summary=dict(version=VERSION,status='completed_independently_reviewed',methods=[r['method'] for r in published_rows],rows=published_rows,
        conclusions=review['conclusions'],assessment_policy=read(OFFICIAL/'assessment_policy.json'),
        complete_check=ident(RUN/'complete-check.json'),independent_analysis=ident(independent/'independent_analysis.md'),
        root_review=ident(ROOT/'analysis/root_review_20261016/final_review.json'),
        current_context=str(CONTEXT/'experiments/boundary_three_20261015.md'),
        legacy_scope='Old20 and pairwise Purity outputs stay at original paths; this version is current28. B0 is historical unmatched reference.')
    atomic(OFFICIAL/'summary.json',js(summary));journal.append(ident(OFFICIAL/'summary.json'))
    deps=dict(completion=ident(RUN/'complete-check.json'),root_review=ident(ROOT/'analysis/root_review_20261016/final_review.json'),
        raw_validation=ident(ROOT/'analysis/root_review_20261016/raw_validation.json'),
        independent_analysis=ident(independent/'independent_analysis.md'),independent_checks=ident(independent/'checks.json'),
        user_policy=ident(OFFICIAL/'assessment_policy.json'),backup=ident(BACKUP/'manifest.json'))
    complete=dict(status='succeeded',version=VERSION,n_methods=28,new_unique_endpoints=8,nonmask_studies=1,
        historical20_raw_unchanged=True,dependencies=deps,
        outputs={p.name:ident(p) for p in sorted(OFFICIAL.iterdir()) if p.is_file() and p.name!='complete.json'})
    atomic(OFFICIAL/'complete.json',js(complete));journal.append(ident(OFFICIAL/'complete.json'))
    atomic(POST/'publication_journal.json',js(dict(status='published',files=journal)))
    verify()


def verify():
    baseline()
    journal=read(POST/'publication_journal.json')
    for rec in journal['files']:assert sha(rec['path'])==rec['sha256'],rec['path']
    completion=read(OFFICIAL/'complete.json')
    for rec in list(completion['dependencies'].values())+list(completion['outputs'].values()):
        assert sha(rec['path'])==rec['sha256'],rec['path']
    cols,rows=table(OFFICIAL/'main_table.csv');oldcols,oldrows=table(OLD/'main_table.csv')
    by={r['method']:r for r in rows}
    assert len(rows)==len(by)==28 and 'A1_CORE' not in by and 'A2_NATIVE' not in by
    assert set(NEW)<=set(by)
    for old in oldrows:
        assert all(by[old['method']][k]==old[k] for k in oldcols if k not in MUTABLE)
    scaling=read(OFFICIAL/'purity_scaling.json')
    assert scaling['comparison_set']==list(by)
    for profile in ('scclassify','fixed'):
        ps=[float(r[profile+'_positive_F1']) for r in rows];ns=[float(r[profile+'_negative_F1']) for r in rows]
        assert scaling[profile]['positive_F1']==dict(min=min(ps),max=max(ps))
        assert scaling[profile]['negative_F1']==dict(min=min(ns),max=max(ns))
        for r,p,n in zip(rows,ps,ns):
            p=(p-min(ps))/(max(ps)-min(ps));clean=1-(n-min(ns))/(max(ns)-min(ns))
            close(r[profile+'_Purity_F1'],0 if p+clean==0 else 2*p*clean/(p+clean))
    _,compact=table(OFFICIAL/'main_table_compact.csv')
    assert len(compact)==28 and [r['method'] for r in compact]==list(by)
    for r in compact:
        assert all(r[k]==by[r['method']][k] for k in r)
    _,comp=table(OFFICIAL/'dual_profile_comparisons.csv')
    for row in comp:
        a,b=by[row['candidate']],by[row['control']];profile=row['profile']
        for sign in ('positive','negative'):
            close(row['delta_'+sign+'_F1'],float(a[profile+'_'+sign+'_F1'])-float(b[profile+'_'+sign+'_F1']))
    for m in NEW:
        assert {(r['control'],r['profile']) for r in comp if r['candidate']==m}>={('B0','fixed'),('B0','scclassify'),('BASE_MATCH','fixed'),('BASE_MATCH','scclassify')}
    for p in CONTEXT.glob('*.md'):
        if p.name in ['README.md'] or p.name.startswith(('01_','02_','03_','04_','05_','06_','07_')):
            original=Path(read(BACKUP/'manifest.json')['files'][str(p)]['path']).read_bytes()
            assert p.read_bytes().endswith(original) and p.read_text().count(MARKER+'_BEGIN')==1
    atomic(POST/'verification.json',js(dict(status='passed',n_methods=28,new_unique_endpoints=8,
        historical20_non_Purity_strings_preserved=True,old20_files_untouched=True,
        dual_F1_parallel=True,B0_and_BASE_MATCH_comparisons_present_for_all_new8=True,
        table_compact_summary_registry_context_sources_hashes_consistent=True,
        artifacts=ident(OFFICIAL/'complete.json'),publication_journal=ident(POST/'publication_journal.json'))))
    print('verified official28, dual profiles, B0 controls, immutable old20 and context preservation')


def snapshot():
    verify()
    files=[Path(x['path']) for x in read(POST/'publication_journal.json')['files']]
    files+=list((ROOT/'analysis/independent_20261016').glob('*'))
    files+=list((ROOT/'analysis/root_review_20261016').glob('*'))
    files+=[POST/'prepared.json',POST/'verification.json',POST/'publication_journal.json',
            BACKUP/'manifest.json',POST/'status_final.json',
            ROOT/'handoff/model_configuration.json',RUN/'complete-check.json',
            ROOT/'continuation_morph_first_20261015/state.json',
            WORK/'docs/BIDCell_Morph_First_Continuation_Addendum_20261015.md']
    files+=list(OLD.glob('*.json'))+list(OLD.glob('*.csv'))
    manifest=[]
    for p in sorted(set(files)):
        if not p.is_file() or p.stat().st_size>5*1024*1024:continue
        rel=p.relative_to(WORK);dest=TRACKED/'snapshot'/rel
        atomic(dest,p.read_bytes());assert sha(dest)==sha(p)
        manifest.append(dict(source=str(p),tracked_path=str(dest.relative_to(REPO)),bytes=p.stat().st_size,sha256=sha(p)))
    atomic(TRACKED/'snapshot_manifest.json',js(dict(schema='boundary-results-snapshot-v1',
        scope='Exact local publication/context/analysis snapshots; full matrices/checkpoints remain in source paths recorded by registry.',
        source_scientific_commit=SOURCE_COMMIT,files=manifest)))
    print('snapshotted',len(manifest),'files to',TRACKED)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','publish','verify','snapshot'])
    args=parser.parse_args();globals()[args.action]()
