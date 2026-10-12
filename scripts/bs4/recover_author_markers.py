"""Author-aligned marker report recovery; no upstream computation or annotation."""
import argparse,json,math,shutil,subprocess,time
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.crgf.common import atomic_json,identity,sha256,verify_outputs,source_identity,Monitor
from scripts.cos.diagnostics import METRICS,load_anchor
from .stage_population import load_population,align_stage,postprocess_increment
from .author_stage import load_stage_table
from .comparison import population_scores
from .author_marker import fixed_scores
from .author_stage_summary import records as stage_summary_records,publish as publish_stage_summaries,remap_coverage as remap_stage_coverage
from .tables import write_table
PAIRS=[('P1','B'),('P2','B'),('T1','B'),('T2','B'),('T2','T1'),('T1','S'),('T2','S')]
SOURCES=['scripts/bs4/recover_author_markers.py','scripts/bs4/author_stage_summary.py','scripts/bs4/author_stage.py','scripts/bs4/author_marker.py','scripts/bs4/comparison.py','scripts/bs4/diagnostics.py','scripts/bs4/stage_population.py','scripts/bs4/tables.py','scripts/cos/diagnostics.py','scripts/crgf/common.py']
GiB=1024**3

def read(p):return json.loads(Path(p).read_text())
def replaced(relative):
    parts=relative.parts
    if relative.as_posix() in ('complete.json','metric_coverage.csv','direction_screen.json'):return True
    if parts[0] in ('stage_domains','fixed_B1QC','sensitivity'):return True
    if parts[0]=='diagnostics' and len(parts)>1:
        return parts[1]=='stage_population' or parts[1].startswith('postprocess_') or parts[1] in ('stage_population_domains.csv','stage_population_contract.json')
    return False

def verify_execution_review(path,root,output,source,profile_audit):
    receipt=read(path)
    if receipt.get('review_scope')!='report_only_author_marker_alignment' or receipt.get('independent') is not True or receipt.get('approved_to_execute') is not True:raise ValueError('independent report-only execution permission required')
    if receipt.get('original_root')!=str(root) or receipt.get('output_root')!=str(output):raise ValueError('report recovery review scope mismatch')
    if receipt.get('source')!=source:raise ValueError('reviewed recovery source mismatch')
    if receipt.get('profile_audit')!=identity(profile_audit):raise ValueError('reviewed profile audit mismatch')
    expected={str(p):sha256(p) for p in (root/'configs/queue.json',root/'source/frozen.json',root/'compute_complete.json',root/'source/context_author_rule_pre_update_snapshot.json')}
    expected['/home/zwan0726/work/outputs/bidcell_metric_protocol_review_20261008/publish_expression_metrics.R']='2d7f0e0da83fed939e5cc2316ea159f5962477e214d00685c73f0f01142775ec'
    if sha256('/home/zwan0726/work/outputs/bidcell_metric_protocol_review_20261008/publish_expression_metrics.R')!=expected['/home/zwan0726/work/outputs/bidcell_metric_protocol_review_20261008/publish_expression_metrics.R']:raise ValueError('archived author source drift')
    if receipt.get('inputs')!=expected:raise ValueError('reviewed original execution identity drift')
    evidence=receipt.get('evidence',{})
    if not evidence:raise ValueError('independent execution evidence absent')
    for p,value in evidence.items():
        if sha256(p)!=value:raise ValueError('reviewed evidence drift '+p)
    return identity(path)

def normalization_records(aligned,domain,summary,arm,anchor,stage):
    for key in set(domain)&set(summary):
        if domain[key]!=summary[key]:raise ValueError('normalization domain/source disagreement '+key)
    base={**domain,**summary};rows=[]
    for metric in METRICS:
        reasons=aligned[metric+'_reason']
        rows.append(dict(base,method=arm,anchor=anchor,stage=stage,metric=metric,n_evaluable=int(reasons.eq('available').sum()),n_unavailable=int(reasons.ne('available').sum()),**{'n_'+reason:int(reasons.eq(reason).sum()) for reason in ('missing_stage_id','label_ineligible','zero_reference_counts','undefined_original_metric','nonfinite_original_metric')}))
    return rows

def validate_reused_tables(comparison,original):
    unchanged=[];empty=[]
    def parsed(path):
        try:return pd.read_csv(path),False
        except pd.errors.EmptyDataError:return pd.DataFrame(),True
    for rel,obj in original.items():
        if Path(rel).parts[0] not in ('fixed_B1QC','sensitivity'):continue
        new=comparison/rel
        if identity(new)!=obj:raise ValueError('unchanged final/sensitivity identity drift')
        # Historical empty-domain CSVs may have no columns. Identity above still
        # requires the exact original bytes; compare their empty parsed tables.
        if rel.endswith(('.csv','.csv.gz')):
            old_frame,old_empty=parsed(obj['path']);new_frame,new_empty=parsed(new)
            if old_empty!=new_empty:raise ValueError('empty-table representation drift')
            if old_empty:
                allowed={f'sensitivity/ownQC_{arm}/paired_metrics.csv' for arm in ('B','S','P1','P2','T1','T2')}
                if rel not in allowed or Path(obj['path']).read_bytes()!=b'\n':raise ValueError('unexpected empty original report table '+rel)
                empty.append(dict(relative_path=rel,original=obj,reason='single-method ownQC domain has no method pair; original newline-only table preserved'))
            pd.testing.assert_frame_equal(old_frame,new_frame,check_exact=True)
        unchanged.append(rel)
    return unchanged,empty

def compare_stage_summaries(comparison,original_comparison):
    summaries=[comparison/'diagnostics/postprocess_cos_increment.csv',*sorted((comparison/'stage_domains').glob('*/postprocess_cos_increment.csv'))]
    changed=[]
    for path in summaries:
        new=pd.read_csv(path);old=pd.read_csv(original_comparison/path.relative_to(comparison));keys=['comparison','from_stage','to_stage','anchor','metric'];a=new.set_index(keys);b=old.set_index(keys)
        if not a.index.equals(b.index):raise ValueError('comparison row inventory/order drift')
        for key in a.index:
            columns=[c for c in a.columns if not ((pd.isna(a.loc[key,c]) and pd.isna(b.loc[key,c])) or a.loc[key,c]==b.loc[key,c])]
            if columns:changed.append(dict(zip(keys,key),changed_fields=columns,before={c:None if pd.isna(b.loc[key,c]) else b.loc[key,c].item() if isinstance(b.loc[key,c],np.generic) else b.loc[key,c] for c in columns},after={c:None if pd.isna(a.loc[key,c]) else a.loc[key,c].item() if isinstance(a.loc[key,c],np.generic) else a.loc[key,c] for c in columns}))
    return summaries,changed

def recompute_population_reports(methods,q,original,comparison):
    scopes=[original/'fixed_B1QC',*sorted((original/'sensitivity').iterdir())];rows=[]
    labels=load_anchor(q['anchors']['B1']);prepared=Path(q['prepared']);markers=read(prepared/'markers.json');refs=(prepared/'reference_genes.txt').read_text().splitlines();score_tables={}
    for arm,spec in methods.items():
        matrix=Path(spec['matrix']);ids=pd.read_csv(matrix/'cell_mapping.csv').canonical_cell_id.to_numpy();genes=pd.read_csv(matrix/'gene_mapping.csv').gene.tolist()
        score_tables[arm]=fixed_scores(np.load(matrix/'counts.npy',mmap_mode='r'),ids,genes,labels,markers,refs)
    expected={'fixed_B1QC','historical_ABC_commonQC','six_methods_commonQC','four_new_arms_commonQC',*[a+'_'+b+'_commonQC' for a,b in PAIRS],*['ownQC_'+a for a in methods]}
    if {p.name for p in scopes}!=expected:raise ValueError('frozen comparison population inventory drift')
    for scope in scopes:
        pop=read(scope/'population.json');relative=scope.relative_to(original);selected={scope.name[6:]:methods[scope.name[6:]]} if scope.name.startswith('ownQC_') else methods
        population_scores(selected,q,comparison/relative,pop['ids'],pop['label'],score_tables=score_tables)
        if read(comparison/relative/'population.json')!=pop:raise ValueError('frozen population order/label drift')
        for old in sorted(scope.iterdir()):
            new=comparison/relative/old.name
            if not new.exists():raise ValueError('recomputed population artifact missing')
            if old.suffix=='.json':same=read(old)==read(new);n=0;changed=0
            else:
                try:a=pd.read_csv(old);b=pd.read_csv(new)
                except pd.errors.EmptyDataError:
                    if scope.name not in ['ownQC_'+x for x in methods] or old.name!='paired_metrics.csv' or old.read_bytes()!=b'\n' or new.read_bytes()!=b'\n':raise
                    a=b=pd.DataFrame()
                if list(a.columns)!=list(b.columns) or len(a)!=len(b):raise ValueError('population report schema/order drift')
                equal=(a.eq(b)|(a.isna()&b.isna()));n=len(a);changed=int((~equal).any(axis=1).sum());same=changed==0
            rows.append(dict(scope=scope.name,artifact=str(relative/old.name),n_rows=n,changed_rows=changed,exact_values_unchanged=same,original=identity(old),recomputed=identity(new)))
    main=pd.read_csv(comparison/'fixed_B1QC/paired_metrics.csv');screen=[]
    for arm in ('P1','P2','T1','T2'):
        part=main[(main.comparison==arm+'-B')&(main.type=='__all__')].set_index('metric');dp=float(part.loc['positive_F1','mean_delta']);dn=float(part.loc['negative_F1','mean_delta'])
        screen.append(dict(arm=arm,delta_positive_F1=dp,delta_negative_F1=dn,direction_screen=bool(dp>=0 and dn<0),interpretation='descriptive direction only; all-metric tradeoffs remain required'))
    atomic_json(comparison/'direction_screen.json',dict(raw_unrounded=True,rows=screen))
    return rows

def artifact_index(comparison,original):
    rows=[dict(relative_path=rel,action='reused_readonly_exact',authority_path=obj['path'],sha256=obj['sha256']) for rel,obj in original.items() if not replaced(Path(rel))]
    for p in sorted(comparison.rglob('*')):
        if p.is_file() and not p.is_symlink():rows.append(dict(relative_path=str(p.relative_to(comparison)),action='recomputed_author_marker_scores_and_domains' if p.name!='metric_coverage.csv' else 'coverage_extended',authority_path=str(p),sha256=sha256(p)))
    return rows

def output_manifest(output):
    return {str(p.relative_to(output)):identity(p) for p in sorted(output.rglob('*')) if p.is_file() and p!=output/'complete.json' and '.git' not in p.parts}

def validate_original_dependencies(root,q,protected):
    methods={**q['historical_methods'],**q['methods']};stage_methods={a:dict(s) for a,s in methods.items()};stage_methods['B']['evaluation']=str(root/'diagnostics/B_native_replay/evaluation')
    admission_path=root/'tests/admission.json';formal_review=read(root/'source/review_receipt.json')
    if formal_review['evidence'][str(admission_path)]!=sha256(admission_path):raise ValueError('original admitted B table authority changed')
    stage_methods['B']['stage_table_authority']=read(admission_path)['evidence']['B_native_recursive_artifacts']
    stage_methods['B']['stage_table_authority_receipt']=str(admission_path)
    machine_report=read(root/'comparison/complete.json')
    verify_outputs(machine_report)
    for arm,spec in stage_methods.items():
        for kind in ('matrix','evaluate'):
            declared=machine_report['arm_receipts'][arm][kind]
            if sha256(declared['path'])!=declared['sha256']:raise ValueError('original published '+kind+' receipt changed '+arm)
            protected[declared['path']]=identity(declared['path'])
        validationfile=Path(spec['evaluation'])/'diagnostics/stages/validation.json'
        if arm=='B':
            declared=read(root/'tests/admission.json')['evidence']['diagnostics/B_native_replay/complete.json']
            if sha256(declared['path'])!=declared['sha256']:raise ValueError('admitted B replay receipt drift')
            validation_authority=read(declared['path'])['outputs']['diagnostics']
        else:
            ev=read(machine_report['arm_receipts'][arm]['evaluate']['path'])
            matching=[v for v in ev['outputs'].values() if Path(v['path']).resolve()==validationfile.resolve()]
            if len(matching)!=1:raise ValueError('stage validation absent from original evaluation receipt')
            validation_authority=matching[0]
        if sha256(validationfile)!=validation_authority['sha256']:raise ValueError('stage validation receipt changed '+arm)
        protected[str(validationfile)]=identity(validationfile)
    paper=Path(q['prepared'])/'complete.json'
    for arm,spec in methods.items():
        evaluation=read(machine_report['arm_receipts'][arm]['evaluate']['path'])
        profiles=Path(spec['evaluation'])/'profiles_complete.json'
        matches=[v for v in evaluation['outputs'].values() if Path(v['path']).resolve()==profiles.resolve()]
        if len(matches)!=1 or matches[0]['sha256']!=sha256(profiles):raise ValueError('published profiles receipt drift '+arm)
        declared=read(profiles)['dependencies']['prepared']
        if Path(declared['path']).resolve()!=paper.resolve() or declared['sha256']!=sha256(paper):raise ValueError('published scoring reference receipt drift '+arm)
        protected[str(profiles)]=identity(profiles)
    protected[str(paper)]=identity(paper)
    for arm in ('T1','T2'):
        frozen=Path(q['methods'][arm]['cost_root'])/'configs/frozen.json'
        if read(admission_path)['arm_frozen'][arm]['sha256']!=sha256(frozen):raise ValueError('original admitted training freeze drift '+arm)
        training=read(frozen)['identities']['prepared']
        if Path(read(q['methods'][arm]['config'])['prepared_manifest']).resolve()!=Path(training['path']).resolve():raise ValueError('training config/preparation identity mismatch '+arm)
        if sha256(training['path'])!=training['sha256']:raise ValueError('frozen training preparation drift '+arm)
        protected[str(frozen)]=identity(frozen);protected[training['path']]=identity(training['path'])
    return stage_methods

def source_state(repo):
    return dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True,timeout=30).strip(),tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=repo,text=True,timeout=30).strip(),checkout=str(repo),files={p:sha256(repo/p) for p in subprocess.check_output(['git','ls-files'],cwd=repo,text=True,timeout=30).splitlines()})

def run(root,output,review_receipt,profile_audit):
    root=Path(root).resolve();output=Path(output).resolve();repo=Path(__file__).resolve().parents[2]
    if root/'recovery' not in output.parents or output==root or root/'comparison' in output.parents:raise ValueError('recovery cannot write original comparison')
    source=source_state(repo);execution_review=verify_execution_review(review_receipt,root,output,source,profile_audit)
    if subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True,timeout=30).strip():raise ValueError('recovery implementation must be committed clean')
    if (output/'complete.json').exists():
        complete=read(output/'complete.json');verify_outputs(complete)
        if complete.get('execution_review')!=execution_review or source_state(repo)!=complete['source']:raise ValueError('completed recovery review/source identity drift')
        if complete['execution_source']!=source_identity(SOURCES) or any(sha256(repo/p)!=expected for p,expected in complete['source']['files'].items()):raise ValueError('recovery source drift')
        for obj in complete['protected_inputs'].values():
            if identity(obj['path'])!=obj:raise ValueError('protected source/input drift')
        print('Validated completed report-only recovery; no writes');return
    comparison=output/'comparison'
    if comparison.exists():raise ValueError('preserve incomplete recovery; choose explicit new attempt')
    output.mkdir(parents=True,exist_ok=True)
    if subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True,timeout=30).strip():raise ValueError('recovery implementation must be committed clean')
    execution=source_identity(SOURCES);started=time.monotonic();q=read(root/'configs/queue.json')
    audit=read(profile_audit);verify_outputs(audit)
    if audit.get('status')!='passed' or audit.get('semantic_differences')!=0:raise ValueError('author profile audit must pass without unresolved differences')
    audit_rows=pd.read_csv(audit['outputs']['profile_audit.csv']['path'])
    expected_profiles={'paper_aligned_v1','remote_nuclear_annotation_raw','windows_style_maxpearson_raw'}
    if len(audit_rows)!=108 or audit_rows.duplicated(['method','profile','metric']).any() or set(audit_rows.method)!={'B','S','P1','P2','T1','T2'} or set(audit_rows.profile)!=expected_profiles:raise ValueError('complete 18-profile audit required')
    if any(set(part.metric)!=set(METRICS) for _,part in audit_rows.groupby(['method','profile'])) or not audit_rows.n_semantic_difference.eq(0).all():raise ValueError('profile metric audit incomplete')
    for p,h in audit['execution_source'].items():
        if sha256(repo/p)!=h:raise ValueError('profile audit implementation drift')
    for obj in audit['inputs'].values():
        if identity(obj['path'])!=obj:raise ValueError('profile audit input drift')
    original_complete=read(root/'compute_complete.json')
    if original_complete['status']!='succeeded' or original_complete['required_missing'] or len(original_complete['artifacts'])!=17:raise ValueError('original full computation incomplete')
    verify_outputs(dict(outputs=original_complete['artifacts']))
    protected={str(p):identity(p) for p in [root/'compute_complete.json',root/'configs/queue.json',root/'source/frozen.json',root/'source/review_receipt.json',root/'comparison/complete.json']}
    protected[execution_review['path']]=execution_review
    protected[str(Path(profile_audit).resolve())]=identity(profile_audit)
    protected.update({obj['path']:obj for obj in audit['inputs'].values()})
    protected.update({obj['path']:obj for obj in audit['outputs'].values()})
    rule_snapshot=root/'source/context_author_rule_pre_update_snapshot.json';protected[str(rule_snapshot)]=identity(rule_snapshot)
    for obj in read(rule_snapshot)['files']:
        if Path(obj['snapshot']).name in ('04_metrics_framework.md','06_paper_evaluation_protocol_20261008.md'):
            if sha256(obj['snapshot'])!=obj['sha256']:raise ValueError('immutable author rule snapshot drift')
            protected[obj['snapshot']]=identity(obj['snapshot'])
    stage_methods=validate_original_dependencies(root,q,protected)
    original={str(p.relative_to(root/'comparison')):identity(p) for p in sorted((root/'comparison').rglob('*')) if p.is_file()}
    # Upper estimate exceeds all current original comparison allocations; no copy of biology.
    old_bytes=sum(p.stat().st_blocks*512 for p in (root/'comparison').rglob('*') if p.is_file())
    planned=max(GiB,3*old_bytes);required=math.ceil(planned*1.2)+15*GiB;free=shutil.disk_usage(output).free
    if free<required:raise ValueError('insufficient recovery budget with 20 percent buffer and 15 GiB reserve')
    atomic_json(output/'resource_plan.json',dict(status='passed',free_bytes=free,new_allocation_upper_bytes=planned,buffer_fraction=.2,reserve_bytes=15*GiB,required_free_bytes=required,scope='report tables and complete source bundle/clone only; historical links excluded'))
    atomic_json(output/'protected_before.json',dict(original_report=original,protected_inputs=protected))

    atomic_json(output/'execution_source.json',source)
    with Monitor(output/'monitor.json'):
        comparison.mkdir();index=[]
        for rel,obj in original.items():
            relative=Path(rel)
            if replaced(relative):continue
            dest=comparison/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.symlink_to(obj['path'])
            index.append(dict(relative_path=rel,action='reused_readonly_exact',authority_path=obj['path'],sha256=obj['sha256']))
        population=load_population(q['population']);normalization=[];authority={};single_stage=[]
        for arm,spec in stage_methods.items():
            validation=read(Path(spec['evaluation'])/'diagnostics/stages/validation.json')
            for stage in sorted(validation['count_artifacts']):
                for anchor,anchor_spec in q['anchors'].items():
                    raw,provenance=load_stage_table(spec,anchor,stage,q['prepared']);aligned,domain=align_stage(raw,population,load_anchor(anchor_spec))
                    dest=output/'normalized_stages'/arm/f'fixed_{anchor}_{stage}.csv';write_table(aligned,dest,na_rep='NA')
                    key='/'.join((arm,anchor,stage));authority[key]=provenance
                    for obj in provenance['inputs'].values():protected[obj['path']]=obj
                    normalization.extend(normalization_records(aligned,domain,provenance['summary'],arm,anchor,stage))
                    single_stage.extend(stage_summary_records(aligned,arm,anchor,stage))
                print('normalized',arm,stage,flush=True)
        stage_summaries=publish_stage_summaries(single_stage,authority,comparison)
        atomic_json(output/'stage_input_authority.json',authority)
        atomic_json(output/'stage_domain_audit_schema.json',dict(version='author-marker-stage-domains-v1',zero_reference_counts='descriptive only; does not invalidate a metric',n_zero_reference_counts='legacy compatibility exclusion-count column, always zero',descriptive_zero_fields=['n_zero_reference_count_ids','n_zero_reference_stage_rows','n_zero_reference_known_rows'],score_change_tolerance='5e-15 only for decimal CSV serialization comparisons; author R binary fixture exact',missing_rows='NA with missing_stage_id; never fabricated zero',fixed_label_policy='original historical label eligibility unchanged',four_endpoint_domain='separate finite intersection for each actual author metric'))
        pd.DataFrame(normalization).drop(columns='author_score_changes').to_csv(output/'stage_domain_audit.csv',index=False)
        postprocess_increment(stage_methods,q['anchors'],PAIRS,population,comparison/'diagnostics',prepared=q['prepared'])
        for before,after in [('pre','postconnect_canonical'),('postconnect_canonical','tile_merge'),('tile_merge','final')]:
            postprocess_increment(stage_methods,q['anchors'],[p for p in PAIRS if 'S' not in p],population,comparison/'stage_domains'/(before+'_to_'+after),stages=(before,after),prepared=q['prepared'])
        population_audit=recompute_population_reports(methods={**q['historical_methods'],**q['methods']},q=q,original=root/'comparison',comparison=comparison)
        unchanged=[rel for rel,obj in original.items() if not replaced(Path(rel))]
        for rel in unchanged:
            if identity(comparison/rel)!=original[rel]:raise ValueError('read-only reused report changed '+rel)
        summaries,changed=compare_stage_summaries(comparison,root/'comparison')
        atomic_json(output/'regression.json',dict(status='passed',unchanged_reference_only_artifacts=unchanged,n_unchanged=len(unchanged),population_reports=population_audit,four_endpoint_rows=sum(len(pd.read_csv(p)) for p in summaries),changed_rows=changed,not_recomputed=['train','predict','postprocess','matrix','scClassify','three-profile raw metrics verified against author','ownership transfers','Purity verified from unchanged paper F1']))
        coverage=pd.read_csv(root/'comparison/metric_coverage.csv')
        coverage,old_summary_inputs=remap_stage_coverage(coverage,q['methods'],stage_summaries);protected.update(old_summary_inputs)
        for i,path in coverage.artifact_path.items():
            if isinstance(path,str):
                try:relative=Path(path).relative_to(root/'comparison')
                except ValueError:continue
                if replaced(relative):
                    corrected=comparison/relative
                    if not corrected.is_file():raise ValueError('corrected coverage artifact absent '+str(corrected))
                    coverage.loc[i,'artifact_path']=str(corrected)
        extra=pd.DataFrame([dict(metric_id='author_marker_per_metric_finite_'+str(p.relative_to(comparison)),profile='fixed_stage_B1',comparison='B/S/P1/P2/T1/T2',required_this_run=True,status='partial_domain_NA',artifact_path=str(p)) for p in summaries])
        pd.concat([coverage,extra],ignore_index=True).to_csv(comparison/'metric_coverage.csv',index=False)
        index=artifact_index(comparison,original)
        pd.DataFrame(index).to_csv(output/'artifact_index.csv',index=False)
        for obj in protected.values():
            if identity(obj['path'])!=obj:raise ValueError('protected original/source changed '+obj['path'])
        for rel,obj in original.items():
            if identity(root/'comparison'/rel)!=obj:raise ValueError('original comparison changed '+rel)
        if source_identity(SOURCES)!=execution or source_state(repo)!=source or subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True,timeout=30).strip():raise ValueError('execution source changed during recovery')
        if shutil.disk_usage(output).free<15*GiB:raise ValueError('recovery breached free-space reserve')
    outputs=output_manifest(output)
    atomic_json(output/'complete.json',dict(version='BS4-report-author-marker-recovery-v1',status='succeeded',report_authority=str(comparison),original_machine_comparison=str(root/'comparison'),original_compute_receipt=identity(root/'compute_complete.json'),execution_source=execution,execution_review=execution_review,source=source,protected_inputs=protected,outputs=outputs,wall_seconds=time.monotonic()-started,scientific_acceptance=False,reason='user clarification: author F1 NA/NaN to zero only; Precision/Recall unchanged; zero-reference counts descriptive; fixed historical label eligibility and profile populations preserved',scope='report-only recovery; original successful computations and machine report preserved'))
    print('report-only recovery succeeded',output,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--review-receipt',required=True);p.add_argument('--profile-audit',required=True);a=p.parse_args();run(a.root,a.output,a.review_receipt,a.profile_audit)
