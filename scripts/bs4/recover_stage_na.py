"""Report-only recovery: preserve all original computation and expose a new report view."""
import argparse,json,math,shutil,subprocess,time
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.crgf.common import atomic_json,identity,sha256,verify_outputs,source_identity,Monitor
from scripts.cos.diagnostics import METRICS,load_anchor
from .stage_population import load_population,align_stage,postprocess_increment
from .stage_reference import load_stage_table
from .tables import write_table
PAIRS=[('P1','B'),('P2','B'),('T1','B'),('T2','B'),('T2','T1'),('T1','S'),('T2','S')]
SOURCES=['scripts/bs4/recover_stage_na.py','scripts/bs4/stage_reference.py','scripts/bs4/stage_population.py','scripts/bs4/tables.py','scripts/cos/diagnostics.py','scripts/crgf/common.py']
GiB=1024**3

def read(p):return json.loads(Path(p).read_text())
def replaced(relative):
    parts=relative.parts
    if relative.as_posix() in ('complete.json','metric_coverage.csv'):return True
    if parts[0]=='stage_domains':return True
    if parts[0]=='diagnostics' and len(parts)>1:
        return parts[1]=='stage_population' or parts[1].startswith('postprocess_') or parts[1] in ('stage_population_domains.csv','stage_population_contract.json')
    return False

def verify_execution_review(path,root,output,source):
    receipt=read(path)
    if receipt.get('review_scope')!='report_only_stage_reference_NA' or receipt.get('independent') is not True or receipt.get('approved_to_execute') is not True:raise ValueError('independent report-only execution permission required')
    if receipt.get('original_root')!=str(root) or receipt.get('output_root')!=str(output):raise ValueError('report recovery review scope mismatch')
    if receipt.get('source')!=source:raise ValueError('reviewed recovery source mismatch')
    expected={str(p):sha256(p) for p in (root/'configs/queue.json',root/'source/frozen.json',root/'compute_complete.json')}
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

def artifact_index(comparison,original):
    rows=[dict(relative_path=rel,action='reused_readonly_exact',authority_path=obj['path'],sha256=obj['sha256']) for rel,obj in original.items() if not replaced(Path(rel))]
    for p in sorted(comparison.rglob('*')):
        if p.is_file() and not p.is_symlink():rows.append(dict(relative_path=str(p.relative_to(comparison)),action='recomputed_stage_availability' if p.name!='metric_coverage.csv' else 'coverage_extended',authority_path=str(p),sha256=sha256(p)))
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

def run(root,output,review_receipt):
    root=Path(root).resolve();output=Path(output).resolve();repo=Path(__file__).resolve().parents[2]
    if root/'recovery' not in output.parents or output==root or root/'comparison' in output.parents:raise ValueError('recovery cannot write original comparison')
    source=source_state(repo);execution_review=verify_execution_review(review_receipt,root,output,source)
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
    original_complete=read(root/'compute_complete.json')
    if original_complete['status']!='succeeded' or original_complete['required_missing'] or len(original_complete['artifacts'])!=17:raise ValueError('original full computation incomplete')
    verify_outputs(dict(outputs=original_complete['artifacts']))
    protected={str(p):identity(p) for p in [root/'compute_complete.json',root/'configs/queue.json',root/'source/frozen.json',root/'source/review_receipt.json',root/'comparison/complete.json']}
    protected[execution_review['path']]=execution_review
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
        population=load_population(q['population']);normalization=[];authority={}
        for arm,spec in stage_methods.items():
            validation=read(Path(spec['evaluation'])/'diagnostics/stages/validation.json')
            for stage in sorted(validation['count_artifacts']):
                for anchor,anchor_spec in q['anchors'].items():
                    raw,provenance=load_stage_table(spec,anchor,stage,q['prepared']);aligned,domain=align_stage(raw,population,load_anchor(anchor_spec))
                    dest=output/'normalized_stages'/arm/f'fixed_{anchor}_{stage}.csv';write_table(aligned,dest,na_rep='NA')
                    key='/'.join((arm,anchor,stage));authority[key]=provenance
                    for obj in provenance['inputs'].values():protected[obj['path']]=obj
                    normalization.extend(normalization_records(aligned,domain,provenance['summary'],arm,anchor,stage))
                print('normalized',arm,stage,flush=True)
        atomic_json(output/'stage_input_authority.json',authority)
        pd.DataFrame(normalization).drop(columns='finite_values_replaced_by_NA').to_csv(output/'stage_domain_audit.csv',index=False)
        postprocess_increment(stage_methods,q['anchors'],PAIRS,population,comparison/'diagnostics',prepared=q['prepared'])
        for before,after in [('pre','postconnect_canonical'),('postconnect_canonical','tile_merge'),('tile_merge','final')]:
            postprocess_increment(stage_methods,q['anchors'],[p for p in PAIRS if 'S' not in p],population,comparison/'stage_domains'/(before+'_to_'+after),stages=(before,after),prepared=q['prepared'])
        # All fixed-final and sensitivity tables retain their original value and NA domains.
        unchanged,empty=validate_reused_tables(comparison,original)
        summaries,changed=compare_stage_summaries(comparison,root/'comparison')
        atomic_json(output/'regression.json',dict(status='passed',unchanged_fixed_final_and_sensitivity_tables=unchanged,n_unchanged=len(unchanged),empty_original_tables=empty,four_endpoint_rows=sum(len(pd.read_csv(p)) for p in summaries),changed_rows=changed,not_recomputed=['train','predict','postprocess','matrix','scClassify','raw final metrics','ownership transfers','Purity']))
        coverage=pd.read_csv(root/'comparison/metric_coverage.csv')
        extra=pd.DataFrame([dict(metric_id='counts_derived_zero_reference_NA_'+str(p.relative_to(comparison)),profile='fixed_stage_B1',comparison='B/S/P1/P2/T1/T2',required_this_run=True,status='partial_domain_NA',artifact_path=str(p)) for p in summaries])
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
    atomic_json(output/'complete.json',dict(version='BS4-report-stage-reference-NA-recovery-v1',status='succeeded',report_authority=str(comparison),original_machine_comparison=str(root/'comparison'),original_compute_receipt=identity(root/'compute_complete.json'),execution_source=execution,execution_review=execution_review,source=source,protected_inputs=protected,outputs=outputs,wall_seconds=time.monotonic()-started,scientific_acceptance=False,reason='legacy stage score tables without zero-reference flags admitted zero-reference F1/Recall as zeros; counts-derived NA applied before metric-specific four-endpoint intersection',scope='report-only recovery; original successful computations and machine report preserved'))
    print('report-only recovery succeeded',output,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--review-receipt',required=True);a=p.parse_args();run(a.root,a.output,a.review_receipt)
