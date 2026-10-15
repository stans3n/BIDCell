"""Manual-only append, with an immutable before/after journal and final audit."""
from __future__ import annotations
import csv
import io
import math
from pathlib import Path
from . import NEW, PARENTS, comparisons
from .common import (atomic,atomic_json,check_identity,check_receipt,dependencies,digest,identity,read,
                     require_manual_resume,sha256,within)
from .publication import RESULT_ROOT,PROTOCOL,validate_bundle,read_strings

CONTEXT=Path('/home/zwan0726/work/context')
TARGETS=[CONTEXT/name for name in ('README.md','01_project_context.md','02_paths_and_environment.md',
          '03_experiment_matrix.md','04_metrics_framework.md','05_decisions_and_protocols.md',
          '06_paper_evaluation_protocol_20261008.md','07_evaluation_dual_annotation_20261013.md')]
MARKER='REP1_BT_GT_FUSION_20261016'


def write_bytes(path,data):atomic(path,lambda f:f.write(data))


def analysis_review(cfg):
    path=within(cfg['run_root'],'results_analysis.json');r=read(path)
    if r.get('dependencies')!=dependencies(cfg,'RESULTS_ANALYSIS') or r.get('reviewer')!='primary':raise ValueError('results analysis binding/reviewer mismatch')
    expected=identity(within(cfg['run_root'],'evaluation8/complete.json'))
    if r.get('evaluation8')!=expected:raise ValueError('results analysis refers to another evaluation')
    for key in ('dual_F1_parallel','negative_results_retained','local_RNA_guardrails_reviewed','no_owner_truth_claim'):
        if r.get(key) is not True:raise ValueError('results analysis incomplete: '+key)
    if r.get('comparisons_reviewed')!=[list(x) for x in comparisons()]:raise ValueError('results analysis comparison coverage incomplete')
    for key in ('selection_summary','local_RNA_explanation'):
        if not isinstance(r.get(key),str) or not r[key].strip():raise ValueError('results analysis lacks '+key)
    return r


def context_block(cfg,rows,registry,analysis):
    by={r['method']:r for r in rows}
    lines=['<!-- '+MARKER+'_BEGIN -->',
           '**Rep1 B/T + GT 融合：生产、八端点评价与30方法登记已验收。**',
           '协议 `'+PROTOCOL+'`；30个方法条目，实际唯一 mask 内容数 '+str(registry['unique_mask_contents'])+'。',
           'fixed 与 scClassify 正/负 F1 并列；P↑、N↓。下方历史记录原样保留。','',
           '| 方法 | sc P↑ | sc N↓ | fixed P↑ | fixed N↓ | capture % |',
           '|---|---:|---:|---:|---:|---:|']
    differences=[]
    for name in NEW:
        row=by[name];lines.append('| '+' | '.join([name]+[row[k] for k in ('scclassify_positive_F1','scclassify_negative_F1','fixed_positive_F1','fixed_negative_F1','capture_pct')])+' |')
        parent=by[PARENTS[name]]
        values=[]
        for key in ('scclassify_positive_F1','scclassify_negative_F1','fixed_positive_F1','fixed_negative_F1'):
            try:value=float(row[key])-float(parent[key])
            except (ValueError,TypeError):value=float('nan')
            values.append(format(value,'.12g') if math.isfinite(value) else 'NA（原值不可用）')
        differences.append(name+' 相对 '+PARENTS[name]+' 的四项原始差值（sc P/N、fixed P/N）：'+', '.join(values)+'。')
    lines += ['']+differences+['',analysis['selection_summary'],'',analysis['local_RNA_explanation'],'',
              'GT 指 Geometry+Type，评分不使用 cell-specific q，但保留原 A1/type/donor 资格。',
              'capture 0.5pp 与 local retention .995 护栏未放宽；局部 RNA 转移的全部报警及负结果保留在评价和转移账中。',
              'B0 为训练历史不配对的历史参照，BASE_MATCH 为机制对照。单 Rep1、无独立膜/人工 owner 真值，不能声称真实边界准确率。',
              '[30方法主表]('+str(RESULT_ROOT/'main_table.csv')+') · [artifact registry]('+str(RESULT_ROOT/'artifact_registry.json')+') · [八端点评价]('+str(Path(cfg['run_root'])/'evaluation8/complete.json')+')',
              '[同步与最终验收]('+str(RESULT_ROOT/'context_sync_complete.json')+')',
              '<!-- '+MARKER+'_END -->','']
    return '\n'.join(lines).encode()


def commit_append_journal(journal_root,updates,binding=None):
    """Recover partially installed appends without rewriting any prior bytes."""
    root=Path(journal_root);manifest=root/'journal.json'
    if manifest.exists():
        journal=read(manifest)
        if journal.get('binding')!=binding:raise ValueError('context append dependency drift')
    else:
        root.mkdir(parents=True,exist_ok=True);entries=[]
        for i,(target,after) in enumerate(updates):
            target=Path(target);before=target.read_bytes() if target.exists() else None
            if before is not None:
                if before not in after:raise ValueError('context history must remain byte-for-byte in the new version')
                saved=root/(str(i)+'.before')
                if saved.exists() and saved.read_bytes()!=before:raise ValueError('uncommitted context journal drift')
                write_bytes(saved,before)
            saved=root/(str(i)+'.after')
            if saved.exists() and saved.read_bytes()!=after:raise ValueError('uncommitted context plan drift')
            write_bytes(saved,after)
            entries.append(dict(target=str(target),before=identity(root/(str(i)+'.before')) if before is not None else None,
                                after=identity(root/(str(i)+'.after'))))
        journal=dict(schema='rep1-bt-gt-context-journal-v1',binding=binding,entries=entries)
        atomic_json(manifest,journal)
    for entry in journal['entries']:
        for k in ('before','after'):
            if entry[k]:check_identity(entry[k])
        target=Path(entry['target']);current=sha256(target) if target.exists() else None
        if current==entry['after']['sha256']:continue
        expected=entry['before']['sha256'] if entry['before'] else None
        if current!=expected:raise ValueError('context changed outside the append journal: '+str(target))
        write_bytes(target,Path(entry['after']['path']).read_bytes())
    outputs={e['target']:identity(e['target']) for e in journal['entries']}
    return dict(journal=identity(manifest),outputs=outputs)


def synchronize(cfg,grant,publication):
    require_manual_resume(cfg,grant);validate_bundle(RESULT_ROOT)
    check_receipt(within(cfg['run_root'],'evaluation8/complete.json'),dependencies(cfg,'EVALUATION8'),cfg['run_root'])
    if publication.get('published_root')!=str(RESULT_ROOT) or not publication.get('old28_unchanged'):raise ValueError('publication proof missing')
    for obj in publication['outputs'].values():check_identity(obj)
    columns,rows=read_strings(RESULT_ROOT/'main_table.csv');registry=read(RESULT_ROOT/'artifact_registry.json')
    analysis=analysis_review(cfg);block=context_block(cfg,rows,registry,analysis)
    binding=dict(dependencies(cfg,'CONTEXT_SYNC'),analysis=identity(within(cfg['run_root'],'results_analysis.json')),
                 table=identity(RESULT_ROOT/'main_table.csv'),registry=identity(RESULT_ROOT/'artifact_registry.json'),grant=identity(grant))
    journal=within(cfg['run_root'],'context_append_journal')
    if not (journal/'journal.json').exists():
        updates=[]
        for target in TARGETS:
            previous=target.read_bytes()
            if MARKER.encode() in previous:raise ValueError('untracked existing context version marker')
            updates.append((target,block+b'\n'+previous))
        experiment=CONTEXT/'experiments/rep1_bt_gt_fusion_20261016.md'
        if experiment.exists():raise FileExistsError('untracked existing experiment record')
        updates.append((experiment,b'# Rep1 B/T + GT fusion\n\n'+block+b'\nSource commit: '+cfg['source_commit'].encode()+b'\nRun: '+cfg['run_root'].encode()+b'\n'))
        target=CONTEXT/'experiment_registry.csv';previous=target.read_bytes()
        reader=csv.DictReader(io.StringIO(previous.decode('utf-8-sig')));fields=reader.fieldnames;old=list(reader)
        if any(r['experiment_id'].startswith('BTGT-20261016-') for r in old):raise ValueError('untracked registry duplicate')
        out=io.StringIO(newline='');writer=csv.DictWriter(out,fieldnames=fields,lineterminator='\r\n')
        for name in NEW:
            writer.writerow(dict(experiment_id='BTGT-20261016-'+name,logical_method=name,status='COMPLETED',
                                 scope_reason='Rep1_GT_postprocessing_own_matrix_dual_F1_BT;negative_results_retained',
                                 evidence_root=str(Path(cfg['run_root'])/'endpoints'/name)))
        updates.append((target,previous+out.getvalue().encode()))
        proof=commit_append_journal(journal,updates,binding)
    else:proof=commit_append_journal(journal,[],binding)
    for target in TARGETS:
        text=target.read_bytes()
        if not text.startswith(block) or text.count(('<!-- '+MARKER+'_BEGIN -->').encode())!=1:raise ValueError('context version coverage mismatch')
    _,registered=read_strings(CONTEXT/'experiment_registry.csv')
    if sum(r['experiment_id'].startswith('BTGT-20261016-') for r in registered)!=2:raise ValueError('experiment registry coverage mismatch')
    result=dict(status='succeeded',dependencies=dependencies(cfg,'FINAL_REGISTRATION'),manual_resume=identity(grant),
                publication=identity(RESULT_ROOT/'publication_complete.json'),artifact_registry=identity(RESULT_ROOT/'artifact_registry.json'),
                results_analysis=identity(within(cfg['run_root'],'results_analysis.json')),
                table=identity(RESULT_ROOT/'main_table.csv'),methods=30,new_methods=list(NEW),unique_mask_contents=registry['unique_mask_contents'],
                old28_raw_preserved=True,context_sync_complete=True,**proof)
    atomic_json(RESULT_ROOT/'context_sync_complete.json',result)
    for obj in result['outputs'].values():check_identity(obj)
    validate_bundle(RESULT_ROOT)
    atomic_json(within(cfg['run_root'],'final_acceptance.json'),dict(result,context_receipt=identity(RESULT_ROOT/'context_sync_complete.json')))
    return result
