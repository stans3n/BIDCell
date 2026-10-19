"""Twelve strictly ordered computation stages, no autonomous result interpretation."""
from pathlib import Path
import csv,json,os,shutil,time,zipfile
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from scripts.rep1_triple_simple_boundary.common import atomic_json,atomic_csv,identity,check_identity,read,digest
from scripts.rep1_triple_simple_boundary.inputs import matrix_arrays
from .config import R05,REGISTRY,NAME,MARKERS,STATES,FATES,artifact,beta_spec
from . import local,compute

def export(src,dst):
    dst=Path(dst)
    if dst.exists():
        if identity(src)['sha256']==identity(dst)['sha256']:return
        raise FileExistsError('stage attempted overwrite: '+str(dst))
    shutil.copyfile(src,dst)

def runroot(cfg):return Path(cfg['run_dir'])
def stage(cfg,n):return runroot(cfg)/'stages'/f'{n:02d}'

def preflight(cfg,progress):
    root=runroot(cfg);old=cfg['old_config'];inputs=cfg['inputs']
    receipt=read(R05/'compute_complete.json')
    history_state=read(R05/'state.json')
    if history_state['completed']!=15 or history_state['total']!=15 or history_state['exit_code']!=0:raise ValueError('r05 15/15 receipt not complete')
    if receipt.get('status')!='succeeded' or receipt.get('comparisons')!=26 or len(receipt.get('methods',[]))!=12:raise ValueError('r05 incomplete')
    for rec in receipt['outputs'].values():check_identity(rec)
    # Same cached row universe and exact historical interface geometry, locked before new metrics.
    den=pd.read_csv(Path(old['domain_manifest']).parent/'denominators.csv')
    interface=int(den[(den.region=='interface')&(den.unit=='molecule')].denominator.iloc[0])
    if interface!=cfg['interface_expected'] or inputs['coordinate_cache_validation']['rows']!=cfg['eligible']:raise ValueError('source universe differs from expected contract')
    sources=[identity(R05/'compute_complete.json'),identity(old['input_manifest']),identity(old['domain_manifest']),identity(inputs['fixed_population']),identity(inputs['metadata']),identity(inputs['reference_region']),identity(REGISTRY/'main_table.csv')]
    cursor=0
    for i,rec in enumerate(inputs['coordinates']):
        check_identity(rec);co=np.load(rec['path'],mmap_mode='r')
        if co.dtype.names!=('transcript_id','x','y','gene') or rec['row_start']!=cursor or rec['row_stop']!=cursor+len(co):raise ValueError('coordinate IDs/row spans invalid')
        cursor+=len(co);sources.append(rec);progress('lock_coordinates',i+1,len(inputs['coordinates']))
    if cursor!=cfg['eligible']:raise ValueError('raw universe coverage')
    for name,spec in cfg['specs'].items():
        for key in ('final','machine','fixed_scores','sc_scores'):sources.append(identity(spec[key]))
        for leaf in ('counts.npy','cell_mapping.csv','gene_mapping.csv','complete.json'):sources.append(identity(Path(spec['matrix'])/leaf))
    legacy=Path(cfg['specs']['P2_MORPH_GT']['root'])/'gt'
    for leaf in ('domain/domain.npz','domain/expression.npz','domain/complete.json','scores/proposals.npy','scores/candidate_scores.npy','recovery/state.json'):sources.append(identity(legacy/leaf))
    frozen=Path(old['domain_manifest']).parent
    for leaf in ('cells.csv','interfaces/assignment.npy','interfaces/pairs.csv','region_bits.npy'):sources.append(identity(frozen/leaf))
    for sub,pattern in (('molecule_strata','chunk_*.npz'),('interfaces','exposure_*.npz')):
        sources.extend(identity(p) for p in sorted((frozen/sub).glob(pattern)))
    ref=read(inputs['reference_region']);machine=read(inputs['endpoints']['BASE_MATCH']['machine']);genes=Path(machine['genes']).read_text().splitlines()
    panel=[]
    depdir=Path(ref['source']['path'] if isinstance(ref['source'],dict) else ref['source']).parent if 'source' in ref else Path('/home/zwan0726/work/projects/BIDCell/example_data/sc_references')
    evalmarkers=read(Path(inputs['evaluation']['prepared'])/'markers.json')
    loss={}
    for kind,filename in [('positive','sc_breast_markers_pos.csv'),('negative','sc_breast_markers_neg.csv')]:
        path=depdir/filename
        if path.exists():
            sources.append(identity(path));loss[kind]=pd.read_csv(path,index_col=0)
    for g,lineage in MARKERS.items():
        nonzero={}
        for kind,t in loss.items():
            if g in t.index:nonzero[kind]=[str(c) for c in t.columns if pd.to_numeric(pd.Series([t.loc[g,c]]),errors='coerce').fillna(0).iloc[0]!=0]
            elif g in t.columns:nonzero[kind]=[str(x) for x in t.index[pd.to_numeric(t[g],errors='coerce').fillna(0)!=0]]
        ep={kind:[t for t,v in values.items() if g in v] for kind,values in evalmarkers.items() if isinstance(values,dict)}
        panel.append(dict(gene=g,lineage='B' if lineage==1 else 'T',in_313_type_prior=g in ref['gene_names'],prior_types=len(ref['type_names']),loss_nonzero=json.dumps(nonzero),evaluation_marker_sets=json.dumps(ep),labels='frozen B1-derived scClassify',dependency='REFERENCE-CONSISTENCY PROXY',panel_origin='existing bt.PANELS; BIDCell Fig3e T markers, reciprocal historical B markers'))
    sources.extend(identity(Path(inputs['evaluation']['prepared'])/p) for p in ('markers.json','markers.rds','reference_metadata.rds','reference_mean_logcounts.csv','reference_detection_proportions.csv','reference_genes.txt','R_packages.csv','scClassify_model.rds'))
    sources.extend([identity(inputs['evaluation']['protocol']),identity(machine['genes']),identity(machine['baseline_config']),identity(machine['preannotation']),identity(machine['transcripts']),identity(machine['nuclei_mask']),ref['source']])
    unique={p['path']:p for p in sources};sources=list(unique.values())
    lock=dict(schema='boundary-goal-v3-lock',one_time_configuration_lock_not_preregistration=True,raw_RNA_rows=cursor,interface_RNA_rows=interface,ROI='historical BASE_MATCH positive-positive 8-neighbour interface <=2um; nucleus excluded; includes nearby BASE background',unique_interface='minimum distance_squared then (cell_i,cell_j), historical assignment',boundary_bins=['[0,1]','(1,2]'],nearest_T_bins=['[0,10]','(10,20]','(20,30]','(30,40]','(40,50]','(50,100]','(100,observed_max]'],nearest_geometry='fixed nuclear centroids; ownQC descriptive uses own centroids',markers=MARKERS,states=list(STATES),fates=list(FATES),matching='exact same-ID complete nuclear pixels; any violation fails affected analysis',CI='paired fixed spatial 1000um blocks, percentile 95%, 2000 draws, seed20261019',exit_denominators='ON_TARGET and OFF_TARGET each use their own original host risk set',beta_only_change='score coefficient 1 -> 0; reference gates retained',BLCR='NOT_DEFINED / NOT_APPLICABLE; optional custom candidate, no invented formula',fixed_roi_off_target_fraction='N_opposite / D_fixed_marker_RNA; literature-inspired local adaptation, not an original-paper standard metric',source_files=sources,source_code_digest=cfg['source_digest'])
    atomic_json(root/'EVALUATION_LOCK.json',lock);atomic_csv(root/'INPUT_HASHES.csv',pd.DataFrame(sources));atomic_csv(root/'MARKER_PANEL_AND_DEPENDENCY.csv',pd.DataFrame(panel))
    (root/'ROI_AND_REFERENCE_PROVENANCE.md').write_text('固定 reference、panel、ROI 和分母已一次性锁定。全球 RNA D=34,442,716；interface D=7,541,976。界面范围沿用 r05 BASE_MATCH 正 owner 邻接并扩张≤2µm、排除核，包含扩张区域的 background pixels；唯一 interface assignment 防止重复。0–1 和1–2µm基于边界端点 raster 欧式最短距离，与 nearest T 核质心距离分开。\n\n六 marker 共享313gene/15type prior 和 B1-derived fixed labels：REFERENCE-CONSISTENCY PROXY。可报告指标和同父消融，不能据此宣称真实 ownership accuracy 或跨患者泛化。单Rep1的空间CI只描述切片内部不确定性。BLCR历史公式未在既有代码/文档中找到，记为 NOT_DEFINED / NOT_APPLICABLE，不影响任何核心实验完成状态。\n')
    (root/'ETA_ESTIMATE.md').write_text('按历史同型 endpoint 阶段估算：r01 score5.10min，topology5.47min，validation1.22min，matrix含oracle1.98min，evaluate42.52min；r05同型score6.62–7.17min/topology7.01–12.02min/validate1.94–2.04min/matrix2.9–3.1min/evaluate55.0–55.3min。前者时间差由完成产物mtime、matrix/evaluate由receipt wall_seconds取得。复用已冻结domain，两次score/topology+beta0 matrix/evaluate约1–2h；新增local/CI/hash/打包后暂估2–4h，单线程R及I/O可能延长，非保证。未用17.2h/15线性估算。\n')
    return [root/p for p in ('EVALUATION_LOCK.json','INPUT_HASHES.csv','MARKER_PANEL_AND_DEPENDENCY.csv','ROI_AND_REFERENCE_PROVENANCE.md','ETA_ESTIMATE.md')]

def global_metrics(cfg,progress):
    root=runroot(cfg);out=stage(cfg,1);oldtable=pd.read_csv(REGISTRY/'main_table.csv');rows=[];inputs=cfg['inputs'];frozen=Path(cfg['old_config']['domain_manifest']).parent
    for k,(name,spec) in enumerate(cfg['specs'].items()):
        ids,genes,counts=matrix_arrays(spec);total=np.asarray(counts.sum(1));detected=(counts>0).sum(1);qc=(total>=20)&(total<=2000)&(detected>=20)
        before=oldtable[oldtable.method==name];row=before.iloc[0].to_dict() if len(before)==1 else dict(method=name)
        row.update(method=name,n_segmented_cells=len(ids),assigned_RNA=int(total.sum()),eligible_RNA_D=cfg['eligible'],capture_pct=100*total.sum()/cfg['eligible'],scclassify_qc_cells=int(qc.sum()),scclassify_qc_rejected=int((~qc).sum()),mean_transcripts_per_cell=float(total.mean()),median_transcripts_per_cell=float(np.median(total)),mean_genes_per_cell=float(detected.mean()),median_genes_per_cell=float(np.median(detected)),matrix_path=spec['matrix'],matrix_source_sha256=identity(Path(spec['matrix'])/'complete.json')['sha256'])
        for profile,key in [('fixed','fixed_scores'),('scclassify','sc_scores')]:
            f=pd.read_csv(spec[key])
            for metric in ('positive_F1','negative_F1'):
                v=f[metric].to_numpy(float);ok=np.isfinite(v);row[profile+'_'+metric]=float(v[ok].mean()) if ok.any() else None;row[profile+'_'+metric+'_n']=int(ok.sum());row[profile+'_population_n']=len(v)
        if name in cfg['inputs']['endpoints']:sp=R05/'reused'/name/'structure/structure.json'
        else:sp=Path(spec['evaluation'])/'structure/structure.json'
        if sp.exists():row.update(read(sp));row['structure_source']=str(sp)
        elif name in ('B0','B1'):
            mask=tifffile.imread(spec['final']);boxes=ndi.find_objects(mask);d4=d8=0
            for cid in ids:
                a=mask[boxes[int(cid)-1]]==cid;d4+=ndi.label(a,ndi.generate_binary_structure(2,1))[1]>1;d8+=ndi.label(a,np.ones((3,3),bool))[1]>1
            row.update(disconnected_cells_4=int(d4),disconnected_cells_8=int(d8),structure_source='new exact full ID D4/D8 from locked existing raster')
            atomic_json(out/(name+'_topology.json'),dict(disconnected_cells_4=int(d4),disconnected_cells_8=int(d8),source=identity(spec['final'])));del mask
        cp=R05/'analysis/all12/boundary/methods'/name/'capture_foreground.csv'
        if name in ('B0','B1'):
            mask=tifffile.imread(spec['final']);assigned=den=0
            for i,rec in enumerate(inputs['coordinates']):
                co=np.load(rec['path'],mmap_mode='r');st=np.load(frozen/'molecule_strata'/f'chunk_{i:04d}.npz');use=local.in_region(st['region_bits'],'interface');good=use&(co['x']>=0);den+=int(use.sum());assigned+=int(np.count_nonzero(mask[co['y'][good],co['x'][good]]))
            del mask
        else:
            c=pd.read_csv(artifact(cp));c=c[(c.region=='interface')&(c.unit=='molecule')].iloc[0];den=int(c.denominator);assigned=int(c.numerator)
        if den!=cfg['interface_expected']:raise ValueError('global reuse interface D mismatch')
        row.update(interface_assigned_RNA=assigned,interface_raw_RNA_D=den,interface_capture_pct=100*assigned/den,identity_role='primary baseline' if name=='B0' else '4000 additional updates; unmatched training' if name=='B1' else 'matched training diagnostic' if name=='BASE_MATCH' else 'candidate/control')
        rows.append(row);progress('reuse_global',k+1,len(cfg['specs']))
    atomic_csv(root/'EXISTING_METHODS_MASTER_METRICS.csv',pd.DataFrame(rows));return [root/'EXISTING_METHODS_MASTER_METRICS.csv',*out.glob('*_topology.json')]

def local_curves(cfg,progress):
    out=stage(cfg,2);ctx=local.context(cfg,out);specs={m:cfg['specs'][m] for m in ('B0','P2_MORPH','P2_MORPH_GT')};local.distance_curves(cfg,specs,out,ctx)
    return list(out.glob('*'))

def transfers(cfg,progress):
    out=stage(cfg,3);ctx=local.context(cfg,stage(cfg,2));specs={m:cfg['specs'][m] for m in ('B0','P2_MORPH','P2_MORPH_GT')};pairs=[('P2_MORPH','P2_MORPH_GT'),('B0','P2_MORPH_GT')];local.stream(cfg,specs,pairs,out,ctx,progress);local.summaries(out);return list(out.glob('*'))

def initial_ci(cfg,progress):
    # CI file has its own stage receipt; source sufficient statistics stay immutable.
    out=stage(cfg,4);ctx=local.context(cfg,stage(cfg,2));p=stage(cfg,3);tmp=out/'stats';tmp.mkdir(exist_ok=True);shutil.copyfile(p/'block_sums.npz',tmp/'block_sums.npz');shutil.copyfile(p/'schema.json',tmp/'schema.json');local.intervals(cfg,tmp,ctx);return list(tmp.glob('*'))

def beta_local(cfg,progress):
    out=stage(cfg,9);ctx=local.context(cfg,stage(cfg,2));specs={m:cfg['specs'][m] for m in ('P2_MORPH','P2_MORPH_GT')};specs[NAME]=beta_spec(cfg);pairs=[('P2_MORPH',NAME),('P2_MORPH','P2_MORPH_GT'),(NAME,'P2_MORPH_GT')];local.distance_curves(cfg,specs,out,ctx);local.stream(cfg,specs,pairs,out,ctx,progress);local.summaries(out);local.intervals(cfg,out,ctx);score_contrasts(cfg,specs,pairs,out,ctx);return list(out.glob('*'))

def score_contrasts(cfg,specs,pairs,out,ctx):
    rows=[];n=int(ctx['cells'].block_1000.max())+1;rng=np.random.default_rng(cfg['seed']);W=np.vstack([np.bincount(rng.integers(n,size=n),minlength=n) for _ in range(cfg['bootstrap_replicates'])])
    for a,b in pairs:
        tables={m:{p:pd.read_csv(specs[m][p+'_scores']).set_index('cell_id') for p in ('fixed','sc')} for m in (a,b)}
        common=np.intersect1d(tables[a]['sc'].index,tables[b]['sc'].index);common=np.intersect1d(common,ctx['fixed'].index)
        for metric in ('positive_F1','negative_F1'):
            selected=common.copy()
            for m in (a,b):
                for p in ('fixed','sc'):selected=selected[np.isfinite(tables[m][p][metric].reindex(selected))]
            for p in ('fixed','sc'):
                x=tables[a][p].loc[selected,metric].to_numpy(float);y=tables[b][p].loc[selected,metric].to_numpy(float);bi=ctx['cells'].loc[selected,'block_1000'].to_numpy();s=np.bincount(bi,weights=y-x,minlength=n);den=np.bincount(bi,minlength=n);draw=np.divide(W@s,W@den,out=np.full(len(W),np.nan),where=(W@den)>0);good=np.isfinite(draw);lo,hi=np.quantile(draw[good],[.025,.975]) if good.any() else (np.nan,np.nan)
                rows.append(dict(source=a,target=b,profile=p,metric=metric,population='pairwise commonQC four-way finite fixed IDs',n=len(selected),source_mean=x.mean() if len(x) else np.nan,target_mean=y.mean() if len(y) else np.nan,delta=(y-x).mean() if len(x) else np.nan,ci_low=lo,ci_high=hi,blocks=n,block_um=1000))
    atomic_csv(Path(out)/'paired_global_F1.csv',pd.DataFrame(rows))

def aggregate(cfg,progress,partial=False):
    root=runroot(cfg);a=stage(cfg,3);mass=pd.read_csv(a/'mass.csv');flows=pd.read_csv(a/'flows.csv');risk=pd.read_csv(a/'risk.csv');ci=pd.read_csv(stage(cfg,4)/'stats/intervals.csv');curves=pd.read_csv(stage(cfg,2)/'distance.csv')
    if not partial:
        b=stage(cfg,9);mass=pd.concat([mass,pd.read_csv(b/'mass.csv')],ignore_index=True).drop_duplicates(['method','region','state']);flows=pd.concat([flows,pd.read_csv(b/'flows.csv')],ignore_index=True).drop_duplicates();risk=pd.concat([risk,pd.read_csv(b/'risk.csv')],ignore_index=True).drop_duplicates(['method','region']);ci=pd.concat([ci,pd.read_csv(b/'intervals.csv')],ignore_index=True).drop_duplicates();curves=pd.concat([curves,pd.read_csv(b/'distance.csv')],ignore_index=True).drop_duplicates(['method','population','direction','marker','bin_index'])
        # Global/local G-only table is written once below; interrupted aggregation may resume.
    for name,frame in [('BOUNDARY_LOCAL_MARKER_RESULTS.csv',mass),('MARKER_AWARE_TRANSFERS.csv',flows),('ROI_RISKSET_COUNTS.csv',risk),('BOUNDARY_SPATIAL_INTERVALS.csv',ci),('BIDCELL_STYLE_DISTANCE_CURVE.csv',curves),('FIGURE_DATA.csv',curves)]:atomic_csv(root/name,frame)
    export(a/'correspondence.csv',root/'METHOD_CORRESPONDENCE_AND_COVERAGE.csv')
    pairs=[('B0','P2_MORPH_GT'),('P2_MORPH','P2_MORPH_GT')]+([] if partial else [('P2_MORPH',NAME),(NAME,'P2_MORPH_GT')]);contrasts=[]
    for s,t in pairs:
        left=mass[mass.method==s].set_index(['region','state']);right=mass[mass.method==t].set_index(['region','state'])
        for key in left.index:
            x=left.loc[key];y=right.loc[key]
            if int(x.fixed_raw_marker_D)!=int(y.fixed_raw_marker_D):raise ValueError('different fixed raw marker denominator')
            contrasts.append(dict(source=s,target=t,region=key[0],state=key[1],metric={'OPPOSITE':'fixed_roi_off_target_fraction_delta','CONSISTENT':'fixed_roi_consistent_fraction_delta','OTHER_OR_UNMATCHED':'fixed_roi_other_unmatched_fraction_delta','UNASSIGNED':'fixed_roi_unassigned_fraction_delta'}[key[1]],source_count=int(x['count']),target_count=int(y['count']),delta_count=int(y['count']-x['count']),fixed_raw_marker_D=int(x.fixed_raw_marker_D),delta_fraction=y.fraction-x.fraction,source_capture=x.capture,target_capture=y.capture,capture_delta=y.capture-x.capture))
    comp=pd.DataFrame(contrasts);atomic_csv(root/'ALL_LOCAL_CONTRASTS.csv',comp);atomic_csv(root/'B0_FINAL_LOCAL_COMPARISON.csv',comp[comp.source=='B0'])
    if not partial:
        paired=pd.read_csv(stage(cfg,9)/'paired_global_F1.csv');paired['metric_domain']='global_F1';lc=comp[(comp.source==NAME)|(comp.target==NAME)].copy();lc['metric_domain']='local_mass_balance';lci=ci[(ci.source==NAME)|(ci.target==NAME)].copy();lci['metric_domain']='local_spatial_CI';atomic_csv(root/'G_ONLY_MATCHED_RESULTS.csv',pd.concat([paired,lc,lci],ignore_index=True))
    off=comp[comp.state=='OPPOSITE'].copy();off['source_fixed_roi_off_target_fraction']=off.source_count/off.fixed_raw_marker_D;off['target_fixed_roi_off_target_fraction']=off.target_count/off.fixed_raw_marker_D;off['fixed_roi_off_target_fraction_delta']=off.delta_fraction;off['metric_provenance']='literature-inspired local adaptation; not an original-paper standard metric';off=off.merge(ci[ci.metric=='fixed_roi_off_target_fraction_delta'],on=['source','target','region','metric'],how='left');atomic_csv(root/'FIXED_ROI_OFF_TARGET_RESULTS.csv',off)
    # Validation is denominator/conservation based, never sign based.
    for (m,r),part in mass.groupby(['method','region']):
        if set(part.state)!=set(STATES) or part.fixed_raw_marker_D.nunique()!=1 or int(part['count'].sum())!=int(part.fixed_raw_marker_D.iloc[0]):raise ValueError('four-state mass balance failed')
    f=flows[flows.kind=='RECIPIENT_FATE']
    for key,part in f.groupby(['source','target','region']):
        if set(part.fate)!=set(FATES) or int(part['count'].sum())!=int(part.all_transferred.iloc[0]):raise ValueError('fates fail full transfer denominator')
    per_gene=pd.read_csv(a/'per_gene.csv')
    if not partial:per_gene=pd.concat([per_gene,pd.read_csv(stage(cfg,9)/'per_gene.csv')],ignore_index=True).drop_duplicates()
    atomic_csv(root/'MARKER_AWARE_TRANSFERS_PER_GENE.csv',per_gene)
    summary=dict(state='PARTIAL_BLOCKED' if partial else 'COMPUTED_AWAITING_USER_RESUME',raw_global_D=cfg['eligible'],interface_D=cfg['interface_expected'],methods=sorted(mass.method.unique()),BLCR_status='NOT_DEFINED / NOT_APPLICABLE',fixed_roi_metric='fixed_roi_off_target_fraction = N_opposite / D_fixed_marker_RNA',beta0_complete=not partial,scientific_interpretation='deferred until user manual Goal resume',limitations=['REFERENCE-CONSISTENCY PROXY','single Rep1 spatial CI is within-section'],mass_balance_all_pass=True,recipient_fates_all_pass=True,canonical_table_updated=False,context_updated=False)
    if not partial:
        spec=beta_spec(cfg);summary['beta0_matrix']=dict(**identity(Path(spec['matrix'])/'counts.npy'),cells=107955,genes=313,matrix_receipt=identity(Path(spec['matrix'])/'complete.json'),csv=identity(Path(spec['matrix'])/'expr_mat.csv'));atomic_json(root/'BETA0_CELL_BY_GENE_MATRIX.json',summary['beta0_matrix'])
        g=read(stage(cfg,8)/'complete.json');old=pd.read_csv(root/'EXISTING_METHODS_MASTER_METRICS.csv');row={k:v for k,v in g.items() if k not in ('status','method','structure','fixed','capture')};row.update(g['structure']);row.update(method=NAME,identity_role='reference-gated geometry-only scoring',assigned_RNA=g['capture']['assigned'],eligible_RNA_D=cfg['eligible'],capture_pct=100*g['capture']['assigned']/cfg['eligible'])
        mm=mass[(mass.method==NAME)&(mass.region=='INTERFACE_0_2')].iloc[0];row.update(interface_assigned_RNA=int(mm.captured_all_gene),interface_raw_RNA_D=int(mm.fixed_raw_all_gene_D),interface_capture_pct=100*mm.capture)
        atomic_csv(root/'ALL_METHODS_MASTER_METRICS.csv',pd.concat([old,pd.DataFrame([row])],ignore_index=True));summary['beta0_evaluation']=g
    atomic_json(root/'RESULTS_SUMMARY.json',summary)
    (root/'RESULT_VALIDATION.md').write_text('PASS：固定RNA universe、四状态mass balance、完整recipient fate分母、所有同父/跨方法逐cell/gene的matrix delta、核ID一对一对应均已程序断言。'+(' β1 full discrete regression和β0独立matrix/oracle/full四F1评价完成。' if not partial else ' β1回归失败，β0支路BLOCKED。')+' 科学结果正负方向均不作为完成关卡。未修改已发布主表、manuscript或context。\n')
    (root/'EVIDENCE_FOR_MANUSCRIPT.md').write_text('自动数字收据，等待用户恢复后的科学分析。\n\n四F1/global capture/QC/D4D8见 EXISTING_METHODS_MASTER_METRICS.csv 和 ALL_METHODS_MASTER_METRICS.csv；BIDCell Fig3e核距离曲线见 BIDCELL_STYLE_DISTANCE_CURVE.csv；FastReseg启发式host-exit/recipient fate见 MARKER_AWARE_TRANSFERS.csv；RNA-level ≤2µm限制是本研究扩展，非原文完全复刻；空间区间见BOUNDARY_SPATIAL_INTERVALS.csv；同父消融见G_ONLY_MATCHED_RESULTS.csv与ALL_LOCAL_CONTRASTS.csv。所有结论需联合报告capture、other/unassigned去向与D4/D8代价。\n')
    (root/'PRO_HANDOFF.md').write_text('WAITING_FOR_USER_RESUME\n\n用户明确 /goal resume 后，先 check_status.sh --verify；若仍RUNNING只汇报状态。完成后派两个 Astra Max只读Agent分别分析global+B0权衡与local+β0/β1；主对话一次合并SCIENTIFIC_VERDICT.md和MANUSCRIPT_PATCH_RECOMMENDATIONS.md，亲自同步canonical主表 /home/zwan0726/work/outputs/bidcell_historical_dual_bt_gt_fusion_20261017/main_table.csv 及现有 /home/zwan0726/work/context/。不能提前分析或写回。B0主基线，B1多4000updates，BASE_MATCH训练匹配诊断。正/负/混合结果均准确报告；没有人工owner truth不否定实测F1与同父消融。\n')
    names=['FIXED_ROI_OFF_TARGET_RESULTS.csv','BOUNDARY_LOCAL_MARKER_RESULTS.csv','MARKER_AWARE_TRANSFERS.csv','MARKER_AWARE_TRANSFERS_PER_GENE.csv','ROI_RISKSET_COUNTS.csv','BOUNDARY_SPATIAL_INTERVALS.csv','BIDCELL_STYLE_DISTANCE_CURVE.csv','FIGURE_DATA.csv','METHOD_CORRESPONDENCE_AND_COVERAGE.csv','ALL_LOCAL_CONTRASTS.csv','B0_FINAL_LOCAL_COMPARISON.csv','RESULTS_SUMMARY.json','RESULT_VALIDATION.md','EVIDENCE_FOR_MANUSCRIPT.md','PRO_HANDOFF.md']
    if not partial:names+=['G_ONLY_MATCHED_RESULTS.csv','BETA0_CELL_BY_GENE_MATRIX.json','ALL_METHODS_MASTER_METRICS.csv']
    return [root/name for name in names]

def package(cfg,progress,partial=False):
    root=runroot(cfg)
    required=['RUN_MANIFEST.json','EVALUATION_LOCK.json','ROI_AND_REFERENCE_PROVENANCE.md','ETA_ESTIMATE.md','MARKER_PANEL_AND_DEPENDENCY.csv','CODE_REVIEW.md','EXISTING_METHODS_MASTER_METRICS.csv','BIDCELL_STYLE_DISTANCE_CURVE.csv','BOUNDARY_LOCAL_MARKER_RESULTS.csv','FIXED_ROI_OFF_TARGET_RESULTS.csv','MARKER_AWARE_TRANSFERS.csv','MARKER_AWARE_TRANSFERS_PER_GENE.csv','B0_FINAL_LOCAL_COMPARISON.csv','METHOD_CORRESPONDENCE_AND_COVERAGE.csv','BOUNDARY_SPATIAL_INTERVALS.csv','ROI_RISKSET_COUNTS.csv','FIGURE_DATA.csv','RESULT_VALIDATION.md','RESULTS_SUMMARY.json','EVIDENCE_FOR_MANUSCRIPT.md','PRO_HANDOFF.md','status.py','check_status.sh']
    if not partial:required += ['BETA0_CELL_BY_GENE_MATRIX.json','G_ONLY_MATCHED_RESULTS.csv','ALL_METHODS_MASTER_METRICS.csv']
    for name in required:
        p=root/name
        if not p.is_file() or p.stat().st_size==0:raise ValueError('required artifact missing '+name)
        if p.suffix=='.csv' and len(pd.read_csv(p))==0:raise ValueError('empty required table '+name)
    # Complete checksum list includes big beta0 products but excludes live status/logs.
    immutable_root=required+['INPUT_HASHES.csv','ALL_LOCAL_CONTRASTS.csv','run_boundary_goal_v3.sh']
    paths=[root/x for x in immutable_root if (root/x).is_file()]
    paths += [p for folder in ('CODE_SNAPSHOT','stages') for p in (root/folder).rglob('*') if p.is_file() and not p.name.startswith('.') and 'tmp' not in p.relative_to(root).parts]
    ids=[identity(p) for p in sorted(paths)]
    (root/'SHA256SUMS').write_text(''.join(x['sha256']+'  '+str(Path(x['path']).relative_to(root))+'\n' for x in ids))
    # A portable light bundle: all root evidence, code and stage receipts, no matrices/tensors.
    tmpzip=root/'.boundary_evidence.zip.tmp'
    with zipfile.ZipFile(tmpzip,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(root.iterdir()):
            if p.is_file() and p.name in immutable_root+['SHA256SUMS']:z.write(p,p.name)
        for folder in ('CODE_SNAPSHOT','receipts'):
            for p in sorted((root/folder).rglob('*')):
                if p.is_file():z.write(p,str(p.relative_to(root)))
    with zipfile.ZipFile(tmpzip) as z:
        if z.testzip() is not None:raise ValueError('zip checksum error')
    os.replace(tmpzip,root/'boundary_evidence.zip')
    return [root/x for x in required]+[root/'SHA256SUMS',root/'boundary_evidence.zip']
