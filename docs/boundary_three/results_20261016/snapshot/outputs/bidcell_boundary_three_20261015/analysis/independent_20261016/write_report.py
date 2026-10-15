"""Render the independently computed scientific evidence as a readable report."""
import json
from pathlib import Path
import numpy as np
import analyze as a

d=json.loads((a.OUT/'main_evidence.json').read_text())
def f(x):return 'NA' if x is None else f'{x:.9f}'
def ci(x):return '['+', '.join(f(v) for v in x)+']'
def link(label,p):return f'[{label}]({p})'
def mdtable(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])
R=a.RUN;O=a.OUT;hist=Path('/home/zwan0726/work/outputs/bidcell_historical_dual_20261013')
text=[]
text.append('''# BIDCell 三方向：独立结果分析（2026-10-16）

**审查判断：可在所述范围内分享，并须保留局部 RNA 代价、双评价分歧和单样本限制。** 本分析由 resume 后新建的独立结果分析者完成，未参与三个方向的原实现或代码审计；未重跑、修改实验、冻结门槛或正式主表。

主要结论是：A1 建立了给定已有类型先验时的 Core 核内表达增量；这没有转化为当前 A2 Core 评分的采用价值，GTC 同时劣于 GT 和 SHUFFLED。固定标签评价上，完整 Morph 的新增几何分支有清楚的条件性收益；直接 DAPI 分支的负 marker F1 收益也应明确保留。P2 在固定标签 positive F1、capture、连通性上有收益，但 scClassify negative F1 的权衡仍存在。没有已验证的三模块联合方案。

本报告按用户在结果分析期间的新要求，**并列、同等展示与解释 scClassify 和固定标签的 P↑/N↓**；冻结程序原先按 scClassify 主口径得到的 status 原样保留，称为“冻结程序判定”，不冒充新的双口径门槛。当前双口径评价/发布身份为 `DUAL-F1-v1.0-28methods-20261016`，底层冻结计分身份仍为 v1.2。后追加的 B0 为历史参照，BASE_MATCH 仍承担机制增量的匹配对照。Purity 不替代这四个 F1 端点。
''')
text.append('## 证据、独立核验与分母\n')
text.append(f"科学源码实际 HEAD 为 `ccb846cbf344bfbe965178cd8f9b738e3654c8b0`，工作目录干净；run 为 `{R.name}`。{link('官方语义完整性检查',R/'complete-check.json')}实际记录七节点全部完整、无延期/失败；本分析另从 raw CSV 和整数矩阵独立核验，而非接受 producer 的科学标签。当前计分身份为 `SC-PRIMARY-FIXED-v1.2-20methods-20261014`，QC 为完整 313 基因上的 RNA 20–2000、genes≥20 后才对齐 307 参考基因。{link('冻结共同契约','/home/zwan0726/work/docs/BIDCell_Linear_Execution_Plan_20261015.md')}；{link('评价规则','/home/zwan0726/work/context/07_evaluation_dual_annotation_20261013.md')}。\n")
text.append(f"独立脚本没有导入科学仓库 helper。逐一核验十个唯一 mask 端点的原始 integer counts/ID、RNA/genes/QC、annotation/六项 marker 的 finite 人口、matrix receipt 的分子守恒及 raw 汇总；随后从 per-cell 分数与冻结核质心块重算全部规定比较的空间 bootstrap，并与保存的 bootstrap arrays 对照（差异容差 2e−14）。A1 另从 fold counts 和 donor 表重算两方向与全部 20 shuffle。新增固定标签空间区间与 B0 参照也使用原始 per-cell 分数；B0 原始核质心与本轮冻结空间块的兼容性已核验。共 {len(json.loads((O/'checks.json').read_text()))} 项断言通过，详见 {link('核验明细',O/'checks.json')}。矩阵和关键输入的实际文件 SHA256 见 {link('source_identities.json',O/'source_identities.json')}。\n")
text.append('所有 mask 保留原始 107,955 ID；capture 分母固定 34,442,716 分子。固定标签面板 103,000 ID，F1 实际有限值为 **85,421**，其余保留 NA。scClassify 使用每端点 own-QC 人口（下表），包括作者规则下合法的零值；不把 NA 补零。A1 不是新的 mask/F1 方法；A2_NATIVE 是 BASE_MATCH 的 validated-reuse 别名。\n')
order=['BASE_MATCH','V2_MORPH','P2_BASE','P2_MORPH','MORPH_NO_DAPI','MORPH_SIMPLE_FUSION','MORPH_NO_GEO','A2_GT','A2_GTC','A2_SHUFFLED']
rows=[]
b0=d['B0_historical_reference'];br=b0['reported_scalar_row']
rows.append(['B0（历史）',f(float(br['scclassify_positive_F1'])),f(float(br['scclassify_negative_F1'])),f(float(br['fixed_positive_F1'])),f(float(br['fixed_negative_F1'])),f'{float(br["capture_pct"]):.6f}',br['scclassify_qc_cells'],br['disconnected_cells_8']])
for n in order:
 e=d['endpoints'][n];m=e['metrics'];rows.append([n,*[f(m[z]['mean']) for z in ['scclassify_positive_F1','scclassify_negative_F1','fixed_positive_F1','fixed_negative_F1']],f'{e["capture_pct"]:.6f}',e['qc_cells'],e['disconnected8']])
text.append(mdtable(['端点','sc P↑','sc N↓','fixed P↑','fixed N↓','capture %','sc QC n','8邻接不连通 n'],rows)+'\n')
text.append('表中原始 F1 不跨两种标签人口直接比较大小；两种口径各自在自己的合法人口中计算方法差值。所有精确值、六项指标 finite 分母与零值人数保存在 '+link('main_evidence.json',O/'main_evidence.json')+' 与 '+link('raw_metric_denominators.csv',O/'raw_metric_denominators.csv')+'。\n')
text.append('“突出”在本报告中指相对已注明匹配对照的增量，并非全历史集合的新绝对纪录。历史 B1 的 sc P/N=0.338736026/0.023489147、fixed=0.408445355/0.028323037，在四个 F1 方向均优于本轮十个比较端点；本轮不能声称超过完整 28 方法集合的最佳绝对 F1。历史 B1 同样不替代 BASE_MATCH 的机制对照。'+link('历史20方法原始表',hist/'main_table.csv')+'。\n')
text.append('## A1：信息增量成立，适用范围仍有限\n')
x=d['a1'];s=x['spatial']['1000']
text.append(f"主要配对人口 **88,102/107,955（81.6099%）**，可评价 88,106，48 个 1000µm 块、2,000 有效重抽样。Core−Type = **{f(x['core_minus_type'])} nats/分子**，95% CI {ci(s['core_minus_type_ci95'])}；Core−20 次匹配 shuffle 均值 = **{f(x['core_minus_shuffle'])}**，CI {ci(s['core_minus_shuffle_ci95'])}。两者在 500/2000µm 也保持正区间，支持冻结 `GO_A2`。分子加权 Core−Type 为 {f(x['molecule_weighted_gain'])}，主人口仍为细胞等权，未改用大核主导的统计量。{link('raw per-cell',R/'A1_CORE/per_cell_scores.csv')}；{link('覆盖',R/'A1_CORE/cell_coverage.csv')}。\n")
folds=x['cross_directions'];eff=x['shuffle_seed_effects']
text.append(f"独立重算 F0→F1 与 F1→F0 的各细胞留出均值增量分别为 {f(folds[0]['cell_equal_core_minus_type'])}、{f(folds[1]['cell_equal_core_minus_type'])}（每方向在其留出分母内归一；正式两方向合并则在每细胞按留出分子量加权，所以不应把这两个数直接平均当正式主值）。20 个 shuffle 的 Core 优势均为正，范围 {f(min(z['core_minus_shuffle'] for z in eff))}–{f(max(z['core_minus_shuffle'] for z in eff))}；它们不是 20 个独立生物重复。profile split cosine={f(x['split_diagnostics']['split_cosine'])}，去除经验类型均值后的 cosine={f(x['split_diagnostics']['within_type_centered_cosine'])}，说明简单 split 相似性部分由共享类型结构贡献。\n")
text.append('相同 gate 在类型内限定 A2 适用范围：B、CD8T/CD8Tex、DC、Endothelial、Epithelial、Fibroblasts、Malignant、Mono/Macro、Myofibroblasts、Plasma 共十类。CD4Tconv/Treg 的 Core−Type 仅 0.000334474，CI 跨零；Mast、NK、Neutrophils、SMC 的有效人口不足 1,000，保持 INCONCLUSIVE，不因描述性均值有利而放宽准入。'+link('type_scope 原始产物',R/'A1_CORE/type_scope.json')+'。\n')
text.append('信号不只出现在高 Core 数：paired [20,50)、[50,100)、≥100 的 Core−Type 分别为 0.125007533、0.258035130、0.448560770；同型邻居组为 0.267805004，异型邻居组为 0.231304474（邻居组允许重叠，不能相加）。但困难同型核内诊断的 3,212,786 留出分子全部已可由几何区分；因此该诊断不能支持真实边缘 owner 准确率。核外重叠覆盖 15,212,765 个 G 分子，Core/donor 完整的共同支持部分为 11,224,447（73.7831%）；GT/GTC 在其中 3,018,255 个 G 分子上产生评分赢家分歧。它只证明能改变决策，未提供正确性真值。'+link('支持与覆盖诊断',R/'A1_CORE/support_diagnostics.json')+'。\n')
text.append('## A2：Core 评分未超过其同域对照；GT 是更强的已测简化候选\n')
text.append('GT/GTC/SHUFFLED 的固定域完全相同：1,317,991 核外像素、2,940,738 G 分子，来自 BASE 正 ID 界面 2µm 带及冻结支持/相邻候选；Core 与被编辑分子不交叠。GT、GTC、SHUFFLED 分别接受 312,867、393,259、403,334 像素；结构拒绝率为 6.01%、7.97%、8.13%。所有四输出 capture 精确相同（21,494,924/34,442,716），全片前景并集/核保留约束已在运行期及完整性检查核验。'+link('domain',R/'A2_CORE/domain.json')+'；'+link('proposal/rejection',R/'A2_CORE/proposal_and_rejection_counts.csv')+'；'+link('实际结构算法',a.SRC/'scripts/boundary_three/core_refine.py')+'。\n')
def delta_table(keys):
 rows=[]
 for k in keys:
  c=d['comparisons'][k];m=c['metrics'];rows.append([k.replace('_minus_',' − '),f(m['positive_F1']['delta']),f(m['negative_F1']['delta']),f(m['positive_F1']['fixed_delta']),f(m['negative_F1']['fixed_delta'])])
 return mdtable(['比较','sc ΔP↑','sc ΔN↓','fixed ΔP↑','fixed ΔN↓'],rows)
text.append(delta_table(['A2_GT_minus_A2_NATIVE','A2_GTC_minus_A2_NATIVE','A2_GTC_minus_A2_GT','A2_GTC_minus_A2_SHUFFLED'])+'\n')
text.append('GTC−GT 在两个评价口径都恶化，且四个 1000µm CI 均处于不利方向：sc ΔP CI [−0.001393297, −0.000619463]、ΔN [0.000052968, 0.000195442]；fixed ΔP [−0.000863134, −0.000621451]、ΔN [0.000134140, 0.000242735]。GTC−SHUFFLED 也在两口径点估计全面更差，fixed 两个区间均不利。各 A2 输出的 QC ID 集完全相同、进入/退出均为 0；因此不是通过 QC 人口筛选造成的排序，反而是同群体内的真实评分变化。GTC 相对 GT 的不连通细胞也增加 66→72。**A1 信息阳性不支持当前 GTC 固定规则的特异 Core 采用价值。**\n')
text.append('GTC−native 仍有可记录的收益：fixed 两项主尺度 CI 均有利（ΔP [0.000832890, 0.001189537]；ΔN [−0.000552283, −0.000392729]），sc 的 ΔP 明确为正、ΔN CI 跨零；冻结程序为 exploratory_signal。更简的 GT−native 在 sc 和 fixed 的四个主尺度 CI 全有利，且固定标签负 F1 下降 −0.000662596 在本轮相对 native 的新端点中最突出，500/2000µm 同样有利。但 GT 的原始 Myofibroblasts、Endothelial、Mono/Macro RNA retention 分别为 0.962530、0.974211、0.977627，高 RNA 四分位为 0.992438，违反保留的局部 0.995 护栏。因此 GT 只是有进一步采用评估价值的简化候选，不能称已通过完整稳健门槛；这里的 GT−native 区间为对已测输出新增的描述性分析。\n')
text.append('尤其，GT 虽不在评分公式中使用 cell-specific q，其合法修正域仍受 A1 Core 覆盖、适用类型和 donor 可用性限制；不能将本结果概括为“已验证完全不依赖 Core 的全域方法”。当前最小已测候选是 **BASE native + 在该冻结 Core 可用域内的固定 G+T 修正**，而非未测的更宽域方法。\n')
text.append('## P2：捕获与连通性受益，双标签结论不同\n')
text.append(delta_table(['P2_BASE_minus_BASE_MATCH','P2_MORPH_minus_V2_MORPH','V2_MORPH_minus_BASE_MATCH','P2_MORPH_minus_P2_BASE'])+'\n')
text.append('BASE 上 P2 的 sc ΔP 和 ΔN 的 1000µm CI 都大于零：positive 增益明确，同时 negative 代价也明确。Morph 上相同规则的 sc 两个 CI 跨零，点估计仍是权衡。固定标签上，两个底座的 positive 增益均有正 CI，negative 点估计小幅降低但 CI 均跨零；应明确报告固定标签 positive 的收益，不应把 uncertain negative 写成已确认双赢。冻结程序仍分别记录 tradeoff_or_negative。四端点共有 ID 敏感性未消除这些主方向。\n')
text.append('P2_BASE capture +0.463087754 pp、不连通 85→23；P2_MORPH +0.392547440 pp、不连通 64→15。postconnect 的前景并集相同，随后清岛/merge 使 final 净增加分子 159,500 和 135,204：背景→细胞为 216,415/185,649，细胞→背景为 56,915/50,445。初始干预的分子在 final 保留约 99.56%，同时存在下游新增变化，不能把最终增捕全当成冲突裁决步骤自身的前景变化。'+link('BASE 阶段转移',R/'P2_BASE/postprocess_transfer.csv')+'；'+link('Morph 阶段转移',R/'P2_MORPH/postprocess_transfer.csv')+'。\n')
ip=d['p2_interaction']['positive_F1'];inn=d['p2_interaction']['negative_F1']
text.append(f"四格交互 `(P2_MORPH−MORPH)−(P2_BASE−BASE)`：sc ΔP={f(ip['delta_difference_in_differences'])}、ΔN={f(inn['delta_difference_in_differences'])}，1000µm CI 分别 {ci(ip['spatial']['1000']['ci95'])}、{ci(inn['spatial']['1000']['ci95'])}，均跨零；2000µm 两个区间转为负，应保留尺度敏感性，不能宣称协同。fixed 交互 ΔP={f(ip['fixed_delta_difference_in_differences'])}，主 CI {ci(ip['fixed_spatial']['1000']['ci95'])} 为负，支持 P2 的 positive 收益在 Morph 上小于 BASE；fixed ΔN={f(inn['fixed_delta_difference_in_differences'])}，CI {ci(inn['fixed_spatial']['1000']['ci95'])} 跨零。Morph 相对 BASE 在同 P2 条件下 fixed P/N 两项 CI 都有利，但 sc CI 跨零且有局部 RNA/结构代价。{link('完整双口径比较',O/'dual_profile_comparisons.csv')}。\n")
text.append('## Morph：新增几何在固定指标上有突出贡献，attention 必要性尚未建立\n')
text.append(delta_table(['V2_MORPH_minus_MORPH_NO_DAPI','V2_MORPH_minus_MORPH_SIMPLE_FUSION','V2_MORPH_minus_MORPH_NO_GEO','MORPH_NO_DAPI_minus_BASE_MATCH','MORPH_SIMPLE_FUSION_minus_BASE_MATCH','MORPH_NO_GEO_minus_BASE_MATCH'])+'\n')
text.append('**应突出记录 fixed 结果。** full−BASE 的 fixed ΔP=+0.000965586、ΔN=−0.000307370，两个主尺度及 500/2000µm CI 均有利；sc 是 P 增加、N 略增加的权衡。full−NO_GEO 的 fixed ΔP=+0.000921291（CI [0.000624857, 0.001213990]）、ΔN=−0.000283013（CI [−0.000426068, −0.000135645]）同样跨尺度清楚；这支持新增几何分支在保留直接 DAPI 和该 attention 结构时的条件性贡献。sc 的 full−NO_GEO 同时增加 P 和 N，不能被 fixed 结果抹去；完整模型不连通也比 NO_GEO 多 3 个。\n')
text.append('full−NO_DAPI 的 fixed ΔN=−0.000207928，主 CI [−0.000355426, −0.000044700] 且两种敏感尺度均有利；fixed ΔP 虽为正，主 CI 跨零。故直接 DAPI 分支最清楚的当前贡献是固定标签 negative F1，而非已证明所有主指标改善。删除分支还改变容量；NO_DAPI 仍使用历史核图/新增几何，不等于没有 DAPI 来源信息。NO_GEO 只删除新增几何 encoder，BASE 的 SearchArea/核图/候选等先验仍在。上述语义由实际 variant 代码和初始化映射控制。'+link('实际变体代码',a.SRC/'bidcell/model/model/morphology_variants.py')+'。\n')
text.append('simple fusion 保留图像、几何、gate、FFN 和零残差，只以两层局部卷积替换两层 cross-attention。simple−BASE 在 fixed 上两项主 CI 均有利；但 capture −0.588858904 pp 超过冻结全局 0.5 pp 损失护栏，同型/异型竞争域分别 −1.243832304/−0.962483386 pp。sc 的 N 优势还会从 own-QC −0.000012630 翻成 common-QC +0.000016634。full−simple 的 fixed 两个 1000µm CI 跨零，P 的 500/2000µm CI 为正，不能称等价或确认复杂 attention 必需。NO_GEO−BASE 的 fixed/sc 改善都很小，主要 CI 全跨零；sc 虽方向有利，其高 RNA 四分位 retention=0.965442，不能据点估计把它升级成稳健默认。\n')
text.append('训练全部为 3,625 次更新、3,998 访问、373 空访问。实际新增参数 NO_DAPI/simple/NO_GEO 为 184,712/186,736/165,432；全网注册参数分别 27,337,930/27,339,954/27,318,650。canonical full 新增参数为191,800，simple 仅少 5,064（约2.64%模块参数）。实际训练 941.10/951.57/775.03 秒，预测 4566.99/4493.16/3037.78 秒；历史 full 训练收据为902.28秒。不同调用的这些收据不是重复配对速度基准，未建立 simple 的大幅训练提速；不得把门槛预算估计当耗时。A1 469.29 秒、P2 两节点约 13760.45/14169.66 秒（含 native replay）、A2 8614.60 秒；A2 原 85,352 秒是规划估计而非本次实测。'+link('实际 timing 明细',O/'main_evidence.json')+'。\n')
text.append('## 人口变化、目标竞争域与空间不确定性\n')
text.append('A2 四输出 QC 人口完全一致，所以采用价值的主对照结论不是 QC migration。P2_BASE 进入/退出 118/36，P2_MORPH 为 100/34；NO_DAPI、simple、NO_GEO 相对 BASE 为 200/224、186/232、167/201。相同 QC ID 内的 sc ΔP/ΔN 均已单列，simple 的 N 符号反转是具体人口敏感例子。其余比较也应保留重新判型带来的标签迁移：full−BASE 的 common-QC 标签改变 11,396 个，两个 P2 分别 7,440、7,159 个。\n')
text.append('双评价分歧并非都来自人口不同。在 full/BASE 两方法×两标签都有限的同一 85,221 ID 上，sc ΔN=+0.000110321 而 fixed ΔN=−0.000303052；P2_BASE 同一 85,273 ID 上 sc ΔN=+0.000212472 而 fixed=−0.000058597。因此历史标签固定与方法自身表达重判型确实给出不同评价结论，不能只选择有利口径。固定标签自身还保留历史锚点的局限；sc 的标签变化也不是 independent owner 真值。\n')
text.append('变化确实主要发生在预定义核外竞争域：A2 的实际分子变化全部在该域；P2 两底座 final 的 changed molecules 中占比为 99.5167%/99.6041%；三条 Morph 相对 BASE 为 94.5610%/94.6347%/95.4733%。这些域由完整核支持定义，既不等同 decoder 每个 view 的可见候选，也不是已知真实生物边界。分区转移和捕获变化只支持“干预作用在目标操作域”，不能把全局 F1 的增加归因成这些分子的 owner 正确率提高。原始 domain 分母：same-type 4,002,710、heterotypic 11,206,635、unknown 3,420 分子；unknown 小组及 single-Unknown 558 分子须保留小分母限制。OOB 的合格分母为 0，因此局部 capture=NA。\n')
text.append('所有已规定 sc 与新增 fixed/B0 field intervals 使用原点 (0,0) 的 1000µm 主块、500/2000µm 敏感块，原核质心固定；统计量是 paired pooled sums/finite counts 的差，不是块均值的均值。2,000 次 PCG64 抽样，mask 指标 seed=20261013，A1 seed=20261015；各已报告区间均 2,000 次有效，达到至少 1,900 的冻结标准。LOBO min/max 与全部尺度区间保存在 '+link('sc spatial',O/'spatial_uncertainty.csv')+'、'+link('fixed spatial',O/'fixed_spatial_uncertainty.csv')+' 和 '+link('B0 spatial',O/'B0_spatial_uncertainty.csv')+'。A2_GTC/native 的 N、NO_GEO/BASE 等有 LOBO/尺度敏感性；不能只引用有利尺度。局部 adverse 组跨类型/计数/密度/多个空间尺度会重叠，不是独立受损细胞数。\n')
text.append('## B0：可作论文历史主参照，同时保留 BASE_MATCH 机制对照\n')
text.append('B0 的 raw 重评分与正式历史 20 表相符：sc P/N=0.326633914/0.025728133（n=103,299），fixed=0.399806382/0.034806660（n=85,421），capture=74.635661717%，面积均值=164.114343940µm²，不连通92。下表覆盖八个新 mask、旧 full Morph，并加 BASE_MATCH 作为背景解释。A2_NATIVE 不重复列。'+link('B0 raw 评分与来源',hist/'methods/B0/input.json')+'；'+link('B0 native 评价',Path('/home/zwan0726/work/outputs/bidcell_new_methods_20261013/baselines/B0/evaluation/complete.json'))+'。\n')
rows=[]
for name in order:
 r=d['B0_reference_comparisons'][name+'_minus_B0'];m=r['metrics'];rows.append([name,*[f(m[z]['delta']) for z in ['scclassify_positive_F1','scclassify_negative_F1','fixed_positive_F1','fixed_negative_F1']],f'{r["capture_delta_pp"]:.6f}',f'{100*r["area_mean_relative_change"]:.3f}'])
text.append(mdtable(['相对 B0','sc ΔP↑','sc ΔN↓','fixed ΔP↑','fixed ΔN↓','Δcapture pp','Δmean area %'],rows)+'\n')
text.append('相对 B0，所有上述端点的双口径 P/N 均有利，且 1000µm 四个区间均有利；这些对比可以作为论文结果的突出差值记录。但 capture 下降 11.7648–12.8168 pp、面积均值缩小 21.60–23.73%，不能省略。BASE_MATCH 本身相对 B0 已有 sc ΔP=+0.007443072、ΔN=−0.001835512、fixed ΔP=+0.002213971、ΔN=−0.003372977，以及 capture −12.227911992 pp。因此大量 B0 差值在新增机制之前就存在。B0 初始化、训练轨迹/访问顺序不匹配；用 B0 作为论文历史主参照可以回答“相对历史复现成品的表现”，无法单独证明 Core、P2 或 Morph 的增量。请把 BASE_MATCH 同时保留，避免把历史背景变化全归因于本轮模块。未针对 B0 重新定义或运行冻结门槛。\n')
text.append('## 可支持的组合与仍未测的主张\n')
text.append('当前没有足以删除权衡和局部损失的统一默认赢家。若优先核外现有前景中的固定修正，应保留 GT 作为最小已测候选，并明确其 Core 可用域依赖和局部 RNA 流失；GTC 不应因 A1 阳性被保留为必要组件。若强调固定标签表达质量，完整 Morph 及其新增几何分支的收益值得保留，不能只按 sc 的 NO_GEO 点估计删掉几何；DAPI 的固定负 marker 收益也应写出。P2 在固定 positive、捕获与连通性上值得作为明确的多指标权衡记录，而不是概括成“普遍无效”。\n')
text.append('仅 full Morph×P2 的四格已测；它没有建立跨尺度稳健协同。A2+Morph、A2+P2、三者堆叠、改顺序保护 Core、新 loss、A3、额外 seed/架构/sweep 均未测，不能把分开观察的差值相加当组合收益。相同 Core 核内分子分半并不提供独立组织验证；类型先验可能使用全核表达，核/质表达分布差异仍在。开发用 Rep1、单训练 seed、历史 full Morph 初态由冻结代码/seed 重建，以及没有独立 owner/边界参考的限制必须与结论一起保留；空间 CI 不回答跨样本或优化随机性的泛化问题。\n')
text.append('## 可复现文件\n')
text.append(f"所有新增分析只写入 `{O}`。精确的核心证据、比较、原始分母、空间区间、局部 RNA/结构检查和输入 SHA256 分别见 {link('main_evidence.json',O/'main_evidence.json')}、{link('comparisons.csv',O/'comparisons.csv')}、{link('dual_profile_comparisons.csv',O/'dual_profile_comparisons.csv')}、{link('B0_reference_comparisons.csv',O/'B0_reference_comparisons.csv')}、{link('local_guardrails.csv',O/'local_guardrails.csv')}。脚本按下列顺序重建；最后两步反映完成后明确追加的双评价与 B0 报告要求，不修改冻结实验。\n")
text.append('```bash\n'+'\n'.join(f'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/zwan0726/miniforge3/envs/bidcell_local/bin/python {O/name}'+(' --matrices' if name=='analyze.py' else '') for name in ['analyze.py','fixed_profile_analysis.py','b0_references.py','finalize_evidence.py','write_report.py'])+'\n```\n')
(O/'independent_analysis.md').write_text('\n'.join(text))
print('Written',O/'independent_analysis.md')
