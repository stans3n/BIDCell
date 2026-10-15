<!-- BOUNDARY_THREE_20261016_BEGIN -->
**2026-10-16 当前入口：三方向7节点、完整评价、独立Sol/xhigh结果分析及主对话复核完成；正式主表28个唯一mask方法。** 当前汇总协议`DUAL-F1-v1.0-28methods-20261016`。按用户本日指令，fixed与scClassify正/负F1并列评估，fixed突出结果必须注明；下方sc主/fixed辅与20/18方法段落是保留的历史政策/版本。冻结程序原判定和计分参数不改写，Purity仍只辅助。

**论文比较加入B0，并保留BASE_MATCH机制对照。** 相对B0，两套F1均改善，但capture低约12pp且B0训练历史不配对；BASE_MATCH本身已有大部分背景差异。每项相关实验都同时给B0/BASE_MATCH，不能把所有B0差值归因于新增模块。

A1有条件表达信息增量并支持A2，但GTC在两套P/N F1均劣于GT/SHUFFLED；GT只是在冻结Core可用域内的简化候选，局部RNA损失仍在。full Morph相对NO_GEO的fixed两项F1跨尺度有利、DAPI的fixed负F1贡献也保留；sc方向有权衡。P2有fixed正F1、capture与连通性收益，sc负F1代价仍在，未建立协同。simple fusion有fixed改善但capture护栏失败；没有已验证的三模块组合或统一默认赢家，未自动追加实验。

历史20原始数据/文件完整保留；新增8端点，A1无mask/F1行、A2_NATIVE仅复用别名。Purity新集合版本`PURITY-28methods-v1.0-20261016`及缩放端点单独保存。单Rep1/seed、空间CI不等于生物重复或owner真值的限制继续适用。

[本轮记录](/home/zwan0726/work/context/experiments/boundary_three_20261015.md) · [当前28主表](/home/zwan0726/work/outputs/bidcell_historical_dual_boundary_three_20261016/main_table.csv) · [含B0双口径比较](/home/zwan0726/work/outputs/bidcell_historical_dual_boundary_three_20261016/dual_profile_comparisons.csv) · [独立分析](/home/zwan0726/work/outputs/bidcell_boundary_three_20261015/analysis/independent_20261016/independent_analysis.md) · [主对话复核](/home/zwan0726/work/outputs/bidcell_boundary_three_20261015/analysis/root_review_20261016/final_review.json) · [同步验证](/home/zwan0726/work/outputs/bidcell_boundary_three_20261015/post_resume_20261016/verification.json)。
<!-- BOUNDARY_THREE_20261016_END -->

<!-- PAIRED_V2_20261014_BEGIN -->
**2026-10-14 配对实验已完成：计算、完整评价、正式20方法同步、独立Astra/Max分析C及主控复核均通过。** E0 BASE_MATCH/E2 V2_INTERACT各从冻结初态完成3625真实更新/3998访问/373空访问；E1 V2_MORPH为限定历史复用。当前协议`SC-PRIMARY-FIXED-v1.2-20methods-20261014`；下方18/14方法段落为保留的历史版本，原18原始非Purity指标未改写。

**研究结论：旧Morph对匹配基线、新interaction对旧Morph都呈正负F1权衡；完整E2对匹配基线仅为探索信号。** E1−E0 ΔP=+0.001304731、ΔN=+0.000075019；E2−E1 ΔP=−0.000079060、ΔN=−0.000149128；E2−E0 ΔP=+0.001225671、ΔN=−0.000074110，但有localized_adverse。三比较均未达到预设priority_signal，不据此替换当前默认方法或扩大架构/扫参；若继续，仅对命名的不确定性做一次预定配对复核，未在本任务自动运行。

scClassify P↑/N↓主判断，fixed仅辅助，Purity按20集合缩放且只辅助。单seed/Rep1、历史Morph初态重建和整网附加FP32 1e-6检查失败限制保留；空间bootstrap不是跨seed或owner准确率证据。

[本轮记录](/home/zwan0726/work/context/experiments/paired_v2_20261014.md) · [C独立报告](/home/zwan0726/work/outputs/bidcell_v2_paired_20261014/final_analysis/report_C.md) · [主控复核](/home/zwan0726/work/outputs/bidcell_v2_paired_20261014/post_resume/report_C_crosscheck.json) · [完整主表](/home/zwan0726/work/outputs/bidcell_historical_dual_20261013/main_table.csv) · [精简主表](/home/zwan0726/work/outputs/bidcell_historical_dual_20261013/main_table_compact.csv) · [矩阵/评价溯源](/home/zwan0726/work/outputs/bidcell_historical_dual_20261013/artifact_registry_paired_20261014.json) · [最终验收](/home/zwan0726/work/outputs/bidcell_v2_paired_20261014/final_acceptance.json)。
<!-- PAIRED_V2_20261014_END -->

<!-- SC_PRIMARY_FIXED_20261013_BEGIN -->
**2026-10-13 当前必遵守选优规则：主要依据重新 scClassify 判型后的 positive F1（越高越好）和 negative F1（越低越好）判断；Purity 用于辅助综合比较，固定标签 F1 暂不参与选优。** 此处“原始/未缩放 F1”均指重新 scClassify 判型后计算的 F1，不指固定标签 F1。不得仅凭 Purity 排名替代正负 F1 判断，也不得在主口径没有提升时改用固定标签宣布获胜。

按既有评价输出契约保留两套结果、人口/有限值人数、QC、未判型比例及配对诊断；固定标签只用于辅助解释，不作为选优依据。正负 F1 方向存在权衡时明确报告权衡。捕获、结构等指标用于说明代价和结论边界。

权威执行说明：[07_evaluation_dual_annotation_20261013.md](/home/zwan0726/work/context/07_evaluation_dual_annotation_20261013.md)，选优补充 `SC-PN-SELECTION-v1.0-20261013`。这一主辅优先级是基于论文指标的本项目规则，不冒称 BIDCell 原文明定 Purity 仅作辅助。本规则优先于下方旧选优口径；历史实验记录、原结果及计分协议身份保留。计数、QC、参考模型、分类参数及依赖身份一致时可复用已核验判型；重新计算所需评分/汇总并记录来源。
<!-- SC_PRIMARY_FIXED_20261013_END -->

<!-- THREE_ARM_FINAL_20261013_BEGIN -->
**2026-10-13 三臂正式计算及 B0/B1 重评完成，结果为 V1 阴性、M 有代价的 marker 改善。** 三臂同seed1234新鲜原主干、同3998访问/3625真实更新；18stage完整计算终态，原15个三臂stage验收后复用，未重训。B0原始34,442,716行全量独立验收及Astra High审计通过，仅补原生CSV与新版证据；原计数/owner/源码/三臂receipt保持。

当前主口径继续为scClassify，固定历史103000 ID/85421可评/17579 NA并列。R/B相对B0两口径P下降/N上升；M两口径P升/N降，但capture74.636%→62.276%、面积−23.02%，固定positive Recall微降，pooled marker仍未优于额外训练的B1。B1共同四端点sc N方向反转保留；单Rep1/seed、历史B0初始化/order未配对、无owner GT，不作为全面升级。本轮保留现有默认基线。

[实验记录](/home/zwan0726/work/context/experiments/new_methods_20261013.md) / [完整报告](/home/zwan0726/work/outputs/bidcell_new_methods_20261013/final_analysis/final_report.md) / [最终独立及管理验收](/home/zwan0726/work/outputs/bidcell_new_methods_20261013/final_acceptance.json)。计算完成与科学验收分别记录，最终科学签署状态以外置验收为准；原历史内容保留。
<!-- THREE_ARM_FINAL_20261013_END -->




<!-- BS4_AUTHOR_FINAL_20261012_BEGIN -->
**2026-10-12 BS4作者规则结果追加。** 固定103000历史B1标签人口、主final85421有限，P1/P2/T2满足未舍入ΔP F1≥0且ΔN F1<0；T1不满足。P1 capture−0.824276pp且不连通52→434；P2 capture+0.275643pp且不连通52→25，但paper N和多类型reference有反向；T2 macro反向与结构代价仍在。没有全面净收益结论，不替换B1或自动追加/组合实验。

遵循用户2026-10-12作者规则：匹配类型零表达F1=0/Recall=0合法，Precision按实际NaN，逐metric有限域；paper完整unmatched走作者零分分支，固定历史17579不可评价标签与真正缺行不补0。已撤回旧S合法零被称作错误评分/误纳与85103域微负论断，旧99文件及原NA恢复尝试永久保留。

[作者规则独立分析](/home/zwan0726/work/outputs/cos_bs4_20261012T013909+1100/tests/final_independent_analysis/author_marker_v1/final_analysis.md) / [逐required最终验收](/home/zwan0726/work/outputs/cos_bs4_20261012T013909+1100/tests/final_independent_analysis/author_marker_v1/final_acceptance.json) / [实验记录](/home/zwan0726/work/context/experiments/cos_bs4_20261012_author_marker.md) / [产物索引](/home/zwan0726/work/outputs/cos_bs4_20261012T013909+1100/tests/final_independent_analysis/author_marker_v1/artifact_index.md)
<!-- BS4_AUTHOR_FINAL_20261012_END -->

<!-- SSP_OVERNIGHT_FINAL_20261011_BEGIN -->
**2026-10-11：S/SP 计算及独立科学核验完成。** 两臂各 4000 真实更新、全片三 profile 与 B 代价分析齐全。主固定历史 B1 标签人口 103000（有限值 85421、NA 17579）全部保留；report ownQC 缺行显式 NA，仅恢复汇总，11 个成功阶段未重跑。S−B 为 P F1 略降/N F1 降低的权衡；SP−S 为双向不利，不改写历史默认 B1。最终 context/管理验收以 [独立最终验收](/home/zwan0726/work/outputs/cos_ssp_20261011T005741+1100/final_acceptance.json) 为准。

[本轮记录](/home/zwan0726/work/context/experiments/cos_ssp_20261011.md) / [独立最终分析](/home/zwan0726/work/outputs/cos_ssp_20261011T005741+1100/final_analysis.md) / [完整产物索引](/home/zwan0726/work/outputs/cos_ssp_20261011T005741+1100/artifact_index.md)。本轮与旧 ABC 的 102935 commonQC 人口及 Purity 集合不同，不能直接跨集合解释提升；未自动开始后续实验。
<!-- SSP_OVERNIGHT_FINAL_20261011_END -->

<!-- EGR_RESULTS_SYNC_20261009_BEGIN -->
## 2026-10-09 已完成结果与缓存状态

pilot、CRGF 第一阶段 B1/GR，以及后续 E/G/R 消融均已有最终结果。E/G/R 实际运行 `20261008T121953+1100` 已完成训练、全片输出、五组统一 scFFPE/scClassify 评价及独立分析；[实际记录](experiments/crgf_egr_20261008.md)与[最终报告](/home/zwan0726/work/outputs/crgf_egr_20261008T121953+1100/final_analysis.md)为结果依据。本轮未建立新增配置相对 B1 的全面净收益；G 相对 E 有小幅 marker 增量，GR 协同未建立。推荐默认底座 B1，结论限于单 seed/Rep1；这是该次消融的推荐，不覆盖后续 COS 的独立决策。下方“E/G/R 未启动、evaluator 待实施”属于同步前历史状态。

旧 pilot 两套特征缓存已按用户授权清理：2573 个 `.pt`，约 7.798 GiB；指标、最终结果、checkpoint、源码、配置、日志和 checksum sidecar 保留。详见[清理记录](/home/zwan0726/work/docs/BIDCell_Cache_Cleanup_20261009.md)。缓存为 `pruned/regenerable`，完整重放需重建，旧整包 hash 清单不能再声明全部文件齐全。

<!-- EGR_RESULTS_SYNC_20261009_END -->

<!-- EGR_PROTOCOL_20261008_BEGIN -->
# 2026-10-08 后续研究路线

当前方法是 backbone h2 内部 CRGF，未增加 UNet 编码层级。原生分割头有学习分数，但尚无独立候选归属头或 learned 修改 gate。未来完整机制为候选表示增强 → 候选归属预测 → 选择性修改，详见 [完整机制路线](/home/zwan0726/work/docs/BIDCell_CRGF_Full_Mechanism_Roadmap_20261008.md)。

下一次授权运行建议先补 E/G/R，与已有 B1/GR 构成五组；三组保持原六项 loss。研究结论必须容许阴性，不把单 seed“未见收益”写成路线永久无效。当前只交付 [E/G/R 执行计划](/home/zwan0726/work/docs/BIDCell_CRGF_EGR_Ablation_Execution_Plan_20261008.md)，没有自动启动训练或第二阶段。
<!-- EGR_PROTOCOL_20261008_END -->

---

<!-- CRGF1008_CURRENT_BEGIN -->
# 当前方法与研究结论 — 2026-10-08

CRGF-1008 v1.1 第一阶段及独立最终分析 **COMPLETE**。方法来源：[执行授权](/home/zwan0726/work/docs/1008_execution_prompt.md)、[change plan](/home/zwan0726/work/docs/updated_change_plan_1008.md)、[设计审查](/home/zwan0726/work/docs/updated_change_plan_1008_review.md)。真实实验：[记录](/home/zwan0726/work/context/experiments/crgf_stage1_20261008.md)；结论来源：[最终分析](/home/zwan0726/work/outputs/crgf_1008_20261008T031528+1100/final_analysis.md)。

- **B0**：身份已核验的历史 step4000 checkpoint/mask/matrix 锚点，复用而未重训；文件名不作历史实际更新数的精确证明。
- **B1**：原网络从 B0 权重和 BN 初始化，fresh Adam，新增 4,000 次实际 optimizer 更新。
- **F1**：同一 B0 起点，在 h2（C=128）加入 D=32 几何/mean-other 候选关系融合，固定 residual_scale=.1；主干、原头、新模块联合训练相同 4,000 次更新。
- B1/F1 相同完整训练池、patch/增强序列和 optimizer 设置。原六 loss、全候选竞争、原平均背景、核恢复和 native 后处理语义保留。F1−B1 为主要比较；F1−B0 混合额外训练和模块作用。

**本轮结论：工程链有效，整体净收益未建立。** F1 的轻微 capture 增量伴随面积增加、marker/结构代价；不能证明真实 owner 更准、纯度改善、独立泛化或 novelty。一个 seed、一张已接触的 Rep1 组织，实际有效更新仅覆盖完整训练池的 27.52%，不称充分收敛。正式 Purity F1 与其他定义/输入缺口如实保留。

当前授权范围已闭合，第二阶段未实施；不据此自动改 loss、加图像/router、续训或扩展消融。下方 MaskAligned A/B 和 legacy_v1 内容是保留的历史方法与结果，旧“当前”措辞不覆盖本节，原 A ROI pilot 的停止扩展结论保持其原范围。
<!-- CRGF1008_CURRENT_END -->

---

<!-- MASKALIGNED_A_CURRENT_BEGIN -->
Current implementation overlay (2026-10-07): the candidate-pixel residual A and E02 score/E03 distance/E04 absolute controls are implemented in the isolated baseline-derived `feature/maskaligned-a-pilot` worktree. Four single-configuration ROI pilots are complete; all fail pooled operational screens. Both Astra agents recommend category 3: the current difference-based A does not warrant full-field expansion. This conclusion is limited to one seed, 500 updates and the native objective. Larger A/B experiments and B remain proposed; legacy_v1 remains distinct. [Actual record](experiments/maskaligned_a_pilot_20261007.md). Earlier proposal-only statements below are historical.
<!-- MASKALIGNED_A_CURRENT_END -->

# Current method identity — 2026-10-07

Current proposal sources: `/home/zwan0726/work/docs/ISBI_Paper_Blueprint_v2.md`, `BIDCell_AB_Change_Plan_MaskAligned.md`, matrix optimization plan v2. A is a frozen-backbone candidate-pixel relative-geometry residual adapter; B adds aligned raw-DAPI CNN+concat. Both must produce native final masks and their own matrix. These are **proposed methods**, not implemented results.

`baseline` is the audited school official-style run; `legacy_v1` is the existing ambiguity loss-weighting implementation. It is not A. The old V1→V2→V3 progression below is historical; it no longer mandates successful V1 weighting before A/B. C/attention/H&E/owner-label accuracy remain outside this phase. No formal A/B, Rep2 or final test has run.

Current execution boundaries and version decisions: [05_decisions_and_protocols.md](05_decisions_and_protocols.md). Actual results: [preparation record](experiments/remote_preparation_20261007.md).

---

# Preserved historical project context (current-focus wording below is superseded)

# Project Context

> **Canonical scope:** Stable scientific and experimental context.  
> Paths belong in `02_paths_and_environment.md`; run results belong in `experiments/`.

## 1. Project Objective

The project extends BIDCell for whole-cell instance segmentation in subcellular spatial transcriptomics (SST), using DAPI-derived nuclear morphology and spatial transcript information.

The research focus is **cell-cell interface ambiguity**: neighbouring candidate cells may both plausibly explain transcripts near their shared interface. The project tests whether explicitly identifying and progressively resolving these ambiguous regions improves the biological quality of the final connected segmentation mask without causing global structural drift.

The current method roadmap is deliberately staged so that each additional mechanism is justified by the previous stage.

## 2. BIDCell Baseline

The baseline is the official BIDCell Xenium-style pipeline reproduced on the full 10x Xenium FFPE Human Breast Cancer Rep1 dataset.

Conceptually:

```text
spatial transcripts + gene identity
        -> per-gene expression maps
        -> candidate-specific BIDCell input
        -> UNet3+ candidate foreground/background prediction
        -> prediction combination and morphological postprocessing
        -> connected whole-cell instance mask
        -> point-in-mask transcript assignment
        -> cell x gene matrix
```

DAPI is primarily used to obtain nucleus anchors for candidate cells. In the baseline model, the CNN input is driven by candidate-specific transcript expression maps rather than raw DAPI as an image feature stream.

Exact baseline paths and environment are canonical in `02_paths_and_environment.md`.

Actual baseline execution history is canonical in `experiments/baseline_reproduction.md`.

## 3. Full Reproduction Status

**Status: VERIFIED / COMPLETED.**

The full Rep1 reproduction completed:

```text
raw Xenium inputs
-> nuclei segmentation
-> expression maps
-> patch generation
-> nucleus-level gene matrix
-> nuclei preannotation
-> official 1-epoch / 4000-step training
-> step-4000 full-image prediction
-> prediction merging
-> morphological postprocessing
-> connected whole-cell mask
-> final cell-level expression matrix
-> QC scripts
```

This establishes the engineering baseline for controlled V1/V2/V3 comparisons.

Live calibration recovered historical QC and benchmark measurements, recorded in the appended baseline verification note. A complete validated Tier 1 paper-comparison table is still unavailable; do not promote proxies or conflicting assignment measurements to final values.

## 4. Research Problem

BIDCell learns candidate-cell segmentation using transcript-derived expression maps and biological losses, but it does not explicitly formulate every ambiguous boundary transcript as a local candidate-ownership decision.

At neighbouring cells, a transcript can lie inside multiple candidate search areas. A naive strategy that strengthens supervision over all overlap pixels may include many easy or irrelevant cases and may also encourage competing cells to expand simultaneously.

The project therefore decomposes the problem into three questions:

1. **WHERE?** Which candidate-overlap regions are genuinely ambiguous and deserve stronger supervision?
2. **WHAT INFORMATION?** Which structural cues help resolve those regions beyond loss reweighting?
3. **WHEN + WHO?** Which residual hard cases need explicit refinement, and which candidate should win?

## 5. Boundary / Ambiguity Motivation

Candidate overlap is treated as a **necessary but not sufficient** condition for geometrical competition.

For a transcript `t`, candidate cells come from the existing BIDCell search areas:

\[
C(t)=\{c: t \in SearchArea(c)\}
\]

A transcript is potentially competitive only when:

\[
|C(t)| \ge 2
\]

For candidate `c`, use the shortest Euclidean distance from the transcript to the full nucleus mask, not centroid distance:

\[
d_c(t)=\min_{p\in N_c}\|x_t-p\|_2
\]

Normalize by equivalent nucleus radius:

\[
r_c=\sqrt{Area(N_c)/\pi}
\]

\[
\tilde d_c(t)=\frac{d_c(t)}{r_c+\epsilon}
\]

For the two principal candidates `A` and `B`, chosen by stable ordering of normalized distance:

\[
M_t=\frac{|\tilde d_A(t)-\tilde d_B(t)|}{\tilde d_A(t)+\tilde d_B(t)+\epsilon}
\]

Interpretation:

- `M_t -> 0`: A and B are geometrically similar candidates; high ambiguity.
- larger `M_t`: one candidate has a clearer geometry advantage; lower ambiguity.

V1/V2 use a fixed ambiguity rule:

\[
A_\tau(t)=I(|C(t)|\ge2)\cdot I(M_t<\tau)
\]

The threshold is selected through the V1 experiment sequence rather than assumed to be universally optimal.

## 6. Experimental Roadmap

### V1 — Geometry-Guided Ambiguity-Weighted BIDCell

**Question: WHERE?**

V1 tests whether stronger supervision targeted at genuine candidate-cell geometry competition improves the final segmentation without increasing model capacity.

```text
candidate search-area overlap
+
normalized two-nucleus distance margin
-> fixed ambiguity mask
-> selective weighting of existing biological losses
```

### V2 — Boundary-Conditioned Morphology Fusion / Cross-Attention

**Question: WHAT INFORMATION?**

V2 is a future plan, not an implemented or completed experiment in the current context.

It keeps the selected V1 ambiguity definition fixed and adds candidate-specific morphology / geometry features, potentially including:

- candidate nucleus mask;
- normalized distance to self nucleus;
- normalized distance to competing nucleus;
- relative distance advantage;
- candidate overlap;
- local DAPI intensity;
- DAPI gradient.

The planned causal test compares simple fusion methods with local cross-attention so that architecture complexity is retained only if it provides measurable value.

### V3 — Learnable Sparse Pairwise Competition

**Question: WHEN + WHO?**

V3 is a future plan, not an implemented or completed experiment in the current context.

The planned design adds:

- learned routing for hard transcripts / candidate regions;
- explicit candidate competition;
- segmentation-logit feedback so the method changes the final connected mask rather than only post-hoc transcript assignment.

The former global pairwise-competition idea is retained as a **V3-global control**, not as the current V2.

## 7. Current Focus: V1

### Hypothesis

If geometry identifies the candidate-cell regions where ownership is genuinely ambiguous, then selectively strengthening the existing BIDCell biological supervision in those regions can improve boundary-related biological quality while leaving non-ambiguous interiors and overall structure largely unchanged.

### Mechanism

For candidate-specific ambiguity, V1 constructs an ambiguity signal only for candidates participating in the selected competition and weights selected losses as:

\[
w=1+\lambda_{boundary}A_\tau
\]

The preferred implementation supports mean normalization:

\[
\tilde w=\frac{w}{E[w]}
\]

This separates local emphasis from a global increase in loss scale.

### Expected Effect

The intended evidence pattern is:

```text
boundary-related biological metric improves
+
assigned transcript fraction does not materially degrade
+
Tier 2 structural safety remains stable
+
mask changes are concentrated near ambiguous / boundary regions
```

No improvement should be claimed until the required experiments and frozen metrics have actually been run.

### What V1 Changes

Current V1 implementation is designed to change only supervision / loss routing:

- existing BIDCell search areas define candidates;
- full-nucleus nearest-pixel distances are used;
- distances are normalized by equivalent nucleus radius;
- the two most competitive candidates define `M_t`;
- a fixed ambiguity threshold selects transcript-occupied pixels;
- A/B determine the margin; at selected pixels, candidate-specific weights cover every eligible search-area channel, including third or later candidates, rather than only A/B;
- main V1 weighting applies to:
  - Overlap loss;
  - Positive-marker loss;
  - Negative-marker loss;
- CellCalling weighting exists only as a disabled-by-default ablation option;
- resource and diagnostic logging are added for experimental control.

### What V1 Must NOT Change

Main V1 must not introduce:

- a new segmentation architecture;
- a new UNet3+ head;
- cross-attention or Transformer blocks;
- a DAPI feature encoder;
- learnable routing;
- explicit A/B transcript ownership classifier;
- post-hoc reassignment as the primary method;
- modified morphological postprocessing semantics;
- weighted nucleus encapsulation or oversegmentation losses in the main configuration.

The baseline and V1 trainable parameter counts should therefore remain equal.

## 8. Experimental Control Principles

Final causal comparisons should keep the following fixed wherever applicable:

```text
same raw Xenium Rep1 data
same filtered transcripts
same Cellpose nuclei
same candidate search areas
same preannotation / reference mapping
same marker lists
same train/test patch logic
same random seed where applicable
same final total_steps = 4000
same optimiser / learning rate for V1
same prediction-combination procedure
same morphological postprocessing
same evaluation code
same hardware / CPU worker settings where feasible
```

Development ablations must use the same fixed patch manifest so that V1 variants are not compared on different subsets.

## 9. Frozen Design Decisions

The following are current decisions unless the user explicitly revises the method:

1. **V1 answers WHERE only.**
2. **UNet3+ architecture remains unchanged in V1.**
3. Candidate membership comes from existing BIDCell search-area channels.
4. Geometry uses nearest full-nucleus mask distance, not centroid distance.
5. Nucleus-size normalization uses equivalent radius.
6. When candidate count exceeds two, the two smallest normalized distances define the principal competition; total candidate count remains a diagnostic.
7. The main V1 loss placement is Overlap + Positive Marker + Negative Marker, subject to E4 ablation before final selection.
8. CellCalling weighting is an E4 control and is disabled by default.
9. Mean-normalized weighting is preferred to avoid global loss-scale confounding.
10. `tau` must be selected after E2; the implementation default `0.15` is not automatically the scientific winner.
11. E3/E4/E5 must share a fixed development patch manifest.
12. E5 must be derived from the actual E4 winner; pre-generated default E5 configs are not authoritative if E4 selects a different loss placement.
13. Final V1 4000-step training occurs only after E1 -> E2 -> E3 -> E4 -> E5 decisions are frozen.
14. `geometry_only` is not part of the currently implemented V1 toolchain because a scientifically defined candidate construction was not implemented for that standalone mode. Do not assume that the older plan's geometry-only ablation exists.
15. Tier 1/2 final evaluation must use the validated evaluation approach; do not substitute speculative proxies.

## 10. Historical Decisions / Superseded Ideas

The following are retained only for provenance and should not drive current implementation unless explicitly revived:

### Superseded: boundary-weighting-only V1

An earlier V1 treated candidate overlap / boundary emphasis more directly without the current normalized two-nucleus geometry ambiguity definition.

**Superseded by:** current Geometry-Guided Ambiguity-Weighted V1.

### Superseded: pairwise competition as V2

An earlier roadmap placed pairwise transcript competition directly in V2.

**Superseded by:** V2 morphology-conditioned feature fusion, with global explicit competition moved to V3 as a control.

### Superseded: early boundary-head / pseudo-boundary proposal

The older reproduction context proposed a generic boundary-weighted loss and later an auxiliary boundary head as a possible next step.

**Superseded by:** the fused V1/V2/V3 roadmap. V1 does not add a boundary head.

### Superseded: immediate complexity escalation

The current roadmap explicitly avoids adding cross-attention or learned routing before V1 establishes that targeted ambiguity supervision is useful.

## Source Context

Derived from:

- `boundary_change_plan_fused_v2_20260902(1).md`
- `v1_geometry_ambiguity_context_after_codex_20260903.md`
- `v1_codex_change_output.md`
- `bidcell_full_reproduction_context_20260723_official4000_qc_updated(1).md`
