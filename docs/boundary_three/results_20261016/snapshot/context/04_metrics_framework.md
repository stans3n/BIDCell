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

<!-- AUTHOR_MARKER_ZERO_RULE_20261012_BEGIN -->
# 2026-10-12 当前指标规则：保留作者 marker F1 补零与合法 Recall=0

适用指标：下文 #8 positive-marker、#9 negative-marker Precision/Recall/F1；#10 Purity 使用相应 author-aligned 原始 F1 汇总。本次用户明确要求以作者评分行为为准，优先于 BS4-OVERNIGHT-v2.0 及恢复计划中把零参考计数细胞六项分数一律改为 NA 的冲突条款。下文历史文字和历史产物保留，不代表这些冲突条款继续有效。

| 情形 | F1 | Precision | Recall | 评价规则 |
|---|---|---|---|---|
| 已进入 marker 类型计算，计算所得 F1 为 NA/NaN | 按作者 `f1[is.na(f1)] <- 0` 置 0 | 保留作者计算结果 | 保留作者计算结果 | 不将 P/R 的 NA 一律补零，也不把所有非有限值都当作作者明确补零范围 |
| 已知且匹配的类型，评价表达向量全零，评价域内 marker 集合非空 | 0 | NaN | 0 | F1=0、Recall=0 是合法结果；Recall 分母是 marker 数量，不能因表达总计数为 0 就称其分母为 0 |
| 有完整输入行，但类型未匹配 marker 列表、未进入作者类型计算 | 0 | 0 | 0 | 作者函数追加全零行；报告未匹配数。真正缺失标签先显式映射并验证，不直接假设 R NA 标签安全 |
| 文件/细胞行/counts/必需依赖真正缺失或损坏 | 不伪造 0 | 不伪造 0 | 不伪造 0 | 缺项保留原因；工程缺项仍须处理，不能以补零冒充完成 |

执行与汇总约束：

- 只有确认来自实际输入和作者评分函数的零分才作为有效数值；不因 `score == 0` 或 `zero_reference_counts == true` 自动改 NA、判错、排除细胞或阻断验收。零参考计数可以作为描述性标志，不能单独决定六项指标无效。
- 对当前 BS4 已知、匹配类型的阶段评分，保留作者 F1=0、Recall=0；四端点比较按每个指标实际输出逐项判定共同有效域，合法 0 参与均值与分母。Precision 的 NaN 不使同一细胞的 F1/Recall 自动失效。不预先硬编码修正后的有效人数或差值。
- `paper_aligned_v1` 遵循作者评分及未匹配补零规则，并保留其 own QC、表达预处理和注释定义。固定历史标签诊断与 paper profile 的人口/标签不同，不能仅修正补零就称其完整复现论文；不擅自把历史标签不可评价者全部填 0。此次明确纠正的是合法 marker 零分被强制改 NA 的规则。
- 对已冻结评价人口，pooled marker F1 汇总包含作者产生的有效零分；P/R 单独报告有效数、NA 和原因。QC/标签/参考基因集合不得为此静默更换，也不得将零表达函数测试理解为绕过 paper QC。
- 已保存的原报告、NA 恢复尝试和旧验收记录保留。若受影响报告需重算，使用另行标识的版本，记录口径差异并重新核验；这份规则澄清不等于 BS4 恢复或最终验收已完成。

依据：[作者 publish 源码](https://github.com/SydneyBioX/CellSPA/blob/publish/R/expression_metrics.R)；[已归档源码，第 52–85 行](/home/zwan0726/work/outputs/bidcell_metric_protocol_review_20261008/publish_expression_metrics.R:52)，SHA256 `2d7f0e0da83fed939e5cc2316ea159f5962477e214d00685c73f0f01142775ec`；[详细评价协议](/home/zwan0726/work/context/06_paper_evaluation_protocol_20261008.md)。2026-10-12 直接调用该归档函数（Metrics 0.1.4、BiocParallel::SerialParam），全零已知类型/简单正例/未匹配类型分别得到 `(F1,Precision,Recall)=(0,NaN,0)/(1,1,1)/(0,0,0)`；这是函数级验证，不是全部实验结果重评，也不代表核验了最新远端源码。

<!-- AUTHOR_MARKER_ZERO_RULE_20261012_END -->

---

<!-- EGR_PROTOCOL_20261008_BEGIN -->
# 2026-10-08 论文规则核验更新（优先于下文历史缺口）

详见 [评价规则与人工事项](/home/zwan0726/work/context/06_paper_evaluation_protocol_20261008.md)。frozen 1–43/Tier1B 的指标身份保留；本次补证据和实施规则，不重写旧结果。

- 作者 marker 代码对未进入 marker 类型计算的细胞追加 P/R/F1 全零；已计算细胞的 F1 NA→0，P/R 不同样统一补零。旧工程 profile 的 Unknown→NA 保留为历史选择。
- Purity 的 positive/negative 独立 min–max + 调和组合已用作者 Source Data 23 行重建，最大误差约 1.33e-14；源数据支持按细胞加权汇总，具体推断边界见 context/06。新实验缩放集合和零范围规则属于项目协议。
- 作者匹配 scFFPE reference 已取得，307 genes 的 mean/detection 已生成。#7 不再属于输入缺失；#10 不再统称公式未知，剩余是 scClassify/QC/evaluator 集成及统一重评。
- 官方 QC 默认/教程是 transcripts 20–2000（含端点）、genes≥20、logNormCounts；不能据此声称原论文所有图逐数复现。论文 scClassify 与教程 max-Pearson 分开。
- 新 paper_aligned_v1 尚未执行；旧 paper_protocol_complete=false 仍是真实运行状态。所有新指标另存协议版本，不混入历史 NA/raw profiles。
<!-- EGR_PROTOCOL_20261008_END -->

---

<!-- CRGF1008_CURRENT_BEGIN -->
# 最新指标应用记录 — CRGF-1008，2026-10-08

冻结指标身份保留；本节登记已完成实验的实际计算域与限制，不改变旧定义。来源：[最终分析](/home/zwan0726/work/outputs/crgf_1008_20261008T031528+1100/final_analysis.md)、[52行比较表](/home/zwan0726/work/outputs/crgf_1008_20261008T031528+1100/final_comparison.csv)、[剩余问题](/home/zwan0726/work/outputs/crgf_1008_20261008T031528+1100/remaining_issues.md)。

- 首要比较 F1 对匹配 B1；B0 为历史锚点。两方法各 107,955 个细胞、313 genes，capture 分母固定 34,442,716 条过滤后 transcript；own-mask owner/counts/完整 CSV 验收通过。
- F1 capture 66.002568% vs B1 65.469619%，差 +0.532949 个百分点；平均面积 136.717577 vs 134.975897 µm²，差 +1.2904%。最终 mask spacing 为 1 µm/pixel，`.2125` lookup 缩放不能作面积换算。
- 两套 profiles 分开。核预注释 all-cell 汇总保留 `NA/partial_domain_NA`（1,040 Unknown）；106,915 known-type 有限域均值仅为补充。Windows-style max-Pearson 用各自表达推断类型，存在人口重注释的解释限制。P/R 的有限域分母另列。
- negative marker F1 增大表示高表达 gene 与负 marker 的重叠增多，不能当作 purity 提高。两 profiles 的 positive F1 微降、negative F1 微升；本轮整体净收益未建立。
- 同ID核包含100%是 native 核恢复约束，mask IoU/边界带变化是变化量，都不是准确率证据。own-mask raster 形态不冒充官方 polygon 指标。
- 正式 #7/#10/#19/#42 缺输入或定义继续 NA；#40/#41 Rep2、#43 多 seed 未运行；#30 V3 coarse/refined 不适用。旧B0错误编号#30仅在最终表映射为 `engineering_pre_post`，原历史产物不改写。
- #38 在最终表解析为单次完整推理 F1/B1=0.9825617；原先阶段 metrics 的 pending 状态保留。无重复/顺序控制，不能宣称固有加速。matrix 缓存提速归计数工程，不归模型贡献。

`evaluation_executed=true`、完整工程终点和 goal 完成均不等于 `paper_protocol_complete=true`；后者仍为 false。下面 A ROI pilot 的 operational vetoes 只适用于其冻结范围，不自动移作 CRGF 科学成功规则。
<!-- CRGF1008_CURRENT_END -->

---

<!-- MASKALIGNED_A_CURRENT_BEGIN -->
Current pilot application overlay (2026-10-07): metric identities below are retained. MA-A-PILOT-1.1 uses pooled count-weighted capture over all 542680 filtered core transcripts, equal-core foreground, and median area/final 8-connected fragmentation of 961 fixed complete cells. The eligible-cell matrix excludes edge-cell owners; these and background are counted separately. Per-block screens are diagnostic and fully reported. Capture loss <=0.5 percentage points, absolute relative foreground/median-area drift <=5%, and fragmentation increase <=0.1 percentage points are pre-result operational vetoes, not scientific success criteria. All four pilots fail pooled screens. Two official engineering annotation profiles remain separate with NA/Unknown and raw denominators. Official primary Purity F1 remains missing_definition. B-to-T curves are ROI-restricted/censored; final disconnected rates of zero reflect native postprocessing. [Actual record](experiments/maskaligned_a_pilot_20261007.md).
<!-- MASKALIGNED_A_CURRENT_END -->

# Current application note — 2026-10-07

The frozen 1–43/Tier1B list below remains preserved. Apply it with the newer MaskAligned proposal's method identities and conditional/deferred items, without inventing absent formulas or treating the draft as a signed paper protocol.

Windows→remote 63-item definitions/capability audit: [CSV](/home/zwan0726/work/outputs/remote_prep_20261007T122955/windows_remote_coverage.csv), [Markdown](/home/zwan0726/work/outputs/remote_prep_20261007T122955/windows_remote_coverage.md). Executed school engineering metrics: [CSV](/home/zwan0726/work/outputs/remote_prep_20261007T122955/evaluation/metrics.csv). Official CellSPA pinned revision `1e892c470177fcea351c9175a9153e5480e93646` is now directly invoked; the old remote purity proxy is not used.

Primary Purity F1 stays missing_definition: cross-method scaling cohort/ranges/degeneracy and class rules are not frozen. #7 lacks reference detection input; #19 lacks thresholds; #42 lacks independent groups. Marker P/R retain NA; 1040 original Unknown nuclear annotations retain missing results. Separate remote-nuclear/raw and Windows-style-max-Pearson/raw engineering profiles must never be silently merged. Neither is the frozen paper annotation/QC/normalization protocol.

EP-SMOKE/CORE/FINAL/COST/EXT requirements remain; EP-FAST-DEV is added only as screening. Evaluator exit0/evaluation_executed does not mean CORE complete or paper_protocol_complete. Final-test mode is prohibited in this preparation build. Current registry and protocol versions are in [05_decisions_and_protocols.md](05_decisions_and_protocols.md).

---

# Frozen Evaluation Framework

> **Status:** FROZEN  
> **Evaluation target:** Final connected whole-cell instance segmentation mask.  
> **Canonical source:** The frozen metric framework dated 2026-09-03.  
> Do not change metric definitions or comparison roles unless the user explicitly revises the framework.

## 1. Evaluation Principles

The final evaluation has five scientific tiers:

| Evaluation Tier | Purpose |
|---|---|
| **Tier 1** | BIDCell / CellSPA-aligned biological segmentation performance |
| **Tier 2** | Structural safety and mask integrity |
| **Tier 3** | Boundary-aware mechanism and change locality |
| **Tier 4** | Computational efficiency and resource cost |
| **Tier 5** | Robustness and generalisation |

Interpretation order:

```text
Tier 1: Did biological segmentation quality improve?
+
Tier 2: Did the connected mask remain structurally valid?
+
Tier 3: Did the proposed mechanism change the intended boundary regions?
+
Tier 4: What computational cost was introduced?
+
Tier 5: Does the result generalise beyond one run/sample?
```

Tier 1 + Tier 2 determine whether the final segmentation is better and safe. Tier 3 is mechanism evidence, not ground-truth accuracy. Tier 4 measures cost. Tier 5 tests robustness.

No new transcript-ground-truth protocol should be invented for this project unless explicitly requested later.

## 2. Tier 1 — BIDCell / CellSPA-Aligned Biological Performance

Where applicable, final paper-level values should use the official / validated CellSPA implementation rather than development proxies.

| # | Metric | Definition / Source | Compared Against | Purpose | Direction |
|---|---|---|---|---|---|
| 1 | Number of segmented cells | Final connected mask instance count | Paper vs reproduction; reproduction vs V1/V2/V3 | Detect abnormal merging, splitting, disappearance or creation | Stable / plausible |
| 2 | Assigned transcript fraction | Fraction of transcripts captured by final masks | Paper vs reproduction; reproduction vs V1/V2/V3 | Measure transcript capture without rewarding indiscriminate expansion | Higher with caution |
| 3 | Transcripts per cell | Final assignment summary | Paper vs reproduction; reproduction vs V1/V2/V3 | Detect expansion/shrinkage and biological content shift | Stable / plausible |
| 4 | Genes per cell | Final cell x gene matrix summary | Paper vs reproduction; reproduction vs V1/V2/V3 | Measure retained biological information | Stable / plausible |
| 5 | Cell area | Final connected mask | Paper vs reproduction; reproduction vs V1/V2/V3 | Detect morphology drift | Stable distribution |
| 6 | Expression-reference Pearson | Validated CellSPA/reference pipeline | Paper vs reproduction; reproduction vs V1/V2/V3 | Agreement of segmented-cell expression with reference | Higher |
| 7 | Non-zero expression Pearson | Validated CellSPA/reference pipeline | Paper vs reproduction; reproduction vs V1/V2/V3 | Consistency of gene detection patterns | Higher |
| 8 | Positive-marker Precision / Recall / F1 | Official/validated marker evaluation | Paper vs reproduction; reproduction vs V1/V2/V3 | Retention of expected marker expression | Higher |
| 9 | Negative-marker Precision / Recall / F1 | Official/validated CellSPA definition | Paper vs reproduction; reproduction vs V1/V2/V3 | Detect unwanted marker contamination | Better according to official definition |
| 10 | Purity F1 | Official CellSPA for final reporting | Paper vs reproduction; reproduction vs V1/V2/V3 | Overall marker-based biological purity | Higher |
| 11 | Nearest-neighbour contamination | CellSPA-style neighbour contamination | Paper/reproduction where comparable; reproduction vs V1/V2/V3 | Primary boundary-relevant biological contamination measure | Lower |

### Tier 1 Interpretation

**Number of cells** is not monotonic: more cells are not automatically better.

**Assigned transcript fraction** is only favourable when purity, contamination, morphology and nucleus-related guardrails remain stable or improve.

**Transcripts/cell, genes/cell and cell area** are descriptive quality indicators; large shifts can expose trivial mask expansion or shrinkage.

**Expression-reference Pearson** is directly paper-comparable only when the same or sufficiently matched reference, cell-type mapping and normalization are used. Otherwise label the evaluation setup separately.

**Purity F1** must use the validated/official implementation for final reporting. A development proxy cannot be labelled as the paper-equivalent metric.

**Nearest-neighbour contamination** is a key boundary-aware metric. Lower contamination is desired, but not at the expense of severe capture loss or structural failure.

### Tier 1 Supplementary CellSPA Analyses

Where feasible, retain as supplementary rather than headline metrics:

- fraction of cells expressing each gene;
- gene-level expression concordance;
- gene detection consistency;
- morphology descriptors such as elongation, circularity, sphericity, compactness, convexity, eccentricity and solidity;
- cell-vs-nucleus matched morphology correlations;
- cell-type composition concordance when a matched reference is available;
- local cell-type Shannon entropy and CV-based spatial descriptors.

These do not replace the frozen headline metrics.

## 3. Tier 2 — Structural Safety and Mask Integrity

Tier 2 is a guardrail layer. It does not claim exact whole-cell ground-truth boundary accuracy.

| # | Metric | Definition / Source | Compared Against | Purpose | Direction |
|---|---|---|---|---|---|
| 12 | Nucleus containment rate | Final mask + nuclei | reproduction vs V1/V2/V3 | Verify nuclei remain inside corresponding cells | High / stable |
| 13 | Nucleus coverage distribution | Final mask + nuclei | reproduction vs V1/V2/V3 | Measure completeness of nucleus coverage | Stable / high |
| 14 | Zero-nucleus cell fraction | Final mask + nuclei | reproduction vs V1/V2/V3 | Detect invalid instances without nuclei | Lower / near zero |
| 15 | Multi-nucleus cell fraction | Final mask + nuclei | reproduction vs V1/V2/V3 | Detect cell merging | Lower / near zero |
| 16 | Split-nucleus fraction | Final mask + nuclei | reproduction vs V1/V2/V3 | Detect over-segmentation around nuclei | Lower / near zero |
| 17 | Disconnected-cell fraction | Connected instance mask | reproduction vs V1/V2/V3 | Detect fragmented instances | Lower / near zero |
| 18 | Cell / nucleus area ratio | Final mask + nuclei | reproduction vs V1/V2/V3 | Detect expansion/shrinkage relative to nucleus | Stable distribution |
| 19 | Extreme-area outlier fraction | Final mask | reproduction vs V1/V2/V3 | Detect giant or abnormally small cells | Lower / stable |
| 20 | Foreground fraction | Final mask | reproduction vs V1/V2/V3 | Detect global mask expansion/contraction | Stable |
| 21 | Cell-cell boundary pixel fraction | Final mask | reproduction vs V1/V2/V3 | Describe overall interface structure | Stable / descriptive |

A Tier 1 improvement is not convincing if it coincides with clear Tier 2 failure, such as strong foreground expansion or increased multi-nucleus merging.

## 4. Tier 3 — Boundary-Aware Mask Mechanism

Tier 3 asks **where the segmentation changed**, not whether a boundary is objectively ground-truth correct.

| # | Metric | Definition / Source | Compared Against | Purpose | Direction |
|---|---|---|---|---|---|
| 22 | Changed pixel fraction | Inter-version connected masks | Baseline->V1; V1->V2; V2->V3 | Quantify total mask change | Diagnostic |
| 23 | Changed-cell fraction | Inter-version matched instances | Same transitions | Quantify fraction of materially modified cells | Diagnostic |
| 24 | Per-cell inter-version mask IoU distribution | Matched inter-version masks | Baseline vs V1; V1 vs V2; V2 vs V3 | Measure per-cell change magnitude | Diagnostic only |
| 25 | 1 um boundary-band change fraction | Inter-version change within boundary band | Same transitions | Test very local interface change | Higher concentration desired |
| 26 | 2 um boundary-band change fraction | Same | Same transitions | Test boundary locality | Higher concentration desired |
| 27 | 3 um boundary-band change fraction | Same | Same transitions | Test boundary locality | Higher concentration desired |
| 28 | 5 um boundary-band change fraction | Same | Same transitions | Distinguish local change from global drift | Diagnostic |
| 29 | Non-boundary interior preservation | Inter-version masks | Same transitions | Verify interior stability | Higher |
| 30 | V3 coarse-mask -> refined-mask change fraction | V3 coarse vs final connected mask | V3 mechanism only | Prove pairwise competition feeds back into segmentation | Non-zero and mechanism-consistent |

### V1 Mechanism Logic

For reproduction -> V1, desired evidence is:

```text
mask changes concentrated near candidate cell-cell interfaces
+
non-boundary interiors largely preserved
+
Tier 1 purity and/or neighbour contamination improves
```

### V2 Mechanism Logic

For V1 -> V2, desired evidence is:

```text
additional changes remain predominantly boundary-localised
+
Tier 1 improves relative to V1
+
Tier 2 remains stable
```

### V3 Mechanism Logic

V3 must produce a measurable final-mask change after explicit competition. A method that only changes a transcript assignment table is insufficient to support a segmentation-refinement claim.

### Deliberately Excluded from Final Tier 3 Performance

The following may be used for development/mechanism inspection but are **not** final segmentation-performance metrics in the frozen framework:

- boundary transcript assignment rate;
- candidate-overlap transcript ratio;
- candidate-overlap area ratio;
- top-1/top-2 assignment margin;
- assignment-change rate;
- marker-consistent reassignment rate;
- assignment entropy;
- transcript ownership accuracy proxies;
- ambiguity-router score distributions.

## 5. Tier 4 — Computational Efficiency

| # | Metric | Definition / Source | Compared Against | Purpose | Direction |
|---|---|---|---|---|---|
| 31 | Training wall time | Measured full training stage | reproduction vs V1/V2/V3 | End-to-end training cost | Lower, all else equal |
| 32 | Inference wall time | Measured prediction stage | reproduction vs V1/V2/V3 | Deployment/prediction cost | Lower, all else equal |
| 33 | Postprocessing / refinement time | Measured postprocess/refinement stage | reproduction vs V1/V2/V3 | Isolate refinement overhead | Lower, all else equal |
| 34 | Peak GPU VRAM | Measured stage peak | reproduction vs V1/V2/V3 | Hardware feasibility | Lower, all else equal |
| 35 | Peak CPU RAM | Measured peak RSS / external cross-check where needed | reproduction vs V1/V2/V3 | Host memory requirement | Lower, all else equal |
| 36 | Trainable parameter count | Model metadata | reproduction vs V1/V2/V3 | Quantify capacity increase | V1 should equal baseline |
| 37 | Training steps per second | Training monitor | reproduction vs V1/V2/V3 | Compare throughput | Higher |
| 38 | Total inference overhead vs BIDCell | Derived from measured stage times | V1/V2/V3 vs reproduction | Summarise extra deployment cost | Lower |
| 39 | V3 sparse refinement speed-up vs V3-global | Measured V3 variants | V3-final vs V3-global | Test sparse-routing efficiency claim | Higher speed-up |

For V1, the resource monitor implementation can support wall time, peak CPU RSS, CUDA allocated/reserved peaks, trainable parameter count, training steps and steps/s. CPU RSS stage isolation and multiprocessing child aggregation have known limitations and must be reported as such if they affect final values.

The correct V3 efficiency claim is **lower refinement cost than V3-global**, not necessarily lower total cost than original BIDCell.

## 6. Tier 5 — Robustness and Generalisation

| # | Metric / Analysis | Compared Against | Purpose | Priority |
|---|---|---|---|---|
| 40 | Rep1 -> Rep2 headline metric consistency | Same method across Rep1/Rep2 | Test trend transfer | Highest |
| 41 | Independent Rep2 performance | BIDCell vs V2/V3-final on Rep2 | Test final-method generalisation | Highest |
| 42 | 95% confidence intervals for headline metrics | Within/between methods | Quantify uncertainty | High |
| 43 | Random-seed stability | Optional repeated runs | Test optimisation stability | Optional if compute allows |

Priority order under limited compute:

```text
Rep2 generalisation
>
confidence intervals
>
multiple full random-seed runs
```

## 7. Ablation Comparison Metrics

Ablations do not get a different metric definition. They use the same frozen metrics appropriate to the scientific question.

| Experiment | Main comparisons | Required metric emphasis |
|---|---|---|
| V1-E3 ambiguity definition | overlap-only vs overlap+geometry | Development biological metrics + geometry diagnostics; final winner later judged by frozen Tier 1/2/3 |
| V1-E4 loss placement | Overlap, markers, combined, CellCalling control | Purity/contamination/capture + structural guardrails |
| V1-E5 lambda sensitivity | lambda variants | Same core quality metrics + stability across lambda |
| V1-E6 full main run | reproduction vs V1 | Full Tier 1 + Tier 2 + Tier 4 |
| V1-E7 locality | reproduction vs V1 masks | Tier 3 |
| V2 fusion ablation | V1 vs concat/FiLM/cross-attention | Tier 1/2/3/4 |
| V3-global vs V3-final | global vs sparse explicit competition | Tier 1/2/4; Tier 3 as mechanism support |

Development proxies may screen configurations, but final claims must use the frozen reporting implementation.

## 8. Reporting Format

### Primary Results Table

The paper-level main table should use this structure:

| Metric | BIDCell paper | BIDCell reproduction | V1 | V2 | V3-global | V3-final |
|---|---:|---:|---:|---:|---:|---:|
| Number of cells | | | | | | |
| Assigned transcripts (%) | | | | | | |
| Transcripts / cell | | | | | | |
| Genes / cell | | | | | | |
| Cell area | | | | | | |
| Expression-reference Pearson | | | | | | |
| Non-zero expression Pearson | | | | | | |
| Positive-marker F1 | | | | | | |
| Negative-marker F1 | | | | | | |
| Purity F1 | | | | | | |
| Nearest-neighbour contamination | | | | | | |
| Nucleus containment | — | | | | | |
| Multi-nucleus fraction | — | | | | | |
| Disconnected-cell fraction | — | | | | | |
| Training time | — | | | | | |
| Inference time | — | | | | | |
| Peak VRAM | — | | | | | |

Do not fill absent values with estimates.

### Recommended Supplementary Figures

1. **Morphology distributions** — cell area, elongation, compactness, solidity, eccentricity, cell/nucleus area ratio.
2. **Neighbour contamination curve** — contamination as a function of distance to heterogeneous neighbours.
3. **Boundary change locality** — mask-change rate as a function of distance from the baseline boundary.

## 9. Metric Interpretation by Comparison

### BIDCell paper vs reproduction

Use primarily Tier 1 to establish reproduction fidelity and identify reference/dataset/evaluation differences.

### Reproduction vs V1

Use Tier 1 + Tier 2 + Tier 3 + Tier 4.

Question:

> Does geometry-guided targeted supervision improve boundary-related biological quality without damaging global segmentation structure?

### V1 vs V2

Use Tier 1 + Tier 2 + Tier 3 + Tier 4.

Question:

> Does morphology/geometry-conditioned feature fusion add value beyond loss reweighting?

### V2 vs V3-global

Use Tier 1 + Tier 2 + Tier 3 + Tier 4.

Question:

> Does explicit candidate competition add segmentation benefit after V2?

### V3-global vs V3-final

Use Tier 1 + Tier 2 + Tier 4, with Tier 3 as mechanism support.

Question:

> Can sparse learned routing retain global-competition quality at lower refinement cost?

### Rep1 vs Rep2

Use Tier 1 + selected Tier 2 + Tier 5.

Question:

> Does the final improvement generalise beyond the Rep1 development dataset?

## 10. Frozen Rules

1. Final evaluation target is the **connected whole-cell instance segmentation mask** and its downstream biological/structural consequences.
2. Do not invent a separate transcript ownership ground-truth protocol.
3. Do not report predicted/estimated values as measured.
4. Use official/validated CellSPA implementations for final paper-compatible values wherever applicable.
5. A higher assigned transcript fraction is not automatically better if purity, contamination or structural integrity worsens.
6. Inter-version IoU is allowed only as a **change metric**, not segmentation accuracy.
7. The frozen framework does **not require**:
   - Dice against vendor segmentation;
   - IoU against vendor segmentation;
   - AJI;
   - PQ;
   - Boundary F1 against vendor segmentation;
   - ASSD;
   - HD95.
8. Vendor segmentation, if ever used later, must be labelled as a vendor/pseudo-reference rather than manual whole-cell ground truth.
9. Metric definitions and comparison roles must not be changed without explicit user approval.

## Source Context

Derived from:

- `frozen_metric_framework_boundary_bidcell_20260903.md`
- `boundary_change_plan_fused_v2_20260902(1).md` only where consistent with the frozen metric file
- `v1_geometry_ambiguity_context_after_codex_20260903.md` for V1 resource-monitor limitations
