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
**2026-10-12 BS4四臂追加。** P1/P2在同B pre只改connect，训练not_required、预测reused；T1α0β1/T2α.25β1从原B1 fresh Adam独立4000真实更新，所有POS权1，同views/teacher。正式仅四臂，B disabled是验证，不是第五臂；S复用历史。

固定103000历史B1标签人口、主final85421有限，P1/P2/T2满足未舍入ΔP F1≥0且ΔN F1<0；T1不满足。P1 capture−0.824276pp且不连通52→434；P2 capture+0.275643pp且不连通52→25，但paper N和多类型reference有反向；T2 macro反向与结构代价仍在。没有全面净收益结论，不替换B1或自动追加/组合实验。

[作者规则独立分析](/home/zwan0726/work/outputs/cos_bs4_20261012T013909+1100/tests/final_independent_analysis/author_marker_v1/final_analysis.md) / [逐required最终验收](/home/zwan0726/work/outputs/cos_bs4_20261012T013909+1100/tests/final_independent_analysis/author_marker_v1/final_acceptance.json) / [实验记录](/home/zwan0726/work/context/experiments/cos_bs4_20261012_author_marker.md) / [产物索引](/home/zwan0726/work/outputs/cos_bs4_20261012T013909+1100/tests/final_independent_analysis/author_marker_v1/artifact_index.md)
<!-- BS4_AUTHOR_FINAL_20261012_END -->

<!-- SSP_OVERNIGHT_FINAL_20261011_BEGIN -->
**2026-10-11：S/SP 计算及独立科学核验完成。** 两臂各 4000 真实更新、全片三 profile 与 B 代价分析齐全。主固定历史 B1 标签人口 103000（有限值 85421、NA 17579）全部保留；report ownQC 缺行显式 NA，仅恢复汇总，11 个成功阶段未重跑。S−B 为 P F1 略降/N F1 降低的权衡；SP−S 为双向不利，不改写历史默认 B1。最终 context/管理验收以 [独立最终验收](/home/zwan0726/work/outputs/cos_ssp_20261011T005741+1100/final_acceptance.json) 为准。

[本轮记录](/home/zwan0726/work/context/experiments/cos_ssp_20261011.md) / [独立最终分析](/home/zwan0726/work/outputs/cos_ssp_20261011T005741+1100/final_analysis.md) / [完整产物索引](/home/zwan0726/work/outputs/cos_ssp_20261011T005741+1100/artifact_index.md)。本轮与旧 ABC 的 102935 commonQC 人口及 Purity 集合不同，不能直接跨集合解释提升；未自动开始后续实验。
<!-- SSP_OVERNIGHT_FINAL_20261011_END -->

<!-- COS_LOSS_ABC_FINAL_20261010_BEGIN -->
**2026-10-10：COS-LOSS-ABC-v2.0 已完成独立最终验收。** A/B/C各4000真实更新、全片16阶段、三profile及完整独立科学/工程核查均完成。主固定B1标签共同人口102935（可评价85290）：B−A为marker权衡，C−B为双向负信号，未建立新增干预净收益；科学阴性不影响工程完成。此机制筛选不改写历史默认B1决策。[本轮记录](/home/zwan0726/work/context/experiments/cos_loss_abc_20261010.md) / [最终分析](/home/zwan0726/work/outputs/cos_loss_20261009T160351+1100/final_analysis.md) / [验收](/home/zwan0726/work/outputs/cos_loss_20261009T160351+1100/final_acceptance.json)。原run_status/3999 heartbeat保留为计算交接快照，当前独立状态以final_acceptance为准。未自动开始后续实验。
<!-- COS_LOSS_ABC_FINAL_20261010_END -->

<!-- EGR_RESULTS_SYNC_20261009_BEGIN -->
## 2026-10-09 已完成结果与缓存状态

pilot、CRGF 第一阶段 B1/GR，以及后续 E/G/R 消融均已有最终结果。E/G/R 实际运行 `20261008T121953+1100` 已完成训练、全片输出、五组统一 scFFPE/scClassify 评价及独立分析；[实际记录](experiments/crgf_egr_20261008.md)与[最终报告](/home/zwan0726/work/outputs/crgf_egr_20261008T121953+1100/final_analysis.md)为结果依据。本轮未建立新增配置相对 B1 的全面净收益；G 相对 E 有小幅 marker 增量，GR 协同未建立。推荐默认底座 B1，结论限于单 seed/Rep1；这是该次消融的推荐，不覆盖后续 COS 的独立决策。下方“E/G/R 未启动、evaluator 待实施”属于同步前历史状态。

旧 pilot 两套特征缓存已按用户授权清理：2573 个 `.pt`，约 7.798 GiB；指标、最终结果、checkpoint、源码、配置、日志和 checksum sidecar 保留。详见[清理记录](/home/zwan0726/work/docs/BIDCell_Cache_Cleanup_20261009.md)。缓存为 `pruned/regenerable`，完整重放需重建，旧整包 hash 清单不能再声明全部文件齐全。

<!-- EGR_RESULTS_SYNC_20261009_END -->

<!-- EGR_PROTOCOL_20261008_BEGIN -->
# 2026-10-08 待执行消融登记说明

[E/G/R 执行计划](/home/zwan0726/work/docs/BIDCell_CRGF_EGR_Ablation_Execution_Plan_20261008.md) 定义 E（自身表达）、G（自身表达+显式几何）、R（自身表达+关系），与已完成 B1、GR=F1 比较。每组独立分支、各 4000 次匹配更新；已有 B1/GR mask/matrix 复用并统一重评。

状态为 **PLANNED / NOT STARTED**；本轮不把计划实验写入已完成运行记录，也不改旧登记表的 measured values。E 是近似容量对照，非严格等参数。实现、测试、启动与结果状态须在真实发生后登记。
<!-- EGR_PROTOCOL_20261008_END -->

---

<!-- CRGF1008_CURRENT_BEGIN -->
# 当前实验登记 — CRGF-1008，2026-10-08

最新登记已追加至 [experiment_registry.csv](/home/zwan0726/work/context/experiment_registry.csv)，旧行原样保留。共同 run_id 为 `20261008T031528+1100`；详细实际记录：[crgf_stage1_20261008.md](/home/zwan0726/work/context/experiments/crgf_stage1_20261008.md)。

| ID | 范围 | 状态 |
|---|---|---|
| CRGF1008-STAGE1-20261008T031528+1100 | 完整匹配队列及最终分析 | COMPLETED；工程终点成立、净收益未建立 |
| CRGF1008-VALIDATION-20261008T031528+1100 | 实施与必要短验证 | COMPLETED / PASS；启动前失败日志保留 |
| CRGF1008-B1-20261008T031528+1100 | B0→原网络，新增4,000实际更新、全片闭环 | COMPLETED；一次成功尝试/阶段 |
| CRGF1008-F1-20261008T031528+1100 | B0→内部CRGF，新增同4,000实际更新、全片闭环 | COMPLETED；一次成功尝试/阶段 |
| CRGF1008-ANALYSIS-20261008T031528+1100 | 用户恢复后独立Astra High核验与分析 | COMPLETED；[最终报告](/home/zwan0726/work/outputs/crgf_1008_20261008T031528+1100/final_analysis.md) |

B0 复用 `BASE-001` 的历史 checkpoint/mask/matrix，不登记为新训练。B1/F1 各双 shift 覆盖 35,418 patches、自己的 final mask 和 matrix；主要比较 F1−B1。capture +0.532949 个百分点与面积 +1.2904% 并存，marker/结构无一致改善，不登记科学成功或“赢家”。正式 Purity 仍缺定义。

本轮已完成最终分析并关闭 goal；第二阶段、Rep2、多 seed、额外消融未运行。下方准备阶段及 A/B/V1 的“未运行/阻断/下一门槛”是各自旧任务记录，不将它们作为已授权 CRGF 第一阶段的阻断，也不改写其历史结果。
<!-- CRGF1008_CURRENT_END -->

---

<!-- MASKALIGNED_A_CURRENT_BEGIN -->
Current E00–E05 ROI task is COMPLETE: shared cache and E00 four-dev reference; E01 ordered incremental acceptance on the first dev ROI only; E02/E03/E04/E05 fixed single-configuration endpoints and final artifact verification. E02/E04/E05 each have 500 actual optimizer calls (420 G-signal views, 80 G-empty views); E03 has zero. All four pooled operational screens fail. User decision category 3: stop expansion of current A. [Registry](experiment_registry.csv), [record](experiments/maskaligned_a_pilot_20261007.md) and linked final artifacts supersede preparation-stage pending claims below. The full A/B matrix remains planned and unauthorized; this is not complete A ablation or full-image E01.
<!-- MASKALIGNED_A_CURRENT_END -->

# Current preparation registry — 2026-10-07

Canonical current rows: [experiment_registry.csv](experiment_registry.csv). Detailed actual run: [remote_preparation_20261007.md](experiments/remote_preparation_20261007.md).

| ID | Logical method / scope | Current state |
|---|---|---|
| BASE-001 | School historical baseline | COMPLETED; current artifact acceptance PASS |
| PREP-MATRIX-20261007 | Engineering only | COMPLETED; small/native and full nuclear/final integer equivalence |
| PREP-EVAL-20261007 | Engineering official CellSPA profiles | COMPLETED execution; paper protocol BLOCKED |
| PREP-RUNNER-20261007 | Actual optimized matrix→validation→evaluation; upstream imported | COMPLETED; inherited locks and resume checked |
| PREP-DISCONNECT-A-20261007 | Deterministic queue after actual Mac disconnect | COMPLETED / PASS; stage 1→2 at 14:04 during actual disconnect |
| PREP-DISCONNECT-B-20261007 | New remote model tool calls after actual disconnect | COMPLETED / PASS; independent new model/tool calls at 14:05 during actual disconnect |
| V1-REMOTE-001 | legacy_v1 source regression suite | FAILED overall: 23 pass, 2 fail; diagnostic evidence preserved |
| A/B E00–E21 | Proposed MaskAligned method experiments | NOT YET RUN; research gates and separate authorization required |

Preparation and manual-disconnect handoff are complete. Formal selection remains blocked by official Purity scaling/reference/class protocol, split/buffer/eligibility and numerical guardrails. Candidate residual implementation, E01/gradient and DAPI controls have not been run; old V1 tests do not satisfy them. No unconditional A/B run list is scheduled.

Measured timing and scope limitations: [readiness](/home/zwan0726/work/outputs/remote_prep_20261007T122955/readiness.md), [matrix performance](/home/zwan0726/work/outputs/remote_prep_20261007T122955/matrix_benchmark.md). Original V1 sequence below is preserved historical design, not the current next-action queue.

---

# Historical experiment matrix (preserved)

# Experiment Matrix

> **Canonical scope:** Experiment definitions, dependencies and status.  
> Actual measured results belong in `experiments/`; paths belong in `02_paths_and_environment.md`.

## 1. Experiment Naming Convention

Use stable IDs that preserve run history:

```text
BASE-001                 completed official reproduction
V1-REMOTE-001            first remote V1 validation attempt
V1-E1-001                first geometry sanity run
V1-E2-001                first threshold-sensitivity run
V1-E3-001                first ambiguity-definition ablation run set
V1-E4-001                first loss-placement ablation run set
V1-E5-001                first lambda-sensitivity run set
V1-E6-001                first full 4000-step V1 run
V1-E7-001                first baseline-to-V1 locality evaluation
```

If an experiment is repeated, increment the final run suffix rather than overwriting history.

## 2. Current Stage

**Current stage: V1 checkout and integration audit complete; remote validation blocked by missing pytest.**

The baseline and V1 code preparation are complete. Live calibration found no V1 scientific run evidence. The active checkout exactly equals freshly fetched origin V1; see `02_paths_and_environment.md#current-branch`. Read-only imports and source inspection did not complete remote validation.

## 3. Experiment Levels

The project uses four operational evidence levels for V1:

### Tier 0 — Reproduction / Sanity

Purpose:

> Prove the baseline pipeline is usable and the V1 implementation behaves correctly before causal comparison.

Includes baseline reproduction, remote dependency/tests and V1-E1 geometry sanity.

### Tier 1 — Main Experiment

Purpose:

> Test the core V1 hypothesis at full Rep1 scale.

Includes V1-E6 full 4000-step comparison and V1-E7 locality evidence.

### Tier 2 — Ablation

Purpose:

> Identify which V1 component actually causes any improvement.

Includes V1-E3 ambiguity definition and V1-E4 loss placement.

### Tier 3 — Sensitivity

Purpose:

> Determine whether the method depends excessively on a narrow threshold or coefficient.

Includes V1-E2 `tau` sensitivity and V1-E5 `lambda` sensitivity.

These operational tiers are experiment-design categories. They are distinct from the **evaluation tiers** in `04_metrics_framework.md`.

## 4. Master Experiment Matrix

| ID | Level | Version | Experiment | Purpose | Initialization | Key Change | Status | Result Record |
|---|---|---|---|---|---|---|---|---|
| `BASE-001` | Tier 0 | Baseline | Official full Rep1 reproduction | Establish engineering baseline and frozen step-4000 outputs | Official BIDCell configuration | None | **COMPLETED** | `experiments/baseline_reproduction.md` |
| `V1-PREP-001` | Preparation | V1 | Geometry-ambiguity code implementation | Implement frozen V1 without running production data | Frozen baseline code | Geometry/weighting tools, diagnostics, resource monitor | **COMPLETED** | `experiments/v1_runs.md#v1-prep-001--code-preparation` |
| `V1-REMOTE-001` | Tier 0 | V1 | Remote dependency + unit/augmentation validation | Verify V1 implementation in the real BIDCell environment | V1 commit | No scientific change; runtime validation only | **BLOCKED** by missing pytest in canonical environment | No actual run record yet |
| `V1-E1-001` | Tier 0 | V1 | Geometry Sanity | Verify that low `M_t` identifies meaningful candidate competition regions | Baseline/V1 preprocessing inputs; no retraining | Compute/visualize ambiguity geometry | **BLOCKED** by `V1-REMOTE-001` | No actual run record yet |
| `V1-E2-001` | Tier 3 | V1 | Fixed Threshold Sensitivity | Test whether ambiguity forms a stable controllable spectrum | E1-validated geometry; no retraining | `tau` in `{0.05,0.10,0.15,0.20,0.30}` | **BLOCKED** by E1 | No actual run record yet |
| `V1-E3-001` | Tier 2 | V1 | Ambiguity Definition Ablation | Determine whether geometry filtering adds value beyond overlap-only selection | Fixed development patch manifest; selected `tau` | `overlap_only` vs `overlap_geometry` | **BLOCKED** by E2 + manifest | No actual run record yet |
| `V1-E4-001` | Tier 2 | V1 | Loss Placement Ablation | Determine which existing biological losses benefit from targeted weighting | Same manifest, `tau`, seed and development budget | Overlap-only; markers-only; overlap+markers; CellCalling control | **BLOCKED** by E3 | No actual run record yet |
| `V1-E5-001` | Tier 3 | V1 | Boundary Weight Sensitivity | Test robustness to `lambda_boundary` | Actual E4 winner | `lambda` in `{0.5,1,2,5}` with mean normalization | **BLOCKED** by E4 | No actual run record yet |
| `V1-E6-001` | Tier 1 | V1 | Full Rep1 4000-step V1 | Test the V1 core hypothesis at production scale | Frozen best E2/E3/E4/E5 choices | Selected geometry ambiguity + selected loss weighting | **BLOCKED** by E5 | No actual run record yet |
| `V1-E7-001` | Tier 1 | V1 | Change Locality Evaluation | Test whether segmentation changes are concentrated near intended boundary regions | Baseline connected mask + V1 connected mask | No retraining; inter-version locality evaluation | **BLOCKED** by E6 | No actual run record yet |

## 5. Baseline / Reproduction

### `BASE-001` — Official BIDCell Rep1 Full Reproduction

**Status:** COMPLETED.

Scientific role:

- common baseline for all later versions;
- establishes the exact preprocessing/training/prediction/postprocessing chain;
- provides frozen step-4000 artifacts.

The run record is canonical in `experiments/baseline_reproduction.md`.

## 6. V1 Sanity Checks

### `V1-REMOTE-001` — Remote Validation

**Status:** BLOCKED by missing pytest in the canonical environment; checkout and read-only integration audit complete.

Readiness evidence: the target ref, baseline inputs, Python environment and basic dependency imports are verified. Required V1 tests, augmentation alignment and real-data/GPU validation remain unrun. No branch preparation or tests were performed in the context calibration.

Completion requirements:

- live repository branch/commit confirmed;
- remote dependencies load correctly;
- required V1 test suites pass;
- augmentation test that was skipped locally executes and passes, or any failure is recorded and resolved;
- no production data is overwritten.

This is a runtime validation, not a scientific result.

### `V1-E1-001` — Geometry Sanity

**Status:** BLOCKED by remote validation.

Purpose:

> Verify that the current `M_t` construction identifies geometrically plausible competition at neighbouring nuclei.

Required observations:

- candidate overlap aligns with real competing search areas;
- low `M_t` is concentrated around plausible competition interfaces;
- high `M_t` generally reflects a clearer candidate advantage;
- at least several representative ROIs are manually inspected.

The implementation currently uses deterministic seeded ROI selection rather than guaranteed density-stratified selection. This is acceptable for the first E1 run; if selected ROIs fail to cover sparse/medium/dense regions, density-stratified selection should be added before accepting E1.

E1 is a geometry validity gate, not a segmentation-accuracy result.

## 7. V1 Main Experiment

### `V1-E6-001` — Full 4000-step Rep1

**Status:** BLOCKED until E2/E3/E4/E5 are frozen.

The final configuration must freeze:

```text
ambiguity definition*
tau*
loss placement*
lambda*
```

Production controls should otherwise match the baseline as specified in `01_project_context.md#8-experimental-control-principles`.

Primary evaluation uses the frozen framework in `04_metrics_framework.md`.

### `V1-E7-001` — Change Locality

**Status:** BLOCKED until the full V1 connected mask exists.

Purpose:

> Establish whether the V1-induced mask change is boundary-local rather than global drift.

Required comparison:

```text
baseline connected mask
vs
V1 connected mask
```

Use the Tier 3 evaluation metrics defined in `04_metrics_framework.md`.

## 8. V1 Ablations

### `V1-E3-001` — Ambiguity Definition Ablation

Current implemented comparison:

```text
overlap_only
vs
overlap_geometry
```

The older fused plan also described a standalone `geometry_only` variant. The latest implementation explicitly did **not** implement this mode because candidate construction for geometry-only selection was not scientifically defined in the existing pipeline.

Therefore:

- do not treat `geometry_only` as a currently required runnable arm;
- adding it later requires an explicit method decision and context update.

An optional prediction-uncertainty filter remains analysis-only / not required for the current V1 main line.

### `V1-E4-001` — Loss Placement Ablation

Development-scale comparison:

```text
Official / unweighted BIDCell control
Weight Overlap only
Weight Positive + Negative Marker only
Weight Overlap + Positive + Negative Marker
Weight CellCalling + Overlap + Markers   # control only
```

Use identical:

```text
tau
lambda
seed
patch manifest
training budget
```

The CellCalling arm tests the concern that strengthening expansion pressure on both competing candidates can worsen competition.

## 9. Parameter Sensitivity Experiments

### `V1-E2-001` — Tau Sensitivity

Planned values:

```text
0.05, 0.10, 0.15, 0.20, 0.30
```

No training is required for the threshold-distribution sanity analysis.

Important control:

The implementation's default `tau = 0.15` is not a frozen winner. E2 determines the selected value.

### `V1-E5-001` — Lambda Sensitivity

Planned values:

```text
0.5, 1, 2, 5
```

Important control:

E5 must be derived from the **actual E4 winner**. If generated configs assume Overlap + Marker but E4 selects another placement, the default E5 configs must not be used as the final sensitivity set.

## 10. Development Patch Manifest Dependency

Before E3/E4/E5, create or select one fixed development patch manifest and keep it unchanged across the compared arms.

The V1 implementation supports manifest creation/selection, but it is not automatically enforced unless the relevant manifest options are used.

**Status:** PLANNED / not yet recorded as created.

Failure to fix the patch subset invalidates causal interpretation of small-development comparisons.

## 11. Future V2 / V3

These are roadmap entries only. Do not create run logs or mark them active until V1 evidence justifies progression.

| Version | Core question | Planned mechanism | Status |
|---|---|---|---|
| V2 | WHAT INFORMATION? | Candidate morphology/geometry feature stream + fusion; cross-attention retained only if superior to simpler fusion | **NOT YET REQUIRED** |
| V3-global | WHO? global control | Explicit candidate competition for all candidate-overlap cases | **NOT YET REQUIRED** |
| V3-final | WHEN + WHO? | Learned sparse routing + explicit competition + segmentation feedback | **NOT YET REQUIRED** |

Progression gates are defined in `01_project_context.md`; evaluation remains fixed by `04_metrics_framework.md`.

## 12. Execution Dependencies

Canonical sequence:

```text
BASE-001 completed
    -> V1-PREP-001 completed
    -> V1-REMOTE-001
    -> V1-E1
    -> V1-E2
    -> fixed development manifest
    -> V1-E3
    -> V1-E4
    -> V1-E5
    -> freeze final V1 config
    -> V1-E6 full 4000-step
    -> prediction + postprocessing + connected mask
    -> Tier 1/2/3/4 evaluation
    -> V1-E7 locality result
```

Do not run the entire sequence as an unconditional batch. E1–E5 contain scientific decision gates.

## 13. Completion Criteria

### Remote validation complete when

- live commit is verified;
- real remote dependencies and augmentation path pass;
- no unresolved blocker remains for E1.

### E1 complete when

- geometry outputs are generated;
- representative ROIs are inspected;
- geometry is accepted or the failure is recorded and corrected.

### E2 complete when

- all planned thresholds are summarized;
- a stable `tau*` is selected with rationale.

### E3 complete when

- overlap-only and overlap+geometry are compared under the same development controls;
- one ambiguity definition is selected.

### E4 complete when

- loss placements are compared on the same manifest/seed/budget;
- one placement is selected.

### E5 complete when

- lambda sensitivity is run on the E4 winner;
- one robust `lambda*` is selected.

### E6 complete when

- final 4000-step training finishes;
- prediction/postprocessing finish;
- V1 connected mask and required outputs exist;
- the run is recorded in `experiments/v1_runs.md`.

### V1 evaluation complete when

- required Tier 1/2/3/4 metrics are calculated according to `04_metrics_framework.md`;
- actual measured values are recorded in the V1 run log;
- structural guardrails are checked;
- locality evidence is available;
- README status is updated.

## Source Context

Derived from:

- `boundary_change_plan_fused_v2_20260902(1).md`
- `v1_geometry_ambiguity_context_after_codex_20260903.md`
- `v1_codex_change_output.md`


## 2026-09-09 - V1 Checkout Alignment (Not a Run)

Checkout preparation COMPLETED. Previous branch: Bidcell_local; previous HEAD: e565988cd2e78e622c68bd0a5649a1ec8b9b281f. Current branch: feature/geometry-ambiguity-v1; current HEAD and freshly fetched origin V1: 58ef3dfeb1828e63cb29f162e1bb4f1f81db9cce. Upstream: origin/feature/geometry-ambiguity-v1. Tracked diff against origin V1 is empty. Authorized obsolete tracked edits were discarded. No untracked/ignored collision occurred; those files remain untouched. All essential V1 files exist; model.py equals baseline snapshot. No clean, merge, commit, push, tests or experiments occurred. This entry supersedes earlier checkout-pending statements; remote validation and E1-E7 remain NOT YET RUN. Current stage: read-only integration/initialization audit before remote validation.
