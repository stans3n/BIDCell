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

## 汇总双口径主表入口（2026-10-13，18方法）

| 表格 | 路径 | 内容 |
|---|---|---|
| 精简主表 | [main_table_compact.csv](/home/zwan0726/work/outputs/bidcell_historical_dual_20261013/main_table_compact.csv) | 18 行、13 列，主要指标快速查阅 |
| 完整主表 | [main_table.csv](/home/zwan0726/work/outputs/bidcell_historical_dual_20261013/main_table.csv) | 18 行、50 列，含双口径六项评分/人数、结构和空间指标 |

范围：原14个历史终点B1、COS四臂、A/B/C、S/SP、P1/P2/T1/T2，加经审计的B0、V1_REGION、V1_BOUNDARY、V2_MORPH；B1不重复。协议仍为 `SC-PRIMARY-FIXED-v1.0-20261013`。两套Purity各按这18方法重算，原14方法的非Purity数据保留；本轮报告的五方法集合Purity不改写或跨集合比较。[原14版本归档](/home/zwan0726/work/outputs/bidcell_historical_dual_extension_20261013/archive_manifest.json) / [扩展与审计证据](/home/zwan0726/work/outputs/bidcell_historical_dual_extension_20261013) / [规则说明](/home/zwan0726/work/context/07_evaluation_dual_annotation_20261013.md)。跨实验初始化、训练预算与真值限制保留。

<!-- THREE_ARM_FINAL_20261013_BEGIN -->
**2026-10-13 三臂正式计算及 B0/B1 重评完成，结果为 V1 阴性、M 有代价的 marker 改善。** 三臂同seed1234新鲜原主干、同3998访问/3625真实更新；18stage完整计算终态，原15个三臂stage验收后复用，未重训。B0原始34,442,716行全量独立验收及Astra High审计通过，仅补原生CSV与新版证据；原计数/owner/源码/三臂receipt保持。

当前主口径继续为scClassify，固定历史103000 ID/85421可评/17579 NA并列。R/B相对B0两口径P下降/N上升；M两口径P升/N降，但capture74.636%→62.276%、面积−23.02%，固定positive Recall微降，pooled marker仍未优于额外训练的B1。B1共同四端点sc N方向反转保留；单Rep1/seed、历史B0初始化/order未配对、无owner GT，不作为全面升级。本轮保留现有默认基线。

[实验记录](/home/zwan0726/work/context/experiments/new_methods_20261013.md) / [完整报告](/home/zwan0726/work/outputs/bidcell_new_methods_20261013/final_analysis/final_report.md) / [最终独立及管理验收](/home/zwan0726/work/outputs/bidcell_new_methods_20261013/final_acceptance.json)。计算完成与科学验收分别记录，最终科学签署状态以外置验收为准；原历史内容保留。
<!-- THREE_ARM_FINAL_20261013_END -->



<!-- AUTHOR_MARKER_ZERO_RULE_20261012_BEGIN -->
**2026-10-12 评价规则优先更新：遵循作者 marker F1 的 NA/NaN→0 行为，保留合法 Recall=0。** 已知、匹配类型且评价表达全零、marker 非空时，作者结果是 F1=0、Precision=NaN、Recall=0；不能仅因零参考计数把六项分数一律改 NA。此规则优先于冲突的 BS4 计划/恢复逻辑。缺文件或缺细胞行仍不能伪造 0；历史 profile 和原结果保留。

指标清单和判定规则在 [04_metrics_framework.md](/home/zwan0726/work/context/04_metrics_framework.md:1)，详细来源在 [06_paper_evaluation_protocol_20261008.md](/home/zwan0726/work/context/06_paper_evaluation_protocol_20261008.md:1)。`03_experiment_matrix.md` 是实验登记与比较关系，指标规则以 04/06 为入口。本次仅更新 context 规则，未完成或签署 BS4 报告恢复验收。

<!-- AUTHOR_MARKER_ZERO_RULE_20261012_END -->
<!-- BS4_AUTHOR_FINAL_20261012_BEGIN -->
**2026-10-12 BS4：作者规则修订版独立验收完成。** 遵循用户2026-10-12作者规则：匹配类型零表达F1=0/Recall=0合法，Precision按实际NaN，逐metric有限域；paper完整unmatched走作者零分分支，固定历史17579不可评价标签与真正缺行不补0。已撤回旧S合法零被称作错误评分/误纳与85103域微负论断，旧99文件及原NA恢复尝试永久保留。

固定103000历史B1标签人口、主final85421有限，P1/P2/T2满足未舍入ΔP F1≥0且ΔN F1<0；T1不满足。P1 capture−0.824276pp且不连通52→434；P2 capture+0.275643pp且不连通52→25，但paper N和多类型reference有反向；T2 macro反向与结构代价仍在。没有全面净收益结论，不替换B1或自动追加/组合实验。

[作者规则独立分析](/home/zwan0726/work/outputs/cos_bs4_20261012T013909+1100/tests/final_independent_analysis/author_marker_v1/final_analysis.md) / [逐required最终验收](/home/zwan0726/work/outputs/cos_bs4_20261012T013909+1100/tests/final_independent_analysis/author_marker_v1/final_acceptance.json) / [实验记录](/home/zwan0726/work/context/experiments/cos_bs4_20261012_author_marker.md) / [产物索引](/home/zwan0726/work/outputs/cos_bs4_20261012T013909+1100/tests/final_independent_analysis/author_marker_v1/artifact_index.md)。原17stage/旧机器comparison保留；最终阶段报告以新author-aligned recovery及本验收为权威。goal管理完成由主控工具另确认。
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
# 最新入口：CRGF 后续路线与论文评价规则已整理

2026-10-08。第一阶段 B1/GR 已完成，整体净收益尚未建立。本次完成文档与评价证据同步；E/G/R 训练、第二阶段均未启动。

- [完整机制路线](/home/zwan0726/work/docs/BIDCell_CRGF_Full_Mechanism_Roadmap_20261008.md)：已实现 CRGF、候选归属评分 COS、选择性修改 SOR，以及后续决策。
- [E/G/R 执行计划](/home/zwan0726/work/docs/BIDCell_CRGF_EGR_Ablation_Execution_Plan_20261008.md)：三个独立实验分支、统一初始化、串行全流程、Astra High 实施/评估、Sol High 主控、稳定启动后暂停并人工恢复。
- [评价规则与人工事项](/home/zwan0726/work/context/06_paper_evaluation_protocol_20261008.md)：作者规则、Source Data 数值重建与本项目选择分别登记。Unknown 补零与 Purity 公式已有证据；scFFPE reference 已取得，新的 evaluator/scClassify 重评待实施。
- 完整 CRGF 分支已推送 [stans3n/BIDCell](https://github.com/stans3n/BIDCell/tree/feature/crgf-1008-stage1-20261008T031528)，commit `f840a8cf92e7ef987f8ee92f14a2e0ee628d9952`。未合并，也未创建 PR。

旧“Purity 缺定义 / reference detection 缺输入 / 未 push”仅描述当时状态；以本次更新和 context/06 为当前依据。旧实验数值、NA/profile 不改写。同步前六个 context 文件完整保存在 [历史快照](/home/zwan0726/work/context/history/before_egr_protocol_sync_20261008/ARCHIVE.md)。
<!-- EGR_PROTOCOL_20261008_END -->

---

<!-- CRGF1008_CURRENT_BEGIN -->
# 当前状态 — CRGF-1008 第一阶段及最终分析完成

更新：2026-10-08，Australia/Sydney。run_id：`20261008T031528+1100`。实施、短验证、B1/F1 匹配继续训练、双 shift 全片输出、own-mask matrix/可用 metrics 和独立 Astra High 最终分析均已完成；goal 工具状态为 `complete`。

**工程终点成立，F1 相对匹配 B1 的整体净收益尚未建立。** capture 增加 0.532949 个百分点，平均细胞面积增加 1.2904%；两套 annotation profiles 的 positive marker F1 微降、negative marker F1 微升，不连通细胞由 60 增至 80。正式 Purity F1 仍缺定义，单 seed/Rep1、无真实 owner 标签的限制保留。

| 当前事项 | 状态 / 入口 |
|---|---|
| 完整实验与最终核验 | COMPLETED；B1/F1 各 4,000 次实际更新，11 个计算阶段均一次成功 |
| 实验记录 | [CRGF 第一阶段记录](/home/zwan0726/work/context/experiments/crgf_stage1_20261008.md) |
| 实验登记 | [登记表](/home/zwan0726/work/context/experiment_registry.csv) / [登记说明](/home/zwan0726/work/context/03_experiment_matrix.md) |
| 最终报告 | [final_analysis.md](/home/zwan0726/work/outputs/crgf_1008_20261008T031528+1100/final_analysis.md) |
| 比较表与证据 | [52 行比较表](/home/zwan0726/work/outputs/crgf_1008_20261008T031528+1100/final_comparison.csv) / [产物索引](/home/zwan0726/work/outputs/crgf_1008_20261008T031528+1100/artifact_index.md) |
| 未解决研究问题 | [remaining_issues.md](/home/zwan0726/work/outputs/crgf_1008_20261008T031528+1100/remaining_issues.md)；不构成本轮工程重跑要求 |
| 当前后续边界 | 本轮闭合；第二阶段、Rep2、多 seed、消融、阈值搜索均未自动开展 |

CRGF 是 h2 内部几何/候选关系融合、联合训练主干的方法，与下方历史 MaskAligned A 冻结主干 ROI pilot 和 legacy_v1 区分。旧 A pilot 的停止扩展结论继续保留，不能移作 CRGF 的实验身份或判定规则。以下旧“current / next action / not yet run”文字按其日期、方法和当时授权解读；当前 CRGF 状态以本节为准。

更新前的 11 个当前文件已逐字节保存在 [历史快照](/home/zwan0726/work/context/history/before_crgf_stage1_final_20261008/ARCHIVE.md)，原有 history 和实验记录保持原样。
<!-- CRGF1008_CURRENT_END -->

---

<!-- MASKALIGNED_A_CURRENT_BEGIN -->
Current authorized MaskAligned A ROI task is **COMPLETE**: implementation, ordered incremental acceptance, four fixed single-configuration pilots, final artifact verification and an evidence-based decision. E02/E04/E05 each executed 500 updates; E03 used a fixed rule with zero updates. All four pooled operational screens fail. Both Astra agents recommend category 3: **do not expand the current A configuration**. E05 capture fell 1.7067 percentage points and median area fell 9.09%. Real-data compute: 37.18 minutes; outputs at runner receipt: 10.894 GiB. Local commit `c93d39d3d0261823dd5363e9672debf865189708` on `feature/maskaligned-a-pilot`. [Delivery](/home/zwan0726/work/outputs/maskaligned_a_pilot_20261007T035938Z/DELIVERY.md); [canonical record](experiments/maskaligned_a_pilot_20261007.md). This overlay supersedes preparation-stage status and next-action statements preserved below. No full-field A/B, Rep2, final test, push or merge occurred.
<!-- MASKALIGNED_A_CURRENT_END -->

# BIDCell project context

Current update: 2026-10-07, Australia/Sydney. This remains the canonical context directory; earlier content is preserved under [history](history/before_remote_preparation_20261007/README.md).

## Current status

School remote preparation has completed repository/environment audit, baseline offline artifact and integer-matrix acceptance, official CellSPA engineering evaluation, matrix optimization, and a persistent downstream queue execution. **Actual disconnect tests A and B both passed during the user-confirmed 14:03–14:12 Sydney outage on 2026-10-07.** The requested preparation phase is complete.

GPU was manually repaired by the user and then successfully revalidated in the actual BIDCell Conda environment. Earlier CUDA-failure observations are obsolete. Do not repeat driver repair, reinstall/purge NVIDIA/CUDA packages, run apt autoremove, or reboot as part of this task.

The active research proposal is the **MaskAligned A/B** plan: candidate-pixel residual geometry adapter, then raw-DAPI CNN+concat. The existing geometry ambiguity loss-weighting branch is **legacy_v1**, not A and not a mandatory prerequisite. New method/evaluation documents are still proposed/draft; precise scientific choices have not silently been frozen.

| Item | Status |
|---|---|
| School historical baseline | COMPLETED; new offline integrity acceptance PASS |
| Optimized final and nuclear matrix equivalence | COMPLETED; exact full integer matches |
| Official engineering evaluator execution | COMPLETED; scientific protocol incomplete |
| Persistent downstream matrix→validation→evaluation | COMPLETED; upstream stages imported from verified existing artifacts |
| Disconnect A / B | COMPLETED / PASS separately; actual outage, new independent CLI tool calls verified |
| legacy_v1 remote tests | FAILED overall: two retained failures; not new A experiments |
| Formal A/B, Rep2, final test | NOT YET RUN; not authorized in this preparation phase |

## Read in this order

1. [Project/method identities](01_project_context.md)
2. [Verified paths and environment](02_paths_and_environment.md)
3. [Experiment registry and next gates](03_experiment_matrix.md)
4. [Metric framework and protocol boundaries](04_metrics_framework.md)
5. [Decisions and versions](05_decisions_and_protocols.md)
6. [This actual preparation record](experiments/remote_preparation_20261007.md)

The detailed deliverables live in [remote_prep_20261007T122955](/home/zwan0726/work/outputs/remote_prep_20261007T122955); they are evidence, not a second mutable project context. [Delivery index](/home/zwan0726/work/outputs/remote_prep_20261007T122955/DELIVERY.md), [readiness](/home/zwan0726/work/outputs/remote_prep_20261007T122955/readiness.md), [disconnect runbook](/home/zwan0726/work/outputs/remote_prep_20261007T122955/disconnect_runbook.md).

## Next authorized action

Stop at this completed preparation boundary. See the disconnect acceptance report in the evidence directory; future research work requires a new instruction and resolution of the documented protocol gates. Do not start formal A/B search, Rep2, final-test evaluation, or reinterpret missing Purity F1 as a proxy-selected winner.

## Maintenance

Preserve historical runs, original artifacts, untracked files and old context. New outputs get separate IDs/paths. Keep measured school remote results, Windows historical reports, proposed methods and frozen decisions distinct. Pipeline success, evaluation execution and paper protocol completeness are separate fields. A missing definition/input blocks only its dependent computation. Never change scientific definitions, references, normalization, NA rules or eligibility silently.
