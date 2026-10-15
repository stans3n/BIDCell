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
**2026-10-12 BS4作者规则决策追加。** 固定103000历史B1标签人口、主final85421有限，P1/P2/T2满足未舍入ΔP F1≥0且ΔN F1<0；T1不满足。P1 capture−0.824276pp且不连通52→434；P2 capture+0.275643pp且不连通52→25，但paper N和多类型reference有反向；T2 macro反向与结构代价仍在。没有全面净收益结论，不替换B1或自动追加/组合实验。

遵循用户2026-10-12作者规则：匹配类型零表达F1=0/Recall=0合法，Precision按实际NaN，逐metric有限域；paper完整unmatched走作者零分分支，固定历史17579不可评价标签与真正缺行不补0。已撤回旧S合法零被称作错误评分/误纳与85103域微负论断，旧99文件及原NA恢复尝试永久保留。

新四端点非S F1/Recall n85421，T1−S/T2−S均85349；Precision分别85294/85291。S缺130固定ID其中72known仍NA。全部5×4stage F1共同85421，P1/T2 final P保持正；六final/18profile/16敏感性/Purity复核未变。单Rep1/seed、共享marker弱标签、transductive、无外部owner真值；无显著性或practical tolerance。scalar_reduction_review_required=True；旧CUDA失败保留，不保证普遍bitwise。

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
# 2026-10-08 当前决策增补

1. 本轮交付 [完整机制路线](/home/zwan0726/work/docs/BIDCell_CRGF_Full_Mechanism_Roadmap_20261008.md) 和 [E/G/R 执行计划](/home/zwan0726/work/docs/BIDCell_CRGF_EGR_Ablation_Execution_Plan_20261008.md)；执行训练须用户后续启动 goal。本轮已授权并完成完整 CRGF 分支向 Stans3n GitHub 推送，没有 merge。
2. E/G/R 保留原六 loss，先完成机制归因；COS/SOR/新 loss 是下一阶段，不能自动叠加。每组独立分支，common 实现同源且代码运行前冻结。
3. 后续主控 Sol High；代码/测试/runner/evaluator 修改及最终分析由 Astra High 子代理完成。稳定正式启动后按用户条件暂停 goal，后台脚本继续；用户人工恢复后分析完成才能标 complete。
4. [评价规则与人工事项](/home/zwan0726/work/context/06_paper_evaluation_protocol_20261008.md) 登记作者明确规则、数值重建和项目选择。Unknown 官方补零不回写旧工程 NA；新 scFFPE/scClassify/QC profile 必须统一重评全部五组。
5. 用户无需事先手工配置 R 或新增已有 reference；agent 先处理常规环境工作。真正需要人工的模型选择、goal 启动/恢复及条件式外部访问/数据事项详见 context/06 第 8 节。
<!-- EGR_PROTOCOL_20261008_END -->

---

<!-- CRGF1008_CURRENT_BEGIN -->
# 最新执行决策与边界 — 2026-10-08

CRGF-1008 v1.1 依据 [1008执行prompt](/home/zwan0726/work/docs/1008_execution_prompt.md) 的明确授权完成实施、匹配全量队列、稳定启动后暂停及用户恢复后的独立最终分析。goal 已由工具标为 `complete`；[完成证据](/home/zwan0726/work/outputs/crgf_1008_20261008T031528+1100/final_delivery.json)。

1. 冻结单一代码版本 `f840a8cf92e7ef987f8ee92f14a2e0ee628d9952`，独立 baseline-derived worktree。Astra implementer 单写实验代码；用户通知后独立 Astra analyst 完成核验，实际请求均为 `gpt-6-astra/high`，不声称验证接口未暴露的底层模型。
2. B1/F1 同 B0 权重/BN 起点，fresh Adam、seed20261008、完整池、相同采样/增强，各4,000实际更新；FP32、单GPU串行。未因九小时窗口截断或扩增预算。
3. 保留原六loss、完整候选组、平均背景和native输出语义；Unknown历史训练 -1→末类SMC共同保留并披露，评价仍NA；不将共同保留说成正确监督。
4. 独立队列自行完成既定 metrics；暂停期间未调用 analyst。用户通知恢复后才进行科学分析，分析未改实验代码、重训或覆盖旧产物。
5. [最终结论](/home/zwan0726/work/outputs/crgf_1008_20261008T031528+1100/final_analysis.md)：工程终点成立，F1对B1整体净收益未建立。保留单seed/组织、无owner真值、正式Purity缺口及B0历史不可纯归因限制。
6. 本轮闭合；第二阶段、图像/router、新loss、多seed/Rep2、额外训练与消融未自动开展。后续研究单独决定，当前没有工程修复或重跑阻断。

下方2026-10-07的准备/A pilot授权边界保留为历史，其“未授权/未实施”只按当时任务解释；CRGF新增授权不追溯改变原A pilot结论、旧run状态或评价定义。[本轮记录](/home/zwan0726/work/context/experiments/crgf_stage1_20261008.md)；[更新前快照](/home/zwan0726/work/context/history/before_crgf_stage1_final_20261008/ARCHIVE.md)。
<!-- CRGF1008_CURRENT_END -->

---

<!-- MASKALIGNED_A_CURRENT_BEGIN -->
Current frozen overlay: MA-A-PILOT-1.1, seed 20261007; 8 train/4 dev blocks with unchanged 384-pixel cores and 500-view sequence. The initial 96-pixel halo failed before learned outcomes; revised halo144/context672 passed full candidate-ID train/dev isolation and global bbox+56 closure. Common train-only lambda=69.95866394042969; E03 beta=0.8890590740563818, b=1.2457919120788574. Original six losses/reductions and native decoding/dual shift/connectivity are retained. Ordered first-dev-only E01 PASS; all 20 endpoint ROI chains and 8630 score archives PASS. All four fixed pilots are COMPLETE and fail pooled operational vetoes. Both Astra agents select category 3: stop expansion of current A, bounded to this difference representation/seed/500 updates/native objective. Official Purity F1 remains missing_definition; historical full-field backbone/preannotation exposure is disclosed. Full-field measured-component planning subtotal for E00+four controls is 6.2–21.2 hours plus separate unmeasured reconciliation, matrix and evaluation terms; cache alone projects 102.16 GiB. [Record](experiments/maskaligned_a_pilot_20261007.md). This overlay supersedes preparation-stage pending statements below. No full-field/B/Rep2/final-test/push/merge was authorized or performed.
<!-- MASKALIGNED_A_CURRENT_END -->

# Decisions, versions and evidence boundaries

Updated 2026-10-07 Australia/Sydney. User instruction takes precedence; preserve historical context and artifacts.

| Item | Version / status | Interpretation |
|---|---|---|
| Frozen metric list | 2026-09-03, retained | Required names/roles; not all precise computation/selection details were frozen |
| Blueprint v2 | actual docs SHA in evidence; proposed | A/B claim and control proposal |
| MaskAligned plan | MA-AB-1.0-proposed | A candidate-pixel residual geometry; B raw DAPI CNN+concat; legacy weighting distinct |
| Evaluation adaptation | MA-EVAL-1.0-draft | Not complete paper protocol |
| Matrix plan | MA-MATRIX-OPT-2.0-proposed | Engineering optimization implemented and validated this run; no method speed-up claim |
| Engineering code | remote-prep-1.1; content hash manifest | Recursive resume validation added; original 1.0 manifest retained with evidence |
| Count contract | native processed/.2125 round + OpenCV nearest resized mask | Full equivalent axis lookup and integer aggregation; no change of ownership |
| Official CellSPA | 1e892c470177fcea351c9175a9153e5480e93646 | Functions sourced unchanged; named engineering adapters/NA joins disclosed |
| GPU restore | user manual repair + remote tensor verification | current PASS, old failure obsolete; no further driver intervention |
| Disconnect | actual 2026-10-07 14:03–14:12 Sydney interval | A PASS; B PASS via independent remote CLI; desktop continuity and power recovery not inferred |

**Frozen execution boundaries for this task:** no formal A/B search, no Rep2, no final-test evaluation, no rerun of baseline training/inference, no overwriting original data/reports/configs. CPU matrix/evaluation timings are school remote measurements; Windows numbers remain historical. New method implementation/selection thresholds/splits are proposed or missing, not silently frozen by this engineering work.

**Research decisions still needed:** primary reference/QC/normalization/type mapping and empty-class eligibility; #10 method scaling cohort/range/degenerate behavior; #11 official pair/bins and any scalar definition; #19 bounds/edge rules; capture/structure/contamination guardrails; physical split/buffer and independent CI units; geometry/window definitions. Existing full-field backbone exposure must be disclosed. Missing decisions do not invalidate completed integer equivalence.

Evidence/version hashes: [/home/zwan0726/work/outputs/remote_prep_20261007T122955/evidence/input_document_and_untracked_hashes.json](/home/zwan0726/work/outputs/remote_prep_20261007T122955/evidence/input_document_and_untracked_hashes.json); [source manifest](/home/zwan0726/work/outputs/remote_prep_20261007T122955/source_manifest.json); [cache dependencies](/home/zwan0726/work/outputs/remote_prep_20261007T122955/cache_dependencies.json).
