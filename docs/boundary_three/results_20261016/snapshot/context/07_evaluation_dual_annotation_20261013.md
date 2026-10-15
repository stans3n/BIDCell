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

# 当前评价规则：scClassify 正负 F1 主判断与辅助对照

协议：`SC-PRIMARY-FIXED-v1.0-20261013`。适用于此 BIDCell 项目此后的每一个实验及本次历史重评。此规则优先于旧文档中把固定标签设为唯一主口径、或声称 scClassify 尚未实施的表述；历史实验记录与原结果保留，不追溯改写。

## 当前选优优先级（2026-10-13 更新）

选优补充：`SC-PN-SELECTION-v1.0-20261013`。计分、QC、分母与已发布产物的协议身份继续沿用各自记录；本补充明确后续分析和方法选择的优先级。

- **主要依据重新 scClassify 判型后的 positive F1（越高越好）和 negative F1（越低越好）判断；Purity 用于辅助综合比较，固定标签 F1 暂不参与选优。**
- “原始 F1”“未缩放 F1”是每个方法自己的最终表达经 scClassify 重新判型后、尚未用于 Purity 缩放合成的正负 marker F1，不是固定标签 F1。scClassify 是判型方法，positive/negative F1 才是此处的主要评价指标。
- 正负 F1 方向存在权衡时，明确报告权衡；不得仅凭 Purity 综合排名宣布主指标全面改善。固定标签结果不用于替代主排名、决定获胜或在主口径未改善时转而宣称性能提升。
- 固定标签及既有四端点配对结果可继续按下方契约计算、保留并用于机制解释；它们的固定标签可评价域不替代 scClassify 主评价人口。捕获率、未判型比例、结构、参考相关等用于说明代价和结论边界。
- Purity 必须标明比较集合与缩放端点，不能跨不同集合直接比较；即使展示 Purity 排序，也同时保留 scClassify 正负 F1，且不将该排序自动作为选优结论。

**与论文的关系：** [BIDCell Methods 的 Evaluation metrics and settings、公式 22，以及 Fig. 3d](https://www.nature.com/articles/s41467-023-44560-w) 使用 scClassify 注释分割细胞，计算正负 marker Precision/Recall/F1，并报告缩放合成的 Purity F1 与其他方法比较；论文还结合捕获、形态、参考相关、邻域污染等维度。上面的“正负 F1 主要、Purity 辅助、固定标签不参与选优”是本项目确定的判断优先级，沿用论文评价口径，但不能表述成论文明确规定了同样的唯一主辅层级或固定标签后备规则。

## 最终 evaluate 的必需输出

1. **主口径**：使用每个方法自己的最终分割计数矩阵，对其 QC 后表达以同一冻结 scClassify 参考模型重新注释，然后计算论文对齐的 CellSPA 表达指标。不得把核预注释或最大 Pearson 判型改名为 scClassify。
2. **固定标签对照**：使用相同参考基因与 marker 评分规则，保留同一份预先冻结的基线细胞 ID、历史标签与不可评价原因。当前 Rep1 使用历史 B1 的 103,000 个 ID，通常有 85,421 个可评价标签；人数须由实际数据核验，不硬填。固定历史 B1 标签不是 nuclei_cell_type.h5 核标签。其他数据集须先冻结其统一基线人口和标签。
3. **同一主表**并列列出两套 positive/negative marker Precision、Recall、F1，附各指标真实有限值人数；同时报告 QC 前后人数、未判型比例、分子捕获、每细胞转录本/基因数、面积和不连通数。选优主要依据重新 scClassify 判型后的 positive/negative F1；Purity 仅作辅助，固定标签只用于对照解释、暂不参与选优，不根据结果有利程度更换主口径。
4. 另给同细胞配对表：某方法与基线的 scClassify 和固定标签四个端点使用同一批可评价 ID，报告人数及类型变化比例。两套全体汇总人口不同，不能把所有差异归因于换标签。
5. 保存按类型的表达相关、检测率相关、类型组成及邻域负 marker 曲线。重复性、多 seed 或真实 owner 精度没有运行时标为未测，不伪造论文全部评价类别已完成。

## 冻结的当前 Rep1 计算规则

- 参考：匹配 scFFPE，30,365 个参考细胞，307 个参考基因、17 个类型。空间计数保留 313 个基因；QC/归一化在完整空间基因域执行，再按名称对齐参考交集。
- QC：`20 <= transcripts <= 2000` 且 `genes >= 20`；随后 `scater::logNormCounts`。所有方法使用相同阈值，各自通过 QC 的 ID 可以不同。
- scClassify 1.5.1；共享参考模型 SHA256 `37f75b893ac0e4bd334939115f754acf350a108c61a12ac1cb3c680867442df6`。参数来源为已冻结的本地论文复现设置，不称其与作者历史参数逐项完全一致。具体参数见 [当前实际协议](/home/zwan0726/work/projects/BIDCell_crgf_egr_common_20261008T121953/scripts/crgf/configs/egr_evaluation_protocol.json)。
- marker：调用已归档作者源码；`q=0.9`、positive `t=0.25`、negative `t=1`。未匹配类型的 paper 输出按作者补零；固定历史不可评价标签/真正缺行保留 NA 和原因。匹配类型合法 F1=0/Recall=0 保留，Precision 的 NaN 不扩散至 F1。
- 每个 profile 的 F1 对其实际评价人口进行细胞等权平均。Precision/Recall 分别记录有限值、NA 和分母；不得默默把 finite mean 当全体均值。
- Purity F1 对预先记录的共同方法集合分别 min–max 缩放 P/N，再使用论文公式；scClassify 与固定标签分开缩放，并记录端点。不同方法集合的 Purity 不直接跨表比较。
- 捕获率分母固定为该实验的全部合格转录本，不随表达 QC 或标签改变。当前 Rep1 为 34,442,716。面积使用最终 mask 的 1 µm/pixel；`.2125` 为坐标 lookup 缩放，不用于面积换算。结构指标取最终 mask，不从训练 loss 推断。
- 局部空间多样性沿用作者 `calSpatialMetricsDiversity` 默认 `bin_width=500`，在最终细胞质心的 µm 坐标上按作者 `ceiling`/`cut` 生成分箱，并保存实际分箱和人数。转录本数、基因数、面积的 CV 按作者实现对 `log10` 后的值计算，SD 使用原值；不擅自改成原值 CV。`unassigned` 是否作为类型参与须明确记录，本轮按作者函数保留字符串类型。作者源码 blob 为 `b5901cc0d9041e3c82006bd8bca3c7687a9da414`，[已核验源码](/home/zwan0726/work/outputs/bidcell_historical_dual_20261013/sources/publish_spatialDiversity_metrics.R)。

## 缓存、重评与完成条件

- scClassify 参考模型不因每次修改模型或 loss 而重新训练。每份新最终表达矩阵需评价一次，不要求每个训练 step 判型。
- 只有计数/ID/基因、QC、归一化、参考模型、分类参数、版本及标签映射的身份一致，才能复用判型。必须核验原收据、文件哈希和实际 QC/标签人口，记录缓存复用；不得仅凭文件名或同一模型名字复用。
- 重评可复用符合上述条件的 scClassify 注释，但应重新计算要求的评分/汇总并核对原结果。原产物缺失时从可验证的压缩备份恢复到新目录，保留原目录和失败记录。
- 最终报告须含 scClassify 主表、固定标签对照、分母及配对敏感性；有必需产物缺失时明确标记未完成。不能仅凭训练、预测或单套工程指标成功宣称最终 evaluate 已完成。
- 保留重评源文件、有效协议、实现哈希、每方法依赖、复用/重算状态、阶段耗时及比较集合。运行前和结果分析前均应读取本文件。

## 来源与历史说明

[BIDCell 论文 Methods](https://www.nature.com/articles/s41467-023-44560-w) 说明对分割细胞使用 scClassify 注释；[作者 CellSPA 教程](https://sydneybiox.github.io/CellSPA/articles/CellSPA.html) 展示 Pearson 示例，两者不能混称。详细作者评分与证据边界继续以 [06_paper_evaluation_protocol_20261008.md](06_paper_evaluation_protocol_20261008.md) 为准，本文件更新的是当前主口径与必需对照输出。

scClassify 已在 E/G/R、COS、ABC、S/SP、BS4 以及本轮执行过。旧文件中的“待安装/待实施”是历史状态。部分历史最终 mask/矩阵已不在原路径；不能凭旧整包 manifest 宣称所有产物仍在。

## 历史重评与本轮三臂汇总产物

2026-10-13初版包含B1、COS四臂、A/B/C、S/SP、P1/P2/T1/T2共14个历史终点。判型缓存经模型、计数、QC、归一化和标签身份核验，作者函数重算六项marker指标。其完整原版本已归档，原14行非Purity值保留。

三臂及B0独立科学审计通过后，加入B0、V1_REGION、V1_BOUNDARY、V2_MORPH，共18方法；复用本轮已验收的计数/QC/判型/评分结果，新增四行空间六指标按同一作者函数及500µm bin规则实算。B1已有一行，不重复追加。

| 表格 | 路径 | 规模 |
|---|---|---|
| 精简主表 | [main_table_compact.csv](/home/zwan0726/work/outputs/bidcell_historical_dual_20261013/main_table_compact.csv) | 18 行、13 列 |
| 完整双口径主表 | [main_table.csv](/home/zwan0726/work/outputs/bidcell_historical_dual_20261013/main_table.csv) | 18 行、50 列 |

- 每套人口与六项finite人数单列。Purity各自使用明确18方法集合，端点见 [purity_scaling.json](/home/zwan0726/work/outputs/bidcell_historical_dual_20261013/purity_scaling.json)；原14集合及本轮五方法集合另保留，不跨集合比较scaled值。
- [同細胞配对](/home/zwan0726/work/outputs/bidcell_historical_dual_20261013/paired_vs_B1.csv) / [类型转移](/home/zwan0726/work/outputs/bidcell_historical_dual_20261013/type_changes_vs_B1.csv) / [覆盖](/home/zwan0726/work/outputs/bidcell_historical_dual_20261013/metric_coverage.json) / [验证](/home/zwan0726/work/outputs/bidcell_historical_dual_20261013/validation.json) / [当前完成收据](/home/zwan0726/work/outputs/bidcell_historical_dual_20261013/complete.json)。
- [原14版本与实际归档身份](/home/zwan0726/work/outputs/bidcell_historical_dual_extension_20261013/archive_manifest.json) / [18方法扩展证据与批准](/home/zwan0726/work/outputs/bidcell_historical_dual_extension_20261013)。最终科学审计仍以各实验的独立记录为准。
- C/SP原计数由已验哈希的压缩备份恢复；其最终mask缺失，几何采用原已核验最终细胞统计，不冒称重测。其余原12方法及新增四方法按实际最终mask计算。原实验输出保持。
- 原14行evaluation_wall_seconds是复用判型后的重评耗时，新四行来自本轮fresh全面evaluate阶段；费用scope不同，不用于模型速度排名。

主表是统一评价结果，跨实验训练初始化/预算、多seed/Rep2和真实owner真值限制保持；科学指标定义与本文件上方冻结规则不变。未来版本更新本节和README入口并归档。
