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
# 2026-10-12 规则澄清：作者 marker 零分优先于冲突的 BS4 计划

当前权威说明见 [指标框架顶部的作者 marker 补零规则](/home/zwan0726/work/context/04_metrics_framework.md:1)。本文件第 3.2–3.3 节此前已记录作者 F1 NA→0 和未匹配类型 P/R/F1 全零行为；本次进一步明确已知类型零表达的边界，并按用户指令纠正 BS4 的冲突计划。

作者 `.calculate_marker_purity` 对匹配类型的实际计算结果只显式执行 `f1[is.na(f1)] <- 0`，没有统一填补 P/R。对评价表达全零、marker 集合非空的已知类型细胞，结果为 **F1=0、Precision=NaN、Recall=0**；这两个 0 合法，不能直接称为“错误评分”或“误纳细胞”。未匹配 marker 类型但具有完整输入行的细胞，另由作者分支追加 P/R/F1 全零。缺文件、缺行和损坏数据不属于这些补零规则。

BS4 第 8.1–8.2 节及 report_stage_na 恢复逻辑若仅凭零参考计数把六项指标全部改 NA，与本次明确规则冲突时应修正。当前 BS4 的已知类型阶段比较保留作者有效零分，按各指标真实有限值分别确定四端点域；保留 Precision 的 NaN，不将其扩散到 F1/Recall。原固定标签不可评价人口、paper ownQC/注释和其他历史 profile 不在本次被静默合并或重定义。

证据为下文 C2 的归档作者源码（SHA256 `2d7f0e0da83fed939e5cc2316ea159f5962477e214d00685c73f0f01142775ec`）。2026-10-12 已直接调用原函数验证上述零表达与未匹配分支。保留原计划、失败恢复和原始结果；后续仅对受影响评价/报告以新版本核验，不因这两个合法零分要求重训或认定实验失败。本段记录规则修订，不签署任何尚未完成的实验验收。

<!-- AUTHOR_MARKER_ZERO_RULE_20261012_END -->

---

# BIDCell / CellSPA 评价规则：已核验证据与执行边界

更新：2026-10-08，Australia/Sydney。适用：后续 CRGF E/G/R 及已有 B1/GR 的统一重评。本文件补充 [冻结指标框架](/home/zwan0726/work/context/04_metrics_framework.md)，不修改旧实验的数值、NA 或 annotation profile。

**当前结论：Purity 的计算公式、论文表格的缩放端点及官方未匹配类型补零行为已有证据；匹配的 scFFPE 参考也已取得。旧文档“正式 Purity 缺定义”“没有 reference detection 输入”的描述是当时状态，不能再作为永久缺口。新 evaluator、scClassify 注释和统一重评仍待实施。**

## 1. 证据级别与来源

- **作者明确**：论文 Methods、官方仓库源码或官方教程直接说明。教程默认值不能自动等同于原论文全部图表的运行配置。
- **数值重建**：由作者发布的 Source Data 重算吻合；只能确认数值支持的内容，不能倒推出唯一原始程序。
- **项目协议**：为了 E/G/R 可重复比较而选择的设置，必须独立标识，不能冒称作者已规定。
- **待执行**：输入/定义已有，尚未接入并运行本地 evaluator。不是“数据未知”。

| ID | 原始来源 | 本地证据 |
|---|---|---|
| P1 | [BIDCell 论文，Methods 与公式 22](https://www.nature.com/articles/s41467-023-44560-w) | 文献身份：Nature Communications，2024，DOI 10.1038/s41467-023-44560-w |
| P2 | [作者 Source Data 工作簿](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41467-023-44560-w/MediaObjects/41467_2023_44560_MOESM4_ESM.xlsx) | [source_data.xlsx](/home/zwan0726/work/outputs/bidcell_metric_protocol_review_20261008/source_data.xlsx)，主要使用 SuppFig4、SuppFig13、Fig3c |
| P3 | [论文补充材料](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41467-023-44560-w/MediaObjects/41467_2023_44560_MOESM1_ESM.pdf) | [supplement.pdf](/home/zwan0726/work/outputs/bidcell_metric_protocol_review_20261008/supplement.pdf) |
| C1 | [CellSPA publish/processing.R](https://github.com/SydneyBioX/CellSPA/blob/publish/R/processing.R) | [源码副本](/home/zwan0726/work/outputs/bidcell_metric_protocol_review_20261008/publish_processing.R) |
| C2 | [CellSPA publish/expression_metrics.R](https://github.com/SydneyBioX/CellSPA/blob/publish/R/expression_metrics.R) | [源码副本](/home/zwan0726/work/outputs/bidcell_metric_protocol_review_20261008/publish_expression_metrics.R) |
| C3 | [CellSPA 官方教程](https://sydneybiox.github.io/CellSPA/articles/CellSPA.html) | [固定版本教程](/home/zwan0726/work/outputs/bidcell_metric_protocol_review_20261008/CellSPA_vignette_pinned.Rmd) |
| C4 | [作者 scFFPE reference](https://raw.githubusercontent.com/SydneyBioX/CellSPA/publish/inst/extdata/sce_FFPE_full.rds) | [sce_FFPE_full.rds](/home/zwan0726/work/outputs/bidcell_metric_protocol_review_20261008/sce_FFPE_full.rds) |
| V1 | 本地数值/文件核验 | [verification.json](/home/zwan0726/work/outputs/bidcell_metric_protocol_review_20261008/verification.json)、[Purity 重建结果](/home/zwan0726/work/outputs/bidcell_metric_protocol_review_20261008/purity_minmax_reconstruction.json) |

作者 publish 树记录为 `b780c7cd589880e0f916b3b4a318db1862b13a7a`；这是 **tree SHA**，不得写成 commit SHA。源码副本以 V1 的 SHA256 为准。现有工程环境 CellSPA pin 为 `1e892c470177fcea351c9175a9153e5480e93646`。两个版本的相关 correlation 调用存在 `use_nan=FALSE` 差异；实施时保存实际版本与等价性检查，不能混用无记录。

## 2. 数据、注释与 QC

### 2.1 训练参考与评价参考分开

论文训练使用 TISCH 乳腺数据参考；生物学评价使用匹配的 Chromium scFFPE。现有模型训练参考、核预注释及其历史行为在 E/G/R 保持不变。评价换用匹配 scFFPE 不等于更改训练监督。[P1、C3、C4]

已取得的 C4 含 30,365 个细胞、307 个 genes、17 个 `graph_cluster_anno` 类型，仅含 `logcounts` assay。已有两个 307×17、无非有限值的派生表：

- [reference_mean_logcounts.csv](/home/zwan0726/work/outputs/bidcell_metric_protocol_review_20261008/reference_mean_logcounts.csv)
- [reference_detection_proportions.csv](/home/zwan0726/work/outputs/bidcell_metric_protocol_review_20261008/reference_detection_proportions.csv)

原空间矩阵保留全部 313 genes；与参考比较使用固定交集 307 genes，按名称对齐。参考缺少 `AKR1C1, ANGPT2, BTNL9, CD8B, POLR2J3, TPSAB1`。不能补零冒充真实参考检测率，不能因此删除训练输入六个基因。CD8B 在空间矩阵存在时，邻域污染图仍可使用。

作者类型名中的 `Firoblast` 拼写要在显式映射表中处理，不能靠模糊匹配静默合并。输出原始标签、标准化标签、映射版本、未匹配数。

### 2.2 注释方法

论文说明使用 **scClassify** 注释空间细胞；官方 CellSPA 教程还展示按 reference Pearson 最大值给类型的流程。两者不是同一 annotation protocol。现有核预注释与 Windows-style max-Pearson 工程 profiles 均不应直接改名为论文 scClassify 结果。[P1、C2、C3]

下一次论文对齐重评应训练/固定同一 scClassify reference 模型，对五个方法各自 QC 后的表达做注释，保存版本、训练设置、预测层级、置信/拒识输出和逐细胞标签。现有 R 环境尚未安装 scClassify；属于可由执行 agent 完成的环境工作。论文没有提供本项目可直接复用的全部运行配置；用官方可复现设置时需记录为本地复现配置，不能声称逐个参数均与作者历史运行一致。

对 fixed-ID 机制诊断，可另用固定核标签，但必须单列 profile，不能混入主表。方法导致的重新注释应报告转移表、unassigned 比例和每类细胞数量，防止只凭换标签解释质量改善。

### 2.3 空间 QC 与归一化

`processingSPE` 的官方默认值及教程明确使用：

```text
20 <= total_transciprts <= 2000
20 <= total_genes <= Inf
随后 scater::logNormCounts
PCA/UMAP seed=2023（属于降维展示，不是模型训练 seed）
```

比较符号含两端，字段 `total_transciprts` 保留作者拼写。教程中的 `subset(..., 1:500)` 是演示截取，生产评价禁止照搬。[C1、C3]

每个方法对自己的完整 313-gene matrix 计算 QC，再在统一参考交集上计算相关生物指标。保存 QC 前后数量、排除原因、gene intersection 和 normalization size factors。五组阈值相同不代表最终通过 QC 的细胞集合相同；另报共同通过集合的配对诊断。capture 的全部 transcript 分母不改为 QC 通过细胞的分母。

**证据边界**：上述阈值是官方代码/教程明确值，不能据此保证论文所有 figure 都用了同一阈值。作者 RDS 的 30,365 cells 与论文 Source Data 某些图中的 Chromium 22,294 cells 也不是同一个已核验 QC 人口。RDS 只有 logcounts，无法从它精确重建原 raw-count QC；若追求原图逐数复现，需补原始数据和具体筛选步骤。

### 2.4 reference 汇总

`processingRef` 在 logcounts 上按参考类型计算：`mean` 和 `prop.detected`。前者是细胞级 logcounts 的平均，不是先平均 raw counts 再取 log；后者是该类型中检测到该基因的细胞比例。作者提供的 reference logcounts 不再当 raw counts 重新 normalize。[C1]

## 3. Marker、Unknown 与汇总

### 3.1 Marker 生成

对参考类型 k，定义 `w_gk = mean_gk - mean_{other types}(mean_g)`，其他类型均值是按类型等权。positive 取 `w > quantile(w,0.9)`；negative 取 `w < quantile(w,0.1)`。严格不等号及 R 默认 quantile type=7 保留。[C1]

删除出现类型数 `>= t * n_types` 的 marker。函数默认 `t=.25`；官方教程 positive 使用 .25，negative 显式使用 **t=1**。不能把 negative 也错误设成 .25。[C1、C3]

### 3.2 每个细胞的 P/R/F1

对细胞所属类型的 marker list，在实际评价 gene 域内构造二值 `actual`。令 marker 占比为 m，对该细胞表达 x 构造 `predicted = x > quantile(x, 1-m)`。positive 与 negative 都使用高表达判断，negative 不反向取低表达。[C2]

```text
Precision = TP/(TP+FP)
Recall    = TP/(TP+FN)
F1        = 2*Precision*Recall/(Precision+Recall)
```

官方实现使用 Metrics 包；计算所得 F1 若为 NA/NaN 则置 0。已计算细胞的 Precision/Recall 非有限值不由这行代码统一置零，必须另报有限值数量与缺失原因。

### 3.3 未匹配类型与 Unknown

官方 `.calculate_marker_purity` 对**没有进入任何 marker 类型计算的细胞**追加全零行，包含 F1、Precision、Recall，并恢复原细胞顺序。[C2，作者明确]

因此，字符串 Unknown/unassigned 或其他未匹配标签且未被分到 marker 类型时，官方 marker 输出为零。不要把这条规则泛化为任意 R NA 标签都能安全直接送入函数；真正缺失标签需先显式映射为未匹配标签，并通过 fixture 验证。

旧 CRGF 核注释工程 profile 将 1,040 Unknown 保留为 NA，是历史工程选择，不是作者这一函数的规则。旧产物保持不动；新的 `paper_aligned_v1` 使用官方补零行为，同时报告未匹配数和比例。不能只补 negative=0 后宣传更纯，因为 positive 也为零，类型拒识本身仍须解释。

### 3.4 Marker F1 如何汇总

新的项目复现主表对 QC 后全部细胞的 positive/negative F1 分别取**细胞等权算术平均**，包括官方补零细胞。按类型分别报告均值/数量；类型宏平均只能作单独补充，不能替代 pooled mean。

证据强度为 **官方补零代码 + Source Data 数值重建**，不是声称找到了作者最终汇总脚本：

- SuppFig13 的 BIDCell 类型均值直接宏平均为 0.3401217247；排除 unassigned 为 0.3601288850，均不等于论文 pooled positive=0.3710524864。
- 由 Fig3c 类型比例加权可得 known-only positive=0.3711567752、negative=0.0361745922。
- 在总量 103,209 中加入 **29 个零分细胞**可重建 positive=0.3710524863796221、negative=0.036164427738950226，与作者表数值吻合。

“29”是表格代数推断，与官方补零一致；没有逐个核对作者历史标签，不能宣称发现了 29 个具体细胞。[P2、V1]

## 4. Purity F1：公式已闭合，缩放集合须有版本

对一个明确比较集合 M 的每个方法 m，先获得上节的原始均值 Pm、Nm，再分别 min–max：

```text
P'm = (Pm - min_M P)/(max_M P - min_M P)
N'm = (Nm - min_M N)/(max_M N - min_M N)
Purity_m = 2*P'm*(1-N'm)/(P'm + 1-N'm)
```

positive 越大越好，negative 越小越好。[P1 公式 22、P2 数值重建]

### 4.1 已确认的论文数值事实

SuppFig4 的 23 行用以下端点复算，最大绝对误差 `1.3322676295501878e-14`：

| 项 | min | max |
|---|---:|---:|
| positive mean F1 | 0.352258364094892 | 0.377909831704304 |
| negative mean F1 | 0.0258949160179617 | 0.053829046681464 |

BIDCell 表值 0.678834322622687，复算为 0.6788343226226913。Voronoi 对应分子分母均零，表中为 0。端点与组合公式由 Source Data 验证；多个比较集合可以共享这些极值，不能因此证明作者唯一用了“这 23 行”作为运行时集合。[V1]

### 4.2 不可冒称已确认的内容

- E/G/R 的缩放集合由本项目定义；作者不可能预先规定我们的五个方法。
- 新数据/新 reference 不能借用上表极值，从而把新值冒充同尺度论文分数。
- 若整个集合的 P 或 N 无变异，论文表没有提供这一退化场景的唯一规则。R `scales::rescale` 的常见中点处理不能反推作者当时一定使用它。
- 改变比较集合会改变 Purity，即使某方法的原始输出不变。因此原始 P/N 永远同时发布，不能跨批次直接比较重新缩放后的 Purity。

E/G/R 执行计划将 M 固定为 B1/E/G/R/GR，所有输入成功后统一计算；退化范围记 `not_identifiable_zero_range` 并保留 raw P/N，不任意补 0.5。这个退化规则是项目选择，不登记为作者明确规则。

## 5. Correlation、邻域污染与形态的口径

| 指标 | 已核实行为 | 执行要求与边界 |
|---|---|---|
| 单细胞 reference Pearson | `calExpressionCorrelation` 将每个空间细胞与各参考类型向量比较，取最大相关及其类型 | 属于官方代码指标；不等于论文按类型聚合表达的相关 |
| 按类型平均表达 Pearson | 论文描述按类型比较平均表达 | 明确构造空间 logcounts 类型均值与参考 mean 的向量，再算 Pearson；保存每类 n，常量/空类另标原因 |
| 按类型非零比例 Pearson | 空间每类各基因检测比例与 reference prop.detected 比较 | 参考已具备；gene 域、类型域和缺失类保持显式 |
| 官方聚合函数注意事项 | 已核验 `calAggExpressionCorrelation` 的空间侧固定为 `rowMeans(spe_exprs != 0)`，即使传入 `ref_exprs_values='mean'` | 不因函数名就把结果标为平均表达 Pearson；分别保存源码行为指标与论文定义重实现指标 |
| 最近异类邻居污染 | 官方按对方类型最近欧氏距离分箱，表达阈值 0，统计本类细胞中该 marker 的阳性比例 | 使用细胞 centroid 的 µm 坐标；旧核 centroid 工程曲线单列 |
| 形态 | 官方教程生成 polygon/alpha shape 再计算对应几何 | 当前 own-mask raster 面积/连接性是工程结构指标；不冒充全部 polygon 指标 |

邻域距离箱：`[0,10], (10,20], (20,30], (30,40], (40,50], (50,100], (100,max]`，按官方代码处理末端范围与空箱；保存每箱分母、marker 阳性数和距离单位。Fig3e 的 CD4/CD8A/CD8B 曲线与文字其他 marker 描述区分，图复现按该图 marker 清单，不把曲线压成无定义单值。可用经小样本等价验证的 KD-tree，避免创建全片 N×N 距离矩阵。

## 6. 现有工程规则继续有效

- final masks 用各自 owner/counts；共享坐标缓存不共享 owner。processed transcript 坐标除 `.2125` 后按原生取整与 cv2 最近邻等价轴映射查询，不能简化为直接 1 µm round。面积 spacing=1 µm/pixel，不用 `.2125` 换算面积。
- 当前整片 capture 分母为 34,442,716 条已过滤 transcript；报告细胞分配、背景、有效域及计数守恒。107,955 核 ID、313 genes 是本次数据身份，不是所有未来样本常量。
- 同 ID 核包含率 100% 受 native 核恢复约束；mask IoU、边界带变化量、cell→cell 转移不是真值准确率。
- Frozen #19 极端面积阈值、#42 空间不确定性分组是项目自定义护栏，论文不替本项目给出阈值。先发布原始分布和描述性分层，不伪造阈值/独立重复。
- #30 V3 coarse/refined 在当前 CRGF 不适用；#39 稀疏修正效率待 SOR 后适用；#40/#41 Rep2 与 #43 多 seed 尚未运行，属于范围未覆盖。

## 7. 下一次评价交付必须包含

完整 frozen #1–43 与 Tier1B 定义保留在 [context/04](/home/zwan0726/work/context/04_metrics_framework.md)。本文件集中补充已核实的论文/官方规则，不宣称所有自定义阈值与作者历史配置都已确定。E/G/R 的环境→QC→归一化→空间注释→评价操作链、现有/待实现 evaluator 和逐项覆盖要求见 [消融执行计划 §9.1–9.3](/home/zwan0726/work/docs/BIDCell_CRGF_EGR_Ablation_Execution_Plan_20261008.md)。执行时必须输出 `metric_coverage.csv`，逐项区分已计算、实际失败、项目协议待定、未纳入及不适用，不能只给“evaluate 成功”。

`evaluation_protocol.json`、实际包/源码 pin、reference/gene/marker/annotation/QC hashes、每组 QC 通过 cell IDs、逐细胞标签与 P/R/F1、分类型均值/数量、raw P/N、Purity 缩放集合/端点、correlation 的具体定义、邻域曲线及分母、mask/matrix 身份。

同时记录 `evaluation_executed` 与 `paper_protocol_complete`。后者只有注释、QC、参考和必需指标已按声明协议运行并核验才可设 true；即便 true，也不自动等于原论文每张图逐数复现。旧工程 profiles、官方教程 profile、论文对齐 profile 各自命名，不拼接成一个主表。

## 8. 需要用户做什么，以及 agent 可以自行完成什么

| 事项 | 当前状态 | 谁处理 / 是否现在阻塞 |
|---|---|---|
| 启动后续 E/G/R goal | 本轮仅写计划 | 用户在准备运行时发 `/goal`；不用现在确认训练启动 |
| Sol High 主控选择 | 计划指定 `gpt-6.1-sol/high`；文档不能切换实际模型 | 用户在客户端选定；执行者核对可见设置 |
| Astra High 子代理 | 计划指定 `gpt-6-astra/high` | 执行者实际调用；若账号/主机不支持才需用户处理访问，不能静默换模型 |
| 后台完成后恢复 | 用户要求人工恢复 | 用户查询状态后 `/goal resume`，再由 Astra 分析 |
| scClassify/R 依赖 | 参考已在本地；scClassify 尚待安装和小样本验证 | agent 在隔离环境安装并锁版本，无需用户手工配环境；只有实际权限/联网失败才转交 |
| 原论文 raw scFFPE 与精确 QC 人口 | 当前 RDS 只有 logcounts，足够 reference mean/detection；不足以回放 raw QC | 本轮 E/G/R 不要求用户补数据；若要求原图逐数复现，agent 先查公开下载，只有不可公开取得的数据/作者具体配置才需用户提供 |
| E/G/R 的缩放集合/退化约定 | 见 §4.2，是项目设计 | 随执行计划采用并记录；不是要求用户猜作者实现，不改旧 frozen 数值 |
| Rep2/额外 seed/新 dataset | 不在本轮三组范围 | 后续单独扩展时才核对数据路径/资源；当前无需添加 |
| 真实 transcript owner 标签 | 当前没有，也不是原 BIDCell 路线的必需输入 | 不要求用户为本轮临时标注；不报告真值 owner accuracy |

本文件没有要求用户事先重装 CUDA、修驱动、重复下载现有 reference 或手工搬运已存在数据。后续执行前由 agent 检查磁盘/GPU和实际 import 路径；机器当前情况以现场只读检查为准。
