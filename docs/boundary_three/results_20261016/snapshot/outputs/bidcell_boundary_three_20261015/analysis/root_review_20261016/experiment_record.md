# BIDCell Core / P2 / Morph 三方向（2026-10-15执行；2026-10-16结果验收）

本轮7个节点已经正式 complete-check 逐项核验，全部 COMPLETED，无失败/延期/跳过。接续程序自然完成，未重复启动；用户批准的顺序是 P2_MORPH → SIMPLE_FUSION → NO_GEO → A2，Morph共享16h、随后A2独立32h，实际完成即退出，未追加窗口。全部完成时间为2026-10-16 01:29:14 Australia/Sydney。

科学执行保持 commit `ccb846cbf344bfbe965178cd8f9b738e3654c8b0`。计算结束后按要求新建独立 `gpt-6.1-sol/xhigh` 结果分析者（实际显式指定，fork_turns=none，spawn accepted），与原实现者/代码审计者不同。独立680项检查通过；主对话另从十个唯一端点的完整整数矩阵、原始逐细胞分数/注释复算QC与两套六指标，检查A2逐基因总量守恒；独立复算A1主值/三尺度区间，再交叉核验336个点值/共同人口项、56个双口径主尺度区间及12个B0区间。完整mask/候选/owner语义由正式complete-check覆盖，不把主表算术核对冒充重新跑全部历史实验。

按用户2026-10-16新增指令，**fixed与scClassify的正、负F1并列评估**，任一口径突出均明确注明；P↑、N↓，不能用其中有利的一套抹去另一套。现行结果汇总协议为 `DUAL-F1-v1.0-28methods-20261016`。程序原先sc主/fixed辅的判定是冻结计算记录，保留原值，不冒称已经按新双口径重跑gate。计分模型、QC、原始六指标、缺失规则与科学门槛均未改变。

每端点保留原107955 ID；capture分母34442716。scClassify按各自ownQC人口；固定历史面板103000 ID，F1有限85421，其他保留NA。已知零表达按作者合法F1=0/Recall=0、Precision可能NA处理；缺行/不可用标签不补0。两口径人口不同，不比较跨口径绝对F1大小；另列commonQC及两方法×两口径共同有限ID诊断。


## 实际端点与对照

| 端点 | sc P↑ | sc N↓ | fixed P↑ | fixed N↓ | capture % | sc QC n | 8邻接不连通 |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.326633914 | 0.025728133 | 0.399806382 | 0.034806660 | 74.635662 | 103299 | 92 |
| BASE_MATCH | 0.334076985 | 0.023892621 | 0.402020353 | 0.031433682 | 62.407750 | 103286 | 85 |
| V2_MORPH | 0.335381716 | 0.023967640 | 0.402985940 | 0.031126313 | 62.276207 | 103282 | 64 |
| P2_BASE | 0.335098193 | 0.024058134 | 0.403218569 | 0.031374627 | 62.870837 | 103368 | 23 |
| P2_MORPH | 0.335798609 | 0.024008838 | 0.403853686 | 0.031100467 | 62.668754 | 103348 | 15 |
| MORPH_NO_DAPI | 0.335006605 | 0.023939880 | 0.402786658 | 0.031334241 | 62.014206 | 103262 | 66 |
| MORPH_SIMPLE_FUSION | 0.334599599 | 0.023879991 | 0.402719991 | 0.031259244 | 61.818891 | 103240 | 69 |
| MORPH_NO_GEO | 0.334310121 | 0.023848212 | 0.402064649 | 0.031409326 | 62.406478 | 103252 | 61 |
| A2_GT | 0.336355543 | 0.023754367 | 0.403778875 | 0.030771086 | 62.407750 | 103286 | 66 |
| A2_GTC | 0.335340713 | 0.023877425 | 0.403032807 | 0.030959080 | 62.407750 | 103286 | 72 |
| A2_SHUFFLED | 0.335983766 | 0.023801446 | 0.403655335 | 0.030826992 | 62.407750 | 103286 | 67 |


论文结果可用B0作历史主参照；每项BASE_MATCH相关实验同时给出B0和BASE_MATCH，A2仍保留GT/SHUFFLED机制对照。相对B0，本轮所有上述输出在两套P/N F1都改善，1000µm主区间均有利，但capture下降11.7648–12.8168pp、面积缩小21.60–23.73%。BASE_MATCH本身对B0已改善sc P/N +0.007443072/−0.001835512、fixed +0.002213971/−0.003372977，且capture已下降12.227911992pp。B0初始化/训练历史/访问顺序没有严格匹配，不能将这些共同背景差异全部归因于Core、P2或Morph。

## A1与A2

A1有88102配对细胞（全体81.6099%）、88106可评价细胞。Core−Type=0.239635648 nats/分子（1000µm 95% CI [0.216328094,0.263690909]）；Core−20shuffle均值=0.326542687（[0.316705801,0.337419563]）。两方向、各shuffle及500/2000µm均支持条件信息增量；十类通过原类型内门槛，其他类保持inconclusive。给定可能使用全核信息的历史类型标签、分子split并非独立组织，不能据此宣称真实核外owner正确。

A2_NATIVE复用BASE_MATCH；GT/GTC/SHUFFLED在相同1317991像素/2940738 G分子的合法域内操作，核/前景保持，assigned均21494924，逐基因总量也相同，QC进入/退出为0。GTC−GT的sc ΔP/ΔN=−0.001014829/+0.000123058，fixed=−0.000746067/+0.000187993；四个主尺度CI均不利，且不连通66→72。GTC也劣于SHUFFLED，故当前cell-specific Core评分未提供特异增量。

GT−native的sc +0.002278557/−0.000138254、fixed +0.001758521/−0.000662596，四个主区间均有利；fixed负F1改善在本轮新输出对native中最突出。不过Myofibroblasts/Endothelial/Mono-Macro的RNA retention约0.962530/0.974211/0.977627，未通过局部0.995护栏。最小已测候选为“BASE native + 本冻结Core可用域内G+T修正”；GT评分不使用cell-specific q，但域依赖A1 Core覆盖、适用类型和donor，不是完全脱离Core的全域方案。A2实际节点wall8614.60秒；原85352秒/23.71h是保守规划估计，不是实测全程。

## P2与Morph：突出fixed收益，也保留冲突

P2_BASE−BASE的sc ΔP/ΔN=+0.001021208/+0.000165513，fixed=+0.001198216/−0.000059055；P2_MORPH−full的sc +0.000416893/+0.000041198，fixed +0.000867747/−0.000025846。两底座fixed positive主CI均正，negative小幅改善的CI均跨零；sc negative代价仍在。capture分别增加0.463087754/0.392547440pp，不连通85→23及64→15。最终增捕含清岛/merge的后续变化，不能说冲突裁决本身新增前景。四格fixed positive交互−0.000330469（CI [−0.000623533,−0.000016500]），支持Morph底座上的P2增量减弱；fixed negative交互CI跨零。sc两个主交互区间均跨零且尺度敏感，未建立协同。

**完整Morph新增几何的fixed贡献最清楚。** full−NO_GEO的fixed ΔP=+0.000921291（CI [0.000624857,0.001213990]）、ΔN=−0.000283013（[−0.000426068,−0.000135645]），三个尺度均有利；sc同时增加P和N，full不连通比NO_GEO多3个。full−NO_DAPI的fixed N改善−0.000207928（CI [−0.000355426,−0.000044700]），fixed P主CI跨零。full−BASE的fixed两项亦有利，而sc呈权衡；不能只按sc点值删去有fixed贡献的几何或DAPI分支。这里NO_DAPI去掉直接图像分支，核图/几何仍有历史DAPI来源；NO_GEO只去新增几何encoder，原有几何先验仍在。

simple fusion−BASE在fixed两项主CI均有利，但capture−0.588858904pp超过0.5pp冻结护栏；sc N从ownQC−0.000012630变成commonQC+0.000016634。full−simple两个fixed主CI跨零，不能证明两者等价，也未证明复杂attention必要。NO_GEO−BASE四项F1主CI均跨零，高RNA组retention0.965442；保留探索信号及局部代价。

本轮十个候选中P2_MORPH的fixed P最高（0.403853686），A2_GT的fixed N最低（0.030771086），是固定口径内部的权衡。未刷新全28方法绝对纪录：历史B1 fixed P/N仍为0.408445355/0.028323037。历史方法的训练也非全部严格配对，此处“突出”均指明确命名的比较，不据横比宣布机制因果赢家。

## 竞争域、人口与不确定性

A2实际变化100%在冻结竞争域；P2两底座约99.52%/99.60%，三Morph对BASE约94.56%–95.47%。这是changed molecule的操作域分布，不是owner准确率或改进比例。full−BASE的commonQC标签变化11396个；P2_BASE/P2_MORPH的QC进入/退出118/36及100/34，三Morph对BASE为200/224、186/232、167/201。固定人口和sc重判型在相同共同ID上仍可反向，不能只用人口不同解释全部分歧。

主尺度1000µm为48块，敏感尺度500/2000µm为176/12块；2000次PCG64、mask seed20261013/A1 seed20261015，各主区间有效次数2000。LOBO、所有尺度、commonQC、RNA保留/结构局部护栏与NA分母均保留。局部受损分组互相重叠，不是独立受损细胞人数；空间CI不代表跨seed、跨样本重复，也未校正全部探索比较的多重性。单Rep1/seed、历史full初态重建、变体容量变化、没有独立owner/边界真值仍限制推广。

仅fullMorph×P2四格已测；A2+Morph、A2+P2、三者叠加未测，不能相加单项收益。没有消除双口径分歧与局部代价的统一默认赢家；不自动追加A3、loss、扫参或新窗口。

## 正式结果与版本

正式主表扩为28个唯一mask方法：历史20原始非Purity字符串逐项保留，增加2个P2、3个Morph、3个A2。A1仅进入研究记录，A2_NATIVE仅是别名。旧20目录是冻结输入，全部原样保留；当前28使用新目录，避免使冻结receipt失效。Purity集合版本`PURITY-28methods-v1.0-20261016`；两套缩放端点分别记录在purity_scaling.json。本次新增值未超出旧端点，所以缩放数值未因集合扩大而变化；仍必须记录新集合，不能混用pairwise Purity。

- [完整28方法主表](/home/zwan0726/work/outputs/bidcell_historical_dual_boundary_three_20261016/main_table.csv) / [精简主表](/home/zwan0726/work/outputs/bidcell_historical_dual_boundary_three_20261016/main_table_compact.csv)
- [含B0的本轮端点表](/home/zwan0726/work/outputs/bidcell_historical_dual_boundary_three_20261016/boundary_endpoints_with_B0.csv) / [双口径全部对照差值](/home/zwan0726/work/outputs/bidcell_historical_dual_boundary_three_20261016/dual_profile_comparisons.csv)
- [当前评价政策](/home/zwan0726/work/outputs/bidcell_historical_dual_boundary_three_20261016/assessment_policy.json) / [协议](/home/zwan0726/work/outputs/bidcell_historical_dual_boundary_three_20261016/protocol.json) / [Purity端点](/home/zwan0726/work/outputs/bidcell_historical_dual_boundary_three_20261016/purity_scaling.json)
- [矩阵/评价/历史来源registry](/home/zwan0726/work/outputs/bidcell_historical_dual_boundary_three_20261016/artifact_registry.json) / [正式发布验收](/home/zwan0726/work/outputs/bidcell_historical_dual_boundary_three_20261016/complete.json)
- [独立结果分析](/home/zwan0726/work/outputs/bidcell_boundary_three_20261015/analysis/independent_20261016/independent_analysis.md) / [B0全部空间区间](/home/zwan0726/work/outputs/bidcell_boundary_three_20261015/analysis/independent_20261016/B0_spatial_uncertainty.csv)
- [主对话原始复算](/home/zwan0726/work/outputs/bidcell_boundary_three_20261015/analysis/root_review_20261016/raw_validation.json) / [独立结果交叉核验](/home/zwan0726/work/outputs/bidcell_boundary_three_20261015/analysis/root_review_20261016/independent_crosscheck.json)
- [主对话复核](/home/zwan0726/work/outputs/bidcell_boundary_three_20261015/analysis/root_review_20261016/final_review.json) / [同步核验](/home/zwan0726/work/outputs/bidcell_boundary_three_20261015/post_resume_20261016/verification.json)
- [冻结计算完整性](/home/zwan0726/work/outputs/bidcell_boundary_three_20261015/boundary-three-20261015-r02/complete-check.json) / [旧20版本](/home/zwan0726/work/outputs/bidcell_historical_dual_20261013/main_table.csv)

结果与context逐字快照、本次脚本在分支`results/boundary-three-20261016`提交；科学运行分支保持冻结commit。完整矩阵/checkpoint留在registry所列路径，未复制进Git或清理。
