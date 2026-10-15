# 三方向实验结果与context同步（2026-10-16）

七个科学节点、正式完整性检查、resume后新建Sol/xhigh独立结果分析、主对话复核和本地结果同步均完成。独立分析680项检查通过；主对话另外复算十个唯一比较端点的完整矩阵/QC/原始分数，核对336项点值/共同人口、56个双口径空间区间和12个B0区间。

科学执行源码固定在`ccb846cbf344bfbe965178cd8f9b738e3654c8b0`，原科学工作目录未修改。该结果分支`results/boundary-three-20261016`只增加事后分析/同步脚本与精确快照，不改变训练、冻结配置或gate。完整矩阵、checkpoint及owner产物仍留在registry所列路径。

## 当前入口

- [本轮研究记录](/home/zwan0726/work/context/experiments/boundary_three_20261015.md)
- [独立结果报告](/home/zwan0726/work/outputs/bidcell_boundary_three_20261015/analysis/independent_20261016/independent_analysis.md)
- [主对话复核](/home/zwan0726/work/outputs/bidcell_boundary_three_20261015/analysis/root_review_20261016/final_review.json)
- [正式28方法主表](/home/zwan0726/work/outputs/bidcell_historical_dual_boundary_three_20261016/main_table.csv)
- [含B0的本轮端点表](/home/zwan0726/work/outputs/bidcell_historical_dual_boundary_three_20261016/boundary_endpoints_with_B0.csv)
- [双口径对照差值](/home/zwan0726/work/outputs/bidcell_historical_dual_boundary_three_20261016/dual_profile_comparisons.csv)
- [矩阵/评价/历史来源registry](/home/zwan0726/work/outputs/bidcell_historical_dual_boundary_three_20261016/artifact_registry.json)
- [正式发布收据](/home/zwan0726/work/outputs/bidcell_historical_dual_boundary_three_20261016/complete.json)
- [同步验证](/home/zwan0726/work/outputs/bidcell_boundary_three_20261015/post_resume_20261016/verification.json)

按用户在分析中新增的两项指令，fixed与scClassify正负F1并列评估，明确突出固定标签上的收益；B0加入所有BASE_MATCH相关比较，可作为论文历史主参照，同时保留BASE_MATCH的匹配机制比较。原程序sc主/fixed辅的gate作为冻结记录保留，不改写为新的双口径gate。

现行汇总协议为`DUAL-F1-v1.0-28methods-20261016`，底层计分定义/模型/QC保持原冻结协议。Purity单列`PURITY-28methods-v1.0-20261016`的集合与缩放端点；历史20文件全部原样保留。历史20加8个独立mask端点，A1不是mask/F1端点，A2_NATIVE只是BASE_MATCH别名。历史方法`P2`与本轮`P2_BASE/P2_MORPH`为不同方法名，不能混用。

## 结论范围

A1有条件信息增量；A2 GTC在两套F1均未优于GT/SHUFFLED。GT是较强的已测简化候选，但域仍受Core可用性限制，并有局部RNA代价。完整Morph新增几何在固定P/N上有明确条件收益，DAPI的固定negative收益也保留；sc口径有权衡。P2改善fixed positive、capture和连通性，同时存在sc negative代价，未建立协同。simple fusion有固定F1收益，但capture护栏失败。

相对B0的双口径F1改善伴随约12pp capture损失，且B0训练历史未配对；不能全部归因于本轮机制。本轮没有刷新历史全表F1纪录，也没有已验证的三方向联合方案。单Rep1/seed、历史full初态重建与缺乏owner真值继续限制推广；空间CI是场内条件不确定性。

## 可追溯快照

`snapshot_manifest.json`将每个原始绝对路径映射到本Git仓库内的精确快照，保存字节数与SHA256。`snapshot/`包含当前正式表/协议/registry、context、两位分析者的证据和脚本、旧20版本及接续/状态收据；它不是完整矩阵或checkpoint的备份。副本保持原绝对溯源路径，当前磁盘数据与副本一致性在提交前逐文件核验。

`post_resume_20261016/before_publication/manifest.json`记录同步前context及历史20的完整备份；新context只追加当前覆盖段，旧正文原样保留。生产者原始Morph收据里的`MEASURED_AWAITING_INDEPENDENT_ANALYSIS`是计算终态记录，事后科学结论见新正式summary和独立/主对话复核，未篡改原receipt。

复现脚本位于本仓库`scripts/post_resume/`。下列命令均不启动实验队列：

```bash
/home/zwan0726/miniforge3/envs/bidcell_local/bin/python /home/zwan0726/work/projects/BIDCell_boundary_three_results_20261016/scripts/post_resume/boundary_three_root_review.py
/home/zwan0726/miniforge3/envs/bidcell_local/bin/python /home/zwan0726/work/projects/BIDCell_boundary_three_results_20261016/scripts/post_resume/crosscheck_boundary_analysis.py
/home/zwan0726/miniforge3/envs/bidcell_local/bin/python /home/zwan0726/work/projects/BIDCell_boundary_three_results_20261016/scripts/post_resume/publish_boundary_three.py verify
```

前两条为有成本的只读复算，已有通过证据无需为查看结果重复运行。`prepare/publish/snapshot`负责授权的版本化同步及快照，保留备份并检查未知外部修改；不依赖AI续跑科学节点。
