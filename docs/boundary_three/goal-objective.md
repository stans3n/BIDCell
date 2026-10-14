按照下述分工，完成 BIDCell 三方向实验的代码实现、独立审计、自动化执行准备和后台启动。之后按我的明确要求暂停 goal；等我 resume 后，再分析实验结果并同步主表和 context。

这是执行授权。请持续完成已授权的实施、测试、修复、本地 Git 提交和后台实验启动，不要只给计划，也不要在常规步骤反复请求确认。

一、目标、模型与阶段边界

研究主线：
针对细胞边缘及核外竞争区域的 transcript 归属问题，分别验证 Core 表达证据、P2 后处理裁决和 Morph 形态分支的作用。严格区分信息可预测性、最终指标改善和真实 owner 正确性。

主对话：
- 模型：gpt-6.1-sol
- reasoning_effort：xhigh
- 职责：任务拆分、接口协调、集成、审计问题闭环、实验准入和状态判断。

实现阶段创建三个 subagent：
1. gpt-6-astra / xhigh：A_CORE，完成 A1 和计划规定的条件 A2。
2. gpt-6-astra / xhigh：P2，完成 BASE_MATCH 与旧 V2_MORPH 两底座验证。
3. gpt-6-astra / xhigh：Morph，完成去 DAPI、simple fusion、去新增几何三个变体。

实现集成后：
- 新建一名 gpt-6.1-sol / xhigh subagent，独立审计代码与 changeplan 一致性。
- 审计问题闭环后，新建一名 gpt-6.1-sol / high subagent，完成自动化执行脚本。
- 自动化脚本完成后，由独立的 gpt-6.1-sol / xhigh 审计者复审脚本与最终集成版本。

我 resume 后：
- 新建一名 gpt-6.1-sol / xhigh subagent 分析结果，不能由原实现者的总结代替独立结果分析。
- 主对话复核后完成主表及 context 同步。

创建代理时必须实际指定 model 和 reasoning_effort，并记录真实配置。若当前接口的 full-history fork 不允许覆盖模型，使用有限上下文或不继承历史的方式，提供完整任务说明、文件路径和接口合同。
指定模型不可用时如实说明，不静默替换，也不将同一代理自审称为独立审计。
遵守实际并发上限；实现阶段最多主对话加三个 Astra 活跃代理，后续角色分阶段启动。

建立或沿用一个覆盖上述完整目标的 goal，不自行设置 token budget。后台启动后的暂停不等于目标完成。

二、先读取这些文件

工作区：
/home/zwan0726/work

必读：
1. /home/zwan0726/work/docs/BIDCell_A_Core_Loss_Validation_ChangePlan_20261015.md
2. /home/zwan0726/work/docs/BIDCell_P2_Transfer_ChangePlan_20261015.md
3. /home/zwan0726/work/docs/BIDCell_Morph_Ablation_ChangePlan_20261015.md
4. /home/zwan0726/work/docs/BIDCell_Linear_Execution_Plan_20261015.md
5. /home/zwan0726/work/docs/BIDCell_Three_ChangePlans_Audit_20261015.md
6. /home/zwan0726/work/context/07_evaluation_dual_annotation_20261013.md

背景：
/home/zwan0726/work/docs/BIDCell_Three_Directions_Summary_and_Tonight_Plan_20261015.md

读取适用的 AGENTS.md、实际源代码和输入收据。现有审计是设计自审，不能作为新代码已经通过独立审计或运行验收的证明。

三、分支、隔离与并行开发

先确认真正的 Git 仓库、当前修改和已有产物，不覆盖用户未提交内容。

创建新的集成分支，并为三个实现代理分别准备独立分支和隔离 worktree/checkout。不要让三个代理在同一工作目录中切换分支或同时修改公共文件。
提交指本地 Git commit，记录每个代理与最终集成版本的 commit SHA。

遵循执行 plan 的新源码及输出目录约定：
- 新源码：/home/zwan0726/work/projects/BIDCell_boundary_three_20261015
- 新产物：/home/zwan0726/work/outputs/bidcell_boundary_three_20261015

若目录已有内容，先核验归属和版本，再决定复用或建立明确的新 run ID；历史源码、输入和已验收结果保持可追溯。

主对话先固定公共接口：
- endpoint registry；
- 配置与变体标识；
- 输入/输出及 receipt schema；
- execution_status 与 science_status；
- 评价调用、资源锁和恢复接口。

明确公共文件负责人。每名实现代理交付：
- 实际代码与必要测试；
- changeplan 条目到实现位置的映射；
- 真实 mini 验证结果；
- 偏差、限制、未完成项；
- commit SHA 和集成说明。

实现全部已规定的实验入口；条件实验可以暂不运行，但不能仅留下占位函数后声称实现完成。A3、新 loss 训练、额外架构和扫参不在本轮实施范围。

四、独立代码审计与修复闭环

gpt-6.1-sol / xhigh 审计者必须阅读集成后的实际 diff、关键调用链和测试结果，独立核验，不能只接受实现代理的文字报告。

审计至少包括：
- 是否逐项遵循 changeplan，是否夹带改变科学问题的修改；
- 数据泄漏、坐标/基因/ID 错配、分母和缺失值处理；
- A1 分子拆分、Type/Core/shuffle 对照及固定 gate；
- A2 同域对照、结构约束、固定 capture 和最终矩阵；
- P2 的 native replay、完整核距离、候选不变与 RNG；
- Morph 的初始化映射、实际删除 DAPI、训练/预测一致性；
- 历史复用边界、评价协议、路径隔离和恢复正确性；
- 资源、并发、进程及输出完整性风险。

输出逐项 findings：严重性、文件位置、影响、修复建议及验证方式。
主对话组织修复，审计者复核关闭问题。所有影响科学有效性、数据完整性或无人值守执行的阻塞项解决后，才进入正式启动。
审计结论必须绑定实际 commit、源码/config hash 和验证结果。不要为通过验收而降低冻结科学门槛。

五、由 gpt-6.1-sol / high 完成自动化脚本

自动化脚本必须按执行 plan 严格串行执行完整科学节点：

A1_CORE
→ P2_BASE
→ MORPH_NO_DAPI
→ P2_MORPH
→ A2_CORE（条件）
→ MORPH_SIMPLE_FUSION
→ MORPH_NO_GEO
→ 汇总与打包

前三项固定，A2 不插队。P2_MORPH 使用已有 full V2_MORPH，不是去 DAPI 变体。

将预定条件直接写入普通程序：
- A1 合法阴性不阻断 P2/Morph；
- P2_BASE 科学阴性不自动取消 P2_MORPH；
- A2 仅在 A1=GO_A2、实现已准入且预算允许时运行；
- 预算不足、科学跳过、工程失败分别记录；
- 工程失败按既定策略停止/恢复，不能伪装成科学阴性；
- 新 mask 必须完成预测/后处理、own matrix、完整评价和必要诊断，才能标记实验完成。

后台执行不得依赖 AI 持续在线、逐阶段发命令或临时判断。暂停 goal 后，普通后台进程仍应按冻结规则继续。
科学 gate、预算判断、跳过、阶段切换、有限重试和最终打包均须自动完成。

实现并验证以下命令接口：
- prepare / validate / launch
- status：人类可读
- status --json：机器可读
- tail：查看日志
- resume：恢复后台实验队列，与恢复 Codex goal 明确区分
- complete-check：核验完整产物
- 安全停止接口

status 至少显示：
run ID、host、进程是否存活、当前实验/阶段、heartbeat 与最后进展时间、训练更新或 tile 进度、已完成/延期/跳过节点、失败原因、日志路径。
能够区分“进程存在但没有进展”“进程已退出”“收据不完整”，不能仅检测 PID。

使用 tmux 或经过验证的独立后台机制，持久保存状态、日志、checkpoint、锁与原子收据。继承执行 plan 的 GPU 锁、资源限制、恢复和预算规则。
预算按真实全链成本验收；8 小时只是 plan 的初始参数，不伪造一晚全部完成的承诺。

脚本提交后，由 gpt-6.1-sol / xhigh 审计者再次审计最终集成版本，重点检查：
- 串行顺序和科学 gate；
- 阴性、失败与延期的状态传播；
- 锁、后台脱离、恢复和重复启动防护；
- 完成条件、状态命令和输出身份；
- 自动执行中没有隐含 AI 依赖。

六、启动后台后暂停 goal

通过最终审计、真实 mini 和资源/预算准入后，直接启动已授权的正式后台队列。

暂停 goal 前必须完成：
1. 代码及脚本已本地提交，记录最终 commit SHA。
2. 审计与准入绑定实际将执行的版本。
3. 后台进程已真实启动，首个正式阶段出现有效进展。
4. 从独立 shell 调用 status、status --json 和日志命令，确认可用。
5. 后台不依赖当前 AI 工具调用维持生命周期。
6. 写好持久化交接文件，记录分支、源码/输出根、进程/session 身份、运行命令、审计、状态接口、实验清单和 resume 后待办。

向我交付可直接复制的实际命令，使用完整路径，不使用尚未验证的占位命令。

满足这些条件后，我明确要求你：
- 调用 goal 暂停工具，将 goal 设置为 paused；
- 报告工具实际返回的状态；
- 停止 AI 工作并结束本轮；
- 保留后台实验进程运行；
- 不继续轮询、不安排 AI 自动唤醒、不提前做最终结果分析；
- 不将 goal 标为 complete。

“脚本文件写完”或“commit 成功”本身不足以到达暂停点。后台启动失败时先修复，不能报告已经成功交接。

七、我 resume 后的工作

读取交接文件和实际队列状态，检查产物、receipt、hash、日志及完成条件。
区分全部完成、科学性跳过、时间延期、工程失败；不要重复启动已完成实验。未完成项如实保留，不能用部分结果冒充完整结论。

创建新的 gpt-6.1-sol / xhigh 结果分析 subagent，由其独立完成：
- A1 信息增量、覆盖及是否支持 A2；
- A2 四组实际采用价值；
- P2 两底座及四格比较；
- Morph 去 DAPI、simple fusion、去新增几何的贡献；
- scClassify P/N、capture、RNA 保留、结构、QC 人口变化、空间不确定性；
- 改善是否发生在预定义竞争域；
- 哪些结论仅属探索，哪些需要重复或独立 owner 参考；
- 三方向是否值得融合及最小有效组合。

评价主次沿用冻结协议，阴性和权衡结果同样保留，不只汇报有利端点。

主对话复核分析和数据后，已获授权完成结果同步：
- 更新经过核验的正式主表及相关汇总/registry；
- 更新 /home/zwan0726/work/context/experiments/ 下本轮记录；
- 按现有组织方式更新 context 的索引和当前结论；
- 保留旧版本、来源、评价身份及实际完成状态；
- 保持历史原始指标不被误改；新增比较集合涉及 Purity 重算时记录新版本和缩放端点；
- A1 的信息检验进入研究记录，不伪装成具有 mask/F1 的新模型端点；
- 失败、延期或未评价端点不填虚构分数、不标完成。

同步后检查主表、报告、context 和实际产物的一致性，并提交对应修改。
只有应完成的实验、独立结果分析、主对话复核和同步均完成后，才把整体 goal 标记为 complete。