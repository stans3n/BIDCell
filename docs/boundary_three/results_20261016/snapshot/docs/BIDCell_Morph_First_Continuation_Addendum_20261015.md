# Morph 优先的实验接续补充计划

日期：2026-10-15，Australia/Sydney。

本补充计划响应本次侧对话中“先完成剩余 Morph 消融，再完成 A2；人工分析在另一对话进行”的要求。用户已明确选择“允许连续完成，采用上述有界窗口”，批准 Morph 最多 16 小时、随后 A2 最多 32 小时。状态：后台接续已启用，按下述顺序自动接续；当前阶段以接续目录中的 state.json 为准。

## 目标与实际顺序

当前运行 `boundary-three-20261015-r02` 的 P2_MORPH 自然完成后，按以下顺序接续：

1. MORPH_SIMPLE_FUSION：检验简单融合能否保留形态模块收益，区分输入信息与 attention 结构贡献。
2. MORPH_NO_GEO：检验去掉新增几何分支后的变化，与已完成的 NO_DAPI 共同解释信息来源。
3. A2_CORE：沿用 A1 已通过的 GO_A2 条件和冻结适用范围，完成 NATIVE、GT、GTC、SHUFFLED 的比较。

每项仍完成自身训练（如适用）、预测、后处理、表达矩阵、完整评价和完整性检查。人工科学分析不作为接续门槛；不自动更新正式主表或 context，不调用或恢复主线程 Codex goal。

不重跑已完成的 A1_CORE、P2_BASE、MORPH_NO_DAPI；不打断当前 P2_MORPH。保留既有训练预算、seed、科学规则、评价口径及失败处理。

## 使用现有接口接续

原配置及原计划文件均被当前运行的身份收据绑定，不原地改写。使用现有 `resume --budget-hours` 的两个执行窗口实现实际顺序：

- Morph 窗口最多 16 小时。A2 启动门槛为 30.1361 小时，在此窗口暂缓；随后执行 SIMPLE_FUSION 和 NO_GEO。每个 Morph 的启动门槛为 7.375 小时；两项冻结估计合计 11 小时。
- 两项 Morph 均完成且当前执行窗口退出后，单独给 A2 最多 32 小时。已完成节点由原程序验证复用；A2 的科学条件和资源准入仍由原程序检查。
- 两个窗口均在任务完成后立即退出，不占满预算。16/32 小时是上限和准入安排，不是实测完成时间。它们最多新增 48 小时执行预算；若超出估计、资源不足或发生错误，不自动追加窗口。

接续程序仅在上一窗口自然完成、supervisor/worker 均退出且队列锁释放后调用正式 resume 接口。遇到用户停止、科学/完整性失败、非预期外部续跑或配置身份变化即停止接续，不覆盖停机请求、不重试失败科学节点、不并行启动第二个实验。

## 文件与状态

- 科学配置：`/home/zwan0726/work/outputs/bidcell_boundary_three_20261015/boundary-three-20261015-r02/configs/run.json`
- 接续目录：`/home/zwan0726/work/outputs/bidcell_boundary_three_20261015/continuation_morph_first_20261015`
- 接续程序：该目录中的 `continue_queue.py`。
- 窗口与授权状态：该目录中的 `continuation.json`；现为 `APPROVED_TIME_WINDOWS`，含用户原话及记录时间。
- 启动后接续状态与日志：该目录中的 `state.json`、`resume_morph.log`、`resume_a2.log`。
- 后台 session：`boundary-three-morph-first-20261015`；启动核验见该目录中的 `launch_receipt.json`，记录真实 PID、进程身份、锁及原实验保持运行的状态。
- 单独创建接续目录中的 `cancel` 文件可取消未来接续；不终止正在运行的实验。正在运行的实验仍使用原 `safe-stop` 接口。

只读检查：

```bash
python3 /home/zwan0726/work/outputs/bidcell_boundary_three_20261015/continuation_morph_first_20261015/continue_queue.py --check
```

用户已明确允许延长。接续进程在当前窗口自然完成后调用正式 resume 接口，不再受原窗口今晚 23 点截止限制；每个新窗口仍受各自 16/32 小时上限约束。

## 核对依据

原 runtime 支持在不修改科学配置的情况下，以新预算 resume；已完成节点核验复用，DEFERRED_TIME 在 resume 时重新检查。其完整节点时间准入仍为 `estimate * 1.25 + 1800s`。本补充调整执行窗口和实际优先级，不改算法或伪造新科学准入。

来源：`/home/zwan0726/work/projects/BIDCell_boundary_three_20261015/scripts/boundary_three/runtime.py` 的 `launch` 与 `run_queue`；原 `/home/zwan0726/work/docs/BIDCell_Linear_Execution_Plan_20261015.md` 保留为原始冻结记录。
