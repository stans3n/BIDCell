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

<!-- THREE_ARM_FINAL_20261013_BEGIN -->
**2026-10-13 三臂正式计算及 B0/B1 重评完成，结果为 V1 阴性、M 有代价的 marker 改善。** 三臂同seed1234新鲜原主干、同3998访问/3625真实更新；18stage完整计算终态，原15个三臂stage验收后复用，未重训。B0原始34,442,716行全量独立验收及Astra High审计通过，仅补原生CSV与新版证据；原计数/owner/源码/三臂receipt保持。

当前主口径继续为scClassify，固定历史103000 ID/85421可评/17579 NA并列。R/B相对B0两口径P下降/N上升；M两口径P升/N降，但capture74.636%→62.276%、面积−23.02%，固定positive Recall微降，pooled marker仍未优于额外训练的B1。B1共同四端点sc N方向反转保留；单Rep1/seed、历史B0初始化/order未配对、无owner GT，不作为全面升级。本轮保留现有默认基线。

[实验记录](/home/zwan0726/work/context/experiments/new_methods_20261013.md) / [完整报告](/home/zwan0726/work/outputs/bidcell_new_methods_20261013/final_analysis/final_report.md) / [最终独立及管理验收](/home/zwan0726/work/outputs/bidcell_new_methods_20261013/final_acceptance.json)。计算完成与科学验收分别记录，最终科学签署状态以外置验收为准；原历史内容保留。
<!-- THREE_ARM_FINAL_20261013_END -->


<!-- BS4_AUTHOR_FINAL_20261012_BEGIN -->
**2026-10-12 BS4来源追加。** 正式source仍 `/home/zwan0726/work/projects/BIDCell_cos_bs4_20261012T013909+1100` commit `1bc74b38e3a1e2ef8798a09c85c4d90ffca199ac`。新author报告恢复 `/home/zwan0726/work/outputs/cos_bs4_20261012T013909+1100/recovery/report_author_marker_v2`，source `94b6dfdfe5ff31a484b89fa60e02945bba35ceb1`；原queue/227证据/计算产物不改。原17stage约12.726653h，报告恢复另计。

[作者规则独立分析](/home/zwan0726/work/outputs/cos_bs4_20261012T013909+1100/tests/final_independent_analysis/author_marker_v1/final_analysis.md) / [逐required最终验收](/home/zwan0726/work/outputs/cos_bs4_20261012T013909+1100/tests/final_independent_analysis/author_marker_v1/final_acceptance.json) / [实验记录](/home/zwan0726/work/context/experiments/cos_bs4_20261012_author_marker.md) / [产物索引](/home/zwan0726/work/outputs/cos_bs4_20261012T013909+1100/tests/final_independent_analysis/author_marker_v1/artifact_index.md)
<!-- BS4_AUTHOR_FINAL_20261012_END -->

<!-- EGR_PROTOCOL_20261008_BEGIN -->
# 2026-10-08 GitHub 与评价参考更新

完整 CRGF commit `f840a8cf92e7ef987f8ee92f14a2e0ee628d9952` 已推送至 `git@github.com:stans3n/BIDCell.git` 的 `feature/crgf-1008-stage1-20261008T031528`。原 checkout/代码 commit 保持不变；此状态取代下文历史“没有 push”的当前解释。

评价审计根：`/home/zwan0726/work/outputs/bidcell_metric_protocol_review_20261008`。作者 `sce_FFPE_full.rds` 及 307×17 reference mean/detection 表已存在；scClassify 安装及新 evaluator 接入待后续 agent 完成，不要求用户重复添加现有数据。版本、来源和人工事项见 [评价规则与人工事项](/home/zwan0726/work/context/06_paper_evaluation_protocol_20261008.md)。

E/G/R 的新分支/worktrees 由后续 goal 按 [E/G/R 执行计划](/home/zwan0726/work/docs/BIDCell_CRGF_EGR_Ablation_Execution_Plan_20261008.md) 创建；本文不把建议路径登记为已经存在的运行环境。
<!-- EGR_PROTOCOL_20261008_END -->

---

<!-- CRGF1008_CURRENT_BEGIN -->
# CRGF-1008 已完成实验的路径与身份 — 2026-10-08

下表来自最终核验记录，不把历史 runner 身份当作仍活动的进程。计算与分析已结束；旧环境/旧方法路径在下方保留。

| 项目 | 实际值 / 证据 |
|---|---|
| run_id / host | `20261008T031528+1100` / `MCV-CGB4` |
| 实验 checkout | `/home/zwan0726/work/projects/BIDCell_crgf_1008_20261008T031528` |
| branch / frozen commit | `feature/crgf-1008-stage1-20261008T031528` / `f840a8cf92e7ef987f8ee92f14a2e0ee628d9952` |
| base commit | `2fae67ad74eaa9ec77b5d6fed6178f7cd9feca9d`；baseline 历史输入保留 |
| Python / Rscript | `/home/zwan0726/miniforge3/envs/bidcell_local/bin/python` / `/home/zwan0726/work/envs/bidcell_eval_r_v1/bin/Rscript` |
| 输出根 | `/home/zwan0726/work/outputs/crgf_1008_20261008T031528+1100` |
| 配置与冻结身份 | [run.json](/home/zwan0726/work/outputs/crgf_1008_20261008T031528+1100/configs/run.json) / [implementation identity](/home/zwan0726/work/outputs/crgf_1008_20261008T031528+1100/source/implementation_identity.json) |
| 最终报告 / 产物索引 | [final_analysis.md](/home/zwan0726/work/outputs/crgf_1008_20261008T031528+1100/final_analysis.md) / [artifact_index.md](/home/zwan0726/work/outputs/crgf_1008_20261008T031528+1100/artifact_index.md) |
| 完成交付 / goal 证据 | [final_delivery.json](/home/zwan0726/work/outputs/crgf_1008_20261008T031528+1100/final_delivery.json) / [goal_completion.json](/home/zwan0726/work/outputs/crgf_1008_20261008T031528+1100/analysis/goal_completion.json) |
| Codex chat ID | `01a11725-1986-7c00-b406-687c75356616`；与 run_id 区分，未暴露独立 CLI session/goal ID |

历史 tmux 名 `crgf-20261008T031528_1100`、runner PID `81888` 和初始训练 PID `81968` 仅为启动审计标识；最终独立检查确认进程结束、锁可获取。`run_status.json` 的 `awaiting_user_analysis` 是后台计算终点，后续分析已由 `final_delivery.json` / goal `complete` 闭合；不要因此重复启动队列。

[完整实验记录](/home/zwan0726/work/context/experiments/crgf_stage1_20261008.md)；[启动时交接](/home/zwan0726/work/outputs/crgf_1008_20261008T031528+1100/launch_handoff.md)保留为历史时点。没有 push/merge、改写 baseline 或分析期间改实验源码。
<!-- CRGF1008_CURRENT_END -->

---

<!-- MASKALIGNED_A_CURRENT_BEGIN -->
Current A source: `/home/zwan0726/work/projects/BIDCell_maskaligned_a`, branch `feature/maskaligned-a-pilot`, clean local commit `c93d39d3d0261823dd5363e9672debf865189708`, direct parent `2fae67ad74eaa9ec77b5d6fed6178f7cd9feca9d`. Pilot commands used `/home/zwan0726/miniforge3/envs/bidcell_local/bin/python -B`, explicit worktree cwd and `PYTHONPATH` pointing to this worktree; actual imports were verified. Executed provenance remains baseline HEAD plus source/dirty hash `17010a641231ebc113bf9869660cfec8719cebe8ede24341c00fa29bf246e639`; the later commit matches all executed scientific source files. Outputs: `/home/zwan0726/work/outputs/maskaligned_a_pilot_20261007T035938Z/revision2`; parent contains preserved failed/superseded artifacts, command ledger and delivery. Original checkout, its two untracked files and baseline artifacts are retained. Current-format full-field shared cache projects 102.16 GiB before other outputs; no full-field job ran. [Source equivalence](/home/zwan0726/work/outputs/maskaligned_a_pilot_20261007T035938Z/revision2/committed_source_equivalence.json).
<!-- MASKALIGNED_A_CURRENT_END -->

# Current verified paths and environment — 2026-10-07

| Role | Current verified path/state |
|---|---|
| Original checkout | `/home/zwan0726/work/projects/BIDCell`; feature/geometry-ambiguity-v1 @ 58ef3dfeb1828e63cb29f162e1bb4f1f81db9cce; tracked clean; original two untracked files retained |
| Baseline link | `/home/zwan0726/work/baseline/latest_bidcell_official_run` |
| Resolved baseline | `/home/zwan0726/work/baseline/runs/bidcell_xenium_breast_official_step4000_20260723_013920` |
| Isolated new engineering code | `/home/zwan0726/work/projects/bidcell_remote_prep_v1` |
| Acceptance/run evidence | `/home/zwan0726/work/outputs/remote_prep_20261007T122955` |
| Baseline Python | `/home/zwan0726/miniforge3/envs/bidcell_local/bin/python`; package imports original checkout |
| Isolated official R evaluation | `/home/zwan0726/work/envs/bidcell_eval_r_v1`; source/package lock in acceptance evidence |
| Fixed coordinate cache | `/home/zwan0726/work/cache/bidcell_remote_prep_v1`; content and contract keys, not file-exists reuse |
| Machine artifact snapshot | `/home/zwan0726/work/projects/bidcell_remote_prep_v1/configs/machine.json` |
| Reference/annotation snapshot | `/home/zwan0726/work/projects/bidcell_remote_prep_v1/configs/reference.json` |
| Draft protocol snapshot | `/home/zwan0726/work/projects/bidcell_remote_prep_v1/configs/protocol.json` |
| Persistent queue | `/home/zwan0726/work/outputs/remote_prep_20261007T122955/pipeline_execution/pipeline.json`; state under `state/` |
| Windows inputs | `/home/zwan0726/work/docs/windows_baseline`; historical report only |

Host MCV-CGB4, user zwan0726; running kernel 7.0.0-34-generic. User restored driver 595.91.07; real BIDCell PyTorch 2.3.1+cu121/CUDA build12.1 tensor+matmul PASS on RTX2080Ti/11264MiB. NVIDIA driver, nvidia-smi, kernel modules and PyTorch runtime all PASS. New kernel7.0.0-38 was installed by the user but is not running; no reboot. Do not repeat repair or apt autoremove.

Full current environment/repository evidence: [audit report](/home/zwan0726/work/outputs/remote_prep_20261007T122955/environment_repository_audit.md), [GPU test](/home/zwan0726/work/outputs/remote_prep_20261007T122955/evidence/gpu_restored_validation.json). Old pytest-missing claim is obsolete: pytest9.1.1 is present. Legacy suite ran with two failures; see actual run record. Historical artifact names/paths below remain useful, but current machine status above supersedes September observations.

---

# Historical paths/environment record (preserved)

# Paths and Environment

> **Canonical scope:** Mutable filesystem, Git and runtime facts live here.  
> **Live calibration:** 2026-09-09, Australia/Sydney. `VERIFIED` means inspected in this pass; historical observations are explicitly identified. No source, dataset or experiment output was changed.

## 1. Repository

### Repository Root

`/home/zwan0726/work/projects/BIDCell` — **VERIFIED**, existing directory.

### Current Branch

| Fact | Live value | Status / evidence |
|---|---|---|
| Local branch | feature/geometry-ambiguity-v1 | VERIFIED after checkout |
| Local HEAD | 58ef3dfeb1828e63cb29f162e1bb4f1f81db9cce | VERIFIED: equals freshly fetched origin V1 |
| Working tree | Tracked tree exactly equals origin V1 | VERIFIED: empty tracked diff |
| Upstream | origin/feature/geometry-ambiguity-v1 | VERIFIED |
| Untracked | bidcell/processing/preannotate.py.backup_20260716; bidcell_local_constraints.txt | Retained |
| Ignored | 211 files: existing example artifacts/references, bytecode, PDM build files and example config | Retained; no target collision |

Previous branch: Bidcell_local. Previous HEAD: e565988cd2e78e622c68bd0a5649a1ec8b9b281f.
The two obsolete tracked config/preannotate edits were discarded with explicit user authorization.
No untracked or ignored path collided with V1 tracked paths. No clean, merge, commit, push or stash restoration occurred.
Checkout preparation is complete; remote validation and E1-E7 remain unrun.

### Important Branches and Commits

| Role | Ref | Commit |
|---|---|---|
| Baseline snapshot | origin/baseline/bidcell-step4000 | 2fae67ad74eaa9ec77b5d6fed6178f7cd9feca9d |
| Active V1 | origin/feature/geometry-ambiguity-v1 | 58ef3dfeb1828e63cb29f162e1bb4f1f81db9cce |
| Historical boundary-weighted V1 | origin/feature/boundary-weighted-loss-v1 | e5c28a79e66c21b2f302784e59983f0cc5b081be |

Fresh origin fetch succeeded after a sandbox SSH-permissions failure and authorized unsandboxed retry.
Frozen baseline execution metadata records the previous HEAD plus local changes, not the later baseline snapshot.
Before checkout, five saved Python snapshots matched the old working tree; this is historical provenance and does not describe the new V1 checkout.

### Git Remote

| Remote | Fetch and push URL | Status |
|---|---|---|
| origin | `git@github.com:stans3n/BIDCell.git` | VERIFIED; replaces historical HTTPS spelling |
| upstream | `https://github.com/SydneyBioX/BIDCell.git` | VERIFIED configured URL; upstream was not fetched |

## 2. Dataset

### Raw Data Root

`/home/zwan0726/work/data/Xenium_FFPE_Human_Breast_Cancer_Rep1/outs` — **VERIFIED**, existing directory; 10x Xenium FFPE Human Breast Cancer Rep1.

Paths below resolve as raw data root + relative name. Both are existing regular files (**VERIFIED**):

| Relative file | Bytes | Header inspection |
|---|---:|---|
| `morphology_mip.ome.tif` | 627807005 | TIFF series shape `(25778, 35416)` |
| `transcripts.csv.gz` | 1416091672 | Existence/metadata only; no decompression or full-data recount |

### Preprocessed Data

Use the frozen baseline inputs in Section 3. Historical transcript/gene/nucleus counts remain measured run facts in the baseline record; this pass read status/QC/log evidence and TIFF headers without recomputing the dataset.

## 3. BIDCell Full Reproduction

### Baseline Reproduction Root

Stable symlink:

`/home/zwan0726/work/baseline/latest_bidcell_official_run`

Resolved target:

`/home/zwan0726/work/baseline/runs/bidcell_xenium_breast_official_step4000_20260723_013920`

**VERIFIED:** `readlink -f` and live directory inspection match the old context. The model run ID is `2026_07_23_04_18_24`. All relative baseline paths below resolve under this target; this rule gives their resolved absolute paths without duplicating the root in every row. Treat this tree and every baseline artifact as read-only by project policy; no claim of filesystem-enforced immutability is made.

### Config

`configs/xenium_breast_official_step4000.yaml` — **VERIFIED**, actual frozen YAML, 1741 bytes.

| Control | Actual YAML value |
|---|---|
| CPU workers | `cpus: 4` |
| Nuclei tiles | `max_height: 12000`, `max_width: 16000` |
| Other nuclei controls | `diameter: null`, `crop_nuclei_to_ts: false`, `use_cpu: false`; `stitch_nuclei_fovs: false` |
| Model / patch | `name: custom`, `patch_size: 48` |
| Training budget | `total_epochs: 1`, `total_steps: 4000` |
| Optimizer | `adam`, learning rate `1e-5`, beta1 `0.9`, beta2 `0.999`, weight decay `0.0001` |
| Original loss weights | `ne_weight`, `os_weight`, `cc_weight`, `ov_weight`, `pos_weight`, `neg_weight`: all `1.0` |
| Save frequencies | `model_freq: 1000`, `sample_freq: 100` |
| Prediction | `test_epoch: 1`, `test_step: 4000` |
| Postprocessing | `patch_size_mp: 1024` |
| Experiment directories | `dir_id: last`, `model_dir: models`, `test_output_dir: test_output`, `samples_dir: samples` |
| Coordinate controls | target pixel `1.0` um; base image pixels `0.2125` um; transcript bases `1.0`; global shifts `0`; `shift_to_origin: false` |
| Transcript fields | `x_location`, `y_location`, `feature_name` |
| Filter prefixes | `NegControlProbe_`, `antisense_`, `NegControlCodeword_`, `BLANK_`, `Blank-`, `NegPrb` |
| Elongated types | Endothelial, Fibroblasts, Myofibroblasts, SMC |

**VERIFIED input/output settings:** `files.data_dir` equals the resolved baseline root; `fp_dapi` and `fp_transcripts` equal the raw-root files in Section 2. Reference files are absolute paths in the actual YAML, under the repository: `example_data/sc_references/sc_breast.csv`, `sc_breast_markers_pos.csv`, `sc_breast_markers_neg.csv` (respectively `fp_ref`, `fp_pos_markers`, `fp_neg_markers`). These paths exist. No scientific/control discrepancy was found against the values recorded in the project context, experiment matrix or baseline record. `dir_id: last` is the frozen YAML setting, not permission to select a different run.

### Preprocessed Outputs

Nuclei, resized DAPI, processed transcripts, gene names, expression maps, preannotation and the nucleus matrix are in the shared inventory below.

### Step-4000 Checkpoint

The official checkpoint is the `epoch_1_step_4000.pth` entry in the shared inventory below. Identity verification follows the table.

### Segmentation Outputs

The raw and connected TIFFs and final cell matrix are listed with the inputs and checkpoint in this shared inventory. Every row **exists**, is **VERIFIED**, and resolves under the baseline target above. Directory sizes are `stat` entry sizes, not recursive content sizes.

| Relative path | Type | Bytes |
|---|---|---:|
| `nuclei.tif` | Regular file | 164909968 |
| `dapi_resized.tif` | Regular file | 82455112 |
| `transcripts_processed.csv` | Regular file | 2557112272 |
| `all_gene_names.txt` | Regular file | 1862 |
| `expr_maps` | Directory | 20480 |
| `nuclei_cell_type.h5` | Regular file | 1729328 |
| `cell_gene_matrices/nuclei/expr_mat.csv` | Regular file | 69017391 |
| `model_outputs/2026_07_23_04_18_24/models/epoch_1_step_4000.pth` | Regular file | 326093130 |
| `model_outputs/2026_07_23_04_18_24/test_output` | Directory | 4096 |
| `model_outputs/2026_07_23_04_18_24/test_output/epoch_1_step_4000.tif` | Regular file | 164909968 |
| `model_outputs/2026_07_23_04_18_24/test_output/epoch_1_step_4000_connected.tif` | Regular file | 164909968 |
| `cell_gene_matrices/2026_07_23_04_18_24/expr_mat.csv` | Regular file | 69411114 |

Nuclei and resized DAPI TIFF headers both report `(5478, 7526)`.

**Checkpoint identity — VERIFIED read-only on CPU:** `torch.load(..., map_location="cpu", weights_only=True)` returned a dict with `epoch` = `1`, `model_state_dict` (OrderedDict, 240 entries), and `optimizer_state_dict` (dict with two entries). There is no explicit step field; step-4000 attribution comes from its filename, frozen YAML, pipeline/status records and matching recorded SHA256. No GPU loading or model execution occurred.

Current SHA256 matches the July 28 benchmark manifest for the config, nuclei, connected mask, final expression matrix and checkpoint. The checkpoint digest also matches `status/checkpoint_validation.json`. The large processed-transcript CSV was not hashed or recounted.

### Logs

The following are **VERIFIED** relative baseline paths; files are regular files and directories exist:

| Path | Type | Bytes |
|---|---|---:|
| `logs` | Directory | 4096 |
| `status` | Directory | 4096 |
| `metadata` | Directory | 4096 |
| `qc` | Directory | 4096 |
| `logs/080_bidcell_official_step4000_full_pipeline.log` | Regular file | 73420 |
| `logs/081_step4000_overview_qc.log` | Regular file | 4787 |
| `logs/082_step4000_roi_transcript_qc.log` | Regular file | 3000 |
| `logs/083_step4000_seam_qc.log` | Regular file | 3484 |
| `logs/084_transcript_boundary_training_influence.log` | Regular file | 2183 |
| `status/PIPELINE_COMPLETED.json` | Regular file | 1454 |
| `status/checkpoint_validation.json` | Regular file | 1514 |
| `status/prediction_validation.json` | Regular file | 755 |

All seven pipeline stages have `*_COMPLETED.json` records; retained `*_RUNNING.json` files describe stage starts, not currently running work. `PIPELINE_COMPLETED.json` records completion on July 23. Metadata includes saved code/diffs, package records, GPU snapshots and an output inventory.

### Baseline QC

**VERIFIED:** all four QC logs and their named outputs exist. Relative evidence files:

- `qc/step4000/overview/081_step4000_overview_metrics.json`
- `qc/step4000/roi_transcripts/082_step4000_roi_transcript_metrics.json`
- `qc/step4000/roi_transcripts/082_step4000_roi_summary.csv`
- `qc/step4000/seams/083_step4000_seam_diagnostics.json`
- `qc/step4000/seams/083_step4000_ranked_seam_rois.csv`
- `qc/transcript_training_influence/084_selected_patch.json`
- `qc/transcript_training_influence/084_checkpoint_boundary_influence_metrics.csv`
- `qc/transcript_training_influence/084_gene_boundary_saliency_by_checkpoint.csv`

QC scripts directory: `/home/zwan0726/work/scripts/bidcell_qc` — **VERIFIED**. Files `081_qc_full_overview_step4000.py`, `082_qc_roi_transcript_boundary_step4000.py`, `083_qc_seam_rois_step4000.py`, `084_visualize_transcript_boundary_training_influence.py` all exist. Their historical results and limits are recorded only in the appended baseline verification note.

## 4. V1

### Source Code

**VERIFIED in the active V1 checkout** after alignment:

```text
bidcell/model/utils/geometry_ambiguity.py
bidcell/model/utils/patch_manifest.py
bidcell/utils/resource_monitor.py
scripts/v1/geometry_sanity.py
scripts/v1/threshold_sweep.py
scripts/v1/make_dev_configs.py
scripts/v1/locality_eval.py
scripts/v1/common.py
docs/geometry_ambiguity_v1.md
tests/test_v1_geometry_ambiguity.py
tests/test_v1_loss_weighting.py
tests/test_resource_monitor.py
tests/test_v1_tools.py
```

Source inspection confirms existing search-area candidates, full-nucleus `cKDTree` nearest-point queries, full-area equivalent-radius normalization, stable two-candidate margin, transcript-supported `M < tau` selection, and weighting of existing losses. CellCalling is disabled by default; nucleus encapsulation and oversegmentation stay unweighted. `bidcell/model/model/model.py` has no diff from the baseline ref; no new trainable capacity was introduced by the inspected changes. Parameter equality is a source-level conclusion here, not a newly measured runtime count. Postprocessing changes add output routing/resource logging without changing its morphology operations.

All essential V1 files now exist in the live checkout. Checkout equality is not runtime validation. Current readiness/status is canonical in `03_experiment_matrix.md`.

### Planned V1 Experiment Root

`/home/zwan0726/work/experiments/boundary_aware_bidcell_v1`

**VERIFIED absent** on this machine; its parent `experiments` is also absent. It remains a planned path, not an existing output root. Proposed relative layout remains:

```text
configs/dev/
configs/e5_selected/
configs/full/
geometry/e1_sanity/
geometry/e2_tau_sweep/
dev_runs/
full/
evaluation/tier1/
evaluation/tier2/
evaluation/tier3/
evaluation/tier4/
logs/
manifests/
```

### V1 Configs

**UNVERIFIED / NOT YET RUN:** No real production V1 config or selected development manifest was found. Final settings must follow E2–E5 decisions.

### V1 Checkpoints

**UNVERIFIED / NOT YET RUN:** No real Rep1 V1 checkpoint was found in the inspected project/output records.

### V1 Outputs and Logs

**UNVERIFIED / NOT YET RUN:** No actual V1 remote-validation or E1–E7 run record or attributable artifact was found. Inspection covered canonical context, the planned root, work-level output/benchmark/log/script locations and fetched Git tree. This is evidence-bounded absence, not proof about every possible external location. Existing baseline ROI replay records are not geometry-ambiguity V1 runs.

## 5. Evaluation

### Frozen Evaluation Definition

`04_metrics_framework.md` remains the canonical, unchanged five-tier definition.

### Tier 1 / Tier 2 Evaluation Code

Discovered custom evaluator: `/home/zwan0726/work/scripts/bidcell_benchmark/benchmark_bidcell_step4000_baseline.py` — **VERIFIED existing file**.

Discovered output root: `/home/zwan0726/work/benchmarks/bidcell_step4000_baseline` — **VERIFIED existing directory**. Its `benchmark_manifest.json`, `benchmark.log`, `benchmark_summary.md`, `metric_status.csv`, `tier1_summary.json`, `tier2_summary.json`, `tier2_transcript_assignment_summary.json`, and associated CSV/JSON outputs attribute a completed July 28 benchmark to the frozen baseline; key artifact hashes match live files.

**UNVERIFIED:** equivalence of this custom evaluator to the final frozen CellSPA reporting protocol. It contains explicit proxies, skipped metrics and historical tier labels. Discovery of a script does not establish scientific validation. Measured evidence, assignment conflicts and missing official values are canonical in the appended baseline run note.

Existing baseline replay output directories:

- `/home/zwan0726/work/benchmarks/bidcell_step4000_roi_replay`
- `/home/zwan0726/work/benchmarks/bidcell_step4000_roi_replay_fixed`

**VERIFIED existing directories.** The fixed replay manifest explicitly identifies baseline step-4000 ROI inference with no training; it is not V1-E1/E2. Replay script: `/home/zwan0726/work/scripts/bidcell_benchmark/replay_bidcell_step4000_roi_inference.py` — **VERIFIED existing file**.

### V1 Tier 3 Code

`scripts/v1/locality_eval.py` exists in the active V1 checkout. **VERIFIED code; production use UNVERIFIED / NOT YET RUN.**

### Resource Monitoring

The fetched V1 source implements `<experiment_output>/evaluation/tier4/resource_summary.json`. **VERIFIED implementation; production output UNVERIFIED / NOT YET RUN.** Older benchmark filenames using `tier3` for efficiency do not redefine the frozen framework's Tier 4.

## 6. Environment

| Fact | Live value | Status |
|---|---|---|
| Home | `/home/zwan0726` | VERIFIED |
| Work root | `/home/zwan0726/work` | VERIFIED |
| Conda environment | `bidcell_local` | VERIFIED environment prefix and executable |
| Python | `/home/zwan0726/miniforge3/envs/bidcell_local/bin/python`, version `3.10.20` | VERIFIED |
| BIDCell import | `/home/zwan0726/work/projects/BIDCell/bidcell/__init__.py` | VERIFIED after checkout; imports active V1 |
| GPU | NVIDIA GeForce RTX 2080 Ti, total `11264 MiB`; `529 MiB` used at inspection | VERIFIED unsandboxed `nvidia-smi`, 2026-09-09 00:25 Sydney |
| Driver | `595.71.05` | VERIFIED |
| Driver-reported CUDA support | `13.2` | VERIFIED; not a toolkit/runtime installation claim |
| PyTorch CUDA build | `12.1` | VERIFIED `torch.version.cuda` |
| Installed CUDA toolkit | Not inspected / UNVERIFIED | Do not infer from nvidia-smi |
| System RAM | `31725860 kB` MemTotal, approximately `30.26 GiB` | VERIFIED `/proc/meminfo` |

The historical baseline reported about 10.56 GiB available VRAM; it is retained as a run-time observation, not today's total device capacity. Sandboxed `nvidia-smi` failed and imports warned that NVML was unavailable; unsandboxed inspection succeeded. This was a tool-access limitation, not evidence of a broken host driver. No V1 GPU operation or augmentation test was performed.

### Important Packages

**VERIFIED installed metadata:**

| Package | Version |
|---|---|
| torch | 2.3.1+cu121 |
| torchvision | 0.18.1+cu121 |
| numpy | 1.26.4 |
| imgaug | 0.4.0 |
| cellpose | 3.1.1.1 |
| opencv-python | 4.11.0.86 |
| opencv-python-headless | 4.11.0.86 |
| natsort | 8.4.0 |
| segmentation-models-pytorch | 0.5.0 |
| scipy | 1.15.3 |
| pandas | 2.3.3 |
| tifffile | 2025.5.10 |
| h5py | 3.16.0 |
| PyYAML | 6.0.3 |

Read-only imports of `bidcell`, `cv2`, `imgaug`, `natsort`, and `segmentation_models_pytorch` succeeded with bytecode writes disabled. These checks do not constitute `V1-REMOTE-001` completion.

Constraints file: repository-relative `bidcell_local_constraints.txt` — **VERIFIED existing, untracked, inspected**:

```text
numpy==1.26.4
imgaug==0.4.0
cellpose==3.1.1.1
opencv-python<4.12
opencv-python-headless<4.12
```

### Hardware-Specific Baseline Adaptation

The actual frozen nuclei tile controls are recorded once in Section 3's YAML table. Frozen pipeline metadata confirms they replaced official defaults `24000 x 32000` to fit historical GPU memory.

## 7. Path Verification Status

Live paths, refs and versions above replace the previous historical-only verification. Remaining limits:

- V1 checkout preparation is complete. Remote validation is blocked by missing pytest in the canonical environment; tests and augmentation remain unrun.
- Planned V1 output root is absent; do not write into baseline instead.
- Official CellSPA equivalence and a unified transcript-assignment headline remain **UNVERIFIED / CONFLICT — REQUIRES VERIFICATION** as detailed in the baseline record.
- Full-data counts were not recomputed; historical measured counts retain their original provenance.
- Fresh/default baseline initialization semantics are established by frozen run_script.py and saved train.py. No pristine initial checkpoint was saved: step 0 contains one Adam update. A complete validated frozen Tier 1 table remains unavailable.

## 8. Path Safety Rules

1. Treat the frozen baseline tree, checkpoint and existing benchmark/replay outputs as read-only.
2. Never write V1 outputs inside the baseline root or overwrite earlier experiments.
3. Do not infer missing paths or treat a Git ref as the active checkout.
4. Record an experiment only after it actually runs; code/import checks do not complete E1–E7.
5. Preserve untracked/ignored files and historical run records. Obsolete tracked edits were discarded with explicit authorization during V1 alignment.
6. Keep mutable absolute paths here; other context files should use canonical section references.

## Source Context

Original historical sources: `bidcell_full_reproduction_context_20260723_official4000_qc_updated(1).md`, `v1_geometry_ambiguity_context_after_codex_20260903.md`, `v1_codex_change_output.md`.

Calibration evidence: live filesystem metadata and TIFF headers; successful origin fetch and Git object inspection; frozen YAML/status/log/code snapshots; CPU checkpoint structure and recorded hash comparisons; package metadata/import checks; host GPU inspection; existing QC and July 28 benchmark records. No experiment, evaluation or test suite was run in this calibration.


## 2026-09-09 - V1 Checkout Alignment (Not a Run)

Checkout preparation COMPLETED. Previous branch: Bidcell_local; previous HEAD: e565988cd2e78e622c68bd0a5649a1ec8b9b281f. Current branch: feature/geometry-ambiguity-v1; current HEAD and freshly fetched origin V1: 58ef3dfeb1828e63cb29f162e1bb4f1f81db9cce. Upstream: origin/feature/geometry-ambiguity-v1. Tracked diff against origin V1 is empty. Authorized obsolete tracked edits were discarded. No untracked/ignored collision occurred; those files remain untouched. All essential V1 files exist; model.py equals baseline snapshot. No clean, merge, commit, push, tests or experiments occurred. This entry supersedes earlier checkout-pending statements; remote validation and E1-E7 remain NOT YET RUN. Current stage: read-only integration/initialization audit before remote validation.


### Post-checkout environment inspection

Python 3.10.20 imports active V1 at the canonical checkout with bytecode writes disabled. Critical dependency imports and installed versions above remain valid. pytest has neither import specification nor distribution metadata: an ENVIRONMENT BLOCKER for the requested test stage; no package was installed. Host GPU metadata remains RTX 2080 Ti / 11264 MiB / driver 595.71.05. No CUDA computation was performed.
