# Morph implementation and validation

Role: Morph implementer, explicitly requested `gpt-6-astra / xhigh`.
Branch: `feature/boundary-three-morph`; isolated worktree
`/home/zwan0726/work/projects/BIDCell_boundary_three_morph_20261015`.
Formal inputs are read-only historical artifacts. Every new output is under
`/home/zwan0726/work/outputs/bidcell_boundary_three_20261015`.
No full formal training is performed by this implementation task.

## Change-plan mapping

| Plan requirement | Implementation |
|---|---|
| Full canonical reconstruction, seed20261013 | `morphology_variants.canonical_full`; must equal historical `64c4e90ce5e14c948ce7bd3f523fb3b50f9f1734c322625826a7c3f99bcf4957`; `morph.prepare_morph` saves actual tensors and file hash |
| NO_DAPI deletion and memory columns0:64 | `MorphologyVariant('no_dapi')`; no image module is registered; `Evidence` does not open/mmap DAPI; aux excludes DAPI |
| NO_GEO deletion and memory columns64:96 | `MorphologyVariant('no_geo')`; no geometry encoder; aux omits geometry; shared image encoding once per view |
| SIMPLE_FUSION fixed two-block design | `LocalConvolutionFusion`; separate blocks; PW128→64/GN8/SiLU/DW5×5/PW64→64; copied LN/gate/FFN; zero-padding; isolated seed2026101501 |
| Exact common tensor mapping | `initialization_map`, per-tensor origin/hash/shape; all common encoders/projections/GN/LN/gate/FFN copied, output remains zero |
| Explicit common train/predict dispatch | `variant_spec`, `dataset`, `forward`; rejects presence/absence mismatch, `model.py` forwards optional keys |
| Shared untrained backbone | `make_model` loads only original `shared_initial`; original trained historical endpoint used exclusively for replay |
| Original six losses / FP32 Adam | unchanged `BaseSix`; original Adam lr1e−5/betas(.9,.999)/eps1e−8/weight_decay1e−4; no new loss, scheduler, AMP, TF32 or candidate splitting |
| Matched visits | `validate_cursor`, full frozen views/candidate IDs/patch hashes; formal exactly3998 visits/373 empty/3625 updates; original Unknown indexing preserved |
| Step zero and strict resume | `snapshot`, `restore`, `training_header`; model/BN/Adam/Python/NumPy/Torch/CUDA/imgaug RNG/cursor/effective sequence, explicit variant and pilot isolation |
| Branch activity and actual effects | updates 2/32/64/100 gradient groups plus same-weight residual-disabled h3/logit/native-owner differences; diagnostic model/BN hash invariant and RNG restored |
| Native prediction | both shifts0/24, complete K, existing `get_seg_mask` average-background/tie/nuclear restoration, existing stripe merge |
| Native primary endpoint | existing native postprocess; machine paths explicitly point to own masks/matrix; no P2/A2 |
| Own complete evaluator | parent `evaluation.build_and_evaluate`, original independent matrix oracle, own-QC/scClassify, fixed panel and all engineering/paper profiles |
| Required final diagnostics | parent comparisons against BASE_MATCH and old full V2_MORPH; canonical detailed reports plus top-level `boundary_strata.csv`/`paired_comparisons.csv` with source identities |
| Complete means full chain | `run_node` writes outer common receipt only after prediction/native post/matrix/evaluation/pair comparisons; `validate_node` rechecks endpoint and comparison semantics and hashes |

## Integration API

- `prepare_morph(config)` must run before final source/config freeze. It saves the actual canonical tensor archive, all three genuine formal step-zero states, initialization maps and own machine/config files. It never adopts pilot or trained endpoint weights.
- `run_node(config,node)` handles exactly `MORPH_NO_DAPI`, `MORPH_SIMPLE_FUSION`, `MORPH_NO_GEO`.
- `validate_node(config,node)` requires all formal artifacts and semantic validations.
- `validate_mini(config,output)` uses `common.compute_lock` for the real heavy work. Formal `run_node` never reacquires the runtime-owned lock.
- Parent owns all shared contracts/freeze/evaluation/reports; this commit includes only Morph-owned changes plus minimal agreed copied model/data adaptations.

## Validation evidence and retained failures

Small CPU tests: `tests/boundary_three/test_morph_variants.py` cover reconstruction, every common mapping, RNG isolation, actual missing modules, simple-block independence, zero residual, learned branch gradients, strict incompatible state rejection and bitwise CPU model/Adam/RNG uninterrupted/resume.

Real mini artifacts live under `mini/morph/`. First attempt passed independent historical-source CPU and GPU dual-shift ROI replay, then failed an audit-only CPU/CUDA comparison in the initialization mapper; mapping comparison now explicitly moves both tensors to CPU. No scientific forward changed.

Second attempt again passed full replay, then failed an overbroad assertion that original SearchArea must rotate exactly. Native code recomputes dilation after augmentation with even-sized/ellipse kernels. That original behavior has not been changed. The corrected audit tests derived geometry vectors/covariances/scalars for equivariance and DAPI alignment, and separately checks channels11/12 against the actual native support/multiplicity. On the frozen first12-candidate view, native support transform discrepancies are0,545,217,553,77,512,142,590 candidate-pixels across the eight transforms; the full focused check passes. This is a documented historical property, not a new tolerance for geometry errors.

The accepted complete real mini is `mini/morph/attempt_08/complete.json`. Its actual compute time was 221.429 seconds, shared-resource wait 0.105 seconds and process peak RSS 6,124,457,984 bytes. It used a real 144×144 ROI with 18,469 retained original transcript IDs/genes. All three 100-update temporary models completed both shifts, native postprocessing, independent integer matrix validation and all authoritative evaluator profiles. No formal full training was performed. The temporary trained checkpoints were deleted; initial states, receipts, matrices and evaluators remain inspectable.

`mini/morph/final_validation.json` rechecks 294 retained artifact identities, current scientific source/config identity and all three own complete evaluators after the process exits. `mini/morph/prepare_validation/validation.json` separately verifies genuine zero-update initializer preparation for all three variants and repeated preparation without changing their files (13.385 seconds). `mini/morph/discarded_failed_temporary_checkpoints.json` records removal of the failed update 32 resume checkpoint; no trained temporary checkpoints remain.

Attempts 03/04 stopped while waiting for the compute lock, before scientific work. Attempt 05 exposed unequal CPU thread setup in the historical replay harness; both constructors now use the frozen numerical setup and exact equality remains required. Attempt 06 passed historical replay, then exposed repeated Adam trials sharing the CPU step tensor of the reference state; independent counter copies and an exact durable-reference comparison fixed that harness bug. Attempt 07 completed NO_DAPI's full chain, then exposed an overlapping live GPU state-dictionary copy before SIMPLE_FUSION training; mini step-zero snapshots now use independent CPU tensors. The final resume test executes directly from the actual disk-restored state. Failed logs and earlier valid intermediate receipts remain distinct; no acceptance tolerance was relaxed.

Final focused tests: 7 passed in 1.000 second, including the regression that repeated diagnostic updates cannot modify the durable reference. The historical full replay passes exact CPU logits, actual GPU logits for both shifts, decoded/merged pre masks, native ROI final masks and the historical full-slide pre-mask interior. Canonical full tensor SHA remains `64c4e90ce5e14c948ce7bd3f523fb3b50f9f1734c322625826a7c3f99bcf4957`.

## Scientific limits

Canonical full Morph is reconstructed from the frozen code/seed and checked against the earlier recorded hash. It is not an independently archived historical step-zero state. Full CPU/GPU replay checks actual trained weights and native outputs. The ROI native postprocess equivalence check compares identical local domains; it does not claim that ROI tiling equals every full-slide tile halo.

Branch deletion changes capacity; simple fusion changes both operator and capacity/padding. These are the prespecified conditional branch/module comparisons, not pure-information or equal-parameter controls. All endpoints retain the original nucleus-derived priors. A single seed/developed Rep1 and no independent owner truth limit inference. Owner-change diagnostics show that learned residuals can alter assignments, not that any change is correct.

GPU backward can have the existing native numerical variability. Admission measures five same-state native repeats; next-step gradients must fit the original RMS envelope, logits/BN/RNG must be exact, and both updates must satisfy the independently calculated original Adam formula. Exact snapshot restoration and CPU uninterrupted/resume are checked separately. No science threshold is relaxed.

## Native orchestration follow-up

Parent requested an explicit native tile stop adapter after the initial implementation commit. `scripts/crgf/postprocess.py` now accepts optional `stop_requested` and `progress_callback` attributes. It submits at most `cpu_workers` tile tasks at once, commits their original tile receipts, checks stop after each group and each128-ID split chunk, and rebuilds from the same immutable tile/split outputs on resume. Native seed+row×1000003+col, tile computation, order and final merge are unchanged. Morph uses the frozen resource worker count (4), with inherited numerical worker threads1. It checks stop before native postprocess, own matrix/evaluation, each comparison and final node receipt. Formal code never acquires a second compute lock.

`test_morph_native_stop.py` invokes the original frozen postprocess as an independent oracle, then actually stops the new implementation after 2/9 tiles and after a durable split chunk, resumes it, and requires exact final-mask equality. An initial test caught a local callback name colliding with the original boundary array; the helper was renamed before the passed run. Mini lock waiting permits 3600 seconds as explicitly authorized by the parent; formal 600-second resource waiting is unchanged.

Measured parameter counts: backbone27,153,218; full Morph191,800 added/27,345,018 total; NO_DAPI184,712 added/27,337,930 total; SIMPLE_FUSION186,736 added/27,339,954 total; NO_GEO165,432 added/27,318,650 total. Active gradient counts are recorded from the actual per-parameter nonzero gradients grouped by registered module, rather than assuming all registered trainable parameters receive nonzero gradients.

## Actual mini effects and resource measurements

Each variant completed 100 updates / 106 visits / 6 empty visits. The initial logits are exactly the shared backbone's logits. Every registered module group has positive gradient norm at updates 2/32/64/100, and actual residual and logit deltas are positive. Native owner changes can be zero at individual probes; that is reported, not converted into a failed or positive scientific result.

| Variant | 100-update production seconds | Pilot incl. resume/max-K diagnostics | Own complete mini chain subtotal | Training K=40 peak allocation | Owner changed pixels at 2/32/64/100 |
|---|---:|---:|---:|---:|---|
| NO_DAPI |20.595|44.964|64.421|3,767,417,856B|5/0/0/2|
| SIMPLE_FUSION |20.514|43.476|62.735|3,620,921,856B|1/0/0/2|
| NO_GEO |15.435|35.460|54.123|3,710,356,480B|5/0/0/2|

The complete-chain subtotals include training diagnostics, ROI prediction, native postprocessing and own matrix/full evaluation. The overall 221.429 seconds also includes shared ROI construction and historical replay. All prediction pools were enumerated: maximum K is 41 at shift 0 and 40 at shift 24, with actual unsplit inference peaking at 1.629 GB or less. The accepted run used actual GPU tensors and a real RTX 2080 Ti (11 GiB).

All three disk-restored model/BN/Adam/RNG/cursor states are bitwise exact, as are the next logits/BN/RNG. Next GPU parameter/optimizer states are not bitwise identical, matching the pre-existing native backward nondeterminism. Their largest parameter differences are 2.2431e−6 / 2.8526e−6 / 2.9989e−6; every gradient tensor passes the prespecified five-repeat native RMS envelope, and both updates pass the independent Adam formula. CPU uninterrupted/resume is bitwise exact. These checks establish engineering replay/resume, not scientific equivalence between trained runs.

## Complete-node forecast

`mini/morph/cost_forecast.json` and its reproducible `build_cost_forecast.py` bind the mini receipt, actual historical timing receipts, measured artifact sizes and the parent's complete shared-report smoke. Forecast components per formal node are training 1128 s, prediction 5708 s, native post 5079 s, own matrix/oracle 300 s, full own evaluator 2446 s, two complete paired reports 3600 s and initialization/receipts/I/O 600 s: 18,861 s, rounded to a 5.5-hour scheduling allowance.

Training/prediction use the larger of measured mini projection and historical full Morph, then 25% headroom. Native post uses actual historical four-worker full-slide timing plus 50% for bounded-group load imbalance. The full evaluator uses measured full-field BASE_MATCH timing plus 25%, not ROI linear scaling. Pair-report timing was not directly instrumented in the parent smoke: 1800 s/pair is explicitly an estimate. This is a conservative scheduling forecast, not a guaranteed duration or a claimed full formal run.

Measured historical complete trees occupy about 2.57 GB each, the complete pair report 212.5 MB, and the shared full coordinate cache 688.9 MB. Reserve 6 GiB per new Morph node for final artifacts, extra saved prediction patches, step zero, two reports and temporary writes; all three require 18 GiB, plus 1 GiB shared cache allowance and the separate 10 GiB global reserve. No new P2/A2 endpoint is included in these primary Morph costs.
