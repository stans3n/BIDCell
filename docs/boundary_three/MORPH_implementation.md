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
| Branch activity and actual effects | updates2/32/64/100 gradient groups plus same-weight residual-disabled h3/logit/native-owner differences; diagnostic model/BN hash invariant and RNG restored |
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

The current full real mini is `mini/morph/attempt_03`; its results and cost section will be completed only from actual receipts. Trained temporary endpoints are discarded after each completed own mini chain; initial states, receipts, matrices and evaluators remain inspectable. Interrupted failed attempts remain distinct and cannot become formal initializers.

## Scientific limits

Canonical full Morph is reconstructed from the frozen code/seed and checked against the earlier recorded hash. It is not an independently archived historical step-zero state. Full CPU/GPU replay checks actual trained weights and native outputs. The ROI native postprocess equivalence check compares identical local domains; it does not claim that ROI tiling equals every full-slide tile halo.

Branch deletion changes capacity; simple fusion changes both operator and capacity/padding. These are the prespecified conditional branch/module comparisons, not pure-information or equal-parameter controls. All endpoints retain the original nucleus-derived priors. A single seed/developed Rep1 and no independent owner truth limit inference. Owner-change diagnostics show that learned residuals can alter assignments, not that any change is correct.

GPU backward can have the existing native numerical variability. Admission measures five same-state native repeats; next-step gradients must fit the original RMS envelope, logits/BN/RNG must be exact, and both updates must satisfy the independently calculated original Adam formula. Exact snapshot restoration and CPU uninterrupted/resume are checked separately. No science threshold is relaxed.

## Native orchestration follow-up

Parent requested an explicit native tile stop adapter after the initial implementation commit. `scripts/crgf/postprocess.py` now accepts optional `stop_requested` and `progress_callback` attributes. It submits at most `cpu_workers` tile tasks at once, commits their original tile receipts, checks stop after each group and each128-ID split chunk, and rebuilds from the same immutable tile/split outputs on resume. Native seed+row×1000003+col, tile computation, order and final merge are unchanged. Morph uses the frozen resource worker count (4), with inherited numerical worker threads1. It checks stop before native postprocess, own matrix/evaluation, each comparison and final node receipt. Formal code never acquires a second compute lock.

`test_morph_native_stop.py` invokes the original frozen postprocess as an independent oracle, then actually stops the new implementation after2/9tiles and after a durable split chunk, resumes it, and requires exact final-mask equality. All6 focused tests passed in0.861seconds. An initial test caught a local callback name colliding with the original boundary array; the helper was renamed before this passed run.

Attempt03 only waited for the shared resource lock and was terminated before scientific work to bind this source change. The active real mini is now `mini/morph/attempt_04`, with a parent-authorized mini-only3600-second lock wait. The formal600-second resource wait is unchanged.

Measured parameter counts: backbone27,153,218; full Morph191,800 added/27,345,018 total; NO_DAPI184,712 added/27,337,930 total; SIMPLE_FUSION186,736 added/27,339,954 total; NO_GEO165,432 added/27,318,650 total. Active gradient counts are recorded from the actual per-parameter nonzero gradients grouped by registered module, rather than assuming all registered trainable parameters receive nonzero gradients.
