# CRGF-1008 stage 1

The native BIDCell six-loss source is unchanged. `RelationGeometry` is inserted after conv2 and before all consumers of h2. DataProcessing's legacy tuple interface remains the default; the staged entry uses an explicit complete candidate group, fixed view parameters, full-nucleus nearest-pixel geometry and crop-based native SearchArea/markers.

The strict run schema is `config.RunConfig`; unknown settings fail. The native YAML remains backward compatible and is saved fully resolved. Both methods load the same B0 model/BN, create fresh Adam and use the same frozen full-pool view sequence. Empty views do not count as updates. Unknown=-1 retains native last-type training indexing; evaluation preserves Unknown NA.

Both methods set cuDNN deterministic mode, disable TF32, set CUBLAS_WORKSPACE_CONFIG=:4096:8 and enable deterministic algorithms with warn_only=True. Installed torch 2.3.1 then dispatches bilinear interpolation to its deterministic decomposition. Native CUDA NLL forward has no deterministic implementation and warns; the criterion is unchanged. This is a common reproducibility setting, not a claim that every CUDA operation is bitwise deterministic. See validation numerical diagnostics and retained failed attempts.

## Commands

Run all commands from this checkout with its PYTHONPATH. The interpreter is `/home/zwan0726/miniforge3/envs/bidcell_local/bin/python`.

1. `python -m scripts.crgf.prepare --config /absolute/requested.json`
2. Complete validation, inspect changes, commit the checkout.
3. `python -m scripts.crgf.run_stage1 --config /absolute/output/configs/run.json --freeze`
4. `scripts/crgf/launch_stage1.sh /absolute/output/configs/run.json`
5. `python scripts/crgf/status.py --run-root /absolute/output`
6. `scripts/crgf/resume_stage1.sh /absolute/output/configs/run.json`

The launcher refuses an existing tmux session and confirms a new live runner identity. The runner holds an inherited flock across the stage process and its workers, so a crashed runner cannot allow a second writer while an orphan stage is still running. Recovery checks host/boot/start-ticks, frozen code, live input content hashes, stage receipts and strict checkpoint state. Heartbeats and actual progress are distinct.

A failed/interrupted stage resumes from its validated checkpoints or patch/tile/chunk receipts. A code repair must happen with the affected queue stopped. For a previously completed stage that needs repair, use `python -m scripts.crgf.invalidate --config ... --stage F1_evaluate --reason 'specific repair'`; this preserves old artifacts and attempts and invalidates only that stage and its downstream stages. Review and freeze the new code before resume. Training signatures exclude evaluator implementations; evaluator repairs do not require valid training to repeat. Stage dependency changes are rejected, not silently accepted. The report is rebuilt when upstream status/receipt identities change.

Inference stores atomic groups of 128 masks, never full logits/features. The two full shifts and native fill_grid are preserved. Global nucleus recovery occurs in native postprocessing. Morphology functions retain the original component selection; only no-background/empty guards are fixed. Tiles have coordinate-derived seeds. Split repair uses the exact cell bounding box with scan-order and overlap tie equivalence verified against the original full-field function. Only explicit tile/split manifests are read.

The matrix `lookup_axes`, `grid`, and `aggregate` kernels are AST-identical to the audited generator. Owners are recomputed for each own final mask. Only transcript coordinate/gene/ID caches are shared. Production acceptance independently resizes the full 2D mask with native OpenCV nearest neighbour, re-reads all transcripts, compares every owner and valid/OOB classification, independently aggregates with np.add.at, and verifies CSV/NPY/mappings/conservation. A uint32-compatible nonnegative integer mask is required; empty masks, no background and labels above int32 are tested. Matrix receipts expose assignments with row ranges and coordinate-record transcript IDs.

Morphology is recomputed from the own mask. Both official CellSPA annotation profiles and Unknown/NA domains are preserved. Individual optional components can fail without suppressing unrelated metrics. Metric 30 remains not applicable; pre/post change is a separately named engineering diagnostic. Raw output compilation ends at awaiting_user_analysis, or partial_failed with exact unfinished stages; no scientific conclusion is generated automatically.

Validation outputs and failed attempts live outside this checkout. Fixtures are technical checks, not a scientific ROI pilot or formal training updates. All historical inputs and scripts remain read-only.
