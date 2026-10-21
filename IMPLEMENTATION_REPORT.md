# P2 singleton batch shape repair

The isolated candidate passed all checks. It is not published into the live prediction cache. The main reviewer owns publication, recovery, and Goal Pause.

## Failure and exact change

The frozen scClassify 1.5.1 function `predict_scClassifySingle` assigns `predMat <- vapply(..., character(nrow(predMat)))`. When there is one query cell, `vapply` simplifies its results to a character vector. The immediately following `apply(predMat, 1, ...)` fails because the vector has no dimensions. Existing expression subsetting already preserves dimensions.

`clone_singleton.R` clones the public predictor and its single-classifier function in a new environment whose parent is the unchanged scClassify namespace. It inserts one conditional immediately after the exact original `vapply` assignment: if dimensions are absent, require exactly one query cell and the original number of hierarchy levels, then restore a matrix with one row, the original input cell ID as its row name, and no column names. It changes no scores, gene filtering, parameters, hierarchy traversal, label mapping, or package bindings. The original function formals and every original body statement are asserted identical. The namespace function body is checked again after testing.

Package identity: scClassify 1.5.1, commit `74fe32784b401492da0fd203736f39bcd6f2a2c5`. The original and patched serialized R function body hashes are recorded in `candidate/validation.json`; the corresponding deparsed bodies are in `candidate/original_single.R` and `candidate/patched_single.R`.

## Candidate and provenance

`prepare_candidate.R` reads the original P2 matrix, applies the frozen QC to all 313 genes, normalizes all 77,057 QC-pass cells together, and selects the original 307 reference genes in their original order. The recomputed size-factor CSV is byte-identical to the existing frozen-evaluator output. It verifies the dependencies, IDs, SHA256 hashes, shapes, and RNG state of all 301 committed batches. It processes only the original remaining cell, ID `78712`, beginning from batch 301's saved RNG state. No cell is padded or dropped; the protocol batch size remains 256.

Candidate: `candidate/batch_0302.rds`, 5,637 bytes, SHA256 `98c47288b42a9536222b75d4ecb8a559e6938ccae169f7ed5848e2dfee0e5e81`. It contains the four original loader fields (`ids`, `prediction`, `rng_state`, `dependencies_digest`) plus explicit `repair_provenance`. The latter binds both repair scripts, original model, protocol, matrix, QC and normalization, source evaluator, config, runtime, previous committed batch, function bodies, parameters, and validation outcomes. Its dependencies digest remains the existing P2 annotation digest `afccd14fb5afbd53a3d4fedeb3bb4529a4ff35a839b01eed1683ee77130ec8ea`.

The script has no path that writes into `prediction_chunks`, emits a pending/committed production receipt, changes a package, or starts the pipeline. It refuses to run if its candidate directory or the production batch 302 already exists. The main reviewer can publish this reviewed candidate atomically and use the frozen Python pending-chunk commit machinery with cursor/total 77,057, then resume the unchanged evaluator. That evaluator will reuse batches 1–301, restore each stored RNG state, and load the repaired final batch through its original cache path.

## Validation performed

- All 198 locked R package versions matched the original reference lock.
- Real batch 301, initialized from batch 300 RNG: the original function reproduced the existing cached prediction and final RNG exactly.
- The isolated clone reproduced that original 256-cell prediction, final RNG, and warning sequence exactly.
- The original one-cell call reproduced `dim(X) must have a positive length`.
- The repaired one-cell call returned one prediction and a 1-by-6 hierarchy with the original ID. Its final RNG and warnings exactly matched the original failing call; the RNG did not advance.
- For a test oracle only, the original function processed two identical copies of this normalized query column with distinct test IDs. Both oracle labels and hierarchy rows matched the repaired singleton exactly. This oracle preserves the original per-batch expressed-gene filter; it contributes no duplicated cell to the candidate.
- Full-QC size-factor CSV SHA256 matched exactly. All bound inputs and all 301 existing batch/RDS receipts were unchanged after testing. The package namespace was unchanged, the frozen source Git checkout remained clean, and production batch 302 remained absent.

The original batch emitted five zero-standard-deviation warnings; the original/repaired singleton emitted the same three such warnings. No warning was converted into a new threshold or scoring rule. `sessionInfo()` also emitted an environment-only `timedatectl` permission warning while recording the R session; all validation and the candidate write succeeded.

Evidence: `test.log`, `candidate/validation.json`, `candidate/prior_batches_verified.json`, and `candidate/R_sessionInfo.txt`. Test duration was approximately 34 seconds. The driver command was:

```bash
env R_LIBS=/home/zwan0726/work/outputs/rep2_full_20261021/runtime_cache/R_library \
    R_LIBS_USER='' R_LIBS_SITE='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 R_PROFILE=/dev/null R_ENVIRON=/dev/null \
    /home/zwan0726/work/envs/bidcell_egr_r/bin/Rscript --vanilla \
    /home/zwan0726/work/outputs/rep2_full_20261021/repairs/singleton/prepare_candidate.R \
    /home/zwan0726/work/outputs/rep2_full_20261021
```

This command has already completed successfully and is intentionally not rerunnable over its existing candidate directory. No frozen source, configuration, prior completed stage receipt, historical asset, or shared R package was edited.
