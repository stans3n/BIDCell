# BIDCell three independent arms

This frozen experiment implements `V1_REGION`, `V1_BOUNDARY`, and `V2_MORPH` as separate arms. The native six losses, input tensor, complete candidate group, BatchNorm, native decoder, nucleus restoration and native postprocessing are retained. Each arm starts from `prepared/shared_initial.pth`, seed 1234, and completes 3625 retained Adam updates using the same frozen nonempty views and augmentations. The configured 4726 views come from the historical 14536 training pool; empty views consume the saved cursor but no Adam update. The historical B0 initial RNG and exact ordering were not saved, so that part of B0 is not reproducible.

The Region and Boundary references deliberately have different tolerance-pooling rules. Unknown type `-1` retains native POS/NEG negative-index semantics and ordinary ellipse support; it abstains from the new reference losses. V2 has a zero residual at h3 and uses the full morphology module, without internal candidate chunks, in this admitted configuration.

At V2's first update, internal morphology branches have zero loss-gradient behind the zero projection, while Adam weight decay can still change their parameters. From the second update, the admission evidence requires positive internal branch gradients and measured parameter updates.

Region calibration is explicitly RR-V1.2: eta retains the originally prescribed pooled rule, while lambda uses only the fixed L6-only base64 probe state (expression96 only when the original root-inactive fallback is required). The original pooled-lambda version failed the real gradient-ratio gate and is retained under `admission/V1_REGION/obsolete_pooled_calibration/`. The revision was reviewed before testing; the nominal, sensitivity, capture and ratio thresholds remain unchanged, and no coefficient is selected by final P/N scores.

Run commands from `/home/zwan0726/work/projects/BIDCell_new_methods_20261013`. The Python interpreter is `/home/zwan0726/miniforge3/envs/bidcell_local/bin/python`. CUDA commands must run on the host, where the RTX 2080 Ti is visible. The normal sandbox cannot see this GPU.

From any working directory, the fixed entry point is:

```bash
/home/zwan0726/work/projects/BIDCell_new_methods_20261013/scripts/new_methods/control.sh status
/home/zwan0726/work/projects/BIDCell_new_methods_20261013/scripts/new_methods/control.sh resume
/home/zwan0726/work/projects/BIDCell_new_methods_20261013/scripts/new_methods/control.sh complete-check
```

```bash
export BIDCELL_ROOT=/home/zwan0726/work/outputs/bidcell_new_methods_20261013
export BIDCELL_PY=/home/zwan0726/miniforge3/envs/bidcell_local/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

# Immutable evidence, fresh initialization, views, reference and run configuration.
$BIDCELL_PY -m scripts.new_methods.runner prepare --root "$BIDCELL_ROOT"

# Isolated development states and real ROI chain; no formal training.
$BIDCELL_PY -m scripts.new_methods.runner admit --root "$BIDCELL_ROOT"

# Requires final independent review bound to source digest and admission SHA256.
$BIDCELL_PY -m scripts.new_methods.runner launch --root "$BIDCELL_ROOT"

$BIDCELL_PY -m scripts.new_methods.runner status --root "$BIDCELL_ROOT"
$BIDCELL_PY -m scripts.new_methods.runner tail --root "$BIDCELL_ROOT" --lines 40

# Restart a stopped backend with the exact frozen source/configuration.
$BIDCELL_PY -m scripts.new_methods.runner resume --root "$BIDCELL_ROOT"

# Revalidates every completed stage and its actual artifacts.
$BIDCELL_PY -m scripts.new_methods.runner complete-check --root "$BIDCELL_ROOT"
```

The backend first trains Boundary, Morphology, then Region on one GPU. It then performs each arm's two shifts, native postprocessing, own integer matrix and transcript owners, and all three evaluation profiles. B0 and B1 receive new evaluation outputs from their existing accepted matrices. B0 is the primary comparator; B1 has a different budget. The fixed historical 103000-label population uses metric-specific finite denominators; genuinely missing or historically unavailable labels stay NA. Matched zero-expression author F1/Recall are zero and Precision remains NA.

`admission/` contains measured numerical, calibration, development, checkpoint restoration, input, coverage, resource and ROI evidence. These checkpoints are discarded pilot endpoints and are never formal initialization. The independent audit receipt is `tests/independent_review/approval.json`, with an identical alias at `admission/independent_review.json`. The launcher requires `status: approved`, the final `source_digest`, and the SHA256 of `admission/complete.json`.

The untouched baseline CUDA backward is itself numerically nondeterministic. Independent CPU comparisons are bitwise; CUDA initial logits and BatchNorm are bitwise. CUDA gradient tensor RMS must be at most twice the maximum of five prespecified original repeats plus `1e-7`. Parameter RMS is reported against the native repeat envelope, including sparse near-zero effective-gradient sign flips; it is not falsely described as passing that envelope. Each side's actual Adam parameter and moment update is independently recomputed from its own gradient and common prior state, with parameter tolerance `atol=1e-7, rtol=1e-6`. Moment formulas are evaluated in FP64 and checked against `1e-7 + 4*eps_FP32*sum(abs(update summands))`, which accounts for native FP32 cancellation without dividing by a nearly zero result. First-step parameter differences also have a `2*lr+1e-8` maximum sanity bound. Restoring saved model/Adam/RNG/cursor is exact; a subsequent CUDA step is checked by gradient repeatability and the original Adam formula rather than described as bitwise reproducible.

`source/manifest.json` freezes source files. Each stage stores dependency and output identities and verifies successful outputs before reuse. Inputs are SHA256-checked at training/prediction boundaries, outside the hot loop. Source, configuration, calibration, reference or input drift causes refusal. `run.lock`, `launch.lock`, and the shared `/home/zwan0726/work/outputs/cos_gpu.lock` reject duplicate/conflicting work. A failed child is terminated as a process group; each stage has a finite timeout and at most one retry. Checkpoints retain model, Adam, BatchNorm, Python/NumPy/Torch/CUDA RNG, actual update count, view cursor and empty skips. A failure can recompute work since the last checkpoint (every 25 retained updates); the append-only log distinguishes attempted work from the authoritative checkpoint count.

No backend stage generates or edits source. `resume` skips only verified successful stages, resumes native prediction chunks and training checkpoints, and can recover a failed report separately through the same queue. `status.json` ends at `compute_complete` with `scientific_review: awaiting_user_resume`. That is completion of computation, not scientific approval of the new models. Sydney estimates in `admission/resources.json` separate measured training time from the full chain, whose CPU endpoint forecast uses historical timings with 20% margin; these are estimates, not a morning completion promise.

Producer hashes are component-specific. A separately reviewed repair to `report.py` does not invalidate model or evaluation artifacts. The complete source manifest and launch review must still be renewed before running changed source; source changes are never silently accepted. A prelaunch pending receipt bridges child completion and parent stage-commit writes, so interruption between them can recover verified outputs without rebuilding a protected completed matrix.
