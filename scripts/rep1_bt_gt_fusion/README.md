# Rep1 B/T + GT scientific interface

This package implements scientific stages for the separately audited controller.
It has no detached launcher, scheduler, user-resume creator, or context writer.
The production queue is exactly:

`BT_EXISTING6 -> P2_BASE_GT_PRODUCE -> V2_MORPH_GT_PRODUCE -> PRODUCTION_VALIDATE`

The controller must finish in `PRODUCTION_COMPLETE_AWAITING_MANUAL_RESUME` and
exit. Scientific success is independent of favorable B/T or F1 results.

## Controller API

```python
from scripts.rep1_bt_gt_fusion import NODE_ORDER
from scripts.rep1_bt_gt_fusion.inputs import freeze_inputs, source_dependencies
from scripts.rep1_bt_gt_fusion.common import digest
from scripts.rep1_bt_gt_fusion.science import run_stage, validate_stage, receipt_path

inputs = freeze_inputs(run_root)  # hashes existing data; no scientific experiment
input_digest = digest(inputs)
sources = source_dependencies()  # original path, bytes, SHA256, snapshot_relative

result = run_stage(config, node, progress=on_committed_progress,
                   lock_fds=(run_lock_fd, workspace_compute_lock_fd),
                   validate_execution=check_admission_environment_and_snapshot)
result = validate_stage(config, node)  # read-only receipt/hash/semantic acceptance
```

`config` can be the frozen JSON path or a dictionary for `run_stage`; the other
interfaces take its dictionary. Required fields are:

```json
{
  "run_id": "rep1-bt-gt-fusion-20261016-r01",
  "run_root": "/absolute/new/run/root",
  "source_snapshot": "/absolute/new/run/root/source_snapshot",
  "input_manifest": "/absolute/new/run/root/configs/inputs_manifest.json",
  "source_digest": "controller-final-source-manifest-digest",
  "input_digest": "canonical-digest-of-input-manifest",
  "plan_sha256": "f24c5150a06ced0c4c1ff312a8aeaa5a2b9ed507b160d713030fcba361e671cf",
  "node_order": ["BT_EXISTING6", "P2_BASE_GT_PRODUCE", "V2_MORPH_GT_PRODUCE", "PRODUCTION_VALIDATE"],
  "rscript": "/home/zwan0726/work/envs/bidcell_egr_r/bin/Rscript",
  "r_libs_user": "/home/zwan0726/work/outputs/bidcell_historical_dual_20261013/sources/R_library",
  "r_libs_site": "",
  "source_commit": "audited-final-commit"
}
```

Do not add a self-referential `config_digest`. Receipts compute `digest(config)`
using sorted compact JSON with finite JSON values. Additional controller fields
are included in this digest and therefore must be frozen too.

`run_stage` refuses empty lock descriptors or a missing execution validator. It
checks descriptors with `fstat`; the controller validator must prove lock
ownership, source module locations, admission and interpreter/package identity.
It runs before scientific work and at each R subprocess launch/return. The
controller owns process identity, heartbeat, signals, exit codes and watchdogs.

`progress` receives `{"stage": str, "cursor": int, "total": int, "detail": dict}`
after an atomic scientific checkpoint or verified complete output. Chunk cursors
are scoped by stage and detail, not one global monotone counter. Resumed chunks
are hash-validated before their cursors are reported. A controller must distinguish
validated prior work from advancement when deciding whether a new launch has
made scientific progress. Source hashing and a live process alone are insufficient.
An exception raised by the callback propagates after the committed checkpoint;
this permits a controller to stop safely for resource limits.

## Input and source freeze

`freeze_inputs` writes only `configs/inputs_manifest.json` below its supplied new
root. It validates B0's separate native-matrix chain, five registry adapters,
old complete evaluations and F1 files, A1 inputs, absolute BASE_MATCH coordinate
cache, original transcript/gene identities, fixed B1 panel and author reference
files. It never copies old matrices or reclassifies the six existing endpoints.
`load_inputs` checks the entire existing-file inventory and canonical digest.

Only the already accepted, absolute BASE_MATCH coordinate cache is used. The
current actual cache passed its complete key, row coverage and receipt checks.
A different/corrupt cache fails closed. Automatic construction of a replacement
shared cache for a mismatching key is not implemented; this run must not silently
use a different cache or modify the historical one.

`source_dependencies` enumerates this package's Python/R files, `scripts/__init__.py`,
the imported `boundary_three` core/common/contracts/settings modules, `crgf`
matrix/common modules and `new_methods/author_marker.py`. Old orchestration is
never called. Four external R sources are copied as actual files into
`source_snapshot/external_sources/`:

- `publish_expression_metrics.R`, `publish_processing.R`, `publish_utils.R` from
  `outputs/bidcell_metric_protocol_review_20261008/`;
- `publish_spatialDiversity_metrics.R` from
  `outputs/bidcell_historical_dual_20261013/sources/`.

These four sources have no transitive `source()` calls. The new R wrappers use
only the snapshot allowlist and never evaluate protocol source-path fields.
The historical hardcoded spatial wrapper is not executed. R runs with `--vanilla`,
explicit `R_LIBS_USER`/`R_LIBS_SITE`, and explicit `pass_fds`; `REP1_BT_GT_LOCK_FDS`
records the inherited descriptor numbers. The entropy library above must be
included in the controller's frozen installed-package/environment receipt.

The controller must copy files, reject symlinks/hardlinks, make its snapshot
read-only, disable bytecode writes, use snapshot cwd and isolated Python search
paths, and verify all delayed imports against the manifest. The interface README
and fixture source can additionally be included in the audit manifest; they are
not imported as runtime dependencies.

## Receipt and recovery behavior

Each receipt binds run, stage, canonical config, source, input and plan digests.
Output files have absolute paths, bytes and SHA256. A receipt with changed output
bytes is an error; the package does not delete it and silently retry. Independent
matrix acceptance visits every original transcript with the full native
OpenCV nearest-neighbor resize oracle, including OOB/owner/gene/ID checks.
Final production validation additionally checks exact parent foreground,
nuclear owners, all IDs, assigned totals and every gene's global count.

GT rebuilds the domain separately from each real parent's final mask. Domain,
expression, support, geometry, score, topology and owner/oracle loops commit
recoverable chunks. GT retains its exact sorted proposal cursor, complete mask
and per-proposal acceptance/rejection reason. R prediction batches persist RDS
and post-prediction RNG state; Python fsyncs and hashes each before authorizing
reuse. A recovered R batch restores that RNG state before subsequent prediction.

## Manual evaluation and publication

Only a fresh, explicit user instruction may cause the controller to create
`run_root/control/manual_resume.json`. This package never creates that grant.
Its required fields are `authorization: "explicit_user_resume"`, a nonempty
`user_instruction_utc`, `dependencies(config, "MANUAL_RESUME")`, and
`production_complete: identity(run_root/production_complete.json)`.
The controller must also prove the production worker exited and released locks.

```python
from scripts.rep1_bt_gt_fusion.evaluation import evaluate
from scripts.rep1_bt_gt_fusion.publication import publish, recover_historical_mask

evaluate(config, manual_resume_path, progress=on_committed_progress,
         lock_fds=(run_lock_fd, workspace_compute_lock_fd),
         validate_execution=check_admission_environment_and_snapshot)
# Only after every evaluation/comparison is verified:
publish(config, manual_resume_path, progress=on_committed_progress)
```

Evaluation is serial P2_BASE_GT then V2_MORPH_GT; six old complete evaluations are
reused. Reports include both raw F1/P/R profiles, pairwise four-way common-finite
populations and distinct all-six/all-eight populations, all six B/T markers,
fixed-distance pairwise block intervals, QC/label migration, native structure,
reference correlations, author 500-um spatial metrics, and local RNA transfers
by type, spatial block, density and gene. Guardrails and negative results remain
visible; no significance or multimetric winner gate exists.

Publication stages and atomically installs the new 30-entry directory only at
`/home/zwan0726/work/outputs/bidcell_historical_dual_bt_gt_fusion_20261017`.
It preserves all 28 old non-Purity data strings, accepts the old CSV's UTF-8 BOM,
and emits a UTF-8 header without BOM. Purity is independently rescaled for the
new 30-entry set. Actual mask pixels determine aliases and unique content count.
Publication returns `context_sync_complete: false`; context append and final
independent synchronization validation are controller responsibilities.

The actual historical C and SP final masks are absent. The manifest records their
original expected bytes/SHA256 and deterministic recovery paths, without claiming
their content has been read. This does not block tonight's six scientific inputs.
It does block exact unique-mask verification and publication. If the original
bytes are later found, `recover_historical_mask(config, grant, method, source)`
accepts only the frozen bytes/SHA256 and writes an actual copy to
`run_root/recovered_historical_masks/{C,SP}/final.tif` with a recovery receipt
bound to the original matrix receipt. It cannot admit a new scientific mask,
modify config/input digests or write the old path. No suitable archive has been
located; this is an unresolved data dependency, not a successful publication.

## Verification scope

Run the deterministic fixtures from the repository root with:

`/home/zwan0726/miniforge3/envs/bidcell_local/bin/python -m unittest tests.test_rep1_bt_gt_fusion_science -v`

Fixtures cover hand-calculated B/T panels/bins/ties/empty pools, two rebuilt parent
GT domains against the unmodified definition, margin and topology rejection,
interrupted GT/matrix/bootstrap recovery, an independent legacy full 2D matrix
oracle, four-way finite populations, local gene/spatial/density accounting,
manual gates, R inherited descriptors, all real publication column names/BOM,
and exact historical-mask recovery. Real-input freeze is read-only hashing only.
No real BT6, GT endpoint, classifier, complete evaluation or publication was run
during scientific implementation. Full resource/launch/admission tests remain
the separately audited controller's responsibility.
