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
  "run_id": "rep1-bt-gt-fusion-20261016-r02",
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

The r02 recovery uses the independently reviewed recovery amendment and exact
`recovery_import_seed.json` identity. Run `prepare --import-seed PATH` only after
the controller commit passes review, then run `adopt-checkpoints --config PATH`
from the new read-only snapshot. Adoption holds both runs' launch/writer locks
and the workspace compute lock. An isolated old-snapshot subprocess validates
the old admission, environment, inputs, B/T and completed proposal cursor.
Scientific binary/CSV files are independent byte-identical copies; JSON identity
edges are rehashed leaf-to-root and retain explicit producer/consumer provenance.
Common and GT full-module AST compatibility and 27 byte-fixed scientific sources
are checked. Import receipts never claim new scientific work or grant resume.
Final independent launch admission binds both the immutable plan and the complete
3394-file mapping receipt. Old r01 remains unchanged. P2 still must write its
final TIFF and pass full topology/matrix validation; V2 still runs in full.
The 1,285,263,411 imported bytes are part of the BT/P2 retained budgets; the
retained r01 is reported separately, not charged twice to incremental free space.
Adoption remeasures disk/RAM admission under all five locks before environment
probes or recovery writes. Every partial attempt keeps its resource report; the
completed mapping binds the successful attempt's report and is reused unchanged.
Every lexical target component must be free of symlinks, including dangling and
inside-run links. JSON paths map only selected files and their actual ancestor
directories; unknown producer-run references and transformed key collisions fail.

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

Final GT TIFF installation uses `common.atomic_tiff`: a named same-directory
temporary file, file fsync, read-only mapped uint32/shape/pixel verification in
bounded row blocks, atomic replace and directory fsync. This avoids the fixed
tifffile version's integer `.name` failure with `os.fdopen`, without encoding a
large TIFF in memory. GT scoring, topology, ordering and scientific checkpoint
formats are unchanged. A tiny fixture exercises `refine()` through its final
TIFF and complete receipt, including interruption before and after replace.

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

Spatial evaluation preserves the author's `cut()` range-based binning (nominal
500 um, at least two cuts per axis), entropy, CV of log10 values and SD of raw
values. The adapter retains matrix dimensions for a single observed class and
uses a common complete-finite bin population for all three CV metrics, and
separately for all three SD metrics. Single-cell SD/CV, zero log-mean CV,
constant correlation vectors and fewer than two common finite bins retain NA
with explicit per-bin/per-metric reasons and denominators. Unexpected R errors
still fail. The six author statistic names map explicitly to the existing
`scclassify_spatial_entropy_vs_*` table columns; unknown, duplicate or missing
statistics are rejected. Real-R fixtures compare normal and partially undefined
multi-type data with the untouched author implementation and feed the actual
CSV through both new table rows and prespecified endpoint comparisons.

Publication stages and atomically installs the new 30-entry directory only at
`/home/zwan0726/work/outputs/bidcell_historical_dual_bt_gt_fusion_20261017`.
It preserves all 28 old non-Purity data strings, accepts the old CSV's UTF-8 BOM,
and emits a UTF-8 header without BOM. Purity is independently rescaled for the
new 30-entry set. Actual mask pixels determine aliases and unique content count.
Publication returns `context_sync_complete: false`; context append and final
independent synchronization validation are controller responsibilities.

The historical C and SP final masks were deleted without backup, as confirmed
by the user and recorded in the separate 2026-10-16 RecoveryAmendment. Their old
input manifest and its original `available: false` entries remain unchanged.
Publication retains both historical negative-result rows, all old non-Purity
strings, the original mask identities and matrix/registry provenance. Only these
two names with frozen unavailable inputs may be missing. Any other missing mask,
a newly missing previously available mask, identity drift or incomplete recovery
fails; all eight scientific endpoint masks remain mandatory.

Registry, notes, publication and final registration receipts distinguish
`available_mask_method_entries`, `missing_mask_methods` and
`available_unique_mask_contents`. With C/SP absent, `unique_mask_contents` is
JSON null, `unique_mask_contents_verified` is false, and
`unique_mask_contents_status` is `not_fully_verified_missing_historical_masks`.
Context renders the whole-30 unique count as NA with the reason. Missing records
carry no content hash or alias; readable masks are independently hashed and
deduplicated, including actual new/parent no-op checks. Purity still uses all 30
rows' existing/new raw F1 values. Validation binds these exceptions to the exact
frozen input manifest and original matrix mask identities.

If exact original bytes are later found, `recover_historical_mask(config, grant, method, source)`
accepts only the frozen bytes/SHA256 and writes an actual copy to
`run_root/recovered_historical_masks/{C,SP}/final.tif` with a recovery receipt
bound to the original matrix receipt. It cannot admit a new scientific mask,
modify config/input digests or write the old path. Recovery is optional for the
two explicitly waived historical masks; their contents are never fabricated.

## Verification scope

Run the deterministic fixtures from the repository root with:

`/home/zwan0726/miniforge3/envs/bidcell_local/bin/python -m unittest tests.test_rep1_bt_gt_fusion_science -v`

The TIFF/missing-history repair regressions are
`python -m unittest tests.test_rep1_bt_gt_science_repair -v` with the same fixed
interpreter. They cover actual `refine()` completion and recovery, uint32 TIFF
interruption/corruption, all-30 publication validation with C/SP absent, exact
optional historical recovery, rejected false unique counts, and the actual
context/final-receipt consumer redirected entirely into synthetic fixture files.
`REP1_BT_GT_REPAIR_FIXTURE_ROOT` retains their inputs and receipts in an empty
evidence destination. Real scientific outputs and context are never written.

The suite includes small actual-R spatial calls with inherited lock FDs. Set
`REP1_BT_GT_R_FIXTURE_ROOT` to an empty evidence directory to retain their exact
inputs, snapshot copies, R logs, author comparisons and downstream consumer
receipts; without it those fixtures use temporary directories. Existing fixture
destinations are never overwritten. The separate tiny real-model prediction
recovery fixture accepts an explicit evidence directory and frozen input manifest
via `tests/rep1_bt_gt_prediction_fixture.py`.

Fixtures cover hand-calculated B/T panels/bins/ties/empty pools, two rebuilt parent
GT domains against the unmodified definition, margin and topology rejection,
interrupted GT/matrix/bootstrap recovery, an independent legacy full 2D matrix
oracle, four-way finite populations, local gene/spatial/density accounting,
manual gates, R inherited descriptors, all real publication column names/BOM,
and exact historical-mask recovery. Real-input freeze is read-only hashing only.
No real BT6, GT endpoint, classifier, complete evaluation or publication was run
during scientific implementation. Full resource/launch/admission tests remain
the separately audited controller's responsibility.

## Controller and immutable execution

The controller uses `control.sh`, `bootstrap.py`, `execution.py`, `runtime.py`
and `context_sync.py`. These are separately reviewed after scientific approval.
`prepare --plan --output-root --run-id` requires the clean committed authorized
branch and the exact planned run path. It hashes the actual inputs, copies the
complete dependency closure into `source_snapshot`, disables writes on all copy
objects, and saves the complete manifest and environment identity. No symlink or
hardlink is accepted. Large caches/temp files live outside the snapshot in the
new run. Environment identity covers interpreters, installed Python/R packages,
conda metadata, numerical binaries and every fixed entropy-library file.

Every supervisor/worker/R subprocess validates the snapshot and admission before
execution. Python uses `-I -S -B` and an import guard that checks delayed project
imports against the snapshot manifest. The worker starts an independent session
so tmux parent death cannot send HUP to its R descendants. Launch/writer/workspace
locks are inherited as the same open file descriptions; they are closed without
an explicit unlock. Actual orphan-R fixtures test parent, worker and both deaths.
All controlled environment values also use tmux `new-session -e`, so an existing
server cannot replace temp/cache paths or thread limits before Python imports.

The final auditor creates `admission/final.json`; the controller never signs its
own admission. Required bindings are source commit/digest, actual snapshot root,
canonical config/input/environment digests, plan/prompt SHA256, exact node order,
manifest/environment identities, and actual fixture/audit evidence. The reviewer
must be `gpt-6-astra/xhigh` with `implementation_author: false`.

`validate --config --admission` verifies that receipt. `launch --config` dispatches
only the four production nodes to tmux. Status `DISPATCHED_NOT_YET_VERIFIED` is
not launch success: the operator must check real process identities, all locks,
two fresh heartbeats, and a new committed scientific checkpoint (or a valid
already-completed terminal receipt). Revalidation of old chunks does not count
as new scientific computation. `resume-production` is an explicit tracked retry;
it validates identities and completed receipts, skips only valid completions,
and refuses surviving descendants/corrupt outputs. It never authorizes evaluation.
Real R prediction commits are recognized separately, with their RDS identity,
endpoint dependency digest, production/grant bindings and cursor checked. Reused
prediction batches do not advance the scientific progress counter.

`status --config --json` and `tail --config` are strictly read-only. They report
observed STALE separately from disk state; no restart, grant or control-file
creation occurs. Heartbeat period is 10s; >60s is stale. Scientific no-progress
>900s is an inspection flag, never an automatic kill/restart. Only after the
queue exits, `complete-check --config --phase production` obtains the compute
locks and validates production. It does not run evaluation.

Only after a new explicit user resume, an exited production hold and validated
receipts may the operator invoke `grant-resume --user-instruction-utc`. No grant
is prepared by launch/hold/status or a timer. `evaluate --manual-resume` starts
the separate serial manual evaluation phase; `publish --manual-resume` requires
its completed result and a primary results-analysis record. That derived
`results_analysis.json` must bind EVALUATION8 and all prespecified comparisons,
document the parallel F1 interpretation and local RNA limitations, and retain
negative results. This is results review, not another user approval request.
An interrupted manual evaluation uses the explicit `resume-evaluation` command
with the same valid manual grant and source/config/production identity. It only
accepts a previous evaluation attempt, retains the fresh STARTING grace, and
reuses valid complete endpoints/R batches. Production retry cannot roll evaluation
back into the production hold. These recovery commands never create a grant.

Publication then performs the scientific bundle validation and an atomic,
recoverable append of the eight context entry files, new experiment record and
two registry entries. Old context bytes are preserved in the append journal.
The final context/registration receipt binds table, registry, analysis and all
updated files. Goal completion remains a main-chat action after verification.

Storage budget separates 4GiB retained node room, 1GiB shared room, 5GiB additional
temporary/atomic overlap and a separate 15GiB reserve: initial incremental peak
10GiB, initial required free 25GiB. Existing bytes reduce remaining requirements;
unused endpoint room also covers manual evaluation. RAM admission uses a
conservative 20GiB workflow allowance plus 6GiB reserve, anchored by historical
full native oracle monitors and actual small fixtures, not tiny-test scaling.

Controller tests are `tests.test_rep1_bt_gt_controller` and the actual-host
`tests/rep1_bt_gt_runtime_fixture.py suite EVIDENCE_ROOT`. Engineering schemas,
synthetic grants and patched test backends are rejected by formal CLI/config
validation and cannot authorize the real run. Fixtures never use real endpoint
inputs or write historical outputs/context.
