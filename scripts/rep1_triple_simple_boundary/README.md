# Rep1 Triple / Simple scientific interface

This package implements the science in the four plans approved on 2026-10-17.
`approved_dag.json` is an exact byte copy of the audited DAG, checked on import.
The implementation creates four endpoints: P2_MORPH_GT, P2_SIMPLE, SIMPLE_GT,
and P2_SIMPLE_GT. It contains no launch, queue, publication, context update,
or controller entry point. `bootstrap.py` is reserved for the controller;
scientific resampling lives in `spatial_bootstrap.py`.

## Controller API

```python
from scripts.rep1_triple_simple_boundary import inputs, frozen_domains, science

# Live preparation only: source_dependencies() uses git to enumerate tracked
# sources. The supervisor copies all tracked repository dependencies and the
# four listed external R sources into source_snapshot before freezing domains.
source_files = inputs.source_dependencies()
manifest = inputs.freeze_inputs(run_root, inspected_inputs_path, plan_manifest_path,
                                progress=callback)
domain_receipt = frozen_domains.freeze(run_root, manifest, source_digest,
                                       progress=callback)

# Invoke these imports from the immutable snapshot in the admitted worker.
result = science.run_stage(config, stage, progress=callback,
                          lock_fds=(run_fd, workspace_fd),
                          validate_execution=validator,
                          compute_authorization=runtime_payload)
science.validate_stage(config, stage)
path = science.receipt_path(config, stage)
```

`config` may be a dictionary or an absolute JSON path. Its schema is
`rep1-triple-simple-boundary-run-v1`. Required fields are `run_id`, `run_root`,
`source_snapshot`, `source_digest`, `input_manifest`, `input_digest`,
`domain_manifest`, `domain_digest`, `plan_manifest`, `plan_digest`,
`comparison_manifest`, `python`, `rscript`, `resources`, and `stage_order`.
The input digest is `common.digest(manifest)`; the domain digest is the SHA256
of `frozen_domain/complete.json`; the plan digest is the SHA256 of the approved
plan manifest. The comparison manifest is the byte-identical approved DAG.
`resources.p2_workers` is 1, 2, or 4. The R environment additionally accepts
`r_libs_user` and `r_libs_site`.

The immutable configuration contains only this authorization declaration:

```json
{"compute_authorization":{"mode":"admitted_compute","allowed_stages":["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15"]}}
```

The controller constructs a runtime payload after successful admission, without
rewriting the configuration:

```python
dict(authorization="admitted_compute", run_id=cfg["run_id"],
     config_digest=common.digest(cfg), admission=common.identity(admission_path),
     stage=stage, production=common.identity(production_receipt) if evaluation else None)
```

The admission receipt must bind the same config digest, have a successful
decision and zero open blockers. Evaluation payloads additionally bind the
correct endpoint's preceding successful production receipt. The controller's
validator remains responsible for independent admission, immutable source and
resource validation. It is called before and after R and while R is alive;
it should use its validated fingerprint cache for cheap repeated checks.

All stages require inherited run/workspace file descriptors and the validator.
Production and evaluation run in snapshot modules; source paths are resolved
from `__file__` or `source_snapshot/external_sources`, while frozen data paths
refer to the read-only original inputs. `source_dependencies()` is never called
from `run_stage`. The API currently rehashes the input manifest leaves on each
stage entry. The isolated input fixture reports the measured cost before the
first stage-01 semantic receipt; a controller-side validation cache must be
audited separately before replacing this validation.

Callbacks receive `{stage, cursor, total, detail}`. Committed chunks expose an
absolute `detail.receipt`; `common.last_committed_witness()` returns the last
such path. The supervisor must verify its identity/dependencies/outputs and
deduplicate witnesses. Revalidating old evidence is not new scientific advance.
R batch witnesses use `.rds.committed.json` with the batch hash, dependency
digest, cursor, and total.

`run_stage` performs stages 01–14. Stage 15 is a supervisor terminal commit with
exit code zero, `terminal_state=COMPUTE_COMPLETE_AWAITING_MANUAL_RESUME`, and the
identity of `compute_complete.json`; it is validated by `validate_stage`.
Scientific negative values, no-op endpoints, retention alarms, and unfavorable
comparisons never authorize stopping. Publication, interpretation, and context
sync require a separate user-authorized manual receipt and are outside this
package's implementation scope.

## Science and recovery

P2 uses the original candidate enumeration, shuffled RNG stream, per-tile seed,
native cleanup, complete global nuclear coordinates, exact int64 squared
distance, and minimum-ID tie rule. Full disabled Simple replay is stage 05;
the real small-ROI and maximum-tile fixtures do not replace it. GT reconstructs
each parent's domain independently and retains A1 eligibility, type prior,
score/margin and complete-cell topology. P2_SIMPLE_GT resolves only the validated
stage-06 P2_SIMPLE parent in this run.

Every new final is rebuilt into the full 107955 × 313 int64 matrix. Independent
validation reads the original 34442716 eligible CSV rows, uses the actual 2-D
OpenCV label resize and pandas rounding, and reconstructs all owners/counts.
Assigned totals are computed from data. The historical Simple total is a
comparison value, not an endpoint lookup or production constant.

The BASE competition region, 2 µm interface band, Unknown priority, all frozen
cell/position/type/density/shape strata, 32 ROI slots, original molecule points,
and global W at 1000/500/2000 µm are frozen before any new final. The 32 ROI
SeedSequence indices are `[20261017, region_index, type_index, density_index]`;
same/heterotypic/Unknown use type indices 0/1/2, and control strata use the
separate nonnegative sentinel 3. Empty strata retain a reason and are not
replaced. Overlapping strata and interface exposures remain explicitly
nonadditive.

Stage 04 uses the nine then-available endpoint matrices and score tables. It
does not run stage-12 old-endpoint completion early. Missing old full 4/8
structure, missing official B/T, and their local structure costs are marked
pending. Existing old six-method spatial files are traced through the 30-method
registry and original comparison receipts, including all bins and six scalars.
Seven old official six-marker profiles are traced through `evaluation8`; old
commonID6/commonQC6/commonID8 populations are not reused as new common populations.
The two old GT full structural products are validated and reused when present.

Stage 12 supplements the old eight endpoints without reclassifying them.
Stage 13 checks stage 04's config/domain/W/population/coefficients and reuses its
completed endpoint partitions, pairwise/factorial/raw-score jobs, B/T pair
profiles, and full transfer cores. It adds new endpoints/pairs, common-all12
populations, and the now-available structure costs. Producer/consumer receipts
are retained. Same-run immutable partitions use hard links; historical files
are copied or read without mutation.

Every FULL_EVALUATE stage (03/07/09/11) also completes exactly two paired
comparisons before its full-evaluation receipt: candidate minus its true parent,
and candidate minus BASE_MATCH. The parent mapping is P2_MORPH_GT→P2_MORPH,
P2_SIMPLE→MORPH_SIMPLE_FUSION, SIMPLE_GT→MORPH_SIMPLE_FUSION, and
P2_SIMPLE_GT→this run's validated P2_SIMPLE. `evaluation_pairs.py` uses only
the existing parent/BASE matrices, scores, annotations and centroids; it does
not call old-eight completion or R classification. For each of the six metrics,
the same four-way finite population is used for sc and fixed profiles, with
the actual IDs, hashes and all QC/missing/nonfinite exclusions retained. B/T
instead uses fixed-panel intersect both ownQC, without the score-finite filter:
both full six-marker/seven-bin curves and signed candidate-minus-control
0–40 µm CI at all three scales are required. Empty populations retain complete
NA rows. Semantic validation independently checks actual arrays, every frozen
stratum key, source marker counts, pools and frozen-W B/T intervals.

These eight paired jobs are members of the already frozen 26 pairs. Stage 04
reuses the two P2_MORPH_GT evaluation producers; stage 13 prefers its validated
stage-04 producers, then reuses the other six evaluation producers. Scientific
data and bootstrap sufficient statistics are hardlinked, with distinct consumer
receipts binding the original producer. No extra pair or repeated CI job is
introduced. Additional evaluation wrapper/coverage/raw-summary/consumer
receipts have a separately reported storage allowance in the SCI-01 handoff.

The final comparison set is 26 pairs, 12 conditional edges, 6 second-order and
1 third-order contrast. There are also 12 raw full-profile and 12 common-all12
endpoint mean jobs: 69 final comparison jobs. Pairwise four-way finite,
eight-cell factorial common, twelve-method common, raw ownQC and raw fixed
full-panel populations remain separate. A finite zero remains a value.

All 313 genes appear in the complete sparse transfer schema, including owner 0,
Unknown, source and target IDs/types, density/crowding/shape, spatial block,
support multiplicity and interface membership. Sparse rows contain every
observed combination, with complete full-matrix delta conservation. Gross
outgoing/incoming transfers and net differences are separate. Pixel, molecule,
cell and interface costs use fixed denominators and shared spatial weights.
Local spatial categorical aggregation is exactly checked against the expanded
individual-stratum implementation; it changes only the scan strategy.

Spatial bootstrap uses 2000 shared PCG64/20261013 rows, pooled block sums/counts,
percentile CI, valid-replicate and occupied-block rules, and LOBO diagnostics.
Temporary batches contain 100 replicates each. Intervals are losslessly stored
as deterministic `intervals.json.gz` (safe JSON, gzip mtime zero); recovery
validates its bytes and reads the same complete values. After complete intervals, exact
block sufficient statistics, W/ID/definition dependencies and a batch identity
list have committed, successful temporary batches are hash-verified and retired.
All 2000 samples are exactly reconstructible from retained statistics and W.
Retirement validates every surviving file before deleting any; interrupted
retirement resumes. Corrupt, failed, and uncommitted bytes are retained. This
policy applies only to new-run successful temporaries; original inputs are
never removed. Mutable GT/moment/statistical cursor recovery retires only the
previous verified committed snapshot after the next snapshot commits.

Atomic file failures preserve unique same-directory temporary files. A corrupt
committed chunk fails before regeneration; recovery must not overwrite it to
make an existing receipt match. R callback/validator engineering exceptions
terminate only the owned R process group, wait for its actual exit while the
locks stay open, and record the real exit code. No inactivity timer or Goal
pause kills computation. The controller separately handles parent-death/orphan
admission and process identity.

Full evaluation requires ownQC313/logNormCounts, frozen reference307
scClassify, all six author and fixed metrics, QC/unassigned/missing/unknown,
full structural distributions with 4/8 connectivity and nuclear preservation,
reference expression/detection definitions, author spatial diversity, and all
six B/T markers with full seven-bin curves and fixed 0–40 µm paired CI. A matrix
or F1 output alone cannot satisfy semantic completion. Completion checks actual
unique metric/profile/scale/stratum keys against the frozen definitions, actual
finite-population IDs and contrast coefficients, all six marker/bin grids,
regional and interface rows, full313 gene deltas and all changed pixel chunks.
Changing the hash of a truncated table cannot make it complete.

## Plan-to-code and validation

| Approved requirement | Implementation | Fixture |
|---|---|---|
| Parent graph and static scope | contracts, inputs, science | F03, F12, F13 |
| P2 identity/native disabled replay | p2, original p2_rules | F01, real mini, max 4 tiles |
| Per-parent GT/A1/topology | gt, original core_refine/core_scores | F02, actual final-TIFF/recovery |
| Full313 matrix and independent CSV oracle | matrix | F04, uint32/OOB/gene/duplicate cases |
| Complete coordinate producer admission | inputs | F04, actual input adapter |
| BASE domains, interfaces, strata, ROI | frozen_domains, roi | F05, F08, corrupt exposure recovery |
| Four-way/eight/twelve populations and raw scores | populations | F06 |
| Shared W, 19 contrasts and every stratum | comparisons, spatial_bootstrap | F06, F07, real-N resource probe |
| Full transfer/local/interface costs | boundary_analysis, local_statistics | F08 categorical/ledger conservation |
| Complete evaluation and R snapshots | evaluation, paper_evaluation.R, spatial_diversity.R | F11, R max batch/recovery, R failure |
| Parent/BASE pairing before each full evaluation | evaluation_pairs, contracts, populations, comparisons, bt, coverage | SCI-01 four-parent map, actual complete/empty grids, re-signed truncation rejection and two-hop reuse |
| Six-marker official/fixed B/T | bt | F07 hand calculations, seven real legacy profiles |
| Atomic recovery and stage-04 reuse | common and all chunk writers | scientific F10 |
| Full semantic coverage / terminal evidence | coverage, science | F12/F13 |
| Runtime/supervisor isolation, orphan locks | separate controller work | F09 and controller portions F10–F14 pending |

Fixture commands and actual results are recorded in
`docs/rep1_triple_simple_science_20261017.md`. A fixture marked pending is not
launch evidence. No fixture is a formal four-endpoint result or a full disabled
Simple replay. The independent scientific audit and subsequent controller
audit/admission remain required.

Measured analysis output is larger than the initial planning allowance. The
complete resource projection, including full scoring/local/spatial/interface
statistics, 29-column transfer payloads, B/T, ROI, receipts and temporary
overlap, is in the handoff document. It is an estimate with explicit assumptions;
the original 24 GiB peak / 6 GiB analysis allowance is not a validated admission.
