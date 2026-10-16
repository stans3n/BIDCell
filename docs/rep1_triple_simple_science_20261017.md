# Scientific implementation handoff — 2026-10-17

This is scientific implementation and fixture evidence for independent review.
It is not code-audit approval, launch approval, or formal four-endpoint output.
The plan audit is `PASS_FOR_IMPLEMENTATION` with zero plan blockers; subsequent
scientific-code, supervisor and resource admission audits are distinct.

Base: `a205f46409ead17753ef71b57acd966aae792158`.
Branch: `feature/rep1-triple-simple-boundary-20261017`.
The committed implementation identity is recorded by
`tests/rep1_triple_source_manifest.py` in the external source manifest
`/home/zwan0726/work/docs/BIDCell_Rep1_Triple_Simple_ScienceSourceManifest_20261017.SCI01.json`.
This avoids embedding a self-referential commit hash in committed documentation.
The external implementation evidence manifest at
`/home/zwan0726/work/docs/BIDCell_Rep1_Triple_Simple_ScienceImplementation_20261017.SCI01.json`
binds that source manifest, actual fixture receipts/logs, resource assumptions
and remaining work. The original two reports without `.SCI01` remain byte
identical and bind the original science commit
`758b8e35aa51c16614b3782cdb5788f2f30dc988`. No pre-task source was changed.

## SCI-01 repair for independent re-review

Independent review1 of the original science commit found one blocker:
FULL_EVALUATE could complete before the required parent/BASE paired products.
The original review at
`/home/zwan0726/work/docs/BIDCell_Rep1_Triple_Simple_ScienceCodeAudit_20261017.review1.json`
is retained. This repair is submitted for independent re-review; it does not
declare that blocker independently closed or authorize launch.

`evaluation_pairs.py` now requires the two candidate-minus-control pairs at
each of stages 03/07/09/11 before full completion. The controls are respectively
P2_MORPH/BASE, MORPH_SIMPLE_FUSION/BASE, MORPH_SIMPLE_FUSION/BASE, and the
validated current-run P2_SIMPLE/BASE. It reads existing scores, full matrices,
annotations and centroids without triggering old-eight completion or R.
Actual six-metric four-way finite IDs and exclusions, both full profiles,
every frozen stratum and three global W scales are retained. The separate B/T
commonQC population retains all six markers and seven bins for both endpoints,
and signed fixed 0–40 µm paired CI at all three scales. Empty populations have
complete NA coverage and continue. Validation recomputes the actual finite
populations, marker numerators/pools and fixed-W B/T intervals; changing hashes
around an incomplete table cannot satisfy full evaluation.

The eight pair jobs already belong to the frozen 26-pair/69-job final analysis.
Stage 04 links the first two producers; stage 13 validates and prefers stage 04,
then links the other six evaluation producers. Same-run scientific data,
curves, compressed intervals and sufficient statistics are hardlinked rather
than recomputed or copied. Consumer receipts retain producer identity and the
config/source/input/domain/W/population/coefficient binding. The DAG and old
eight-method completion order are unchanged.

The SCI-01 storage increment is explicitly estimated at 64 MiB (67,108,864 B)
for the four evaluation paired wrappers, coverage/raw summaries and additional
consumer/inventory receipts. The scientific payload for these eight jobs was
already included in the original 26-pair/69-job estimate. The increment is
added to the original 15,223,392,152 B projection, giving approximately
15,290,501,016 B (14.2404 GiB), before rounding to the candidate 15 GiB analysis
allowance. This accounting does not silently consume and then add the existing
1 GiB inventory allowance twice. It is an estimate, not a measured full-run
upper bound or admission approval. Actual synthetic apparent/unique/allocated
bytes and the two-hop inode evidence are retained in
`F/SCI01_attempt_03/test_SCI01_complete_actual_pairs_required_and_reused/paired_reuse_storage.json`.
That record is written before intentional table corruption; the fixture then
re-signs every enclosing output hash and confirms semantic full-evaluation
validation still rejects the incomplete table. Its final corrupted test tree
is deliberately retained as rejection evidence, not an accepted science result.

For its two pairs, six IDs and three strata, the producer occupied 3,066,914
logical / 4,128,768 allocated bytes across 369 files. The initial consumer's
368 files added only 375,377 unique logical / 405,504 allocated bytes; the
second consumer added 373,795 / 405,504 bytes. Across all three trees this is
8,777,704 apparent bytes but 3,816,086 unique logical / 4,939,776 allocated
bytes after `(device,inode)` deduplication. These measurements demonstrate
reuse accounting at fixture scale; they do not bound full-scene metadata.

The API, stage mapping, plan-to-code table, scientific invariants, stage-04/13
reuse, failure/recovery and controller responsibilities are documented in
`scripts/rep1_triple_simple_boundary/README.md`. Scientific resampling uses
`spatial_bootstrap.py`; `bootstrap.py`, `runtime.py`, `execution.py`, `cli.py` and
`control.sh` remain reserved for the subsequent controller implementation.

## Approved identities

| Input | SHA256 |
|---|---|
| Launch prompt | 07052b65e43dfcbe23ee7416db2e1ccc47dce149ef2c7fd1d3c505e81bf5e4b8 |
| Triple plan | 74a365a18acc3df80332c9f4950cba927cfdf5566df6e69e3a3826d5367ecfd9 |
| Fixed boundary plan | 0bf85031be2e859025b697c2e128948b2995eafaeb81506203b24a0329be1580 |
| Simple controls plan | 11869b117cab49a64622275267aef074626342ba69412b3cd4284868e88a4226 |
| Execution plan | 176d610af6629699b5ea86d95a6ec8801df9e646b6ee44f1b335a3766613638d |
| Inspected inputs, 299 files | 06fc8b2521bdb38863dc7cdc4a8cd3758f2761ab8305412ed7d02db81c2c6995 |
| DAG | 9585ee48fa04961542aaeb4476eeb1e2f6c605b4a4a530eabbd94ea464a06be3 |
| Plan manifest | e7c1aa8ce8c87a7419f585e019ea99aa1a615ca477a8cae1d021e2d4fcb3250b |
| Root plan audit JSON | 4ec6026e72b52d80a11ad2f6415373b432bcc810894b7a56a823444ffe893743 |
| Root old-six spatial trace | 63656c51af6860498c4a19421500f598cb7846ca3d0642f9eefc4ad154c15040 |
| Root old-seven official B/T trace | 6799216927b886d722cd6812e33451391342ecdcf2e0370963647ee2fb4123bc |

The last two reports extend read-only input evidence without changing approved
plan bytes. The spatial trace binds complete original bins, six correlations,
original ownQC/annotation and receipts; original 30-method non-Purity values are
retained. The B/T trace binds seven original official ownQC populations and all
42 rows per endpoint. New-code numerical compatibility is independently tested.
The actual isolated extended input freeze binds 4,858 files and 13,221,340,757
bytes, including all owner/evaluation leaves and both trace reports.

## Commands and actual results

Run from the repository root using
`/home/zwan0726/miniforge3/envs/bidcell_local/bin/python`.
All commands set `PYTHONPATH=.` and
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`. The R wrapper uses
the frozen four-thread R protocol. Real heavy fixtures acquire
`/home/zwan0726/work/outputs/cos_gpu.lock`; heavy fixtures ran serially.
Output prefix `F` below means
`/home/zwan0726/work/outputs/bidcell_rep1_triple_simple_science_fixtures_20261017`.

| Fixture | Exact command suffix / output | Actual result |
|---|---|---|
| SCI-01 final full regression | `-m unittest tests.test_rep1_triple_simple_science tests.test_rep1_triple_evaluation_pairs -v`, `REP1_TRIPLE_SCIENCE_FIXTURE_ROOT=F/SCI01_attempt_03`; log `F/SCI01_attempt_03.log` | PASS, 30 tests, 112.913 s; includes two-hop reuse and re-signed truncation rejection |
| Original F01–F08 and scientific F10–F14 synthetic suite | `-m unittest tests.test_rep1_triple_simple_science -v`, `REP1_TRIPLE_SCIENCE_FIXTURE_ROOT=F/attempt_06/synthetic`; log `F/attempt_06/synthetic.log` | PASS at original science commit, 27 tests, 41.859 s |
| Four actual maximum-candidate Simple tiles | `tests/rep1_triple_p2_resource_fixture.py --output F/attempt_01/F14_max_tiles` | PASS, 4/4, 1319.594 s; `complete.json` and four native/P2 tile receipts |
| Actual Simple small ROI, workers 1/4, interrupted tile recovery | `tests/rep1_triple_real_fixtures.py mini --output F/attempt_02/mini` | PASS; `fixture_receipt.json`; fixed 128×128 ROI, complete global nuclei, no full-slide replay claim |
| Actual maximum R prediction batch and recovery | `tests/rep1_triple_real_fixtures.py r_batch --output F/attempt_06/r_batch` | PASS; actual 256 cells ×313 genes, annotation/RDS/committed receipt bytes unchanged on recovery; log and receipt retained |
| Actual R callback and execution-validator failures | `tests/rep1_triple_real_fixtures.py r_failure --output F/attempt_05/r_failure` | PASS; both owned R processes reaped with actual exit -15; failed pending evidence retained |
| Seven original official B/T profiles | `tests/rep1_triple_real_fixtures.py old_bt --output F/attempt_03/old_bt` | PASS; all original recipients/targets, full nearest/min-ID ties, 7 bins ×6 markers numerically identical for all seven endpoints |
| Actual input adapter and first semantic witness | `tests/rep1_triple_inputs_fixture.py --output F/attempt_04/inputs` | PASS; full cache producer identity, all 69 chunks/row spans, old trace adapters and first old-BASE full-ID/nuclear/matrix receipt |
| Real-N conservative complete-group statistics | `tests/rep1_triple_statistics_resource_fixture.py --output F/attempt_06/statistics_resource` | PASS resource probe; 107955 actual IDs, 1600 groups, 2000 replicates, all three actual block universes |
| Actual old BASE→Simple max-change 29-column sparse resource payload | `tests/rep1_triple_transfer_resource_fixture.py --output F/attempt_06/transfer_resource_with_budget --manifest F/attempt_04/inputs/configs/inputs_manifest.json --statistics F/attempt_06/statistics_resource/fixture_receipt.json` | PASS resource probe; all69 owner chunks enumerated; selected chunk37, 85321 changed/unique rows; full resource projection in receipt |

The old-seven B/T fixture was invoked through its Python function in an isolated
serial driver; the table gives the equivalent supported CLI. R failure coverage
was extended after the initial callback-only pass. Final compilation and
`git diff --check` are recorded in the external evidence manifest.
The actual-input/P2/R/group/transfer fixtures above remain the original
measurements bound by review1 and the preserved implementation report. They
were not rerun for SCI-01. Only the complete synthetic regression and the
three new paired-evaluation tests were rerun. Earlier SCI-01 attempts 01
(three tests, 72.785 s) and 02 (30 tests, 113.701 s) passed and are retained;
attempt 03 adds the final two-hop and re-signed-truncation assertions.

The 27 retained synthetic tests include the original four exhaustive native/P2
fixtures and four GT final-TIFF/repair fixtures, adapted to the new package and
three GT endpoint names. They also cover independent full313 CSV/count/owner
oracle, large uint32 IDs, OOB, gene association, duplicate IDs, parent-relative
conservation and dynamic-parent rejection; complete coordinate cache producer
keys; Unknown-priority/interface overlap; four-way/eight/twelve finite
populations and 19 hand-calculated contrasts; fixed ROI seeds; complete sparse
ledger/gross-net accounting; categorical versus expanded spatial groups;
actual full-pair table keys and truncation rejection; atomic retained failures;
bootstrap batch/corruption/retirement recovery; deterministic JSON gzip;
independent compute/publication authorization; R-resource sampler failure
rejection; and rejecting matrix-only claims of full evaluation.
The three SCI-01 tests additionally cover all four true-parent/BASE mappings,
actual finite/QC separation (including finite zero), signed B/T changes,
all six markers/seven bins/three scales, legal empty populations, absence of
old-eight completion or R reclassification, required paired evidence before
full evaluation, two-hop immutable partition reuse, and refusal of re-signed
incomplete paired tables.

ROI seed indices are exactly `[20261017, region_index, type_index, density]`.
Known interface classes use 0/1/2 for same/heterotypic/Unknown. Control regions
use a separate nonnegative sentinel 3. This convention is tested before any
formal ROI selection. Overlapping interface exposures separately retain any
Unknown, any heterotypic pair and multiple known types; unique assignment still
uses the frozen minimum `(distance_squared, i, j)` rule.

## Resource evidence and admission implications

| Probe | Actual measurement | Limits |
|---|---|---|
| Four native/P2 max tiles | K=5199/4922/4732/4578; 5 processes; peak PSS 1,600,477,184 B; private 778,571,776 B; summed RSS 4,877,504,512 B | Native saved tile pixels and candidate/RNG-enter-cleanup identities exact; separate cleanup exit RNG retained. This is not full disabled replay or whole-stage peak |
| Real mini | Peak PSS 1,362,591,744 B, 5 processes | FD inheritance list empty: evidence not collected, not a lock PASS |
| R maximum batch | Final peak PSS 2,387,611,648 B; previous valid run2,405,395,456 B; 28 samples; 2 processes | Actual R process inherited both run/workspace lock inode FDs. No grandchild observed; parent-death lifecycle remains controller F09 |
| Group bound, final compressed representation | Peak PSS 7,752,315,904 B; private 7,751,979,008 B; 29 samples; 15.104 s | Every actual ID in all1600 groups is an allocation bound, not formal frozen memberships |
| Sparse resource payload | Real max changed chunk37: 500000 rows /85321 changed; final resource receipt records exact PSS and wall time | Real owners/genes/types/positions/cell properties; resource-only region/density/interface columns are explicitly high-entropy bounded values, never scientific assignments |
| Root full-shape oracle allocation | 25779×35416 uint32; 3,651,956,256 array B; PSS 3,875,246,080 B; peak RSS 4,044,587,008 B | Allocation only; not full independent matrix-oracle execution |

Root allocation source:
`/home/zwan0726/work/docs/BIDCell_Rep1_Triple_Simple_RootOracleArrayProbe_20261017.json`.
The extended input freeze took 68.776 s; complete leaf rehash took 23.688 s;
first old-BASE semantic receipt then took 3.012 s. Thus the measured stage-01
entry path before first semantic evidence was 26.701 s. This is a measured
warm-filesystem path, not permission to skip input identity checks.

The statistical resource probe uses actual 48/176/12 block universes for
1000/500/2000 µm, PCG64 W, real 107955 IDs, an explicit 1600-group bound over the
prespecified upper count1498, and independently dense statistics. Comparing
all three old uncompressed and new compressed outputs confirms exactly equal
JSON values. Their 2,138,816 bytes become 326,209 bytes; at 69×12 jobs this saves
1,500,838,596 bytes without dropping values, W, sufficient statistics or receipts.

The following projections include complete interval JSON.gz, block sufficient
statistics, retired-batch identities, final receipts and aggregate gzip tables.
They are estimates from the dense probe, not absolute compression guarantees.

| Analysis component | Count basis | Projected bytes |
|---|---|---:|
| Raw, paired, factorial and all12 scores | 69 jobs ×12 profile/metrics ×3 scales ×1600 groups; 3,974,400 CI rows | 3,692,380,716 |
| Cell transfer and 4/8 structure costs | 26 ×9 metrics ×3 scales ×4993 group bound; 3,505,086 rows | 3,256,368,749 |
| Full pixel/molecule spatial strata | 26 ×2 units ×6 metrics ×3 scales ×1836 groups; 1,718,496 rows | 1,596,553,314 |
| All-overlap interface statistics | 26 ×2 units ×3 metrics ×3 scales ×1604 groups; 750,672 rows | 697,405,097 |
| Regional and fixed-window paired B/T CI | 2340 +468 rows | 2,608,747 |
| Sparse full29 gene/owner and full28 pixel ledgers | All26 pairs at actual old BASE→Simple change workload, measured high-entropy full-column compression | 3,580,540,373 |
| B/T full profiles | 108 profiles × largest real old profile footprint | 115,833,780 |
| New B/T plots/receipts allowance | Explicit estimate | 134,217,728 |
| All32 ROI data,12 masks, all26 comparison panels | Explicit estimate,128 µm crops | 536,870,912 |
| Full cell distributions/raw tables/populations/regional endpoints | Explicit estimate | 536,870,912 |
| Nested inventories, long stratum names, state snapshots | Explicit estimate | 1,073,741,824 |
| **Total analysis retained estimate** | Includes every category above | **15,223,392,152 =14.178 GiB** |

The 4993 local-group bound includes 3393 actual type×block/RNA/crowding
combinations in addition to the1600 frozen-group bound. Actual old full changes
are 3,913,530 molecules and 2,786,442 pixels, with148464 BASE interface pairs.
The measured per-unique-row compressed sizes are20.621/20.461 bytes. These old
workloads do not bound new endpoints. Absolute raw all-change bounds are also
reported and exceed available disk; resource admission must combine defensible
workload estimates and bounded per-chunk checks, retaining engineering failure
evidence if estimates are exceeded.

Scoring alone projects about10,097 s (2.80 h) under the all-ID membership load;
other analysis tasks add runtime. This is not the former1-hour estimate. Only
one group projector/job is retained; repeated denominator block counts are
cached within its bounded population set. Temporary overlap includes the
2000×G sample array (25.6 MB forG1600,79.9 MB forG4993), at most20 compressed
100-replicate batches awaiting final commit, the active sparse chunk, and one
previous committed cursor snapshot until the next is installed. Completed
bootstrap batches retire only after all remaining bytes verify against their
committed identities; interrupted/corrupt/uncommitted artifacts remain.

The original24 GiB total /6 GiB analysis budget is **not validated**. If analysis
is rounded to15 GiB, that planning decomposition becomes33 GiB plus the separate
15 GiB reserve. Available disk at the last science measurement was47.84 GiB,
slightly below48 GiB. The root must reconcile actual allocated existing/new
sizes, remaining peak and improved representation before launch. No old files
were deleted and no output/step was removed to satisfy the former estimate.
RAM measurements likewise describe individual probes, not an asserted whole
stage; the controller must validate the full20–22 GiB stage envelope plus6 GiB
reserve. Large temporaries remain on work storage.

## Retained failures and remaining work

- Initial resource script failed before computation because `psutil` was absent;
  `/proc` sampling replaced that unavailable dependency.
- `attempt_02/r_batch` numerics/recovery succeeded, but its sampler thread hit
  `ProcessLookupError` during normal process exit. Its original log is retained,
  and `fixture_receipt_correction.json` explicitly invalidates that resource
  measurement. Final code handles exit races and raises on unexpected sampler
  errors from both context exit and result access. The affected R measurement
  was rerun successfully in `attempt_03/r_batch`. After switching R temporaries
  to unique same-directory `tempfile` names, the final actual256 batch and
  recovery passed again in `attempt_06/r_batch`.
- `attempt_03/inputs` rejected an original evaluation8 B/T receipt because the
  adapter expected an absent `n_curves` field. The adapter now verifies the
  actual unique42-row table and original receipt outputs. The failed log is
  retained; `attempt_04/inputs` passed full validation.
- Earlier12/25/26 synthetic and pre-gzip resource runs remain as superseded
  evidence. The final27-test suite and final compressed-resource run are the
  current measurements. The intentionally interrupted/corrupt fixtures retain
  their evidence.

No scientific-code placeholder is declared complete. Formal frozen-domain
preparation, full disabled Simple replay, all four full productions and full
evaluations, final12-method analysis and compute-complete have not been run.
The small fixtures do not substitute for those required formal DAG nodes.

Independent review1 completed with only SCI-01 open; the repair still requires
independent re-review. Controller source freezing,
interpreter isolation, resource admission/failure controls, host parent-death
and orphan-lock rejection, read-only status/tail, startup proof, and controller
portions of F09–F14 remain separate pending work. Resource capacity reconciliation
above is an explicit unresolved admission condition. Publication, independent
result interpretation, formal34-method registration and context sync remain
behind their later manual gate; none occurred in this implementation phase.

## Temporary disk audit notes for controller review

The3.652 GB resized oracle is an in-memory NumPy/OpenCV array, not disk spill.
All significant atomic writers use unique same-directory files on work. A
successful rename installs that temporary as a retained output; do not count
it once as retained and again as an equal extra temporary. During replacement,
the old committed output and new temporary can coexist. Failed temporaries are
retained and must be charged to actual usage before any restart.

| Stage / writer | Simultaneously present successful-path temporary / overlap |
|---|---|
| Matrix generation | One 107955×313 int64 atomic counts array:270,319,320 B plus NPY header. CSV is streamed through one same-disk file, not staged as a second encoded memory/disk copy. One500000-row owner chunk is2 MB; aggregate/oracle compressed arrays are retained, sequential writes. The full resized oracle never spills |
| GT apply | The previous verified full-mask cursor and the next atomic mask can coexist:2×164,909,712 B, plus two decision vectors of one byte per proposal and a small state receipt. Final TIFF is an additional retained164,909,712 B. Only the verified previous cursor retires |
| GT domain / score | Query, expression, support, legal-domain, geometry and score chunks remain retained. Each writer is sequential; merged query/domain/expression/proposal/score files are additional retained representations. They are part of the GT4.5 GiB planning allowance, not a hidden temporary exemption. Their new-parent totals have not been measured full-slide |
| P2 four workers | Each worker writes at most one atomic1024×1024 uint32 TIFF at a time:four×4 MiB; postconnect and cleanup tile TIFFs then remain retained. Claim/details NPZ is another bounded active tile write; candidate count is measured by the max-tile fixture. Tile outputs, all merged stage TIFFs, split-cleanup chunks and replay matrices/validation are retained within the P2/replay allowance |
| P2 full-mask merge | One full uint32 stage TIFF atomic write is164,909,712 B; all successful previous stage files remain retained. Endpoint final is currently an additional atomic copy of predictions/final.tif, charged as retained, not assumed deduplicated |
| R | One maximum256-cell prediction RDS plus pending JSON uses unique `tempfile` paths. Batch RDS, committed receipt and final combined raw-prediction RDS all remain retained. Real batch prediction_chunks used20,480 allocated B; the post-exit Rtmp directory used4096 B. That does not prove a whole-run Rtmp ceiling; the controller monitors its actual path under the evaluation output |
| Bootstrap | One active group/metric/scale holds at most20 successful100-replicate NPZ batches until final commit. AtG4993, raw sample payload across all batches is79,888,000 B (RAM complete sample array is separately the same size); one active batch3,994,400 B; one output temporary and complete sufficient statistics can coexist before verified retirement. Persistent intervals/NPZ/identity lists are included in the analysis estimate |
| Local spatial cursor | One previous verified snapshot and the next snapshot coexist. With236 blocks,1836 strata,5 flow counts andint64, raw block sufficient arrays are17,331,840 B per snapshot plus totals, across all three scales; compression is not assumed for the bound |
| Transfer/ROI | One sparse500000-molecule chunk has at most120 MB raw full29 keys+count payload; a128-row full-width pixel chunk has at most223,492,096 B raw full28 keys+count. NPZ writing is sequential. ROI data/panels remain retained; rasterized molecule panels do not serialize one PDF vector object per molecule |

The pipeline stages are serial; matrix, R, GT and P2 temporary peaks are not
added as if they ran concurrently. Within P2, four child writers do coexist.
Bootstrap and local spatial writers are sequential within one comparison.
These code bounds explain why a revised successful-path temporary allowance
may be smaller than the original6 GiB, but this implementation handoff does
**not** approve a4 GiB temporary budget or a46 GiB admission. Supervisor
per-write checks must include current failed/partial artifacts, actual retained
production/domain growth and library Rtmp behavior. The original4.5 GiB GT and
5.5 GiB P2/replay values include all named retained products by definition; they
remain historical extrapolations until validated against current frozen inputs
and actual growth. The independent auditor/root own that resource decision.
