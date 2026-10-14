# A_CORE implementation and validation

Owner: explicitly requested A_CORE implementation agent, actual spawned model
`gpt-6-astra`, reasoning `xhigh`. Worktree:
`/home/zwan0726/work/projects/BIDCell_boundary_three_a_core_20261015`, branch
`feature/boundary-three-a-core`. Parent owns shared contracts/evaluation/reporting.
No historical sources/results are written. No A3, new loss, model training,
architecture, sweep, formal experiment or scientific threshold change is added.

## Plan-to-code mapping

| Frozen changeplan requirement | Implementation | Evidence |
|---|---|---|
| Native ties-to-even/OpenCV coordinate contract; vendor owner ignored; full denominator | `a_core._inputs`, `_coordinates` consume the original per-record cache only after original full 2D oracle receipt and input/cache identities agree; chunk ranges and gene bounds checked | `test_native_lookup_independent_two_dimensional_oracle`; original validated cache receipt; real mini provenance |
| Original full ID/type/area/centroid mapping; 313 genes; training-only reference | `_inputs`; `core_scores.reference_arrays`, `legal_types` | exact nucleus/preannotation ID/type equality, computed area/centroids; name/order validation; Unknown test |
| No irrelevant Evidence DAPI or raster-count load | `core_refine.NuclearSupport` contains only complete nuclei/support logic; reference uses only names/indices/theta/valid | input manifest states no DAPI; implementation imports no Evidence |
| Full-nucleus EDT≥2, empty Core no whole-cell fallback | `NuclearSupport.core_image` pads each complete nucleus with a real zero halo | specimen-edge, touching/degenerate Core fixture |
| SHA256 molecule split; F0/F1≥10; no leakage or duplicate IDs; coverage | `_count_core`, `molecule_fold`, `cross_scores` | per-chunk source IDs/fold/row/gene/pixel, global ID uniqueness, count conservation, leakage challenge |
| Fixed type-shrunk q; no double-counted Type term | `profile`, `cross_scores`, `score_proposals` | zero-count fallback, normalization, reversed heldout expression test |
| Cell-equal primary log predictive increment | `cross_scores`, `_write_statistics` | within-cell fold pooling by heldout n, then cell-equal bootstrap; molecule-weighted auxiliary explicitly named |
| 20 typed/count-bin derangements, no self-maps; paired population | `matched_donors`, `cross_scores` | all seeds persisted; actual donor/recipient counts in shuffle CSV; singleton/no-reference fixtures |
| Fixed shared-weight 1000µm block bootstrap; 500/2000 sensitivity; unchanged gate | `paired_bootstrap`, `decide`, `_write_statistics` | fixed PCG64, 2000 resamples, same weights, all population/interval boundary tests |
| Split stability, type residual stability, type/count/density/neighbour strata | `cross_scores`, `_strata` | per-cell raw/profile/type-centered similarities; frozen full-support same/heterotypic flags |
| Nuclear heldout candidate diagnosis vs outer support only | `_support_diagnostics` | every candidate/geometry saved; same-type heldout expression margins separate; outer owner accuracy explicitly unavailable |
| A2 full-count donor seed, no Core/edited-molecule overlap | `_a2_masks` | new full-count donor table; nuclear-exterior domain assertion |
| Same-domain NATIVE/GT/GTC/SHUFFLED; candidate-wide abstention | `interface_band`, `fixed_domain`, `_a2_masks` | fixed baseline 8-neighbour cell interfaces, Euclidean band2, baseline adjacency/support, all candidate eligibility/donor checks |
| Mean per-molecule pixel log score, log2 margin, deterministic ties/order | `score_proposals` | multigene/pixel test, current-owner ties, global-ID ties, disabled identity |
| Complete connectivity; no new island or attachment to orphan island | `CompleteCellTopology`, `topology_oracle`, `apply_proposals` | full-cell bboxes explicitly include all baseline components/all future proposal positions; independent full-image oracle in tests and every mini proposal |
| Constant foreground/capture; no postprocess/P2 rerun | `_a2_masks`, `_run_a2`, `validate_node` | exact foreground/nuclear masks; own integer matrix assigned and eligible counts compared exactly across four arms |
| Validated NATIVE reuse; three new own matrices/full evaluators; final comparisons | `_run_a2` calls parent `evaluation.validated_reuse/build_and_evaluate` and `reports.compare_endpoints` | three required GTC contrasts plus full frozen reports; no fabricated native prediction |
| A1 exact chunk recovery; A2 exact proposal/mask recovery | `_count_core`, `apply_proposals` | immutable per-range sparse counts; atomic generation snapshots and cursors; interrupted/direct output equivalence |
| Input/config/source/output receipts, semantic validation and failures | `run_node`, `validate_node`, shared `common` | no node complete receipt before every required artifact/evaluator; failures record INVALID separately from legal negatives |

## Explicit preformal clarification for audit

The plan says a type-limited A1 signal must not force A2 outside its established
scope, but does not give an applicability statistic. Parent integration explicitly
instructed a conservative clarification before viewing formal A1 results: retain
the unchanged global A1 gate, apply that exact same gate and coverage thresholds
per type to describe established scope, and make every A2 arm abstain for types
without established scope (`INCONCLUSIVE_SCOPE`). There is no lower sample/block
threshold or new positive-mean gate. This can produce legitimate zero-action A2.
The independent auditor must review this clarification against §3.5/§4.1; it is
not concealed as original plan text. Descriptive strata do not set applicability.

The A2 mini intentionally exercises mechanics on its A1-evaluable ROI cells,
without claiming formal GO or using the scope gate to suppress the engineering
fixture. Formal `_run_a2` requires complete global GO and applies the established
type scope. Mini output remains under `mini/a_core`, outside every formal node.

## Recovery and resource behavior

A1 commits each input chunk with source/config-specific dependencies, sparse
integer increments, and exact molecule IDs/folds. Recovery reconstructs totals
from each range exactly once; it never resumes a floating accumulator blindly.
A2 saves the sorted proposal cursor, decision array, and complete current mask
as a new generation before atomically publishing its state. A crash before
publication discards the uncommitted generation; accepted operations never run
twice against an authoritative mask. No stochastic decision is made in A2.
Primary statistics/support reports are deterministic stages and may be recomputed
when interrupted; they cannot create a completed node early. Formal runtime owns
the resource lock; the standalone mini acquires the same shared lock once.

## Limitations

Rep1 remains a development specimen. Historical type labels may use all nuclear
expression; only Core profile counts and heldout molecules are split here.
Nuclear heldout diagnosis does not establish real outer owner correctness.
Complete-nucleus support is not the decoder's per-view visibility. Confidence
intervals condition on this molecular split and 20 donor permutations. They are
not independent-tissue replication or full-randomization p values. Sparse Core
coverage and nuclear/cytoplasmic shifts remain explicit. A2 fixes capture and
cannot recover background RNA. The coordinate mini is an explicitly translated
native-grid fixture; its scale=1 CSV is not silently substituted into formal data.

## Validation status

Initial implementation passed eight focused tests; final focused suite passes
12 tests. Real validation and measured planning evidence are recorded below.
No formal science node has been run by this agent.

Pre-admission memory refinement: A2 candidate scores are allocated directly as a
36-byte structured record per candidate edge, avoiding a Python tuple/object
copy of the complete score table. Full support has 77,913,750 native memberships
and maximum K=14 in the verified frozen input. Query pixels are expression-bearing
subsets, so this is an upper bound, not a measured new-result population.

Support CSR, complete geometry and sparse-expression diagnostics emit progress
and honor stop requests at bounded nuclear-group boundaries. These subphases do
**not** save partial CSR/geometry/score accumulators; after interruption, their
current deterministic subphase is rebuilt from its inputs. Previously committed
molecule chunks are reused exactly. The restart cost can include one complete
support/geometry/diagnostics pass and must be included in conservative budgeting.
No progress counter is advertised as a persisted scientific cursor for these
subphases. Bootstrap restarts from its fixed RNG seed if interrupted before its
artifact is published.

Additional tests now compare complete/elongated support against the actual frozen
`Evidence.support` implementation, constructing only its nuclear fields and never
running its DAPI initializer. Complete nuclear distances are checked against a
brute-force full-nucleus oracle for a nucleus longer than a decoder view. Vector
winner reduction is tested for empty support and exact ties; recovery rejects
changed dependencies. Focused suite: 10 passed.

Mini CSV qv=20 is an explicit eligibility token for molecules already selected by
the original processed-input contract; the original qv values are not a mini
score feature. Original transcript IDs, genes, integer native coordinates and
row order are preserved before local coordinate translation.


## Real validation evidence (2026-10-15)

All paths below are relative to the isolated output root
`/home/zwan0726/work/outputs/bidcell_boundary_three_20261015/`.

- `mini/a_core/mini_receipt.json`: **passed**, on source commit `5216b32`.
  The original receipt is preserved. Its source manifest records actual executed
  source; the draft config's null source/input digest is not represented as a
  final frozen-source receipt.
- The real 384×384 native-grid ROI contains 693 complete nuclei and complete
  baseline cells, 125,283 original eligible molecules and all 313 spatial genes.
  Mini coordinates were mapped in the original native coordinate system before
  integer translation. The A1 mini has 555 paired cells in one 1000µm block;
  its unchanged gate correctly returns **INCONCLUSIVE**. A2 mini execution is an
  engineering exercise, not a GO claim or an exception to the formal gate.
- Exact Core chunk stop/resume matched uninterrupted integer counts. An actual
  A2 interruption occurred, and recovery matched final masks, sorted proposals
  and all candidate scores. The independent complete-image topology oracle
  checked every proposed transfer in all three arms. Accepted transfers were
  GT=239, GTC=308 and SHUFFLED=293; these counts document exercised mechanics.
- Four actual mini endpoints each built its own 693×313 integer matrix and ran
  the full evaluator. Every arm assigned exactly 60,682 of 125,283 eligible
  molecules. The ROI native endpoint is explicitly `A2_MINI_NATIVE`; formal
  `A2_NATIVE` remains a validated historical reuse.
- Measured mini wall=675.49s, including compute=127.89s and serialized resource
  wait=547.60s. Python peak RSS=854,769,664 bytes. This is **not** a measurement of
  aggregate Python plus child-R peak memory. Four fresh full evaluator runs took
  25.52s, 24.57s, 24.49s and 24.59s. A1 execution took 5.23s; uninterrupted A2
  mask/domain/topology mechanics took 3.28s. Total retained mini data was 68.93MB
  before the final input-only probe and validation evidence were added.
- The ROI's observed maximum K=13 is not relabeled as global K=14. Complete-cell
  retention removed one contributing candidate. A separate bounded original-input
  probe, `mini/a_core/maximum_support/receipt.json`, queried the frozen global
  maximum using **all 107,955 complete nuclei**, recovered exactly 14 candidates,
  and matched each geometry value to brute distance over the entire nucleus.
  It computed no expression efficacy score, loaded no DAPI, took 73.77s compute
  (61.35s complete support scan), and used peak Python RSS=854,712,320 bytes.
- `mini/a_core/validation_after_mini/source_binding.json`, its executable
  `verify_binding.py`, and `source_delta.patch` bind the passed mini's unchanged
  calculations to final source. Exact AST comparison passes after explicitly
  removing only progress callbacks/arguments and an unused chunk-index rename.
  The mini-used A1 validation branch is unchanged. New formal A2 comparison
  completion validation and the parent report-path fix are separately tested;
  the old mini receipt is not relabeled as executing those additions.

The final comparison validator requires exactly GTC–GT, GTC–NATIVE and
GTC–SHUFFLED, binds each report under its candidate endpoint directory, checks
its receipt identity, calls the full shared comparison validator, checks the
population/contrast names and requires exact agreement with the persisted
scientific decision. Missing comparisons, receipt corruption and altered
summary values fail bounded unit tests.

Additional stop/progress callbacks now cover every secondary molecule scan,
A2 domain/score generation and A2 geometry. They do not persist partial state:
those deterministic stages restart. Tests verify callback/no-callback equality,
actual bounded interruption and exact recomputation. Proposal application retains
its existing committed cursor, decisions and full-mask snapshot every 1,000
proposals. The calculation/recovery format and all scientific gates are unchanged.

## Conservative full-node resource forecast

Machine-readable inputs, formulas, per-phase timings and limitations are in
`mini/a_core/resource_forecast.json`. These are planning estimates, **not measured
formal runtimes**. The mini/full ratios are 155.78× cells, 274.92× molecules and
430.10× the A1 worst support-edge population. Full input support memberships
are bounded by 77,913,750. The input-only A2 bound is 3,001,399 possible band
pixels and 10,388,161 support edges before type/donor/adjacency restrictions;
three 36-byte score tables therefore require at most 1,121,921,388 bytes.

| Node | Nominal measured scaling | Conservative admission allowance | Incremental retained/transient disk | RAM allowance |
|---|---:|---:|---:|---:|
| A1 | 1,271s total over all measured phases | 3,600s, including hashes/I/O and one deterministic diagnostic restart | 6GiB | 20GiB analytical Python allowance |
| Conditional A2 | 20,476s for all mask mechanics using the largest candidate-edge ratio | 85,352s (23.71h) over all phases below | 12GiB | 20GiB analytical aggregate allowance; child-R peak unmeasured |

The conservative A2 allowance explicitly includes complete-support/domain/
geometry scoring, all three proposal/topology passes, all three own matrices,
three fresh full R evaluations, validated native reuse, three complete
comparisons at 1,800s each, one deterministic pre-proposal restart and I/O.
The parent measured an actual shared complete report at 212,511,669 bytes;
its observed 147s file-timestamp span is only a lower-bound inference, so it is
not used as a whole-run timing measurement. Historical per-arm matrix and
full-evaluation sizes are 479,485,398 and 191,060,052 bytes, respectively.

Checkpoint traffic is material: a native uint32 mask is 164,909,840 bytes.
The all-domain upper bound permits 9,006 snapshot batches across three arms,
up to **1.485TB cumulative mask writes plus equal SHA readback**. The forecast
uses an explicitly assumed combined 100MiB/s I/O rate; no bandwidth guarantee
is claimed. Obsolete generations are deleted after atomic publication, so
peak retained/live mask snapshots are approximately 660MB, not terabytes.
The mask-scaling allowance also includes some mini checkpoint overhead; adding
full traffic separately deliberately errs conservatively.

Actual A2 may be far smaller, or legally zero-action, after unchanged type/donor/
adjacency restrictions. The 23.71h envelope is not a prediction that actual A2
will take a day. It prevents a false guarantee that the whole chain fits the
plan's initial eight-hour budget. Budget deferral remains distinct from a legal
scientific negative and an engineering failure. The runtime must account for
current stage completion, resources and reserve before formal admission.

## Local implementation commits and remaining audit

- `0cde5d2`: complete A1 and conditional four-arm A2 implementation, tests and
  plan mapping; parent integrated as `94e4945`.
- `5216b32`: bounded structured score allocations, support/geometry restart
  disclosure and expanded frozen-support tests; source used by the real mini.
- Final follow-up commit adds strict comparison completion validation, bounded
  progress callbacks, the input-only global maximum probe and final evidence.

No A1/A2 formal outcome, efficacy ranking or adopted method is reported here.
Independent review must adjudicate the explicit type-scope clarification,
conditional A2 budget envelope and source correspondence before formal admission.
