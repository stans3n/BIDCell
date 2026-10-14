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

Focused unit tests: 8 passed at initial implementation. Actual mini receipt,
measured full-node/disk estimate and final local commit will be appended after
real validation finishes. No formal science node has been run by this agent.

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
