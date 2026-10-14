# P2 transfer implementation

Implementer: explicitly requested `gpt-6-astra`, `xhigh`; this is implementation
evidence, not an independent audit. Branch `feature/boundary-three-p2` starts at
`64a18abcd76f30b26aeaf98502701464ef054a17`. Shared parent-owned contracts,
evaluation, freeze and reports are integrated through their separate commits.

## Plan to code

| Frozen requirement | Implementation |
| --- | --- |
| BASE_MATCH and old full V2_MORPH, immutable historical inputs | `p2_transfer.BASES`, `prepare_node`; shared endpoint registry. Current paired BASE and old new-methods Morph are explicit. |
| Actual disabled native replay before P2 | `p2_native.native_connect` calls current native `postprocess_connect`; `run_node` verifies full historical final pixels, counts, ordered gene/ID mappings, CSV and every transcript-owner chunk before P2. |
| Unchanged current-native proposals and RNG | Observational `NativeWrites` captures final proposal writes from actual native; P2 checks every candidate hash, proposal histogram, candidate order and entry RNG against replay. |
| Frozen complete-nucleus int64 arbitration | `p2_rules.py`, transplanted from the read-only historical BS4 rule; no distance normalization, owner protection, candidate expansion or alternate rule exposed by formal nodes. |
| Cross-crop/multicomponent nuclei, exact ties, missing IDs | `FullNucleusIndex`; complete global raster, bounded query pairs, exact integer distances and smallest global ID tie. |
| Native ordered tiles, islands, splits and nuclear restoration | `p2_native.process`, native `remove_islands` and unchanged `split_patch`; diagnostics never feed canonical masks back into the algorithm. |
| Interrupted recovery | Dependency/hash-bound tile receipts and independent split chunks; reconstruct original ordered merge on resume, then apply manifest-listed split chunks. Bounded submissions and explicit stop boundaries. |
| Canonical stage diagnostics and molecular denominators | `p2_diagnostics.transfer_diagnostics`; overlap overwrite yields unique pixels, exact matrix coordinate records provide molecule multiplicity. Pre, canonical connect, ordered islands merge, cleanup, pre-global-restore and final persist. |
| Competition, type, density, RNA, nuclear shape | Full proposal composition plus winner/second distance/tie per canonical conflict pixel; frozen native-owner type, density and BASE_MATCH RNA groups; descriptive nuclear eccentricity bins. Stage transfers distinguish cell/cell, cell/background and background/cell. |
| Own final matrix and full evaluator | `run_node` calls parent `evaluation.build_and_evaluate`; no historical matrix/annotation copied into new P2 endpoint. Replay separately rebuilds its matrix as an equality gate. |
| Two primary comparisons and four cells | Parent `reports.compare_endpoints` for each node; `factorial_report` only after both complete new P2 endpoints. P2_BASE negative does not gate P2_MORPH. |
| Entry points and receipts | `run_node`, `validate_node`, `validate_mini`; common outer completion only after replay, own evaluation, transfer diagnostics and complete comparisons. |

## Compatibility decision

The current native `postprocess_connect` enumerates `np.unique(img)[1:]`, while
the historical BS4 transplantation used `ids[ids > 0]`. Those differ on a tile
with no zero background. The adapter preserves **current native** enumeration,
including dropping its first positive ID in that case. The single corresponding
line in the copied rule is explicitly documented and tested. Candidate geometry
and P2 arbitration otherwise retain the frozen implementation. This avoids
silently fixing native behavior inside a P2 intervention.

The native observer temporarily substitutes a NumPy proxy only inside each
single-threaded worker. It delegates all operations and records the already
computed boolean final write masks. No input, candidate, RNG operation or return
value is altered. Uninstrumented-current-native tests verify this observation.

## Validation

`tests/boundary_three/test_p2.py` executes 24 independent exhaustive all-global-
nuclear-coordinate oracle cases, native output/proposal/RNG checks, the
no-background edge case, complete multicomponent distances, exact and near ties,
missing IDs, bounded query memory, and native tile/merge order.

Real evidence is under
`/home/zwan0726/work/outputs/bidcell_boundary_three_20261015/mini/p2`:

- `attempt_01/complete.json`: both real chains passed in **1201.48 seconds**.
- Each chain replayed production tile `(0,2048)`, size 1024, using its exact
  historical seed. Native postislands pixels matched the saved historical tile;
  a separate uninstrumented native call also matched the candidate output and
  RNG. Observed candidate counts were 3,346 (BASE) and 3,342 (Morph).
- Both actual 144×144 ROI pre masks came from their respective formal historical
  pre rasters. The 18,469 molecule IDs, genes and quantized global coordinates
  were independently checked against full production coordinate records before
  exact local rebasing. No fresh prediction/training was needed for P2.
- Native ROI final masks and all integer counts, ID/gene maps, CSV and transcript
  owners matched separately run current native pipelines. Each new P2 final
  then built its own matrix and passed all authoritative evaluator profiles.
  BASE had 63 matrix IDs / 58 own-QC IDs; Morph had 62 / 56.
- BASE exercised 1,295 conflict pixels, 48 exact ties and 611 postconnect owner
  changes. Morph exercised 1,389 conflicts, 57 ties and 591 changes. These are
  development coverage diagnostics, not formal efficacy results.
- Both bases were interrupted after a tile boundary and after a split chunk on
  a real overlapping-tile ROI. All five stage TIFF SHA256 values matched the
  uninterrupted run exactly in both recovery cases.
- Peak measured parent RSS was 1,183,498,240 bytes; peak child RSS was
  1,181,556,736 bytes. These are process maxima, not concurrent process-tree
  sums or a measured full-slide peak.
- `candidate_inventory.json` inspects all 48 production pre tiles per base:
  maximum K is **5,201**, at `(4096,0)`, for both. The production tile executed
  by mini is representative, not that maximum-K tile. P2's query allocation
  remains explicitly bounded at 262,144 pairs independent of candidate count.
- `unit_tests.log`: five tests passed, including 24 exhaustive full-coordinate
  cases, an ineligible nearby nucleus, a strict squared-distance difference of
  one at 10¹⁰, and a real stop file that prevents node preparation/completion.
- `final_validation.json` revalidates real artifact hashes, own evaluation and
  every recovery stage using final code. The expensive mini's exact producer
  commit is **`a4ff61c`**. A subsequent change adds only formal-run stop guards;
  AST equality verifies that all transfer helpers used by the mini stayed
  identical, while its numerical producer files retain their exact hashes.
  The control-only difference is explicit rather than represented as a rerun
  of a full formal node.

The frozen transplant source SHA256 is
`779e134aa2b665f38ba05488e42c49a8ab17ea7a87353aa2ee36ce040b2a822e`.
Apart from its explanatory header, the sole copied-rule diff is the native ID
enumeration described above.

`resource_estimate.json` records **17,130 seconds for P2_BASE** and
**18,890 seconds for P2_MORPH**, before the queue's separate 1.25 contingency and
30-minute package reserve. Components include historical full native replay
time (~3,386 seconds), conservative P2 cost at 2.5× native, two validated own
matrix builds, full evaluator, stage diagnostics, comparisons, hashing/I/O and
the second node's four-cell report. Disk estimates are 14 / 16 GiB respectively.
The P2 multiplier and full-slide peak remain estimates; the mini does not claim
to have measured a formal full P2 node or maximum-K P2 tile.

A mini passing does not omit full native final/matrix replay at each formal
node. `validate_mini` expects a valid config whose `output_root` is its isolated
mini output directory. Formal `run_node` acquires no compute lock recursively;
the queue owns that lock. It raises `InterruptedError` at resumable tile/split
or between-stage boundaries, leaving the queue to distinguish a user stop,
budget pause or resource pause.

The final local commit SHA and integration list are recorded in
`mini/p2/P2_handoff.json`; the separately committed shared parent files should
not be cherry-picked again. Final artifact validation command (already run):

```bash
PYTHONPATH=/home/zwan0726/work/projects/BIDCell_boundary_three_p2_20261015 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/zwan0726/miniforge3/envs/bidcell_local/bin/python /home/zwan0726/work/projects/BIDCell_boundary_three_p2_20261015/tests/boundary_three/validate_p2_receipts.py --root /home/zwan0726/work/outputs/bidcell_boundary_three_20261015/mini/p2/attempt_01 --output /home/zwan0726/work/outputs/bidcell_boundary_three_20261015/mini/p2/final_validation.json
```

## Scope and limitations

No formal experiment is launched by this implementation agent. Mini endpoints
are development-only. The P2 metric result is neither owner truth nor evidence
that the network learned boundary features. Nearest nucleus cannot compensate
for a missing proposal. A legal zero intervention remains legal and retains its
full evaluation. Foreground equality is asserted after postconnect only; final
capture may change through islands and merge.

Nuclear shape bins (`eccentricity <= .5`, `(.5,.8]`, `>.8`) are descriptive,
outcome-independent diagnostics; they do not change the algorithm or add a
scientific selection threshold. All principal frozen strata/gates and spatial
uncertainty belong to the shared comparison implementation.
