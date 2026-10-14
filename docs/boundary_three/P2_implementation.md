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

Real mini receipts, timings, recovery evidence, resource estimates and final
local commit are recorded below when the run finishes. A mini passing is not
permission to omit the full native final/matrix replay at each formal node.

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
