# Boundary-three integration contract (2026-10-15)

Scientific definitions are frozen by the four plans in /home/zwan0726/work/docs.
The parent owns contracts.py, common.py, freeze.py, evaluation.py and reports.py.
A_CORE owns a_core.py, core_scores.py, core_refine.py and its tests.
P2 owns p2_transfer.py, p2_rules.py/native adapters and its tests.
Morph owns morph.py, morphology_variants.py, variant data/training/prediction adapters and its tests.
Automation, after code audit, owns runtime.py, cli.py, control.sh and execution tests.
Do not change another owner's files without coordinating.

All scientific modules expose run_node(config, node), validate_node(config, node),
and validate_mini(config, output). config is either a run.json path or its loaded dict.
run_node writes a common schema node complete receipt only after every necessary
artifact and diagnostic passes. validate_node performs semantic checks plus receipt
hash checks. validate_mini returns and persists real evidence including runtime/memory,
input paths, maximum K and interruption/recovery comparison. Conditional A2 must be
fully implemented and admitted before A1 formal results; no placeholder functions.
A1/A2 entry is a_core, both P2 nodes are p2_transfer, all three Morph nodes are morph.
Preparation helpers may be exposed by each module; orchestration must call these
before freezing source/config. Functions must not mutate run config during execution.

Use common.load_run/output_path/endpoint_spec. Node dirs are output_root/NODE;
A2 endpoints are output_root/A2_CORE/endpoints/A2_NAME. Historical endpoint registry
is explicit: BASE_MATCH from paired outputs, V2_MORPH from old new-methods outputs.
New masks use endpoint_spec paths, own integer matrix and full evaluator; parent
provides build_and_evaluate(cfg, endpoint_id) in evaluation.py. A2_NATIVE is verified
historical reuse, not manufactured new prediction. A1 is not a mask/F1 endpoint.

Config keys: version, run_id, source_root, output_root, python, rscript, node_order,
endpoints, inputs, enabled_nodes, seeds, resources, budget_seconds, node_estimates,
source_digest, input_digest. inputs include historical_input_root, shared_initial,
views, predict_shift0, predict_shift24, reference_region, evaluation_queue,
authoritative_protocol, frozen_strata, frozen_counts, pairing_audit.
All paths are absolute. Historical sources and results are read-only. Do not call
legacy prepare/launch, whose hardcoded output roots could overwrite history.

Use common.progress after meaningful resumable work and check stop_requested at
safe boundaries. Pause must preserve exact RNG/cursor/model/BN/optimizer or exact
processed ranges/proposal state. An incomplete node never creates complete.json.
No automatic NaN/OOM/integrity retry, no AI dependence. Use common.compute_lock for
real mini GPU/large CPU work. Formal runtime owns the workspace GPU lock; do not
acquire it recursively in run_node. Parallel small unit tests are allowed.

Receipt dependencies bind config/source/input digests. Nested native receipts may
retain their original validated schema; the outer node receipt is common v1.
Actual configs and all output hashes must be verified; semantic checks cannot be
replaced with a nonempty file check. Record mapping, limitations, deviations,
mini receipts and local commit SHA in docs/boundary_three/{ROLE}_implementation.md.
