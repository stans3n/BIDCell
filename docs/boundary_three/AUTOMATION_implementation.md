# Serial runtime implementation

Author role: user-requested automation implementer, gpt-6.1-sol / high. This is
implementation evidence, not independent review or formal launch authorization.
Scientific source was independently admitted for automation preparation at
42f2d95369aec03f627272f45bdd2804d4702396. The parent integrates subsequent
scientific fixes and obtains a new independent final script/integrated audit.

## Plan mapping

| Requirement | Implementation |
|---|---|
| Fixed full-node order, A1 negative continues, P2_BASE negative does not cancel P2_MORPH | runtime.run_queue uses contracts.NODES; only validated A1 decision controls A2 |
| A2 pre-admission + GO + whole-chain remaining budget | verify_admission requires exact frozen enabled list; run_queue validates A1 and checks estimate×1.25+1800s |
| Minimum three budget | budget_forecast refuses initial window below conservative three-node sum×1.25+1800s |
| Actual unchanged science | node_worker calls actual run_node and validate_node; never tunes constants or rewrites run.json |
| Preparation after committed final source | prepare verifies clean integrated checkout; prepares exact Morph step zero before input/source/config freeze |
| Complete freeze and final audit | provenance.capture_inputs; source_manifest; frozen domains; final reviewer admission + actual environment identity |
| Strict serialization and orphan protection | queue flock + workspace compute flock once/node; worker inherits both descriptions; closing supervisor descriptor retains child lock |
| Internal entrypoint protection | _node requires active lease/current node, direct live supervisor parent and inherited descriptor inode identity; _run requires STARTING plus exclusive one-time lease claim |
| Durable state and observation | atomic queue_state, result/checkpoint receipts, fsynced events/logs, PID+boot+start ticks; real progress file separate from heartbeat |
| Detach/duplicate protection | unique actual tmux session; launch guard + queue lock + bound process checks; actual host fixture exercises duplicate rejection |
| Stop/failure distinction | SafeStop RuntimeError and InterruptedError are controlled pauses; other exceptions remain runtime/integrity failures |
| Finite resource wait and narrow retry | waits <=600s; exactly one transient EIO/ESTALE/ETIMEDOUT retry only with content-verified existing recovery units; worker revalidates scientific cursor/dependencies |
| Budget/resource pause | stop.json requests committed scientific safe boundary, never kills active scientific writes or truncates frozen training |
| Resume | new execution lease with explicit budget; source/config/input stay immutable; validated completed nodes and scientific skips do not rerun; time deferrals eligible again |
| Disk admission | all time-eligible remaining retained outputs + shared cache + maximum atomic extra +10GiB; only provably whole-window time-ineligible tail excluded, with machine-readable derivation |
| Reporting | package aggregates actual receipted tables with distinct Purity cohort/scaling identities, actual boundary categories and decisions; A1 stays information-only; missing metrics remain NA |
| Completion | complete-check runs each actual node's semantic and content validation and distinguishes minimum/all-applicable/deferred/failed/not-enabled |

## Interfaces

Use the integrated source's absolute control.sh path. Every operational command
takes `--config /absolute/runroot/configs/run.json`. Prepare takes
`--template /absolute/template.json --output-root /absolute/new-empty-runroot
--run-id IDENTIFIER`. Validate takes `--admission /absolute/final-review.json`.
Other commands: launch; status; status --json; tail [--node ID] [--lines N]
[--follow]; resume [--budget-hours N]; complete-check; safe-stop [--reason TEXT].
Resume is a background experiment queue operation, distinct from Codex goal
resume. It creates a fresh bounded execution lease without modifying any science
configuration or completed receipt. Known failed nodes require a new reviewed
run; no silent retry of OOM, NaN, drift, or changed algorithms.

Final reviewer admission schema is `boundary-three-launch-admission-v1`, with
status ADMITTED, approved_for_formal_launch=true, source_commit, source_digest,
config_digest, input_digest, exact ordered admitted_nodes, reviewer containing
model gpt-6.1-sol, reasoning_effort xhigh and implementation_author=false,
nonempty evidence identity list, environment identity pointing to this run's
admission/environment.json, and node_admissions mapping each enabled node to
status ADMITTED and its actual mini/resource evidence identity list. Runtime
checks the independent file; implementation does not manufacture approval.

The parent supplies evidence-backed full-chain estimates for every enabled node:
seconds, retained_bytes, atomic_extra_bytes, peak_ram_bytes, peak_gpu_bytes,
evidence list. Resource defaults remain >=1GiB GPU reserve, >=6GiB host reserve,
>=10GiB disk reserve, <=4 workers and one BLAS thread. Available RAM before a
node must cover its whole-node peak plus reserve; while running, reserve alone
is monitored to avoid counting the already allocated working set twice.

## Verification evidence

Focused tests in tests/boundary_three/test_runtime.py cover budget arithmetic,
whole-window disk exclusion/resume expansion, PID reuse/boot identity, real
inherited flock survival, duplicate lock rejection, unauthorized internal worker,
fixture/null-freeze refusal, live-without-progress and incomplete receipts,
controlled SafeStop versus runtime failures, immutable config across stop and
progress, negative gates, GO with insufficient A2 time, failure halting the queue,
same-lease replay refusal, semantic completion and preserved comparison scaling.

Actual host engineering tmux fixture is implemented only in tests, never a
configurable production backend. Its own marker and custom interpreter are
rejected by formal run validation. Real lifecycle evidence resides under
/home/zwan0726/work/outputs/bidcell_boundary_three_20261015/mini/automation/.
The final lifecycle_receipt.json records exact source SHA/hash and all actual
commands, host, detached session/process identities via states/events, cursor
interruption, status JSON/human, tail, duplicate refusal, resume and no replay.

Scientific real minis are the existing independently reviewed A_CORE/P2/Morph
evidence; the runtime fixture adds engineering validation, not scientific scores,
GPU resource admission, or a full-slide evaluator. No formal queue was launched
by this implementation role. Production preparation/final source-input freeze,
R replay/maxK admission, final independent review and formal launch are parent
tasks. Actual tail completion depends on frozen costs and measured elapsed time.
