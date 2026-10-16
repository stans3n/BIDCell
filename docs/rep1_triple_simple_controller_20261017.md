# Rep1 independent compute controller

The independent science audit passed commit790910b. The root dialogue is the
sole controller author. This document describes implementation, not admission
or formal compute success. The same independent auditor must review the final
source, actual host fixtures, frozen input/domain/environment/configuration and
resource gates before launch.

`control.sh` invokes the fixed Python with `-I -S -B`. `bootstrap.py` uses only
snapshot, stdlib and the fixed purelib, disables bytecode and installs the
content-checked project import guard. Source copies are independent read-only
files; exact inventories reject additions, links, writable files and drift.
External author R sources are copied. The explicit per-session tmux environment
overrides old server R/Python/loader/cache settings, with work-storage TMPDIR.

Live checkout permits only `prepare --output-root ... --run-id ...
--plans-manifest ...`. It first copies committed code, then a private snapshot
process freezes input and outcome-independent domains under three inherited
locks. Config, environment and full-content preflight follow in one direction.
The independent final admission binds them without embedding its future hash
in the config. Mutation fingerprints include inode, mode, size, mtime and ctime;
the worker also performs the scientific full-input hashes at each stage.

`launch --config ...` dispatches tmux and returns. It releases the dispatcher
gate immediately; supervisor then owns launch/run/workspace locks and passes
the same OFDs to workers, R and fork descendants. There is no LOCK_UN. An
unresolved prior dispatch refuses a fresh launch. Explicit `resume-compute
--config ... --attempt-reason ...` requires prior identities exited, all locks
free and the same admitted immutable source/config. Code repair needs a newly
reviewed snapshot/run. There is no automatic retry or inactivity termination.

The supervisor runs01–14 serially. A scientific witness is validated before
state progress advances, deduplicated against existing path+SHA; reused consumer
wrappers do not count. Mutable cursor SHA can advance at the same path. Every
10s heartbeat is separate. R batches bind the actual production/dependency
digest. Queue and per-stage logs retain actual stderr and errors. Scientific
negative/no-op/guardrail values never control stage continuation.

After14 complete semantic verification, supervisor commits15 and exits. A
separate entry waiter records the actual OS return code. `complete-check
--phase compute` requires all15 semantic receipts, observed OS exit0, no live
or unobservable entry/supervisor/worker, and all three locks free. It calculates
no missing work. `status --json` and `tail --lines80` are strictly read-only;
stale heartbeats and >900s scientific inactivity are observations only.

The provisional disk decomposition is GT4.5+P2/replay5.5+analysis15+shared2+
temporary4+reserve15=46GiB. The analysis measurement14.178GiB plus explicit
SCI01 metadata64MiB rounds to15GiB. These are estimates with retained output
growth checks, not new-parent absolute bounds. Allocated bytes deduplicate
hard links by device/inode; temporary/failed files never reduce retained
forecasts. A cached <=10s remaining forecast is conservative. RAM peak22GiB
includes the current owned tree, with6GiB available reserve and actual PSS.
All large temporaries use work storage. Unsafe resource/identity/engineering
errors fail the queue and preserve evidence; they do not authorize scope cuts.

`publication-check` first requires the genuine future manual receipt and never
publishes. This compute CLI has no manual-grant, result-analysis, publication,
context-sync, AI call, scheduling, or automatic wakeup implementation. Those
remain explicit post-user-resume root work.

Targeted controller tests are `tests.test_rep1_triple_controller`. The actual
host harness is `tests/rep1_triple_controller_fixture.py suite OUTPUT`, with
OUTPUT containing `controller-fixtures`. It patches science only inside the
isolated test entry and cannot pass the formal loader. It exercises real tmux,
runtime, R wrapper and fork locks; its synthetic numeric data is not a
biological result. Final measured fixture evidence is recorded outside the
repository and bound by the final auditor, rather than asserted here in advance.
