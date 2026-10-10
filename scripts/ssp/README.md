# SSP-OVERNIGHT-v1.0 execution

All writes belong to the isolated SSP checkout and run root. A/B/C and their
accepted inputs/outputs remain read-only. Both new arms independently initialize
from the frozen B1 checkpoint; no stage consumes S weights for SP.

`prepare_and_launch.sh COMMAND RUN_ROOT` is the absolute-path-safe entrypoint.
Run process observation and launch/resume commands in the host PID namespace.
A subsequent AI must use host execution (`require_escalated`); an invisible host
PID in a sandbox is unknown and must never be treated as a dead runner.

- `prepare`: idempotent manifest bootstrap, full-nucleus prototypes, frozen teacher inventory.
- `admission`: execute missing real method/runner/mini tests and collect required evidence.
- `freeze`: finalize commit bindings and freeze every source/input hash before independent approval.
- `export`: export a complete committed Git bundle and source map (including all new files).
- `launch` / `resume`: deploy reviewed code; for a fresh run bootstrap and prepare data, execute missing admission, freeze inputs/source and collect evidence. Stop at the independent approval gate if its receipt is missing. For an admitted run only verify hashes/receipts and current resources, then detach the serial S→SP queue. Existing complete computation refuses restart; a running process or held lock refuses duplicate launch.
- `status`, `process`, `tail`: read actual machine status, identity, and logs.
- `evaluate --arm S|SP`: validate an existing completed evaluation or execute the missing evaluation under run/global locks; refuses concurrent runner/child execution.
- `complete [--arm S|SP]`: inspect validated full-run or arm evaluation receipts.
- `deploy --deployment MANIFEST`: safely clone the reviewed bundle into an absent recorded target, or verify an identical existing deployment.

Independent `tests/independent_review/approval.json` must bind final
`source/frozen.json`, queue and every required evidence file, including
`tests/admission.json`. Scripts never author the independent approval.

A new source revision after a failure requires explicit affected-stage review.
Strict resume never silently accepts a changed source/input/recipe or overwrites
failed attempts. Only whitelisted I/O/interruption failures retry, at most twice;
NaN, identity, mathematics and conservation failures stop that arm. SP is
independent and may continue after an S failure. An eight-hour target miss updates
status and ETA without truncating either arm or evaluation.

`python -m scripts.ssp.handoff --root RUN_ROOT --observe` records real progress
and held locks. Two observations at least 20 seconds apart with increasing
updates, readable update100 full checkpoint, eight probes, and a real duplicate
launch refusal are required before `handoff --root RUN_ROOT` writes the handoff.
Only the controlling AI can conditionally pause its goal, after independent
review and all AI workers finish. The compute runner never calls an AI or goal
tool and terminates at `compute_complete_awaiting_user_resume`.
