#!/usr/bin/env bash
set -euo pipefail
CHECKOUT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=/home/zwan0726/miniforge3/envs/bidcell_local/bin/python
CONFIG="${1:?Usage: launch_stage1.sh /absolute/run.json}"
export PYTHONPATH="$CHECKOUT"
ROOT="$($PYTHON -c 'from scripts.crgf.config import read; import sys; print(read(sys.argv[1]).output_root)' "$CONFIG")"
RUN_ID="$($PYTHON -c 'from scripts.crgf.config import read; import sys; print(read(sys.argv[1]).run_id)' "$CONFIG")"
SESSION="crgf-${RUN_ID//[^a-zA-Z0-9_-]/_}"
mkdir -p "$ROOT/logs"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  printf 'Session already exists: %s\n' "$SESSION" >&2
  exit 1
fi
# Shell printf %q preserves absolute argv safely, including unusual path characters.
printf -v COMMAND 'cd %q && exec %q -u -m scripts.crgf.run_stage1 --config %q >> %q 2>&1' "$CHECKOUT" "$PYTHON" "$CONFIG" "$ROOT/logs/pipeline.log"
OLD_ID=""
if [[ -f "$ROOT/runner_identity.json" ]]; then OLD_ID="$(cat "$ROOT/runner_identity.json")"; fi
tmux new-session -d -s "$SESSION" "$COMMAND"
OK=0
for ((i=0;i<30;i++)); do
  if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    printf 'Queue exited before startup; inspect %s\n' "$ROOT/logs/pipeline.log" >&2
    exit 1
  fi
  if [[ -f "$ROOT/runner_identity.json" ]] && [[ "$(cat "$ROOT/runner_identity.json")" != "$OLD_ID" ]]; then
    if "$PYTHON" -c 'import json,sys; from pathlib import Path; from scripts.crgf.run_stage1 import is_alive; sys.exit(0 if is_alive(json.loads(Path(sys.argv[1]).read_text())) else 1)' "$ROOT/runner_identity.json"; then OK=1; break; fi
  fi
  sleep 1
done
if [[ "$OK" != 1 ]]; then printf 'No live new runner identity confirmed\n' >&2; exit 1; fi
printf 'run_id=%s\ntmux_session=%s\noutput_root=%s\n' "$RUN_ID" "$SESSION" "$ROOT"
cat "$ROOT/runner_identity.json"
