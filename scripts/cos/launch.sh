#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 2 || "$1" != "--config" ]]; then
  echo 'usage: scripts/cos/launch.sh --config /absolute/path/configs/queue.json' >&2
  exit 2
fi
exec /usr/bin/python3 - "$2" <<'PY'
import json,os,subprocess,sys
from pathlib import Path
config=Path(sys.argv[1]).resolve();cfg=json.loads(config.read_text());cwd=cfg['methods']['A']['worktree']
os.chdir(cwd);os.environ['PYTHONPATH']=cwd
os.execv(cfg['python'],[cfg['python'],'-m','scripts.cos.launch','--config',str(config)])
PY
