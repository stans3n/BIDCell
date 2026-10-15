#!/bin/bash
set -euo pipefail
exec /home/zwan0726/miniforge3/envs/bidcell_local/bin/python -I -S -B "$(dirname -- "$0")/bootstrap.py" "$@"
