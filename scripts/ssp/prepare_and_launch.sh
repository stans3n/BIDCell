#!/usr/bin/env bash
set -euo pipefail
SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${SSP_PYTHON:-/home/zwan0726/miniforge3/envs/bidcell_local/bin/python}"
export PYTHONPATH="$SOURCE_DIR${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" -m scripts.ssp.entry "${1:-launch}" --root "${2:?usage: prepare_and_launch.sh COMMAND RUN_ROOT}" "${@:3}"
