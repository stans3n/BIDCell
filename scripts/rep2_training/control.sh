#!/usr/bin/env bash
set -euo pipefail
RUN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON=/home/zwan0726/miniforge3/envs/bidcell_local/bin/python
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$RUN_ROOT/source${PYTHONPATH:+:$PYTHONPATH}"
export NUMBA_CACHE_DIR="$RUN_ROOT/runtime_cache/numba"
export MPLCONFIGDIR="$RUN_ROOT/runtime_cache/matplotlib"
export XDG_CACHE_HOME="$RUN_ROOT/runtime_cache/xdg"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$RUN_ROOT/source"
exec "$PYTHON" -B -m scripts.rep2_training.runner "${@:-status}"
