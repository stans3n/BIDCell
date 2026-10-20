#!/usr/bin/env bash
set -euo pipefail
REP2_SOURCE="$(cd "$(dirname "$0")/../.." && pwd)"
REP2_ROOT="$(dirname "$REP2_SOURCE")"
export PYTHONPATH="$REP2_SOURCE"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NUMBA_CACHE_DIR="$REP2_ROOT/tmp/numba" MPLCONFIGDIR="$REP2_ROOT/tmp/matplotlib" XDG_CACHE_HOME="$REP2_ROOT/tmp/cache" TMPDIR="$REP2_ROOT/tmp"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$REP2_SOURCE"
exec /home/zwan0726/miniforge3/envs/bidcell_local/bin/python -B -m scripts.rep2_full.runner "$@"
