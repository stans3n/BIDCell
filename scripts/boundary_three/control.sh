#!/usr/bin/env bash
set -euo pipefail
SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$SOURCE_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$SOURCE_DIR${PYTHONPATH:+:$PYTHONPATH}"
exec /home/zwan0726/miniforge3/envs/bidcell_local/bin/python -m scripts.boundary_three.cli "$@"
