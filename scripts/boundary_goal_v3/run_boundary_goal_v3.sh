#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$(dirname "$(readlink -f "$0")")/../.."
exec /home/zwan0726/miniforge3/envs/bidcell_local/bin/python -B -m scripts.boundary_goal_v3.pipeline "$@"
