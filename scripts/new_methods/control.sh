#!/usr/bin/env bash
set -euo pipefail
cd /home/zwan0726/work/projects/BIDCell_new_methods_20261013
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
command_name=${1:-status}
if (($#)); then shift; fi
exec /home/zwan0726/miniforge3/envs/bidcell_local/bin/python -m scripts.new_methods.runner "$command_name" --root /home/zwan0726/work/outputs/bidcell_new_methods_20261013 "$@"
