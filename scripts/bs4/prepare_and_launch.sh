#!/bin/bash
set -euo pipefail
cd /home/zwan0726/work/projects/BIDCell_cos_bs4_20261012T013909+1100
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
exec /home/zwan0726/miniforge3/envs/bidcell_local/bin/python -m scripts.bs4.entry launch --root /home/zwan0726/work/outputs/cos_bs4_20261012T013909+1100 "$@"
