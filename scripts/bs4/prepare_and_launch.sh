#!/bin/bash
set -euo pipefail
bs4_checkout="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$bs4_checkout"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
exec /home/zwan0726/miniforge3/envs/bidcell_local/bin/python -m scripts.bs4.entry launch --root /home/zwan0726/work/outputs/cos_bs4_20261012T013909+1100 "$@"
