#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=""
cd /home/zwan0726/work/projects/BIDCell_v2_paired_20261014
action="${1:-status}"
shift || true
if [[ "$action" == "train-status" && $# -gt 0 ]]; then
    exec /home/zwan0726/miniforge3/envs/bidcell_local/bin/python -B -m scripts.paired_v2_pipeline.cli "$action" --root /home/zwan0726/work/outputs/bidcell_v2_paired_20261014 --arm "$1"
fi
exec /home/zwan0726/miniforge3/envs/bidcell_local/bin/python -B -m scripts.paired_v2_pipeline.cli "$action" --root /home/zwan0726/work/outputs/bidcell_v2_paired_20261014 "$@"
