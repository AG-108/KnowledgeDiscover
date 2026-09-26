#!/usr/bin/env bash

set -u

project_dir="${KD_PROJECT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
output_dir="$project_dir/results/server_physo_full_20260925"

cd "$project_dir" || exit 1
mkdir -p "$output_dir"
source hpc/activate_kd_server.sh || exit 1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python -u run_benchmark.py \
  --profile full \
  --tasks sr ode \
  --models physo \
  --seeds 0 \
  --timeout 1800 \
  --resume \
  --device physo=cuda:0 \
  --output-dir "$output_dir" \
  >"$output_dir/run.log" 2>&1
status=$?
printf '%s\n' "$status" >"$output_dir/run.exit"
exit "$status"
