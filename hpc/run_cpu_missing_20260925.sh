#!/usr/bin/env bash

set -u

project_dir="${KD_PROJECT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
preflight_dir="$project_dir/results/server_cpu_missing_preflight_20260925"
output_dir="$project_dir/results/server_cpu_missing_20260925"

cd "$project_dir" || exit 1
mkdir -p "$output_dir"
source hpc/activate_kd_server.sh || exit 1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python -u scripts/run_parallel_benchmark.py \
  --manifest "$preflight_dir/missing_manifest.json" \
  --compatibility "$preflight_dir/compatibility_report.json" \
  --output-dir "$output_dir" \
  --cpu-workers 2 \
  --gpu-workers 1 \
  --timeout 3600 \
  --retries 0 \
  --checkpoint-every 1 \
  >"$output_dir/run.log" 2>&1
status=$?
printf '%s\n' "$status" >"$output_dir/run.exit"
exit "$status"
