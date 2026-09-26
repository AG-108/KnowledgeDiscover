#!/usr/bin/env bash

set -u

project_dir="${KD_PROJECT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
results_dir="$project_dir/results"
preflight_dir="$results_dir/server_gpu_missing_preflight_20260925"
queue_dir="$results_dir/server_gpu_missing_queue_20260925"
physo_exit="$results_dir/server_physo_full_20260925/run.exit"

cd "$project_dir" || exit 1
mkdir -p "$queue_dir"
source hpc/activate_kd_server.sh || exit 1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

printf 'Waiting for PhySO at %s\n' "$(date -Is)" >>"$queue_dir/queue.log"
while [[ ! -f "$physo_exit" ]]; do
  sleep 60
done
printf 'PhySO exit=%s at %s\n' "$(cat "$physo_exit")" "$(date -Is)" >>"$queue_dir/queue.log"

run_stage() {
  local model="$1"
  local timeout_seconds="$2"
  local output_dir="$results_dir/server_${model}_missing_20260925"
  local status

  mkdir -p "$output_dir"
  printf 'Starting %s at %s\n' "$model" "$(date -Is)" >>"$queue_dir/queue.log"
  python -u scripts/run_parallel_benchmark.py \
    --manifest "$preflight_dir/missing_${model}_manifest.json" \
    --compatibility "$preflight_dir/compatibility_report.json" \
    --output-dir "$output_dir" \
    --cpu-workers 1 \
    --gpu-workers 1 \
    --timeout "$timeout_seconds" \
    --retries 0 \
    --checkpoint-every 5 \
    >"$output_dir/run.log" 2>&1
  status=$?
  printf '%s\n' "$status" >"$output_dir/run.exit"
  printf 'Finished %s exit=%s at %s\n' "$model" "$status" "$(date -Is)" >>"$queue_dir/queue.log"
}

run_stage symbolicgpt 3600
run_stage dso 7200
printf 'Queue finished at %s\n' "$(date -Is)" >>"$queue_dir/queue.log"
