#!/usr/bin/env bash

# Override these for a historical worktree or a separate persistent volume.
KD_PROJECT="${KD_PROJECT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
KD_STORAGE_ROOT="${KD_STORAGE_ROOT:-$HOME/.local/share/kd}"

set +e

project_dir="$KD_PROJECT"
persist_dir="${KD_PERSIST_DIR:-$KD_STORAGE_ROOT/KnowledgeDiscover_run_20260917}"
run_log="$persist_dir/logs/full_no_llmsr_resume_20260917.log"
exit_file="$persist_dir/logs/full_no_llmsr_resume_20260917.exit"

cd "$project_dir" || exit 1
mkdir -p "$persist_dir/logs"
rm -f "$exit_file"

OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
JULIA_NUM_THREADS=2 \
"${KD_PYTHON:-python}" -u scripts/run_parallel_benchmark.py \
  --manifest "$project_dir/results/full_benchmark_20260917/gpu/benchmark_manifest.json" \
  --manifest "$project_dir/results/full_benchmark_20260917/cpu_classical/benchmark_manifest.json" \
  --manifest "$project_dir/results/full_benchmark_20260917/cpu_sparse/benchmark_manifest.json" \
  --compatibility "$project_dir/results/compatibility_20260917/compatibility_report.json" \
  --output-dir "$persist_dir/results/full_benchmark_no_llmsr_20260917" \
  --reuse-dir "$persist_dir/results/full_benchmark_20260917/gpu" \
  --reuse-dir "$persist_dir/results/full_benchmark_20260917/cpu_classical" \
  --reuse-dir "$persist_dir/results/full_benchmark_20260917/cpu_sparse" \
  --exclude-model llmsr \
  --cpu-workers 6 \
  --gpu-workers 1 \
  --timeout 21600 \
  --retries 1 \
  --checkpoint-every 10 \
  > "$run_log" 2>&1
status=$?
printf '%s\n' "$status" > "$exit_file"
exit "$status"
