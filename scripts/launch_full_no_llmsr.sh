#!/usr/bin/env bash

# Override these for a historical worktree or a separate persistent volume.
KD_PROJECT="${KD_PROJECT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
KD_STORAGE_ROOT="${KD_STORAGE_ROOT:-$HOME/.local/share/kd}"

set +e
cd "$KD_PROJECT" || exit 1

mkdir -p logs
rm -f logs/full_no_llmsr_20260917.exit
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
JULIA_NUM_THREADS=2 \
"${KD_PYTHON:-python}" -u scripts/run_parallel_benchmark.py \
  --manifest results/full_benchmark_20260917/gpu/benchmark_manifest.json \
  --manifest results/full_benchmark_20260917/cpu_classical/benchmark_manifest.json \
  --manifest results/full_benchmark_20260917/cpu_sparse/benchmark_manifest.json \
  --compatibility results/compatibility_20260917/compatibility_report.json \
  --output-dir results/full_benchmark_no_llmsr_20260917 \
  --reuse-dir results/full_benchmark_20260917/gpu \
  --reuse-dir results/full_benchmark_20260917/cpu_classical \
  --reuse-dir results/full_benchmark_20260917/cpu_sparse \
  --exclude-model llmsr \
  --cpu-workers 6 \
  --gpu-workers 1 \
  --timeout 21600 \
  --retries 1 \
  --checkpoint-every 10 \
  > logs/full_no_llmsr_20260917.log 2>&1
status=$?
printf '%s\n' "$status" > logs/full_no_llmsr_20260917.exit
exit "$status"
