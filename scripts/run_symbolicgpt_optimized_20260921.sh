#!/usr/bin/env bash

# Launch or resume the optimized full SymbolicGPT benchmark. Successful cases
# are discovered only in OUTPUT_DIR; legacy benchmark directories are not used.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-/opt/conda/envs/pdebench/bin/python}"
MANIFEST="results/symbolicgpt_optimized_manifest_20260921/benchmark_manifest.json"
COMPATIBILITY="results/compatibility_20260917/compatibility_report.json"
OUTPUT_DIR="results/symbolicgpt_full_20260918"

cd "$PROJECT_DIR" || exit 1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export JULIA_NUM_THREADS=1

exec "$PYTHON" -u scripts/run_parallel_benchmark.py \
    --manifest "$MANIFEST" \
    --compatibility "$COMPATIBILITY" \
    --output-dir "$OUTPUT_DIR" \
    --cpu-workers 1 \
    --gpu-workers 1 \
    --timeout 3600 \
    --retries 0 \
    --checkpoint-every 10
