#!/usr/bin/env bash

# Resume the September 18 benchmark outputs after a container replacement.
# Successful case directories are reused in place; only interrupted, failed,
# or timed-out cases are executed again.

set -uo pipefail

PROJECT_DIR="/2501001sjlkff1/pdebench/KnowledgeDiscover_run_20260917"
PYTHON="/opt/conda/envs/pdebench/bin/python"
GPU_MANIFEST="results/full_benchmark_20260917/gpu/benchmark_manifest.json"
CPU_CLASSICAL_MANIFEST="results/full_benchmark_20260917/cpu_classical/benchmark_manifest.json"
CPU_SPARSE_MANIFEST="results/full_benchmark_20260917/cpu_sparse/benchmark_manifest.json"
DSCV_MANIFEST="results/dscv_reduced_manifest_20260918/benchmark_manifest.json"
COMPATIBILITY="results/compatibility_20260917/compatibility_report.json"

cd "$PROJECT_DIR" || exit 1
mkdir -p logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export JULIA_NUM_THREADS=1
export JULIA_DEPOT_PATH=/2501001sjlkff1/pdebench/.julia
export JULIA_PKG_SERVER=https://mirrors.nju.edu.cn/julia
export JULIA_PKG_PRECOMPILE_AUTO=0

run_and_record() {
    local label="$1"
    shift
    local rc=0

    printf 'Starting %s at %s\n' "$label" "$(date -Is)"
    "$@" >"logs/${label}.log" 2>&1
    rc=$?
    printf '%s\n' "$rc" >"logs/${label}.exit"
    printf 'Finished %s with rc=%s at %s\n' "$label" "$rc" "$(date -Is)"
    return 0
}

run_gpu_queue() {
    run_and_record symbolicgpt_resume_20260919 \
        "$PYTHON" -u scripts/run_parallel_benchmark.py \
        --manifest "$GPU_MANIFEST" \
        --compatibility "$COMPATIBILITY" \
        --output-dir results/symbolicgpt_full_20260918 \
        --reuse-dir results/full_benchmark_no_llmsr_20260917 \
        --reuse-dir results/full_benchmark_20260917/gpu \
        --exclude-model dso \
        --exclude-model physo \
        --exclude-model spr \
        --exclude-model dlga \
        --exclude-model deepmod \
        --exclude-model pdenet \
        --exclude-model eqgpt \
        --exclude-model llmsr \
        --cpu-workers 1 \
        --gpu-workers 1 \
        --timeout 3600 \
        --retries 0 \
        --checkpoint-every 10

    run_and_record dso_resume_20260919 \
        "$PYTHON" -u scripts/run_parallel_benchmark.py \
        --manifest "$GPU_MANIFEST" \
        --compatibility "$COMPATIBILITY" \
        --output-dir results/dso_full_20260918 \
        --reuse-dir results/full_benchmark_no_llmsr_20260917 \
        --reuse-dir results/full_benchmark_20260917/gpu \
        --exclude-model symbolicgpt \
        --exclude-model physo \
        --exclude-model spr \
        --exclude-model dlga \
        --exclude-model deepmod \
        --exclude-model pdenet \
        --exclude-model eqgpt \
        --exclude-model llmsr \
        --cpu-workers 1 \
        --gpu-workers 1 \
        --timeout 7200 \
        --retries 0 \
        --checkpoint-every 5

    printf 'GPU resume queue finished at %s\n' "$(date -Is)"
}

run_cpu_queue() {
    run_and_record fast_cpu_retry_20260919 \
        "$PYTHON" -u scripts/run_parallel_benchmark.py \
        --manifest "$CPU_CLASSICAL_MANIFEST" \
        --manifest "$CPU_SPARSE_MANIFEST" \
        --compatibility "$COMPATIBILITY" \
        --output-dir results/fast_cpu_baselines_20260918 \
        --reuse-dir results/full_benchmark_no_llmsr_20260917 \
        --reuse-dir results/full_benchmark_20260917/cpu_classical \
        --reuse-dir results/full_benchmark_20260917/cpu_sparse \
        --exclude-model dscv \
        --exclude-model sga \
        --exclude-model llmsr \
        --cpu-workers 4 \
        --gpu-workers 1 \
        --timeout 3600 \
        --retries 0 \
        --checkpoint-every 10

    run_and_record dscv_retry_20260919 \
        "$PYTHON" -u scripts/run_parallel_benchmark.py \
        --manifest "$DSCV_MANIFEST" \
        --compatibility "$COMPATIBILITY" \
        --output-dir results/dscv_reduced_20260918 \
        --reuse-dir results/dscv_reduced_pilot_20260918 \
        --exclude-model llmsr \
        --cpu-workers 4 \
        --gpu-workers 1 \
        --timeout 600 \
        --retries 0 \
        --checkpoint-every 5

    run_and_record sga_retry_20260919 \
        "$PYTHON" -u scripts/run_parallel_benchmark.py \
        --manifest "$CPU_SPARSE_MANIFEST" \
        --compatibility "$COMPATIBILITY" \
        --output-dir results/sga_full_20260918 \
        --reuse-dir results/full_benchmark_no_llmsr_20260917 \
        --reuse-dir results/full_benchmark_20260917/cpu_sparse \
        --exclude-model dscv \
        --exclude-model pdefind \
        --exclude-model weakform \
        --exclude-model sindy \
        --exclude-model pysindy \
        --exclude-model llmsr \
        --cpu-workers 4 \
        --gpu-workers 1 \
        --timeout 3600 \
        --retries 0 \
        --checkpoint-every 1

    printf 'CPU retry queue finished at %s\n' "$(date -Is)"
}

case "${1:-}" in
    gpu)
        run_gpu_queue
        ;;
    cpu)
        run_cpu_queue
        ;;
    *)
        printf 'Usage: %s {gpu|cpu}\n' "$0" >&2
        exit 2
        ;;
esac
