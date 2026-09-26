#!/usr/bin/env bash

# Override these for a historical worktree or a separate persistent volume.
KD_PROJECT="${KD_PROJECT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
KD_STORAGE_ROOT="${KD_STORAGE_ROOT:-$HOME/.local/share/kd}"

# Continue the persistent benchmark queues after the currently running fast
# CPU and PhySO stages finish. Each stage has its own output, log, and exit
# status so an interrupted container can be resumed without mixing budgets.

set -uo pipefail

PROJECT_DIR="$KD_PROJECT"
PYTHON="${KD_PYTHON:-python}"
GPU_MANIFEST="results/full_benchmark_20260917/gpu/benchmark_manifest.json"
CPU_SPARSE_MANIFEST="results/full_benchmark_20260917/cpu_sparse/benchmark_manifest.json"
COMPATIBILITY="results/compatibility_20260917/compatibility_report.json"

cd "$PROJECT_DIR" || exit 1
mkdir -p logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export JULIA_NUM_THREADS=1
export JULIA_DEPOT_PATH=$KD_STORAGE_ROOT/.julia
export JULIA_PKG_SERVER=https://mirrors.nju.edu.cn/julia
export JULIA_PKG_PRECOMPILE_AUTO=0

wait_for_stage() {
    local stage_stem="$1"
    local exit_file="${stage_stem}.exit"
    local pid_file="${stage_stem}.pid"
    local stage_pid=""

    printf 'Waiting for %s at %s\n' "$stage_stem" "$(date -Is)"
    while [[ ! -f "$exit_file" ]]; do
        if [[ -f "$pid_file" ]]; then
            stage_pid="$(head -n 1 "$pid_file")"
        fi
        if [[ -z "$stage_pid" ]] || ! kill -0 "$stage_pid" 2>/dev/null; then
            printf 'Warning: %s has no live process and no exit file; continuing.\n' "$stage_stem"
            break
        fi
        sleep 30
    done
    printf 'Dependency %s finished or stopped at %s\n' "$stage_stem" "$(date -Is)"
}

run_gpu_stage() {
    local label="$1"
    local output_dir="$2"
    local timeout_seconds="$3"
    shift 3
    local log_file="logs/${label}.log"
    local exit_file="logs/${label}.exit"
    local rc=0

    if [[ -f "$exit_file" ]] && [[ "$(head -n 1 "$exit_file")" == "0" ]]; then
        printf 'Skipping completed stage %s at %s\n' "$label" "$(date -Is)"
        return 0
    fi

    printf 'Starting %s at %s\n' "$label" "$(date -Is)"
    "$PYTHON" -u scripts/run_parallel_benchmark.py \
        --manifest "$GPU_MANIFEST" \
        --compatibility "$COMPATIBILITY" \
        --output-dir "$output_dir" \
        --reuse-dir results/full_benchmark_no_llmsr_20260917 \
        --reuse-dir results/full_benchmark_20260917/gpu \
        --cpu-workers 1 \
        --gpu-workers 1 \
        --timeout "$timeout_seconds" \
        --retries 0 \
        --checkpoint-every 10 \
        "$@" >"$log_file" 2>&1
    rc=$?
    printf '%s\n' "$rc" >"$exit_file"
    printf 'Finished %s with rc=%s at %s\n' "$label" "$rc" "$(date -Is)"
    return 0
}

run_cpu_stage() {
    local label="$1"
    local output_dir="$2"
    local timeout_seconds="$3"
    shift 3
    local log_file="logs/${label}.log"
    local exit_file="logs/${label}.exit"
    local rc=0

    if [[ -f "$exit_file" ]] && [[ "$(head -n 1 "$exit_file")" == "0" ]]; then
        printf 'Skipping completed stage %s at %s\n' "$label" "$(date -Is)"
        return 0
    fi

    printf 'Starting %s at %s\n' "$label" "$(date -Is)"
    "$PYTHON" -u scripts/run_parallel_benchmark.py \
        --manifest "$CPU_SPARSE_MANIFEST" \
        --compatibility "$COMPATIBILITY" \
        --output-dir "$output_dir" \
        --reuse-dir results/full_benchmark_no_llmsr_20260917 \
        --reuse-dir results/full_benchmark_20260917/cpu_sparse \
        --cpu-workers 6 \
        --gpu-workers 1 \
        --timeout "$timeout_seconds" \
        --retries 0 \
        --checkpoint-every 5 \
        "$@" >"$log_file" 2>&1
    rc=$?
    printf '%s\n' "$rc" >"$exit_file"
    printf 'Finished %s with rc=%s at %s\n' "$label" "$rc" "$(date -Is)"
    return 0
}

run_gpu_queue() {
    wait_for_stage logs/physo_full_20260918

    # Finish the small PDE-oriented GPU baselines before starting long SR jobs.
    run_gpu_stage gpu_pde_remaining_20260918 results/gpu_pde_remaining_20260918 7200 \
        --exclude-model dso \
        --exclude-model symbolicgpt \
        --exclude-model physo \
        --exclude-model spr \
        --exclude-model llmsr

    # SPR has only the PDE collection, so finishing it gives another complete
    # baseline before the much larger SymbolicGPT and DSO queues.
    run_gpu_stage spr_full_20260918 results/spr_full_20260918 7200 \
        --exclude-model dso \
        --exclude-model symbolicgpt \
        --exclude-model physo \
        --exclude-model dlga \
        --exclude-model deepmod \
        --exclude-model pdenet \
        --exclude-model eqgpt \
        --exclude-model llmsr

    run_gpu_stage symbolicgpt_full_20260918 results/symbolicgpt_full_20260918 3600 \
        --exclude-model dso \
        --exclude-model physo \
        --exclude-model spr \
        --exclude-model dlga \
        --exclude-model deepmod \
        --exclude-model pdenet \
        --exclude-model eqgpt \
        --exclude-model llmsr

    # DSO is last because the initial samples took roughly an hour each.
    run_gpu_stage dso_full_20260918 results/dso_full_20260918 7200 \
        --exclude-model symbolicgpt \
        --exclude-model physo \
        --exclude-model spr \
        --exclude-model dlga \
        --exclude-model deepmod \
        --exclude-model pdenet \
        --exclude-model eqgpt \
        --exclude-model llmsr

    printf 'GPU queue finished at %s\n' "$(date -Is)"
}

run_cpu_queue() {
    wait_for_stage logs/dscv_reduced_20260918

    run_cpu_stage sga_full_20260918 results/sga_full_20260918 3600 \
        --exclude-model dscv \
        --exclude-model pdefind \
        --exclude-model weakform \
        --exclude-model sindy \
        --exclude-model pysindy \
        --exclude-model llmsr

    printf 'CPU queue finished at %s\n' "$(date -Is)"
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
