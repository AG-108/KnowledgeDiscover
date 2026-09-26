#!/usr/bin/env bash
# Source for CPU runs with an existing environment selected in local settings.
KD_PROJECT="${KD_PROJECT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
if [[ -f "$KD_PROJECT/.local/server.env" ]]; then
  source "$KD_PROJECT/.local/server.env" || return
fi
KD_STORAGE_ROOT="${KD_STORAGE_ROOT:-$HOME/.local/share/kd}"
KD_ENV="${KD_ENV:-$KD_STORAGE_ROOT/envs/kd-cpu}"
export KD_PROJECT KD_ENV
export PATH="$KD_ENV/bin:$PATH"
export PYTHONPATH="$KD_PROJECT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-$KD_STORAGE_ROOT/.julia}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-1}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
cd "$KD_PROJECT" || return
