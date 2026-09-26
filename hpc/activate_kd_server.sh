#!/usr/bin/env bash
# Source this file after configuring .local/server.env (see server.env.example).
KD_PROJECT="${KD_PROJECT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
if [[ -f "$KD_PROJECT/.local/server.env" ]]; then
  source "$KD_PROJECT/.local/server.env" || return
fi
KD_STORAGE_ROOT="${KD_STORAGE_ROOT:-$HOME/.local/share/kd}"
KD_ENV_ROOT="${KD_ENV_ROOT:-$KD_STORAGE_ROOT/envs}"
case "${1:-unified}" in
  unified|main|operon) KD_ENV="${KD_ENV:-${KD_UNIFIED_ENV:-$KD_ENV_ROOT/kd-unified-py310}}" ;;
  legacy-main) KD_ENV="${KD_LEGACY_MAIN_ENV:-$KD_ENV_ROOT/kd-main-gpu}" ;;
  legacy-operon) KD_ENV="${KD_LEGACY_OPERON_ENV:-$KD_ENV_ROOT/kd-operon-py310}" ;;
  *) echo "Usage: source hpc/activate_kd_server.sh [unified|main|operon|legacy-main|legacy-operon]" >&2; return 2 ;;
esac
KD_CONDA_SH="${KD_CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"
if [[ ! -f "$KD_CONDA_SH" ]]; then
  echo "Set KD_CONDA_SH to the host's conda.sh in .local/server.env" >&2
  return 1
fi
source "$KD_CONDA_SH" || return
conda activate "$KD_ENV" || return
export KD_PROJECT KD_ENV KD_STORAGE_ROOT
export KD_E2E_SOURCE="${KD_E2E_SOURCE:-$KD_STORAGE_ROOT/external/e2e-source}"
export KD_E2E_CHECKPOINT="${KD_E2E_CHECKPOINT:-$KD_STORAGE_ROOT/checkpoints/e2e/model1.pt}"
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-$KD_STORAGE_ROOT/.julia}"
if [[ -z "${PYTHON_JULIAPKG_EXE:-}" ]]; then
  if [[ -x "$KD_ENV/julia_env/pyjuliapkg/install/bin/julia" ]]; then
    export PYTHON_JULIAPKG_EXE="$KD_ENV/julia_env/pyjuliapkg/install/bin/julia"
  elif [[ -x "$JULIA_DEPOT_PATH/environments/pyjuliapkg/pyjuliapkg/install/bin/julia" ]]; then
    export PYTHON_JULIAPKG_EXE="$JULIA_DEPOT_PATH/environments/pyjuliapkg/pyjuliapkg/install/bin/julia"
  fi
fi
export PYTHONPATH="$KD_PROJECT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export MPLBACKEND="${MPLBACKEND:-Agg}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
cd "$KD_PROJECT" || return
