#!/usr/bin/env bash

# Shared PDEBench/KnowledgeDiscover CPU environment on the EIT Slurm cluster.
export KD_PROJECT="$HOME/pdebench/KnowledgeDiscover_run_20260917"
export KD_ENV="$HOME/pdebench/envs/kd-py39-cpu"

export PATH="$KD_ENV/bin:$PATH"
export PYTHONPATH="$KD_PROJECT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1

# Keep Julia/PySR state on persistent storage and use the reachable mirror.
export JULIA_DEPOT_PATH="$HOME/pdebench/.julia"
export JULIA_PKG_SERVER="https://mirrors.nju.edu.cn/julia"

# One process should use one BLAS/OpenMP thread when benchmark cases are
# parallelized across Slurm CPUs. A job script may override these beforehand.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-1}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

cd "$KD_PROJECT"
