#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
config="${1:-configs/benchmark/benchmark.json}"
if [[ $# -gt 0 ]]; then
    shift
fi

cd "${repo_root}"
exec "${PYTHON:-python}" -u run_benchmark.py --config "${config}" "$@"
