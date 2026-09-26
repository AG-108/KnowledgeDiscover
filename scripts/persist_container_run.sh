#!/usr/bin/env bash

# Override these for a historical worktree or a separate persistent volume.
KD_PROJECT="${KD_PROJECT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
KD_STORAGE_ROOT="${KD_STORAGE_ROOT:-$HOME/.local/share/kd}"

set -u

source_dir=$KD_PROJECT
persist_root=$KD_STORAGE_ROOT
destination="${KD_PERSIST_DIR:-$persist_root/KnowledgeDiscover_run_20260917}"
sync_log="$persist_root/persistence_sync_20260917.log"

log() {
  printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$sync_log"
}

sync_tree() {
  mkdir -p "$destination"
  cp -au "$source_dir"/. "$destination"/
  sync
  log "incremental sync complete"
}

log "starting initial persistent copy from $source_dir to $destination"
mkdir -p "$destination/environment"
cp -a "$source_dir"/. "$destination"/
/opt/conda/bin/conda list -n pdebench --explicit \
  > "$destination/environment/conda-explicit.txt" 2>&1 || true
/opt/conda/bin/conda env export -n pdebench \
  > "$destination/environment/conda-environment.yml" 2>&1 || true
"${KD_PYTHON:-python}" -m pip freeze \
  > "$destination/environment/pip-freeze.txt" 2>&1 || true
sync
log "initial persistent copy complete"

for round in $(seq 1 9); do
  sleep 120
  sync_tree
  log "periodic sync $round/9"
done

log "stopping benchmark for final consistent checkpoint"
tmux has-session -t kd_full_no_llmsr 2>/dev/null \
  && tmux kill-session -t kd_full_no_llmsr || true
sleep 5
pkill -TERM -f "$KD_PROJECT/run_benchmark.py --_case-file" 2>/dev/null || true
pkill -TERM -f 'scripts/run_parallel_benchmark.py' 2>/dev/null || true
sleep 5

# A final non-update copy overwrites any file captured while it was being written.
cp -a "$source_dir"/. "$destination"/
sync

source_results=$(find "$source_dir/results/full_benchmark_no_llmsr_20260917/cases" \
  -name result.json 2>/dev/null | wc -l)
saved_results=$(find "$destination/results/full_benchmark_no_llmsr_20260917/cases" \
  -name result.json 2>/dev/null | wc -l)
source_bytes=$(du -sb "$source_dir" | awk '{print $1}')
saved_bytes=$(du -sb "$destination" | awk '{print $1}')

{
  printf 'checkpoint_completed_at=%s\n' "$(date -Is)"
  printf 'source=%s\n' "$source_dir"
  printf 'destination=%s\n' "$destination"
  printf 'source_result_files=%s\n' "$source_results"
  printf 'saved_result_files=%s\n' "$saved_results"
  printf 'source_bytes=%s\n' "$source_bytes"
  printf 'saved_bytes=%s\n' "$saved_bytes"
  printf 'resume_command=%s\n' \
    "/bin/bash $KD_PROJECT/scripts/launch_full_no_llmsr.sh"
} > "$destination/PERSISTENCE_CHECKPOINT.txt"
sync
log "final checkpoint complete: source_results=$source_results saved_results=$saved_results"
