#!/usr/bin/env bash
# Serve the LLM-SR protocol backbone (meta-llama/Llama-3.1-8B-Instruct) with
# vLLM's OpenAI-compatible server, as LLM-SRBench does.  Weights are read from
# a local directory only; this script never downloads a model.
#
#   LLMSR_BACKBONE_PATH=/path/to/Llama-3.1-8B-Instruct CUDA_VISIBLE_DEVICES=0 \
#     bash scripts/serve_llmsr_backbone.sh
#
# Optional settings (also read from .local/server.env): LLMSR_VLLM_ENV (conda
# env containing vllm), LLMSR_HOST, LLMSR_PORT, LLMSR_MAX_MODEL_LEN,
# LLMSR_GPU_MEMORY_UTILIZATION.  Extra arguments are passed to `vllm serve`.
set -euo pipefail
KD_PROJECT="${KD_PROJECT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
if [[ -f "$KD_PROJECT/.local/server.env" ]]; then
  source "$KD_PROJECT/.local/server.env"
fi
: "${LLMSR_BACKBONE_PATH:?Set LLMSR_BACKBONE_PATH to the local Llama-3.1-8B-Instruct directory}"
if [[ ! -f "$LLMSR_BACKBONE_PATH/config.json" ]]; then
  echo "No config.json in $LLMSR_BACKBONE_PATH; download the model there first" >&2
  exit 1
fi
if [[ -n "${LLMSR_VLLM_ENV:-}" ]]; then
  source "${KD_CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}"
  conda activate "$LLMSR_VLLM_ENV"
fi
if ! command -v vllm >/dev/null 2>&1; then
  echo "vllm is not installed in the active environment; see the LLM-SR section of docs/methods.md" >&2
  exit 1
fi
# Refuse any Hugging Face Hub access: a missing file must fail, not download.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
args=(
  serve "$LLMSR_BACKBONE_PATH"
  --served-model-name meta-llama/Llama-3.1-8B-Instruct
  --dtype auto
  --host "${LLMSR_HOST:-127.0.0.1}"
  --port "${LLMSR_PORT:-10005}"
  --seed 0
  --max-model-len "${LLMSR_MAX_MODEL_LEN:-8192}"
  --gpu-memory-utilization "${LLMSR_GPU_MEMORY_UTILIZATION:-0.90}"
)
exec vllm "${args[@]}" "$@"
