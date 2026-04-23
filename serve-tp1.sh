#!/usr/bin/env bash
# Single-GPU variant (TP=1) — runs the same stack on GPU 1 only. Useful as
# an A/B against serve.sh (TP=2). For dense Qwen3.6-27B on 2× 3090, TP=2 is
# ~1.5× faster due to memory-bandwidth split outweighing PCIe NCCL cost —
# keep this script for reference or for a 1-GPU machine.
#
# Context is capped lower than TP=2: FP8 KV on 1× 24 GB fits ~57k tokens max.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$SCRIPT_DIR/config.env"

: "${MODEL_DIR:?}" "${VENV_DIR:?}" "${LOG_DIR:?}"
: "${PORT_TP1:=1235}"
: "${GPU_INDEX_TP1:=1}"        # Pick the card that isn't driving your desktop
: "${MAX_MODEL_LEN_TP1:=48000}"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

mkdir -p "$LOG_DIR"
export HF_HOME=$LOG_DIR/hf-cache
export TMPDIR=$LOG_DIR/tmp

# Refuse to start if a 123x-range port is taken — TP=1 is for A/B, so we want
# to catch port collisions loudly (unlike serve.sh which auto-frees 1234).
if ss -tln 2>/dev/null | grep -q -E ':123[0-9] '; then
  echo "A server appears to be on a 123x port — stop it first." >&2
  ss -tln | grep -E ':123[0-9] ' >&2
  exit 1
fi

if [[ -n "${GPU_POWER_LIMIT:-}" && -n "${GPU_CLOCK_MAX:-}" ]]; then
  echo "Applying power/clock envelope to GPU ${GPU_INDEX_TP1}..."
  sudo -n nvidia-smi -i "$GPU_INDEX_TP1" -pm ENABLED                              >/dev/null 2>&1 || true
  sudo -n nvidia-smi -i "$GPU_INDEX_TP1" -pl "$GPU_POWER_LIMIT"                   >/dev/null 2>&1 || true
  sudo -n nvidia-smi -i "$GPU_INDEX_TP1" --lock-gpu-clocks="210,${GPU_CLOCK_MAX}" >/dev/null 2>&1 || true
fi

export PYTORCH_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_P2P_DISABLE=1
export VLLM_USE_FLASHINFER_SAMPLER=1

TEMPLATE=$MODEL_DIR/chat_template.no_thinking.jinja
if [[ ! -f $TEMPLATE && -f $MODEL_DIR/chat_template.jinja ]]; then
  printf '%s\n' '{%- set enable_thinking = false %}' > "$TEMPLATE"
  cat "$MODEL_DIR/chat_template.jinja" >> "$TEMPLATE"
fi
TEMPLATE_FLAG=()
[[ -f "$TEMPLATE" ]] && TEMPLATE_FLAG=(--chat-template "$TEMPLATE")

CUDA_VISIBLE_DEVICES="$GPU_INDEX_TP1" vllm serve "$MODEL_DIR" \
  --served-model-name "${SERVED_MODEL_NAME:-qwen3.6-27b}-tp1" \
  "${TEMPLATE_FLAG[@]}" \
  --override-generation-config '{"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0}' \
  --port "$PORT_TP1" \
  --dtype float16 \
  --quantization auto_round \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --tensor-parallel-size 1 \
  --max-model-len "$MAX_MODEL_LEN_TP1" \
  --max-num-seqs "${MAX_NUM_SEQS:-3}" \
  --max-num-batched-tokens 4128 \
  --gpu-memory-utilization 0.95 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --trust-remote-code \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}' \
  2>&1 | tee "$LOG_DIR/vllm-tp1.log"
