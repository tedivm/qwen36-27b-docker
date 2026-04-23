#!/usr/bin/env bash
# Launch vLLM serving Qwen3.6-27B AutoRound-INT4 on 2× 24 GB Ampere GPUs.
# OpenAI-compatible API on the configured port (default 1234).
#
# Flags explained in README.md. TL;DR: MTP spec-decode (n=3), FP8 KV,
# prefix caching opt-in for Qwen3.6 hybrid attention, Unsloth non-thinking
# sampler defaults, thinking hardcoded off via chat-template patch.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$SCRIPT_DIR/config.env"

: "${MODEL_DIR:?}" "${VENV_DIR:?}" "${LOG_DIR:?}" "${PORT:?}" "${SERVED_MODEL_NAME:?}"
: "${MAX_MODEL_LEN:=200000}" "${MAX_NUM_SEQS:=3}" "${GPU_MEMORY_UTIL:=0.92}"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

mkdir -p "$LOG_DIR"
export HF_HOME=$LOG_DIR/hf-cache
export TMPDIR=$LOG_DIR/tmp

# --- Clear the port if anything else (e.g. LM Studio) is holding it -------
if ss -tln 2>/dev/null | grep -q ":${PORT} "; then
  echo "Port ${PORT} is in use — clearing it..."
  fuser -k "${PORT}/tcp" 2>/dev/null || true
  sleep 2
fi

# --- Optional GPU thermal/power envelope ----------------------------------
# Requires passwordless `sudo -n nvidia-smi`. Soft-fails if not available.
if [[ -n "${GPU_POWER_LIMIT:-}" && -n "${GPU_CLOCK_MAX:-}" ]]; then
  echo "Applying GPU power limit (${GPU_POWER_LIMIT}W) and clock lock (max ${GPU_CLOCK_MAX} MHz)..."
  for i in 0 1; do
    sudo -n nvidia-smi -i "$i" -pm ENABLED                          >/dev/null 2>&1 \
      || echo "  warn: couldn't enable persistence on GPU $i"
    sudo -n nvidia-smi -i "$i" -pl "$GPU_POWER_LIMIT"               >/dev/null 2>&1 \
      || echo "  warn: couldn't set power limit on GPU $i"
    sudo -n nvidia-smi -i "$i" --lock-gpu-clocks="210,${GPU_CLOCK_MAX}" >/dev/null 2>&1 \
      || echo "  warn: couldn't lock clocks on GPU $i"
  done
  nvidia-smi --query-gpu=index,power.limit,persistence_mode --format=csv
  echo
fi

export PYTORCH_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_P2P_DISABLE=1
export VLLM_USE_FLASHINFER_SAMPLER=1

# --- Patched chat template: hardcode thinking off -------------------------
TEMPLATE=$MODEL_DIR/chat_template.no_thinking.jinja
if [[ ! -f $TEMPLATE ]]; then
  if [[ -f $MODEL_DIR/chat_template.jinja ]]; then
    printf '%s\n' '{%- set enable_thinking = false %}' > "$TEMPLATE"
    cat "$MODEL_DIR/chat_template.jinja" >> "$TEMPLATE"
    echo "Created $TEMPLATE with thinking forced off."
  else
    echo "  warn: no chat_template.jinja in $MODEL_DIR — using tokenizer default"
    TEMPLATE=""
  fi
fi
TEMPLATE_FLAG=()
[[ -n "$TEMPLATE" ]] && TEMPLATE_FLAG=(--chat-template "$TEMPLATE")

CUDA_VISIBLE_DEVICES=0,1 vllm serve "$MODEL_DIR" \
  --served-model-name "$SERVED_MODEL_NAME" \
  "${TEMPLATE_FLAG[@]}" \
  --override-generation-config '{"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0}' \
  --port "$PORT" \
  --dtype float16 \
  --quantization auto_round \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --tensor-parallel-size 2 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens 4128 \
  --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
  --disable-custom-all-reduce \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --trust-remote-code \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}' \
  2>&1 | tee "$LOG_DIR/vllm.log"
