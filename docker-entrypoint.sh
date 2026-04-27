#!/usr/bin/env bash
set -euo pipefail

# --- Auto-detect GPU count & set TP if not overridden --------------------
if [[ -z "${TENSOR_PARALLEL:-}" ]]; then
    GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
    if (( GPU_COUNT == 0 )); then
        echo "ERROR: No GPUs detected. Pass --gpus to docker run." >&2
        exit 1
    fi
    TENSOR_PARALLEL=$GPU_COUNT
    echo "Detected ${GPU_COUNT} GPU(s), setting TENSOR_PARALLEL=${TENSOR_PARALLEL}"
fi
export TENSOR_PARALLEL

# --- Adjust defaults based on GPU count -----------------------------------
if (( TENSOR_PARALLEL == 1 )); then
    : "${MAX_MODEL_LEN:=48000}"
    : "${GPU_MEMORY_UTIL:=0.95}"
    echo "Single-GPU mode: MAX_MODEL_LEN=${MAX_MODEL_LEN}, GPU_MEMORY_UTIL=${GPU_MEMORY_UTIL}"
fi
export MAX_MODEL_LEN
export GPU_MEMORY_UTIL

# --- Determine visible GPU indices ----------------------------------------
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    CUDA_VISIBLE_DEVICES=""
    for i in $(seq 0 $((TENSOR_PARALLEL - 1))); do
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:+${CUDA_VISIBLE_DEVICES},}${i}"
    done
fi
export CUDA_VISIBLE_DEVICES

# --- Cachedir setup -------------------------------------------------------
export HF_HOME="${MODEL_DIR}/.hf_cache"
export TMPDIR="${MODEL_DIR}/.tmp"
export PIP_CACHE_DIR="${MODEL_DIR}/.pip_cache"
mkdir -p "$HF_HOME" "$TMPDIR" "$PIP_CACHE_DIR"

# --- Logs setup -----------------------------------------------------------
mkdir -p "$LOG_DIR"

# --- Commands --------------------------------------------------------------
CMD="${1:-serve}"

case "$CMD" in
    download)
        if [[ -f "${MODEL_DIR}/config.json" ]]; then
            echo "Model already present at ${MODEL_DIR} — skipping download."
            exit 0
        fi
        echo "Downloading ${MODEL_REPO} to ${MODEL_DIR} ..."
        mkdir -p "$MODEL_DIR"
        HF_HUB_ENABLE_HF_TRANSFER=1 hf download "$MODEL_REPO" \
            --local-dir "$MODEL_DIR"
        echo "Download complete."
        exit 0
        ;;

    serve)
        if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
            if [[ "${MODEL_DOWNLOAD:-0}" == "1" ]]; then
                echo "Model not found and MODEL_DOWNLOAD=1 — downloading now ..."
                HF_HUB_ENABLE_HF_TRANSFER=1 hf download "$MODEL_REPO" \
                    --local-dir "$MODEL_DIR"
                echo "Download complete."
            else
                echo "ERROR: Model not found at ${MODEL_DIR}." >&2
                echo "Run 'docker compose run --rm qwen36 download' or set MODEL_DOWNLOAD=1." >&2
                exit 1
            fi
        fi

        # --- Build generation config JSON from env vars ---------------------
        GEN_CONFIG="{\"temperature\": ${TEMPERATURE}, \"top_p\": ${TOP_P}, \"top_k\": ${TOP_K}, \"min_p\": ${MIN_P}, \"presence_penalty\": ${PRESENCE_PENALTY}, \"repetition_penalty\": ${REPETITION_PENALTY}}"

        # --- Optional reasoning parser --------------------------------------
        REASONING_PARSER_FLAG=()
        [[ -n "${REASONING_PARSER:-}" ]] && REASONING_PARSER_FLAG=(--reasoning-parser "$REASONING_PARSER")

        echo "Starting vLLM server:"
        echo "  Model              : ${MODEL_DIR}"
        echo "  Served as          : ${SERVED_MODEL_NAME}"
        echo "  Port               : ${PORT}"
        echo "  Tensor parallelism : ${TENSOR_PARALLEL}"
        echo "  Max model len      : ${MAX_MODEL_LEN}"
        echo "  Max num seqs       : ${MAX_NUM_SEQS}"
        echo "  GPU memory util    : ${GPU_MEMORY_UTIL}"
        echo "  CUDA devices       : ${CUDA_VISIBLE_DEVICES}"
        echo "  Generation config  : ${GEN_CONFIG}"
        echo ""

        vllm serve "$MODEL_DIR" \
            --served-model-name "$SERVED_MODEL_NAME" \
            --override-generation-config "$GEN_CONFIG" \
            --port "$PORT" \
            --dtype float16 \
            --quantization auto_round \
            --kv-cache-dtype fp8 \
            --enable-prefix-caching \
            --enable-chunked-prefill \
            --tensor-parallel-size "$TENSOR_PARALLEL" \
            --max-model-len "$MAX_MODEL_LEN" \
            --max-num-seqs "$MAX_NUM_SEQS" \
            --max-num-batched-tokens 4128 \
            --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
            --disable-custom-all-reduce \
            --enable-auto-tool-choice \
            --tool-call-parser qwen3_coder \
            --trust-remote-code \
            "${REASONING_PARSER_FLAG[@]}" \
            --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}' \
            2>&1 | tee -a "${LOG_DIR}/vllm.log"
        ;;

    *)
        echo "Unknown command: ${CMD}" >&2
        echo "Usage: docker-entrypoint.sh {download|serve}" >&2
        exit 1
        ;;
esac
