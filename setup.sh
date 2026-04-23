#!/usr/bin/env bash
# First-run setup: create Python venv, install vLLM + auto-round, download
# weights. Re-runnable — skips steps that are already done.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONFIG=$SCRIPT_DIR/config.env
if [[ ! -f $CONFIG ]]; then
  echo "error: $CONFIG not found. Copy config.env.example → config.env and edit." >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$CONFIG"

: "${MODEL_DIR:?set MODEL_DIR in config.env}"
: "${VENV_DIR:?set VENV_DIR in config.env}"
: "${LOG_DIR:?set LOG_DIR in config.env}"

mkdir -p "$LOG_DIR" "$(dirname "$VENV_DIR")" "$(dirname "$MODEL_DIR")"

# Move pip/HF caches onto the same drive as LOG_DIR so we don't fill /.
export TMPDIR=$LOG_DIR/tmp
export PIP_CACHE_DIR=$LOG_DIR/pip-cache
export HF_HOME=$LOG_DIR/hf-cache
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$HF_HOME"

# --- venv ------------------------------------------------------------------
if [[ ! -d $VENV_DIR ]]; then
  echo ">> Creating venv at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel
python -m pip install 'vllm==0.19.1' auto-round hf_transfer

# --- weights ---------------------------------------------------------------
if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo ">> Downloading Lorbus/Qwen3.6-27B-int4-AutoRound to $MODEL_DIR"
  mkdir -p "$MODEL_DIR"
  HF_HUB_ENABLE_HF_TRANSFER=1 hf download Lorbus/Qwen3.6-27B-int4-AutoRound \
    --local-dir "$MODEL_DIR"
else
  echo ">> Model already present at $MODEL_DIR — skipping download"
fi

echo
echo "Setup complete."
echo "  Model : $MODEL_DIR"
echo "  Venv  : $VENV_DIR"
echo "  Logs  : $LOG_DIR"
echo "  Run   : ./serve.sh"
