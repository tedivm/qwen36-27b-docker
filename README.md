# Qwen3.6-27B vLLM Docker

Docker-based vLLM serving for [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) with [Lorbus AutoRound INT4 quant](https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound) and MTP speculative decoding. Model is downloaded at runtime and stored on a host volume so the container can be upgraded without redownloading weights.

Forked from [k0zakinio/qwen36-vllm-setup](https://github.com/k0zakinio/qwen36-vllm-setup).

## What you get

| Metric (dual RTX 3090, TP=2, 200K context) | Value |
|---|---|
| Sustained TPS on coding workloads | **~118** |
| Sustained TPS on prose | ~89 |
| Max context length | 200,000 tokens (172K KV pool headroom) |
| Vision support | yes (MoonViT, via `image_url` content parts) |

### Verified benchmarks (2x RTX 3090, TP=2)

Measured with `bench_tps.py` against the Docker container:

| Workload | Tokens | Time | TPS |
|---|---|---|---|
| Prose (800-word story) | 800 | 8.99s | **88.98** |
| Code (LRU cache impl) | 1200 | 10.16s | **118.13** |

Dense model, not MoE — no tensor shuffling at token boundaries, clean TP=2 split. No NVLink required; PCIe TP is fine on this workload.

## Hardware

Tested on 2x RTX 3090 (48 GB VRAM total). Also works at lower context on a single 24 GB card. GPU count is auto-detected — pass `--gpus all` for multi-GPU or `--gpus '"device=0"'` for single-GPU.

Minimum disk: ~20 GB for weights + ~6 GB for caches.

## Quick start

### Option 1: Docker Compose (recommended)

```bash
# Download model (one-time, stores in ./models)
docker compose run --rm qwen36 download

# Start the server
docker compose up -d

# Stop
docker compose down
```

### Option 2: Docker CLI

```bash
# Build locally
docker build -t qwen36-vllm .

# Or pull from GHCR
docker pull ghcr.io/tedivm/qwen36-27b-docker:latest

# Download model (one-time, stores in /path/on/host/models)
docker run --rm --gpus all \
  -v /path/on/host/models:/data/models \
  qwen36-vllm download

# Start server (auto-detects 1 or 2 GPUs)
docker run -d --name qwen36 --gpus all -p 1234:1234 \
  -v /path/on/host/models:/data/models \
  qwen36-vllm

# Single-GPU override
docker run -d --name qwen36 --gpus '"device=0"' -p 1234:1234 \
  -v /path/on/host/models:/data/models \
  qwen36-vllm

# Upgrade (no redownload needed)
docker stop qwen36 && docker rm qwen36
docker pull ghcr.io/tedivm/qwen36-27b-docker:latest
docker run -d --name qwen36 --gpus all -p 1234:1234 \
  -v /path/on/host/models:/data/models \
  qwen36-vllm
```

## Environment variables

All configuration is via environment variables with sensible defaults:

| Variable | Default | Description |
|---|---|---|
| `MODEL_DIR` | `/data/models` | Model weights path (mount from host) |
| `MODEL_REPO` | `Lorbus/Qwen3.6-27B-int4-AutoRound` | HuggingFace model repo |
| `PORT` | `1234` | API port |
| `SERVED_MODEL_NAME` | `qwen3.6-27b` | Model name for API |
| `MAX_MODEL_LEN` | `200000` | Max context length (auto-lowered to 48000 for single GPU) |
| `MAX_NUM_SEQS` | `3` | Concurrent sequences |
| `GPU_MEMORY_UTIL` | `0.92` | GPU memory fraction (auto-set to 0.95 for single GPU) |
| `TENSOR_PARALLEL` | *(auto)* | Tensor parallelism (auto-detected from GPU count) |
| `TEMPERATURE` | `0.6` | Generation temperature |
| `TOP_P` | `0.95` | Nucleus sampling threshold |
| `TOP_K` | `20` | Top-k sampling |
| `MIN_P` | `0.0` | Min-p sampling threshold |
| `PRESENCE_PENALTY` | `0` | Presence penalty |
| `REPETITION_PENALTY` | `1.0` | Repetition penalty |
| `REASONING_PARSER` | `qwen3` | Reasoning parser (blank to disable) |
| `HF_TOKEN` | *(empty)* | HuggingFace auth token for gated models |

## Usage

**Monitor** (inside the container):
```bash
docker exec -it qwen36 watch-vllm.py
docker exec -it qwen36 watch-vllm.py /data/models/.tmp/vllm.log 24
```

**Benchmark** (inside the container):
```bash
docker exec -it qwen36 python bench_tps.py
```

**OpenAI-compatible API**:
```bash
curl http://localhost:1234/v1/chat/completions \
  -H "Authorization: Bearer any" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-27b","messages":[{"role":"user","content":"hi"}]}'
```

## Server flags

| Flag | Why |
|---|---|
| `--quantization auto_round` | Matches the Lorbus weights |
| `--kv-cache-dtype fp8` | Halves KV memory vs FP16; 200K x 3 seqs fits on 48 GB |
| `--enable-prefix-caching` | Not default for Qwen3.6 hybrid attention; opt in |
| `--enable-chunked-prefill` | Recommended alongside spec-decode for throughput |
| `--speculative-config method=mtp, num_speculative_tokens=3` | ~2x throughput on code; 3 is the sweet spot |
| `--max-num-seqs 3` | Solo user + subagents; raise for more concurrency |
| `--max-num-batched-tokens 4128` | Matches vLLM's CUDA-graph compile range endpoint |
| `--gpu-memory-utilization 0.92` | Leaves CUDA-graph margin |
| `--disable-custom-all-reduce` | No NVLink — stock NCCL is faster |
| `--tool-call-parser qwen3_coder` + `--enable-auto-tool-choice` | OpenAI-style tool calls |
| `--reasoning-parser qwen3` | Enables extended thinking output |

TP=2 beats TP=1 by ~1.5x on dual 3090s. Memory-bandwidth savings from splitting weights across two cards outweigh the PCIe NCCL all-reduce cost.

## Caveats

- **Mamba prefix caching is experimental** for Qwen3.6. vLLM auto-picks the `align` fallback mode for `Qwen3_5ForConditionalGeneration`. Regular-attention layers cache fine (~85% hit rate); Mamba/GDN linear-attention layers re-run prefill on every new request.
- **Spec-decode silently ignores** `min_p` and `logit_bias` per-request params.
- **Deprecation warnings about `Qwen2VLImageProcessorFast` / `use_fast`** are upstream-transformers noise; ignore.
- **CUDA graph mode downgrades** to `PIECEWISE` under spec-decode (FlashInfer limitation) — automatic and expected.

## Acknowledgments

- [k0zakinio/qwen36-vllm-setup](https://github.com/k0zakinio/qwen36-vllm-setup) — the original repo this was forked from; all of the vLLM flag tuning, performance benchmarks, and serve scripts originated there.
- [Lorbus](https://huggingface.co/Lorbus) — AutoRound INT4 quant that preserves the MTP head in BF16 and keeps MoonViT in FP16.
- [Qwen team](https://github.com/QwenLM/Qwen3.6) — the base model and the MTP head.
- Medium article ["An Overnight Stack for Qwen3.6-27B"](https://medium.com/@fzbcwvv/an-overnight-stack-for-qwen3-6-27b-85-tps-125k-context-vision-on-one-rtx-3090-0d95c6291914?postPublishedType=repub) — original source of the AutoRound + MTP + TurboQuant stack.
- [Sandermage's Genesis patches](https://github.com/Sandermage/genesis-vllm-patches) — more aggressive approach with TurboQuant KV; useful reference for pushing further.
