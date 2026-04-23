# Qwen3.6-27B vLLM Setup — AutoRound INT4 + MTP Speculative Decoding

Fast local inference for [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) on dual 24 GB Ampere GPUs. Uses the [Lorbus AutoRound INT4 quant](https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound) (which retains the MTP draft head and MoonViT vision encoder) with vLLM speculative decoding.

## What you get

| Metric (dual RTX 3090, TP=2, 200K context) | Value |
|---|---|
| Sustained TPS on coding workloads | **~80–105** |
| Sustained TPS on prose | ~60 |
| MTP per-position acceptance (code) | 94% / 82% / 72% |
| Mean accepted tokens per spec cycle (code) | 3.48 / 4 |
| Regular-attention prefix cache hit rate (agent workloads) | ~85% |
| Max context length | 200,000 tokens (172K KV pool headroom) |
| Vision support | yes (MoonViT, via `image_url` content parts) |

Dense-model, not MoE — no tensor shuffling at token boundaries, clean TP=2 split. No NVLink required; PCIe TP is fine on this workload.

## Hardware

Tested on 2× RTX 3090 (48 GB VRAM total). Also works at lower context on a single 24 GB card via `serve-tp1.sh` (~48K max, FP8 KV caps out around 57K on one Ampere card).

Minimum disk: ~20 GB for weights + ~6 GB for the Python venv + CUDA wheels.

## Setup

```bash
git clone <this-repo> qwen36-vllm-setup
cd qwen36-vllm-setup
cp config.env.example config.env
$EDITOR config.env          # set MODEL_DIR, VENV_DIR, LOG_DIR
./setup.sh                  # creates venv, installs vLLM + auto-round, downloads weights
./serve.sh                  # starts the server on the configured port
```

Passwordless `sudo -n nvidia-smi` is optional — the script uses it to apply power/clock limits per launch, and soft-fails if credentials aren't cached. Skip this section by editing the loop out of `serve.sh` if you prefer.

## Usage

**Live stats** (rolling window of spec-decode acceptance, TPS, prefix-cache hit rate, KV usage):
```bash
./watch-vllm.py                        # default log path and 12-sample window
./watch-vllm.py "$LOG_DIR/vllm.log"    # explicit log path
./watch-vllm.py "$LOG_DIR/vllm.log" 24 # ~4 min rolling window
```

**Benchmark** (prose vs code A/B, reports TPS + acceptance):
```bash
python bench_tps.py http://localhost:1234/v1 qwen3.6-27b
```

**OpenAI-compatible API**:
```bash
curl http://localhost:1234/v1/chat/completions \
  -H "Authorization: Bearer any" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-27b","messages":[{"role":"user","content":"hi"}]}'
```

## Why these flags

| Flag | Why |
|---|---|
| `--quantization auto_round` | Matches the Lorbus weights |
| `--kv-cache-dtype fp8` | Halves KV memory vs FP16; 200K × 3 seqs fits on 48 GB |
| `--enable-prefix-caching` | Not default for Qwen3.6 hybrid attention; opt in |
| `--enable-chunked-prefill` | Recommended alongside spec-decode for throughput |
| `--speculative-config method=mtp, num_speculative_tokens=3` | ~2× throughput on code; 3 is the sweet spot |
| `--max-num-seqs 3` | Solo user + a couple of subagents; raise if you need more concurrency |
| `--max-num-batched-tokens 4128` | Matches vLLM's CUDA-graph compile range endpoint |
| `--gpu-memory-utilization 0.92` | Leaves CUDA-graph margin; vLLM suggests 0.93 with `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1` |
| `--disable-custom-all-reduce` | No NVLink — stock NCCL is faster without the custom path |
| `--tool-call-parser qwen3_coder` + `--enable-auto-tool-choice` | OpenAI-style tool calls |
| `--chat-template <patched>` | Template has `{%- set enable_thinking = false %}` prepended so thinking is hardcoded off; clients can't re-enable |
| `--override-generation-config` | Unsloth non-thinking defaults (`temp=0.7, top_p=0.8, top_k=20, presence_penalty=1.5, repetition_penalty=1.0`); per-request overrides still work |

Counter-intuitively, TP=2 beats TP=1 by ~1.5× on this hardware. Memory-bandwidth savings from splitting weights across two cards outweigh the PCIe NCCL all-reduce cost, even with `NCCL_P2P_DISABLE=1`.

## Caveats worth knowing

- **Thinking is permanently off** at the inference layer (template-level). To re-enable, delete line 1 of `chat_template.no_thinking.jinja` *and* re-add `--reasoning-parser qwen3` to the serve script.
- **Mamba prefix caching is experimental** for Qwen3.6. vLLM auto-picks the `align` fallback mode (weaker than `all`) for `Qwen3_5ForConditionalGeneration`. Regular-attention layers cache fine (~85% hit rate on agent workloads); Mamba/GDN linear-attention layers re-run prefill on every new request. This is why you'll see high `prompt_tps` bursts even when `Prefix cache hit rate` is showing 85%+ — that's Mamba re-prefill, not attention re-prefill. You can try `--mamba-cache-mode all` to override (more memory, not vLLM-default for this arch).
- **Spec-decode silently ignores** `min_p` and `logit_bias` per-request params. Our `--override-generation-config` sets `min_p=0.0` so this is a no-op server-side.
- **Deprecation warnings about `Qwen2VLImageProcessorFast` / `use_fast`** are upstream-transformers noise; ignore.
- **CUDA graph mode downgrades** to `PIECEWISE` under spec-decode (FlashInfer limitation) — this is automatic and expected.
- **Port conflict:** `serve.sh` `fuser -k`'s the configured port on start. Close LM Studio's local server first or set a different port in `config.env`.

## Acknowledgments

- [Lorbus](https://huggingface.co/Lorbus) — AutoRound INT4 quant that preserves the MTP head in BF16 and keeps MoonViT in FP16.
- [Qwen team](https://github.com/QwenLM/Qwen3.6) — the base model and the MTP head.
- Medium article ["An Overnight Stack for Qwen3.6-27B"](https://medium.com/@fzbcwvv/an-overnight-stack-for-qwen3-6-27b-85-tps-125k-context-vision-on-one-rtx-3090-0d95c6291914?postPublishedType=repub) — original source of the AutoRound + MTP + TurboQuant stack. This repo is the subset that works with stock vLLM 0.19.1 (no TurboQuant, no nightly, no custom patches).
- [Sandermage's Genesis patches](https://github.com/Sandermage/genesis-vllm-patches) — more aggressive approach with TurboQuant KV; not required here but useful reference for pushing further.
