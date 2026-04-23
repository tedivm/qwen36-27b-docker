#!/usr/bin/env python3
"""TPS + spec-decode acceptance A/B. Runs a prose prompt and a coding prompt
back-to-back against the same server, reports end-to-end TPS, spec-decode
acceptance (from /metrics), and a snippet of each response.

Code prompts see materially higher MTP acceptance (>80%) than prose (~40%)
because repeated syntax / identifiers are what the draft head is trained to
predict well. Use this to validate both perf and correctness after launch.

  Usage: python bench_tps.py <base_url> <model>
  e.g.   python bench_tps.py http://localhost:1234/v1 qwen3.6-27b
"""
import json
import re
import sys
import time
import urllib.request

base_url = sys.argv[1].rstrip("/")
model = sys.argv[2]
base_root = base_url[:-3] if base_url.endswith("/v1") else base_url

prompts = {
    "story": (
        "Write a detailed fictional story set on a Martian colony about a botanist who "
        "discovers that the genetically-modified wheat is developing a rudimentary form "
        "of communication. Keep it coherent. At least 800 words."
    ),
    "code": (
        "Write a complete, production-quality Python implementation of an LRU cache "
        "with the following: (1) class `LRUCache` with O(1) get/put, capacity in "
        "constructor, (2) full type annotations on every method, (3) thread-safe via "
        "a single lock, (4) __repr__ that shows current size/capacity/hit-miss stats, "
        "(5) a `@cached(capacity=N)` decorator that wraps a function with its own "
        "LRUCache keyed by args, (6) tests using pytest covering: basic get/put, "
        "eviction order, thread-safety via concurrent.futures, the decorator, and "
        "edge cases (capacity=1, capacity=0, None values). Include docstrings and "
        "runtime-complexity notes. Output nothing but the code — no explanation."
    ),
}


def scrape_spec(metrics_text: str) -> dict:
    """Best-effort parse of vLLM prometheus spec-decode counters.
    Names vary across vLLM versions — we read whatever's present.
    """
    pats = {
        "drafts":       r"^vllm:spec_decode_num_drafts\{[^}]*\}\s+([\d.]+)",
        "drafted_toks": r"^vllm:spec_decode_num_draft_tokens\{[^}]*\}\s+([\d.]+)",
        "accepted":     r"^vllm:spec_decode_num_accepted_tokens\{[^}]*\}\s+([\d.]+)",
    }
    out = {}
    for k, p in pats.items():
        m = re.search(p, metrics_text, re.MULTILINE)
        if m:
            out[k] = float(m.group(1))
    return out


def get_metrics() -> dict:
    try:
        with urllib.request.urlopen(f"{base_root}/metrics", timeout=5) as r:
            return scrape_spec(r.read().decode())
    except Exception:
        return {}


def run(label: str, prompt: str, max_tokens: int = 800):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": False,
    }).encode()
    pre = get_metrics()
    t0 = time.time()
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer any"},
    )
    with urllib.request.urlopen(req, timeout=900) as resp:
        data = json.loads(resp.read())
    elapsed = time.time() - t0
    post = get_metrics()

    u = data.get("usage", {})
    comp = u.get("completion_tokens", 0)
    tps = comp / elapsed if elapsed > 0 else 0
    ddrafts = post.get("drafts", 0) - pre.get("drafts", 0)
    ddrafted = post.get("drafted_toks", 0) - pre.get("drafted_toks", 0)
    daccepted = post.get("accepted", 0) - pre.get("accepted", 0)
    accept_rate = daccepted / ddrafted if ddrafted else 0
    mean_acc_len = daccepted / ddrafts if ddrafts else 0

    print(f"=== {label} ===")
    print(f"  gen_toks       : {comp}")
    print(f"  elapsed        : {elapsed:.2f} s")
    print(f"  TPS            : {tps:.2f}")
    if ddrafts:
        print(f"  spec cycles    : {int(ddrafts)}")
        print(f"  drafted toks   : {int(ddrafted)}")
        print(f"  accepted toks  : {int(daccepted)}")
        print(f"  draft accept % : {accept_rate*100:.1f}%")
        print(f"  mean accept len: {mean_acc_len:.2f}")
    else:
        print("  (no spec-decode metric deltas — may be running without MTP, or")
        print("   /metrics uses different counter names in this vLLM version)")
    print(f"  first 120 chars: {data['choices'][0]['message']['content'][:120]!r}")
    print()


if __name__ == "__main__":
    run("STORY (prose baseline)", prompts["story"])
    run("CODE  (coding workload)", prompts["code"], max_tokens=1200)
