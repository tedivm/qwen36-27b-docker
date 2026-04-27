#!/usr/bin/env python3
"""Live monitor for the vLLM server.

Tails the server log and shows a rolling window of spec-decode acceptance,
generation TPS, prefix cache hit rate, and KV usage. Stdlib-only — run with
system python3, no venv needed.

  Usage: ./watch-vllm.py [LOG [WINDOW_SAMPLES]]

   LOG             path to the server log (defaults to /data/logs/vllm.log)
  WINDOW_SAMPLES  rolling window size in samples (default: 12 = ~2 min)
"""
import collections
import os
import re
import shutil
import subprocess
import sys
import time


def default_log_path() -> str:
    model_dir = os.environ.get("MODEL_DIR", "/data/models")
    return os.path.join(model_dir, ".tmp", "vllm.log")


LOG = sys.argv[1] if len(sys.argv) > 1 else default_log_path()
N = int(sys.argv[2]) if len(sys.argv) > 2 else 12

SPEC_RE = re.compile(
    r"Mean acceptance length: (?P<mean>[\d.]+).*"
    r"Per-position acceptance rate: (?P<p1>[\d.]+), (?P<p2>[\d.]+), (?P<p3>[\d.]+).*"
    r"Avg Draft acceptance rate: (?P<rate>[\d.]+)%"
)
TP_RE = re.compile(
    r"Avg prompt throughput: (?P<prompt>[\d.]+) tokens/s.*"
    r"Avg generation throughput: (?P<gen>[\d.]+) tokens/s.*"
    r"Running: (?P<running>\d+) reqs.*"
    r"GPU KV cache usage: (?P<kv>[\d.]+)%.*"
    r"Prefix cache hit rate: (?P<pcache>[\d.]+)%"
)
TS_RE = re.compile(r"(\d{2}:\d{2}:\d{2})")

spec_buf = collections.deque(maxlen=N)
tp_buf = collections.deque(maxlen=N)


def color(s, code):
    return f"\x1b[{code}m{s}\x1b[0m"


def draw():
    shutil.get_terminal_size((120, 40))
    parts = ["\x1b[H\x1b[J"]
    hdr = f"vLLM live monitor — rolling {N} samples ({N*10}s)   now: {time.strftime('%H:%M:%S')}   Ctrl-C to quit"
    parts.append(color(hdr, "1;36") + "\n")
    parts.append(color(f"log: {LOG}", "2") + "\n\n")

    parts.append(color("── Spec-decode ────────────────────────────────────────────", "1") + "\n")
    parts.append(f"{'time':>8}  {'mean':>5}  {'pos1':>5}  {'pos2':>5}  {'pos3':>5}  {'accept%':>7}\n")
    for t, mean, p1, p2, p3, rate in spec_buf:
        mean_c = "32" if mean >= 3.2 else "33" if mean >= 2.6 else "31"
        rate_c = "32" if rate >= 80 else "33" if rate >= 60 else "31"
        parts.append(
            f"{t:>8}  {color(f'{mean:>5.2f}', mean_c)}  "
            f"{p1:>5.2f}  {p2:>5.2f}  {p3:>5.2f}  "
            f"{color(f'{rate:>6.1f}%', rate_c)}\n"
        )
    if spec_buf:
        n = len(spec_buf)
        avg = lambda i: sum(r[i] for r in spec_buf) / n
        parts.append(color(
            f"{'avg':>8}  {avg(1):>5.2f}  {avg(2):>5.2f}  {avg(3):>5.2f}  {avg(4):>5.2f}  {avg(5):>6.1f}%\n",
            "1;36",
        ))
    else:
        parts.append(color("  (waiting for spec-decode metrics…)\n", "2"))

    parts.append("\n" + color("── Throughput / cache ─────────────────────────────────────", "1") + "\n")
    parts.append(f"{'time':>8}  {'gen_tps':>7}  {'prompt_tps':>10}  {'seqs':>4}  {'kv%':>5}  {'prefix%':>7}\n")
    for t, gen, prompt, running, kv, pc in tp_buf:
        gen_c = "32" if gen >= 60 else "33" if gen >= 20 else "31" if gen > 0 else "2"
        kv_c = "31" if kv >= 80 else "33" if kv >= 50 else "32"
        parts.append(
            f"{t:>8}  {color(f'{gen:>7.1f}', gen_c)}  {prompt:>10.1f}  "
            f"{running:>4d}  {color(f'{kv:>4.1f}%', kv_c)}  {pc:>6.1f}%\n"
        )
    if tp_buf:
        gens = [r[1] for r in tp_buf]
        active = [g for g in gens if g > 1.0]
        peak = max(gens)
        avg_active = sum(active) / len(active) if active else 0.0
        parts.append(color(
            f"\npeak gen_tps: {peak:.1f}   avg gen_tps (active windows): {avg_active:.1f}   "
            f"samples: {len(tp_buf)}\n",
            "1;36",
        ))
    else:
        parts.append(color("  (waiting for throughput metrics…)\n", "2"))

    sys.stdout.write("".join(parts))
    sys.stdout.flush()


def main():
    if not os.path.isfile(LOG):
        sys.stderr.write(f"Log file not found: {LOG}\n")
        sys.exit(1)
    p = subprocess.Popen(
        ["tail", "-n", "0", "-F", LOG],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    draw()
    try:
        for line in p.stdout:
            ts_m = TS_RE.search(line)
            ts = ts_m.group(1) if ts_m else "??:??:??"
            m = SPEC_RE.search(line)
            if m:
                spec_buf.append((
                    ts,
                    float(m.group("mean")),
                    float(m.group("p1")),
                    float(m.group("p2")),
                    float(m.group("p3")),
                    float(m.group("rate")),
                ))
                draw()
                continue
            m = TP_RE.search(line)
            if m:
                tp_buf.append((
                    ts,
                    float(m.group("gen")),
                    float(m.group("prompt")),
                    int(m.group("running")),
                    float(m.group("kv")),
                    float(m.group("pcache")),
                ))
                draw()
    except KeyboardInterrupt:
        pass
    finally:
        p.terminate()
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
