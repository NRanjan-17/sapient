#!/usr/bin/env python3
"""Layer D: the 'language-could-matter share' of each workload.

For Whisper, Kokoro, SmolVLM and serve there is no C++ arm, so a wall-clock race
against SAPIENT-in-C++ is not available. Racing SAPIENT against itself tells you
nothing about the language. What IS informative, and cheap, is where the time
goes: the fraction of leaf samples sitting in hand-written SIMD kernels (which
this study measured as compiling to materially the same work in both languages,
and which showed 1.00x for Q6_K) versus scalar loops, the allocator and memmove
(where a port could plausibly differ).

That fraction is a CEILING on what a port could win in the workload -- the same
logic the prior study applied to LLM decode, generalised to the other modalities.

It is a ceiling, not an estimate: the end-to-end result (SS7) shows even the SIMD
share is not perfectly language-invariant, so the true figure sits between.
"""
import argparse, json, pathlib, re, subprocess, sys, time

# Frames where the thread is NOT executing -- parked workers, kernel waits.
# `sample` samples every thread including idle ones, so leaving these in the
# denominator would measure how many threads the pool keeps parked rather than
# where the work goes. They are reported separately and excluded from the share.
IDLE = re.compile(
    r"__psynch_cvwait|__psynch_mutexwait|swtch_pri|thread_switch|__workq_kernreturn|"
    r"kevent|mach_msg|__semwait_signal|nanosleep|__select|poll|usleep|"
    r"start_wqthread|_pthread_wqthread|__ulock_wait|read$|__read", re.I)

# Leaf-symbol buckets. Order matters: first match wins.
BUCKETS = [
    ("simd_kernel", re.compile(
        r"dot_q\d|_neon|_sdot|_smmla|matmul_nt|sgemm|gemm|flash_attn|"
        r"scaled_dot_product|conv2d|im2col|matrixmultiply", re.I)),
    ("allocator_memmove", re.compile(
        r"malloc|free|realloc|calloc|memmove|memcpy|memset|operator new|"
        r"_platform_mem|szone|nanov2|tiny_malloc", re.I)),
    ("scalar_loop", re.compile(
        r"silu|gelu|softmax|rms_norm|layer_norm|apply_rope|quantize_row|"
        r"unary_f32|binary_f32|snake|lstm|istft|stft|mel|resample|to_f32", re.I)),
    ("threading_sync", re.compile(
        r"rayon|spinpool|parking_lot|pthread|mutex|condvar|semaphore|"
        r"psynch|kevent|ulock", re.I)),
]

def classify(sym):
    if IDLE.search(sym):
        return "idle_blocked"
    for name, rx in BUCKETS:
        if rx.search(sym):
            return name
    return "other"

def leaf_histogram(sample_text):
    """`sample` prints a call tree whose depth is encoded in a prefix of spaces
    and the tree characters '+', '!', '|', ':' -- e.g.

        2060 Thread_13669943   DispatchQueue_1: com.apple.main-thread
        + 2060 start  (in dyld) + 6992  [0x18a7344e4]
        +   2060 main  (in sapient) + 52  [0x102d75a78]

    so the prefix WIDTH is the depth. A frame is a LEAF when the following line
    is not deeper, meaning its samples are time executing there rather than in a
    callee. Counting non-leaf frames would double-count every ancestor."""
    line_re = re.compile(r"^([\s+!|:]*?)(\d+)\s+(.+?)(?:\s+\(in\s|\s+\[0x|\s*$)")
    rows = []
    for ln in sample_text.splitlines():
        m = line_re.match(ln)
        if m:
            sym = m.group(3).strip()
            if sym.startswith("Thread_"):      # thread headers are not frames
                continue
            rows.append((len(m.group(1)), int(m.group(2)), sym))
    hist = {}
    for i, (indent, count, sym) in enumerate(rows):
        deeper = rows[i + 1][0] > indent if i + 1 < len(rows) else False
        if not deeper:                       # leaf frame
            hist[sym] = hist.get(sym, 0) + count
    return hist

def profile(cmd, seconds, settle):
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(settle)
    if proc.poll() is not None:
        raise RuntimeError(f"workload exited before sampling: {' '.join(cmd)}")
    out = pathlib.Path(f"/tmp/sample_{proc.pid}.txt")
    subprocess.run(["sample", str(proc.pid), str(seconds), "-f", str(out)],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try: proc.terminate(); proc.wait(timeout=20)
    except Exception: proc.kill()
    return leaf_histogram(out.read_text(errors="ignore"))

def report(label, hist):
    agg = {}
    for sym, n in hist.items():
        agg[classify(sym)] = agg.get(classify(sym), 0) + n
    idle = agg.pop("idle_blocked", 0)
    on_cpu = sum(agg.values()) or 1          # denominator: executing samples only
    simd = agg.get("simd_kernel", 0)
    matters = agg.get("scalar_loop", 0) + agg.get("allocator_memmove", 0)
    print(f"\n### {label}")
    print(f"    on-CPU leaf samples {on_cpu}  (plus {idle} parked/blocked, excluded)")
    for k in ["simd_kernel", "scalar_loop", "allocator_memmove", "threading_sync", "other"]:
        v = agg.get(k, 0)
        print(f"    {k:<22} {v:>7}  {100*v/on_cpu:5.1f}%")
    print(f"    {'-'*44}")
    print(f"    {'SIMD (lang-invariant)':<22} {simd:>7}  {100*simd/on_cpu:5.1f}%")
    print(f"    {'CEILING for a port':<22} {matters:>7}  {100*matters/on_cpu:5.1f}%")
    work = {s: n for s, n in hist.items() if classify(s) != "idle_blocked"}
    top = sorted(work.items(), key=lambda kv: -kv[1])[:5]
    print("    top on-CPU leaves: "
          + "; ".join(f"{s[:44]} {100*n/on_cpu:.1f}%" for s, n in top))
    return {"on_cpu_leaf_samples": on_cpu, "idle_samples": idle, "buckets": agg,
            "simd_share": simd/on_cpu, "port_ceiling_share": matters/on_cpu,
            "top_leaves": [{"symbol": s, "share": n/on_cpu} for s, n in top]}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="target/release/sapient")
    ap.add_argument("--seconds", type=int, default=8)
    ap.add_argument("--out", default="benchmarks/lang-comparison/results/raw/profile_share.json")
    a = ap.parse_args()
    B = a.bin
    jobs = [
        ("LLM decode (qwen2.5-1.5b-q4)",
         [B, "bench-llm", "openhorizon/qwen2.5-1.5b-q4", "--backend", "cpu",
          "--runs", "4", "--prompt", "Write a long detailed essay about the Roman Empire."], 6),
        # The audio/vision workloads are seconds long, so each is given enough
        # work to outlive the sampling window: 120 s of audio (four Whisper
        # windows), a long TTS paragraph, and a prompt that elicits a full
        # vision answer rather than a one-word one.
        ("Whisper-base transcription",
         [B, "transcribe", "openhorizon/whisper-base", "/tmp/long.wav",
          "--backend", "cpu"], 2.0),
        ("Kokoro-82M TTS",
         [B, "speak", "kokoro-82m",
          "The quick brown fox jumps over the lazy dog. " * 12,
          "-o", "/tmp/k.wav", "--no-play"], 2.0),
        ("SmolVLM-256M vision",
         [B, "see", "/tmp/probe.png", "-p",
          "Describe this image in as much detail as you possibly can.",
          "-m", "smolvlm-256m", "--max-tokens", "192"], 1.0),
    ]
    res = {}
    for label, cmd, settle in jobs:
        try:
            res[label] = report(label, profile(cmd, a.seconds, settle))
        except Exception as e:
            print(f"\n### {label}\n    SKIPPED: {e}")
    pathlib.Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"\nwrote {a.out}")
