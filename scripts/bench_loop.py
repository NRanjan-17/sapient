#!/usr/bin/env python3
"""Reproducible baseline benchmark for the optimisation loop (LOOP_LOG.md / PAPER.md).

Measures, on the machine it runs on, and writes one JSON file:

  LLM     decode tok/s, TTFT, peak RSS      `sapient bench-llm --json` (decode-only)
          + llama.cpp `llama-bench` on the same GGUF when both are available
  Vision  image-encode latency: p50 / max / stdev over N runs, per-stage split
          (`SAPIENT_VISION_TIMING=1 sapient see`)
  Quality perplexity on wikitext-2 (llama.cpp protocol)   `sapient eval-ppl` + `llama-perplexity`
  Size    binary size, version, git revision

Nothing here is tuned per run: same prompt, same deterministic test image
(generated in-process), fixed token counts. Compare JSON files across commits.

Usage:
  python3 scripts/bench_loop.py                       # uses target/release/sapient
  python3 scripts/bench_loop.py --sapient /path/to/sapient --out benchmarks/x.json
  python3 scripts/bench_loop.py --skip-llm            # vision + size only
  python3 scripts/bench_loop.py --skip-ppl            # skip quality/perplexity section
"""
import argparse
import datetime
import glob
import json
import os
import platform
import re
import shutil
import statistics
import struct
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
import zlib

PROMPT = ("Write a detailed 1000-word essay on how neural networks learn through "
          "backpropagation, including the role of gradients and the chain rule.")
LLM_MODELS = ["openhorizon/qwen2.5-1.5b-q4", "openhorizon/llama-3.2-1b-q4"]
VLM_MODEL = "smolvlm-256m"


def sh(cmd, env=None, timeout=3600):
    e = dict(os.environ)
    if env:
        e.update(env)
    r = subprocess.run(cmd, capture_output=True, text=True, env=e, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


def write_test_png(path, size=640):
    """Deterministic RGB test image (gradient + solid block), stdlib only."""
    rows = bytearray()
    for y in range(size):
        rows.append(0)  # filter: none
        for x in range(size):
            if 200 <= y < 440 and 180 <= x < 460:
                rows += bytes((217, 26, 26))
            else:
                rows += bytes((x * 255 // size, y * 255 // size, (x * 7 + y * 13) % 256))

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(bytes(rows), 6))
    png += chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)


def perf_threads():
    """Performance-core count (llama.cpp's best setting on P/E-core Macs)."""
    if sys.platform == "darwin":
        rc, out, _ = sh(["sysctl", "-n", "hw.perflevel0.physicalcpu"])
        if rc == 0 and out.strip().isdigit():
            return int(out.strip())
    return os.cpu_count() or 1


def find_cached_gguf(repo_hint):
    """Locate the GGUF `sapient` downloaded for an alias, for llama-bench."""
    hub = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    hits = [p for p in glob.glob(os.path.join(hub, "models--*", "snapshots", "*", "*.gguf"))
            if repo_hint.lower() in p.lower()]
    return max(hits, key=os.path.getsize) if hits else None


def bench_llm(sapient, model, runs, tokens, backend):
    rc, out, err = sh([sapient, "bench-llm", model, "-p", PROMPT, "--max-tokens", str(tokens),
                       "--runs", str(runs), "--backend", backend, "--json"])
    if rc != 0:
        return {"error": (err or out).strip().splitlines()[-1][:300] if (err or out).strip() else "failed"}
    d = json.loads(out)
    tps = [r["tps"] for r in d["runs"]]
    return {
        "decode_tps_mean": d["summary"]["decode_tps"],
        "decode_tps_runs": tps,
        "decode_tps_stdev": round(statistics.pstdev(tps), 2) if len(tps) > 1 else 0.0,
        "ttft_ms_runs": [r["ttft_ms"] for r in d["runs"]],
        "ttft_ms_mean": d["summary"]["mean_ttft_ms"],
        "peak_rss_mb": d["summary"]["peak_rss_mb"],
        "tokens_per_run": [r["total_tokens"] for r in d["runs"]],
        "threads": d.get("threads"),
        "load_time_ms": d.get("load_time_ms"),
        "method": d.get("method"),
    }


def bench_llama_cpp(gguf, threads, tokens):
    if not shutil.which("llama-bench") or not gguf:
        return {"skipped": "llama-bench or GGUF not found"}
    rc, out, _ = sh(["llama-bench", "-m", gguf, "-t", str(threads), "-ngl", "0",
                     "-p", "0", "-n", str(tokens), "-r", "3", "-o", "json"])
    if rc != 0:
        return {"error": "llama-bench failed"}
    d = json.loads(out)[0]
    ver = verr = ""
    if shutil.which("llama-cli"):
        _, ver, verr = sh(["llama-cli", "--version"])
    m = re.search(r"version: (.+)", ver + verr)
    return {"decode_tps_mean": round(d["avg_ts"], 2), "decode_tps_stdev": round(d["stddev_ts"], 2),
            "threads": threads, "gguf": os.path.basename(gguf), "build": m.group(1).strip() if m else None}


def wikitext_path():
    """Download and cache wikitext-2 test set; return path or None on failure."""
    cache_dir = os.path.expanduser("~/.cache/sapient-bench")
    target = os.path.join(cache_dir, "wikitext-2-raw", "wiki.test.raw")

    if os.path.exists(target):
        return target

    try:
        os.makedirs(cache_dir, exist_ok=True)
        zip_path = os.path.join(cache_dir, "wikitext-2-raw-v1.zip")
        url = "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"
        urllib.request.urlretrieve(url, zip_path)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(cache_dir)

        os.remove(zip_path)

        if os.path.exists(target):
            return target
        return None
    except Exception:
        return None


def bench_ppl(sapient, model, text, chunks, backend):
    """Run sapient eval-ppl and return dict with perplexity metrics."""
    rc, out, err = sh([sapient, "eval-ppl", model, "--file", text, "--ctx", "512",
                       "--chunks", str(chunks), "--backend", backend, "--json"])
    if rc != 0:
        return {"error": (err or out).strip().splitlines()[-1][:300] if (err or out).strip() else "failed"}
    try:
        d = json.loads(out)
        return {
            "perplexity": d.get("perplexity"),
            "perplexity_stderr": d.get("perplexity_stderr"),
            "tokens_scored": d.get("tokens_scored"),
            "chunks": d.get("chunks"),
            "ctx": d.get("ctx"),
            "seconds": d.get("seconds"),
            "method": d.get("method"),
        }
    except Exception:
        return {"error": "JSON parse failed"}


def bench_llama_ppl(gguf, text, chunks, threads):
    """Run llama-perplexity and parse the PPL estimate line; return dict."""
    if not shutil.which("llama-perplexity") or not gguf:
        return {"skipped": "llama-perplexity or GGUF not found"}

    rc, out, err = sh(["llama-perplexity", "-m", gguf, "-f", text, "-c", "512",
                       "--chunks", str(chunks), "-ngl", "0", "-t", str(threads)])
    if rc != 0:
        return {"error": "llama-perplexity failed"}

    # Parse "Final estimate: PPL = X +/- Y" line
    text_combined = out + err
    m = re.search(r"Final estimate:\s+PPL\s*=\s*([\d.]+)\s*\+/-\s*([\d.]+)", text_combined)
    if m:
        return {
            "perplexity": float(m.group(1)),
            "perplexity_stderr": float(m.group(2)),
            "chunks": chunks,
            "threads": threads,
        }
    return {"error": "no Final estimate line"}


VISION_RE = re.compile(r"vision (\d+) ms · prefill (\d+) ms")
STAGE_RE = re.compile(r"\[vision\] (\d+) patches · (.*) ms")


def bench_vision(sapient, image, runs):
    enc, pre, stages = [], [], []
    for _ in range(runs + 1):  # first run is a warm-up (page cache), discarded
        rc, out, err = sh([sapient, "see", image, "-p", "Describe.", "--model", VLM_MODEL,
                           "--max-tokens", "8"], env={"SAPIENT_VISION_TIMING": "1"})
        text = out + err
        m = VISION_RE.search(text)
        if rc != 0 or not m:
            return {"error": text.strip().splitlines()[-1][:300] if text.strip() else "failed"}
        enc.append(int(m.group(1)))
        pre.append(int(m.group(2)))
        s = STAGE_RE.search(text)
        if s:
            stages.append({k: int(v) for k, v in
                           (part.rsplit(" ", 1) for part in s.group(2).split(" · "))})
    enc, pre, stages = enc[1:], pre[1:], stages[1:]
    res = {
        "model": VLM_MODEL, "runs": runs,
        "encode_ms_runs": enc,
        "encode_ms_p50": statistics.median(enc),
        "encode_ms_max": max(enc),
        "encode_ms_min": min(enc),
        "encode_ms_stdev": round(statistics.pstdev(enc), 1),
        "prefill_ms_p50": statistics.median(pre),
    }
    if stages:
        res["stage_ms_p50"] = {k: statistics.median(s[k] for s in stages) for k in stages[0]}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sapient", default="target/release/sapient")
    ap.add_argument("--out", default=None)
    ap.add_argument("--runs", type=int, default=3, help="measured LLM runs (bench-llm adds 1 warm-up)")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--vision-runs", type=int, default=10)
    ap.add_argument("--ppl-chunks", type=int, default=20, help="512-token wikitext-2 chunks to score; 0 disables the quality section")
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--skip-vision", action="store_true")
    ap.add_argument("--skip-ppl", action="store_true")
    a = ap.parse_args()

    sapient = os.path.abspath(a.sapient)
    if not os.path.exists(sapient):
        sys.exit(f"no binary at {sapient} — build with: cargo build --release -p sapient-cli")
    rc, ver, _ = sh([sapient, "--version"])
    rcg, rev, _ = sh(["git", "rev-parse", "--short", "HEAD"])
    rcd, dirty, _ = sh(["git", "status", "--porcelain", "--untracked-files=no"])
    res = {
        "date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "machine": {"platform": platform.platform(), "machine": platform.machine(),
                    "cpu_count": os.cpu_count(), "perf_threads": perf_threads()},
        "sapient": {"version": ver.strip(), "git": rev.strip(), "dirty": bool(dirty.strip()),
                    "binary_bytes": os.path.getsize(sapient), "backend": a.backend},
        "llm": {}, "quality": {}, "vision": None,
    }
    if sys.platform == "darwin":
        rcb, brand, _ = sh(["sysctl", "-n", "machdep.cpu.brand_string"])
        res["machine"]["cpu"] = brand.strip()

    if not a.skip_llm:
        for model in LLM_MODELS:
            print(f"[llm] {model} …", file=sys.stderr)
            entry = {"sapient": bench_llm(sapient, model, a.runs, a.tokens, a.backend)}
            # e.g. "openhorizon/qwen2.5-1.5b-q4" → cached repo dir containing "qwen2.5-1.5b"
            gguf = find_cached_gguf(model.split("/")[1].removesuffix("-q4"))
            entry["llama_cpp"] = bench_llama_cpp(gguf, perf_threads(), a.tokens)
            s, l = entry["sapient"], entry["llama_cpp"]
            if "decode_tps_mean" in s and "decode_tps_mean" in l and s["decode_tps_mean"]:
                entry["llama_cpp_over_sapient"] = round(l["decode_tps_mean"] / s["decode_tps_mean"], 3)
            res["llm"][model] = entry

    if not a.skip_ppl and a.ppl_chunks > 0:
        text = wikitext_path()
        if text is None:
            res["quality"] = {"skipped": "wikitext-2 not available"}
        else:
            for model in LLM_MODELS:
                print(f"[quality] {model} …", file=sys.stderr)
                entry = {"sapient": bench_ppl(sapient, model, text, a.ppl_chunks, a.backend)}
                # e.g. "openhorizon/qwen2.5-1.5b-q4" → cached repo dir containing "qwen2.5-1.5b"
                gguf = find_cached_gguf(model.split("/")[1].removesuffix("-q4"))
                entry["llama_cpp"] = bench_llama_ppl(gguf, text, a.ppl_chunks, perf_threads())
                s, l = entry["sapient"], entry["llama_cpp"]
                if "perplexity" in s and "perplexity" in l and s["perplexity"] and l["perplexity"]:
                    entry["sapient_over_llama_cpp"] = round(s["perplexity"] / l["perplexity"], 4)
                res["quality"][model] = entry

    if not a.skip_vision:
        print("[vision] smolvlm-256m …", file=sys.stderr)
        with tempfile.TemporaryDirectory() as td:
            img = os.path.join(td, "bench_image.png")
            write_test_png(img)
            res["vision"] = bench_vision(sapient, img, a.vision_runs)

    out = a.out or os.path.join("benchmarks", "{}-{}.json".format(
        datetime.date.today().isoformat(), platform.node().split(".")[0] or "host"))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
        f.write("\n")
    print(json.dumps(res, indent=2))
    print(f"\nwrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
