#!/usr/bin/env python3
"""End-to-end A/B: the real engine with Rust leaf kernels vs the same engine
with bit-identical C++ leaf kernels.

Protocol rules, each of which has produced a wrong number in this repo before:

  * decode rate is computed here as (tokens-1)/((elapsed-ttft)/1000). `bench-llm`'s
    own `tps` puts prefill inside the denominator and would understate decode.
  * ABBA interleaving, never all-A-then-all-B, so thermal drift cancels within a
    block instead of accumulating against the arm that ran second.
  * every run is checked for `mmap == False`. The Q4_K_R4 repack SKIPS mmap-backed
    tensors, so an mmap'd load silently runs different kernels than the ones
    under test. macOS reports `vm.page_free_count` (free pages only, not
    reclaimable), so the engine's auto-mmap heuristic trips on almost any model
    here unless inactive pages are reclaimed first -- which this script does.
  * `--backend cpu` explicitly; `auto` would resolve to an accelerator.
  * statistic is the median and a percentile bootstrap CI on the RATIO of
    medians, not a difference of means.
"""
import argparse, ctypes, json, pathlib, random, statistics, subprocess, sys, time

def reclaim_pages(gb=8):
    """Force macOS to reclaim inactive pages so the engine's auto-mmap
    heuristic sees real availability. Without this the model loads mmap'd and
    the kernels under test never run."""
    try:
        n = gb * 1024**3
        b = ctypes.create_string_buffer(n)
        for i in range(0, n, 1 << 20):
            b[i] = 1
        del b
        time.sleep(1)
    except Exception as e:
        print(f"  ! page reclaim failed: {e}", file=sys.stderr)

def compressed_mb():
    """macOS compresses inactive anonymous pages. When the engine's ~1.3 GB of
    heap-resident weights get compressed, every access faults and decompresses,
    which is the leading explanation for the 21 -> 54 tok/s regime swings seen
    on this machine. Recorded per block so contamination is visible, not guessed."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    ps = int(subprocess.run(["sysctl","-n","hw.pagesize"],
                            capture_output=True, text=True).stdout)
    for line in out.splitlines():
        if "occupied by compressor" in line:
            return int(line.split(":")[1].strip().rstrip(".")) * ps // 1024 // 1024
    return -1

def free_mb():
    g = lambda k: int(subprocess.run(["sysctl","-n",k],capture_output=True,text=True).stdout)
    return g("hw.pagesize") * g("vm.page_free_count") // 1024 // 1024

def thermal_ok():
    out = subprocess.run(["pmset","-g","therm"],capture_output=True,text=True).stdout
    for line in out.splitlines():
        if "CPU_Speed_Limit" in line:
            try:
                return int(line.split("=")[1].strip()) == 100, line.strip()
            except Exception:
                return True, line.strip()
    return True, "no CPU speed limit reported"

def run_arm(binary, model, prompt, max_tokens, runs, threads):
    cmd = [str(binary), "bench-llm", model, "--backend", "cpu",
           "--max-tokens", str(max_tokens), "--runs", str(runs),
           "--prompt", prompt, "--json"]
    env = {"SAPIENT_THERMAL": "off", "RAYON_NUM_THREADS": str(threads),
           "PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": str(pathlib.Path.home())}
    last = ""
    for attempt in range(4):
        p = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if p.returncode == 0:
            break
        last = p.stderr[-400:]
        # The engine hits the Hub for a file listing even on a fully cached
        # model, so a transient DNS blip aborts an otherwise fine run.
        print(f"  ! {pathlib.Path(binary).name} attempt {attempt+1} failed, retrying",
              file=sys.stderr)
        time.sleep(5)
    else:
        raise RuntimeError(f"{binary} failed after 4 attempts: {last}")
    # the JSON is the last {...} on stdout
    txt = p.stdout[p.stdout.index("{"):]
    d = json.loads(txt)
    if d["mmap"]:
        raise RuntimeError("model loaded MMAP'd -- R4 repack skipped, kernels under "
                           "test are not running. Reclaim pages and retry.")
    out = []
    for r in d["runs"]:
        dec_s = (r["elapsed_ms"] - r["ttft_ms"]) / 1000.0
        out.append({"decode_tps": (r["total_tokens"] - 1) / dec_s if dec_s > 0 else None,
                    "ttft_ms": r["ttft_ms"], "elapsed_ms": r["elapsed_ms"],
                    "tokens": r["total_tokens"], "load_ms": d["load_time_ms"]})
    return out

def bootstrap_ratio_ci(a, b, iters=10000, seed=12345):
    """Percentile bootstrap CI on median(b)/median(a)."""
    rng = random.Random(seed)
    ratios = []
    for _ in range(iters):
        ra = [a[rng.randrange(len(a))] for _ in a]
        rb = [b[rng.randrange(len(b))] for _ in b]
        ma, mb = statistics.median(ra), statistics.median(rb)
        if ma > 0:
            ratios.append(mb / ma)
    ratios.sort()
    lo = ratios[int(0.025 * len(ratios))]
    hi = ratios[int(0.975 * len(ratios))]
    return lo, hi

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-dir", default="../results/bin")
    ap.add_argument("--arms", default="rust,cpp",
                    help="comma-separated arm names; binaries are sapient-<arm>")
    ap.add_argument("--model", default="openhorizon/qwen2.5-1.5b-q4")
    ap.add_argument("--prompt", default="Write a detailed essay about the Roman Empire.")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--runs", type=int, default=3, help="runs per invocation")
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--out", default="../results/raw/e2e_decode.json")
    args = ap.parse_args()

    bd = pathlib.Path(args.bin_dir).resolve()
    names = [a.strip() for a in args.arms.split(",") if a.strip()]
    arms = {n: bd / f"sapient-{n}" for n in names}
    for k, v in arms.items():
        if not v.exists(): sys.exit(f"missing binary: {v}")
    baseline = names[0]

    ok, msg = thermal_ok()
    print(f"thermal before: {msg}{'' if ok else '  <-- CONTAMINATED'}")
    print(f"threads={args.threads} max_tokens={args.max_tokens} "
          f"blocks={args.blocks} runs/invocation={args.runs}\n")

    samples = {n: [] for n in names}
    per_block = []
    for blk in range(1, args.blocks + 1):
        print(f"── block {blk}/{args.blocks} ──  (free={free_mb()} MB, "
              f"compressed={compressed_mb()} MB)")
        if free_mb() < 3000:
            print(f"   reclaiming pages (free={free_mb()} MB)")
            reclaim_pages()
        # Palindromic order (ABC CBA): every arm is equidistant from the block
        # midpoint, so a monotone thermal drift across the block biases them
        # equally instead of penalising whichever ran last.
        order = names + names[::-1]
        blk_s = {n: [] for n in names}
        for arm in order:
            res = run_arm(arms[arm], args.model, args.prompt,
                          args.max_tokens, args.runs, args.threads)
            tps = [r["decode_tps"] for r in res]
            blk_s[arm] += tps
            samples[arm] += tps
            for r in res:
                r["arm"], r["block"] = arm, blk
            per_block.append({"block": blk, "arm": arm, "runs": res})
            print(f"   {arm:<5} " + " ".join(f"{t:6.2f}" for t in tps))
        base_m = statistics.median(blk_s[baseline])
        print("   block medians: " + "  ".join(
            f"{n}={statistics.median(blk_s[n]):.3f}"
            + (f" ({statistics.median(blk_s[n])/base_m:.4f}x)" if n != baseline else "")
            for n in names))
        ok, msg = thermal_ok()
        if not ok: print(f"   !! thermal: {msg}")

    base = samples[baseline]
    mbase = statistics.median(base)
    print(f"\n{'='*78}")
    print(f"{'arm':<16}{'n':>4}{'median':>9}{'mean':>9}{'sd':>7}{'min':>8}{'max':>8}"
          f"{'vs base':>10}{'95% CI on ratio':>22}")
    print("-"*78)
    summary = {}
    for n in names:
        v = samples[n]
        m = statistics.median(v)
        if n == baseline:
            ci, verdict = (None, None), "baseline"
        else:
            ci = bootstrap_ratio_ci(base, v)
            verdict = ("no significant difference" if ci[0] <= 1.0 <= ci[1]
                       else ("faster" if m > mbase else "slower"))
        cis = "—" if n == baseline else f"[{ci[0]:.4f}, {ci[1]:.4f}]"
        print(f"{n:<16}{len(v):>4}{m:>9.3f}{statistics.mean(v):>9.3f}"
              f"{statistics.stdev(v):>7.3f}{min(v):>8.3f}{max(v):>8.3f}"
              f"{m/mbase:>9.4f}x{cis:>22}")
        summary[n] = {"samples": v, "median": m, "mean": statistics.mean(v),
                      "sd": statistics.stdev(v), "min": min(v), "max": max(v),
                      "ratio_vs_base": m / mbase, "ci95": list(ci), "verdict": verdict}
    print("="*78)
    for n in names:
        if n != baseline:
            print(f"  {n} vs {baseline}: {summary[n]['ratio_vs_base']:.4f}x "
                  f"({(summary[n]['ratio_vs_base']-1)*100:+.2f}%) — {summary[n]['verdict']}")
    print()

    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": vars(args), "baseline": baseline,
        "arms": summary, "per_block": per_block,
    }, indent=2))
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
