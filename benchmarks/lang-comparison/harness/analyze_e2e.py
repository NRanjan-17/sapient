#!/usr/bin/env python3
"""Paired block analysis for the end-to-end A/B.

The experiment is a RANDOMISED BLOCK design: within a block every arm is
measured adjacently, so a block ratio cancels the slow drift that dominates this
machine. Pooling raw samples across blocks -- let alone across sessions -- does
not, and on this hardware between-session drift was measured at +7.1% on an
identical binary, i.e. larger than the effect under study.

So the unit of analysis is the BLOCK RATIO, and the reported statistic is over
those. That also makes the result robust to a single anomalous run: one 48.35
tok/s sample (2.2x every neighbour, on deterministic identical work) perturbs
its block's median barely at all.
"""
import json, pathlib, statistics as st, random, sys

def block_ratios(path, baseline=None):
    d = json.loads(pathlib.Path(path).read_text())
    base = d.get("baseline") or baseline or "rust"
    blocks = {}
    for entry in d["per_block"]:
        blocks.setdefault(entry["block"], {}).setdefault(entry["arm"], []).extend(
            r["decode_tps"] for r in entry["runs"])
    out = {}
    for b, arms in sorted(blocks.items()):
        if base not in arms: continue
        mb = st.median(arms[base])
        for a, v in arms.items():
            if a == base: continue
            out.setdefault(a, []).append(st.median(v) / mb)
    return out, base, blocks

def summarize(name, ratios, base):
    n = len(ratios)
    m = st.median(ratios); mean = st.mean(ratios)
    # Bootstrap over BLOCKS (the independent unit), not over runs.
    rng = random.Random(11); bs = []
    for _ in range(20000):
        s = [ratios[rng.randrange(n)] for _ in range(n)]
        bs.append(st.median(s))
    bs.sort()
    lo, hi = bs[int(.025*len(bs))], bs[int(.975*len(bs))]
    pos = sum(1 for r in ratios if r > 1.0)
    sig = "no significant difference" if lo <= 1.0 <= hi else (
        "SIGNIFICANTLY FASTER" if m > 1 else "SIGNIFICANTLY SLOWER")
    print(f"  {name} vs {base}")
    print(f"    block ratios (n={n}): " + " ".join(f"{r:.4f}" for r in ratios))
    print(f"    median {m:.4f}x ({(m-1)*100:+.2f}%)  mean {mean:.4f}x"
          f"  sd {st.stdev(ratios) if n>1 else 0:.4f}")
    print(f"    95% CI on median block ratio: [{lo:.4f}, {hi:.4f}]"
          f" = [{(lo-1)*100:+.2f}%, {(hi-1)*100:+.2f}%]")
    print(f"    blocks favouring {name}: {pos}/{n}   -> {sig}\n")
    return {"arm": name, "n_blocks": n, "ratios": ratios, "median": m,
            "mean": mean, "ci95": [lo, hi], "blocks_favouring": pos, "verdict": sig}

if __name__ == "__main__":
    paths = sys.argv[1:] or ["../results/raw/e2e_decode_3arm.json"]
    allres = {}
    pooled = {}
    for p in paths:
        print(f"\n=== {pathlib.Path(p).name} ===")
        ratios, base, _ = block_ratios(p)
        for a, r in ratios.items():
            allres.setdefault(p, {})[a] = summarize(a, r, base)
            pooled.setdefault(a, []).extend(r)
    if len(paths) > 1:
        print("=== POOLED BLOCK RATIOS across sessions ===")
        print("(valid where raw pooling is not: each ratio is internally paired,")
        print(" so session-level drift has already been divided out)\n")
        for a, r in pooled.items():
            summarize(a, r, "rust")
