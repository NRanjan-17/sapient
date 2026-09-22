#!/usr/bin/env python3
"""Block-ratio analysis WITH a stationarity gate.

The randomised-block design assumes the machine is locally stationary within a
block. On this laptop that assumption is frequently false: decode rate was
observed swinging 21 -> 54 tok/s on identical deterministic work (macOS memory
compression is the leading suspect; the harness's own 8 GB page reclaim may be
inducing it).

The palindromic arm order measures the BASELINE arm first AND last in every
block, so the assumption is directly testable: drift = median(last baseline
block) / median(first baseline block). A block that fails the gate spans a
regime change and its ratios are not interpretable -- block 7 of session 3, for
instance, has the baseline at 21 tok/s at the start and 48 tok/s at the end,
which would manufacture a spurious 1.19x for whichever arm ran late.

The gate is applied to the BASELINE only, so it cannot be biased by the arms
under test.
"""
import json, pathlib, statistics as st, random, sys

TOL = 0.10  # max |drift| within a block

def load(path):
    d = json.loads(pathlib.Path(path).read_text())
    base = d.get("baseline", "rust")
    blocks = {}
    for e in d["per_block"]:                      # per_block is in execution order
        blocks.setdefault(e["block"], []).append(
            (e["arm"], [r["decode_tps"] for r in e["runs"]]))
    return base, blocks, pathlib.Path(path).name

def analyse(paths):
    kept, dropped, rows = [], [], []
    for p in paths:
        base, blocks, name = load(p)
        for b, seq in sorted(blocks.items()):
            occ = [i for i, (a, _) in enumerate(seq) if a == base]
            if len(occ) < 2:
                continue
            first, last = seq[occ[0]][1], seq[occ[-1]][1]
            drift = st.median(last) / st.median(first) - 1.0
            arms = {}
            for a, v in seq:
                arms.setdefault(a, []).extend(v)
            mb = st.median(arms[base])
            rec = {"session": name, "block": b, "drift": drift,
                   "ratios": {a: st.median(v)/mb for a, v in arms.items() if a != base},
                   "base_tps": mb}
            (kept if abs(drift) <= TOL else dropped).append(rec)
            rows.append(rec)

    print(f"{'session':<26}{'blk':>4}{'base tok/s':>12}{'within-block drift':>20}  {'ratios':<34}{'':>6}")
    print("-"*108)
    for r in rows:
        ok = abs(r["drift"]) <= TOL
        rs = "  ".join(f"{a}={v:.4f}" for a, v in sorted(r["ratios"].items()))
        print(f"{r['session']:<26}{r['block']:>4}{r['base_tps']:>12.2f}"
              f"{r['drift']*100:>19.1f}%  {rs:<34}{'keep' if ok else 'DROP':>6}")
    print("-"*108)
    print(f"kept {len(kept)} blocks, dropped {len(dropped)} for |drift| > {TOL*100:.0f}%\n")

    arms = sorted({a for r in kept for a in r["ratios"]})
    out = {}
    for a in arms:
        v = [r["ratios"][a] for r in kept if a in r["ratios"]]
        if len(v) < 2:
            print(f"  {a}: only {len(v)} usable block(s) — no conclusion"); continue
        rng = random.Random(3); bs = []
        for _ in range(20000):
            s = [v[rng.randrange(len(v))] for _ in v]
            bs.append(st.median(s))
        bs.sort(); lo, hi = bs[int(.025*len(bs))], bs[int(.975*len(bs))]
        m = st.median(v)
        sig = "no significant difference" if lo <= 1.0 <= hi else (
              "SIGNIFICANTLY FASTER" if m > 1 else "SIGNIFICANTLY SLOWER")
        print(f"  {a:<16} n={len(v):<3} median {m:.4f}x ({(m-1)*100:+.2f}%)  "
              f"CI [{lo:.4f}, {hi:.4f}] = [{(lo-1)*100:+.2f}%, {(hi-1)*100:+.2f}%]  "
              f"{sum(1 for x in v if x>1)}/{len(v)} blocks  -> {sig}")
        out[a] = {"n_blocks": len(v), "ratios": v, "median": m, "ci95": [lo, hi],
                  "blocks_favouring": sum(1 for x in v if x > 1), "verdict": sig}
    return out, kept, dropped

if __name__ == "__main__":
    res, kept, dropped = analyse(sys.argv[1:])
    pathlib.Path("../results/raw/e2e_block_analysis.json").write_text(json.dumps(
        {"tolerance": TOL, "kept_blocks": len(kept), "dropped_blocks": len(dropped),
         "arms": res, "blocks": kept}, indent=2))
