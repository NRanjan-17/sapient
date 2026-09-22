#!/usr/bin/env python3
"""Stratify the end-to-end block ratios by PERFORMANCE REGIME.

Block 1 of session C ran with the baseline at ~54 tok/s and showed C++ at
0.99x -- no advantage. Earlier sessions ran at ~20 tok/s and showed +6%. Same
binaries, same model, same kernels. So the effect is regime-dependent and a
single pooled number would misrepresent it.

Working hypothesis: macOS demotes sustained CPU-heavy work to efficiency cores.
The C++ advantage comes from instruction-level parallelism (four independent
sdot->addv chains instead of one). A wide out-of-order P-core already extracts
that parallelism on its own, so unrolling buys little; a narrower E-core cannot,
so unrolling wins. If true, the advantage should be large in the slow regime and
near zero in the fast one -- which is what stratifying tests.
"""
import json, pathlib, statistics as st, random, sys

TOL = 0.10

def blocks_from(path):
    d = json.loads(pathlib.Path(path).read_text())
    base = d.get("baseline", "rust")
    seq = {}
    for e in d["per_block"]:
        seq.setdefault(e["block"], []).append(
            (e["arm"], [r["decode_tps"] for r in e["runs"]]))
    out = []
    for b, s in sorted(seq.items()):
        occ = [i for i, (a, _) in enumerate(s) if a == base]
        if len(occ) < 2: continue
        drift = st.median(s[occ[-1]][1]) / st.median(s[occ[0]][1]) - 1
        arms = {}
        for a, v in s: arms.setdefault(a, []).extend(v)
        mb = st.median(arms[base])
        out.append({"src": pathlib.Path(path).name, "block": b, "drift": drift,
                    "base": mb,
                    "ratios": {a: st.median(v)/mb for a, v in arms.items() if a != base}})
    return out

def ci(v, seed=5):
    rng = random.Random(seed); bs = []
    for _ in range(20000):
        bs.append(st.median([v[rng.randrange(len(v))] for _ in v]))
    bs.sort(); return bs[int(.025*len(bs))], bs[int(.975*len(bs))]

def report(label, blks):
    arms = sorted({a for b in blks for a in b["ratios"]})
    print(f"\n### {label}  (n={len(blks)} blocks, baseline "
          f"{min(b['base'] for b in blks):.1f}-{max(b['base'] for b in blks):.1f} tok/s)")
    for a in arms:
        v = [b["ratios"][a] for b in blks if a in b["ratios"]]
        if len(v) < 2:
            print(f"    {a:<16} n={len(v)} — too few blocks"); continue
        lo, hi = ci(v); m = st.median(v)
        sig = "no significant difference" if lo <= 1.0 <= hi else (
              "FASTER" if m > 1 else "SLOWER")
        print(f"    {a:<16} n={len(v):<3} median {m:.4f}x ({(m-1)*100:+.2f}%)  "
              f"CI [{(lo-1)*100:+.2f}%, {(hi-1)*100:+.2f}%]  "
              f"{sum(1 for x in v if x>1)}/{len(v)}  -> {sig}")

if __name__ == "__main__":
    allb = []
    for p in sys.argv[1:]:
        try: allb += blocks_from(p)
        except FileNotFoundError: print(f"(skip {p})")
    stable = [b for b in allb if abs(b["drift"]) <= TOL]
    print(f"{len(allb)} blocks, {len(stable)} pass the |drift|<={TOL:.0%} stationarity gate")
    THRESH = 30.0     # tok/s; the two regimes are well separated (20-22 vs 36-54)
    slow = [b for b in stable if b["base"] < THRESH]
    fast = [b for b in stable if b["base"] >= THRESH]
    report(f"SLOW regime (baseline < {THRESH:.0f} tok/s)", slow) if slow else None
    report(f"FAST regime (baseline >= {THRESH:.0f} tok/s)", fast) if fast else None
    report("ALL stable blocks pooled", stable)
    pathlib.Path("../results/raw/e2e_regime.json").write_text(json.dumps(
        {"threshold_tps": THRESH, "blocks": stable}, indent=2))
