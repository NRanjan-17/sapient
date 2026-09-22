#!/usr/bin/env python3
"""Collect Criterion estimates into one tidy JSON, with the dispersion that
Criterion's own console output hides. The repo's existing benchmark docs report
bare means with no n, no CI and no spread (docs/BENCHMARKS.md); this study does
not repeat that."""
import json, pathlib, sys, statistics

crit = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "target/criterion")
out = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "results/raw/criterion.json")

rows = []
for est in sorted(crit.glob("*/*/*/new/estimates.json")):
    group, arm, shape = est.parts[-5], est.parts[-4], est.parts[-3]
    e = json.loads(est.read_text())
    samp = est.parent / "sample.json"
    per_iter = []
    if samp.exists():
        s = json.loads(samp.read_text())
        per_iter = [t / n for t, n in zip(s["times"], s["iters"])]
    rows.append({
        "group": group, "arm": arm, "shape": shape,
        "median_ns": e["median"]["point_estimate"],
        "median_ci_lo": e["median"]["confidence_interval"]["lower_bound"],
        "median_ci_hi": e["median"]["confidence_interval"]["upper_bound"],
        "mean_ns": e["mean"]["point_estimate"],
        "std_dev_ns": e["std_dev"]["point_estimate"],
        "n_samples": len(per_iter),
        "min_ns": min(per_iter) if per_iter else None,
        "max_ns": max(per_iter) if per_iter else None,
    })
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(rows, indent=2))
print(f"wrote {len(rows)} rows -> {out}")
