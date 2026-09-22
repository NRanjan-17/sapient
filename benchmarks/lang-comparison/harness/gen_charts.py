#!/usr/bin/env python3
"""Charts for docs/RUST_VS_CPP_REPORT.md.

Palette, spines, dpi and label styling are lifted from
scripts/gen-benchmark-charts.py so the report looks like the rest of docs/assets.
The one deliberate departure: **these charts carry error bars.** The existing
ones do not, which is exactly how a repo ends up publishing +4% deltas that sit
inside a documented 40% thermal swing.
"""
import json, pathlib, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE  = "#fcfcfb"
INK      = "#0b0b0b"
INK_2    = "#52514e"
MUTED    = "#898781"
GRID     = "#e1e0d9"
BASELINE = "#c3c2b7"
RUST     = "#2a78d6"
CPP      = "#1baf7a"
VARIANT  = "#eda100"

def style_axis(ax):
    ax.set_facecolor(SURFACE)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(length=0, labelsize=9.5, colors=INK_2)

def chart_micro(rows, out):
    """Per-shape kernel throughput, Rust vs the Rust variants vs C++."""
    arms = [("rust", "Rust (shipped)", RUST),
            ("rust_unrolled", "Rust + unrolled", VARIANT),
            ("rust_unrolled_vectail", "Rust + unrolled + NEON tail", "#d97706"),
            ("cpp", "C++", CPP)]
    for group, title, fname in [
        ("q4_k_4rows_r4_q8k", "Q4_K  (64% of decode)", "lang_micro_q4k.png"),
        ("q6_k_4rows_r4_q8k", "Q6_K  (32% of decode)", "lang_micro_q6k.png"),
    ]:
        g = [r for r in rows if r["group"] == group]
        if not g: continue
        shapes = sorted({r["shape"] for r in g}, key=lambda s: int(s.split("k=")[1]))
        present = [a for a in arms if any(r["arm"] == a[0] for r in g)]
        fig, ax = plt.subplots(figsize=(9.0, 4.4), facecolor=SURFACE)
        style_axis(ax)
        n = len(present); bw = 0.8 / n
        for i, (arm, label, color) in enumerate(present):
            xs, ys, es = [], [], []
            for j, sh in enumerate(shapes):
                m = next((r for r in g if r["arm"] == arm and r["shape"] == sh), None)
                if not m: continue
                xs.append(j - 0.4 + bw * (i + 0.5))
                ys.append(m["median_ns"])
                es.append(m["std_dev_ns"])
            ax.bar(xs, ys, width=bw * 0.92, label=label, color=color,
                   edgecolor=SURFACE, linewidth=1.0,
                   yerr=es, capsize=2, ecolor=MUTED, error_kw={"linewidth": 0.9})
        ax.set_xticks(range(len(shapes)))
        ax.set_xticklabels([s.replace("_k=", "\nk=") for s in shapes], fontsize=8.5)
        ax.set_ylabel("ns per 4-row group  (lower is better)", fontsize=9.5, color=INK_2)
        ax.set_title(title, fontsize=12.5, fontweight="bold", color=INK)
        ax.legend(loc="upper left", frameon=False, labelcolor=INK_2, fontsize=8.5)
        fig.text(0.01, 0.005, "Criterion, n=100 per arm; bars are medians, whiskers ±1 sd. "
                              "All arms bit-identical.", fontsize=7, color=MUTED)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.savefig(out / fname, dpi=140, facecolor=SURFACE, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out/fname}")

def chart_attribution(rows, out):
    """How much of the C++ micro-advantage is recoverable in Rust."""
    g = [r for r in rows if r["group"] == "q4_k_4rows_r4_q8k"]
    if not g: return
    tot = {}
    for r in g:
        tot.setdefault(r["arm"], []).append(r["median_ns"])
    tot = {k: sum(v) for k, v in tot.items()}
    if "rust" not in tot: return
    order = [("rust", "Rust\n(shipped)", RUST),
             ("rust_unrolled", "+ unrolled\nrow loop", VARIANT),
             ("rust_unrolled_vectail", "+ NEON\nepilogue", "#d97706"),
             ("cpp", "C++", CPP)]
    order = [o for o in order if o[0] in tot]
    fig, ax = plt.subplots(figsize=(7.4, 4.4), facecolor=SURFACE)
    style_axis(ax)
    xs = range(len(order))
    ys = [tot["rust"] / tot[a] for a, _, _ in order]
    ax.bar(xs, ys, width=0.62, color=[c for _, _, c in order],
           edgecolor=SURFACE, linewidth=1.0)
    for x, y in zip(xs, ys):
        ax.text(x, y, f"{y:.3f}×", ha="center", va="bottom", fontsize=9, color=INK)
    ax.axhline(1.0, color=BASELINE, linewidth=1.0, linestyle="--")
    ax.set_xticks(list(xs)); ax.set_xticklabels([l for _, l, _ in order], fontsize=9)
    ax.set_ylabel("speedup vs shipped Rust", fontsize=9.5, color=INK_2)
    ax.set_ylim(0.95, max(ys) * 1.12)
    ax.set_title("Q4_K kernel: how much of the C++ gap is Rust-recoverable",
                 fontsize=12.5, fontweight="bold", color=INK)
    fig.text(0.01, 0.005, "Summed over five real model shapes. Every arm is "
                          "bit-identical to the shipped kernel.", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "lang_attribution.png", dpi=140, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out/'lang_attribution.png'}")

def chart_e2e(blocks_json, out):
    """End-to-end block ratios: the statistic the conclusion rests on.

    Plots every block as a point plus the median and its bootstrap CI, because
    the honest story here is the SPREAD -- the machine swung 2.5x in absolute
    throughput while these ratios stayed put."""
    import statistics as st, random
    data = json.loads(pathlib.Path(blocks_json).read_text())
    blocks = data["blocks"]
    arms = [("rust_unrolled", "Rust\n+ unroll + NEON tail", VARIANT),
            ("rust-unrolled", "Rust\n+ unroll + NEON tail", VARIANT),
            ("cpp", "C++\nleaf kernels", CPP)]
    seen, series = set(), []
    for key, label, color in arms:
        v = [b["ratios"][key] for b in blocks if key in b["ratios"]]
        if v and label not in seen:
            seen.add(label); series.append((label, color, v))
    if not series: return

    fig, ax = plt.subplots(figsize=(7.6, 4.6), facecolor=SURFACE)
    style_axis(ax)
    rng = random.Random(4)
    for i, (label, color, v) in enumerate(series):
        bs = sorted(st.median([v[rng.randrange(len(v))] for _ in v]) for _ in range(20000))
        lo, hi = bs[int(.025*len(bs))], bs[int(.975*len(bs))]
        m = st.median(v)
        xs = [i + (rng.random() - 0.5) * 0.22 for _ in v]
        ax.scatter(xs, v, s=22, color=color, alpha=0.45, edgecolor="none", zorder=2)
        ax.errorbar([i], [m], yerr=[[m - lo], [hi - m]], fmt="o", color=color,
                    markersize=9, capsize=6, linewidth=2, zorder=3,
                    markeredgecolor=SURFACE, markeredgewidth=1.2)
        ax.text(i + 0.30, m, f"{(m-1)*100:+.2f}%", va="center", fontsize=10,
                color=INK, fontweight="bold")
    ax.axhline(1.0, color=BASELINE, linewidth=1.2, linestyle="--", zorder=1)
    ax.text(len(series) - 0.45, 1.0, "shipped Rust", fontsize=8, color=MUTED,
            va="bottom", ha="right")
    ax.set_xticks(range(len(series)))
    ax.set_xticklabels([l for l, _, _ in series], fontsize=9.5)
    ax.set_xlim(-0.5, len(series) - 0.25)
    ax.set_ylabel("decode speedup vs shipped Rust", fontsize=9.5, color=INK_2)
    ax.set_title("End-to-end decode — per-block ratios, median and 95% CI",
                 fontsize=12.5, fontweight="bold", color=INK)
    fig.text(0.01, 0.005,
             f"Qwen2.5-1.5B Q4_K_M, CPU, greedy, token-identical output. Each dot is one "
             f"interleaved block ({data['kept_blocks']} kept, {data['dropped_blocks']} "
             f"dropped by the stationarity gate).", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(out / "lang_e2e.png", dpi=140, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out/'lang_e2e.png'}")

if __name__ == "__main__":
    raw = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "../results/raw")
    out = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "../../../docs/assets")
    out.mkdir(parents=True, exist_ok=True)
    crit = raw / "criterion.json"
    if crit.exists():
        rows = json.loads(crit.read_text())
        chart_micro(rows, out); chart_attribution(rows, out)
    blk = raw / "e2e_block_analysis.json"
    if blk.exists():
        chart_e2e(blk, out)
