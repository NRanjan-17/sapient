# `benchmarks/lang-comparison` — the Rust-vs-C++ study

Everything behind [`docs/RUST_VS_CPP_REPORT.md`](../../docs/RUST_VS_CPP_REPORT.md):
C++ transliterations of SAPIENT's hot CPU kernels, the gates that prove they
compute the same thing, and the harnesses that time them — in isolation and
inside the real engine.

## The question, and why it needs this much machinery

"Would SAPIENT be faster in C++?" cannot be answered by racing SAPIENT against
llama.cpp: that compares two codebases which differ in kernels, threading,
layout and years of tuning, with language as one variable among many. The only
valid control for "SAPIENT in C++" is **SAPIENT's own kernels, compiled as C++,
measured against themselves** — which is what lives here.

## Layout

| Path | What it is |
|---|---|
| `cpp/` | C++ ports + the `extern "C"` ABI. Line-for-line transliterations: same algorithm, same layout, same intrinsics |
| `rust/cpp-kernels/` | Builds the C++ (`cc`) and exposes safe Rust wrappers. Outside the workspace, so `cargo build --workspace` never needs a C++ compiler |
| `rust/harness/` | Shared fixtures, the bit-identity gate, Criterion benches, and the experimental Rust kernel variants |
| `asm/` | Disassembly extraction + instruction census + opcode-histogram diff |
| `harness/` | Build-both-arms script, the ABBA end-to-end driver, result collection |
| `results/` | Every raw number this study reports |

## The gate comes first

`rust/harness/tests/bit_identity.rs` compares each C++ kernel to the **existing
Rust scalar oracles** with exact `f32::to_bits()` equality — the repo's own
standard, not approximate agreement. A kernel that is merely *close* is a
different kernel, and timing it would compare two different computations.

```bash
cd rust/harness && cargo test --release --test bit_identity
```

**No timing number in the report is recorded, or believed, until this is green.**

It already caught one real defect: clang defaults to `-ffp-contract=on` and was
fusing `a*b + c` into an FMA, rounding once where Rust rounds twice. The C++ is
now built `-ffp-contract=off`. A naive port would have shipped different numerics.

## Running it

```bash
./run.sh              # everything: gate, asm diff, micro-benchmarks, end-to-end A/B
./run.sh gate         # correctness only
./run.sh micro        # Criterion, Rust vs C++ vs the experimental Rust variants
./run.sh asm          # instruction census and opcode diff
./run.sh e2e          # build both engine arms and run the ABBA A/B
```

or `just bench-lang` from the repo root.

## What it does to the rest of the repo

Almost nothing, by design:

* the study crates are **excluded** from the workspace, so `cargo build --workspace`
  and CI's `clippy --workspace` behave exactly as before and need no C++ toolchain;
* `sapient-backends-cpu` gains a `cpp-kernels` feature that is **off by default** —
  with it off, the optional dependency is not built and the compiled output is
  unchanged (the baseline binary is byte-size-identical to the pre-study build);
* `[profile.bench-lang]` is additive; the shipping `[profile.release]` is untouched.

## Protocol rules

Each of these has produced a wrong number in this repo before.

* **`--backend cpu` always.** `auto` resolves to an accelerator.
* **Verify `mmap: false`.** The Q4_K_R4 repack *skips* mmap-backed tensors, so an
  mmap'd load silently runs different kernels than the ones under test. On macOS
  the engine reads `vm.page_free_count` — free pages only, not reclaimable ones —
  so auto-mmap trips on almost any model unless inactive pages are reclaimed
  first. `ab_e2e.py` reclaims, then **aborts** if a run still reports `mmap: true`.
* **Never quote `bench-llm`'s `tps` as a decode rate.** It puts prefill in the
  denominator. Compute `(tokens-1)/((elapsed-ttft)/1000)`.
* **Never quote its `peak_rss_mb` as a peak.** It is a single end-of-run sample.
* **ABBA, never all-A-then-all-B**, and report per-block ratios so thermal drift
  is visible rather than averaged away.
* **`SAPIENT_THERMAL=off`** and a pinned `RAYON_NUM_THREADS`.
