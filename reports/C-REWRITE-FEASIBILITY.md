# Would rewriting parts of SAPIENT in C make it faster or more memory-efficient?

> **Analysis date:** 2026-09-22 · **Codebase:** SAPIENT v0.6.0 (`main`, commit `a0965a5`)
> **Question asked:** *"What part of SAPIENT can I modify in C so it becomes faster and more memory efficient?"*
> ## CORRECTION (2026-09-22) — a load-bearing claim below is WRONG
>
> §1.1 argues that Rust and C share the LLVM backend and therefore "identical
> source logic yields identical machine code," so no language-level speedup
> exists. **Measurement disproves this.** A C++ transliteration of
> `dot_q4_k_4rows_r4_q8k_neon` measured **1.298x** — clang emitted 4x the `sdot`
> per loop body and 30 fewer branches than rustc. Shared LLVM guarantees the same
> *instruction set*, not the same *optimisation decisions*.
>
> **The recommendation still stands, for a different reason:** the difference is
> small end-to-end (+5.5%), roughly 40% of it is recoverable by ordinary Rust
> edits, and the other hot kernel measured exactly 1.000x. See
> [`docs/C_AND_RUST_FINDINGS.md`](../docs/C_AND_RUST_FINDINGS.md) for the full
> study, and `benchmarks/lang-comparison/` for the code and raw data.
>
> **Update (2026-09-22):** the suite has since been run — see
> [`TEST-RUN-M2-2026-09-22.md`](TEST-RUN-M2-2026-09-22.md). **322 passed, 0 failed.**
> It corrects the test counts in §0 below and confirms the conclusion here is unchanged
> (correctness tests cannot measure performance).
>
> **Short answer:** **Almost none of it.** Exactly one opportunity is real, it is
> ARM-only, and it wins by *linking* someone else's assembly rather than by
> being written in C. The largest remaining speed and memory wins are both in Rust.

---

## 0. Method, and what this report is *not* based on

**This analysis is based on `docs/BENCHMARKS.md`, not on a test run.**

Two things are worth stating plainly, because they shaped the conclusion:

1. **The test suite was not executed for this report.** The machine used for the
   analysis had no Rust toolchain installed (`cargo`, `rustc`, and `~/.rustup`
   all absent). A backgrounded `cargo test --workspace` reported exit code 0 but
   its actual output was `command not found` — a false pass, noted here so nobody
   later mistakes it for a green run.

2. **The test suite could not have answered this question anyway.** SAPIENT's
   test suite (**342 test attributes; 332 executed, 20 ignored** — measured, see the
   2026-09-22 run; an earlier static grep here said "334 / 19" and undercounted) consists
   of *correctness* gates, not performance measurements. They assert equality:

  | Test | What it asserts |
  |---|---|
  | `q6_k_neon_matches_scalar` | SIMD kernel == scalar reference |
  | `wgpu_logits_match_cpu_llama` | GPU engine == CPU engine |
  | `repacked_engine_logits_are_bit_identical` | optimization changed zero output bits |
  | `stage_albert_matches_reference` | Rust port == PyTorch reference, 1e-5 |

   None of them record time or memory. That evidence lives in
   [`docs/BENCHMARKS.md`](../docs/BENCHMARKS.md), which is what this report reads.

   *(One benchmark does live in the test harness —
   `crates/sapient-generate/tests/moe_bench.rs` — but it is `#[ignore]`d and needs
   Mixtral-8x7B Q4_K_M, ~26 GB, on a 32 GB+ box.)*

---

## 1. The premise, examined: C is not faster than Rust here

Three independent reasons, all verifiable in this repository.

### 1.1 Same compiler backend ~~— OVERTURNED, see the correction above~~

> **This subsection is wrong and is kept only so the correction has something to
> point at.** Measured: the two compilers make different optimisation decisions
> from the same logic, worth 1.298x on the hottest kernel.

Rust and C both compile through LLVM. ~~Identical source logic yields identical
machine code. There is no language-level speedup available.~~ Shared LLVM
guarantees the same *available instructions*; inlining, unrolling and
vectorisation heuristics differ between the two front ends and their LLVM
versions.

### 1.2 The hot kernels are *already* written the way C would be

`crates/sapient-backends/cpu/src/kernels/quant.rs` is **3,787 lines** of
`core::arch::aarch64` NEON intrinsics and raw `core::arch::asm!` blocks emitting
`sdot` and `smmla` instructions directly. Rewriting that in C hands the same
intrinsics to the same compiler and produces the same instructions.

```
crates/sapient-backends/cpu/src/kernels/quant.rs      3787 lines
crates/sapient-backends/cpu/src/kernels/matmul.rs     1536
crates/sapient-backends/cpu/src/spinpool.rs            541
crates/sapient-backends/cpu/src/kernels/attention.rs   530
```

### 1.3 There is no C in the build today, deliberately

`find crates -name 'build.rs' -o -name '*.c' -o -name '*.cpp'` returns **nothing**.
The README headline is *"pure-Rust… No Python · No Docker · No CUDA"*, and the
cross-compilation story (iOS, Android, Raspberry Pi, Windows) depends on it.

### 1.4 On memory specifically: there is no C advantage at all

Rust has no garbage collector and no runtime. A `[u8]` of Q4_K blocks occupies
exactly what it would in C. **Memory efficiency here is decided by data format** —
quantization scheme, mmap vs heap, KV-cache dtype — and every one of those choices
is language-independent. See §4.

---

## 2. Where the performance gap actually is

From `docs/BENCHMARKS.md`, the Jetson AGX Thor (14× Neoverse) decomposition —
the single most useful measurement in the repo, because it **separates kernel
quality from threading**:

| Qwen2.5-1.5B Q4_K_M, dense | SAPIENT | llama.cpp | gap |
|---|---:|---:|---:|
| Decode, **1 thread** | 3.60 tok/s | 6.97 | **1.94×** — single-core kernel quality |
| Multicore scaling, 1  14 threads | 7.4× (53%) | 12.1× (86%) | **1.6×** — threading overhead |
| Decode, 14 threads | 26.68 | 84.45 | 3.16× (combined) |

On Apple M4 the same-session CPU gap is narrower: **1.47×** (Llama-3.2-1B) and
**1.64×** (Qwen2.5-1.5B). On the Metal path SAPIENT is within **0.81–0.93×** of
llama.cpp and holds the **lowest TTFT of the three engines** (52–63 ms vs Ollama's
~130–150 ms).

The repo names the cause of the 1.94× explicitly, in both `docs/ROADMAP.md` and
`CLAUDE.md`:

> "~1.94× single-core kernel quality (llama.cpp = Arm **KleidiAI** microkernels)
> × ~1.6× multicore scaling (per-GEMV rayon fork/join)."

**So the gap is 50/50, and only one half is a kernel problem.**

---

## 3. The finding that should change the plan

From the Raspberry Pi 5 optimization hunt, recorded in `CLAUDE.md`:

> "**Decode is memory-latency-bound, not compute-bound** — SDOT, single-reduction,
> and multi-row/MLP kernels all gave ~0 once Q6_K was vectorized; the practical
> kernel ceiling is 'no scalar K-quant kernels.'"

Corroborating figure: SAPIENT's decode moves **~26 GB/s** of weight traffic against
Thor's **~200 GB/s** roofline.

**During decode the CPU is waiting on RAM, not on arithmetic.** Making the math
faster — in C, in assembly, in anything — does not help when the bottleneck is
fetching weights from memory.

This is why the repo carries a list of **measured no-go** optimizations. Each was
built, benchmarked, and reverted:

| Attempted optimization | Measured result |
|---|---|
| Cache-blocking over activation panels | Thor **+7% TTFT** (worse) |
| x4 register tile | M4 **−5%**, Thor neutral (register spills) |
| SVE port | Pointless — Thor's SVE is 128-bit = NEON width |
| Vectorized SMMLA combine | Neutral (−0.7%, noise); reverted |
| Kokoro windowed streaming decode | NO-GO — AdaIN statistics are global |

This is the trap to avoid. The instinct to "rewrite the inner loop in C" targets
the half of the problem that is not the bottleneck.

---

## 4. Memory: the real wins are all in Rust

Current state is already strong: weights are mmap'd (RSS ≈ file size), the KV cache
is Q8_0, GPU weights stay quantized on-device, and embedding lookup is row-wise
rather than whole-table.

The concrete remaining win is named in `CLAUDE.md`:

> "Remaining heap (~18 GB) is those Q8_0-converted experts; **a first-class Q5_0
> mmap dtype + kernel would zero it** (future)."

**Context:** unsloth "dynamic" quants store some tensors as Q5_0, which SAPIENT
cannot keep as blocks, so the loader re-quantizes them to Q8_0 at load. For
GLM-4.5-Air that is ~18 GB of heap that could be zero-copy mmap instead. The
earlier fix in this same area (F32-expansion  Q8_0 re-quant) already took peak
RSS from **118 GB  72 GB** and decode from **2.45  3.23 tok/s**; finishing the
job with a native Q5_0 dtype removes what remains.

This is a `DType` enum variant plus a GEMV kernel, following the existing Q5_K
code exactly. Order of a few hundred lines of Rust. **C contributes nothing.**

---

## 5. The one place C genuinely buys something

**Link Arm's KleidiAI microkernels for Q4_K / Q8_0 GEMV on aarch64.**
Ceiling: up to **~1.94× single-core** on server ARM.

Note carefully *why* this wins. Not because it is C — because it is **pre-written,
hand-tuned assembly that Arm engineers spent years on**. You would be *linking* it,
not authoring it. Writing your own C kernels from scratch lands roughly where the
existing Rust ones already are (§1.2).

### Costs, which are the actual decision

| Cost | Detail |
|---|---|
| Build complexity | Every target needs a C cross-compiler. `docs/MOBILE.md` documents how painful the *existing* C deps (`onig_sys`, `esaxx-rs`) already are on the Android NDK. |
| Narrow reach | aarch64 only. x86, all three GPU backends (WGSL shaders), Whisper, Kokoro and SNAC get nothing. |
| Positioning | Contradicts the README's headline "pure-Rust" claim and the frictionless cross-compile story. |
| Licensing | **Not a problem.** KleidiAI is Apache-2.0, compatible with AGPL-3.0-only. |

### If you do it

- Gate it behind an **optional Cargo feature**, `aarch64`-only, so the pure-Rust
  build stays the default and mobile / Windows / x86 packaging is untouched.
- Gate correctness on **`tests/cpu_repack.rs::repacked_engine_logits_are_bit_identical`**.
  That test exists precisely to prove an optimization changed no output bits, and
  it is the right gate for swapping in a foreign kernel.
- Also run the scalar-reference family — `q4_k_r4_kernel_matches_single_row`,
  `q6_k_w6a8_neon_matches_scalar` — which pin every kernel to a plain-Rust oracle.

### The other half of the gap is not a C problem

The ~1.6× multicore scaling gap is `rayon` fork/join overhead — roughly **230
barriers per token**. The team already attacked this in Rust with a custom
spin/park threadpool (`crates/sapient-backends/cpu/src/spinpool.rs`, 541 lines),
measuring **+7.7%** on M4 llama-1B and **+5.3%** on 14-core Thor. What remains is
scheduling design — fewer parallel regions per token. C is irrelevant to it.

---

## 6. Recommendation

| Rank | Change | Language | Expected payoff |
|---|---|---|---|
| 1 | `DType::Q5_0` first-class mmap dtype + GEMV kernel | **Rust** | −18 GB heap on GLM-class MoE |
| 2 | Fewer parallel regions per token (threadpool redesign) | **Rust** | up to ~1.6× on many-core ARM |
| 3 | Multi-row / tiled GEMM for wgpu prefill | **WGSL** | TTFT (named P5-remaining in ROADMAP) |
| 4 | Link KleidiAI behind an `aarch64` feature flag | **C (linked)** | up to ~1.94× single-core, ARM only |

**Ranks 1–3 are larger, cheaper, and portable. Rank 4 is real but expensive, and
it is the only item on this list where C appears at all.**

---

## 7. Sources

| Claim | Source |
|---|---|
| Thor 1-thread / multicore decomposition | `docs/BENCHMARKS.md` §"Where the 1.8× goes" |
| M4 head-to-head CPU + Metal + TTFT | `docs/BENCHMARKS.md` §"v0.5.3 head-to-head refresh" |
| KleidiAI named as the single-core cause | `docs/ROADMAP.md:42-52`; `CLAUDE.md` |
| "Decode is memory-latency-bound" | `CLAUDE.md` §SIMD hot paths (RPi5 perf hunt) |
| No-go optimization records | `CLAUDE.md`; `kernels/matmul.rs`; `docs/BENCHMARKS.md` |
| Q5_0 / 18 GB residual heap | `CLAUDE.md` §GLM-4.5-Air, "Q5_0Q8_0 at load (RSS fix)" |
| Spinpool measured deltas | `CLAUDE.md` §"Decode spin/park threadpool" |
| Test counts / no C in build | Measured run 2026-09-22 + scan of `crates/`, `tests/` at `a0965a5` |

---

*If any figure here stops matching the code or the benchmark record, the code and
`docs/BENCHMARKS.md` win — please open a PR to correct this report.*
