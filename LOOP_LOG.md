# LOOP_LOG.md — optimisation loop log

One entry per iteration: what was tried, what was measured, the decision, and the next
hypothesis. Negative results stay in. Companion files: `NOVELTY.md` (prior art),
`PAPER.md` (paper plan), `benchmarks/` (result files), `scripts/bench_loop.py` (harness).

## Standing facts

**Merge rule.** A change merges only if it improves at least one metric with no
correctness regression and no other metric worse beyond noise. Noise on the M4 laptop
is up to ~8% run to run, so decisions use interleaved A/B runs of the two binaries,
not a diff of two result files.

**Hardware the loop can use**

| Target | State |
|---|---|
| Apple M4 (CPU, Metal) | Full access. llama.cpp 0.5.0 (build 11146) and Ollama 0.12.6 installed; ExecuTorch and MLC-LLM are not. |
| Raspberry Pi 5 | One pre-approved action only: replace `~/sapient-vision-test/sapient`, run the `see` timing, read the temperature. The Pi rule (ask before any other command) still holds, and the loop does not ask — so anything else on the Pi is skipped and logged. |
| Jetson AGX Thor | Password SSH; not available to the loop. |
| WASM | No `wasm32` target installed; the tokio/reqwest/mmap/rayon dependency tree does not build for it. Not started. |
| ESP32-S3 / Cortex-M7 | No `no_std` core exists. Scoped at 4–8 weeks (see iteration 1). Not started. |

**Capabilities missing for the paper** (details in `NOVELTY.md`): no VLA model loads;
no early-exit/anytime path; speculative decoding uses a fixed draft length; no weight
streaming or prefetch; KV reuse is in-memory only.

---

## Iteration 1 — 2026-10-02 — baseline, harness, prior-art map

**Research.** Full prior-art search for "deadline-guaranteed VLA inference on edge
hardware" (results in `NOVELTY.md`). Outcome: partially covered. Entropy-adaptive
speculative decoding (AdaEDL; Spec-VLA, WA-SpecDec on VLAs) and KV/token reuse across
frames (VLA-Cache and follow-ons) are published. No paper found schedules on-robot VLA
inference against a wall-clock deadline, and no Raspberry-Pi-class VLA latency study
was found. Competing runtimes exist (vla.cpp, Embodied.cpp).

**Decisions.**
1. MLSys 2027's deadline is 2026-10-30. Not feasible (no VLA runs on Sapient). Realistic
   targets: RSS 2027 or MLSys 2028. Recorded in `PAPER.md`.
2. The paper's claim narrows to a wall-clock deadline scheduler on CPU-only boards with
   a measured miss rate — not a "guarantee", and not speculative decoding or KV reuse.
3. The critical path for any paper experiment is a SmolVLA port.

**Change.**
- `scripts/bench_loop.py`: one-command, reproducible baseline (LLM decode / TTFT / peak
  RSS via `bench-llm`, llama.cpp on the same GGUF, vision encode p50/max/stdev with
  per-stage split, binary size). Deterministic in-process test image; no tuning knobs.
- `benchmarks/2026-10-02-m4-baseline.json`: first committed result file.
- Correctness fix: `llama-3.2-3b` and `mistral-7b` are now flagged `gated` in the
  registry (they download from gated Hugging Face repos; `sapient models` did not warn).

**Measured (Apple M4, CPU, v0.6.0 + today's vision kernels).**

| Metric | Sapient | llama.cpp |
|---|---:|---:|
| Qwen2.5-1.5B Q4_K_M decode | 53.6 tok/s (±0.45) | 62.6 (1.17×) |
| Llama-3.2-1B Q4_K_M decode | 66.5 tok/s (±1.64) | 75.4 (1.13×) |
| TTFT, bench prompt (Qwen / Llama) | 362 / 378 ms | — |
| Peak RSS (Qwen / Llama) | 1177 / 1184 MB | — |
| SmolVLM-256M image encode | p50 606 ms, max 659, stdev 30.8 | — |
| Binary (CPU build) | 52.4 MB on disk | — |

A run minutes earlier read 71.2 tok/s (Llama-1B) and 556 ms (vision p50, stdev 9.0):
the spread sets the noise band above. llama.cpp is build 11146 today; the July tables
in `docs/BENCHMARKS.md` used b9860, so ratios are not comparable across those dates.

**Not done / skipped.**
- No kernel change this iteration (harness and baseline first).
- Pi LLM baseline: needs model downloads on the Pi — outside the approved action. Skipped.
- Ollama, ExecuTorch, MLC-LLM are not in the scripted suite (Ollama needs its daemon;
  the other two are not installed).

**Decision.** Merge on green CI: adds a harness, result file, docs and a two-line
registry fix; no performance metric changes.

**Next hypothesis (iteration 2).** The four Q8_0 linears are ~60% of the vision encode
(Pi: ~2.3 s of 3.5 s). Their kernel pays an f32 scale combine per 32-element block.
A per-256 activation scale with integer-domain block combine (the Q8_K idea already
used for K-quants) should cut that tail. It changes the activation-quantisation
accuracy class, so it needs a quality gate first — which the repo lacks. Order:
(a) add a perplexity/feature-drift gate, (b) then the kernel. If (a) is too large for
one iteration, fall back to the bit-identical lever: thread scaling of the blocked
GEMM (1 → 10 threads gives only ~4× on the M4).

---

## Iteration 2 — 2026-10-02 — perplexity gate (`sapient eval-ppl`) and first quality numbers

**Research.** Technique: a perplexity gate that scores exactly what llama.cpp's
`llama-perplexity` scores (non-overlapping 512-token chunks, second half of each chunk,
BOS at chunk start), so the two engines can be compared on the same GGUF and the same
tokens. Source: llama.cpp `tools/perplexity`. Expected gain: no speed change; it
unlocks every non-bit-identical kernel change (three were blocked on it: Q8_K-style
activations for Q8_0 linears, the vectorised softmax `exp`, MLX 4-bit re-quantisation).

**Change.**
- `sapient eval-ppl <model|file.gguf> --file <text> [--ctx 512] [--chunks 20] [--json]`
  (hidden command). Works on every engine that implements `forward_all_logits`.
- `scripts/bench_loop.py` gains a `quality` section (Sapient + `llama-perplexity` on
  wikitext-2, downloaded on first use). The harness edit was delegated to a Haiku
  subagent and reviewed; one fix after review (`-ngl 0` so llama.cpp scores on CPU).
- `bench-llm` and `eval-ppl` share one model loader.

**Measured** (Apple M4, CPU, wikitext-2 test, 20 × 512-token chunks = 5100 scored
tokens, identical for all runs; `benchmarks/2026-10-02-m4-quality.json`).

| Perplexity (lower is better) | llama.cpp | Sapient, per-32 activation scales | Sapient default (Q8_K activations) |
|---|---:|---:|---:|
| Qwen2.5-1.5B Q4_K_M | 11.720 | 11.738 (+0.15%) | 11.762 (+0.36%) |
| Llama-3.2-1B Q4_K_M | 16.259 | 16.464 (+1.26%) | 16.508 (+1.53%) |

Protocol check: on the first four chunks the running estimates track llama.cpp's
chunk by chunk (7.14 / 10.03 / 10.02 / 10.09 vs 7.08 / 9.92 / 9.92 / 10.02).

What the numbers say:
- The Q8_K activation format (default since July, adopted for speed) costs about
  **+0.2–0.3% perplexity** over the per-32 format. This is its first quality measurement.
- Sapient is **+0.4% (Qwen) to +1.5% (Llama-1B)** worse than llama.cpp on the same file.
  The README's old "zero quality loss" wording was not literally true; the loss is small.
- The absolute standard errors (±0.45, ±0.68) are much larger than these differences,
  but the runs score identical tokens, so the differences are paired. A per-token
  paired interval is not computed yet — treat sub-0.5% differences as indicative.

**Negative / blocked.**
- **Metal perplexity not measured.** `cargo build --features mlx` fails here: Xcode's
  Metal Toolchain component is not installed (`xcodebuild -downloadComponent
  MetalToolchain`). Installing a system component is outside what the loop does on its
  own, so the 4-bit re-quantisation cost stays unmeasured.
- No speed metric changed this iteration.

**Decision.** Merge on green CI: adds a metric (perplexity, one of the listed
benchmarks) and a gate; no performance or correctness regression (new unit test for the
log-likelihood helper; existing suites unchanged).

**Next hypothesis (iteration 3).** Find where Llama-3.2-1B's +1.3% comes from before
adding more quantisation. Two suspects, each testable with `eval-ppl`:
(1) the Q8_0 KV cache (llama.cpp keeps K/V in f16) — Llama-1B's head_dim is 64, so its
cache is Q8_0; (2) int8 activation quantisation in the Q6_K kernels (Llama-1B's tied
embedding/output matrix is Q6_K). Caveat to check first: `eval-ppl` uses `forward_all_logits`, which runs with
`use_cache = false` — it may never touch the quantised KV cache, in which case suspect
(1) is not exercised here and decode-time quality needs a cached-path variant of the
gate. If a suspect explains most of the gap, fix it when the
speed cost is within noise; otherwise record it. Also add a paired per-token interval to
`eval-ppl` so small differences can be called.

---

## Iteration 3 — 2026-10-02 — quality ablation; two protocol bugs and one real tokenizer bug

**Research.** Question from iteration 2: where does Sapient's perplexity gap to
llama.cpp come from? Technique: ablation with paired per-token statistics — switch off
one quantisation at a time and compare log-likelihoods token by token (a paired test is
far tighter than two aggregate perplexities). Sources: llama.cpp keeps K/V in f16 and
quantises activations per 256 (`block_q8_K`); standard paired-difference statistics.

**Corrections to iteration 2 (its numbers were wrong).**
1. *Chunk offset.* llama.cpp tokenises the file with the leading BOS and then cuts
   chunks; `eval-ppl` cut BOS-less tokens, so for models with a BOS every chunk sat one
   token later. The Llama-3.2-1B comparison was therefore not on identical tokens. The
   reported +1.53% gap was mostly this; the real figure is +0.42%.
2. *Stray BOS on Qwen* (below) put a wrong first token in every Qwen chunk. The reported
   +0.36% gap is really +0.09%.
`benchmarks/2026-10-02-m4-quality.json` is left as measured and marked superseded.

**Bug found and fixed (affects real chat, not just the benchmark).** `SapientTokenizer`
picked its BOS by name (`<s>`, `<bos>`, `<|begin_of_text|>`). Qwen2.5's vocabulary
contains `<s>` as an ordinary BPE token (id 128245) and the model has no BOS, so every
Qwen prompt — `chat`, `serve`, `run` — was sent with a stray `<s>` in front. BOS
detection now requires a *special added* token. Two regression tests
(`plain_vocab_token_named_like_bos_is_not_bos`, `special_added_bos_is_detected_and_prepended`).
Llama's `<|begin_of_text|>` (special) is unaffected; verified against the tokenizer files.

**Change.**
- Tokenizer BOS fix (above).
- `eval-ppl`: chunk-offset fix; `--cached` (score through the KV cache, as decode does);
  `--dump-nll <file>` (per-token log-likelihoods).
- `scripts/ppl_paired.py` (written by a Haiku subagent, reviewed): paired comparison of
  two dumps with a 95% interval.
- `SAPIENT_F32_ACT=1`: diagnostic switch that forces the f32-activation reference
  kernels, including the i8mm paths (the first version of the switch missed those and
  that run was discarded).

**Measured** (Apple M4 CPU, wikitext-2, 20 × 512-token chunks, 5100 tokens;
`benchmarks/2026-10-02-m4-quality-ablation.json`).

| Perplexity | Qwen2.5-1.5B Q4_K_M | Llama-3.2-1B Q4_K_M |
|---|---:|---:|
| llama.cpp | 11.720 | 16.259 |
| Sapient default (Q8_K int8 activations) | 11.730 (+0.09%) | 16.328 (+0.42%) |
| Sapient, per-32 int8 activations | 11.674 (−0.39%) | 16.296 (+0.23%) |
| Sapient, f32 activations | 11.638 (−0.70%) | 16.304 (+0.28%) |
| Sapient default, through the KV cache | 11.726 | 16.329 |

Paired, relative to Sapient default (95% interval):

| Switch | Qwen2.5-1.5B | Llama-3.2-1B |
|---|---|---|
| per-32 activation scales | −0.48% (−0.83 .. −0.13), significant | −0.20% (−0.40 .. +0.01), not significant |
| f32 activations | −0.79% (−1.13 .. −0.44), significant | −0.15% (−0.34 .. +0.05), not significant |
| KV cache in the path | −0.04% (−0.54 .. +0.47), none | +0.01% (−0.22 .. +0.24), none |

What this says:
- **Sapient's default is at llama.cpp parity on Qwen (+0.09%) and +0.42% on Llama-1B.**
- **The Q8_0 KV cache costs nothing measurable** on either model. Suspect (1) is cleared.
- **Int8 activation quantisation is the measurable cost on Qwen:** the default (per-256
  scales) is 0.79% worse than f32 activations and 0.48% worse than per-32 scales. On
  Llama-1B neither difference is significant.
- Llama-1B's remaining +0.28% to llama.cpp is present even with f32 activations, so it
  is not activation quantisation. Unexplained (candidates: Q6_K paths, f32 reduction order).

**Decision.** Merge on green CI. Correctness improves (a wrong token is no longer sent to
every Qwen prompt; the gate now scores the same tokens as llama.cpp); no speed path
changes by default (`SAPIENT_F32_ACT` is off unless set). The default activation format
stays Q8_K: it was adopted for a measured +6–45% decode gain and now has a measured
quality price of ≤0.5% on one model and nothing significant on the other. That is a
trade-off, recorded here, not a regression introduced by this change.

**Negative results.**
- The first `SAPIENT_F32_ACT` run was invalid (the switch did not cover the i8mm prefill
  kernels); discarded and re-run.
- No throughput or latency metric improved this iteration.

**Next hypothesis (iteration 4).** Back to the paper's critical path: the first rung of
VLA support. SmolVLA is built on SmolVLM2-500M; Sapient loads only SmolVLM-256M today.
Add SmolVLM2-500M to `sapient see` (same Idefics3 family, larger SigLIP + SmolLM2-360M),
verify it against the existing vision gates, and record its encode / prefill latency and
jitter with `scripts/bench_loop.py`. If the checkpoint needs architecture changes beyond
configuration, log what and scope them.
