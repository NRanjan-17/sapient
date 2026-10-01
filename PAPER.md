# PAPER.md — working paper plan

Maintained by the optimisation loop (`LOOP_LOG.md`). Every number here must come from
a script in this repo and a committed result file. Status: **plan only — no VLA
result exists yet.**

## Venue and timeline

- **MLSys 2027** (the stated target): submission **2026-10-30**, four weeks after this
  plan was written. **Not feasible:** Sapient cannot run any VLA today, and the
  SmolVLA port alone is estimated at 4–8 weeks. Decision (iteration 1): do not aim a
  full paper at this deadline.
- **Realistic targets:** RSS 2027 (extended abstract 2026-12-04, paper 2027-04-16) if
  the VLA port and the first deadline experiments land by about February 2027;
  otherwise the MLSys 2028 cycle (expected autumn 2027). CoRL 2027 dates are not
  announced.

## Problem

A robot control loop needs an action by a wall-clock deadline. On CPU-class edge
boards (Raspberry Pi 5, Jetson CPU), a VLA forward pass takes hundreds of
milliseconds to seconds and its latency moves with thermal state, page-cache state
and storage I/O. Existing work either makes policies tolerate late actions (real-time
chunking, async inference) or cuts average compute (early exit, caching). None found
schedules on-robot inference against a wall-clock deadline (see `NOVELTY.md`).

## Claim (current wording)

On a CPU-only edge board, a deadline scheduler inside the inference engine can keep
the deadline-miss rate of a VLA control loop below a stated threshold by choosing how
much computation to spend per step — exit depth, denoising steps, vision-feature
reuse — from live engine state (measured stage times, thermal headroom, weight
readiness), and returns the best available action when the deadline arrives.

Not claimed: a hard real-time guarantee. The system runs on Linux with `mmap` and
thermal throttling; the claim is a **measured tail bound and miss rate**.

## Method (planned — none of this is built)

1. **VLA support in Sapient.** SmolVLA first (SmolVLM2 backbone with layer skipping,
   flow-matching action expert, action chunking, robot-state input).
2. **Anytime knobs.** Variable denoising-step count; backbone exit depth (needs a
   multi-exit or distilled model — a training dependency this paper must own);
   vision-feature reuse across frames.
3. **Deadline scheduler.** Per-stage latency model fed by the engine's own timers,
   thermal governor state and page/weight readiness; picks the knob setting before
   each step and falls back to the previous chunk's remaining actions on a miss.
4. **Deadline-aware weight readiness** for models larger than RAM (prefetch ordered by
   the next stage's need).

## Experiments (planned)

| Experiment | Metric | Script | Status |
|---|---|---|---|
| LLM decode vs llama.cpp | tok/s, TTFT, peak RSS | `scripts/bench_loop.py` | **measured** (baseline below) |
| Vision encode latency and jitter | p50 / max / stdev, per stage | `scripts/bench_loop.py` | **measured** (M4; Pi by hand) |
| VLA observation-to-action latency | p50 / p99 / jitter, Hz | — | blocked: no VLA runs |
| Deadline-miss rate vs deadline | miss %, action quality | — | blocked |
| Thermal soak: miss rate over 30 min | miss %, temperature | — | blocked |
| Task success under deadline (sim or real arm) | success % | — | blocked |

## Baselines

| Baseline | Available on the M4 today |
|---|---|
| llama.cpp | yes — Homebrew 0.5.0 (build 11146) |
| Ollama | yes — 0.12.6 (not in the scripted suite yet) |
| ExecuTorch | no — not installed |
| MLC-LLM | no — not installed |
| vla.cpp / LeRobot (for VLA rows) | no |

## Results so far (not paper results — engine baselines)

Apple M4, 16 GB, CPU backend, 2026-10-02, `benchmarks/2026-10-02-m4-baseline.json`:

| | Sapient | llama.cpp (4 threads) |
|---|---:|---:|
| Qwen2.5-1.5B Q4_K_M decode, tok/s | 53.6 | 62.6 |
| Llama-3.2-1B Q4_K_M decode, tok/s | 66.5 | 75.4 |
| Qwen2.5-1.5B TTFT (bench prompt), ms | 362 | — |
| Peak RSS, Qwen2.5-1.5B, MB | 1177 | — |
| SmolVLM-256M image encode, p50 / max / stdev, ms | 606 / 659 / 30.8 | — |
| Binary size (CPU build), MB | 52.4 | — |

Run-to-run spread on this laptop is up to about 8% (a run minutes earlier read
71.2 tok/s on Llama-3.2-1B and 556 ms vision p50). Raspberry Pi 5 vision encode,
hand-timed the same day: 3.5 s (see `docs/BENCHMARKS.md`).

## Limitations (already known)

- No VLA runs on Sapient; every VLA number is absent.
- The Pi is reachable only for one pre-approved timing command; no scripted Pi suite.
- Jetson access needs a password and is not available to the loop.
- No WASM build and no `no_std` core exist.
- No quality metric (perplexity, task success) is implemented.
