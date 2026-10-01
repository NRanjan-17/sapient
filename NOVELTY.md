# NOVELTY.md — prior-art map for the paper

Maintained by the optimisation loop (`LOOP_LOG.md`). Updated whenever an idea is
proposed, narrowed, or dropped. Last full search: **2026-10-02** (iteration 1).

Rows marked ✓ were checked against the arXiv abstract or official page. Rows marked
(U) come from search summaries only — the link resolves but the claim was not
confirmed from the primary source. Re-verify any (U) row before citing it.

## Status of the proposed contribution

**Proposed:** deadline-aware VLA inference on edge CPUs — a bounded observation-to-action
latency for robot control loops.

**Verdict (2026-10-02): partially covered. Narrowed — see "What we keep".**
No paper found claims a wall-clock deadline for *on-robot* VLA inference on CPU-class
boards, and no Raspberry-Pi-class VLA latency study was found. But two of the four
proposed ingredients are already published, and "guaranteed worst-case" is not a
claim a Linux + mmap + thermal-throttling system can defend.

## What exists in Sapient today (checked in the code, 2026-10-02)

| Ingredient named in the goal | In the repo? |
|---|---|
| Anytime / early-exit inference | **No.** No multi-exit model, no deadline API. |
| Entropy-based dynamic speculative decoding | **No.** `sapient-generate/src/speculative.rs` drafts a fixed K = 5 tokens; nothing adapts K. |
| NVMe layer streaming with deadline-aware prefetch | **No.** Larger-than-RAM models rely on plain `mmap` (zero-copy MoE expert views); no prefetch logic. |
| Persistent KV cache | **Partly.** In-memory prefix KV reuse across consecutive calls (`Pipeline::enable_prefix_cache`); nothing persisted, nothing across image frames. |
| Any VLA model | **No.** Sapient cannot load SmolVLA, pi0 or OpenVLA. Perception half only (SigLIP tower + LLM). |
| Engine-side thermal governor | **Yes** (`sapient-backends-cpu/src/thermal.rs`), plus mobile thermal hooks. |
| Hand-written ARM int8 kernels, per-stage timing | **Yes.** |

So three of the four named ingredients are proposals, not code, and every VLA
experiment is blocked on a VLA port.

## Closest prior work

### (d) KV / token reuse across frames — covered
| Work | Venue | What it does |
|---|---|---|
| [VLA-Cache](https://arxiv.org/abs/2502.02175) (U) | NeurIPS 2025 | Reuses KV for visually static tokens across frames, training-free. This *is* idea (d). |
| [Adaptive Visual Token Caching](https://arxiv.org/abs/2602.00686) (U) | arXiv 2026 | Learned token caching. |
| [Text-Vision Synergistic Token Caching](https://arxiv.org/abs/2609.34319) (U) | arXiv 2026 | Training-free token caching. |
| [Characterizing VLAs across XPUs / DP-Cache](https://arxiv.org/abs/2604.24447) ✓ | arXiv 2026 | Caching + fusion, up to 6× on edge NPUs. |

### (b) Speculative decoding on action tokens — covered
| Work | Venue | What it does |
|---|---|---|
| [Spec-VLA](https://arxiv.org/abs/2507.22424) (U) | arXiv 2025 | Speculative decoding for OpenVLA with relaxed acceptance, 1.42×. |
| [WA-SpecDec](https://arxiv.org/abs/2608.08725) ✓ | arXiv 2026 | Scene-aware dynamic acceptance, 1.5× over VLA speculative decoding. |
| [AdaEDL](https://arxiv.org/abs/2410.18351) (U) | NeurIPS-W 2024 | Entropy-based early draft stopping — the entropy mechanism itself. |
| [SpecDec++](https://arxiv.org/abs/2405.19715) (U) | arXiv 2024 | Learned adaptive draft length. |
| [OpenVLA-OFT](https://arxiv.org/abs/2502.19645) (U), [PD-VLA](https://arxiv.org/abs/2503.02310) (U) | arXiv 2025 | Parallel decoding removes the autoregressive action tokens (b) depends on. |

Also: speculative decoding only applies to autoregressive action heads (OpenVLA,
pi0-FAST). SmolVLA and pi0 use flow matching — nothing to speculate on.

### (a) Early exit / anytime — covered for compute budgets, open for wall-clock deadlines
| Work | Venue | What it does |
|---|---|---|
| [DeeR-VLA](https://arxiv.org/abs/2411.02359) (U) | NeurIPS 2024 (venue U) | Multi-exit MLLM; exits under average/peak *compute* budgets. |
| [Decoupled Early Exits](https://arxiv.org/abs/2609.29382) ✓ | arXiv 2026 | Exits in backbone, action expert and denoise steps; 79.2% latency cut. Closest to (a); per-task, no deadline. |
| [CEED-VLA](https://arxiv.org/abs/2506.13725) (U), [MoLe-VLA](https://arxiv.org/abs/2503.20384) (U), [DySL-VLA](https://arxiv.org/abs/2602.22896) (U) | 2025–26 | Early-exit decoding / layer skipping, no deadline. |
| [Scheduling Real-time DL Services as Imprecise Computations](https://arxiv.org/abs/2011.01112) (U) | RTCSA 2020 | Mandatory + optional DNN stages against deadlines — the direct ancestor, for CNNs. |
| [Anytime-Lidar](https://arxiv.org/abs/2208.12181) (U), [VALO](https://arxiv.org/abs/2409.11542) (U) | — | Deadline-aware anytime LiDAR detection (perception, not policy). |

An "anytime flow-matching policy" with an explicit deadline: **not found**.

### Async / real-time chunk execution — makes policies tolerate delay, none bound it
[Real-Time Chunking](https://arxiv.org/abs/2506.07339) (U, NeurIPS 2025) ·
[A2C2](https://arxiv.org/abs/2509.23224) (U) · [SmolVLA async](https://arxiv.org/abs/2506.01844) (U) ·
[VLASH](https://arxiv.org/abs/2512.01031) ✓ · [Jetson-PI](https://arxiv.org/abs/2607.12659) ✓ (CoRL 2026, Jetson Orin GPU) ·
[FASTER](https://arxiv.org/abs/2603.19199) (U).

### Deadline-aware serving — off-robot
[Robion](https://arxiv.org/abs/2609.12075) ✓ (SLO-aware VLA serving on multi-GPU edge servers, 98% SLO attainment) ·
[Chronos](https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2026.1873627/full) (U, schedulability for LLM serving) ·
[RT-LM](https://arxiv.org/abs/2309.06619) (U, RTSS 2023).

### (c) Larger-than-RAM execution — covered for throughput, open for deadlines
[LLM in a Flash](https://arxiv.org/abs/2312.11514) (U) · [EdgeMoE](https://arxiv.org/abs/2308.14352) (U) ·
[ActiveFlow](https://arxiv.org/abs/2504.08378) ✓ (cross-layer DRAM/flash preloading — closest) ·
[Lever](https://arxiv.org/abs/2605.16786) ✓ (draft in DRAM, flash-resident target verifies; already combines (b)+(c) for LLMs).
Deadline-aware weight streaming: **not found**.

### Competing runtimes and edge numbers
[vla.cpp](https://arxiv.org/abs/2606.08094) ✓ (C++ runtime, 11 VLAs; via search summary (U): SmolVLA 65 ms on AGX Orin, 142 ms on Orin Nano) ·
[Embodied.cpp](https://arxiv.org/abs/2607.02501) ✓ ·
[Lite VLA (CPU-bound)](https://arxiv.org/abs/2511.05642) ✓ (CPU-only on-board VLA; closest hardware class, no numbers in abstract) ·
[LiteVLA-Edge](https://arxiv.org/abs/2603.03380) (U) · [NanoVLA](https://arxiv.org/abs/2510.25122) (U) ·
[VLA-Perf](https://arxiv.org/abs/2602.18397) (U).
"A VLA runtime" is therefore not a contribution by itself.

### Thermal
[MNN-AECS](https://arxiv.org/abs/2506.19884) (U) — engine-level core selection for energy.

## What we keep, drop, and change

- **Drop (b)** entropy-adaptive speculative decoding as a claimed contribution. Prior art
  covers both the mechanism (AdaEDL) and its use on VLAs (Spec-VLA, WA-SpecDec).
- **Drop (d)** KV reuse across frames as a claimed contribution (VLA-Cache and
  follow-ons). It can still be an engineering feature.
- **Keep (a) + (c), coupled:** a wall-clock deadline scheduler on CPU-only boards that
  picks how much computation to spend (exit depth, denoise steps, vision reuse) from
  live state — I/O/prefetch readiness, thermal headroom, measured stage times.
- **Change the claim wording:** not "guaranteed worst-case latency". The defensible
  claim is a **measured tail bound / deadline-miss rate with an anytime fallback**.
- **Own the training dependency:** anytime exits need a multi-exit or distilled model.
- **Second, cheaper angle:** the first latency and jitter characterisation of a VLA on
  a Raspberry-Pi-class CPU (none found), including the thermal dimension.

## Open questions for the next search
- Any 2026 paper on wall-clock-deadline flow-matching policies (search again before committing).
- Whether vla.cpp reports CPU-only numbers.
- Real-time-systems venues (RTSS/RTAS/EMSOFT 2025–26) for "VLA" or "robot foundation model".
