# SAPIENT and C — what we built, what we measured, what we recommend

**A meeting brief.** Written in plain English. No prior knowledge of the codebase assumed.

**Date:** 2026-09-22 · **Branch:** `c` · **Engine:** SAPIENT v0.6.0

---

## The one-page summary

We were asked two separate questions that are easy to confuse:

| Question | Short answer |
|---|---|
| **1. Should we let C programs *use* SAPIENT?** | **Yes — and it is now built and working.** |
| **2. Should we *rewrite* SAPIENT in C?** | **No.** Measured gain is about 5%, and it is not because of the language. |

The first is about opening SAPIENT to other people. The second is about throwing away the Rust code. They have opposite answers, and this document keeps them apart.

---

# PART 1 — What we changed in SAPIENT (already done)

## 1.1 The problem

Today SAPIENT can be used three ways: the `sapient` command in a terminal, a Swift or Kotlin app on a phone, or over HTTP.

If you write **Python, Go, Node, C#, Java, Julia or Zig**, there was no way in.

Almost every programming language can call C code. That is the one thing they all agree on. So the standard way to open a library to everyone is to give it a **C interface** — a file called a "header" that lists the functions other programs can call.

This is exactly why llama.cpp has a large community. Its `llama.h` file is why `llama-cpp-python`, Ollama and LM Studio exist. None of those are written in C++; they all just *call* it.

## 1.2 What we built

A new component, `crates/sapient-capi`, which produces:

- **`libsapient`** — the library file other programs link against
- **`sapient.h`** — the list of functions they can call

Here is the whole thing from a C program's point of view:

```c
#include "sapient.h"

sapient_error_t *err = NULL;
sapient_options_t opts = sapient_options_default();
opts.max_tokens = 128;

sapient_session_t *s = sapient_session_load("smollm2-135m-q4", &opts, &err);

char *reply = NULL;
sapient_chat(s, "Name one primary colour.", &reply, &err);
puts(reply);                  // -> "The primary colour is actually blue."

sapient_string_free(reply);   // give back what you were given
sapient_session_free(s);
```

21 functions in total: load a model, chat, stream a reply token by token, cancel mid-reply, read the conversation history, reset it, list available models, choose CPU or GPU.

## 1.3 The four rules that make it safe

These matter because a mistake here crashes *other people's programs*, not ours.

**Rule 1 — a crash must never escape into C.**
Rust and C handle failure differently. If Rust panics and that panic reaches C code, the program is corrupted in ways that are very hard to debug. So every single entry point is wrapped in a catcher that turns a panic into a polite error code.

**Rule 2 — return the real error, not a generic one.**
We got this wrong at first. The original code returned the same "bad argument" code no matter what went wrong, so a caller could not tell "you passed nonsense" apart from "the model failed". Now each error carries its own code. There is a test that fails if anyone undoes this.

**Rule 3 — anything handed out must be handed back.**
Rust cleans up automatically; C does not. So every piece of text we return has a matching "I'm done with this" function. Miss one and the program slowly eats memory until it dies.

**Rule 4 — the header and the code must always agree.**
If the header says a function exists and the library disagrees, programs break at link time with a confusing message. A test compares them on every build.

## 1.4 Supporting pieces

| Piece | What it does |
|---|---|
| `scripts/install-capi.sh` | Installs the library so `pkg-config` and CMake find it automatically |
| `examples/c-chat/` | A 30-line working C program. If it stops compiling, we broke the interface |
| `tests/abi_surface.rs` | The agreement check from Rule 4 |

---

# PART 2 — Test cases and results

Everything below was run on an **Apple M2, 8 cores, 16 GB, macOS 27.0**, with Rust 1.98.1.

## 2.0 What you need to run any of this

### Minimum, to build and test the engine

| Requirement | Version | Notes |
|---|---|---|
| Rust | **1.82 or newer** | The project's stated minimum. Measurements here used 1.98.1 |
| Disk | **~12 GB free** | Build artefacts alone reach 9 GB |
| RAM | **8 GB** | 16 GB recommended; some skipped tests need far more |
| OS | macOS, Linux or Windows | No Python, Docker or CUDA needed |

```bash
cargo test --workspace -- --test-threads=1
```

### To build and test the C interface

| Requirement | Notes |
|---|---|
| A C compiler | clang or gcc. Already present on macOS with Xcode tools, or `build-essential` on Linux |
| `make` | Only for the example program |
| Linux only | `libasound2-dev` — the CLI links audio libraries by default |

```bash
cargo build --release -p sapient-capi
cd examples/c-chat && make && ./c-chat "hello"
```

### To run the C++ kernel comparison

| Requirement | Notes |
|---|---|
| A C++17 compiler | clang 15+ or gcc 11+ |
| **An ARM64 processor** | Apple Silicon, Raspberry Pi 5, or an ARM server. **The kernels use ARM instructions and will not build on Intel or AMD** |
| ARM dot-product support | Present on every Apple Silicon chip and on Cortex-A76 and newer |
| `python3` | Only for the analysis scripts, not the measurements |

```bash
cd benchmarks/lang-comparison/rust/harness
cargo bench --bench kernels
```

### For the heavier tests we skipped

20 tests are skipped by default because they download large models:

| Test | Needs |
|---|---|
| Mixtral end-to-end | **26 GB download, 32 GB RAM** |
| GLM-4.5-Air | **63 GB download, 96 GB RAM** |
| Whisper, Kokoro, vision tests | 0.1–2 GB downloads plus network |

Run them with `cargo test --workspace --release -- --ignored`. The machine used for this report has 16 GB, so the two largest were not run.

### What this report was measured on

| | |
|---|---|
| C interface tests, leak check | **Apple M2**, 8 cores, 16 GB, macOS 27.0, Rust 1.98.1 |
| Our three optimisation attempts | Same M2 |
| Kernel and engine comparison | **Apple M5**, 10 cores, 24 GB, macOS 26.6, Rust 1.98.1, clang 21 |

Two different machines, so **do not compare absolute timings between sections** — only ratios within a section.

## 2.1 The full test suite

```
cargo test --workspace -- --test-threads=1
```

| | Before our work | After |
|---|---:|---:|
| Tests passing | 316 | **334** |
| Tests failing | 0 | **0** |
| Skipped (need big model downloads) | 20 | 20 |

We added **18 tests**: 13 checking the C interface logic, 5 checking the header and code agree.

## 2.2 Testing the C interface *from C*

Rust tests call the functions as Rust. That does not prove a C program can use them. So we wrote a test **in C** and ran it. 32 checks:

| Group | What it checks | Result |
|---|---|---|
| Version | The header and library agree on the version number | pass |
| Catalog | Model list works, asking past the end returns nothing instead of crashing | pass |
| **Error paths** | Passing nothing where a model name is expected gives a clear message, not a crash | pass |
| Loading | A real model loads, defaults are correct | pass |
| Normal chat | Returns text, conversation history grows correctly | pass |
| **Streaming = normal** | Text streamed piece by piece is **byte-for-byte identical** to the text returned at the end | pass |
| Cancelling | Stopping mid-reply actually stops it, and returns the partial text | pass |
| History and reset | Three conversations recorded, reset clears them | pass |
| Cleanup | Freeing nothing does not crash | pass |

**32 passed, 0 failed.**

The streaming check is the most important one. If pieces of text were dropped or repeated on the way out, this is where it would show.

## 2.3 Memory leak check

We claimed "everything handed out gets handed back". We then proved it:

```
$ leaks --atExit -- ./abi_test
Process 19593: 0 leaks for 0 total leaked bytes.
```

Over a run that created and destroyed a model session, several replies, error objects and history entries — **zero leaked bytes**.

---

# PART 3 — The C experiment: does C actually run faster?

## 3.1 How the question was answered

Comparing SAPIENT against llama.cpp cannot answer this. They differ in dozens of ways and the language is only one of them.

The only fair test is **SAPIENT against itself**: take SAPIENT's own maths code, rewrite it in C++, change nothing else, and race the two.

That is what was done. Two pieces of maths were chosen because the profiler says they are **96.5% of all the time spent generating text**.

The test ran in stages, each one a gate:

| Stage | Question asked | Result |
|---|---|---|
| Port | Can it be rewritten faithfully in C++? | Yes — about 360 lines |
| **Correctness** | Does the C++ produce *exactly* the same numbers? | **Yes — bit-for-bit, 7 of 7 tests** |
| Machine code | Do the two compilers produce different instructions? | **Yes, meaningfully different** |
| Speed, in isolation | Is the C++ faster on its own? | One kernel **1.298x**, the other **1.000x — no difference at all** |
| Attribution | Can Rust be changed to get the same win? | **Yes, about 40% of it** |
| **In the real engine** | Does any of it survive when actually generating text? | **C++ +5.5%** |

## 3.2 The headline numbers

| What was measured | Result |
|---|---:|
| C++ vs Rust, generating text end to end | **+5.5%** (confidence range +3.4% to +6.4%) |
| The same fix applied **in Rust instead** | **+2.2%** |
| Q4_K maths on its own | 1.298x |
| **Q6_K maths on its own** | **1.000x — literally no difference** |

That last row is the important one. The C++ version of Q6_K ran **26% fewer instructions** and finished in exactly the same time. The computer was waiting for memory, not for maths.

## 3.2b We re-ran it ourselves, per kernel, on a different chip

The numbers above come from an M5. We ran the same two kernels separately on an **M2** to see whether they hold. Speedup against plain Rust, higher is faster:

| Kernel | Rust unrolled | Rust unrolled + vectail | Rust unchecked | C++ |
|---|---:|---:|---:|---:|
| **Q4_K** (64% of decode) | 1.124x | **1.153x** | 1.075x | **1.229x** |
| **Q6_K** (32% of decode) | — | — | — | **1.035x** |

Five matrix sizes per kernel, Criterion, 30 samples each. Four findings:

**Q4_K replicates. Q6_K does not.** The M5 study measured Q6_K at exactly **1.000x**. On the M2 it is **1.035x**, consistently, across all five sizes (1.033–1.036). Same code, same compilers, different chip.

That matters: **the language gap depends on the processor.** One machine is not enough to decide this, which is precisely the caution the original study raised about a Raspberry Pi.

**Rust recovers 70% of the gap here, not 40%.** The existing bit-identical Rust variant with the loop unrolled and the float tail vectorised reaches 1.153x against C++'s 1.229x — closing **70.9%** of the distance, stable across every size (69.5–74.7%).

**Removing bounds checks makes it slower.** The `unchecked` variant measures **1.075x — worse than the checked one at 1.124x**. Reproduced independently. Reaching for `unsafe` here costs performance as well as safety.

**The ranking never wavers:** C++ > vectail > unrolled > unchecked > plain Rust, at all five sizes. None of this is noise.

Raw data: `benchmarks/lang-comparison/results/M2_KERNEL_RESULTS.md`.

## 3.3 Why the 5% is not about the language

Both Rust and C++ are translated by the same underlying compiler technology. What differed was two decisions:

1. The C++ compiler unrolled a loop; the Rust compiler did not.
2. The C++ compiler vectorised a small piece of leftover work; the Rust compiler did not.

Both are ordinary things you can write in Rust by hand. Doing so recovered about 40% of the gap.

So the honest description is: **two compiler decisions, not two languages.**

## 3.4 We tried to optimise the C++ further — and made it slower

To check whether the C++ had headroom left, we hand-optimised the hottest kernel three ways. All three produce **identical numbers**. All three are **slower**.

| Version | What we changed | Speed |
|---|---|---:|
| Original C++ | — | **1.000x** |
| Attempt 1 | Precompute scale values; unroll the 4-row loop | **0.784x** (22% slower) |
| Attempt 2 | Attempt 1 **plus** a smarter way of adding up results | **0.924x** (7.6% slower) |

Here is the change that *did* help, from Attempt 1 to Attempt 2 (+18% on its own). The original adds up four numbers immediately, 32 times per block:

```cpp
// original: a horizontal add every time, 32 per super-block
const int32_t dot_lo = vaddvq_s32(vdotq_s32(vdotq_s32(zero, lo0, xlo0), lo1, xlo1));
isum[r] += (int32_t)sc1 * dot_lo;
```

Our version keeps them apart and adds up once at the end, 4 times per block:

```cpp
// ours: keep it as a vector, fold the scale in, reduce ONCE per block
const int32x4_t dlo = vdotq_s32(vdotq_s32(zero, lo0, xlo0), lo1, xlo1);
vacc = vmlaq_n_s32(vacc, dlo, (int32_t)SC[R][is]);
// ... later, once:
const int32_t isum = vaddvq_s32(vacc);
```

That was a genuine improvement. It still lost overall, because the other change — precomputing the scale values — pushed 64 bytes onto the stack per block and cost more than it saved.

**Conclusion: the C++ is already about as good as it gets. There is no easy 2x hiding in it.**

---

# PART 4 — How the comparison was done

This section exists because the *method* is what makes the numbers believable.

## 4.1 The settings used

| Setting | Value | Why it matters |
|---|---|---|
| Machine | Apple M5, 24 GB | Supports all the fast instructions being tested |
| Rust | 1.98.1 | **Its compiler is newer than the C++ one** — so if anything, this favours Rust |
| C++ | Apple clang 21 | |
| Model | qwen2.5-1.5b, 4-bit, 1.0 GB | A realistic size |
| Threads | Fixed at 10 | Stops thread count varying between runs |
| Temperature control | Turned off | Stops the machine slowing itself down mid-test |
| Output | Checked identical across versions | Proves we compared the same work |

## 4.2 The ordering trick

Computers get slower as they warm up, so whichever version runs second looks worse.

The fix: run them in a mirrored order — **Rust, C++, C++, Rust** — so the baseline is measured both first and last every time. Any drift shows up in the baseline itself and can be subtracted out.

## 4.3 The finding nobody expected

The test machine was **not stable**. The same program doing the same work measured anywhere from **21 to 54 tokens per second — a 2.5x swing.**

The suspected cause is macOS compressing the 1.3 GB of model weights in memory; once compressed, every access has to decompress first.

This is why the study reports confidence ranges instead of single numbers. **Any benchmark on this machine that does not account for this is unreliable** — including some of our own published figures.

## 4.4 Two mistakes we made and caught

Worth mentioning because both produced convincing wrong answers:

**Mistake 1 — random data is not valid data.** We filled test weights with random bytes. Four of those bytes are a special number format; random bytes make them "not a number". Every result came out as `nan` — and the bit-comparison *passed*, because two `nan`s have identical bits.

**Mistake 2 — the compiler deleted our benchmark.** The timing loop called the same function with the same inputs 60,000 times, so the compiler ran it once and reused the answer. 60,000 iterations "took" 0.0002 seconds. We had to explicitly block that optimisation.

## 4.5 Better ways to compare, for next time

| Approach | Why it is better |
|---|---|
| **Count instructions, not seconds** | Instruction counts do not change when the machine speeds up or slows down. Q6_K proves the point: 26% fewer instructions, identical time |
| **Test on a quiet Linux machine** | No memory compression, no efficiency-core scheduling |
| **Test on a Raspberry Pi and a server chip too** | A loop unroll that wins on an M5 may lose on a Pi, which has fewer registers |
| **Measure energy, not just speed** | On phones and edge devices, battery matters more than raw speed |
| **Vary the compiler version instead of the language** | The finding was "one compiler unrolled a loop". Testing compiler settings is far cheaper than rewriting in another language |

---

# PART 5 — Is it worth it, given this will be open to a wider community?

## 5.1 Two different things, two different answers

**Opening SAPIENT to C programs: clearly worth it. Already done.**

Cost was a few days. The benefit is that Python, Go, Node, Java, C# and Zig developers can now use SAPIENT without us writing anything for each one. That is how llama.cpp got its ecosystem.

**Rewriting SAPIENT in C: not worth it.**

| Argument | Detail |
|---|---|
| The gain is small | 5.5%, and about 40% of that can be had in Rust |
| It is not the language | Two compiler decisions, both reproducible in Rust |
| Half the evidence says zero | Q6_K measured exactly 1.000x |
| The scope is enormous | About 360 lines have a C++ version. SAPIENT is **53,000 lines** |
| We would lose the phone apps | Swift and Kotlin bindings are generated from Rust |
| We would lose memory safety | On code that runs untrusted model files |
| We already tried to improve the C++ | All three attempts were slower |

## 5.2 Straight trade-off: Rust vs C

### Staying in Rust

**Advantages**

| | |
|---|---|
| Memory safety | The engine parses model files downloaded from the internet. In Rust a malformed file is an error; in C it is a potential security hole. This is the single biggest one |
| The phone apps come free | Swift and Kotlin bindings are generated automatically from the Rust. A C port throws that away and you write them by hand |
| One toolchain, every platform | `cargo build` cross-compiles to Mac, Linux, Windows, Raspberry Pi, iOS and Android. No per-platform C compiler setup |
| No undefined behaviour | Buffer overruns, use-after-free and data races are compile errors, not 2am bugs |
| Fearless threading | The compiler rejects data races. The engine is heavily threaded, so this is used constantly |
| Dependencies are managed | One file lists them; no vendored headers, no build-system archaeology |
| The team already knows it | 53,000 lines of institutional knowledge |

**Disadvantages**

| | |
|---|---|
| The compiler sometimes optimises worse | Measured: rustc did not unroll a loop clang did. Worth 1.298x on one kernel |
| Some instructions are not available yet | `vmmlaq_s32` is still unstable in Rust, so we emit it via hand-written assembly instead of an intrinsic |
| Fewer people can read it | The C and C++ pool is much larger, which matters for outside contributors |
| Slower to compile | Minutes, not seconds |
| Hand-written assembly is awkward | Possible, but uglier than in C |

### Switching to C

**Advantages**

| | |
|---|---|
| Slightly better code generation, sometimes | +5.5% end to end, measured. About 40% of it is available in Rust anyway |
| Every instruction available immediately | No waiting for a Rust intrinsic to stabilise |
| Larger contributor pool | More people write C than Rust |
| Trivial to embed elsewhere | Though the C interface we just built already gives us this without rewriting anything |
| Faster compiles | |

**Disadvantages**

| | |
|---|---|
| A 53,000-line rewrite | 360 lines have a C++ version today. That is 0.7% |
| Memory safety gone | On code that parses untrusted files |
| The phone SDKs must be rebuilt by hand | Swift and Kotlin bindings no longer generate themselves |
| Build complexity explodes | A C compiler per platform, plus CMake or Make, plus dependency vendoring |
| Half the measured evidence says zero gain | Q6_K came back at exactly 1.000x |
| We already tried to improve the C++ | Three attempts, all slower |
| Years of tuning thrown away | Every fixed bug, every measured decision, re-litigated in a new language |

### The summary line

The gain is **real but small and mostly recoverable in Rust**. The cost is **a total rewrite plus the loss of memory safety and the mobile SDKs**. That is not a close call.

## 5.3 The scope point, concretely

"Convert it to C" sounds like one project. It is not. What has a C++ version today is **2 maths functions**. What does not:

- All the other maths (Q5_K, Q8_0, attention, convolutions, and the entire speech-synthesis set)
- **Anything for Intel and AMD processors** — the C++ work is ARM only
- Model file loading, tokenisers, chat formatting, the HTTP server, sampling, caching
- The four different model engines, plus speech-to-text, text-to-speech, and vision
- All three graphics-card backends

## 5.4 What to do instead

From the study's own recommendations, in order of value:

1. **Write the Intel/AMD maths kernels.** Right now every Intel and AMD user runs the slow path. This is far bigger than 5%. It is also where a careless C++ test would give a *falsely positive* answer — the C++ would look dramatically faster, and all of the difference would be work Rust simply has not been given yet.
2. **Ship the faster Rust kernel we already have** (measured 1.139x), after checking it on a Raspberry Pi.
3. **Fix a memory-detection bug on macOS** that silently disables a fast path — meaning some of our published Mac benchmarks may not be measuring what we think.
4. **Write fast maths for the speech synthesis**, which spends 17.8% of its time in slow code — ten times any other part.
5. **Use the newer instructions** available on M5 chips, which neither language currently uses.

None of these are "rewrite in C". All are Rust work.

---

# PART 6 — What is in this branch

| Path | What |
|---|---|
| `crates/sapient-capi/` | The C interface — library, header, tests |
| `examples/c-chat/` | Working C example |
| `scripts/install-capi.sh` | Installer |
| `benchmarks/lang-comparison/` | The C++ comparison: kernels, harness, raw results |
| `benchmarks/lang-comparison/cpp/variants/` | **Our three failed optimisation attempts, kept as a record** |
| `docs/C_AND_RUST_FINDINGS.md` | This document |

## Reproducing the numbers

```bash
# The test suite
cargo test --workspace -- --test-threads=1          # 334 pass

# The C interface, tested from C
cargo build --release -p sapient-capi
cd examples/c-chat && make && ./c-chat "hello"

# The failed optimisation attempts
cd benchmarks/lang-comparison/cpp/variants
c++ -O3 -std=c++17 -flto -ffp-contract=off -I../include \
    -o ab ../src/q4k.cpp q4k_vecacc_unroll.cpp ab_bench.cpp
./ab 35 60000
```

---

## One correction we owe

An earlier internal note of ours argued that Rust and C *must* produce identical machine code because they share the same compiler backend, so no speed difference was possible.

**That was wrong**, and the measurements here disprove it. The shared backend guarantees the same *available instructions*, not the same *decisions about how to use them*. For one kernel the difference was 1.298x.

The recommendation does not change — but the reason does. It is not "there is no difference". It is **"the difference is real, small, and we can capture it in Rust."**
