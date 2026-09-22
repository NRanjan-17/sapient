# C-ecosystem compatibility

> ## STATUS: IMPLEMENTED (core) — 2026-09-22
>
> §4.1, §4.2 (build), §4.3, §4.5, §4.6 and §4.7 have **shipped** as
> [`crates/sapient-capi`](../crates/sapient-capi). A real C program loads a model and
> streams a reply — see [`examples/c-chat`](../examples/c-chat).
>
> **Still open:** §4.4 (a Python binding) and shipping `libsapient` as a **release
> artifact** (it builds, but `release.yml` does not yet attach it). §3 (licensing) is
> unchanged and remains a decision, not a task.
>
> **Drafted:** 2026-09-22 · **Against:** SAPIENT v0.6.0, `main` @ `a0965a5`
> **Question:** from a community point of view, what would make SAPIENT compatible with
> the current C ecosystem? (Setting aside the mobile/NDK C specifics in
> [`MOBILE.md`](MOBILE.md), which are a separate concern.)

---

## 1. The reframe: C as an *interface*, not an *implementation*

A related analysis, [`reports/C-REWRITE-FEASIBILITY.md`](../reports/C-REWRITE-FEASIBILITY.md),
concluded that rewriting SAPIENT's hot paths **in** C buys nothing — Rust and C share the
LLVM backend, the kernels already use NEON intrinsics and inline `asm!`, and decode is
memory-latency-bound anyway.

**This document is about the opposite direction, and the two do not conflict.** Keeping
100% Rust internals while *exposing* a C ABI makes SAPIENT a first-class citizen of the C
world at no cost to the engine.

The C ABI is the universal calling convention. Python (ctypes/cffi), Go (cgo), Node
(N-API), C#, Java (JNI), Julia, Zig, Ruby, PHP and Lua bind to C and to nothing else.
This is the mechanism behind llama.cpp's ecosystem: `llama.h` is why `llama-cpp-python`,
Ollama, LM Studio and `node-llama-cpp` exist at all. None of them are written in C++;
they all *bind* to it.

---

## 2. Where SAPIENT stood before this work

All verified against the tree at `a0965a5`, *before* `sapient-capi` landed. Kept as the
rationale for why the crate exists.

| Capability | Status |
|---|---|
| Hand-written C ABI | **None.** The single `extern "C"` in the workspace is an *import* — `crates/sapient-audio/src/permissions.rs:108`, the ObjC runtime for the macOS mic prompt |
| `cbindgen.toml` / `sapient.h` | Absent |
| pkg-config `.pc`, CMake config package | Absent — the `CMakeLists.txt` files present are React Native codegen artifacts under `sdks/react-native/` |
| Shipped shared library | **No.** `release.yml` ships the CLI (`.tar.gz`/`.zip`), a Swift XCFramework, an Android AAR and npm packages — no `libsapient.{so,dylib,dll}` + header |
| SDKs | TypeScript, React Native. **No Python** |

### Why `sapient-ffi` does not already cover this

`crates/sapient-ffi` declares `crate-type = ["lib", "staticlib", "cdylib"]`, so a shared
object *is* produced — but its exported symbols are **UniFFI scaffolding**, not a designed
C API:

- The surface is **generated**, and its shape changes between UniFFI releases.
- It traffics in `RustBuffer` and opaque handles with a bespoke serialisation contract.
- It carries a **contract checksum that aborts on mismatch** — the crate pins
  `uniffi = "=0.29.3"` exactly for this reason (see the comment in its `Cargo.toml`).

That is a correct design for what it is: a codegen substrate for Swift and Kotlin. It is
not something a C developer can `#include`, and it should not be repurposed as one.

---

## 3. The blocker that outranks all the engineering

**AGPL-3.0-only is why a C ABI alone will not produce community bindings.**

Linking creates a combined work. Anything linking `libsapient` must therefore itself be
AGPL, or hold a commercial license from OpenHorizon Labs. In practice that excludes:

- every closed-source product,
- most corporate open-source policies (Google bans AGPL internally; many enterprises follow),
- and even permissive OSS projects, which decline viral dependencies on principle.

The contrast with the incumbents is the whole story: **llama.cpp, ggml and whisper.cpp are
all MIT.** That is not incidental to their ecosystems — it is the cause of them.

**This is a deliberate business decision, not an oversight.** `CLAUDE.md` states the
position plainly: the AGPL copyleft is "the moat," and the commercial license is the paid
track. The options below are therefore presented as a choice for OpenHorizon Labs to make,
not a defect to fix.

| Option | Effect | Cost / caveat |
|---|---|---|
| **Linking exception scoped to the C API surface** — in the manner of the GCC Runtime Exception or the GPL Classpath Exception | Third parties may link and publish bindings; the engine stays AGPL, so engine modifications still flow back | Narrowest possible concession, most tailored to the goal — **the recommended option if adoption is wanted without surrendering the moat** |
| **LGPL-3.0 for a `sapient-capi` crate only** | Well-understood by legal teams; dynamic linking unencumbered | Static linking still carries relinking obligations — awkward for embedded and mobile consumers |
| **Keep AGPL-3.0-only unchanged** | Moat and commercial funnel both intact | Community bindings stay sparse; "C-ecosystem compatibility" then means *enterprise integration enablement*, which is a different and entirely valid goal |
| MIT/Apache the header alone | Lets others publish binding *source* | Does not address linking — largely cosmetic |

> **Not legal advice.** Any of these requires counsel, particularly given the inbound
> relicensing grant in `CONTRIBUTING.md` and the dual-license invariants in `CLAUDE.md`.

---

## 4. The engineering, ranked

Worth doing under **any** licensing outcome — enterprise commercial licensees need exactly
the same artifacts.

### 4.1 A hand-written `sapient-capi` crate exporting `sapient.h` — SHIPPED

A narrow, deliberate surface — **not** a re-export of UniFFI's ABI (§2). Sketch only:

```c
sapient_session_t *sapient_session_load(const char *model,
                                        const sapient_opts_t *opts,
                                        sapient_error_t **err);

int  sapient_chat(sapient_session_t *s, const char *prompt,
                  char **out, sapient_error_t **err);

int  sapient_chat_stream(sapient_session_t *s, const char *prompt,
                         bool (*on_token)(const char *tok, void *user_data),
                         void *user_data, sapient_error_t **err);

void     sapient_string_free(char *s);
void     sapient_session_free(sapient_session_t *s);
uint32_t sapient_api_version(void);
```

**Rules that make or break it:**

| Rule | Why |
|---|---|
| Opaque handles only; no Rust types cross the boundary | Layout stability |
| `catch_unwind` on **every** entry point | A panic unwinding into C is undefined behaviour |
| An explicit `*_free` for everything handed out | C has no drop glue |
| UTF-8 `char*` in, caller-freed `char*` out | Lowest common denominator across every binding language |
| Errors as out-params, never `errno` or a global | Thread safety |
| A `bool` return from the token callback | Cancellation, matching the existing `TokenListener` contract |

**The async problem is already solved.** `sapient-ffi`'s `run_async` (spawn onto a runtime
worker, `block_on` the `JoinHandle`) exists because `Pipeline` internals call
`tokio::task::block_in_place`, which panics on a non-worker thread. That pattern transfers
to the C API unchanged.

### 4.2 Ship the library as a release artifact — BUILDS, NOT YET ATTACHED TO RELEASES

`libsapient.{so,dylib,dll}` + `sapient.h` + a `.pc` file, per platform and per variant
(cpu / gpu / metal), alongside the existing CLI archives. Today there is nothing to link
against without building from source.

### 4.3 Build-system integration — SHIPPED

A pkg-config `.pc` and a CMake config package, so that `find_package(sapient)` and
`pkg-config --cflags --libs sapient` both work. C developers expect these; without them
every integration is bespoke.

### 4.4 A Python binding — OPEN

The largest ecosystem in AI, and SAPIENT currently has no entry to it. Either cffi over the
new C ABI, or PyO3 directly (better ergonomics, more work). This is where most of the
potential community actually is.

### 4.5 An ABI-stability guard in CI — SHIPPED

A symbol-and-signature snapshot test that fails on unintended change, plus a
`SAPIENT_API_VERSION` macro and the `sapient_api_version()` accessor above.

Make this an **explicit gate**, not a convention. Precedent from
[`reports/TEST-RUN-M2-2026-09-22.md`](../reports/TEST-RUN-M2-2026-09-22.md): two
feature-gated suites were silently passing **vacuously** in CI for want of one
`--features` flag. An unguarded ABI would fail the same way.

### 4.6 `examples/c-chat/` — SHIPPED

Thirty lines of C and a Makefile. Doubles as documentation and as a compile-time canary
for the header.

### 4.7 Document the threading contract — SHIPPED

`Pipeline.engine` is an `Arc<Mutex<ForwardEngine>>` and the engine is **single-sequence**
(`forward_logits(&[u32])`, KV cache `[1, n_kv, max_seq, head_dim]`). C callers will assume
a handle can be driven from N threads.

State explicitly what is `Send`, what serialises on the engine lock, and whether two
sessions may run concurrently. Absent that, deadlock reports are inevitable.

---

## 5. Effort and the decision

The engineering is **modest** — roughly a few weeks for §4.1–4.3, §4.6 and §4.7. Nothing
in the architecture resists it, and `sapient-ffi` has already proven the async, streaming
and cancellation patterns survive an FFI boundary.

**The license is the actual variable.**

- A C ABI **under AGPL-3.0-only** yields *enterprise integrations that buy commercial
  licenses* — a real and viable business.
- It does **not** yield the llama.cpp-style binding ecosystem, because that ecosystem is
  composed of people who cannot use AGPL code.

Both are legitimate destinations. The technical roadmap in §4 serves either one; decide
which is being built, and it does not change.

---

## 6. Related documents

| Document | Relevance |
|---|---|
| [`reports/C-REWRITE-FEASIBILITY.md`](../reports/C-REWRITE-FEASIBILITY.md) | C as an implementation language — the opposite question, answered "no" |
| [`reports/TEST-RUN-M2-2026-09-22.md`](../reports/TEST-RUN-M2-2026-09-22.md) | Source of the vacuous-CI-gate precedent cited in §4.5 |
| [`MOBILE.md`](MOBILE.md) | The existing UniFFI/Swift/Kotlin surface and its NDK C toolchain constraints |
| `CLAUDE.md` | Dual-license invariants; the `sapient-ffi` design rules |
| `CONTRIBUTING.md` | The inbound relicensing grant that any license change must account for |

---

*The core is implemented; see `crates/sapient-capi/README.md` for the invariants that
must hold when changing it. The two open items are §4.4 (Python) and attaching
`libsapient` to releases (§4.2).*
