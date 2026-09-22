# `c-chat` — SAPIENT from C

Thirty lines of C that load a model and stream a reply. It doubles as a compile-time
canary for the [C ABI](../../crates/sapient-capi/include/sapient.h): if this stops
building or linking, the ABI broke.

```bash
# 1. Build the C library (from the repo root)
cargo build --release -p sapient-capi

# 2. Build and run the example
cd examples/c-chat
make
./c-chat "Why is the sky blue?"

# Pick a different model (any alias from `sapient models`)
SAPIENT_MODEL=qwen2.5-0.5b-q4 ./c-chat "Write a haiku about Rust."
```

The first run downloads weights; later runs start from the cache.

## What it demonstrates

| | |
|---|---|
| Version handshake | compares `SAPIENT_API_VERSION` (header) against `sapient_api_version()` (library) |
| Options | starts from `sapient_options_default()` and overrides — the source-compatible pattern |
| Streaming | `sapient_chat_stream` with a callback; return `false` from it to cancel |
| Error handling | `sapient_error_t**` out-param, read then `sapient_error_free` |
| Ownership | every returned pointer freed exactly once |

## Linking

`make` links the shared library with an rpath into `target/release`. For a
self-contained binary, `make static` links `libsapient.a` instead.

The macOS framework list in the `Makefile` is what the Rust dependency graph pulls in —
`SystemConfiguration`/`CoreFoundation` for reqwest's proxy lookup, `Metal`/`QuartzCore`
for wgpu-hal. It mirrors `scripts/package-swift.sh`, which learned the same list the hard
way (see `docs/MOBILE.md`).

Once `sapient.pc` is installed, all of that collapses to:

```bash
cc -std=c11 -o c-chat main.c $(pkg-config --cflags --libs sapient)
```
