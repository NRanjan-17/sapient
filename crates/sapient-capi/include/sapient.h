/* SPDX-License-Identifier: AGPL-3.0-only
 * Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
 *
 * sapient.h — stable C ABI for the SAPIENT edge inference engine.
 *
 * SAPIENT runs language, vision and speech models entirely on-device. This header is the
 * surface every other language binds to: Python (cffi), Go (cgo), Node (N-API), C#, Java,
 * Julia, Zig. See docs/C-ECOSYSTEM.md.
 *
 *   Link:  -lsapient          (pkg-config --cflags --libs sapient)
 *   Build: cargo build --release -p sapient-capi
 *
 * ── Conventions ──────────────────────────────────────────────────────────────────────
 *
 *   Ownership   Every pointer this library returns to you is yours to free, with exactly
 *               one matching call: sapient_string_free / sapient_error_free /
 *               sapient_session_free. The only exceptions are sapient_version() and
 *               sapient_error_message(), which are owned by the library — do NOT free
 *               them. Freeing twice is undefined behaviour; freeing NULL is always safe.
 *
 *   Errors      Functions returning int yield SAPIENT_OK (0) or a negative
 *               SAPIENT_ERR_* code. Functions returning a pointer yield NULL on failure.
 *               Either way, detail is written to the `err` out-parameter when you pass
 *               one. Passing NULL for `err` is legal and means "status code only".
 *               An error you receive must be freed with sapient_error_free().
 *
 *   Strings     UTF-8, NUL-terminated, in both directions.
 *
 *   Blocking    Every call is synchronous. sapient_session_load() downloads weights on
 *               first use and can take minutes. Call from a background thread.
 *
 *   Threading   A session handle is safe to share across threads. Generation calls on the
 *               SAME session serialize on the engine's internal lock — they do not run in
 *               parallel. Different sessions do run concurrently.
 *
 *   Panics      Never cross this boundary. An internal failure surfaces as
 *               SAPIENT_ERR_PANIC rather than unwinding into your stack frame.
 */

#ifndef SAPIENT_H
#define SAPIENT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Version of THIS ABI, not of the engine. Compare against sapient_api_version() at
 * runtime to detect a header/library mismatch. Bumped on any breaking change here. */
#define SAPIENT_API_VERSION 1u

/* ── Status codes ──────────────────────────────────────────────────────────────────── */

#define SAPIENT_OK                    0  /* success                                     */
#define SAPIENT_ERR_INVALID_ARGUMENT -1  /* NULL pointer, bad UTF-8, out-of-range value */
#define SAPIENT_ERR_LOAD             -2  /* download / resolve / weight-load failure    */
#define SAPIENT_ERR_GENERATION       -3  /* failure after the model was loaded          */
#define SAPIENT_ERR_INTERNAL         -4  /* runtime join, encoding, poisoned lock       */
#define SAPIENT_ERR_PANIC            -5  /* caught Rust panic — always a bug, report it */

/* ── Backend selection ─────────────────────────────────────────────────────────────── */

#define SAPIENT_BACKEND_AUTO   0  /* this build's accelerator, falling back to CPU      */
#define SAPIENT_BACKEND_CPU    1  /* force CPU                                          */
#define SAPIENT_BACKEND_METAL  2  /* force Apple MLX/Metal (needs an mlx build)         */
#define SAPIENT_BACKEND_WGPU   3  /* force portable GPU: Vulkan/DX12/Metal (wgpu build) */

/* ── Opaque handles ────────────────────────────────────────────────────────────────── */

typedef struct sapient_session_t sapient_session_t;
typedef struct sapient_error_t   sapient_error_t;

/* ── Errors ────────────────────────────────────────────────────────────────────────── */

/* Status code carried by an error; SAPIENT_ERR_INVALID_ARGUMENT if err is NULL. */
int         sapient_error_code(const sapient_error_t *err);

/* Message text, valid until sapient_error_free(err). Do NOT free separately.
 * NULL if err is NULL. */
const char *sapient_error_message(const sapient_error_t *err);

/* Release an error. NULL-safe. */
void        sapient_error_free(sapient_error_t *err);

/* ── Strings ───────────────────────────────────────────────────────────────────────── */

/* Release a string this library returned through an out-parameter. NULL-safe.
 * Never pass a pointer from sapient_version() or sapient_error_message(). */
void sapient_string_free(char *s);

/* ── Library info ──────────────────────────────────────────────────────────────────── */

/* Engine version, e.g. "0.6.0". Owned by the library — do NOT free. */
const char *sapient_version(void);

/* ABI version of the loaded library. Compare with SAPIENT_API_VERSION. */
uint32_t    sapient_api_version(void);

/* Point the model cache at a directory (sets HF_HOME). Call before loading. */
int         sapient_set_cache_dir(const char *path, sapient_error_t **err);

/* ── Model catalog ─────────────────────────────────────────────────────────────────── */

/* Number of models in the curated catalog. */
size_t sapient_model_count(void);

/* Alias of catalog entry `index`, or NULL if out of range. Caller frees. */
char  *sapient_model_alias(size_t index);

/* Resolve an alias (fuzzy-matched) to its HuggingFace repo id.
 * On SAPIENT_OK, *out is a caller-owned string. */
int    sapient_resolve_alias(const char *name, char **out, sapient_error_t **err);

/* ── Options ───────────────────────────────────────────────────────────────────────── */

/* Generation options.
 *
 * ALWAYS start from sapient_options_default() and override the fields you care about —
 * that keeps your code source-compatible when fields are added.
 *
 * Negative floats and a negative top_k mean UNSET. Leaving all four sampling fields
 * unset selects greedy (deterministic) decoding. */
typedef struct {
    uint32_t    max_tokens;          /* hard cap on new tokens per reply               */
    float       temperature;         /* < 0 = unset                                    */
    float       top_p;               /* nucleus threshold 0–1; < 0 = unset             */
    int32_t     top_k;               /* < 0 = unset; 0 disables the filter             */
    float       repetition_penalty;  /* 1.0 = off; < 0 = unset                         */
    const char *system_prompt;       /* NULL = none; borrowed during the load call     */
    int         backend;             /* one of SAPIENT_BACKEND_*                       */
} sapient_options_t;

/* 512 tokens, greedy decoding, no system prompt, automatic backend. */
sapient_options_t sapient_options_default(void);

/* ── Sessions ──────────────────────────────────────────────────────────────────────── */

/* Load a model (downloading it on first use) and hold it resident.
 * `opts` may be NULL to accept the defaults. Returns NULL on failure.
 * BLOCKING, potentially for minutes — call from a background thread. */
sapient_session_t *sapient_session_load(const char              *model,
                                        const sapient_options_t *opts,
                                        sapient_error_t        **err);

/* Release a session and unload its model. NULL-safe. */
void sapient_session_free(sapient_session_t *s);

/* One blocking chat turn. On SAPIENT_OK, *out is a caller-owned reply string.
 * The turn is appended to the session transcript. */
int sapient_chat(sapient_session_t *s,
                 const char        *prompt,
                 char             **out,
                 sapient_error_t  **err);

/* Streaming callback. `token` is valid ONLY for the duration of the call — copy it if
 * you need to keep it. Return true to continue generating, false to cancel. */
typedef bool (*sapient_token_cb)(const char *token, void *user_data);

/* Streaming chat turn. `on_token` is invoked on the calling thread for each fragment.
 * On SAPIENT_OK, *out is the full reply (caller-owned) — which, if you cancelled, is the
 * partial text generated so far, and is what gets committed to the transcript. */
int sapient_chat_stream(sapient_session_t *s,
                        const char        *prompt,
                        sapient_token_cb   on_token,
                        void              *user_data,
                        char             **out,
                        sapient_error_t  **err);

/* Clear the conversation, keeping the model loaded. Any system prompt is re-seeded. */
int sapient_session_reset(sapient_session_t *s, sapient_error_t **err);

/* ── Transcript ────────────────────────────────────────────────────────────────────── */

/* Number of messages in the transcript (0 if s is NULL). */
size_t sapient_transcript_len(sapient_session_t *s);

/* Entry `index` as "role\tcontent", or NULL if out of range. Caller frees.
 * Tab-separated rather than a struct, to keep layout out of the ABI. */
char  *sapient_transcript_at(sapient_session_t *s, size_t index);

/* ── Session info ──────────────────────────────────────────────────────────────────── */

/* Model alias this session was loaded with. Caller frees; NULL on failure. */
char *sapient_session_model(sapient_session_t *s);

/* Resolved backend label, e.g. "CPU" or "Metal GPU". Caller frees; NULL on failure. */
char *sapient_session_backend(sapient_session_t *s);

/* Whether weights are memory-mapped (RSS tracks the working set, not file size). */
bool  sapient_session_is_mmap(sapient_session_t *s);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SAPIENT_H */
