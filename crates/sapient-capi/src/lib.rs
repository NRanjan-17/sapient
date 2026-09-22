// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! Stable **C ABI** for embedding SAPIENT — the surface every other language binds to.
//!
//! This is deliberately NOT `sapient-ffi`. That crate exports UniFFI scaffolding
//! (`RustBuffer`, generated handles, a contract checksum that aborts on mismatch) as a
//! codegen substrate for Swift and Kotlin; its shape changes between UniFFI releases and
//! it is not meant to be `#include`d. This crate is a small, hand-written, versioned C
//! API that Python (cffi), Go (cgo), Node (N-API), C#, Java, Julia and Zig can all bind
//! to — see `docs/C-ECOSYSTEM.md`.
//!
//! The canonical declaration of this ABI is [`include/sapient.h`]. Keep the two in sync;
//! `tests/abi_surface.rs` fails the build if they drift.
//!
//! # ABI rules (why the code looks like this)
//!
//! - **Opaque handles only.** No Rust type's layout crosses the boundary. Callers hold
//!   `sapient_session_t*` / `sapient_error_t*` pointers and nothing else.
//! - **No panic may cross into C.** A Rust panic unwinding into a C frame is undefined
//!   behaviour, so every `extern "C"` entry point is wrapped in [`std::panic::catch_unwind`]
//!   and converts a panic into [`SAPIENT_ERR_PANIC`].
//! - **Everything handed out has a matching free.** C has no drop glue:
//!   [`sapient_string_free`], [`sapient_error_free`], [`sapient_session_free`].
//! - **Errors are out-parameters**, never a global or `errno` — that keeps them thread-safe.
//!   Pass `NULL` for `err` to discard the detail and keep only the status code.
//! - **Strings are UTF-8** `char*` in, caller-freed `char*` out.
//!
//! # Threading
//!
//! A `sapient_session_t*` is safe to use from multiple threads. Generation calls on the
//! *same* session **serialize** on the engine's internal lock — they do not run in
//! parallel. Two different sessions run concurrently. See `docs/C-ECOSYSTEM.md` §4.7.
//!
//! # Async
//!
//! Every function here is **blocking**. A private multi-threaded tokio runtime drives the
//! async `Pipeline` internals; futures are spawned onto its workers rather than
//! `block_on`-ed on the caller, because `Pipeline` uses `tokio::task::block_in_place`,
//! which panics on a thread that is not a runtime worker. Call from a background thread.
//!
//! [`include/sapient.h`]: ../../include/sapient.h

#![allow(non_camel_case_types)]

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex, OnceLock};

use sapient_generate::{GenerationConfig, LoadOptions, Pipeline, SamplingStrategy};
use sapient_tokenizers::chat::ChatMessage;

// ── ABI version ───────────────────────────────────────────────────────────────

/// Version of the *C ABI* (not the engine). Bump on any breaking change to the
/// declarations in `sapient.h`. Compare against `SAPIENT_API_VERSION` from the header
/// that a caller was compiled against to detect a header/library mismatch.
pub const SAPIENT_API_VERSION: u32 = 1;

// ── Status codes ──────────────────────────────────────────────────────────────

/// Success.
pub const SAPIENT_OK: c_int = 0;
/// A NULL pointer, non-UTF-8 string, or out-of-range value was passed in.
pub const SAPIENT_ERR_INVALID_ARGUMENT: c_int = -1;
/// Model download, resolution, or weight loading failed.
pub const SAPIENT_ERR_LOAD: c_int = -2;
/// Generation failed after the model was loaded.
pub const SAPIENT_ERR_GENERATION: c_int = -3;
/// An unexpected internal failure (runtime join, poisoned lock, …).
pub const SAPIENT_ERR_INTERNAL: c_int = -4;
/// A Rust panic was caught at the boundary. Always a bug — please report it.
pub const SAPIENT_ERR_PANIC: c_int = -5;

// ── Backend selection ─────────────────────────────────────────────────────────

/// Resolve to whatever accelerator this build supports, falling back to CPU.
pub const SAPIENT_BACKEND_AUTO: c_int = 0;
/// Force the CPU engine.
pub const SAPIENT_BACKEND_CPU: c_int = 1;
/// Force Apple MLX/Metal (requires a `mlx`-featured build on Apple Silicon).
pub const SAPIENT_BACKEND_METAL: c_int = 2;
/// Force the portable GPU path (requires a `wgpu`-featured build).
pub const SAPIENT_BACKEND_WGPU: c_int = 3;

// ── Errors ────────────────────────────────────────────────────────────────────

/// Opaque error detail. Obtained via an `err` out-parameter; freed with
/// [`sapient_error_free`].
pub struct sapient_error_t {
    code: c_int,
    message: CString,
}

/// Internal error carrying a status code and a human-readable reason.
#[derive(Debug)]
struct CapiError {
    code: c_int,
    message: String,
}

impl CapiError {
    fn new(code: c_int, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
    fn invalid(message: impl Into<String>) -> Self {
        Self::new(SAPIENT_ERR_INVALID_ARGUMENT, message)
    }
}

/// Write an error into the caller's out-parameter, if they supplied one.
///
/// `NULL` is an explicit "I only want the status code" — never an error itself.
fn set_error(out: *mut *mut sapient_error_t, e: CapiError) -> c_int {
    let code = e.code;
    if !out.is_null() {
        // Interior NUL bytes can't be represented in C; substitute rather than lose
        // the whole message.
        let msg = CString::new(e.message.replace('\0', "\\0"))
            .unwrap_or_else(|_| CString::new("error message contained invalid bytes").unwrap());
        let boxed = Box::new(sapient_error_t { code, message: msg });
        unsafe { *out = Box::into_raw(boxed) };
    }
    code
}

/// Turn a caught panic payload into a `CapiError`.
fn panic_error(payload: Box<dyn std::any::Any + Send>) -> CapiError {
    let detail = payload
        .downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".to_string());
    CapiError::new(
        SAPIENT_ERR_PANIC,
        format!("panic caught at FFI boundary: {detail}"),
    )
}

/// Guard for functions returning a **status code**.
///
/// On failure the returned code is the error's own `code` — not a fixed sentinel — so
/// callers can distinguish `SAPIENT_ERR_LOAD` from `SAPIENT_ERR_GENERATION` from
/// `SAPIENT_ERR_PANIC` without inspecting the `err` out-parameter. `set_error` returns
/// that code, which is why both arms simply forward it.
fn guard_status(
    err: *mut *mut sapient_error_t,
    f: impl FnOnce() -> Result<c_int, CapiError>,
) -> c_int {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => set_error(err, e),
        Err(payload) => set_error(err, panic_error(payload)),
    }
}

/// Guard for functions returning a **value** (pointer, bool, length), where the ABI has
/// no room for a code. `on_err` is the documented failure sentinel — NULL, `false`, `0` —
/// and the detail goes to `err`.
fn guard<T>(
    err: *mut *mut sapient_error_t,
    on_err: T,
    f: impl FnOnce() -> Result<T, CapiError>,
) -> T {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            set_error(err, e);
            on_err
        }
        Err(payload) => {
            set_error(err, panic_error(payload));
            on_err
        }
    }
}

/// The status code carried by an error. Returns [`SAPIENT_ERR_INVALID_ARGUMENT`] for NULL.
///
/// # Safety
/// `err` must be NULL or a pointer returned through an out-parameter of this library and
/// not yet freed.
#[no_mangle]
pub unsafe extern "C" fn sapient_error_code(err: *const sapient_error_t) -> c_int {
    if err.is_null() {
        return SAPIENT_ERR_INVALID_ARGUMENT;
    }
    (*err).code
}

/// Human-readable error message, valid until [`sapient_error_free`]. **Do not free it**
/// separately. Returns NULL for a NULL error.
///
/// # Safety
/// Same contract as [`sapient_error_code`].
#[no_mangle]
pub unsafe extern "C" fn sapient_error_message(err: *const sapient_error_t) -> *const c_char {
    if err.is_null() {
        return std::ptr::null();
    }
    (*err).message.as_ptr()
}

/// Release an error. Safe to call with NULL. Double-free is undefined behaviour.
///
/// # Safety
/// `err` must be NULL or a pointer from this library that has not already been freed.
#[no_mangle]
pub unsafe extern "C" fn sapient_error_free(err: *mut sapient_error_t) {
    if !err.is_null() {
        drop(Box::from_raw(err));
    }
}

// ── Strings ───────────────────────────────────────────────────────────────────

/// Release a string returned by this library. Safe to call with NULL.
///
/// # Safety
/// `s` must be NULL or a string returned through an out-parameter of this library and not
/// yet freed. Never pass a pointer from [`sapient_version`] or [`sapient_error_message`].
#[no_mangle]
pub unsafe extern "C" fn sapient_string_free(s: *mut c_char) {
    if !s.is_null() {
        drop(CString::from_raw(s));
    }
}

/// Borrow a C string as `&str`, rejecting NULL and non-UTF-8.
unsafe fn cstr(ptr: *const c_char, what: &str) -> Result<&'static str, CapiError> {
    if ptr.is_null() {
        return Err(CapiError::invalid(format!("{what} must not be NULL")));
    }
    CStr::from_ptr(ptr)
        .to_str()
        .map_err(|_| CapiError::invalid(format!("{what} must be valid UTF-8")))
}

/// Move a Rust `String` into a caller-owned C string.
fn into_c_string(s: String) -> Result<*mut c_char, CapiError> {
    CString::new(s.replace('\0', "\\0"))
        .map(CString::into_raw)
        .map_err(|_| CapiError::new(SAPIENT_ERR_INTERNAL, "could not encode string for C"))
}

// ── Runtime ───────────────────────────────────────────────────────────────────

/// Private tokio runtime driving the async `Pipeline` internals. Two workers suffice —
/// inference runs on tokio's blocking pool, not on these.
fn runtime() -> &'static tokio::runtime::Runtime {
    static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .thread_name("sapient-capi")
            .enable_all()
            .build()
            .expect("failed to build sapient-capi tokio runtime")
    })
}

/// Drive a future on the runtime's workers and block the calling (foreign) thread.
///
/// The future MUST run on a worker rather than via `block_on` on the caller: `Pipeline`
/// uses `block_in_place` internally, which panics outside a multi-thread-runtime worker.
fn run_async<T, F>(fut: F) -> anyhow::Result<T>
where
    F: std::future::Future<Output = anyhow::Result<T>> + Send + 'static,
    T: Send + 'static,
{
    let handle = runtime().spawn(fut);
    runtime()
        .block_on(handle)
        .map_err(|e| anyhow::anyhow!("sapient-capi runtime join error: {e}"))?
}

// ── Library info ──────────────────────────────────────────────────────────────

/// Engine version string (e.g. `"0.6.0"`), owned by the library — **do not free**.
#[no_mangle]
pub extern "C" fn sapient_version() -> *const c_char {
    static V: OnceLock<CString> = OnceLock::new();
    V.get_or_init(|| CString::new(env!("CARGO_PKG_VERSION")).expect("version has no NUL"))
        .as_ptr()
}

/// ABI version of this library. Compare with `SAPIENT_API_VERSION` from the header the
/// caller compiled against; a mismatch means header and library disagree.
#[no_mangle]
pub extern "C" fn sapient_api_version() -> u32 {
    SAPIENT_API_VERSION
}

/// Point the model cache at a directory (sets `HF_HOME`). Call before loading. Returns a
/// status code.
///
/// # Safety
/// `path` must be a NUL-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn sapient_set_cache_dir(
    path: *const c_char,
    err: *mut *mut sapient_error_t,
) -> c_int {
    guard_status(err, || {
        let p = cstr(path, "path")?;
        std::env::set_var("HF_HOME", p);
        Ok(SAPIENT_OK)
    })
}

// ── Model catalog ─────────────────────────────────────────────────────────────

/// Number of models in the curated catalog.
#[no_mangle]
pub extern "C" fn sapient_model_count() -> usize {
    sapient_hub::registry::catalog().len()
}

/// Alias of catalog entry `index`, or NULL if out of range. Caller frees with
/// [`sapient_string_free`].
#[no_mangle]
pub extern "C" fn sapient_model_alias(index: usize) -> *mut c_char {
    guard(
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        || match sapient_hub::registry::catalog().get(index) {
            Some(m) => into_c_string(m.alias.to_string()),
            None => Ok(std::ptr::null_mut()),
        },
    )
}

/// Resolve an alias (fuzzy-matched) to its HuggingFace repo id. On success writes a
/// caller-owned string to `out`.
///
/// # Safety
/// `name` must be a NUL-terminated UTF-8 string; `out` must be a valid writable pointer.
#[no_mangle]
pub unsafe extern "C" fn sapient_resolve_alias(
    name: *const c_char,
    out: *mut *mut c_char,
    err: *mut *mut sapient_error_t,
) -> c_int {
    guard_status(err, || {
        if out.is_null() {
            return Err(CapiError::invalid("out must not be NULL"));
        }
        let n = cstr(name, "name")?;
        let repo = sapient_hub::registry::resolve_model_alias(n)
            .map_err(|e| CapiError::invalid(e.to_string()))?;
        *out = into_c_string(repo)?;
        Ok(SAPIENT_OK)
    })
}

// ── Options ───────────────────────────────────────────────────────────────────

/// Generation options. Obtain a populated instance from [`sapient_options_default`] and
/// override fields; that keeps source compatible when fields are added.
///
/// Negative floats and negative `top_k` mean **unset**. Leaving all four sampling fields
/// unset selects greedy (deterministic) decoding.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct sapient_options_t {
    /// Hard cap on new tokens per reply.
    pub max_tokens: u32,
    /// Sampling temperature; `< 0` = unset.
    pub temperature: f32,
    /// Nucleus sampling threshold (0–1); `< 0` = unset.
    pub top_p: f32,
    /// Top-k cutoff; `< 0` = unset, `0` disables the filter.
    pub top_k: i32,
    /// Repetition penalty (1.0 = off); `< 0` = unset.
    pub repetition_penalty: f32,
    /// Optional system prompt, NUL-terminated UTF-8. NULL = none. Borrowed for the
    /// duration of the load call only.
    pub system_prompt: *const c_char,
    /// One of the `SAPIENT_BACKEND_*` constants.
    pub backend: c_int,
}

/// Default options: 512 tokens, greedy decoding, no system prompt, automatic backend.
#[no_mangle]
pub extern "C" fn sapient_options_default() -> sapient_options_t {
    sapient_options_t {
        max_tokens: 512,
        temperature: -1.0,
        top_p: -1.0,
        top_k: -1,
        repetition_penalty: -1.0,
        system_prompt: std::ptr::null(),
        backend: SAPIENT_BACKEND_AUTO,
    }
}

impl sapient_options_t {
    /// All sampling fields unset → greedy; otherwise the combined sampler with
    /// engine-neutral defaults (`top_k 0` / `top_p 1.0` disable those filters, `rp 1.0`
    /// is a no-op). Mirrors `sapient-ffi`'s mapping so both surfaces behave identically.
    fn strategy(&self) -> SamplingStrategy {
        if self.temperature < 0.0
            && self.top_p < 0.0
            && self.top_k < 0
            && self.repetition_penalty < 0.0
        {
            return SamplingStrategy::Greedy;
        }
        SamplingStrategy::Combined {
            top_k: if self.top_k < 0 {
                0
            } else {
                self.top_k as usize
            },
            top_p: if self.top_p < 0.0 { 1.0 } else { self.top_p },
            temperature: if self.temperature < 0.0 {
                0.7
            } else {
                self.temperature
            },
            repetition_penalty: if self.repetition_penalty < 0.0 {
                1.0
            } else {
                self.repetition_penalty
            },
        }
    }

    fn generation_config(&self) -> GenerationConfig {
        GenerationConfig {
            max_new_tokens: self.max_tokens as usize,
            strategy: self.strategy(),
            ..GenerationConfig::default()
        }
    }

    fn backend_kind(&self) -> Result<sapient_generate::GenerationBackend, CapiError> {
        use sapient_generate::GenerationBackend as B;
        match self.backend {
            SAPIENT_BACKEND_AUTO => Ok(B::Auto),
            SAPIENT_BACKEND_CPU => Ok(B::Cpu),
            SAPIENT_BACKEND_METAL => Ok(B::Metal),
            SAPIENT_BACKEND_WGPU => Ok(B::Wgpu),
            other => Err(CapiError::invalid(format!(
                "unknown backend {other} (expected 0=auto, 1=cpu, 2=metal, 3=wgpu)"
            ))),
        }
    }
}

// ── Session ───────────────────────────────────────────────────────────────────

/// An opaque loaded model plus its conversation state.
pub struct sapient_session_t {
    pipeline: Arc<Pipeline>,
    history: Mutex<Vec<ChatMessage>>,
    system_prompt: Option<String>,
    config: GenerationConfig,
    model: String,
}

impl sapient_session_t {
    fn seeded_history(system_prompt: &Option<String>) -> Vec<ChatMessage> {
        match system_prompt {
            Some(s) if !s.is_empty() => vec![ChatMessage::system(s.clone())],
            _ => Vec::new(),
        }
    }

    /// History snapshot plus the new user turn, without mutating state — history is only
    /// committed once a turn succeeds.
    fn messages_with(&self, user_message: &str) -> Vec<ChatMessage> {
        let mut msgs = self
            .history
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        msgs.push(ChatMessage::user(user_message));
        msgs
    }

    fn commit_turn(&self, user_message: &str, reply: &str) {
        let mut h = self.history.lock().unwrap_or_else(|e| e.into_inner());
        h.push(ChatMessage::user(user_message));
        h.push(ChatMessage::assistant(reply));
    }
}

/// Borrow a session handle, rejecting NULL.
unsafe fn session<'a>(s: *mut sapient_session_t) -> Result<&'a sapient_session_t, CapiError> {
    if s.is_null() {
        return Err(CapiError::invalid("session must not be NULL"));
    }
    Ok(&*s)
}

/// Download (if needed) and load a model, returning an owned session handle.
///
/// **Blocking**, and the first call for a given model downloads weights — call from a
/// background thread. Returns NULL on failure, with detail in `err`.
///
/// # Safety
/// `model` must be a NUL-terminated UTF-8 string. `opts` may be NULL (defaults are used);
/// if non-NULL it must point at a valid `sapient_options_t`.
#[no_mangle]
pub unsafe extern "C" fn sapient_session_load(
    model: *const c_char,
    opts: *const sapient_options_t,
    err: *mut *mut sapient_error_t,
) -> *mut sapient_session_t {
    guard(err, std::ptr::null_mut(), || {
        let name = cstr(model, "model")?;
        let options = if opts.is_null() {
            sapient_options_default()
        } else {
            *opts
        };
        let system_prompt = if options.system_prompt.is_null() {
            None
        } else {
            Some(cstr(options.system_prompt, "system_prompt")?.to_string())
        };

        let config = options.generation_config();
        let load_opts = LoadOptions {
            generation: config.clone(),
            backend: options.backend_kind()?,
            ..LoadOptions::default()
        };
        let alias = name.to_string();
        let mut pipeline =
            run_async(async move { Pipeline::from_pretrained_with_opts(&alias, load_opts).await })
                .map_err(|e| CapiError::new(SAPIENT_ERR_LOAD, format!("{e:#}")))?;
        // Multi-turn chats re-send the whole history; the prefix cache keeps the KV for
        // the shared prefix so only the new turn is prefilled.
        pipeline.enable_prefix_cache();

        Ok(Box::into_raw(Box::new(sapient_session_t {
            pipeline: Arc::new(pipeline),
            history: Mutex::new(sapient_session_t::seeded_history(&system_prompt)),
            system_prompt,
            config,
            model: name.to_string(),
        })))
    })
}

/// Release a session and unload its model. Safe to call with NULL. Double-free is
/// undefined behaviour.
///
/// # Safety
/// `s` must be NULL or a handle from [`sapient_session_load`] not already freed, with no
/// other thread using it.
#[no_mangle]
pub unsafe extern "C" fn sapient_session_free(s: *mut sapient_session_t) {
    if !s.is_null() {
        drop(Box::from_raw(s));
    }
}

/// One blocking chat turn. On success writes a caller-owned reply string to `out` (free
/// with [`sapient_string_free`]) and commits the turn to the session history.
///
/// # Safety
/// `s` must be a live session; `prompt` NUL-terminated UTF-8; `out` a valid writable
/// pointer.
#[no_mangle]
pub unsafe extern "C" fn sapient_chat(
    s: *mut sapient_session_t,
    prompt: *const c_char,
    out: *mut *mut c_char,
    err: *mut *mut sapient_error_t,
) -> c_int {
    guard_status(err, || {
        if out.is_null() {
            return Err(CapiError::invalid("out must not be NULL"));
        }
        let sess = session(s)?;
        let user_message = cstr(prompt, "prompt")?;

        let messages = sess.messages_with(user_message);
        let pipeline = Arc::clone(&sess.pipeline);
        let config = sess.config.clone();
        let reply = run_async(async move { pipeline.chat_with_config(&messages, &config).await })
            .map_err(|e| CapiError::new(SAPIENT_ERR_GENERATION, format!("{e:#}")))?;

        sess.commit_turn(user_message, &reply);
        *out = into_c_string(reply)?;
        Ok(SAPIENT_OK)
    })
}

/// Token callback for [`sapient_chat_stream`]. Receives each decoded fragment as a
/// NUL-terminated UTF-8 string that is **only valid for the duration of the call** —
/// copy it if you need to keep it. Return `true` to continue, `false` to cancel.
pub type sapient_token_cb =
    Option<extern "C" fn(token: *const c_char, user_data: *mut c_void) -> bool>;

/// Streaming chat turn: `on_token` is invoked on the calling thread for each fragment as
/// it decodes. Returning `false` from the callback cancels generation.
///
/// On success writes the full (possibly cancelled-partial) reply to `out`, which is also
/// what gets committed to history — on cancel that is intentional, so the transcript
/// matches what the user saw and the prefix cache stays aligned with the engine's KV.
///
/// Error semantics match `sapient-ffi`: a failure *starting* the stream returns a status
/// code, while a failure *mid-stream* arrives in-band as a final `Error: …` fragment,
/// because the token channel carries only text with no sideband.
///
/// # Safety
/// `s` must be a live session; `prompt` NUL-terminated UTF-8; `out` a valid writable
/// pointer; `on_token` a valid function pointer for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn sapient_chat_stream(
    s: *mut sapient_session_t,
    prompt: *const c_char,
    on_token: sapient_token_cb,
    user_data: *mut c_void,
    out: *mut *mut c_char,
    err: *mut *mut sapient_error_t,
) -> c_int {
    guard_status(err, || {
        if out.is_null() {
            return Err(CapiError::invalid("out must not be NULL"));
        }
        let callback = on_token.ok_or_else(|| CapiError::invalid("on_token must not be NULL"))?;
        let sess = session(s)?;
        let user_message = cstr(prompt, "prompt")?;

        let messages = sess.messages_with(user_message);
        let pipeline = Arc::clone(&sess.pipeline);
        let config = sess.config.clone();
        let stream =
            run_async(
                async move { Ok(pipeline.chat_stream_with_config(&messages, &config).await) },
            )
            .map_err(|e| CapiError::new(SAPIENT_ERR_GENERATION, format!("{e:#}")))?;

        // Consume on the caller's thread — `blocking_recv` must not run on a runtime
        // worker. Dropping `rx` early is the cancellation signal to the engine.
        let mut rx = stream.into_inner();
        let mut reply = String::new();
        while let Some(token) = rx.blocking_recv() {
            reply.push_str(&token);
            // A token containing an interior NUL would truncate in C; substitute so the
            // callback still sees the full fragment.
            let c_token = CString::new(token.replace('\0', "\\0")).map_err(|_| {
                CapiError::new(SAPIENT_ERR_INTERNAL, "could not encode token for C")
            })?;
            if !callback(c_token.as_ptr(), user_data) {
                break;
            }
        }
        drop(rx);

        sess.commit_turn(user_message, &reply);
        *out = into_c_string(reply)?;
        Ok(SAPIENT_OK)
    })
}

/// Clear the conversation, keeping the model loaded. The system prompt supplied at load
/// time is re-seeded.
///
/// # Safety
/// `s` must be a live session handle.
#[no_mangle]
pub unsafe extern "C" fn sapient_session_reset(
    s: *mut sapient_session_t,
    err: *mut *mut sapient_error_t,
) -> c_int {
    guard_status(err, || {
        let sess = session(s)?;
        *sess.history.lock().unwrap_or_else(|e| e.into_inner()) =
            sapient_session_t::seeded_history(&sess.system_prompt);
        sess.pipeline.reset_cache();
        Ok(SAPIENT_OK)
    })
}

/// Number of messages in the session transcript.
///
/// # Safety
/// `s` must be a live session handle.
#[no_mangle]
pub unsafe extern "C" fn sapient_transcript_len(s: *mut sapient_session_t) -> usize {
    guard(std::ptr::null_mut(), 0, || {
        Ok(session(s)?
            .history
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len())
    })
}

/// Transcript entry `index` as `"role\tcontent"`, or NULL if out of range. Caller frees
/// with [`sapient_string_free`]. Tab-separated keeps the ABI free of struct layout.
///
/// # Safety
/// `s` must be a live session handle.
#[no_mangle]
pub unsafe extern "C" fn sapient_transcript_at(
    s: *mut sapient_session_t,
    index: usize,
) -> *mut c_char {
    guard(std::ptr::null_mut(), std::ptr::null_mut(), || {
        let sess = session(s)?;
        let h = sess.history.lock().unwrap_or_else(|e| e.into_inner());
        match h.get(index) {
            Some(m) => into_c_string(format!("{}\t{}", m.role, m.content)),
            None => Ok(std::ptr::null_mut()),
        }
    })
}

/// The model alias this session was loaded with. Caller frees with
/// [`sapient_string_free`]; NULL on failure.
///
/// # Safety
/// `s` must be a live session handle.
#[no_mangle]
pub unsafe extern "C" fn sapient_session_model(s: *mut sapient_session_t) -> *mut c_char {
    guard(std::ptr::null_mut(), std::ptr::null_mut(), || {
        into_c_string(session(s)?.model.clone())
    })
}

/// Human-readable resolved backend (e.g. `"CPU"`, `"Metal GPU"`). Caller frees with
/// [`sapient_string_free`]; NULL on failure.
///
/// # Safety
/// `s` must be a live session handle.
#[no_mangle]
pub unsafe extern "C" fn sapient_session_backend(s: *mut sapient_session_t) -> *mut c_char {
    guard(std::ptr::null_mut(), std::ptr::null_mut(), || {
        into_c_string(session(s)?.pipeline.backend_display_label())
    })
}

/// Whether the weights are memory-mapped (RSS tracks the working set, not file size).
///
/// # Safety
/// `s` must be a live session handle.
#[no_mangle]
pub unsafe extern "C" fn sapient_session_is_mmap(s: *mut sapient_session_t) -> bool {
    guard(std::ptr::null_mut(), false, || {
        Ok(session(s)?.pipeline.is_mmap())
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_options_select_greedy() {
        let o = sapient_options_default();
        assert_eq!(o.max_tokens, 512);
        assert_eq!(o.backend, SAPIENT_BACKEND_AUTO);
        assert!(matches!(o.strategy(), SamplingStrategy::Greedy));
    }

    #[test]
    fn any_sampling_field_selects_combined_with_neutral_defaults() {
        let mut o = sapient_options_default();
        o.temperature = 0.8;
        match o.strategy() {
            SamplingStrategy::Combined {
                top_k,
                top_p,
                temperature,
                repetition_penalty,
            } => {
                assert_eq!(top_k, 0); // unset → filter disabled
                assert_eq!(top_p, 1.0); // unset → filter disabled
                assert_eq!(temperature, 0.8);
                assert_eq!(repetition_penalty, 1.0); // unset → no-op
            }
            other => panic!("expected Combined, got {other:?}"),
        }
    }

    #[test]
    fn backend_constants_map_to_engine_kinds() {
        use sapient_generate::GenerationBackend as B;
        let mut o = sapient_options_default();
        for (c, expected) in [
            (SAPIENT_BACKEND_AUTO, B::Auto),
            (SAPIENT_BACKEND_CPU, B::Cpu),
            (SAPIENT_BACKEND_METAL, B::Metal),
            (SAPIENT_BACKEND_WGPU, B::Wgpu),
        ] {
            o.backend = c;
            assert_eq!(o.backend_kind().unwrap(), expected);
        }
        o.backend = 99;
        assert!(o.backend_kind().is_err());
    }

    #[test]
    fn guard_converts_panic_into_error_code() {
        let mut err: *mut sapient_error_t = std::ptr::null_mut();
        let rc = guard_status(&mut err, || -> Result<c_int, CapiError> {
            panic!("boom");
        });
        assert_eq!(rc, SAPIENT_ERR_PANIC);
        assert!(!err.is_null());
        unsafe {
            assert_eq!(sapient_error_code(err), SAPIENT_ERR_PANIC);
            let msg = CStr::from_ptr(sapient_error_message(err)).to_str().unwrap();
            assert!(msg.contains("boom"), "message was {msg:?}");
            sapient_error_free(err);
        }
    }

    #[test]
    fn guard_status_returns_the_real_code_not_a_sentinel() {
        // Regression: the first cut returned a fixed sentinel, so a generation failure
        // was indistinguishable from a bad argument. Callers must be able to branch on
        // the return value alone, without inspecting `err`.
        for code in [
            SAPIENT_ERR_LOAD,
            SAPIENT_ERR_GENERATION,
            SAPIENT_ERR_INTERNAL,
            SAPIENT_ERR_INVALID_ARGUMENT,
        ] {
            let mut err: *mut sapient_error_t = std::ptr::null_mut();
            let rc = guard_status(&mut err, || Err(CapiError::new(code, "x")));
            assert_eq!(rc, code);
            unsafe {
                assert_eq!(sapient_error_code(err), code);
                sapient_error_free(err);
            }
        }
    }

    #[test]
    fn guard_value_returns_its_sentinel_and_still_reports_detail() {
        let mut err: *mut sapient_error_t = std::ptr::null_mut();
        let p = guard(&mut err, std::ptr::null_mut(), || {
            Err::<*mut c_char, _>(CapiError::invalid("nope"))
        });
        assert!(p.is_null());
        unsafe {
            assert_eq!(sapient_error_code(err), SAPIENT_ERR_INVALID_ARGUMENT);
            sapient_error_free(err);
        }
    }

    #[test]
    fn null_error_out_param_is_allowed() {
        // Passing NULL for `err` means "status code only" — it must not crash.
        let rc = guard_status(std::ptr::null_mut(), || {
            Err::<c_int, _>(CapiError::invalid("nope"))
        });
        assert_eq!(rc, SAPIENT_ERR_INVALID_ARGUMENT);
    }

    #[test]
    fn null_inputs_are_rejected_not_dereferenced() {
        unsafe {
            let mut err: *mut sapient_error_t = std::ptr::null_mut();
            assert!(sapient_session_load(std::ptr::null(), std::ptr::null(), &mut err).is_null());
            assert_eq!(sapient_error_code(err), SAPIENT_ERR_INVALID_ARGUMENT);
            sapient_error_free(err);

            assert_eq!(sapient_transcript_len(std::ptr::null_mut()), 0);
            assert!(sapient_session_model(std::ptr::null_mut()).is_null());
            assert!(!sapient_session_is_mmap(std::ptr::null_mut()));
            // Free-with-NULL is explicitly safe.
            sapient_session_free(std::ptr::null_mut());
            sapient_string_free(std::ptr::null_mut());
            sapient_error_free(std::ptr::null_mut());
        }
    }

    #[test]
    fn strings_round_trip_and_free() {
        let p = into_c_string("héllo".to_string()).unwrap();
        unsafe {
            assert_eq!(CStr::from_ptr(p).to_str().unwrap(), "héllo");
            sapient_string_free(p);
        }
    }

    #[test]
    fn interior_nul_is_escaped_not_truncated() {
        let p = into_c_string("a\0b".to_string()).unwrap();
        unsafe {
            assert_eq!(CStr::from_ptr(p).to_str().unwrap(), "a\\0b");
            sapient_string_free(p);
        }
    }

    #[test]
    fn version_and_api_version_are_exposed() {
        let v = unsafe { CStr::from_ptr(sapient_version()).to_str().unwrap() };
        assert_eq!(v, env!("CARGO_PKG_VERSION"));
        assert_eq!(sapient_api_version(), SAPIENT_API_VERSION);
    }

    #[test]
    fn catalog_is_reachable_and_indexable() {
        let n = sapient_model_count();
        assert!(n > 0, "catalog should not be empty");
        let p = sapient_model_alias(0);
        assert!(!p.is_null());
        unsafe {
            assert!(!CStr::from_ptr(p).to_str().unwrap().is_empty());
            sapient_string_free(p);
        }
        assert!(sapient_model_alias(n).is_null(), "out of range → NULL");
    }

    #[test]
    fn resolve_alias_rejects_garbage_and_resolves_real_entries() {
        unsafe {
            let known = CString::new(sapient_hub::registry::catalog()[0].alias).unwrap();
            let mut out: *mut c_char = std::ptr::null_mut();
            let mut err: *mut sapient_error_t = std::ptr::null_mut();
            assert_eq!(
                sapient_resolve_alias(known.as_ptr(), &mut out, &mut err),
                SAPIENT_OK
            );
            assert!(!out.is_null());
            sapient_string_free(out);

            let bogus = CString::new("definitely-not-a-model-xyzzy").unwrap();
            let mut out2: *mut c_char = std::ptr::null_mut();
            assert_eq!(
                sapient_resolve_alias(bogus.as_ptr(), &mut out2, &mut err),
                SAPIENT_ERR_INVALID_ARGUMENT
            );
            sapient_error_free(err);
        }
    }
}
