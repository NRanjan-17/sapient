// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! ABI-stability gate for the C API.
//!
//! Three things must agree, or downstream bindings break silently:
//!   1. the `#[no_mangle] extern "C"` functions in `src/lib.rs`,
//!   2. the declarations in `include/sapient.h`,
//!   3. the committed snapshot in [`EXPECTED_SYMBOLS`] below.
//!
//! Adding a function means adding it to all three — deliberately. Renaming or removing
//! one is a **breaking ABI change**: bump `SAPIENT_API_VERSION` in both `src/lib.rs` and
//! `include/sapient.h`, and say so in the changelog.
//!
//! Why a gate and not a convention: `reports/TEST-RUN-M2-2026-09-22.md` found two GPU
//! test suites that had been passing *vacuously* in CI for want of one `--features` flag.
//! An unguarded ABI rots the same quiet way — except the breakage lands in other people's
//! programs.

use std::collections::BTreeSet;

/// The public C ABI. Sorted. Changing this list is a deliberate act — see the module docs.
const EXPECTED_SYMBOLS: &[&str] = &[
    "sapient_api_version",
    "sapient_chat",
    "sapient_chat_stream",
    "sapient_error_code",
    "sapient_error_free",
    "sapient_error_message",
    "sapient_model_alias",
    "sapient_model_count",
    "sapient_options_default",
    "sapient_resolve_alias",
    "sapient_session_backend",
    "sapient_session_free",
    "sapient_session_is_mmap",
    "sapient_session_load",
    "sapient_session_model",
    "sapient_session_reset",
    "sapient_set_cache_dir",
    "sapient_string_free",
    "sapient_transcript_at",
    "sapient_transcript_len",
    "sapient_version",
];

const HEADER: &str = include_str!("../include/sapient.h");
const SOURCE: &str = include_str!("../src/lib.rs");

/// Exported symbols per the Rust source: every `pub [unsafe] extern "C" fn` that is
/// `#[no_mangle]`.
fn symbols_in_source() -> BTreeSet<String> {
    let mut found = BTreeSet::new();
    let lines: Vec<&str> = SOURCE.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        if line.trim() != "#[no_mangle]" {
            continue;
        }
        // The signature is the next non-attribute line.
        let sig = lines[i + 1..]
            .iter()
            .find(|l| !l.trim_start().starts_with('#'))
            .expect("#[no_mangle] with no following signature");
        assert!(
            sig.contains("extern \"C\""),
            "#[no_mangle] on a non-extern-\"C\" item: {sig}"
        );
        let name = sig
            .split("fn ")
            .nth(1)
            .and_then(|rest| rest.split(['(', '<']).next())
            .expect("could not parse fn name")
            .trim();
        found.insert(name.to_string());
    }
    found
}

/// Function names declared in the header. Matches `name(` preceded by a return type,
/// which is enough for this hand-written header (no macros, no function-like defines).
fn symbols_in_header() -> BTreeSet<String> {
    let mut found = BTreeSet::new();
    for line in HEADER.lines() {
        let t = line.trim();
        // Skip comments, preprocessor lines, and the callback typedef.
        if t.starts_with('*')
            || t.starts_with("/*")
            || t.starts_with('#')
            || t.starts_with("typedef")
        {
            continue;
        }
        let Some(open) = t.find('(') else { continue };
        let before = &t[..open];
        let Some(name) = before.split([' ', '*', '\t']).next_back() else {
            continue;
        };
        if name.starts_with("sapient_") {
            found.insert(name.to_string());
        }
    }
    found
}

#[test]
fn header_matches_the_rust_exports() {
    let src = symbols_in_source();
    let hdr = symbols_in_header();
    assert_eq!(
        src, hdr,
        "\ninclude/sapient.h and src/lib.rs disagree.\n  only in src/lib.rs: {:?}\n  only in sapient.h: {:?}\n",
        src.difference(&hdr).collect::<Vec<_>>(),
        hdr.difference(&src).collect::<Vec<_>>(),
    );
}

#[test]
fn abi_surface_matches_the_committed_snapshot() {
    let actual = symbols_in_source();
    let expected: BTreeSet<String> = EXPECTED_SYMBOLS.iter().map(|s| s.to_string()).collect();
    assert_eq!(
        actual,
        expected,
        "\nThe C ABI changed.\n  added: {:?}\n  removed: {:?}\n\n\
         If intentional: update EXPECTED_SYMBOLS, and for a rename/removal also bump \
         SAPIENT_API_VERSION in src/lib.rs AND include/sapient.h.\n",
        actual.difference(&expected).collect::<Vec<_>>(),
        expected.difference(&actual).collect::<Vec<_>>(),
    );
}

#[test]
fn api_version_agrees_between_header_and_source() {
    let hdr = HEADER
        .lines()
        .find(|l| l.contains("#define SAPIENT_API_VERSION"))
        .and_then(|l| l.split_whitespace().nth(2))
        .map(|v| v.trim_end_matches('u').to_string())
        .expect("SAPIENT_API_VERSION not found in sapient.h");
    assert_eq!(
        hdr,
        sapient::SAPIENT_API_VERSION.to_string(),
        "SAPIENT_API_VERSION differs between include/sapient.h and src/lib.rs"
    );
}

#[test]
fn status_and_backend_constants_agree_between_header_and_source() {
    let defines: Vec<(&str, i32)> = vec![
        ("SAPIENT_OK", sapient::SAPIENT_OK),
        (
            "SAPIENT_ERR_INVALID_ARGUMENT",
            sapient::SAPIENT_ERR_INVALID_ARGUMENT,
        ),
        ("SAPIENT_ERR_LOAD", sapient::SAPIENT_ERR_LOAD),
        ("SAPIENT_ERR_GENERATION", sapient::SAPIENT_ERR_GENERATION),
        ("SAPIENT_ERR_INTERNAL", sapient::SAPIENT_ERR_INTERNAL),
        ("SAPIENT_ERR_PANIC", sapient::SAPIENT_ERR_PANIC),
        ("SAPIENT_BACKEND_AUTO", sapient::SAPIENT_BACKEND_AUTO),
        ("SAPIENT_BACKEND_CPU", sapient::SAPIENT_BACKEND_CPU),
        ("SAPIENT_BACKEND_METAL", sapient::SAPIENT_BACKEND_METAL),
        ("SAPIENT_BACKEND_WGPU", sapient::SAPIENT_BACKEND_WGPU),
    ];
    for (name, rust_value) in defines {
        let needle = format!("#define {name} ");
        let line = HEADER
            .lines()
            .find(|l| l.trim_start().starts_with(&needle))
            .unwrap_or_else(|| panic!("{name} not #defined in sapient.h"));
        let hdr_value: i32 = line
            .trim_start()
            .trim_start_matches(&needle)
            .split_whitespace()
            .next()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| panic!("could not parse the value of {name} in sapient.h"));
        assert_eq!(
            hdr_value, rust_value,
            "{name} is {hdr_value} in sapient.h but {rust_value} in src/lib.rs"
        );
    }
}

#[test]
fn every_exported_function_is_documented_in_the_header() {
    // A bare declaration with no comment above it is how an API becomes unusable.
    for sym in EXPECTED_SYMBOLS {
        let idx = HEADER
            .find(&format!("{sym}("))
            .unwrap_or_else(|| panic!("{sym} missing from sapient.h"));
        let preceding = &HEADER[..idx];
        let has_comment = preceding
            .lines()
            .rev()
            .take(8)
            .any(|l| l.trim_start().starts_with("/*") || l.trim_start().starts_with('*'));
        assert!(
            has_comment,
            "{sym} is declared in sapient.h with no comment"
        );
    }
}
