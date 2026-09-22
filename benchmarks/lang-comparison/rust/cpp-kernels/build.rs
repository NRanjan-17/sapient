// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
//
// Compiles the C++ kernel ports. Deliberately configurable so the study can
// vary the one axis it cares about — the compiler — without editing source:
//
//   CXX                       pick the compiler (Apple clang 21 / brew clang 23 / g++)
//   SAPIENT_CPP_MARCH=1       add an explicit -march (default OFF, to mirror Rust,
//                             which sets no target-cpu — see .cargo/config.toml)
//   SAPIENT_CPP_NO_LTO=1      disable -flto
//   SAPIENT_CPP_EXTRA=...     append arbitrary flags
//
// Every flag actually used is echoed into SAPIENT_CPP_FLAGS so the report can
// state the exact build line rather than a remembered one.

use std::env;

fn main() {
    let src = ["src/q4k.cpp", "src/q6k.cpp"];
    for f in &src {
        println!("cargo:rerun-if-changed=../../cpp/{f}");
    }
    println!("cargo:rerun-if-changed=../../cpp/include/sapient_kernels.h");
    for v in [
        "CXX",
        "SAPIENT_CPP_MARCH",
        "SAPIENT_CPP_NO_LTO",
        "SAPIENT_CPP_EXTRA",
        "SAPIENT_CPP_CONTRACT",
    ] {
        println!("cargo:rerun-if-env-changed={v}");
    }

    let mut b = cc::Build::new();
    b.cpp(true)
        .std("c++17")
        .opt_level(3)
        .include("../../cpp/include")
        .warnings(true);

    let mut flags: Vec<String> = vec!["-O3".into(), "-std=c++17".into()];

    // ── FP contraction: REQUIRED for the comparison to be meaningful ─────────
    // clang defaults to `-ffp-contract=on`, fusing `a*b + c` into one FMA and
    // therefore rounding ONCE where Rust rounds twice. rustc never contracts
    // unless you write `f32::mul_add` explicitly. Left at clang's default the
    // C++ computes genuinely different values (measured: 1-2 ULP), which fails
    // the repo's bit-identity standard and would make any timing a comparison
    // of two different computations rather than of two languages.
    //
    // This divergence is itself a reportable finding — set SAPIENT_CPP_CONTRACT=1
    // to measure what contraction is worth, but then the bit-identity gate
    // legitimately fails and the numbers are labelled accordingly.
    if env::var("SAPIENT_CPP_CONTRACT").is_ok() {
        flags.push("-ffp-contract=fast".into());
        b.flag("-ffp-contract=fast");
    } else {
        flags.push("-ffp-contract=off".into());
        b.flag("-ffp-contract=off");
    }

    if env::var("SAPIENT_CPP_NO_LTO").is_err() {
        b.flag("-flto");
        flags.push("-flto".into());
    }
    // OFF by default: Rust sets no global target-cpu, so neither do we. The
    // NEON/dotprod capability comes from __attribute__((target(...))) on the
    // function, which is the exact analogue of Rust's #[target_feature].
    if env::var("SAPIENT_CPP_MARCH").is_ok() {
        let m = "-march=armv8.2-a+dotprod+i8mm";
        b.flag(m);
        flags.push(m.into());
    }
    if let Ok(extra) = env::var("SAPIENT_CPP_EXTRA") {
        for f in extra.split_whitespace() {
            b.flag(f);
            flags.push(f.into());
        }
    }

    for f in &src {
        b.file(format!("../../cpp/{f}"));
    }
    b.compile("sapient_cpp_kernels");

    let compiler = b.get_compiler();
    println!(
        "cargo:rustc-env=SAPIENT_CPP_COMPILER={}",
        compiler.path().display()
    );
    println!("cargo:rustc-env=SAPIENT_CPP_FLAGS={}", flags.join(" "));
}
