// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! **The blocking gate.** No timing number from this study is recorded, or
//! believed, until every test here is green.
//!
//! The standard is the repo's own: exact `f32::to_bits()` equality against the
//! existing scalar oracles in `sapient_backends_cpu::kernels::quant` — not
//! approximate agreement. A C++ kernel that is merely *close* is a different
//! kernel, and benchmarking it would compare two different computations.

use sapient_backends_cpu::kernels::quant as rk;
use sapient_cpp_kernels as cpp;
use sapient_lang_harness::{has_dotprod, q4_k, q6_k, SHAPES};

/// fp16 → f32, **exhaustively** over all 65 536 bit patterns.
///
/// Rust reads scales with `half::f16::from_le_bytes(..).to_f32()`; the C++ uses
/// an AArch64 `__fp16` hardware convert. Both are IEEE-754 binary16→binary32,
/// which is exact for every input — but "should be" is not evidence, and a
/// mismatch here would silently offset every super-block scale. 65 536 cases is
/// cheap enough to just prove it.
#[test]
fn f16_conversion_matches_half_crate_exhaustively() {
    let mut checked = 0u32;
    for bits in 0u32..=0xFFFF {
        let bits = bits as u16;
        let want = half::f16::from_bits(bits).to_f32();
        let got = cpp::f16_to_f32(bits);
        if want.is_nan() {
            assert!(got.is_nan(), "0x{bits:04x}: half=NaN cpp={got}");
        } else {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "0x{bits:04x}: cpp={got} half={want}"
            );
        }
        checked += 1;
    }
    assert_eq!(checked, 65_536);
}

#[test]
fn cpp_q4_k_4rows_r4_q8k_is_bit_identical_to_rust() {
    if !has_dotprod() {
        eprintln!("SKIP: CPU lacks FEAT_DotProd");
        return;
    }
    for &(name, k) in SHAPES {
        let f = q4_k(0xC0FF_EE01, k);

        let got = unsafe { cpp::dot_q4_k_4rows_r4_q8k(&f.packed, &f.x_i8, &f.x_scales, &f.x_sums) };
        let rust =
            unsafe { rk::dot_q4_k_4rows_r4_q8k_neon(&f.packed, &f.x_i8, &f.x_scales, &f.x_sums) };

        for r in 0..4 {
            // vs the Rust NEON kernel it replaces
            assert_eq!(
                got[r].to_bits(),
                rust[r].to_bits(),
                "{name} (k={k}) row {r}: cpp={} rust_neon={}",
                got[r],
                rust[r]
            );
            // and independently vs the scalar oracle, so a shared bug in both
            // SIMD paths cannot hide
            let oracle = rk::dot_q4_k_row_q8k_scalar(f.row(r), &f.x_i8, &f.x_scales, &f.x_sums);
            assert_eq!(
                got[r].to_bits(),
                oracle.to_bits(),
                "{name} (k={k}) row {r}: cpp={} scalar_oracle={}",
                got[r],
                oracle
            );
        }
    }
}

#[test]
fn cpp_q6_k_4rows_r4_q8k_is_bit_identical_to_rust() {
    if !has_dotprod() {
        eprintln!("SKIP: CPU lacks FEAT_DotProd");
        return;
    }
    for &(name, k) in SHAPES {
        let f = q6_k(0xC0FF_EE02, k);

        let got = unsafe { cpp::dot_q6_k_4rows_r4_q8k(&f.packed, &f.x_i8, &f.x_scales) };
        let rust = unsafe { rk::dot_q6_k_4rows_r4_q8k_neon(&f.packed, &f.x_i8, &f.x_scales) };

        for r in 0..4 {
            assert_eq!(
                got[r].to_bits(),
                rust[r].to_bits(),
                "{name} (k={k}) row {r}: cpp={} rust_neon={}",
                got[r],
                rust[r]
            );
            let oracle = rk::dot_q6_k_row_q8k_scalar(f.row(r), &f.x_i8, &f.x_scales);
            assert_eq!(
                got[r].to_bits(),
                oracle.to_bits(),
                "{name} (k={k}) row {r}: cpp={} scalar_oracle={}",
                got[r],
                oracle
            );
        }
    }
}

/// The ported scalar oracles must match the Rust scalar oracles too — this is
/// what the pure-C++ sanitizer driver runs, so it needs its own gate.
#[test]
fn cpp_scalar_oracles_are_bit_identical_to_rust() {
    for &(name, k) in SHAPES {
        let f4 = q4_k(0xC0FF_EE03, k);
        for r in 0..4 {
            let got = cpp::dot_q4_k_row_q8k_scalar(f4.row(r), &f4.x_i8, &f4.x_scales, &f4.x_sums);
            let want = rk::dot_q4_k_row_q8k_scalar(f4.row(r), &f4.x_i8, &f4.x_scales, &f4.x_sums);
            assert_eq!(got.to_bits(), want.to_bits(), "q4_k scalar {name} row {r}");
        }
        let f6 = q6_k(0xC0FF_EE04, k);
        for r in 0..4 {
            let got = cpp::dot_q6_k_row_q8k_scalar(f6.row(r), &f6.x_i8, &f6.x_scales);
            let want = rk::dot_q6_k_row_q8k_scalar(f6.row(r), &f6.x_i8, &f6.x_scales);
            assert_eq!(got.to_bits(), want.to_bits(), "q6_k scalar {name} row {r}");
        }
    }
}

/// Multiple independent seeds, so the gate is not one lucky byte pattern.
#[test]
fn bit_identity_holds_across_seeds() {
    if !has_dotprod() {
        eprintln!("SKIP: CPU lacks FEAT_DotProd");
        return;
    }
    let k = 2048;
    for seed in [1u64, 7, 42, 1337, 0xDEAD_BEEF, 0x5EED, u64::MAX / 3] {
        let f4 = q4_k(seed, k);
        let a = unsafe { cpp::dot_q4_k_4rows_r4_q8k(&f4.packed, &f4.x_i8, &f4.x_scales, &f4.x_sums) };
        let b =
            unsafe { rk::dot_q4_k_4rows_r4_q8k_neon(&f4.packed, &f4.x_i8, &f4.x_scales, &f4.x_sums) };
        for r in 0..4 {
            assert_eq!(a[r].to_bits(), b[r].to_bits(), "q4_k seed {seed} row {r}");
        }

        let f6 = q6_k(seed, k);
        let a = unsafe { cpp::dot_q6_k_4rows_r4_q8k(&f6.packed, &f6.x_i8, &f6.x_scales) };
        let b = unsafe { rk::dot_q6_k_4rows_r4_q8k_neon(&f6.packed, &f6.x_i8, &f6.x_scales) };
        for r in 0..4 {
            assert_eq!(a[r].to_bits(), b[r].to_bits(), "q6_k seed {seed} row {r}");
        }
    }
}

/// Records the exact C++ build line into the test log, so the report can quote
/// it rather than a remembered flag string.
#[test]
fn report_cpp_build_configuration() {
    println!("C++ compiler: {}", cpp::CPP_COMPILER);
    println!("C++ flags:    {}", cpp::CPP_FLAGS);
    assert!(!cpp::CPP_COMPILER.is_empty());
}

/// The manually-unrolled Rust arm must also be bit-identical, or it is a
/// different kernel and its timing means nothing.
#[test]
#[cfg(target_arch = "aarch64")]
fn rust_unrolled_q4_k_is_bit_identical_to_shipped() {
    if !has_dotprod() {
        eprintln!("SKIP: CPU lacks FEAT_DotProd");
        return;
    }
    use sapient_lang_harness::rust_unrolled::dot_q4_k_4rows_r4_q8k_unrolled as unrolled;
    use sapient_lang_harness::rust_unrolled::dot_q4_k_4rows_r4_q8k_unrolled_unchecked as unchecked;
    use sapient_lang_harness::rust_unrolled::dot_q4_k_4rows_r4_q8k_unrolled_vectail as vectail;
    for &(name, k) in SHAPES {
        for seed in [0xC0FF_EE01u64, 7, 0xDEAD_BEEF] {
            let f = q4_k(seed, k);
            let got = unsafe { unrolled(&f.packed, &f.x_i8, &f.x_scales, &f.x_sums) };
            let want = unsafe {
                rk::dot_q4_k_4rows_r4_q8k_neon(&f.packed, &f.x_i8, &f.x_scales, &f.x_sums)
            };
            let got_u = unsafe { unchecked(&f.packed, &f.x_i8, &f.x_scales, &f.x_sums) };
            let got_v = unsafe { vectail(&f.packed, &f.x_i8, &f.x_scales, &f.x_sums) };
            for r in 0..4 {
                assert_eq!(
                    got[r].to_bits(),
                    want[r].to_bits(),
                    "{name} (k={k}, seed={seed}) row {r}: unrolled={} shipped={}",
                    got[r],
                    want[r]
                );
                assert_eq!(
                    got_u[r].to_bits(),
                    want[r].to_bits(),
                    "{name} (k={k}, seed={seed}) row {r}: unrolled_unchecked={} shipped={}",
                    got_u[r],
                    want[r]
                );
                assert_eq!(
                    got_v[r].to_bits(),
                    want[r].to_bits(),
                    "{name} (k={k}, seed={seed}) row {r}: unrolled_vectail={} shipped={}",
                    got_v[r],
                    want[r]
                );
            }
        }
    }
}
