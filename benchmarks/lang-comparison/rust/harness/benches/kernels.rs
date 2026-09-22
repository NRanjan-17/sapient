// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! Criterion micro-benchmarks: Rust kernel vs its C++ transliteration.
//!
//! Both arms are driven from the SAME fixture in the SAME process, alternating
//! within one Criterion group, so thermal drift and CPU-placement variance hit
//! both arms equally instead of accumulating against whichever ran second.
//!
//! Correctness is asserted at setup time, not assumed: if the two arms ever
//! disagree the bench panics rather than reporting a meaningless speed.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use sapient_backends_cpu::kernels::quant as rk;
use sapient_cpp_kernels as cpp;
use sapient_lang_harness::{has_dotprod, q4_k, q6_k, SHAPES};

fn bench_q4_k(c: &mut Criterion) {
    if !has_dotprod() {
        return;
    }
    let mut g = c.benchmark_group("q4_k_4rows_r4_q8k");
    for &(name, k) in SHAPES {
        let f = q4_k(0xC0FF_EE01, k);

        // Guard: never benchmark two computations that disagree.
        let a = unsafe { rk::dot_q4_k_4rows_r4_q8k_neon(&f.packed, &f.x_i8, &f.x_scales, &f.x_sums) };
        let b = unsafe { cpp::dot_q4_k_4rows_r4_q8k(&f.packed, &f.x_i8, &f.x_scales, &f.x_sums) };
        #[cfg(target_arch = "aarch64")]
        let u = unsafe {
            sapient_lang_harness::rust_unrolled::dot_q4_k_4rows_r4_q8k_unrolled(
                &f.packed, &f.x_i8, &f.x_scales, &f.x_sums,
            )
        };
        for r in 0..4 {
            assert_eq!(a[r].to_bits(), b[r].to_bits(), "{name}: rust/cpp disagree at row {r}");
            #[cfg(target_arch = "aarch64")]
            assert_eq!(a[r].to_bits(), u[r].to_bits(), "{name}: rust/unrolled disagree at row {r}");
        }

        // 4 rows x k weights of useful work per call.
        g.throughput(Throughput::Elements((4 * k) as u64));
        let id = format!("{name}/k={k}");

        g.bench_with_input(BenchmarkId::new("rust", &id), &f, |bch, f| {
            bch.iter(|| unsafe {
                rk::dot_q4_k_4rows_r4_q8k_neon(
                    black_box(&f.packed),
                    black_box(&f.x_i8),
                    black_box(&f.x_scales),
                    black_box(&f.x_sums),
                )
            })
        });
        // Third arm: Rust with the row loop manually unrolled. This is what
        // separates "C++ is faster" from "clang unrolled and rustc didn't".
        #[cfg(target_arch = "aarch64")]
        g.bench_with_input(BenchmarkId::new("rust_unrolled", &id), &f, |bch, f| {
            bch.iter(|| unsafe {
                sapient_lang_harness::rust_unrolled::dot_q4_k_4rows_r4_q8k_unrolled(
                    black_box(&f.packed),
                    black_box(&f.x_i8),
                    black_box(&f.x_scales),
                    black_box(&f.x_sums),
                )
            })
        });
        #[cfg(target_arch = "aarch64")]
        g.bench_with_input(BenchmarkId::new("rust_unrolled_vectail", &id), &f, |bch, f| {
            bch.iter(|| unsafe {
                sapient_lang_harness::rust_unrolled::dot_q4_k_4rows_r4_q8k_unrolled_vectail(
                    black_box(&f.packed),
                    black_box(&f.x_i8),
                    black_box(&f.x_scales),
                    black_box(&f.x_sums),
                )
            })
        });
        #[cfg(target_arch = "aarch64")]
        g.bench_with_input(BenchmarkId::new("rust_unrolled_unchecked", &id), &f, |bch, f| {
            bch.iter(|| unsafe {
                sapient_lang_harness::rust_unrolled::dot_q4_k_4rows_r4_q8k_unrolled_unchecked(
                    black_box(&f.packed),
                    black_box(&f.x_i8),
                    black_box(&f.x_scales),
                    black_box(&f.x_sums),
                )
            })
        });
        g.bench_with_input(BenchmarkId::new("cpp", &id), &f, |bch, f| {
            bch.iter(|| unsafe {
                cpp::dot_q4_k_4rows_r4_q8k(
                    black_box(&f.packed),
                    black_box(&f.x_i8),
                    black_box(&f.x_scales),
                    black_box(&f.x_sums),
                )
            })
        });
    }
    g.finish();
}

fn bench_q6_k(c: &mut Criterion) {
    if !has_dotprod() {
        return;
    }
    let mut g = c.benchmark_group("q6_k_4rows_r4_q8k");
    for &(name, k) in SHAPES {
        let f = q6_k(0xC0FF_EE02, k);

        let a = unsafe { rk::dot_q6_k_4rows_r4_q8k_neon(&f.packed, &f.x_i8, &f.x_scales) };
        let b = unsafe { cpp::dot_q6_k_4rows_r4_q8k(&f.packed, &f.x_i8, &f.x_scales) };
        for r in 0..4 {
            assert_eq!(a[r].to_bits(), b[r].to_bits(), "{name}: arms disagree at row {r}");
        }

        g.throughput(Throughput::Elements((4 * k) as u64));
        let id = format!("{name}/k={k}");

        g.bench_with_input(BenchmarkId::new("rust", &id), &f, |bch, f| {
            bch.iter(|| unsafe {
                rk::dot_q6_k_4rows_r4_q8k_neon(
                    black_box(&f.packed),
                    black_box(&f.x_i8),
                    black_box(&f.x_scales),
                )
            })
        });
        g.bench_with_input(BenchmarkId::new("cpp", &id), &f, |bch, f| {
            bch.iter(|| unsafe {
                cpp::dot_q6_k_4rows_r4_q8k(
                    black_box(&f.packed),
                    black_box(&f.x_i8),
                    black_box(&f.x_scales),
                )
            })
        });
    }
    g.finish();
}

criterion_group!(benches, bench_q4_k, bench_q6_k);
criterion_main!(benches);
