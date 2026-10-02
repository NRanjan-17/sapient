// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! Throughput of the Q8_0 W8A8 GEMM (`matmul_nt` with Q8_0 weights) on the
//! shapes the vision tower and the SmolVLA action expert run. Ignored — a
//! measurement, not a gate:
//!
//! ```bash
//! cargo test -p sapient-backends-cpu --release --test q8_gemm_bench -- --ignored --nocapture
//! RAYON_NUM_THREADS=1 cargo test … # single-core kernel rate
//! ```

use sapient_backends_cpu::kernels::matmul::matmul_nt;
use sapient_backends_cpu::kernels::quant::quantize_q8_0_block;
use sapient_core::{DType, Tensor};

fn q8_weight(n: usize, k: usize) -> Tensor {
    let bytes: Vec<u8> = (0..n * k / 32)
        .flat_map(|b| {
            let blk: Vec<f32> = (0..32)
                .map(|i| (((b * 32 + i) * 2654435761usize % 2003) as f32 / 2003.0 - 0.5) * 0.1)
                .collect();
            quantize_q8_0_block(&blk)
        })
        .collect();
    Tensor::from_quant_bytes(&bytes, vec![n, k], DType::Q8_0).unwrap()
}

#[test]
#[ignore = "throughput measurement"]
fn q8_gemm_throughput() {
    // (label, m, k, n)
    let shapes = [
        ("tower q/k/v   ", 1024usize, 768usize, 768usize),
        ("tower fc1     ", 1024, 768, 3072),
        ("tower fc2     ", 1024, 3072, 768),
        ("expert gate/up", 50, 736, 2048),
        ("expert down   ", 50, 2048, 720),
        ("prefix mlp    ", 78, 960, 2560),
    ];
    println!("threads: {}", rayon::current_num_threads());
    for (label, m, k, n) in shapes {
        let w = q8_weight(n, k);
        let x: Vec<f32> = (0..m * k)
            .map(|i| ((i * 40503 % 1009) as f32 / 1009.0 - 0.5) * 4.0)
            .collect();
        let xt = Tensor::from_f32(&x, vec![m, k]).unwrap();
        // Run for at least 0.3 s: on Apple Silicon a core that has not ramped
        // up reads ~2× low, and the small shapes take under a millisecond.
        let mut best = f64::MAX;
        let begun = std::time::Instant::now();
        while begun.elapsed().as_secs_f64() < 0.3 {
            let t = std::time::Instant::now();
            let y = matmul_nt(&xt, &w).unwrap();
            best = best.min(t.elapsed().as_secs_f64());
            std::hint::black_box(y);
        }
        // Activation quantization alone (what the GEMM does first, serially here).
        let mut q_best = f64::MAX;
        for _ in 0..5 {
            let t = std::time::Instant::now();
            for row in x.chunks_exact(k) {
                std::hint::black_box(
                    sapient_backends_cpu::kernels::quant::quantize_row_to_i8_blocks(row),
                );
            }
            q_best = q_best.min(t.elapsed().as_secs_f64());
        }
        print!("[quantize 1-thread {:5.2} ms] ", q_best * 1e3);
        let gmac = (m * k * n) as f64 / best / 1e9;
        println!(
            "{label} m={m:4} k={k:4} n={n:4}  {:7.2} ms  {gmac:6.1} GMAC/s",
            best * 1e3
        );
    }
}

/// The 4×4 tile kernel alone on cache-resident data — the per-core ceiling the
/// full GEMM (quantization, scheduling, output layout) should approach.
#[cfg(target_arch = "aarch64")]
#[test]
#[ignore = "throughput measurement"]
fn q8_tile_kernel_ceiling() {
    use sapient_backends_cpu::kernels::quant::{
        dot_q8_0_4rows_sdot_x4, dot_q8_0_row_sdot_x4, q8_0_row_scales, quantize_row_to_i8_blocks,
    };
    if !std::arch::is_aarch64_feature_detected!("dotprod") {
        return;
    }
    let k = 768usize;
    let bpr = k / 32;
    let w = q8_weight(4, k);
    let wb = w.as_quant_blocks();
    let rb = bpr * 34;
    let rows: Vec<&[u8]> = (0..4).map(|l| &wb[l * rb..(l + 1) * rb]).collect();
    let mut ws = vec![0.0f32; 4 * bpr];
    for l in 0..4 {
        q8_0_row_scales(rows[l], &mut ws[l * bpr..(l + 1) * bpr]);
    }
    let mut ws_t = vec![0.0f32; bpr * 4];
    for l in 0..4 {
        for b in 0..bpr {
            ws_t[b * 4 + l] = ws[l * bpr + b];
        }
    }
    let xq: Vec<(Vec<i8>, Vec<f32>)> = (0..4)
        .map(|r| {
            let x: Vec<f32> = (0..k)
                .map(|i| ((i * 31 + r * 7) % 97) as f32 - 48.0)
                .collect();
            quantize_row_to_i8_blocks(&x)
        })
        .collect();
    let mut xs_t = vec![0.0f32; bpr * 4];
    for r in 0..4 {
        for b in 0..bpr {
            xs_t[b * 4 + r] = xq[r].1[b];
        }
    }
    let x = [&xq[0].0[..], &xq[1].0[..], &xq[2].0[..], &xq[3].0[..]];
    let iters = 1_000_000usize;
    let macs = (iters * 16 * k) as f64 / 1e9;
    let mut sink = 0.0f32;
    for round in 0..3 {
        let t = std::time::Instant::now();
        for _ in 0..iters {
            let r = unsafe {
                dot_q8_0_4rows_sdot_x4(
                    [rows[0], rows[1], rows[2], rows[3]],
                    &ws_t,
                    4,
                    0,
                    x,
                    &xs_t,
                    4,
                    0,
                )
            };
            sink += r[0][0];
        }
        let s44 = t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        for _ in 0..iters {
            for l in 0..4 {
                let r = unsafe {
                    dot_q8_0_row_sdot_x4(rows[l], &ws[l * bpr..(l + 1) * bpr], x, &xs_t, 4, 0)
                };
                sink += r[0];
            }
        }
        let s14 = t.elapsed().as_secs_f64();
        println!(
            "round {round}: 4x4 tile {:6.1} GMAC/s · 1x4 tile {:6.1} GMAC/s",
            macs / s44,
            macs / s14
        );
    }
    std::hint::black_box(sink);
}

#[test]
#[ignore = "throughput measurement"]
fn exp_approx_rate() {
    use sapient_backends_cpu::kernels::elementwise::exp_approx_slice;
    let src: Vec<f32> = (0..1 << 20).map(|i| -((i % 4099) as f32) * 0.005).collect();
    let (mut a, mut b) = (f64::MAX, f64::MAX);
    for _ in 0..200 {
        let mut v = src.clone();
        let t = std::time::Instant::now();
        exp_approx_slice(&mut v);
        a = a.min(t.elapsed().as_secs_f64());
        std::hint::black_box(&v);
        let mut v = src.clone();
        let t = std::time::Instant::now();
        for x in v.iter_mut() {
            *x = x.exp();
        }
        b = b.min(t.elapsed().as_secs_f64());
        std::hint::black_box(&v);
    }
    let n = src.len() as f64;
    println!(
        "exp_approx {:.2} ns/elem · libm exp {:.2} ns/elem",
        a / n * 1e9,
        b / n * 1e9
    );
}
