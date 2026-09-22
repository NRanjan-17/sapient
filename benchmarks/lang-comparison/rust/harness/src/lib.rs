// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! Shared fixtures for the Rust-vs-C++ kernel study.
//!
//! Both language arms are handed the **same `&[u8]` / `&[i8]` / `&[f32]`
//! slices**, so input identity is guaranteed by construction rather than by
//! two generators agreeing. That is why there is no C++ fixture generator.
//!
//! Byte patterns follow the repo's own test convention
//! (`quant.rs` `mod tests`): pseudo-random payload with the f16 `d`/`dmin`
//! fields overwritten by sane fixed values, so a random byte stream can never
//! produce a NaN or Inf super-block scale and turn a bit-identity assert into a
//! meaningless `NaN != NaN`.

pub mod rust_unrolled;

use half::f16;
use sapient_backends_cpu::kernels::quant::{
    quantize_row_to_q8k, repack_q4_k_rows4, repack_q6_k_rows4, Q4_K_BLOCK_BYTES,
    Q6_K_BLOCK_BYTES, QK_K,
};

/// Reduction lengths taken from the models this engine actually runs.
/// Every one is a multiple of `QK_K` (256), which every K-quant row is.
pub const SHAPES: &[(&str, usize)] = &[
    ("qwen2.5-1.5b hidden", 1536),
    ("llama-3.2-1b hidden", 2048),
    ("mixtral-8x7b hidden", 4096),
    ("qwen2.5-1.5b ffn", 8960),
    ("mixtral-8x7b ffn", 14336),
];

/// Deterministic LCG byte stream — same constants as `quant.rs`'s `lcg_bytes`.
pub fn lcg_bytes(seed: u64, n: usize) -> Vec<u8> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 33) as u8
        })
        .collect()
}

/// Activation vector with a realistic dynamic range.
pub fn lcg_activations(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / u32::MAX as f32) * 3.0 - 1.5
        })
        .collect()
}

/// One 4-row group of weights plus the Q8_K activations that cover it.
pub struct Fixture {
    /// The four rows, un-interleaved — what the single-row oracle consumes.
    pub rows: Vec<u8>,
    /// The same four rows in the R4 interleaved layout the kernel consumes.
    pub packed: Vec<u8>,
    pub row_bytes: usize,
    pub k: usize,
    /// Q8_K activations: int8 quants, one scale per 256, sums per 32.
    pub x_i8: Vec<i8>,
    pub x_scales: Vec<f32>,
    pub x_sums: Vec<i32>,
}

impl Fixture {
    pub fn row(&self, r: usize) -> &[u8] {
        &self.rows[r * self.row_bytes..(r + 1) * self.row_bytes]
    }
}

/// Build a Q4_K 4-row fixture for reduction length `k`.
pub fn q4_k(seed: u64, k: usize) -> Fixture {
    assert_eq!(k % QK_K, 0, "k must be a multiple of {QK_K}");
    let n = 4usize;
    let row_bytes = k / QK_K * Q4_K_BLOCK_BYTES;
    let mut rows = lcg_bytes(seed, n * row_bytes);
    // Sane super-block scales, exactly as the repo's gate tests do.
    for r in 0..n {
        for blk in 0..k / QK_K {
            let base = r * row_bytes + blk * Q4_K_BLOCK_BYTES;
            rows[base..base + 2].copy_from_slice(&f16::from_f32(0.04).to_le_bytes());
            rows[base + 2..base + 4].copy_from_slice(&f16::from_f32(0.02).to_le_bytes());
        }
    }
    let packed = repack_q4_k_rows4(&rows, n, k);
    let x = lcg_activations(seed ^ 0x9E37_79B9, k);
    let (x_i8, x_scales, x_sums) = quantize_row_to_q8k(&x);
    Fixture { rows, packed, row_bytes, k, x_i8, x_scales, x_sums }
}

/// Build a Q6_K 4-row fixture for reduction length `k`.
pub fn q6_k(seed: u64, k: usize) -> Fixture {
    assert_eq!(k % QK_K, 0, "k must be a multiple of {QK_K}");
    let n = 4usize;
    let row_bytes = k / QK_K * Q6_K_BLOCK_BYTES;
    let mut rows = lcg_bytes(seed, n * row_bytes);
    for r in 0..n {
        for blk in 0..k / QK_K {
            let base = r * row_bytes + blk * Q6_K_BLOCK_BYTES;
            // Q6_K keeps d at [208..210]; there is no dmin.
            rows[base + 208..base + 210].copy_from_slice(&f16::from_f32(0.04).to_le_bytes());
        }
    }
    let packed = repack_q6_k_rows4(&rows, n, k);
    let x = lcg_activations(seed ^ 0x51ED_2701, k);
    let (x_i8, x_scales, x_sums) = quantize_row_to_q8k(&x);
    Fixture { rows, packed, row_bytes, k, x_i8, x_scales, x_sums }
}

/// True when this CPU can run the kernels under study.
pub fn has_dotprod() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::is_aarch64_feature_detected!("dotprod")
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}
