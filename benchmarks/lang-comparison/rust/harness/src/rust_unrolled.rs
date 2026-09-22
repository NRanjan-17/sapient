// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! **The decisive arm.** A Rust Q4_K kernel identical to the shipped one except
//! that the `for r in 0..4` row loop is manually unrolled.
//!
//! Why this exists: the C++ port measured ~1.30x faster than the shipped Rust
//! Q4_K kernel. The assembly shows why — clang unrolls the row loop (16 `sdot`
//! per body) and rustc does not (4), so the shipped Rust runs one `sdot`->`addv`
//! dependency chain at a time while C++ interleaves four independent ones.
//!
//! If unrolling the Rust closes the gap, the 1.30x is an **unroll decision**,
//! not a language tax — and the fix is a Rust-side edit, not a rewrite. That is
//! precisely the question this study exists to answer, so it gets its own arm
//! rather than an argument.
//!
//! Nothing here is a proposed change to the shipping kernel: it is an
//! experimental variable, bit-identity gated like every other arm.

#![cfg(target_arch = "aarch64")]

use half::f16;
use sapient_backends_cpu::kernels::quant::{Q4_K_BLOCK_BYTES, QK, QK_K};

/// Unpack the 6-bit (scale, min) pair for sub-block `j` — a verbatim copy of
/// the private `get_scale_min_k4` (quant.rs:596), which is not `pub`.
#[inline(always)]
fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        (
            (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4),
            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
        )
    }
}

/// `sdot` via inline asm — byte-for-byte the shipped helper (quant.rs:804), so
/// this arm differs from the shipped kernel in loop structure ONLY.
#[target_feature(enable = "neon,dotprod")]
#[inline]
unsafe fn sdot_s32(
    acc: std::arch::aarch64::int32x4_t,
    w: std::arch::aarch64::int8x16_t,
    x: std::arch::aarch64::int8x16_t,
) -> std::arch::aarch64::int32x4_t {
    let mut a = acc;
    core::arch::asm!(
        "sdot {0:v}.4s, {1:v}.16b, {2:v}.16b",
        inout(vreg) a,
        in(vreg) w,
        in(vreg) x,
        options(nomem, nostack),
    );
    a
}

/// Row loop unrolled; `isum`/`imin` are eight scalars instead of two `[i32; 4]`
/// arrays, so LLVM can keep them in registers rather than load-modify-storing
/// them through the stack every iteration.
///
/// Per-row arithmetic and its ORDER are unchanged, so the result is bit-identical
/// to `dot_q4_k_4rows_r4_q8k_neon` (gated in `tests/bit_identity.rs`).
///
/// # Safety
/// Same contract as the shipped kernel: caller must have verified
/// `FEAT_DotProd`, and `packed` must be a whole Q4_K_R4 row-group.
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn dot_q4_k_4rows_r4_q8k_unrolled(
    packed: &[u8],
    x_i8: &[i8],
    x_scales: &[f32],
    x_sums: &[i32],
) -> [f32; 4] {
    use std::arch::aarch64::*;
    let mask = vdupq_n_u8(0x0F);
    let mut acc = [0.0f32; 4];
    let nb = packed.len() / (4 * Q4_K_BLOCK_BYTES);
    let mut x_off = 0usize;

    for (b, &db) in x_scales.iter().enumerate().take(nb) {
        let gbase = b * 4 * Q4_K_BLOCK_BYTES;

        let mut dv = [0.0f32; 4];
        let mut dminv = [0.0f32; 4];
        for r in 0..4 {
            let blk = &packed[gbase + r * Q4_K_BLOCK_BYTES..];
            dv[r] = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
            dminv[r] = f16::from_le_bytes([blk[2], blk[3]]).to_f32();
        }

        let mut q_off = 0usize;
        let mut is = 0usize;
        // Eight independent scalar accumulators — the whole point of this arm.
        let (mut is0, mut is1, mut is2, mut is3) = (0i32, 0i32, 0i32, 0i32);
        let (mut im0, mut im1, mut im2, mut im3) = (0i32, 0i32, 0i32, 0i32);

        for _ in 0..(QK_K / 64) {
            let xlo0 = vld1q_s8(x_i8.as_ptr().add(x_off));
            let xlo1 = vld1q_s8(x_i8.as_ptr().add(x_off + 16));
            let xhi0 = vld1q_s8(x_i8.as_ptr().add(x_off + 32));
            let xhi1 = vld1q_s8(x_i8.as_ptr().add(x_off + 48));
            let sum_lo = x_sums[x_off / QK];
            let sum_hi = x_sums[(x_off + 32) / QK];

            // `$r` is a literal, so every offset folds to a constant.
            macro_rules! row {
                ($r:literal, $isum:ident, $imin:ident) => {{
                    let base = gbase + $r * Q4_K_BLOCK_BYTES;
                    let blk = &packed[base..base + Q4_K_BLOCK_BYTES];
                    let scales = &blk[4..16];
                    let qs = &blk[16..Q4_K_BLOCK_BYTES];
                    let (sc1, m1) = get_scale_min_k4(is, scales);
                    let (sc2, m2) = get_scale_min_k4(is + 1, scales);

                    let q0 = vld1q_u8(qs.as_ptr().add(q_off));
                    let q1 = vld1q_u8(qs.as_ptr().add(q_off + 16));
                    let lo0 = vreinterpretq_s8_u8(vandq_u8(q0, mask));
                    let lo1 = vreinterpretq_s8_u8(vandq_u8(q1, mask));
                    let hi0 = vreinterpretq_s8_u8(vshrq_n_u8::<4>(q0));
                    let hi1 = vreinterpretq_s8_u8(vshrq_n_u8::<4>(q1));

                    let zero = vdupq_n_s32(0);
                    let dot_lo = vaddvq_s32(sdot_s32(sdot_s32(zero, lo0, xlo0), lo1, xlo1));
                    let dot_hi = vaddvq_s32(sdot_s32(sdot_s32(zero, hi0, xhi0), hi1, xhi1));

                    $isum += sc1 as i32 * dot_lo + sc2 as i32 * dot_hi;
                    $imin += m1 as i32 * sum_lo + m2 as i32 * sum_hi;
                }};
            }

            row!(0, is0, im0);
            row!(1, is1, im1);
            row!(2, is2, im2);
            row!(3, is3, im3);

            x_off += 64;
            q_off += 32;
            is += 2;
        }

        let isum = [is0, is1, is2, is3];
        let imin = [im0, im1, im2, im3];
        for r in 0..4 {
            acc[r] += db * (dv[r] * isum[r] as f32 - dminv[r] * imin[r] as f32);
        }
    }
    acc
}


/// `get_scale_min_k4` with the three slice indexes unchecked.
///
/// # Safety
/// `scales` must have at least 12 elements and `j < 8`.
#[inline(always)]
unsafe fn get_scale_min_k4_unchecked(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales.get_unchecked(j) & 63, scales.get_unchecked(j + 4) & 63)
    } else {
        (
            (scales.get_unchecked(j + 4) & 0x0F) | ((scales.get_unchecked(j - 4) >> 6) << 4),
            (scales.get_unchecked(j + 4) >> 4) | ((scales.get_unchecked(j) >> 6) << 4),
        )
    }
}

/// Fifth arm: unrolled AND unchecked. Isolates what the bounds checks that
/// survive unrolling are actually worth. Row loop unrolled; `isum`/`imin` are eight scalars instead of two `[i32; 4]`
/// arrays, so LLVM can keep them in registers rather than load-modify-storing
/// them through the stack every iteration.
///
/// Per-row arithmetic and its ORDER are unchanged, so the result is bit-identical
/// to `dot_q4_k_4rows_r4_q8k_neon` (gated in `tests/bit_identity.rs`).
///
/// # Safety
/// Same contract as the shipped kernel: caller must have verified
/// `FEAT_DotProd`, and `packed` must be a whole Q4_K_R4 row-group.
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn dot_q4_k_4rows_r4_q8k_unrolled_unchecked(
    packed: &[u8],
    x_i8: &[i8],
    x_scales: &[f32],
    x_sums: &[i32],
) -> [f32; 4] {
    use std::arch::aarch64::*;
    let mask = vdupq_n_u8(0x0F);
    let mut acc = [0.0f32; 4];
    let nb = packed.len() / (4 * Q4_K_BLOCK_BYTES);
    let mut x_off = 0usize;

    for (b, &db) in x_scales.iter().enumerate().take(nb) {
        let gbase = b * 4 * Q4_K_BLOCK_BYTES;

        let mut dv = [0.0f32; 4];
        let mut dminv = [0.0f32; 4];
        for r in 0..4 {
            let blk = packed.get_unchecked(gbase + r * Q4_K_BLOCK_BYTES..);
            dv[r] = f16::from_le_bytes([*blk.get_unchecked(0), *blk.get_unchecked(1)]).to_f32();
            dminv[r] = f16::from_le_bytes([*blk.get_unchecked(2), *blk.get_unchecked(3)]).to_f32();
        }

        let mut q_off = 0usize;
        let mut is = 0usize;
        // Eight independent scalar accumulators — the whole point of this arm.
        let (mut is0, mut is1, mut is2, mut is3) = (0i32, 0i32, 0i32, 0i32);
        let (mut im0, mut im1, mut im2, mut im3) = (0i32, 0i32, 0i32, 0i32);

        for _ in 0..(QK_K / 64) {
            let xlo0 = vld1q_s8(x_i8.as_ptr().add(x_off));
            let xlo1 = vld1q_s8(x_i8.as_ptr().add(x_off + 16));
            let xhi0 = vld1q_s8(x_i8.as_ptr().add(x_off + 32));
            let xhi1 = vld1q_s8(x_i8.as_ptr().add(x_off + 48));
            let sum_lo = *x_sums.get_unchecked(x_off / QK);
            let sum_hi = *x_sums.get_unchecked((x_off + 32) / QK);

            // `$r` is a literal, so every offset folds to a constant.
            macro_rules! row {
                ($r:literal, $isum:ident, $imin:ident) => {{
                    let base = gbase + $r * Q4_K_BLOCK_BYTES;
                    let blk = packed.get_unchecked(base..base + Q4_K_BLOCK_BYTES);
                    let scales = blk.get_unchecked(4..16);
                    let qs = blk.get_unchecked(16..Q4_K_BLOCK_BYTES);
                    let (sc1, m1) = get_scale_min_k4_unchecked(is, scales);
                    let (sc2, m2) = get_scale_min_k4_unchecked(is + 1, scales);

                    let q0 = vld1q_u8(qs.as_ptr().add(q_off));
                    let q1 = vld1q_u8(qs.as_ptr().add(q_off + 16));
                    let lo0 = vreinterpretq_s8_u8(vandq_u8(q0, mask));
                    let lo1 = vreinterpretq_s8_u8(vandq_u8(q1, mask));
                    let hi0 = vreinterpretq_s8_u8(vshrq_n_u8::<4>(q0));
                    let hi1 = vreinterpretq_s8_u8(vshrq_n_u8::<4>(q1));

                    let zero = vdupq_n_s32(0);
                    let dot_lo = vaddvq_s32(sdot_s32(sdot_s32(zero, lo0, xlo0), lo1, xlo1));
                    let dot_hi = vaddvq_s32(sdot_s32(sdot_s32(zero, hi0, xhi0), hi1, xhi1));

                    $isum += sc1 as i32 * dot_lo + sc2 as i32 * dot_hi;
                    $imin += m1 as i32 * sum_lo + m2 as i32 * sum_hi;
                }};
            }

            row!(0, is0, im0);
            row!(1, is1, im1);
            row!(2, is2, im2);
            row!(3, is3, im3);

            x_off += 64;
            q_off += 32;
            is += 2;
        }

        let isum = [is0, is1, is2, is3];
        let imin = [im0, im1, im2, im3];
        for r in 0..4 {
            acc[r] += db * (dv[r] * isum[r] as f32 - dminv[r] * imin[r] as f32);
        }
    }
    acc
}

/// Sixth arm: unrolled body + **NEON epilogue**.
///
/// The opcode diff against C++ showed Rust emitting 8 scalar `scvtf`, 8 `fmul`,
/// 8 `fcvt` and 4 `fsub` per super-block where C++ emitted none — clang had
/// vectorised the per-super-block float tail across all four rows. This arm
/// does the same by hand.
///
/// Bit-identity is preserved because each lane performs exactly the scalar
/// sequence `db * (dv*isum - dminv*imin)` in the same order, with **no FMA
/// fusion** (`vfmaq_f32` would round once instead of twice and break the gate).
///
/// Row loop unrolled; `isum`/`imin` are eight scalars instead of two `[i32; 4]`
/// arrays, so LLVM can keep them in registers rather than load-modify-storing
/// them through the stack every iteration.
///
/// Per-row arithmetic and its ORDER are unchanged, so the result is bit-identical
/// to `dot_q4_k_4rows_r4_q8k_neon` (gated in `tests/bit_identity.rs`).
///
/// # Safety
/// Same contract as the shipped kernel: caller must have verified
/// `FEAT_DotProd`, and `packed` must be a whole Q4_K_R4 row-group.
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn dot_q4_k_4rows_r4_q8k_unrolled_vectail(
    packed: &[u8],
    x_i8: &[i8],
    x_scales: &[f32],
    x_sums: &[i32],
) -> [f32; 4] {
    use std::arch::aarch64::*;
    let mask = vdupq_n_u8(0x0F);
    let mut acc_v = vdupq_n_f32(0.0);
    let nb = packed.len() / (4 * Q4_K_BLOCK_BYTES);
    let mut x_off = 0usize;

    for (b, &db) in x_scales.iter().enumerate().take(nb) {
        let gbase = b * 4 * Q4_K_BLOCK_BYTES;

        let mut dv = [0.0f32; 4];
        let mut dminv = [0.0f32; 4];
        for r in 0..4 {
            let blk = &packed[gbase + r * Q4_K_BLOCK_BYTES..];
            dv[r] = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
            dminv[r] = f16::from_le_bytes([blk[2], blk[3]]).to_f32();
        }

        let mut q_off = 0usize;
        let mut is = 0usize;
        // Eight independent scalar accumulators — the whole point of this arm.
        let (mut is0, mut is1, mut is2, mut is3) = (0i32, 0i32, 0i32, 0i32);
        let (mut im0, mut im1, mut im2, mut im3) = (0i32, 0i32, 0i32, 0i32);

        for _ in 0..(QK_K / 64) {
            let xlo0 = vld1q_s8(x_i8.as_ptr().add(x_off));
            let xlo1 = vld1q_s8(x_i8.as_ptr().add(x_off + 16));
            let xhi0 = vld1q_s8(x_i8.as_ptr().add(x_off + 32));
            let xhi1 = vld1q_s8(x_i8.as_ptr().add(x_off + 48));
            let sum_lo = x_sums[x_off / QK];
            let sum_hi = x_sums[(x_off + 32) / QK];

            // `$r` is a literal, so every offset folds to a constant.
            macro_rules! row {
                ($r:literal, $isum:ident, $imin:ident) => {{
                    let base = gbase + $r * Q4_K_BLOCK_BYTES;
                    let blk = &packed[base..base + Q4_K_BLOCK_BYTES];
                    let scales = &blk[4..16];
                    let qs = &blk[16..Q4_K_BLOCK_BYTES];
                    let (sc1, m1) = get_scale_min_k4(is, scales);
                    let (sc2, m2) = get_scale_min_k4(is + 1, scales);

                    let q0 = vld1q_u8(qs.as_ptr().add(q_off));
                    let q1 = vld1q_u8(qs.as_ptr().add(q_off + 16));
                    let lo0 = vreinterpretq_s8_u8(vandq_u8(q0, mask));
                    let lo1 = vreinterpretq_s8_u8(vandq_u8(q1, mask));
                    let hi0 = vreinterpretq_s8_u8(vshrq_n_u8::<4>(q0));
                    let hi1 = vreinterpretq_s8_u8(vshrq_n_u8::<4>(q1));

                    let zero = vdupq_n_s32(0);
                    let dot_lo = vaddvq_s32(sdot_s32(sdot_s32(zero, lo0, xlo0), lo1, xlo1));
                    let dot_hi = vaddvq_s32(sdot_s32(sdot_s32(zero, hi0, xhi0), hi1, xhi1));

                    $isum += sc1 as i32 * dot_lo + sc2 as i32 * dot_hi;
                    $imin += m1 as i32 * sum_lo + m2 as i32 * sum_hi;
                }};
            }

            row!(0, is0, im0);
            row!(1, is1, im1);
            row!(2, is2, im2);
            row!(3, is3, im3);

            x_off += 64;
            q_off += 32;
            is += 2;
        }

        // Vector epilogue: one pass over four lanes instead of eight scalar
        // converts and multiplies.
        let isum_v = vcvtq_f32_s32(vld1q_s32([is0, is1, is2, is3].as_ptr()));
        let imin_v = vcvtq_f32_s32(vld1q_s32([im0, im1, im2, im3].as_ptr()));
        let dv_v = vld1q_f32(dv.as_ptr());
        let dminv_v = vld1q_f32(dminv.as_ptr());
        let t = vsubq_f32(vmulq_f32(dv_v, isum_v), vmulq_f32(dminv_v, imin_v));
        acc_v = vaddq_f32(acc_v, vmulq_f32(t, vdupq_n_f32(db)));
    }
    let mut acc = [0.0f32; 4];
    vst1q_f32(acc.as_mut_ptr(), acc_v);
    acc
}
