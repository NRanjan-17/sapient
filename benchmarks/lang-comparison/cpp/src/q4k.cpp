// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
//
// Line-for-line C++ port of the Q4_K kernels in
// crates/sapient-backends/cpu/src/kernels/quant.rs.
//
// Each function below names the Rust original and its line range. The loop
// nest, the index arithmetic, and — critically — the ORDER of floating-point
// operations are preserved exactly, because the gate is bit-identity, not
// approximate agreement.

#include "sapient_kernels.h"

#include <arm_neon.h>
#include <string.h>

// ── fp16 → f32 ───────────────────────────────────────────────────────────────
// Rust uses `half::f16::from_le_bytes([lo, hi]).to_f32()`. Both that and the
// AArch64 FCVT below are IEEE-754 binary16→binary32, which is exact for every
// input (binary32 can represent every binary16 value, including subnormals,
// infinities and NaN payloads). Gated by `f16_conversion_matches_half_crate`.
extern "C" float sapient_cpp_f16_to_f32(uint16_t bits) {
    __fp16 h;
    memcpy(&h, &bits, sizeof h);
    return (float)h;
}

static inline float f16_at(const uint8_t *p) {
    uint16_t bits;
    memcpy(&bits, p, sizeof bits);   // the bytes are little-endian on-disk
    __fp16 h;
    memcpy(&h, &bits, sizeof h);
    return (float)h;
}

// ── get_scale_min_k4 (quant.rs:596-605) ──────────────────────────────────────
// Unpacks the 6-bit (scale, min) pair for K-quant sub-block j from the 12-byte
// `scales` field. The +4 / high-bit packing is ggml's; changing it silently
// corrupts every Q4_K tensor (this is the Q6_K-postmortem bug class).
static inline void get_scale_min_k4(size_t j, const uint8_t *scales,
                                    uint8_t *sc, uint8_t *m) {
    if (j < 4) {
        *sc = scales[j] & 63;
        *m  = scales[j + 4] & 63;
    } else {
        *sc = (uint8_t)((scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4));
        *m  = (uint8_t)((scales[j + 4] >> 4)   | ((scales[j]     >> 6) << 4));
    }
}

// ── dot_q4_k_4rows_r4_q8k_neon (quant.rs:1410-1472) ──────────────────────────
// Four Q4_K rows in the R4 interleaved layout x one Q8_K activation row.
// Per-row `isum`/`imin` accumulate in the integer domain across the four
// 64-weight groups; one f32 fma per row per super-block.
//
// 64.1% of measured decode time — the single most important function here.
__attribute__((target("dotprod")))
extern "C" sapient_f32x4 sapient_cpp_dot_q4_k_4rows_r4_q8k(
    const uint8_t *packed, size_t packed_len,
    const int8_t  *x_i8,
    const float   *x_scales, size_t x_scales_len,
    const int32_t *x_sums)
{
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    sapient_f32x4 acc = {{0.0f, 0.0f, 0.0f, 0.0f}};

    const size_t nb = packed_len / (4 * SAPIENT_Q4_K_BLOCK_BYTES);
    const size_t nlim = (x_scales_len < nb) ? x_scales_len : nb;  // Rust: .take(nb)
    size_t x_off = 0;

    for (size_t b = 0; b < nlim; ++b) {
        const float db = x_scales[b];
        const size_t gbase = b * 4 * SAPIENT_Q4_K_BLOCK_BYTES;

        float dv[4], dminv[4];
        for (size_t r = 0; r < 4; ++r) {
            const uint8_t *blk = packed + gbase + r * SAPIENT_Q4_K_BLOCK_BYTES;
            dv[r]    = f16_at(blk);
            dminv[r] = f16_at(blk + 2);
        }

        size_t q_off = 0, is = 0;
        int32_t isum[4] = {0, 0, 0, 0};
        int32_t imin[4] = {0, 0, 0, 0};

        for (size_t g = 0; g < SAPIENT_QK_K / 64; ++g) {
            const int8x16_t xlo0 = vld1q_s8(x_i8 + x_off);
            const int8x16_t xlo1 = vld1q_s8(x_i8 + x_off + 16);
            const int8x16_t xhi0 = vld1q_s8(x_i8 + x_off + 32);
            const int8x16_t xhi1 = vld1q_s8(x_i8 + x_off + 48);
            const int32_t sum_lo = x_sums[x_off / SAPIENT_QK];
            const int32_t sum_hi = x_sums[(x_off + 32) / SAPIENT_QK];

            for (size_t r = 0; r < 4; ++r) {
                const uint8_t *blk = packed + gbase + r * SAPIENT_Q4_K_BLOCK_BYTES;
                const uint8_t *scales = blk + 4;
                const uint8_t *qs = blk + 16;

                uint8_t sc1, m1, sc2, m2;
                get_scale_min_k4(is,     scales, &sc1, &m1);
                get_scale_min_k4(is + 1, scales, &sc2, &m2);

                const uint8x16_t q0 = vld1q_u8(qs + q_off);
                const uint8x16_t q1 = vld1q_u8(qs + q_off + 16);
                const int8x16_t lo0 = vreinterpretq_s8_u8(vandq_u8(q0, mask));
                const int8x16_t lo1 = vreinterpretq_s8_u8(vandq_u8(q1, mask));
                const int8x16_t hi0 = vreinterpretq_s8_u8(vshrq_n_u8(q0, 4));
                const int8x16_t hi1 = vreinterpretq_s8_u8(vshrq_n_u8(q1, 4));

                const int32x4_t zero = vdupq_n_s32(0);
                // Rust reaches this instruction through inline `asm!`; C++ uses
                // the intrinsic. Same instruction — that is the point.
                const int32_t dot_lo =
                    vaddvq_s32(vdotq_s32(vdotq_s32(zero, lo0, xlo0), lo1, xlo1));
                const int32_t dot_hi =
                    vaddvq_s32(vdotq_s32(vdotq_s32(zero, hi0, xhi0), hi1, xhi1));

                isum[r] += (int32_t)sc1 * dot_lo + (int32_t)sc2 * dot_hi;
                imin[r] += (int32_t)m1 * sum_lo + (int32_t)m2 * sum_hi;
            }
            x_off += 64;
            q_off += 32;
            is    += 2;
        }

        for (size_t r = 0; r < 4; ++r) {
            acc.v[r] += db * (dv[r] * (float)isum[r] - dminv[r] * (float)imin[r]);
        }
    }
    return acc;
}

// ── dot_q4_k_row_q8k_scalar (quant.rs:394-429) ───────────────────────────────
// The scalar oracle, ported so the pure-C++ sanitizer driver can cross-check
// without linking Rust.
extern "C" float sapient_cpp_dot_q4_k_row_q8k_scalar(
    const uint8_t *row, size_t row_len,
    const int8_t *x_i8, const float *x_scales, const int32_t *x_sums)
{
    float acc = 0.0f;
    size_t x_off = 0;
    const size_t nblk = row_len / SAPIENT_Q4_K_BLOCK_BYTES;

    for (size_t b = 0; b < nblk; ++b) {
        const uint8_t *block = row + b * SAPIENT_Q4_K_BLOCK_BYTES;
        const float d    = f16_at(block);
        const float dmin = f16_at(block + 2);
        const uint8_t *scales = block + 4;
        const uint8_t *qs = block + 16;

        size_t q_off = 0, is = 0;
        int32_t isum = 0, imin = 0;
        for (size_t g = 0; g < SAPIENT_QK_K / 64; ++g) {
            uint8_t sc1, m1, sc2, m2;
            get_scale_min_k4(is,     scales, &sc1, &m1);
            get_scale_min_k4(is + 1, scales, &sc2, &m2);
            const int8_t *xlo = x_i8 + x_off;
            const int8_t *xhi = x_i8 + x_off + 32;
            int32_t dot_lo = 0, dot_hi = 0;
            for (size_t l = 0; l < 32; ++l) {
                dot_lo += (int32_t)(qs[q_off + l] & 0x0F) * (int32_t)xlo[l];
                dot_hi += (int32_t)(qs[q_off + l] >> 4)   * (int32_t)xhi[l];
            }
            isum += (int32_t)sc1 * dot_lo + (int32_t)sc2 * dot_hi;
            imin += (int32_t)m1 * x_sums[x_off / SAPIENT_QK]
                  + (int32_t)m2 * x_sums[(x_off + 32) / SAPIENT_QK];
            x_off += 64;
            q_off += 32;
            is    += 2;
        }
        acc += x_scales[b] * (d * (float)isum - dmin * (float)imin);
    }
    return acc;
}
