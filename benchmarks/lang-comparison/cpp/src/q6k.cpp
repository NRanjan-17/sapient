// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
//
// Line-for-line C++ port of the Q6_K kernels in
// crates/sapient-backends/cpu/src/kernels/quant.rs.
//
// Q6_K block layout (210 bytes / 256 weights), quant.rs:2556-2563:
//   [0..127]   ql    — low nibbles
//   [128..191] qh    — upper 2 bits, 4 per byte
//   [192..207] scales — i8, one per 16-element group
//   [208..209] d      — f16 super-block scale
//
// The +0/+2/+4/+6 per-half scale indexing below is the exact indexing ggml's
// dequantize_row_q6_K uses. Getting it wrong is the historical token-salad bug
// (CLAUDE.md, "Q6_K dequantization scale indexing"). Do not "simplify" it.

#include "sapient_kernels.h"

#include <arm_neon.h>
#include <string.h>

static inline float f16_at(const uint8_t *p) {
    uint16_t bits;
    memcpy(&bits, p, sizeof bits);
    __fp16 h;
    memcpy(&h, &bits, sizeof h);
    return (float)h;
}

// ── dot_q6_k_4rows_r4_q8k_neon (quant.rs:2119-2196) ──────────────────────────
// Four Q6_K rows (R4 interleaved) x one Q8_K activation row. Per-row integer
// scale accumulation; one f32 fma per row per super-block. Q6_K has no min
// term — the -32 bias is folded into the int8 quants before the sdot.
//
// 32.4% of measured decode time.
__attribute__((target("dotprod")))
extern "C" sapient_f32x4 sapient_cpp_dot_q6_k_4rows_r4_q8k(
    const uint8_t *packed, size_t packed_len,
    const int8_t  *x_i8,
    const float   *x_scales, size_t x_scales_len)
{
    const uint8x16_t mask0f = vdupq_n_u8(0x0F);
    const uint8x16_t mask3  = vdupq_n_u8(0x03);
    const int8x16_t  m32    = vdupq_n_s8(32);

    sapient_f32x4 acc = {{0.0f, 0.0f, 0.0f, 0.0f}};
    const size_t nb = packed_len / (4 * SAPIENT_Q6_K_BLOCK_BYTES);
    const size_t nlim = (x_scales_len < nb) ? x_scales_len : nb;  // Rust: .take(nb)
    size_t x_off = 0;

    for (size_t b = 0; b < nlim; ++b) {
        const float db = x_scales[b];
        const size_t gbase = b * 4 * SAPIENT_Q6_K_BLOCK_BYTES;

        size_t ql_off = 0, qh_off = 0, sc_base = 0, xo = x_off;
        int32_t isum[4] = {0, 0, 0, 0};

        for (size_t h = 0; h < SAPIENT_QK_K / 128; ++h) {
            for (size_t li = 0; li < 2; ++li) {
                const size_t l0 = li * 16;          // Rust: for &l0 in &[0, 16]
                const size_t is = l0 / 16;

                const int8x16_t xv1 = vld1q_s8(x_i8 + xo + l0);
                const int8x16_t xv2 = vld1q_s8(x_i8 + xo + 32 + l0);
                const int8x16_t xv3 = vld1q_s8(x_i8 + xo + 64 + l0);
                const int8x16_t xv4 = vld1q_s8(x_i8 + xo + 96 + l0);

                for (size_t r = 0; r < 4; ++r) {
                    const uint8_t *block = packed + gbase + r * SAPIENT_Q6_K_BLOCK_BYTES;
                    const uint8_t *ql = block;
                    const uint8_t *qh = block + 128;
                    const int8_t  *sc = (const int8_t *)(block + 192);

                    const uint8x16_t ql_lo = vld1q_u8(ql + ql_off + l0);
                    const uint8x16_t ql_hi = vld1q_u8(ql + ql_off + l0 + 32);
                    const uint8x16_t qhv   = vld1q_u8(qh + qh_off + l0);

                    const uint8x16_t q1 = vorrq_u8(
                        vandq_u8(ql_lo, mask0f),
                        vshlq_n_u8(vandq_u8(qhv, mask3), 4));
                    const uint8x16_t q2 = vorrq_u8(
                        vandq_u8(ql_hi, mask0f),
                        vshlq_n_u8(vandq_u8(vshrq_n_u8(qhv, 2), mask3), 4));
                    const uint8x16_t q3 = vorrq_u8(
                        vshrq_n_u8(ql_lo, 4),
                        vshlq_n_u8(vandq_u8(vshrq_n_u8(qhv, 4), mask3), 4));
                    const uint8x16_t q4 = vorrq_u8(
                        vshrq_n_u8(ql_hi, 4),
                        vshlq_n_u8(vandq_u8(vshrq_n_u8(qhv, 6), mask3), 4));

                    // Rust's `group!` macro, expanded. sc_off is +0/+2/+4/+6.
                    #define SAPIENT_Q6K_GROUP(Q, SC_OFF, XV)                       \
                        do {                                                       \
                            const int8x16_t qm =                                   \
                                vsubq_s8(vreinterpretq_s8_u8(Q), m32);             \
                            const int32_t dot =                                    \
                                vaddvq_s32(vdotq_s32(vdupq_n_s32(0), qm, XV));     \
                            isum[r] += (int32_t)sc[sc_base + is + (SC_OFF)] * dot; \
                        } while (0)

                    SAPIENT_Q6K_GROUP(q1, 0, xv1);
                    SAPIENT_Q6K_GROUP(q2, 2, xv2);
                    SAPIENT_Q6K_GROUP(q3, 4, xv3);
                    SAPIENT_Q6K_GROUP(q4, 6, xv4);

                    #undef SAPIENT_Q6K_GROUP
                }
            }
            xo      += 128;
            ql_off  += 64;
            qh_off  += 32;
            sc_base += 8;
        }

        for (size_t r = 0; r < 4; ++r) {
            const uint8_t *block = packed + gbase + r * SAPIENT_Q6_K_BLOCK_BYTES;
            const float d = f16_at(block + 208);
            acc.v[r] += db * d * (float)isum[r];
        }
        x_off += SAPIENT_QK_K;
    }
    return acc;
}

// ── dot_q6_k_row_q8k_scalar (quant.rs:1878-1921) ─────────────────────────────
// Scalar oracle, for the pure-C++ sanitizer driver.
extern "C" float sapient_cpp_dot_q6_k_row_q8k_scalar(
    const uint8_t *row, size_t row_len,
    const int8_t *x_i8, const float *x_scales)
{
    float acc = 0.0f;
    size_t x_off = 0;
    const size_t nblk = row_len / SAPIENT_Q6_K_BLOCK_BYTES;

    for (size_t bi = 0; bi < nblk; ++bi) {
        const uint8_t *block = row + bi * SAPIENT_Q6_K_BLOCK_BYTES;
        const uint8_t *ql = block;
        const uint8_t *qh = block + 128;
        const int8_t  *sc = (const int8_t *)(block + 192);
        const float d = f16_at(block + 208);

        size_t ql_off = 0, qh_off = 0, sc_base = 0, xo = x_off;
        int32_t isum = 0;

        for (size_t hh = 0; hh < SAPIENT_QK_K / 128; ++hh) {
            for (size_t li = 0; li < 2; ++li) {
                const size_t l0 = li * 16;
                const size_t is = l0 / 16;
                // (sub, sc_off, x_add) tuples, exactly as the Rust array.
                static const size_t SUB[4][3] = {{0,0,0},{1,2,32},{2,4,64},{3,6,96}};
                for (size_t t = 0; t < 4; ++t) {
                    const size_t sub = SUB[t][0], sc_off = SUB[t][1], x_add = SUB[t][2];
                    int32_t dot = 0;
                    for (size_t l = 0; l < 16; ++l) {
                        const size_t bb = l0 + l;
                        uint8_t q;
                        switch (sub) {
                            case 0: q = (uint8_t)((ql[ql_off + bb] & 0x0F)
                                        | ((qh[qh_off + bb] & 3) << 4)); break;
                            case 1: q = (uint8_t)((ql[ql_off + bb + 32] & 0x0F)
                                        | (((qh[qh_off + bb] >> 2) & 3) << 4)); break;
                            case 2: q = (uint8_t)((ql[ql_off + bb] >> 4)
                                        | (((qh[qh_off + bb] >> 4) & 3) << 4)); break;
                            default: q = (uint8_t)((ql[ql_off + bb + 32] >> 4)
                                        | (((qh[qh_off + bb] >> 6) & 3) << 4)); break;
                        }
                        const size_t xi = xo + x_add + l0 + l;
                        dot += ((int32_t)q - 32) * (int32_t)x_i8[xi];
                    }
                    isum += (int32_t)sc[sc_base + is + sc_off] * dot;
                }
            }
            xo      += 128;
            ql_off  += 64;
            qh_off  += 32;
            sc_base += 8;
        }
        acc += x_scales[bi] * d * (float)isum;
        x_off += SAPIENT_QK_K;
    }
    return acc;
}
