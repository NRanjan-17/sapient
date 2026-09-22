// Optimized Q4_K 4-row R4 Q8_K kernel. Must stay BIT-IDENTICAL to the original.
//
// Three changes, none of which touch float op order:
//   1. Hoist get_scale_min_k4 + per-row pointers out of the group loop.
//   2. Keep the dot products in int32x4_t and fold the sub-block scale with
//      vmlaq_n_s32, reducing ONCE per super-block instead of twice per
//      (row, group). That removes 28 of 32 vaddvq_s32 horizontal reductions
//      per super-block. Exact: sc*(a+b+c+d) == sc*a+sc*b+sc*c+sc*d in int32.
//   3. Unroll the 4-row loop so isum/imin/dv/dminv stay in registers.
#include "sapient_kernels.h"
#include <arm_neon.h>
#include <string.h>

static inline float f16_at_o2(const uint8_t *p) {
    uint16_t bits; memcpy(&bits, p, sizeof bits);
    __fp16 h; memcpy(&h, &bits, sizeof h);
    return (float)h;
}

static inline void gsm2(size_t j, const uint8_t *scales, uint8_t *sc, uint8_t *m) {
    if (j < 4) { *sc = scales[j] & 63; *m = scales[j + 4] & 63; }
    else {
        *sc = (uint8_t)((scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4));
        *m  = (uint8_t)((scales[j + 4] >> 4)   | ((scales[j]     >> 6) << 4));
    }
}

__attribute__((target("dotprod")))
extern "C" sapient_f32x4 sapient_cpp_dot_q4_k_4rows_r4_q8k_opt2(
    const uint8_t *packed, size_t packed_len,
    const int8_t  *x_i8,
    const float   *x_scales, size_t x_scales_len,
    const int32_t *x_sums)
{
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    const int32x4_t  zero = vdupq_n_s32(0);
    sapient_f32x4 acc = {{0.0f, 0.0f, 0.0f, 0.0f}};

    const size_t nb   = packed_len / (4 * SAPIENT_Q4_K_BLOCK_BYTES);
    const size_t nlim = (x_scales_len < nb) ? x_scales_len : nb;
    size_t x_off = 0;

    for (size_t b = 0; b < nlim; ++b) {
        const float db = x_scales[b];
        const uint8_t *r0 = packed + b * 4 * SAPIENT_Q4_K_BLOCK_BYTES;
        const uint8_t *r1 = r0 + SAPIENT_Q4_K_BLOCK_BYTES;
        const uint8_t *r2 = r1 + SAPIENT_Q4_K_BLOCK_BYTES;
        const uint8_t *r3 = r2 + SAPIENT_Q4_K_BLOCK_BYTES;
        const uint8_t *rp[4] = {r0, r1, r2, r3};

        // (1) all 8 sub-block (scale,min) pairs per row, unpacked once.
        uint8_t SC[4][8], MN[4][8];
        float dv[4], dminv[4];
        for (size_t r = 0; r < 4; ++r) {
            dv[r]    = f16_at_o2(rp[r]);
            dminv[r] = f16_at_o2(rp[r] + 2);
            const uint8_t *sp = rp[r] + 4;
            for (size_t j = 0; j < 8; ++j) gsm2(j, sp, &SC[r][j], &MN[r][j]);
        }

        // (2) vector accumulators — one horizontal reduction per row per block.
        int32_t isum[4] = {0, 0, 0, 0};
        int32_t imin[4] = {0, 0, 0, 0};
        size_t q_off = 0, is = 0;

        for (size_t g = 0; g < SAPIENT_QK_K / 64; ++g) {
            const int8x16_t xlo0 = vld1q_s8(x_i8 + x_off);
            const int8x16_t xlo1 = vld1q_s8(x_i8 + x_off + 16);
            const int8x16_t xhi0 = vld1q_s8(x_i8 + x_off + 32);
            const int8x16_t xhi1 = vld1q_s8(x_i8 + x_off + 48);
            const int32_t sum_lo = x_sums[x_off / SAPIENT_QK];
            const int32_t sum_hi = x_sums[(x_off + 32) / SAPIENT_QK];

            // (3) unrolled over the four rows.
            #define ROW(R, VACC)                                                     \
            do {                                                                     \
                const uint8_t *qs = rp[R] + 16 + q_off;                              \
                const uint8x16_t q0 = vld1q_u8(qs);                                  \
                const uint8x16_t q1 = vld1q_u8(qs + 16);                             \
                const int8x16_t lo0 = vreinterpretq_s8_u8(vandq_u8(q0, mask));       \
                const int8x16_t lo1 = vreinterpretq_s8_u8(vandq_u8(q1, mask));       \
                const int8x16_t hi0 = vreinterpretq_s8_u8(vshrq_n_u8(q0, 4));        \
                const int8x16_t hi1 = vreinterpretq_s8_u8(vshrq_n_u8(q1, 4));        \
                const int32x4_t dlo =                                                \
                    vdotq_s32(vdotq_s32(zero, lo0, xlo0), lo1, xlo1);                \
                const int32x4_t dhi =                                                \
                    vdotq_s32(vdotq_s32(zero, hi0, xhi0), hi1, xhi1);                \
                isum[R] += (int32_t)SC[R][is] * vaddvq_s32(dlo)                      \
                         + (int32_t)SC[R][is + 1] * vaddvq_s32(dhi);                 \
                imin[R] += (int32_t)MN[R][is] * sum_lo                               \
                         + (int32_t)MN[R][is + 1] * sum_hi;                          \
            } while (0)

            ROW(0, _); ROW(1, _); ROW(2, _); ROW(3, _);
            #undef ROW

            x_off += 64; q_off += 32; is += 2;
        }

        for (size_t r = 0; r < 4; ++r) {
            acc.v[r] += db * (dv[r] * (float)isum[r] - dminv[r] * (float)imin[r]);
        }
    }
    return acc;
}
