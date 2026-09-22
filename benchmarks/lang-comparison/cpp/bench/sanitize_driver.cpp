// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
//
// Pure-C++ driver for the AddressSanitizer / UndefinedBehaviorSanitizer pass.
//
// Why this exists instead of running the Rust gate under sanitizers: mixing a
// sanitised C++ object into an unsanitised Rust test binary is fragile (the
// interceptors need to be present from process start, and Rust's allocator and
// panic machinery are not instrumented). A standalone driver keeps the
// sanitised surface to exactly the code under test.
//
// It rebuilds the same fixture shape the Rust harness uses — pseudo-random
// payload with valid f16 super-block scales — and drives both the SIMD kernels
// and the scalar oracles, checking they agree. Memory-safety defects (the thing
// a C++ port would newly expose, and Rust would not) show up as ASan reports.
//
//   clang++ -std=c++17 -O1 -g -fsanitize=address,undefined \
//           -fno-omit-frame-pointer -Iinclude \
//           src/q4k.cpp src/q6k.cpp bench/sanitize_driver.cpp -o /tmp/sanitize
//   /tmp/sanitize

#include "sapient_kernels.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

// Same LCG constants as the Rust harness, so the byte patterns are comparable.
static std::vector<uint8_t> lcg_bytes(uint64_t seed, size_t n) {
    std::vector<uint8_t> v(n);
    uint64_t s = seed;
    for (size_t i = 0; i < n; ++i) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        v[i] = (uint8_t)(s >> 33);
    }
    return v;
}

static std::vector<float> lcg_activations(uint64_t seed, size_t n) {
    std::vector<float> v(n);
    uint64_t s = seed;
    for (size_t i = 0; i < n; ++i) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        v[i] = ((float)(uint32_t)(s >> 33) / (float)UINT32_MAX) * 3.0f - 1.5f;
    }
    return v;
}

// Q8_K activation quantisation — mirrors quantize_row_to_q8k (quant.rs:366).
static void quantize_q8k(const std::vector<float> &x, std::vector<int8_t> &q,
                         std::vector<float> &scales, std::vector<int32_t> &sums) {
    size_t nsuper = x.size() / SAPIENT_QK_K;
    q.assign(x.size(), 0);
    scales.assign(nsuper, 0.f);
    sums.assign(x.size() / SAPIENT_QK, 0);
    for (size_t b = 0; b < nsuper; ++b) {
        float mx = 0.f;
        for (size_t i = 0; i < SAPIENT_QK_K; ++i)
            mx = std::fmax(mx, std::fabs(x[b * SAPIENT_QK_K + i]));
        float scale = mx > 0.f ? mx / 127.f : 1.f;
        float inv = scale > 0.f ? 1.f / scale : 0.f;
        for (size_t i = 0; i < SAPIENT_QK_K; ++i) {
            float r = std::round(x[b * SAPIENT_QK_K + i] * inv);
            r = std::fmin(127.f, std::fmax(-127.f, r));
            q[b * SAPIENT_QK_K + i] = (int8_t)r;
        }
        scales[b] = scale;
        for (size_t j = 0; j < SAPIENT_QK_K / SAPIENT_QK; ++j) {
            int32_t acc = 0;
            size_t base = b * SAPIENT_QK_K + j * SAPIENT_QK;
            for (size_t l = 0; l < SAPIENT_QK; ++l) acc += q[base + l];
            sums[b * (SAPIENT_QK_K / SAPIENT_QK) + j] = acc;
        }
    }
}

// Interleave 4 rows into the R4 layout — mirrors repack_q4_k_rows4 (quant.rs:1229).
static std::vector<uint8_t> repack_r4(const std::vector<uint8_t> &rows, size_t k,
                                      size_t block_bytes) {
    size_t nb = k / SAPIENT_QK_K;
    std::vector<uint8_t> out(rows.size());
    for (size_t r = 0; r < 4; ++r)
        for (size_t b = 0; b < nb; ++b) {
            const uint8_t *src = rows.data() + (r * nb + b) * block_bytes;
            uint8_t *dst = out.data() + (b * 4 + r) * block_bytes;
            memcpy(dst, src, block_bytes);
        }
    return out;
}

static void set_f16(uint8_t *p, float v) {   // little-endian, as on disk
    __fp16 h = (__fp16)v;
    uint16_t bits;
    memcpy(&bits, &h, sizeof bits);
    memcpy(p, &bits, sizeof bits);
}

static int failures = 0;

static void check(const char *what, float a, float b) {
    if (memcmp(&a, &b, sizeof a) != 0) {
        printf("  MISMATCH %-28s simd=%.9g scalar=%.9g\n", what, a, b);
        ++failures;
    }
}

int main() {
    const size_t shapes[] = {1536, 2048, 4096, 8960, 14336};
    printf("sanitizer driver: Q4_K + Q6_K, %zu shapes x 3 seeds\n",
           sizeof(shapes) / sizeof(*shapes));

    for (size_t si = 0; si < sizeof(shapes) / sizeof(*shapes); ++si) {
        size_t k = shapes[si], nb = k / SAPIENT_QK_K;
        for (uint64_t seed : {1ULL, 0xC0FFEE01ULL, 0xDEADBEEFULL}) {
            // ── Q4_K ──
            {
                size_t rb = nb * SAPIENT_Q4_K_BLOCK_BYTES;
                auto rows = lcg_bytes(seed, 4 * rb);
                for (size_t r = 0; r < 4; ++r)
                    for (size_t b = 0; b < nb; ++b) {
                        uint8_t *blk = rows.data() + r * rb + b * SAPIENT_Q4_K_BLOCK_BYTES;
                        set_f16(blk, 0.04f);
                        set_f16(blk + 2, 0.02f);
                    }
                auto packed = repack_r4(rows, k, SAPIENT_Q4_K_BLOCK_BYTES);
                auto x = lcg_activations(seed ^ 0x9E3779B9ULL, k);
                std::vector<int8_t> q; std::vector<float> sc; std::vector<int32_t> su;
                quantize_q8k(x, q, sc, su);

                auto got = sapient_cpp_dot_q4_k_4rows_r4_q8k(
                    packed.data(), packed.size(), q.data(), sc.data(), sc.size(), su.data());
                for (size_t r = 0; r < 4; ++r) {
                    float want = sapient_cpp_dot_q4_k_row_q8k_scalar(
                        rows.data() + r * rb, rb, q.data(), sc.data(), su.data());
                    check("q4_k", got.v[r], want);
                }
            }
            // ── Q6_K ──
            {
                size_t rb = nb * SAPIENT_Q6_K_BLOCK_BYTES;
                auto rows = lcg_bytes(seed ^ 0xABCDULL, 4 * rb);
                for (size_t r = 0; r < 4; ++r)
                    for (size_t b = 0; b < nb; ++b)
                        set_f16(rows.data() + r * rb + b * SAPIENT_Q6_K_BLOCK_BYTES + 208, 0.04f);
                auto packed = repack_r4(rows, k, SAPIENT_Q6_K_BLOCK_BYTES);
                auto x = lcg_activations(seed ^ 0x51ED2701ULL, k);
                std::vector<int8_t> q; std::vector<float> sc; std::vector<int32_t> su;
                quantize_q8k(x, q, sc, su);

                auto got = sapient_cpp_dot_q6_k_4rows_r4_q8k(
                    packed.data(), packed.size(), q.data(), sc.data(), sc.size());
                for (size_t r = 0; r < 4; ++r) {
                    float want = sapient_cpp_dot_q6_k_row_q8k_scalar(
                        rows.data() + r * rb, rb, q.data(), sc.data());
                    check("q6_k", got.v[r], want);
                }
            }
        }
    }

    // fp16 conversion, exhaustive — cheap, and it touches every code path.
    for (uint32_t b = 0; b <= 0xFFFF; ++b) (void)sapient_cpp_f16_to_f32((uint16_t)b);

    printf(failures ? "FAILED: %d mismatches\n" : "OK: all kernels agree with their oracles\n",
           failures);
    return failures ? 1 : 0;
}
