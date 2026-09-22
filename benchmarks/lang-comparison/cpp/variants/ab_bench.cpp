// Bit-identity gate + timing for the original vs optimized Q4_K kernel.
#include "sapient_kernels.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <cstdint>

extern "C" sapient_f32x4 sapient_cpp_dot_q4_k_4rows_r4_q8k(
    const uint8_t*, size_t, const int8_t*, const float*, size_t, const int32_t*);
extern "C" sapient_f32x4 sapient_cpp_dot_q4_k_4rows_r4_q8k_opt(
    const uint8_t*, size_t, const int8_t*, const float*, size_t, const int32_t*);
extern "C" sapient_f32x4 sapient_cpp_dot_q4_k_4rows_r4_q8k_opt2(
    const uint8_t*, size_t, const int8_t*, const float*, size_t, const int32_t*);

static inline void escape(void* p) { asm volatile("" : : "g"(p) : "memory"); }

static uint64_t s = 0x243F6A8885A308D3ULL;
static uint32_t rnd() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (uint32_t)(s >> 32); }

int main(int argc, char** argv) {
    // Shapes matching a real qwen2.5-1.5b row: k=8960 -> 35 super-blocks.
    const size_t nb = (argc > 1) ? (size_t)atoi(argv[1]) : 35;
    const size_t iters = (argc > 2) ? (size_t)atoi(argv[2]) : 200000;

    std::vector<uint8_t> packed(nb * 4 * SAPIENT_Q4_K_BLOCK_BYTES);
    std::vector<int8_t>  x_i8(nb * SAPIENT_QK_K);
    std::vector<float>   x_scales(nb);
    std::vector<int32_t> x_sums(nb * (SAPIENT_QK_K / SAPIENT_QK));

    for (auto& b : packed)   b = (uint8_t)(rnd() & 0xFF);
    // Bytes 0..3 of each 144-byte block are the f16 (d, dmin). Random bytes
    // there decode to NaN/Inf and poison the whole accumulation, so write
    // real half-precision values instead.
    for (size_t blk = 0; blk < nb * 4; ++blk) {
        uint8_t* p = packed.data() + blk * SAPIENT_Q4_K_BLOCK_BYTES;
        __fp16 d    = (__fp16)(0.001f + (float)(rnd() % 500) / 100000.0f);
        __fp16 dmin = (__fp16)(0.0005f + (float)(rnd() % 300) / 200000.0f);
        memcpy(p,     &d,    2);
        memcpy(p + 2, &dmin, 2);
    }
    for (auto& v : x_i8)     v = (int8_t)(rnd() & 0xFF);
    for (auto& v : x_scales) v = (float)((rnd() % 1000)) / 4000.0f;
    for (size_t i = 0; i < x_sums.size(); ++i) {
        int32_t acc = 0;
        for (size_t l = 0; l < SAPIENT_QK; ++l) acc += x_i8[i * SAPIENT_QK + l];
        x_sums[i] = acc;
    }

    // ── correctness: must be BIT-identical, not merely close ──
    sapient_f32x4 a = sapient_cpp_dot_q4_k_4rows_r4_q8k(
        packed.data(), packed.size(), x_i8.data(), x_scales.data(), x_scales.size(), x_sums.data());
    sapient_f32x4 b = sapient_cpp_dot_q4_k_4rows_r4_q8k_opt2(
        packed.data(), packed.size(), x_i8.data(), x_scales.data(), x_scales.size(), x_sums.data());

    int bitfail = 0;
    for (int r = 0; r < 4; ++r) {
        uint32_t ba, bb; memcpy(&ba, &a.v[r], 4); memcpy(&bb, &b.v[r], 4);
        if (ba != bb) { bitfail++; printf("  MISMATCH row %d: %.9g (0x%08x) vs %.9g (0x%08x)\n", r, a.v[r], ba, b.v[r], bb); }
    }
    printf("bit-identity: %s   [%.6g %.6g %.6g %.6g]\n",
           bitfail ? "FAIL" : "PASS", a.v[0], a.v[1], a.v[2], a.v[3]);
    if (bitfail) return 1;

    // ── timing: interleaved A/B, best-of-5, to blunt frequency drift ──
    double best_o = 1e30, best_p = 1e30;
    volatile float sink = 0;
    for (int rep = 0; rep < 5; ++rep) {
        auto t0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < iters; ++i) {
            escape(packed.data()); escape(x_i8.data());
            sapient_f32x4 r = sapient_cpp_dot_q4_k_4rows_r4_q8k(
                packed.data(), packed.size(), x_i8.data(), x_scales.data(), x_scales.size(), x_sums.data());
            sink += r.v[0]; escape((void*)&sink);
        }
        auto t1 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < iters; ++i) {
            escape(packed.data()); escape(x_i8.data());
            sapient_f32x4 r = sapient_cpp_dot_q4_k_4rows_r4_q8k_opt2(
                packed.data(), packed.size(), x_i8.data(), x_scales.data(), x_scales.size(), x_sums.data());
            sink += r.v[0]; escape((void*)&sink);
        }
        auto t2 = std::chrono::steady_clock::now();
        double o = std::chrono::duration<double>(t1 - t0).count();
        double p = std::chrono::duration<double>(t2 - t1).count();
        if (o < best_o) best_o = o;
        if (p < best_p) best_p = p;
        printf("  rep %d: orig %.4fs  opt %.4fs  (%.3fx)\n", rep, o, p, o / p);
    }
    printf("\nBEST: orig %.4fs  opt %.4fs  speedup %.4fx\n", best_o, best_p, best_o / best_p);
    printf("(nb=%zu super-blocks, %zu iters, sink=%g)\n", nb, iters, (float)sink);
    return 0;
}
