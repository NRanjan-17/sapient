// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
//
// C++ transliterations of SAPIENT's hot CPU kernels, for the Rust-vs-C++
// language study (docs/RUST_VS_CPP_REPORT.md).
//
// CONTRACT: these are line-for-line ports of the Rust originals in
// crates/sapient-backends/cpu/src/kernels/quant.rs — SAME algorithm, SAME data
// layout, SAME intrinsics, SAME order of floating-point operations. The only
// deliberate difference is that C++ calls the `vdotq_s32` / `vmmlaq_s32`
// intrinsics where Rust must emit inline `asm!` (`vmmlaq_s32` is still unstable
// in Rust: E0658 / rust#117223). That difference IS the experiment.
//
// Any change here must keep the bit-identity gate green
// (benchmarks/lang-comparison/rust/harness/tests/bit_identity.rs).

#ifndef SAPIENT_KERNELS_H
#define SAPIENT_KERNELS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Block geometry — mirrors quant.rs:32-36, :588-591.
#define SAPIENT_QK                 32
#define SAPIENT_QK_K              256
#define SAPIENT_Q4_K_BLOCK_BYTES  144
#define SAPIENT_Q6_K_BLOCK_BYTES  210
#define SAPIENT_Q8_0_BLOCK_BYTES   34

// A homogeneous float aggregate: returned in v0..v3 under AArch64 AAPCS,
// exactly like Rust's `[f32; 4]`, so the FFI boundary adds no store/reload.
typedef struct {
    float v[4];
} sapient_f32x4;

// Two activation rows x four weight rows — mirrors Rust `[[f32; 2]; 4]`.
typedef struct {
    float v[4][2];
} sapient_f32x4x2;

// ── Q4_K ─────────────────────────────────────────────────────────────────────

// Port of `dot_q4_k_4rows_r4_q8k_neon` (quant.rs:1410-1472).
// 64.1% of measured decode time. Requires FEAT_DotProd.
sapient_f32x4 sapient_cpp_dot_q4_k_4rows_r4_q8k(
    const uint8_t *packed, size_t packed_len,
    const int8_t  *x_i8,
    const float   *x_scales, size_t x_scales_len,
    const int32_t *x_sums);

// Port of `dot_q4_k_4rows_r4_x2_q8k_smmla` (quant.rs:1487-1609).
// The prefill kernel. Requires FEAT_I8MM. This is the arm where Rust cannot
// use the intrinsic on stable and must emit inline asm.
sapient_f32x4x2 sapient_cpp_dot_q4_k_4rows_r4_x2_q8k_smmla(
    const uint8_t *packed, size_t packed_len,
    const int8_t  *x0_i8, const float *x0_scales, const int32_t *x0_sums,
    const int8_t  *x1_i8, const float *x1_scales, const int32_t *x1_sums,
    size_t x_scales_len);

// ── Q6_K ─────────────────────────────────────────────────────────────────────

// Port of `dot_q6_k_4rows_r4_q8k_neon` (quant.rs:2119-2196).
// 32.4% of measured decode time. Requires FEAT_DotProd.
sapient_f32x4 sapient_cpp_dot_q6_k_4rows_r4_q8k(
    const uint8_t *packed, size_t packed_len,
    const int8_t  *x_i8,
    const float   *x_scales, size_t x_scales_len);

// ── Scalar oracles (for the pure-C++ sanitizer driver) ───────────────────────

// Port of `dot_q4_k_row_q8k_scalar` (quant.rs:394-429).
float sapient_cpp_dot_q4_k_row_q8k_scalar(
    const uint8_t *row, size_t row_len,
    const int8_t *x_i8, const float *x_scales, const int32_t *x_sums);

// Port of `dot_q6_k_row_q8k_scalar` (quant.rs:1878-1921).
float sapient_cpp_dot_q6_k_row_q8k_scalar(
    const uint8_t *row, size_t row_len,
    const int8_t *x_i8, const float *x_scales);

// ── fp16 → f32, proven bit-equal to the `half` crate's `to_f32` ──────────────
float sapient_cpp_f16_to_f32(uint16_t bits);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // SAPIENT_KERNELS_H
