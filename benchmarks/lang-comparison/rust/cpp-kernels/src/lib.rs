// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! Safe Rust bindings to the C++ kernel ports used by the Rust-vs-C++ study.
//!
//! The wrappers take the **same argument types** as their Rust counterparts in
//! `sapient_backends_cpu::kernels::quant`, so a benchmark or the engine can
//! swap one for the other by changing only the path. Each wrapper is
//! `#[inline(always)]` so the only call that survives is the FFI call itself —
//! the wrapper must not add a second frame that the Rust arm doesn't pay.

/// AArch64 AAPCS returns a 4-float homogeneous aggregate in `v0..v3`, exactly
/// like Rust's `[f32; 4]`, so this crosses the boundary in registers.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct F32x4 {
    v: [f32; 4],
}

extern "C" {
    fn sapient_cpp_dot_q4_k_4rows_r4_q8k(
        packed: *const u8,
        packed_len: usize,
        x_i8: *const i8,
        x_scales: *const f32,
        x_scales_len: usize,
        x_sums: *const i32,
    ) -> F32x4;

    fn sapient_cpp_dot_q6_k_4rows_r4_q8k(
        packed: *const u8,
        packed_len: usize,
        x_i8: *const i8,
        x_scales: *const f32,
        x_scales_len: usize,
    ) -> F32x4;

    fn sapient_cpp_dot_q4_k_row_q8k_scalar(
        row: *const u8,
        row_len: usize,
        x_i8: *const i8,
        x_scales: *const f32,
        x_sums: *const i32,
    ) -> f32;

    fn sapient_cpp_dot_q6_k_row_q8k_scalar(
        row: *const u8,
        row_len: usize,
        x_i8: *const i8,
        x_scales: *const f32,
    ) -> f32;

    fn sapient_cpp_f16_to_f32(bits: u16) -> f32;
}

/// The exact compiler and flag string the C++ was built with — recorded at
/// build time so the report quotes the real build line, not a remembered one.
pub const CPP_COMPILER: &str = env!("SAPIENT_CPP_COMPILER");
pub const CPP_FLAGS: &str = env!("SAPIENT_CPP_FLAGS");

/// C++ port of `dot_q4_k_4rows_r4_q8k_neon` (quant.rs:1410).
///
/// # Safety
/// Same contract as the Rust original: `packed` must be a whole Q4_K_R4
/// row-group covered by the Q8_K activations, and the CPU must have
/// `FEAT_DotProd`.
#[inline(always)]
pub unsafe fn dot_q4_k_4rows_r4_q8k(
    packed: &[u8],
    x_i8: &[i8],
    x_scales: &[f32],
    x_sums: &[i32],
) -> [f32; 4] {
    sapient_cpp_dot_q4_k_4rows_r4_q8k(
        packed.as_ptr(),
        packed.len(),
        x_i8.as_ptr(),
        x_scales.as_ptr(),
        x_scales.len(),
        x_sums.as_ptr(),
    )
    .v
}

/// C++ port of `dot_q6_k_4rows_r4_q8k_neon` (quant.rs:2119).
///
/// # Safety
/// As above; `packed` must be a whole Q6_K_R4 row-group.
#[inline(always)]
pub unsafe fn dot_q6_k_4rows_r4_q8k(packed: &[u8], x_i8: &[i8], x_scales: &[f32]) -> [f32; 4] {
    sapient_cpp_dot_q6_k_4rows_r4_q8k(
        packed.as_ptr(),
        packed.len(),
        x_i8.as_ptr(),
        x_scales.as_ptr(),
        x_scales.len(),
    )
    .v
}

/// C++ port of the scalar oracle `dot_q4_k_row_q8k_scalar` (quant.rs:394).
pub fn dot_q4_k_row_q8k_scalar(row: &[u8], x_i8: &[i8], x_scales: &[f32], x_sums: &[i32]) -> f32 {
    unsafe {
        sapient_cpp_dot_q4_k_row_q8k_scalar(
            row.as_ptr(),
            row.len(),
            x_i8.as_ptr(),
            x_scales.as_ptr(),
            x_sums.as_ptr(),
        )
    }
}

/// C++ port of the scalar oracle `dot_q6_k_row_q8k_scalar` (quant.rs:1878).
pub fn dot_q6_k_row_q8k_scalar(row: &[u8], x_i8: &[i8], x_scales: &[f32]) -> f32 {
    unsafe {
        sapient_cpp_dot_q6_k_row_q8k_scalar(
            row.as_ptr(),
            row.len(),
            x_i8.as_ptr(),
            x_scales.as_ptr(),
        )
    }
}

/// C++ fp16→f32, for the gate that proves it matches the `half` crate bit-exactly.
pub fn f16_to_f32(bits: u16) -> f32 {
    unsafe { sapient_cpp_f16_to_f32(bits) }
}
