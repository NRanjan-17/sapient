// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! Element-wise CPU kernels — arithmetic, activations, and mathematical ops.
//!
//! All kernels operate on F32 tensors. Binary ops support same-shape operands
//! only (broadcasting handled by the dispatch layer after shape inference).

use sapient_core::error::{Result, SapientError};
use sapient_core::{DType, Tensor};

// ── Helper ────────────────────────────────────────────────────────────────────

/// Below this many elements the rayon fork/join costs more than the map.
const UNARY_PAR_MIN: usize = 1 << 16;

/// Apply a unary f32 function element-wise.
///
/// Large tensors (a vision tower's `[1024, 3072]` MLP activation is 3M
/// elements per layer) map in parallel — element-wise, so bit-identical to the
/// serial path.
fn unary_f32<F: Fn(f32) -> f32 + Sync>(x: &Tensor, f: F) -> Result<Tensor> {
    if x.dtype() != DType::F32 {
        return Err(SapientError::TypeMismatch {
            expected: "f32".into(),
            got: x.dtype().to_string(),
        });
    }
    let src = x.to_f32_cow();
    let data: Vec<f32> = if src.len() >= UNARY_PAR_MIN {
        use rayon::prelude::*;
        src.par_iter().map(|&v| f(v)).collect()
    } else {
        src.iter().map(|&v| f(v)).collect()
    };
    Tensor::from_f32(&data, x.shape().clone())
}

/// Apply a binary f32 function element-wise (same shape only).
fn binary_f32<F: Fn(f32, f32) -> f32>(a: &Tensor, b: &Tensor, f: F) -> Result<Tensor> {
    // Handle scalar broadcast (numel == 1).
    let a_cow = a.to_f32_cow();
    let a_data = a_cow.as_ref();
    let b_cow = b.to_f32_cow();
    let b_data = b_cow.as_ref();

    let (out, shape) = if a_data.len() == b_data.len() {
        let out: Vec<f32> = a_data
            .iter()
            .zip(b_data.iter())
            .map(|(&x, &y)| f(x, y))
            .collect();
        (out, a.shape().clone())
    } else if b_data.len() == 1 {
        let scalar = b_data[0];
        let out: Vec<f32> = a_data.iter().map(|&x| f(x, scalar)).collect();
        (out, a.shape().clone())
    } else if a_data.len() == 1 {
        let scalar = a_data[0];
        let out: Vec<f32> = b_data.iter().map(|&y| f(scalar, y)).collect();
        (out, b.shape().clone())
    } else {
        return Err(SapientError::ShapeMismatch {
            expected: a.shape().dims().to_vec(),
            got: b.shape().dims().to_vec(),
        });
    };

    Tensor::from_f32(&out, shape)
}

// ── Arithmetic ────────────────────────────────────────────────────────────────

pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    binary_f32(a, b, |x, y| x + y)
}
pub fn sub(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    binary_f32(a, b, |x, y| x - y)
}
pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    binary_f32(a, b, |x, y| x * y)
}
pub fn div(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    binary_f32(a, b, |x, y| x / y)
}
pub fn pow(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    binary_f32(a, b, |x, y| x.powf(y))
}

pub fn neg(x: &Tensor) -> Result<Tensor> {
    unary_f32(x, |v| -v)
}
pub fn abs(x: &Tensor) -> Result<Tensor> {
    unary_f32(x, |v| v.abs())
}
pub fn sqrt(x: &Tensor) -> Result<Tensor> {
    unary_f32(x, |v| v.sqrt())
}
pub fn exp(x: &Tensor) -> Result<Tensor> {
    unary_f32(x, |v| v.exp())
}
pub fn log(x: &Tensor) -> Result<Tensor> {
    unary_f32(x, |v| v.ln())
}
pub fn erf(x: &Tensor) -> Result<Tensor> {
    unary_f32(x, erf_approx)
}
pub fn floor(x: &Tensor) -> Result<Tensor> {
    unary_f32(x, |v| v.floor())
}
pub fn ceil(x: &Tensor) -> Result<Tensor> {
    unary_f32(x, |v| v.ceil())
}
pub fn round(x: &Tensor) -> Result<Tensor> {
    unary_f32(x, |v| v.round())
}

// ── Activations ───────────────────────────────────────────────────────────────

pub fn relu(x: &Tensor) -> Result<Tensor> {
    unary_f32(x, |v| v.max(0.0))
}

pub fn sigmoid(x: &Tensor) -> Result<Tensor> {
    unary_f32(x, |v| 1.0 / (1.0 + (-v).exp()))
}

pub fn tanh_act(x: &Tensor) -> Result<Tensor> {
    unary_f32(x, |v| v.tanh())
}

/// GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
pub fn gelu(x: &Tensor) -> Result<Tensor> {
    const SQRT_2_OVER_PI: f32 = 0.797_884_56;
    const COEF: f32 = 0.044_715;
    unary_f32(x, |v| {
        let inner = SQRT_2_OVER_PI * (v + COEF * v * v * v);
        0.5 * v * (1.0 + inner.tanh())
    })
}

/// Polynomial `exp` for one value: range reduction `x = n·ln2 + r`,
/// `|r| ≤ ln2/2`, a degree-6 polynomial in `r` (Cephes `expf` coefficients),
/// then `· 2ⁿ` by exponent arithmetic. Branch-free straight-line f32 code, so a
/// loop over it auto-vectorizes (NEON / SSE / AVX) — the libm `exp` it replaces
/// is a scalar call. Relative error ≤ ~2e-7; the input is clamped to
/// `[-87, 88]`, so very negative inputs return ~1.6e-38 instead of 0.
///
/// NOT bit-identical to `f32::exp`: use it only on paths whose accuracy is
/// gated numerically (the SmolVLA vision path), not where bit-stable output
/// matters (`sapient see` decodes greedily and a near-tie can flip).
#[inline(always)]
pub fn exp_approx(x: f32) -> f32 {
    const LOG2E: f32 = std::f32::consts::LOG2_E;
    const LN2_HI: f32 = 0.693_359_4;
    const LN2_LO: f32 = -2.121_944_4e-4;
    // Adding and subtracting 1.5·2²³ rounds to the nearest integer.
    const ROUND: f32 = 12_582_912.0;
    let x = x.clamp(-87.0, 88.0);
    let n = (x * LOG2E + ROUND) - ROUND;
    let r = (x - n * LN2_HI) - n * LN2_LO;
    let mut p = 1.987_569_1e-4_f32;
    p = p * r + 1.398_199_9e-3;
    p = p * r + 8.333_452e-3;
    p = p * r + 4.166_579_6e-2;
    p = p * r + 1.666_666_5e-1;
    p = p * r + 0.5;
    let e = p * r * r + r + 1.0;
    // 2ⁿ: n is an integer in [-126, 127].
    let pow2 = f32::from_bits(((n as i32 + 127) as u32) << 23);
    e * pow2
}

/// Four [`exp_approx`] at once. The same operations in the same order with
/// separate multiplies and adds (no FMA) → bit-identical to the scalar function.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn exp_approx_neon(x: std::arch::aarch64::float32x4_t) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    let f = |v: f32| vdupq_n_f32(v);
    let x = vminq_f32(vmaxq_f32(x, f(-87.0)), f(88.0));
    let n = vsubq_f32(
        vaddq_f32(vmulq_f32(x, f(std::f32::consts::LOG2_E)), f(12_582_912.0)),
        f(12_582_912.0),
    );
    let r = vsubq_f32(
        vsubq_f32(x, vmulq_f32(n, f(0.693_359_4))),
        vmulq_f32(n, f(-2.121_944_4e-4)),
    );
    let mut p = f(1.987_569_1e-4);
    p = vaddq_f32(vmulq_f32(p, r), f(1.398_199_9e-3));
    p = vaddq_f32(vmulq_f32(p, r), f(8.333_452e-3));
    p = vaddq_f32(vmulq_f32(p, r), f(4.166_579_6e-2));
    p = vaddq_f32(vmulq_f32(p, r), f(1.666_666_5e-1));
    p = vaddq_f32(vmulq_f32(p, r), f(0.5));
    let e = vaddq_f32(vaddq_f32(vmulq_f32(vmulq_f32(p, r), r), r), f(1.0));
    let pow2 = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(
        vcvtq_s32_f32(n),
        vdupq_n_s32(127),
    )));
    vmulq_f32(e, pow2)
}

/// In-place [`exp_approx`] over a slice — NEON on aarch64 (LLVM does not
/// auto-vectorize the scalar form), element-wise elsewhere. Same values either way.
pub fn exp_approx_slice(xs: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let mut chunks = xs.chunks_exact_mut(4);
        for c in &mut chunks {
            // SAFETY: NEON is baseline on aarch64; `c` holds exactly 4 f32.
            unsafe { vst1q_f32(c.as_mut_ptr(), exp_approx_neon(vld1q_f32(c.as_ptr()))) };
        }
        for v in chunks.into_remainder() {
            *v = exp_approx(*v);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    for v in xs.iter_mut() {
        *v = exp_approx(*v);
    }
}

/// In-place tanh-approximation GELU over a slice using [`exp_approx`]
/// (`tanh(y) = 1 − 2 / (e^{2y} + 1)`).
fn gelu_fast_slice(xs: &mut [f32]) {
    const SQRT_2_OVER_PI: f32 = 0.797_884_56;
    const COEF: f32 = 0.044_715;
    let scalar = |v: &mut f32| {
        let x = *v;
        let inner = SQRT_2_OVER_PI * (x + COEF * x * x * x);
        let t = 1.0 - 2.0 / (exp_approx(2.0 * inner) + 1.0);
        *v = 0.5 * x * (1.0 + t);
    };
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let mut chunks = xs.chunks_exact_mut(4);
        for c in &mut chunks {
            // SAFETY: NEON is baseline on aarch64; `c` holds exactly 4 f32.
            // Mirrors `scalar` operation for operation.
            unsafe {
                let x = vld1q_f32(c.as_ptr());
                let x3 = vmulq_f32(vmulq_f32(vmulq_n_f32(x, COEF), x), x);
                let inner = vmulq_n_f32(vaddq_f32(x, x3), SQRT_2_OVER_PI);
                let e = exp_approx_neon(vmulq_n_f32(inner, 2.0));
                let one = vdupq_n_f32(1.0);
                let t = vsubq_f32(one, vdivq_f32(vdupq_n_f32(2.0), vaddq_f32(e, one)));
                let y = vmulq_f32(vmulq_n_f32(x, 0.5), vaddq_f32(one, t));
                vst1q_f32(c.as_mut_ptr(), y);
            }
        }
        chunks.into_remainder().iter_mut().for_each(scalar);
    }
    #[cfg(not(target_arch = "aarch64"))]
    xs.iter_mut().for_each(scalar);
}

/// [`gelu`] with `tanh` computed through [`exp_approx`]
/// (`tanh(y) = 1 − 2 / (e^{2y} + 1)`): the same tanh-approximation GELU,
/// vectorized, ~1e-6 absolute from [`gelu`]. Same caveat as `exp_approx`.
pub fn gelu_fast(x: &Tensor) -> Result<Tensor> {
    if x.dtype() != DType::F32 {
        return Err(SapientError::TypeMismatch {
            expected: "f32".into(),
            got: x.dtype().to_string(),
        });
    }
    let mut data = x.to_f32_cow().into_owned();
    if data.len() >= UNARY_PAR_MIN {
        use rayon::prelude::*;
        data.par_chunks_mut(4096).for_each(gelu_fast_slice);
    } else {
        gelu_fast_slice(&mut data);
    }
    Tensor::from_f32_vec(data, x.shape().clone())
}

/// Exact (erf-based) GELU: `0.5 * x * (1 + erf(x / √2))`.
///
/// This is the variant used by HuggingFace/OpenAI Whisper (`activation_function
/// = "gelu"`), as distinct from the tanh approximation in [`gelu`]. The two
/// differ by < 1e-3 per element but the error compounds across a Whisper
/// encoder/decoder stack, so the audio path uses this exact form.
pub fn gelu_erf(x: &Tensor) -> Result<Tensor> {
    const INV_SQRT_2: f32 = std::f32::consts::FRAC_1_SQRT_2;
    unary_f32(x, |v| 0.5 * v * (1.0 + erf_approx(v * INV_SQRT_2)))
}

/// SiLU / Swish: x * sigmoid(x)
pub fn silu(x: &Tensor) -> Result<Tensor> {
    unary_f32(x, |v| v / (1.0 + (-v).exp()))
}

/// Hard Swish: x * relu6(x + 3) / 6
pub fn hard_swish(x: &Tensor) -> Result<Tensor> {
    unary_f32(x, |v| v * (v + 3.0).clamp(0.0, 6.0) / 6.0)
}

pub fn leaky_relu(x: &Tensor, alpha: f32) -> Result<Tensor> {
    unary_f32(x, |v| if v >= 0.0 { v } else { alpha * v })
}

pub fn clip(x: &Tensor, min: Option<f32>, max: Option<f32>) -> Result<Tensor> {
    unary_f32(x, |v| {
        let v = min.map_or(v, |lo| v.max(lo));
        max.map_or(v, |hi| v.min(hi))
    })
}

// ── Erf approximation (Abramowitz & Stegun) ───────────────────────────────────

fn erf_approx(x: f32) -> f32 {
    let sign = x.signum();
    let x = x.abs();
    // Rational approximation — max error ~1.5e-7.
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (0.254_829_59
            + (-0.284_496_74 + (1.421_413_74 + (-1.453_152_03 + 1.061_405_43 * t) * t) * t) * t)
            * t
            * (-x * x).exp();
    sign * y
}

#[cfg(test)]
mod tests {
    #[test]
    fn exp_approx_tracks_exp() {
        let mut worst = 0.0f32;
        let mut x = -87.0f32;
        while x <= 88.0 {
            let (a, e) = (super::exp_approx(x), x.exp());
            worst = worst.max(((a - e) / e).abs());
            x += 0.0137;
        }
        assert!(worst < 4e-7, "max relative error {worst}");
        assert_eq!(super::exp_approx(0.0), 1.0);
        // The slice form (NEON on aarch64) gives the same bits as the scalar.
        let src: Vec<f32> = (0..1003).map(|i| (i as f32 - 700.0) * 0.173).collect();
        let mut v = src.clone();
        super::exp_approx_slice(&mut v);
        for (x, got) in src.iter().zip(&v) {
            assert_eq!(got.to_bits(), super::exp_approx(*x).to_bits(), "x = {x}");
        }
        // Clamped tails stay finite and positive.
        assert!(super::exp_approx(-1000.0) > 0.0 && super::exp_approx(-1000.0) < 1e-37);
        assert!(super::exp_approx(1000.0).is_finite());
    }

    #[test]
    fn gelu_fast_tracks_gelu() {
        let xs: Vec<f32> = (0..4001).map(|i| (i as f32 - 2000.0) * 0.01).collect();
        let t = sapient_core::Tensor::from_f32(&xs, vec![xs.len()]).unwrap();
        let (a, b) = (super::gelu_fast(&t).unwrap(), super::gelu(&t).unwrap());
        for ((x, a), b) in xs.iter().zip(a.as_f32_slice()).zip(b.as_f32_slice()) {
            assert!((a - b).abs() < 5e-6, "x={x}: {a} vs {b}");
        }
    }

    use super::*;

    fn t(data: &[f32]) -> Tensor {
        Tensor::from_f32(data, vec![data.len()]).unwrap()
    }

    #[test]
    fn test_add() {
        assert!(
            (add(&t(&[1.0, 2.0]), &t(&[3.0, 4.0]))
                .unwrap()
                .as_f32_slice()[0]
                - 4.0)
                .abs()
                < 1e-6
        );
    }
    #[test]
    fn test_relu() {
        let r = relu(&t(&[-1.0, 0.0, 1.0])).unwrap();
        let d = r.as_f32_slice();
        assert_eq!(d, &[0.0, 0.0, 1.0]);
    }
    #[test]
    fn test_sigmoid() {
        let v = sigmoid(&t(&[0.0])).unwrap().as_f32_slice()[0];
        assert!((v - 0.5).abs() < 1e-6);
    }
    #[test]
    fn test_gelu() {
        let v = gelu(&t(&[0.0])).unwrap().as_f32_slice()[0];
        assert!(v.abs() < 1e-5);
    }
    #[test]
    fn test_erf() {
        let v = erf_approx(0.0);
        assert!(v.abs() < 1e-6, "erf(0) should be ~0, got {v}");
    }
    #[test]
    fn test_gelu_erf() {
        // Exact GELU: g(0)=0, g(1)=0.8413447, g(-1)=-0.1586553.
        let out = gelu_erf(&t(&[0.0, 1.0, -1.0])).unwrap();
        let v = out.as_f32_slice();
        assert!(v[0].abs() < 1e-6);
        assert!((v[1] - 0.841_344_7).abs() < 1e-4, "g(1)={}", v[1]);
        assert!((v[2] - (-0.158_655_3)).abs() < 1e-4, "g(-1)={}", v[2]);
    }
    #[test]
    fn test_scalar_broadcast() {
        let a = t(&[1.0, 2.0, 3.0]);
        let b = t(&[2.0]);
        let r = mul(&a, &b).unwrap();
        assert_eq!(r.as_f32_slice(), &[2.0, 4.0, 6.0]);
    }
}
