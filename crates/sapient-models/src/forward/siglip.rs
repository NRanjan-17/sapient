// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! SigLIP vision tower + Idefics3 connector — the vision half of SmolVLM.
//!
//! `SiglipVision::encode` turns preprocessed pixels `[3, S, S]` into visual
//! token embeddings `[n_vis, text_hidden]` ready to splice into the text
//! model's embedding sequence at the `<image>` token positions:
//!
//! 1. patch embedding: `conv2d` stride=patch (existing kernel) + learned
//!    position embeddings — `[n_patches, vision_hidden]`
//! 2. N pre-LN transformer blocks (LayerNorm+bias, full non-causal attention
//!    via an explicit all-zeros mask — same invariant as the Whisper encoder:
//!    `mask=None` means CAUSAL in the CPU kernel — and a `gelu_pytorch_tanh`
//!    MLP, which is the existing `gelu` kernel, NOT `gelu_erf`)
//! 3. `post_layernorm`
//! 4. Idefics3 pixel shuffle (scale s): `[h·w, c]` → `[h/s · w/s, c·s²]` with
//!    the exact transformers ordering `out[hj][wj] = concat_{dh, dw} in[hj·s+dh][wj·s+dw]`
//!    (dh outer, dw inner, channels innermost) — getting this ordering wrong
//!    is this model's token-salad class of bug
//! 5. modality projection: linear `c·s² → text_hidden` (no bias)
//!
//! CPU-only and f32 (the tower is ~93 M params; one 512² image is 1024
//! patches × 12 layers — comfortably fast on the parallel conv/GEMM kernels).

use std::collections::HashMap;

use anyhow::{anyhow, Result};
use sapient_backends_cpu::kernels::elementwise::{exp_approx_slice, gelu, gelu_fast};
use sapient_core::{Shape, Tensor};

use super::backend::{LlmBackend, LlmBackendDispatch, LlmBackendKind};
use super::common::{merge_heads, split_heads};

const LN_EPS: f32 = 1e-6;

/// Vision-tower dimensions (SmolVLM-256M: 768/12L/12H, image 512, patch 16,
/// scale 4, text hidden 576).
#[derive(Debug, Clone)]
pub struct SiglipConfig {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub intermediate: usize,
    pub image_size: usize,
    pub patch: usize,
    pub scale_factor: usize,
    pub text_hidden: usize,
}

impl SiglipConfig {
    pub fn n_patches_side(&self) -> usize {
        self.image_size / self.patch
    }
    pub fn n_patches(&self) -> usize {
        self.n_patches_side() * self.n_patches_side()
    }
    /// Visual tokens after pixel shuffle.
    pub fn n_visual_tokens(&self) -> usize {
        self.n_patches() / (self.scale_factor * self.scale_factor)
    }
}

/// The loaded tower: weight map (keys as in the HF checkpoint, e.g.
/// `model.vision_model.encoder.layers.0.self_attn.q_proj.weight`) + CPU dispatch.
pub struct SiglipVision {
    cfg: SiglipConfig,
    weights: HashMap<String, Tensor>,
    backend: LlmBackendDispatch,
    head_dim: usize,
    /// Checkpoint key prefix up to (excluding) `.embeddings…` — SmolVLM uses
    /// `model.vision_model`, Gemma3/MedGemma use `vision_tower.vision_model`.
    prefix: String,
    /// Vectorized polynomial `exp` in the attention softmax and the GELU
    /// instead of libm calls — see [`with_fast_math`](Self::with_fast_math).
    fast_math: bool,
    /// int8 attention (`dense_attention_int8`) — see
    /// [`with_int8_attention`](Self::with_int8_attention).
    int8_attention: bool,
}

impl SiglipVision {
    pub fn new(cfg: SiglipConfig, weights: HashMap<String, Tensor>) -> Result<Self> {
        Self::with_prefix(cfg, weights, "model.vision_model")
    }

    pub fn with_prefix(
        cfg: SiglipConfig,
        weights: HashMap<String, Tensor>,
        prefix: &str,
    ) -> Result<Self> {
        let head_dim = cfg.hidden / cfg.heads;
        let backend = LlmBackendDispatch::from_kind(LlmBackendKind::Cpu)
            .map_err(|e| anyhow!("vision backend: {e}"))?;
        Ok(Self {
            cfg,
            weights,
            backend,
            head_dim,
            prefix: prefix.to_string(),
            fast_math: false,
            int8_attention: false,
        })
    }

    /// Run the tower's attention in int8 (Q·Kᵀ and P·V on the `sdot` GEMM tile,
    /// per-32 scales, K mean-centred first) instead of f32 SGEMM. Approximate —
    /// for paths gated on numeric error only (SmolVLA `fast`). Needs aarch64
    /// `dotprod`, `head_dim % 32 == 0` and a patch count divisible by 32;
    /// otherwise the f32 path runs.
    pub fn with_int8_attention(mut self, on: bool) -> Self {
        self.int8_attention = on;
        self
    }

    /// Use the vectorized polynomial `exp` (softmax) and GELU. ~2e-7 relative
    /// per `exp`, so the output is no longer bit-identical to the default path.
    /// Off for `sapient see` (greedy decoding can flip on a near-tie); SmolVLA
    /// turns it on with its Q8_0 vision path, where the action error is
    /// measured. On a Pi 5 these two were ~1.1 s of a 2.4 s image encode.
    pub fn with_fast_math(mut self, on: bool) -> Self {
        self.fast_math = on;
        self
    }

    pub fn config(&self) -> &SiglipConfig {
        &self.cfg
    }

    fn get(&self, name: &str) -> Result<&Tensor> {
        self.weights
            .get(name)
            .ok_or_else(|| anyhow!("vision weight missing: {name}"))
    }

    fn opt(&self, name: &str) -> Option<&Tensor> {
        self.weights.get(name)
    }

    fn layer_norm(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        let w = self.get(&format!("{prefix}.weight"))?.clone();
        let b = self.opt(&format!("{prefix}.bias")).cloned();
        self.backend
            .layer_norm(x, &w, b.as_ref(), LN_EPS)
            .map_err(|e| anyhow!("{e}"))
    }

    fn linear(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        let w = self.get(&format!("{prefix}.weight"))?.clone();
        let b = self.opt(&format!("{prefix}.bias")).cloned();
        self.backend
            .linear_3d_bias(x, &w, b.as_ref())
            .map_err(|e| anyhow!("{e}"))
    }

    fn attention(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        #[cfg(target_arch = "aarch64")]
        if self.int8_attention
            && self.head_dim % 32 == 0
            && q.shape().dims()[2] % 32 == 0
            && std::arch::is_aarch64_feature_detected!("dotprod")
        {
            let seq = q.shape().dims()[2];
            return dense_attention_int8(
                q,
                k,
                v,
                self.cfg.heads,
                self.head_dim,
                attn_tile_rows(seq),
            );
        }
        dense_full_attention(q, k, v, self.cfg.heads, self.head_dim, self.fast_math)
    }

    /// Preprocessed pixels `[3, S, S]` → raw tower features
    /// `[n_patches · hidden]` (post `post_layernorm`, before any connector).
    /// The Idefics3 connector continues in [`encode`](Self::encode); Gemma3's
    /// pool/norm/project connector consumes these directly.
    pub fn encode_features(&self, pixels: &[f32]) -> Result<Vec<f32>> {
        let s = self.cfg.image_size;
        let c = self.cfg.hidden;
        let n_patch = self.cfg.n_patches();
        if pixels.len() != 3 * s * s {
            anyhow::bail!("expected {}x{s}x{s} pixels, got {}", 3, pixels.len());
        }

        // ── 1. patch embedding: conv2d stride=patch → [1, c, side, side] ────
        let x = Tensor::from_f32(pixels, Shape::new([1, 3, s, s])).map_err(|e| anyhow!("{e}"))?;
        let pw = self.get(&format!(
            "{}.embeddings.patch_embedding.weight",
            self.prefix
        ))?;
        let pb = self.opt(&format!("{}.embeddings.patch_embedding.bias", self.prefix));
        let patches = sapient_backends_cpu::kernels::conv2d::conv2d(
            &x,
            pw,
            pb,
            [self.cfg.patch, self.cfg.patch],
            [0, 0, 0, 0],
            [self.cfg.patch, self.cfg.patch],
            [1, 1],
            1,
        )
        .map_err(|e| anyhow!("{e}"))?;
        // [1, c, side, side] → [n_patch, c] (patch-major, channels contiguous).
        let pv = patches.to_f32_vec();
        let mut h = vec![0.0f32; n_patch * c];
        for ci in 0..c {
            for p in 0..n_patch {
                h[p * c + ci] = pv[ci * n_patch + p];
            }
        }
        // + learned position embeddings [n_patch, c].
        let pos = self
            .get(&format!(
                "{}.embeddings.position_embedding.weight",
                self.prefix
            ))?
            .to_f32_vec();
        if pos.len() != h.len() {
            anyhow::bail!("position embedding {} != patches {}", pos.len(), h.len());
        }
        for (a, b) in h.iter_mut().zip(&pos) {
            *a += b;
        }
        let mut x =
            Tensor::from_f32(&h, Shape::new([1, n_patch, c])).map_err(|e| anyhow!("{e}"))?;

        // Per-stage wall-clock breakdown, printed under SAPIENT_VISION_TIMING
        // (same idea as SAPIENT_KOKORO_TIMING): [norm, qkv, attn, out_proj, fc1, gelu, fc2].
        let timing = std::env::var_os("SAPIENT_VISION_TIMING").is_some();
        let mut st = [std::time::Duration::ZERO; 7];
        let mut mark = std::time::Instant::now();
        let mut lap = |i: usize, mark: &mut std::time::Instant| {
            let now = std::time::Instant::now();
            st[i] += now - *mark;
            *mark = now;
        };

        // ── 2. transformer blocks (pre-LN) ───────────────────────────────────
        for l in 0..self.cfg.layers {
            let p = format!("{}.encoder.layers.{l}", self.prefix);
            // attn
            let normed = self.layer_norm(&x, &format!("{p}.layer_norm1"))?;
            lap(0, &mut mark);
            let q = split_heads(
                &self.linear(&normed, &format!("{p}.self_attn.q_proj"))?,
                self.cfg.heads,
                self.head_dim,
            )?;
            let k = split_heads(
                &self.linear(&normed, &format!("{p}.self_attn.k_proj"))?,
                self.cfg.heads,
                self.head_dim,
            )?;
            let v = split_heads(
                &self.linear(&normed, &format!("{p}.self_attn.v_proj"))?,
                self.cfg.heads,
                self.head_dim,
            )?;
            lap(1, &mut mark);
            let attn = self.attention(&q, &k, &v)?;
            let attn = merge_heads(&attn)?;
            lap(2, &mut mark);
            let attn = self.linear(&attn, &format!("{p}.self_attn.out_proj"))?;
            x = self.backend.add(&x, &attn).map_err(|e| anyhow!("{e}"))?;
            lap(3, &mut mark);
            // mlp
            let normed = self.layer_norm(&x, &format!("{p}.layer_norm2"))?;
            lap(0, &mut mark);
            let up = self.linear(&normed, &format!("{p}.mlp.fc1"))?;
            lap(4, &mut mark);
            // gelu_pytorch_tanh
            let up = if self.fast_math {
                gelu_fast(&up)
            } else {
                gelu(&up)
            }
            .map_err(|e| anyhow!("{e}"))?;
            lap(5, &mut mark);
            let down = self.linear(&up, &format!("{p}.mlp.fc2"))?;
            x = self.backend.add(&x, &down).map_err(|e| anyhow!("{e}"))?;
            lap(6, &mut mark);
        }
        if timing {
            let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
            eprintln!(
                "[vision] {} patches · norm {:.0} · qkv {:.0} · attn {:.0} · out_proj {:.0} · fc1 {:.0} · gelu {:.0} · fc2 {:.0} ms",
                n_patch, ms(st[0]), ms(st[1]), ms(st[2]), ms(st[3]), ms(st[4]), ms(st[5]), ms(st[6])
            );
        }
        let x = self.layer_norm(&x, &format!("{}.post_layernorm", self.prefix))?;
        Ok(x.to_f32_vec())
    }

    /// Idefics3/SmolVLM path: tower features → pixel shuffle → modality
    /// projection → `[n_visual_tokens · text_hidden]`.
    pub fn encode(&self, pixels: &[f32]) -> Result<Vec<f32>> {
        let c = self.cfg.hidden;
        let side = self.cfg.n_patches_side();
        let xv = self.encode_features(pixels)?;

        // ── 3. pixel shuffle: [side, side, c] → [side/s², c·s²] ──────────────
        let sf = self.cfg.scale_factor;
        let out_side = side / sf;
        let cs2 = c * sf * sf;
        let mut shuffled = vec![0.0f32; out_side * out_side * cs2];
        for hj in 0..out_side {
            for wj in 0..out_side {
                let dst = (hj * out_side + wj) * cs2;
                for dh in 0..sf {
                    for dw in 0..sf {
                        let src_patch = (hj * sf + dh) * side + (wj * sf + dw);
                        let d = dst + (dh * sf + dw) * c;
                        shuffled[d..d + c].copy_from_slice(&xv[src_patch * c..(src_patch + 1) * c]);
                    }
                }
            }
        }

        // ── 4. modality projection: [n_vis, c·s²] → [n_vis, text_hidden] ────
        let n_vis = out_side * out_side;
        let shuffled =
            Tensor::from_f32(&shuffled, Shape::new([1, n_vis, cs2])).map_err(|e| anyhow!("{e}"))?;
        let proj = self.linear(&shuffled, "model.connector.modality_projection.proj")?;
        Ok(proj.to_f32_vec())
    }
}

/// Query rows per attention tile: a tile's score block is `rows × seq` f32.
/// Swept on a Pi 5 (tower attention, ms): 16 KB 3542 · 64 KB 1078 · **256 KB
/// 846** · 1 MB 987 · untiled 1051 — small tiles lose to SGEMM re-packing K/V
/// on every call, large ones fall out of cache. 256 KB was also best-or-equal
/// on an M4. `SAPIENT_ATTN_TILE_KB` overrides for tuning.
fn attn_tile_rows(seq: usize) -> usize {
    static KB: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let kb = *KB.get_or_init(|| {
        std::env::var("SAPIENT_ATTN_TILE_KB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&kb| kb > 0)
            .unwrap_or(256)
    });
    (kb * 1024 / 4 / seq.max(1)).max(1)
}

/// Dense NON-CAUSAL attention for the vision tower: per head,
/// `S = softmax(Q·Kᵀ/√d)`, `O = S·V`, through blocked SGEMM.
///
/// Tiled over query rows: each tile computes its `rows × seq` score block,
/// softmaxes it and multiplies by V while it is still cache-resident, instead
/// of materialising the whole `seq × seq` matrix per head (4 MB at 1024
/// patches, 64 MB at 4096 — streamed three times on a small-cache CPU). Tiles
/// are independent, so (head, tile) pairs run in parallel; each tile is the
/// same K reduction as the untiled product → bit-identical to it. For long
/// sequences this is still far faster than the flash row-loop, which is shaped
/// for long-KV DECODE (one query row at a time).
fn dense_full_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    n_heads: usize,
    head_dim: usize,
    fast_exp: bool,
) -> Result<Tensor> {
    let seq = q.shape().dims()[2];
    dense_full_attention_tiled(q, k, v, n_heads, head_dim, attn_tile_rows(seq), fast_exp)
}

/// [`dense_full_attention`] with an explicit tile height (query rows per tile).
fn dense_full_attention_tiled(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    n_heads: usize,
    head_dim: usize,
    tile: usize,
    fast_exp: bool,
) -> Result<Tensor> {
    use rayon::prelude::*;
    use sapient_backends_cpu::kernels::matmul::sgemm_serial;
    let dims = q.shape().dims().to_vec(); // [1, h, seq, hd]
    let seq = dims[2];
    let qv = q.to_f32_cow();
    let kv = k.to_f32_cow();
    let vv = v.to_f32_cow();
    let scale = 1.0 / (head_dim as f32).sqrt();
    let tile = tile.max(1);
    let head_len = seq * head_dim;
    let mut out = vec![0.0f32; n_heads * head_len];

    out.par_chunks_mut(head_len)
        .enumerate()
        .for_each(|(h, out_h)| {
            let qh = &qv[h * head_len..(h + 1) * head_len];
            let kh = &kv[h * head_len..(h + 1) * head_len];
            let vh = &vv[h * head_len..(h + 1) * head_len];
            out_h
                .par_chunks_mut(tile * head_dim)
                .enumerate()
                .for_each(|(ti, out_t)| {
                    let r0 = ti * tile;
                    let rows = out_t.len() / head_dim;
                    let q_t = &qh[r0 * head_dim..(r0 + rows) * head_dim];
                    // S = Q_t · Kᵀ  (B[d][j] = K[j][d]).
                    let mut scores = vec![0.0f32; rows * seq];
                    sgemm_serial(rows, head_dim, seq, q_t, kh, 1, head_dim, &mut scores);
                    // Row-wise softmax.
                    // The scalar libm `exp` is the default: a polynomial exp
                    // (~2e-7 rel. error) is not bit-identical and flipped a
                    // greedy near-tie in a `sapient see` reply (2026-10-02).
                    // `fast_exp` opts in for paths gated on numeric error
                    // (SmolVLA: action error vs the reference).
                    for row in scores.chunks_exact_mut(seq) {
                        let mut mx = f32::NEG_INFINITY;
                        for x in row.iter_mut() {
                            *x *= scale;
                            if *x > mx {
                                mx = *x;
                            }
                        }
                        let mut sum = 0.0f32;
                        if fast_exp {
                            for x in row.iter_mut() {
                                *x -= mx;
                            }
                            exp_approx_slice(row);
                            sum = row.iter().sum();
                        } else {
                            for x in row.iter_mut() {
                                *x = (*x - mx).exp();
                                sum += *x;
                            }
                        }
                        let inv = 1.0 / sum;
                        for x in row.iter_mut() {
                            *x *= inv;
                        }
                    }
                    // O_t = S · V  (B[s][d] = V[s][d]).
                    sgemm_serial(rows, seq, head_dim, &scores, vh, head_dim, 1, out_t);
                });
        });
    Tensor::from_f32(&out, Shape::new([1, n_heads, seq, head_dim])).map_err(|e| anyhow!("{e}"))
}

/// One head's K and Vᵀ as Q8_0 rows plus their block-major f32 scales — the
/// "weights" of the two int8 attention GEMMs.
#[cfg(target_arch = "aarch64")]
struct Int8HeadKv {
    k: Vec<u8>,
    k_scales_t: Vec<f32>,
    vt: Vec<u8>,
    vt_scales_t: Vec<f32>,
}

/// Quantize `rows` rows of `width` values (`src[r·width..]`) to Q8_0, also
/// returning the scales widened to f32, block-major (`[bi · rows + r]`).
#[cfg(target_arch = "aarch64")]
fn q8_0_rows(src: impl Fn(usize, &mut [f32]), rows: usize, width: usize) -> (Vec<u8>, Vec<f32>) {
    use sapient_backends_cpu::kernels::quant::quantize_q8_0_block;
    let bpr = width / 32;
    let mut bytes = Vec::with_capacity(rows * bpr * 34);
    let mut scales_t = vec![0.0f32; bpr * rows];
    let mut buf = vec![0.0f32; width];
    for r in 0..rows {
        src(r, &mut buf);
        for (b, blk) in buf.chunks_exact(32).enumerate() {
            let qb = quantize_q8_0_block(blk);
            scales_t[b * rows + r] = half::f16::from_le_bytes([qb[0], qb[1]]).to_f32();
            bytes.extend_from_slice(&qb);
        }
    }
    (bytes, scales_t)
}

/// Quantize `rows` activation rows to int8 with per-32 scales, returning the
/// scales block-major as the GEMM tile wants them.
#[cfg(target_arch = "aarch64")]
fn i8_rows(x: &[f32], rows: usize, width: usize) -> (Vec<i8>, Vec<f32>) {
    use sapient_backends_cpu::kernels::quant::quantize_row_to_i8_blocks_into;
    let bpr = width / 32;
    let mut q = vec![0i8; rows * width];
    let mut sc = vec![0.0f32; rows * bpr];
    for r in 0..rows {
        quantize_row_to_i8_blocks_into(
            &x[r * width..(r + 1) * width],
            &mut q[r * width..(r + 1) * width],
            &mut sc[r * bpr..(r + 1) * bpr],
        );
    }
    let mut sc_t = vec![0.0f32; bpr * rows];
    for r in 0..rows {
        for b in 0..bpr {
            sc_t[b * rows + r] = sc[r * bpr + b];
        }
    }
    (q, sc_t)
}

/// int8 version of [`dense_full_attention`] (non-causal, same layout).
///
/// Both products run on the W8A8 `sdot` tile with per-32-element scales, the
/// format the Q8_0 linears already use:
/// * `S = Q·Kᵀ`: K is first **mean-centred over the sequence** per channel.
///   That adds the same constant `q·mean` to every score of a query row, which
///   softmax ignores — but it removes K's shared offset so the int8 grid is
///   spent on what distinguishes the keys (the SageAttention observation).
/// * `O = P·V`: the unnormalized `exp(s − max)` (largest value 1) is quantized
///   per 32 keys, multiplied with Vᵀ quantized per 32 keys, and the output row
///   divided by the row sum at the end.
///
/// The softmax itself stays f32 (polynomial `exp`). Approximate: SmolVLA gates
/// it on action error; `sapient see` never uses it.
#[cfg(target_arch = "aarch64")]
fn dense_attention_int8(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    n_heads: usize,
    head_dim: usize,
    tile: usize,
) -> Result<Tensor> {
    use rayon::prelude::*;
    use sapient_backends_cpu::kernels::elementwise::exp_approx_slice;
    use sapient_backends_cpu::kernels::quant::q8_0_gemm_nt_serial;
    let seq = q.shape().dims()[2];
    let hd = head_dim;
    let qv = q.to_f32_cow();
    let kv = k.to_f32_cow();
    let vv = v.to_f32_cow();
    let scale = 1.0 / (hd as f32).sqrt();
    let head_len = seq * hd;
    // Multiple of 4 so tiles map onto the 4×4 GEMM tile.
    let tile = (tile.max(4) / 4) * 4;

    let heads: Vec<Int8HeadKv> = (0..n_heads)
        .into_par_iter()
        .map(|h| {
            let kh = &kv[h * head_len..(h + 1) * head_len];
            let vh = &vv[h * head_len..(h + 1) * head_len];
            let mut mean = vec![0.0f32; hd];
            for row in kh.chunks_exact(hd) {
                for (m, x) in mean.iter_mut().zip(row) {
                    *m += x;
                }
            }
            for m in mean.iter_mut() {
                *m /= seq as f32;
            }
            let (k, k_scales_t) = q8_0_rows(
                |j, buf| {
                    for ((o, x), m) in buf.iter_mut().zip(&kh[j * hd..(j + 1) * hd]).zip(&mean) {
                        *o = x - m;
                    }
                },
                seq,
                hd,
            );
            let (vt, vt_scales_t) = q8_0_rows(
                |d, buf| {
                    for (j, o) in buf.iter_mut().enumerate() {
                        *o = vh[j * hd + d];
                    }
                },
                hd,
                seq,
            );
            Int8HeadKv {
                k,
                k_scales_t,
                vt,
                vt_scales_t,
            }
        })
        .collect();

    let mut out = vec![0.0f32; n_heads * head_len];
    out.par_chunks_mut(head_len)
        .enumerate()
        .for_each(|(h, out_h)| {
            let hk = &heads[h];
            let qh = &qv[h * head_len..(h + 1) * head_len];
            out_h
                .par_chunks_mut(tile * hd)
                .enumerate()
                .for_each(|(ti, out_t)| {
                    let r0 = ti * tile;
                    let rows = out_t.len() / hd;
                    let (qi8, qs_t) = i8_rows(&qh[r0 * hd..(r0 + rows) * hd], rows, hd);
                    let mut scores = vec![0.0f32; rows * seq];
                    // SAFETY: dotprod checked by the caller; sizes by construction.
                    unsafe {
                        q8_0_gemm_nt_serial(
                            &qi8,
                            &qs_t,
                            rows,
                            hd,
                            &hk.k,
                            &hk.k_scales_t,
                            seq,
                            &mut scores,
                            seq,
                        )
                    };
                    let mut sums = vec![0.0f32; rows];
                    for (row, sum) in scores.chunks_exact_mut(seq).zip(sums.iter_mut()) {
                        let mx = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                        for x in row.iter_mut() {
                            *x = (*x - mx) * scale;
                        }
                        exp_approx_slice(row);
                        *sum = row.iter().sum();
                    }
                    let (pi8, ps_t) = i8_rows(&scores, rows, seq);
                    // SAFETY: as above.
                    unsafe {
                        q8_0_gemm_nt_serial(
                            &pi8,
                            &ps_t,
                            rows,
                            seq,
                            &hk.vt,
                            &hk.vt_scales_t,
                            hd,
                            out_t,
                            hd,
                        )
                    };
                    for (o, sum) in out_t.chunks_exact_mut(hd).zip(&sums) {
                        let inv = 1.0 / sum;
                        for x in o.iter_mut() {
                            *x *= inv;
                        }
                    }
                });
        });
    Tensor::from_f32(&out, Shape::new([1, n_heads, seq, head_dim])).map_err(|e| anyhow!("{e}"))
}

#[cfg(test)]
mod tests {
    /// The pixel-shuffle ordering must be exactly transformers'
    /// `Idefics3Connector.pixel_shuffle`: out[hj][wj] = concat over dh (outer),
    /// dw (inner) of in[hj·s+dh][wj·s+dw], channels innermost.
    /// Tiling the tower attention over query rows must not change a single
    /// bit, and the result must match a naive reference.
    /// int8 attention tracks the f32 kernel on realistic-scale data (keys with
    /// a large shared offset — what the mean-centring is for).
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn int8_attention_tracks_f32() {
        use super::{Shape, Tensor};
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return;
        }
        let (heads, seq, hd) = (2usize, 96usize, 64usize);
        let gen = |salt: usize, offset: f32| -> Vec<f32> {
            (0..heads * seq * hd)
                .map(|i| {
                    (((i * 2654435761usize + salt * 97) % 2003) as f32 / 2003.0 - 0.5) * 4.0
                        + offset * ((i % hd) as f32 / hd as f32)
                })
                .collect()
        };
        let t = |d: Vec<f32>| Tensor::from_f32(&d, Shape::new([1, heads, seq, hd])).unwrap();
        let (q, k, v) = (t(gen(1, 0.0)), t(gen(2, 20.0)), t(gen(3, 0.0)));
        let exact = super::dense_full_attention_tiled(&q, &k, &v, heads, hd, seq, false)
            .unwrap()
            .to_f32_vec();
        for tile in [4usize, 30, 96] {
            let got = super::dense_attention_int8(&q, &k, &v, heads, hd, tile)
                .unwrap()
                .to_f32_vec();
            let (mut e, mut m) = (0.0f32, 0.0f32);
            for (a, b) in got.iter().zip(&exact) {
                e = e.max((a - b).abs());
                m = m.max(b.abs());
            }
            assert!(e < 0.03 * m, "tile {tile}: max err {e} (max |ref| {m})");
        }
    }

    #[test]
    fn tiled_attention_is_bit_identical_and_matches_naive() {
        use super::{Shape, Tensor};
        let (heads, seq, hd) = (2usize, 37usize, 8usize);
        let gen = |salt: usize| -> Vec<f32> {
            (0..heads * seq * hd)
                .map(|i| (((i * 2654435761usize + salt * 97) % 2003) as f32 / 2003.0 - 0.5) * 3.0)
                .collect()
        };
        let (qd, kd, vd) = (gen(1), gen(2), gen(3));
        let t = |d: &[f32]| Tensor::from_f32(d, Shape::new([1, heads, seq, hd])).unwrap();
        let (q, k, v) = (t(&qd), t(&kd), t(&vd));

        let whole = super::dense_full_attention_tiled(&q, &k, &v, heads, hd, seq, false)
            .unwrap()
            .to_f32_vec();
        for tile in [1usize, 5, 16, 36] {
            let tiled = super::dense_full_attention_tiled(&q, &k, &v, heads, hd, tile, false)
                .unwrap()
                .to_f32_vec();
            for (i, (a, b)) in tiled.iter().zip(&whole).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "tile {tile}, element {i}");
            }
        }

        // Naive reference in f64.
        let scale = 1.0 / (hd as f64).sqrt();
        for h in 0..heads {
            for i in 0..seq {
                let at = |d: &[f32], r: usize, c: usize| d[(h * seq + r) * hd + c] as f64;
                let s: Vec<f64> = (0..seq)
                    .map(|j| (0..hd).map(|d| at(&qd, i, d) * at(&kd, j, d)).sum::<f64>() * scale)
                    .collect();
                let mx = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = s.iter().map(|x| (x - mx).exp()).collect();
                let z: f64 = e.iter().sum();
                for d in 0..hd {
                    let want: f64 = (0..seq).map(|j| e[j] / z * at(&vd, j, d)).sum();
                    let got = whole[(h * seq + i) * hd + d] as f64;
                    assert!((got - want).abs() < 1e-4, "h{h} i{i} d{d}: {got} vs {want}");
                }
            }
        }
    }

    #[test]
    fn pixel_shuffle_ordering_matches_reference() {
        // 4×4 grid, c=1, s=2 → 2×2 output with 4 channels each.
        let side = 4usize;
        let c = 1usize;
        let sf = 2usize;
        let xv: Vec<f32> = (0..side * side).map(|i| i as f32).collect();
        let out_side = side / sf;
        let cs2 = c * sf * sf;
        let mut shuffled = vec![0.0f32; out_side * out_side * cs2];
        for hj in 0..out_side {
            for wj in 0..out_side {
                let dst = (hj * out_side + wj) * cs2;
                for dh in 0..sf {
                    for dw in 0..sf {
                        let src_patch = (hj * sf + dh) * side + (wj * sf + dw);
                        let d = dst + (dh * sf + dw) * c;
                        shuffled[d..d + c].copy_from_slice(&xv[src_patch * c..(src_patch + 1) * c]);
                    }
                }
            }
        }
        // Reference (worked by hand from the transformers view/permute chain):
        // out[0][0] = [in(0,0), in(0,1), in(1,0), in(1,1)] = [0, 1, 4, 5]
        assert_eq!(&shuffled[..4], &[0.0, 1.0, 4.0, 5.0]);
        // out[1][1] = [in(2,2), in(2,3), in(3,2), in(3,3)] = [10, 11, 14, 15]
        assert_eq!(&shuffled[12..16], &[10.0, 11.0, 14.0, 15.0]);
    }
}
