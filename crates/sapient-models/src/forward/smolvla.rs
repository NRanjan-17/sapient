// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! SmolVLA — a vision-language-action policy (LeRobot `lerobot/smolvla_base`).
//!
//! One inference turns camera image(s) + a language instruction + the robot
//! state into a **chunk of future actions** (`chunk × max_action_dim`, 50 × 32):
//!
//! 1. **Prefix** — SigLIP + connector image tokens (×√hidden), language token
//!    embeddings (×√hidden) and one state token (`state_proj`) run through the
//!    first 16 layers of SmolVLM2-500M's text model. Each layer's post-RoPE
//!    keys and raw values are kept: that K/V cache is all the action expert
//!    ever sees of the observation.
//! 2. **Action expert** — a 16-layer, 720-wide transformer over the 50 noisy
//!    action tokens. **Even layers** are self-attention over `[prefix K/V ;
//!    action K/V]` (causal inside the chunk); **odd layers** are
//!    cross-attention: the expert's `k_proj`/`v_proj` re-project the VLM
//!    layer's cached K/V and the action tokens attend to the prefix only.
//! 3. **Flow matching** — `num_steps` (10) forward-Euler steps from noise
//!    (t = 1) to actions (t = 0): `x ← x − v(x, t) / num_steps`.
//!
//! Things that differ from the stock VLM path and are easy to get wrong:
//! * RoPE base is **10 000** here, not SmolVLM2's configured 100 000 (the
//!   reference hard-codes it for both the VLM layers and the expert).
//! * The prefix is **not causal**: image and language tokens attend to each
//!   other bidirectionally, none of them attends to the state token, and the
//!   state token attends to everything.
//! * Padded language tokens are never attended to and do not advance the
//!   position counter, so this engine simply **drops** them (same result for
//!   every real token, shorter prefix).
//! * Cross-attention layers rebase the action positions to `0..chunk`;
//!   self-attention layers continue after the prefix (`n_prefix..`).
//!
//! CPU, f32. Validated stage by stage against the LeRobot reference
//! (`scripts/gen_smolvla_fixture.py`, `tests/smolvla_reference.rs`).

use std::collections::HashMap;

use anyhow::{anyhow, bail, Result};
use rayon::prelude::*;
use sapient_backends_cpu::kernels::matmul::{matmul_nt, sgemm_serial};
use sapient_core::{DType, Shape, Tensor};

use super::common::embed_tokens;
use super::siglip::{SiglipConfig, SiglipVision};

const VLM_PREFIX: &str = "model.vlm_with_expert.vlm.";
const EXPERT_PREFIX: &str = "model.vlm_with_expert.lm_expert.";
const TEXT: &str = "model.text_model";

/// SmolVLA dimensions. Defaults are `lerobot/smolvla_base`; the widths are
/// re-read from the checkpoint's tensor shapes in [`SmolVla::from_weights`].
#[derive(Debug, Clone)]
pub struct SmolVlaConfig {
    pub vlm_hidden: usize,
    pub expert_hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// Actions per chunk (`chunk_size`).
    pub chunk: usize,
    pub max_state_dim: usize,
    pub max_action_dim: usize,
    /// Flow-matching Euler steps.
    pub num_steps: usize,
    /// Every n-th expert layer (0, n, 2n, …) is self-attention; the rest are
    /// cross-attention over the VLM K/V.
    pub self_attn_every: usize,
    pub min_period: f64,
    pub max_period: f64,
    pub rms_eps: f32,
    pub rope_base: f32,
}

impl Default for SmolVlaConfig {
    fn default() -> Self {
        Self {
            vlm_hidden: 960,
            expert_hidden: 720,
            layers: 16,
            heads: 15,
            kv_heads: 5,
            head_dim: 64,
            chunk: 50,
            max_state_dim: 32,
            max_action_dim: 32,
            num_steps: 10,
            self_attn_every: 2,
            min_period: 4e-3,
            max_period: 4.0,
            rms_eps: 1e-5,
            rope_base: 10_000.0,
        }
    }
}

/// The observation as the action expert sees it: per VLM layer, the post-RoPE
/// keys and the values of the `n` prefix tokens, both `[kv_heads, n, head_dim]`.
pub struct PrefixCache {
    pub n: usize,
    pub keys: Vec<Vec<f32>>,
    pub values: Vec<Vec<f32>>,
}

/// A loaded SmolVLA policy.
pub struct SmolVla {
    cfg: SmolVlaConfig,
    vision: SiglipVision,
    w: HashMap<String, Tensor>,
}

/// Convert a float tensor to F32 (exact for F16/BF16 sources).
fn to_f32(t: Tensor) -> Result<Tensor> {
    if t.dtype() == DType::F32 {
        return Ok(t);
    }
    let dims = t.shape().dims().to_vec();
    Tensor::from_f32_vec(t.to_f32_vec(), Shape::new(dims)).map_err(|e| anyhow!("{e}"))
}

impl SmolVla {
    /// Build from the checkpoint's tensors (keys as in `model.safetensors`).
    ///
    /// Everything except the token-embedding table is widened to F32 so the
    /// engine reproduces the f32 reference exactly; quantizing the linears is
    /// a later, separately measured step. The unused `lm_head` is dropped.
    pub fn from_weights(mut cfg: SmolVlaConfig, weights: HashMap<String, Tensor>) -> Result<Self> {
        let mut vision_w = HashMap::new();
        let mut w = HashMap::new();
        for (name, t) in weights {
            if let Some(rest) = name.strip_prefix(VLM_PREFIX) {
                if rest.starts_with("lm_head") {
                    continue;
                }
                if rest.starts_with("model.vision_model") || rest.starts_with("model.connector") {
                    vision_w.insert(rest.to_string(), t);
                } else if rest.ends_with("embed_tokens.weight") {
                    w.insert(rest.to_string(), t);
                } else {
                    w.insert(rest.to_string(), to_f32(t)?);
                }
            } else if let Some(rest) = name.strip_prefix(EXPERT_PREFIX) {
                w.insert(format!("expert.{rest}"), to_f32(t)?);
            } else if let Some(rest) = name.strip_prefix("model.") {
                w.insert(format!("head.{rest}"), to_f32(t)?);
            }
        }

        let dims = |name: &str| -> Result<Vec<usize>> {
            w.get(name)
                .map(|t: &Tensor| t.shape().dims().to_vec())
                .ok_or_else(|| anyhow!("SmolVLA weight missing: {name}"))
        };
        cfg.vlm_hidden = dims(&format!("{TEXT}.embed_tokens.weight"))?[1];
        cfg.expert_hidden = dims("expert.norm.weight")?[0];
        let q = dims("expert.layers.0.self_attn.q_proj.weight")?[0];
        let kv = dims(&format!("{TEXT}.layers.0.self_attn.k_proj.weight"))?[0];
        if q != cfg.heads * cfg.head_dim || kv != cfg.kv_heads * cfg.head_dim {
            bail!(
                "SmolVLA attention shape mismatch: q_proj out {q}, k_proj out {kv}, expected \
                 {} heads / {} KV heads of dim {}",
                cfg.heads,
                cfg.kv_heads,
                cfg.head_dim
            );
        }
        cfg.max_state_dim = dims("head.state_proj.weight")?[1];
        cfg.max_action_dim = dims("head.action_in_proj.weight")?[1];
        cfg.layers = (0..)
            .take_while(|i| w.contains_key(&format!("{TEXT}.layers.{i}.input_layernorm.weight")))
            .count();
        if !w.contains_key(&format!(
            "expert.layers.{}.input_layernorm.weight",
            cfg.layers - 1
        )) {
            bail!("SmolVLA: expert must have one layer per VLM layer");
        }

        let pos = vision_w
            .get("model.vision_model.embeddings.position_embedding.weight")
            .ok_or_else(|| anyhow!("SmolVLA: vision position embedding missing"))?
            .shape()
            .dims()
            .to_vec();
        let patch_w = vision_w
            .get("model.vision_model.embeddings.patch_embedding.weight")
            .ok_or_else(|| anyhow!("SmolVLA: vision patch embedding missing"))?
            .shape()
            .dims()
            .to_vec();
        let side = (pos[0] as f64).sqrt() as usize;
        let proj_in = vision_w
            .get("model.connector.modality_projection.proj.weight")
            .ok_or_else(|| anyhow!("SmolVLA: connector projection missing"))?
            .shape()
            .dims()[1];
        let vcfg = SiglipConfig {
            hidden: pos[1],
            layers: (0..)
                .take_while(|i| {
                    vision_w.contains_key(&format!(
                        "model.vision_model.encoder.layers.{i}.layer_norm1.weight"
                    ))
                })
                .count(),
            heads: 12,
            intermediate: vision_w
                .get("model.vision_model.encoder.layers.0.mlp.fc1.weight")
                .ok_or_else(|| anyhow!("SmolVLA: vision MLP missing"))?
                .shape()
                .dims()[0],
            image_size: side * patch_w[2],
            patch: patch_w[2],
            scale_factor: ((proj_in / pos[1]) as f64).sqrt() as usize,
            text_hidden: cfg.vlm_hidden,
        };
        let vision = SiglipVision::new(vcfg, vision_w)?;
        Ok(Self { cfg, vision, w })
    }

    pub fn config(&self) -> &SmolVlaConfig {
        &self.cfg
    }

    /// Side length of the square image the vision tower expects.
    pub fn image_size(&self) -> usize {
        self.vision.config().image_size
    }

    fn get(&self, name: &str) -> Result<&Tensor> {
        self.w
            .get(name)
            .ok_or_else(|| anyhow!("SmolVLA weight missing: {name}"))
    }

    /// `y = x·Wᵀ (+ b)` over `rows` row vectors; `name` without `.weight`.
    fn linear(&self, x: &[f32], rows: usize, name: &str) -> Result<Vec<f32>> {
        let w = self.get(&format!("{name}.weight"))?;
        let in_dim = w.shape().dims()[1];
        if x.len() != rows * in_dim {
            bail!("{name}: input {} != {rows} x {in_dim}", x.len());
        }
        let xt = Tensor::from_f32(x, Shape::new([rows, in_dim])).map_err(|e| anyhow!("{e}"))?;
        let mut y = matmul_nt(&xt, w)
            .map_err(|e| anyhow!("{name}: {e}"))?
            .to_f32_vec();
        if let Some(b) = self.w.get(&format!("{name}.bias")) {
            let b = b.as_f32_slice();
            for row in y.chunks_exact_mut(b.len()) {
                for (v, bi) in row.iter_mut().zip(b) {
                    *v += bi;
                }
            }
        }
        Ok(y)
    }

    /// RMSNorm over rows of width `weight.len()`.
    fn rms_norm(&self, x: &[f32], name: &str) -> Result<Vec<f32>> {
        let w = self.get(name)?.as_f32_slice();
        let mut out = vec![0.0f32; x.len()];
        for (src, dst) in x.chunks_exact(w.len()).zip(out.chunks_exact_mut(w.len())) {
            let ms = src.iter().map(|v| v * v).sum::<f32>() / w.len() as f32;
            let inv = 1.0 / (ms + self.cfg.rms_eps).sqrt();
            for ((d, s), wi) in dst.iter_mut().zip(src).zip(w) {
                *d = s * inv * wi;
            }
        }
        Ok(out)
    }

    /// SwiGLU MLP: `down(silu(gate(x)) · up(x))`.
    fn mlp(&self, x: &[f32], rows: usize, layer: &str) -> Result<Vec<f32>> {
        let mut g = self.linear(x, rows, &format!("{layer}.mlp.gate_proj"))?;
        let u = self.linear(x, rows, &format!("{layer}.mlp.up_proj"))?;
        for (gi, ui) in g.iter_mut().zip(&u) {
            *gi = *gi / (1.0 + (-*gi).exp()) * ui;
        }
        self.linear(&g, rows, &format!("{layer}.mlp.down_proj"))
    }

    // ── prefix ──────────────────────────────────────────────────────────────

    /// Preprocessed pixels `[3, S, S]` in `[-1, 1]` → `[n_img_tokens · vlm_hidden]`
    /// (SigLIP + connector, before the √hidden scale).
    pub fn embed_image(&self, pixels: &[f32]) -> Result<Vec<f32>> {
        self.vision.encode(pixels)
    }

    /// Build the prefix embeddings `[n, vlm_hidden]`: every image's tokens,
    /// then the language tokens, then the state token. `lang_tokens` must hold
    /// only real tokens (no padding); `state` is padded to `max_state_dim`.
    pub fn embed_prefix(
        &self,
        images: &[&[f32]],
        lang_tokens: &[u32],
        state: &[f32],
    ) -> Result<Vec<f32>> {
        let h = self.cfg.vlm_hidden;
        let scale = (h as f32).sqrt();
        let mut embs = Vec::new();
        for pixels in images {
            embs.extend(self.embed_image(pixels)?.into_iter().map(|v| v * scale));
        }
        let table = self.get(&format!("{TEXT}.embed_tokens.weight"))?;
        let lang = embed_tokens(table, lang_tokens)?;
        embs.extend(lang.as_f32_slice().iter().map(|v| v * scale));

        if state.len() > self.cfg.max_state_dim {
            bail!(
                "state has {} dims, the policy takes at most {}",
                state.len(),
                self.cfg.max_state_dim
            );
        }
        let mut padded = vec![0.0f32; self.cfg.max_state_dim];
        padded[..state.len()].copy_from_slice(state);
        embs.extend(self.linear(&padded, 1, "head.state_proj")?);
        Ok(embs)
    }

    /// Run the prefix through the VLM layers, keeping each layer's K/V.
    ///
    /// The last token must be the state token (it is the only one the others
    /// may not attend to). Also returns the final-norm hidden states — unused
    /// by action sampling, kept for validation.
    pub fn prefix_pass(&self, embs: &[f32]) -> Result<(PrefixCache, Vec<f32>)> {
        let c = &self.cfg;
        let h = c.vlm_hidden;
        let n = embs.len() / h;
        if n < 2 || embs.len() != n * h {
            bail!("prefix must be [n ≥ 2, {h}]");
        }
        // Image + language attend among themselves; the state token (last)
        // attends to everything and is attended to only by itself.
        let allow: Vec<bool> = (0..n * n)
            .map(|ij| ij % n < n - 1 || ij / n == n - 1)
            .collect();
        let positions: Vec<usize> = (0..n).collect();

        let mut x = embs.to_vec();
        let mut cache = PrefixCache {
            n,
            keys: Vec::with_capacity(c.layers),
            values: Vec::with_capacity(c.layers),
        };
        for l in 0..c.layers {
            let p = format!("{TEXT}.layers.{l}");
            let normed = self.rms_norm(&x, &format!("{p}.input_layernorm.weight"))?;
            let mut q = self.linear(&normed, n, &format!("{p}.self_attn.q_proj"))?;
            let mut k = self.linear(&normed, n, &format!("{p}.self_attn.k_proj"))?;
            let v = self.linear(&normed, n, &format!("{p}.self_attn.v_proj"))?;
            rope(&mut q, c.heads, c.head_dim, &positions, c.rope_base);
            rope(&mut k, c.kv_heads, c.head_dim, &positions, c.rope_base);
            let k = heads_major(&k, n, c.kv_heads, c.head_dim);
            let v = heads_major(&v, n, c.kv_heads, c.head_dim);
            let att = attention(&q, &k, &v, n, n, c, &allow);
            cache.keys.push(k);
            cache.values.push(v);

            let o = self.linear(&att, n, &format!("{p}.self_attn.o_proj"))?;
            for (xi, oi) in x.iter_mut().zip(&o) {
                *xi += oi;
            }
            let normed = self.rms_norm(&x, &format!("{p}.post_attention_layernorm.weight"))?;
            let m = self.mlp(&normed, n, &p)?;
            for (xi, mi) in x.iter_mut().zip(&m) {
                *xi += mi;
            }
        }
        let out = self.rms_norm(&x, &format!("{TEXT}.norm.weight"))?;
        Ok((cache, out))
    }

    // ── action expert ───────────────────────────────────────────────────────

    /// Embed the noisy action chunk `[chunk, max_action_dim]` at flow time `t`
    /// → `[chunk, expert_hidden]`.
    pub fn embed_suffix(&self, x_t: &[f32], t: f32) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let e = c.expert_hidden;
        let action = self.linear(x_t, c.chunk, "head.action_in_proj")?;
        let time = sinusoidal_time_embedding(t, e, c.min_period, c.max_period);
        let mut cat = Vec::with_capacity(c.chunk * 2 * e);
        for row in action.chunks_exact(e) {
            cat.extend_from_slice(row);
            cat.extend_from_slice(&time);
        }
        let mut hid = self.linear(&cat, c.chunk, "head.action_time_mlp_in")?;
        for v in hid.iter_mut() {
            *v /= 1.0 + (-*v).exp(); // silu
        }
        self.linear(&hid, c.chunk, "head.action_time_mlp_out")
    }

    /// One evaluation of the velocity field: `v(x_t, t)` as `[chunk, max_action_dim]`.
    pub fn denoise_step(&self, cache: &PrefixCache, x_t: &[f32], t: f32) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let (s, n, hd) = (c.chunk, cache.n, c.head_dim);
        let kv_w = c.kv_heads * hd;
        let mut x = self.embed_suffix(x_t, t)?;

        // Self-attention: all prefix tokens + causal inside the chunk.
        let allow_self: Vec<bool> = (0..s * (n + s))
            .map(|ij| {
                let (i, j) = (ij / (n + s), ij % (n + s));
                j < n || j - n <= i
            })
            .collect();
        let allow_cross = vec![true; s * n];
        let pos_self: Vec<usize> = (n..n + s).collect();
        let pos_cross: Vec<usize> = (0..s).collect();

        for l in 0..c.layers {
            let p = format!("expert.layers.{l}");
            let normed = self.rms_norm(&x, &format!("{p}.input_layernorm.weight"))?;
            let mut q = self.linear(&normed, s, &format!("{p}.self_attn.q_proj"))?;
            let att = if c.self_attn_every > 0 && l % c.self_attn_every == 0 {
                let mut k = self.linear(&normed, s, &format!("{p}.self_attn.k_proj"))?;
                let v = self.linear(&normed, s, &format!("{p}.self_attn.v_proj"))?;
                rope(&mut q, c.heads, hd, &pos_self, c.rope_base);
                rope(&mut k, c.kv_heads, hd, &pos_self, c.rope_base);
                let k = heads_major(&k, s, c.kv_heads, hd);
                let v = heads_major(&v, s, c.kv_heads, hd);
                // [prefix ; suffix] per KV head.
                let join = |pre: &[f32], suf: &[f32]| -> Vec<f32> {
                    let mut out = Vec::with_capacity(c.kv_heads * (n + s) * hd);
                    for h in 0..c.kv_heads {
                        out.extend_from_slice(&pre[h * n * hd..(h + 1) * n * hd]);
                        out.extend_from_slice(&suf[h * s * hd..(h + 1) * s * hd]);
                    }
                    out
                };
                let k = join(&cache.keys[l], &k);
                let v = join(&cache.values[l], &v);
                attention(&q, &k, &v, s, n + s, c, &allow_self)
            } else {
                // Re-project the VLM layer's K/V: [kv_heads, n, hd] → rows of
                // kv_heads·hd → expert k_proj / v_proj → back to heads-major.
                let k_rows = seq_major(&cache.keys[l], n, c.kv_heads, hd);
                let v_rows = seq_major(&cache.values[l], n, c.kv_heads, hd);
                debug_assert_eq!(k_rows.len(), n * kv_w);
                let k = self.linear(&k_rows, n, &format!("{p}.self_attn.k_proj"))?;
                let v = self.linear(&v_rows, n, &format!("{p}.self_attn.v_proj"))?;
                rope(&mut q, c.heads, hd, &pos_cross, c.rope_base);
                let k = heads_major(&k, n, c.kv_heads, hd);
                let v = heads_major(&v, n, c.kv_heads, hd);
                attention(&q, &k, &v, s, n, c, &allow_cross)
            };
            let o = self.linear(&att, s, &format!("{p}.self_attn.o_proj"))?;
            for (xi, oi) in x.iter_mut().zip(&o) {
                *xi += oi;
            }
            let normed = self.rms_norm(&x, &format!("{p}.post_attention_layernorm.weight"))?;
            let m = self.mlp(&normed, s, &p)?;
            for (xi, mi) in x.iter_mut().zip(&m) {
                *xi += mi;
            }
        }
        let out = self.rms_norm(&x, "expert.norm.weight")?;
        self.linear(&out, s, "head.action_out_proj")
    }

    /// Integrate the flow from `noise` (t = 1) to actions (t = 0) in
    /// `num_steps` Euler steps. Returns `[chunk, max_action_dim]` in the
    /// policy's normalized action space.
    pub fn sample_actions(&self, cache: &PrefixCache, noise: &[f32]) -> Result<Vec<f32>> {
        let c = &self.cfg;
        if noise.len() != c.chunk * c.max_action_dim {
            bail!("noise must be [{}, {}]", c.chunk, c.max_action_dim);
        }
        let dt = -1.0f64 / c.num_steps as f64;
        let mut x = noise.to_vec();
        for step in 0..c.num_steps {
            let t = (1.0 + step as f64 * dt) as f32;
            let v = self.denoise_step(cache, &x, t)?;
            for (xi, vi) in x.iter_mut().zip(&v) {
                *xi += dt as f32 * vi;
            }
        }
        Ok(x)
    }
}

/// `[seq, heads·hd]` (seq-major, as a linear produces it) → `[heads, seq, hd]`.
fn heads_major(x: &[f32], seq: usize, heads: usize, hd: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for s in 0..seq {
        for h in 0..heads {
            let src = (s * heads + h) * hd;
            let dst = (h * seq + s) * hd;
            out[dst..dst + hd].copy_from_slice(&x[src..src + hd]);
        }
    }
    out
}

/// Inverse of [`heads_major`].
fn seq_major(x: &[f32], seq: usize, heads: usize, hd: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for s in 0..seq {
        for h in 0..heads {
            let src = (h * seq + s) * hd;
            let dst = (s * heads + h) * hd;
            out[dst..dst + hd].copy_from_slice(&x[src..src + hd]);
        }
    }
    out
}

/// In-place rotate-half RoPE on seq-major `[seq, heads·hd]` data.
fn rope(x: &mut [f32], heads: usize, hd: usize, positions: &[usize], base: f32) {
    let half = hd / 2;
    let inv: Vec<f32> = (0..half)
        .map(|i| 1.0 / base.powf(2.0 * i as f32 / hd as f32))
        .collect();
    for (row, &pos) in x.chunks_exact_mut(heads * hd).zip(positions) {
        for head in row.chunks_exact_mut(hd) {
            for i in 0..half {
                let (sin, cos) = (pos as f32 * inv[i]).sin_cos();
                let (a, b) = (head[i], head[i + half]);
                head[i] = a * cos - b * sin;
                head[i + half] = b * cos + a * sin;
            }
        }
    }
}

/// Masked grouped-query attention.
///
/// `q` is seq-major `[sq, heads·hd]`; `k`/`v` are heads-major
/// `[kv_heads, sk, hd]`; `allow[i·sk + j]` says whether query `i` may attend to
/// key `j`. Returns seq-major `[sq, heads·hd]`. A fully masked row yields a
/// uniform distribution, like the reference's `finfo.min` fill.
fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    sq: usize,
    sk: usize,
    c: &SmolVlaConfig,
    allow: &[bool],
) -> Vec<f32> {
    let (heads, hd) = (c.heads, c.head_dim);
    let rep = heads / c.kv_heads;
    let scale = (hd as f32).powf(-0.5);
    let qh = heads_major(q, sq, heads, hd);
    let mut out_h = vec![0.0f32; heads * sq * hd];
    out_h
        .par_chunks_mut(sq * hd)
        .enumerate()
        .for_each(|(h, out)| {
            let kvh = h / rep;
            let q_h = &qh[h * sq * hd..(h + 1) * sq * hd];
            let k_h = &k[kvh * sk * hd..(kvh + 1) * sk * hd];
            let v_h = &v[kvh * sk * hd..(kvh + 1) * sk * hd];
            let mut scores = vec![0.0f32; sq * sk];
            sgemm_serial(sq, hd, sk, q_h, k_h, 1, hd, &mut scores);
            for (i, row) in scores.chunks_exact_mut(sk).enumerate() {
                let mut mx = f32::MIN;
                for (j, s) in row.iter_mut().enumerate() {
                    *s = if allow[i * sk + j] {
                        *s * scale
                    } else {
                        f32::MIN
                    };
                    mx = mx.max(*s);
                }
                let mut sum = 0.0f32;
                for s in row.iter_mut() {
                    *s = (*s - mx).exp();
                    sum += *s;
                }
                let inv = 1.0 / sum;
                for s in row.iter_mut() {
                    *s *= inv;
                }
            }
            sgemm_serial(sq, sk, hd, &scores, v_h, hd, 1, out);
        });
    seq_major(&out_h, sq, heads, hd)
}

/// Sine-cosine embedding of the flow time `t` (openpi's
/// `create_sinusoidal_pos_embedding`): periods log-spaced between
/// `min_period` and `max_period`, computed in f64, laid out `[sin…, cos…]`.
fn sinusoidal_time_embedding(t: f32, dim: usize, min_period: f64, max_period: f64) -> Vec<f32> {
    let half = dim / 2;
    let mut out = vec![0.0f32; dim];
    for i in 0..half {
        let frac = if half > 1 {
            i as f64 / (half - 1) as f64
        } else {
            0.0
        };
        let period = min_period * (max_period / min_period).powf(frac);
        let arg = t as f64 * (1.0 / period * 2.0 * std::f64::consts::PI);
        out[i] = arg.sin() as f32;
        out[half + i] = arg.cos() as f32;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heads_major_round_trips() {
        let (seq, heads, hd) = (3, 4, 2);
        let x: Vec<f32> = (0..seq * heads * hd).map(|i| i as f32).collect();
        let hm = heads_major(&x, seq, heads, hd);
        // token 1, head 2 → heads-major slot (2, 1)
        assert_eq!(hm[(2 * seq + 1) * hd], x[(heads + 2) * hd]);
        assert_eq!(seq_major(&hm, seq, heads, hd), x);
    }

    #[test]
    fn rope_matches_the_shared_kernel() {
        let (seq, heads, hd) = (5usize, 3usize, 8usize);
        let x: Vec<f32> = (0..seq * heads * hd)
            .map(|i| ((i * 37 % 101) as f32 / 101.0) - 0.5)
            .collect();
        let positions = [0usize, 1, 2, 7, 40];
        let mut ours = x.clone();
        rope(&mut ours, heads, hd, &positions, 10_000.0);

        let hm = heads_major(&x, seq, heads, hd);
        let t = Tensor::from_f32(&hm, Shape::new([1, heads, seq, hd])).unwrap();
        let r = sapient_backends_cpu::kernels::rope::apply_rope(&t, &positions, 10_000.0)
            .unwrap()
            .to_f32_vec();
        let r = seq_major(&r, seq, heads, hd);
        for (a, b) in ours.iter().zip(&r) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn time_embedding_endpoints() {
        let e = sinusoidal_time_embedding(1.0, 8, 4e-3, 4.0);
        // Slowest period = 4.0 → angle π/2 at t = 1.
        assert!((e[3] - 1.0).abs() < 1e-6 && e[7].abs() < 1e-6);
        // Fastest period = 4e-3 → 250 full turns.
        assert!(e[0].abs() < 1e-4 && (e[4] - 1.0).abs() < 1e-6);
    }

    /// One query, two keys, identity-like values: masking a key must move all
    /// the weight to the other one, and a GQA group must share its K/V head.
    #[test]
    fn attention_respects_mask_and_gqa() {
        let c = SmolVlaConfig {
            heads: 2,
            kv_heads: 1,
            head_dim: 2,
            ..Default::default()
        };
        let q = vec![1.0, 0.0, 0.0, 1.0]; // [sq=1, heads=2, hd=2]
        let k = vec![1.0, 0.0, 0.0, 1.0]; // [kv=1, sk=2, hd=2]
        let v = vec![10.0, 0.0, 0.0, 20.0];
        let masked = attention(&q, &k, &v, 1, 2, &c, &[true, false]);
        assert_eq!(masked, vec![10.0, 0.0, 10.0, 0.0]);

        let open = attention(&q, &k, &v, 1, 2, &c, &[true, true]);
        let s = (2.0f32).powf(-0.5);
        let p = s.exp() / (s.exp() + 1.0); // head 0 favours key 0
        assert!((open[0] - 10.0 * p).abs() < 1e-5);
        assert!((open[3] - 20.0 * p).abs() < 1e-5); // head 1 favours key 1
    }
}
