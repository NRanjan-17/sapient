// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! SmolVLA against the LeRobot reference, stage by stage.
//!
//! The fixture (`tests/fixtures/smolvla_base.safetensors`) holds the inputs and
//! every intermediate of `lerobot/smolvla_base` run in f32 on CPU — written by
//! `scripts/gen_smolvla_fixture.py`. This test loads the real checkpoint and
//! checks, in order: image embedding → prefix embeddings → per-layer prefix K/V
//! → prefix output → suffix embedding → one velocity → the full 10-step action
//! chunk.
//!
//! Ignored by default: it needs the 0.9 GB checkpoint. Point
//! `SAPIENT_SMOLVLA_DIR` at a directory holding its `model.safetensors`:
//!
//! ```bash
//! SAPIENT_SMOLVLA_DIR=~/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/<rev> \
//!   cargo test -p sapient-models --release --test smolvla_reference -- --ignored --nocapture
//! ```

use std::collections::HashMap;
use std::path::PathBuf;

use sapient_core::Tensor;
use sapient_models::forward::{SmolVla, SmolVlaConfig, SmolVlaQuant};

fn fixture() -> HashMap<String, Tensor> {
    let p =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/smolvla_base.safetensors");
    sapient_io::load_safetensors(&p).expect("load SmolVLA fixture")
}

fn fx(f: &HashMap<String, Tensor>, name: &str) -> Vec<f32> {
    f.get(name)
        .unwrap_or_else(|| panic!("fixture tensor missing: {name}"))
        .to_f32_vec()
}

/// The generator's procedural 512×512 image, preprocessed to `[3, S, S]` in
/// `[-1, 1]` (mirrors `test_image` in `scripts/gen_smolvla_fixture.py`).
fn test_image(size: usize) -> Vec<f32> {
    let mut px = vec![0.0f32; 3 * size * size];
    for y in 0..size {
        for x in 0..size {
            let rgb = if (160..352).contains(&y) && (144..368).contains(&x) {
                [217u8, 26, 26]
            } else {
                [
                    (x * 255 / size) as u8,
                    (y * 255 / size) as u8,
                    ((x * 7 + y * 13) % 256) as u8,
                ]
            };
            for (c, v) in rgb.iter().enumerate() {
                px[c * size * size + y * size + x] = *v as f32 / 255.0 * 2.0 - 1.0;
            }
        }
    }
    px
}

/// (max abs error, max abs reference value) over the compared elements.
fn err(got: &[f32], want: &[f32]) -> (f32, f32) {
    assert_eq!(got.len(), want.len(), "length mismatch");
    let mut e = 0.0f32;
    let mut m = 0.0f32;
    for (g, w) in got.iter().zip(want) {
        e = e.max((g - w).abs());
        m = m.max(w.abs());
    }
    (e, m)
}

fn check(stage: &str, got: &[f32], want: &[f32], tol: f32) {
    let (e, m) = err(got, want);
    println!("{stage:28} max_err {e:.3e}  (max |ref| {m:.3}, tol {tol:.0e})");
    assert!(e <= tol, "{stage}: max_err {e} > {tol}");
}

/// Rows of `full` (`[n_full, width]`) where `keep` is set.
fn keep_rows(full: &[f32], width: usize, keep: &[bool]) -> Vec<f32> {
    full.chunks_exact(width)
        .zip(keep)
        .filter(|(_, k)| **k)
        .flat_map(|(r, _)| r.iter().copied())
        .collect()
}

#[test]
#[ignore = "needs the lerobot/smolvla_base checkpoint: set SAPIENT_SMOLVLA_DIR"]
fn smolvla_matches_lerobot_reference() {
    let dir = std::env::var("SAPIENT_SMOLVLA_DIR").expect("set SAPIENT_SMOLVLA_DIR");
    let weights = sapient_io::load_safetensors(&PathBuf::from(dir).join("model.safetensors"))
        .expect("load SmolVLA checkpoint");
    let model = SmolVla::from_weights(SmolVlaConfig::default(), weights).expect("build SmolVLA");
    let c = model.config().clone();
    let f = fixture();

    // ── inputs ──────────────────────────────────────────────────────────────
    let pixels = test_image(model.image_size());
    let lang_mask = fx(&f, "input.lang_mask");
    let lang: Vec<u32> = fx(&f, "input.lang_tokens")
        .iter()
        .zip(&lang_mask)
        .filter(|(_, m)| **m > 0.5)
        .map(|(t, _)| *t as u32)
        .collect();
    let state = fx(&f, "input.state");
    let noise = fx(&f, "input.noise");
    // The reference keeps padded language tokens in the prefix; we drop them.
    let keep: Vec<bool> = fx(&f, "prefix.pad_mask").iter().map(|m| *m > 0.5).collect();
    let n = keep.iter().filter(|k| **k).count();
    println!(
        "prefix: {n} real tokens of {} (reference pads language to 48)",
        keep.len()
    );

    // ── 1. vision ───────────────────────────────────────────────────────────
    let t = std::time::Instant::now();
    let img = model.embed_image(&pixels).unwrap();
    let vision_ms = t.elapsed().as_secs_f64() * 1e3;
    check(
        "image embedding",
        &img,
        &fx(&f, "vision.image_embedding"),
        2e-2,
    );

    // ── 2. prefix embeddings ────────────────────────────────────────────────
    let embs = model.embed_prefix(&[&pixels], &lang, &state).unwrap();
    assert_eq!(embs.len(), n * c.vlm_hidden);
    check(
        "prefix embeddings",
        &embs,
        &keep_rows(&fx(&f, "prefix.embs"), c.vlm_hidden, &keep),
        0.5, // values reach ~±300 after the √hidden scale; stored as f16
    );

    // ── 3. prefix pass: per-layer K/V + output ──────────────────────────────
    let t = std::time::Instant::now();
    let (cache, out) = model.prefix_pass(&embs).unwrap();
    let prefix_ms = t.elapsed().as_secs_f64() * 1e3;
    assert_eq!(cache.n, n);
    for layer in [0usize, 1, 15] {
        for (what, ours) in [
            ("keys", &cache.keys[layer]),
            ("values", &cache.values[layer]),
        ] {
            // Fixture layout [kv_heads, n_full, hd]: filter the padded rows per head.
            let full = fx(&f, &format!("prefix.kv.{layer}.{what}"));
            let per_head = full.len() / c.kv_heads;
            let want: Vec<f32> = full
                .chunks_exact(per_head)
                .flat_map(|h| keep_rows(h, c.head_dim, &keep))
                .collect();
            check(&format!("prefix layer {layer} {what}"), ours, &want, 5e-2);
        }
    }
    check(
        "prefix output",
        &out,
        &keep_rows(&fx(&f, "prefix.out"), c.vlm_hidden, &keep),
        5e-2,
    );

    // ── 4. suffix embedding at t = 1 ────────────────────────────────────────
    let suffix = model.embed_suffix(&noise, 1.0).unwrap();
    check(
        "suffix embedding (t=1)",
        &suffix,
        &fx(&f, "suffix.embs_t1"),
        1e-4,
    );

    // ── 5. one velocity ─────────────────────────────────────────────────────
    let t = std::time::Instant::now();
    let v = model.denoise_step(&cache, &noise, 1.0).unwrap();
    let step_ms = t.elapsed().as_secs_f64() * 1e3;
    check("velocity (t=1)", &v, &fx(&f, "denoise.v_t1"), 1e-4);

    // ── 6. full action chunk ────────────────────────────────────────────────
    let t = std::time::Instant::now();
    let actions = model.sample_actions(&cache, &noise).unwrap();
    let sample_ms = t.elapsed().as_secs_f64() * 1e3;
    check(
        "actions (10 Euler steps)",
        &actions,
        &fx(&f, "actions.normalized"),
        1e-4,
    );

    println!(
        "timing (f32, unquantized): vision {vision_ms:.0} ms · prefix {prefix_ms:.0} ms · \
         one denoise step {step_ms:.0} ms · {} steps {sample_ms:.0} ms",
        c.num_steps
    );
}

/// Action error and time of each Q8_0 choice, against the f32 LeRobot reference.
///
/// The yardstick is LeRobot's own default precision: it runs the VLM in bf16,
/// which moves the action chunk by `BF16_MAX` (max abs, measured with the same
/// inputs) from the f32 reference.
#[test]
#[ignore = "needs the lerobot/smolvla_base checkpoint: set SAPIENT_SMOLVLA_DIR"]
fn smolvla_quantized_action_error() {
    let dir = PathBuf::from(std::env::var("SAPIENT_SMOLVLA_DIR").expect("set SAPIENT_SMOLVLA_DIR"));
    let f = fixture();
    let lang_mask = fx(&f, "input.lang_mask");
    let lang: Vec<u32> = fx(&f, "input.lang_tokens")
        .iter()
        .zip(&lang_mask)
        .filter(|(_, m)| **m > 0.5)
        .map(|(t, _)| *t as u32)
        .collect();
    let (state, noise) = (fx(&f, "input.state"), fx(&f, "input.noise"));
    let want = fx(&f, "actions.normalized");

    let q = |vision, vlm, expert| SmolVlaQuant {
        vision,
        vlm,
        expert,
    };
    let configs = [
        ("f32", SmolVlaQuant::NONE),
        ("vision", q(true, false, false)),
        ("vlm", q(false, true, false)),
        ("expert", q(false, false, true)),
        ("vision+vlm", q(true, true, false)),
        ("all", SmolVlaQuant::ALL),
    ];
    for (name, quant) in configs {
        let weights = sapient_io::load_safetensors(&dir.join("model.safetensors")).unwrap();
        let model = SmolVla::from_weights_quant(SmolVlaConfig::default(), weights, quant).unwrap();
        let pixels = test_image(model.image_size());
        let mut best = [f64::MAX; 3];
        let mut actions = Vec::new();
        for _ in 0..3 {
            let t0 = std::time::Instant::now();
            let embs = model.embed_prefix(&[&pixels], &lang, &state).unwrap();
            let t1 = std::time::Instant::now();
            // embed_prefix includes the (tiny) language/state embedding.
            let cache = model.prefix_cache(&embs).unwrap();
            let t2 = std::time::Instant::now();
            actions = model.sample_actions(&cache, &noise).unwrap();
            let t3 = std::time::Instant::now();
            for (b, d) in best.iter_mut().zip([t1 - t0, t2 - t1, t3 - t2]) {
                *b = b.min(d.as_secs_f64() * 1e3);
            }
        }
        let (max, _) = err(&actions, &want);
        let rms = (actions
            .iter()
            .zip(&want)
            .map(|(a, w)| ((a - w) as f64).powi(2))
            .sum::<f64>()
            / want.len() as f64)
            .sqrt();
        println!(
            "{name:11} max_err {max:.3e}  rms {rms:.3e} | vision {:4.0} ms · prefix {:4.0} ms · \
             denoise {:4.0} ms · total {:4.0} ms",
            best[0],
            best[1],
            best[2],
            best.iter().sum::<f64>()
        );
        // Regression guards at 2× the measured values (f32 must stay exact).
        let (max_tol, rms_tol) = if quant == SmolVlaQuant::NONE {
            (1e-4, 1e-5)
        } else {
            (7e-2, 8e-3)
        };
        assert!(
            max <= max_tol && rms <= rms_tol,
            "{name}: action error max {max} rms {rms}"
        );

        // What fewer Euler steps cost (information only — a coarser integral
        // of the same flow, compared with the 10-step f32 reference).
        if quant == SmolVlaQuant::ALL {
            let embs = model.embed_prefix(&[&pixels], &lang, &state).unwrap();
            let cache = model.prefix_cache(&embs).unwrap();
            for steps in [5usize, 3] {
                let t = std::time::Instant::now();
                let a = model.sample_actions_steps(&cache, &noise, steps).unwrap();
                let ms = t.elapsed().as_secs_f64() * 1e3;
                let (max, _) = err(&a, &want);
                let rms = (a
                    .iter()
                    .zip(&want)
                    .map(|(a, w)| ((a - w) as f64).powi(2))
                    .sum::<f64>()
                    / want.len() as f64)
                    .sqrt();
                println!(
                    "all, {steps} steps  max_err {max:.3e}  rms {rms:.3e} | denoise {ms:4.0} ms"
                );
            }
        }
    }
}
