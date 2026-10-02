// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! `VlaPipeline` end to end against the LeRobot reference: download
//! `lerobot/smolvla_base` + the SmolVLM2 tokenizer, preprocess a camera frame,
//! tokenize the instruction, and reproduce the reference action chunk from the
//! same state and start noise (fixture written by
//! `scripts/gen_smolvla_fixture.py`).
//!
//! Ignored: downloads ~0.9 GB.
//! `cargo test -p sapient-generate --release --test vla_e2e -- --ignored --nocapture`

use std::path::PathBuf;

use sapient_generate::{SmolVlaQuant, VlaPipeline, SMOLVLA_REPO};

const TASK: &str = "Pick up the red cube and place it in the box.";

/// The generator's procedural camera frame (`test_image` in the script).
fn test_frame(size: u32) -> image::RgbImage {
    image::RgbImage::from_fn(size, size, |x, y| {
        if (160..352).contains(&y) && (144..368).contains(&x) {
            image::Rgb([217, 26, 26])
        } else {
            image::Rgb([
                (x * 255 / size) as u8,
                (y * 255 / size) as u8,
                ((x * 7 + y * 13) % 256) as u8,
            ])
        }
    })
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "downloads lerobot/smolvla_base (~0.9 GB)"]
async fn smolvla_pipeline_reproduces_reference_actions() {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../sapient-models/tests/fixtures/smolvla_base.safetensors");
    let fx = sapient_io::load_safetensors(&fixture).expect("load SmolVLA fixture");
    let get = |name: &str| fx[name].to_f32_vec();

    // The exact f32 path; the default (Q8_0) is checked at the end.
    let vla = VlaPipeline::from_pretrained_with(SMOLVLA_REPO, SmolVlaQuant::NONE)
        .await
        .expect("load SmolVLA");
    assert_eq!(
        (vla.state_dim(), vla.action_dim(), vla.chunk_len()),
        (6, 6, 50)
    );

    // Tokenization must match the reference processor exactly.
    let want_ids: Vec<u32> = get("input.lang_tokens")
        .iter()
        .zip(get("input.lang_mask"))
        .filter(|(_, m)| *m > 0.5)
        .map(|(t, _)| *t as u32)
        .collect();
    assert_eq!(vla.tokenize(TASK).unwrap(), want_ids);

    let pixels = vla.preprocess_rgb(&test_frame(512));
    let state = get("input.state");
    let chunk = vla
        .predict_with_noise(&[pixels], TASK, &state[..6], &get("input.noise"))
        .expect("predict");
    assert_eq!((chunk.steps, chunk.dim), (50, 6));
    // smolvla_base ships no plain action statistics → normalized space.
    assert!(!chunk.robot_units);

    let want = get("actions.normalized");
    let mut max_err = 0.0f32;
    for step in 0..chunk.steps {
        for (g, w) in chunk.row(step).iter().zip(&want[step * 32..step * 32 + 6]) {
            max_err = max_err.max((g - w).abs());
        }
    }
    let t = &chunk.timing;
    println!(
        "actions max_err {max_err:.3e} · prefix {} tokens · vision {:.0} ms · prefix {:.0} ms · \
         denoise {:.0} ms · total {:.0} ms",
        t.prefix_tokens, t.vision_ms, t.prefix_ms, t.denoise_ms, t.total_ms
    );
    assert!(
        max_err < 1e-4,
        "actions diverge from the reference: {max_err}"
    );

    // A seeded run is reproducible.
    let px = vla.preprocess_rgb(&test_frame(512));
    let a = vla
        .predict(std::slice::from_ref(&px), TASK, &state[..6], 42)
        .unwrap();
    let b = vla.predict(&[px], TASK, &state[..6], 42).unwrap();
    assert_eq!(a.actions, b.actions);
    drop(vla);

    // Default precision (Q8_0 linears): bounded action error against the f32
    // reference. Measured 3.5e-2 max; LeRobot's own bf16 default moves the
    // same chunk by 1.4e-2. The bound is a regression guard, not a target.
    let q8 = VlaPipeline::from_pretrained(SMOLVLA_REPO).await.unwrap();
    let px = q8.preprocess_rgb(&test_frame(512));
    let chunk = q8
        .predict_with_noise(&[px], TASK, &state[..6], &get("input.noise"))
        .unwrap();
    let mut q_err = 0.0f32;
    for step in 0..chunk.steps {
        for (g, w) in chunk.row(step).iter().zip(&want[step * 32..step * 32 + 6]) {
            q_err = q_err.max((g - w).abs());
        }
    }
    println!(
        "Q8_0 default: actions max_err {q_err:.3e} · total {:.0} ms",
        chunk.timing.total_ms
    );
    assert!(q_err < 7e-2, "Q8_0 action error grew: {q_err}");
}
