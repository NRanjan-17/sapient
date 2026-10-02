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

/// Sapient on REAL robot observations: 24 frames (two cameras, 480×640) from
/// the SO-100 dataset `lerobot/svla_so100_pickplace`, against LeRobot's own
/// f32 and bf16 outputs and the recorded actions. The file is written by
/// `scripts/smolvla_dataset_eval.py` (not in the repo).
///
/// `SAPIENT_SMOLVLA_DATASET=<file>` (default
/// `~/.cache/sapient-bench/smolvla_so100.safetensors`).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs lerobot/smolvla_base and the dataset file from scripts/smolvla_dataset_eval.py"]
async fn smolvla_real_observations() {
    use sapient_generate::VlaPrecision;
    const DATASET_TASK: &str = "Pick up the cube and place it in the box.";
    let path = std::env::var("SAPIENT_SMOLVLA_DATASET")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").unwrap())
                .join(".cache/sapient-bench/smolvla_so100.safetensors")
        });
    let fx = sapient_io::load_safetensors(&path).expect("load dataset file");
    let get = |name: &str| fx[name].to_f32_vec();
    let n = (0..)
        .take_while(|i| fx.contains_key(&format!("obs.{i}.gt")))
        .count();
    assert!(n > 0, "no observations in {path:?}");

    let frame = |name: &str| -> image::RgbImage {
        let t = &fx[name];
        let d = t.shape().dims().to_vec(); // [H, W, 3]
        let px: Vec<u8> = t.to_f32_vec().iter().map(|v| *v as u8).collect();
        image::RgbImage::from_raw(d[1] as u32, d[0] as u32, px).unwrap()
    };
    let rms = |a: &[f32], b: &[f32]| -> f64 {
        (a.iter()
            .zip(b)
            .map(|(x, y)| ((x - y) as f64).powi(2))
            .sum::<f64>()
            / a.len() as f64)
            .sqrt()
    };
    let max_abs = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
    };
    // First 6 dims of [50, 32] rows.
    let trim =
        |v: Vec<f32>| -> Vec<f32> { v.chunks_exact(32).flat_map(|r| r[..6].to_vec()).collect() };

    let mut lerobot_f32 = Vec::new();
    let mut lerobot_bf16 = Vec::new();
    let mut gt = Vec::new();
    for i in 0..n {
        lerobot_f32.extend(trim(get(&format!("obs.{i}.lerobot_f32"))));
        lerobot_bf16.extend(trim(get(&format!("obs.{i}.lerobot_bf16"))));
        gt.extend(get(&format!("obs.{i}.gt")));
    }
    println!(
        "{n} real observations · recorded action RMS {:.3} (dataset-normalized)",
        rms(&gt, &vec![0.0; gt.len()])
    );
    println!(
        "{:28} vs LeRobot f32: rms {:.3e} max {:.3e} · vs recorded: rms {:.3}",
        "LeRobot bf16 (yardstick)",
        rms(&lerobot_bf16, &lerobot_f32),
        max_abs(&lerobot_bf16, &lerobot_f32),
        rms(&lerobot_bf16, &gt)
    );
    println!(
        "{:28} vs recorded: rms {:.3}",
        "LeRobot f32",
        rms(&lerobot_f32, &gt)
    );

    for precision in [
        VlaPrecision::Exact,
        VlaPrecision::Balanced,
        VlaPrecision::Fast,
    ] {
        let vla = VlaPipeline::from_pretrained_with(SMOLVLA_REPO, precision.quant())
            .await
            .expect("load SmolVLA");
        let want_ids: Vec<u32> = get("obs.0.lang").iter().map(|v| *v as u32).collect();
        assert_eq!(
            vla.tokenize(DATASET_TASK).unwrap(),
            want_ids,
            "tokenization"
        );
        let mut ours = Vec::new();
        let mut total_ms = 0.0;
        for i in 0..n {
            let images = vec![
                vla.preprocess_rgb(&frame(&format!("obs.{i}.top"))),
                vla.preprocess_rgb(&frame(&format!("obs.{i}.wrist"))),
            ];
            let chunk = vla
                .predict_with_noise(
                    &images,
                    DATASET_TASK,
                    &get(&format!("obs.{i}.state")),
                    &get(&format!("obs.{i}.noise")),
                )
                .unwrap();
            total_ms += chunk.timing.total_ms;
            ours.extend(chunk.actions);
        }
        let (r, m) = (rms(&ours, &lerobot_f32), max_abs(&ours, &lerobot_f32));
        println!(
            "{:28} vs LeRobot f32: rms {r:.3e} max {m:.3e} · vs recorded: rms {:.3} · {:.0} ms/chunk",
            format!("Sapient {}", precision.as_str()),
            rms(&ours, &gt),
            total_ms / n as f64
        );
        // Guards: exact must reproduce LeRobot on real frames (the resize path
        // and two cameras included); the quantized modes are bounded at 2× the
        // values measured on 2026-10-02 (balanced 1.38e-2 / 0.166, fast
        // 1.90e-2 / 0.157).
        let (rms_tol, max_tol) = match precision {
            VlaPrecision::Exact => (1e-4, 1e-3),
            VlaPrecision::Balanced => (2.8e-2, 0.33),
            VlaPrecision::Fast => (3.8e-2, 0.31),
        };
        assert!(
            r <= rms_tol && m <= max_tol,
            "{precision:?}: rms {r} max {m}"
        );
    }
}
