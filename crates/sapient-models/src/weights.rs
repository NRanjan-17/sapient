// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! HuggingFace safetensors weight loading and key resolution.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use sapient_core::Tensor;
use sapient_io::SafetensorsLoader;

/// Load and merge safetensors shards from disk.
pub fn load_hf_weights(paths: &[PathBuf]) -> Result<HashMap<String, Tensor>> {
    load_hf_weights_map(paths, |_, t| t)
}

/// Load and merge safetensors shards, quantizing each F16/BF16 linear weight
/// to Q8_0 as it is read (the same `should_quantize_online` rule and
/// `quantize_tensor_to_q8_0` call the engines apply after loading, so the
/// result is byte-identical and their own pass becomes a no-op).
///
/// The difference is peak memory: the whole BF16 checkpoint never sits in
/// memory next to its Q8_0 copy. For SmolLM2-1.7B that is ~1.8 GB instead of
/// 3.4 GB + 1.8 GB, which decides whether it loads under a phone's per-app
/// limit at all.
pub fn load_hf_weights_quantized(paths: &[PathBuf]) -> Result<HashMap<String, Tensor>> {
    use crate::forward::common::{quantize_tensor_to_q8_0, should_quantize_online};
    load_hf_weights_map(paths, |name, t| {
        if should_quantize_online(name, &t) {
            quantize_tensor_to_q8_0(t)
        } else {
            t
        }
    })
}

fn load_hf_weights_map<F>(paths: &[PathBuf], mut transform: F) -> Result<HashMap<String, Tensor>>
where
    F: FnMut(&str, Tensor) -> Tensor,
{
    let mut merged = HashMap::new();
    for path in paths {
        let shard = SafetensorsLoader::load_map(path, &mut transform)
            .with_context(|| format!("failed to load weights from {}", path.display()))?;
        for (k, v) in shard {
            if merged.insert(k.clone(), v).is_some() {
                bail!("duplicate weight key '{k}' in shard {}", path.display());
            }
        }
    }
    Ok(merged)
}

/// Detect the common prefix for transformer weight keys.
pub fn detect_weight_prefix(weights: &HashMap<String, Tensor>) -> String {
    const CANDIDATES: &[&str] = &[
        "model.text_model.",
        "model.language_model.",
        "transformer.",
        "model.",
        "gpt_neox.",
    ];

    for prefix in CANDIDATES {
        let embed_key = format!("{prefix}embed_tokens.weight");
        if weights.contains_key(&embed_key) {
            return prefix.to_string();
        }
    }

    if weights.contains_key("embed_tokens.weight") {
        return String::new();
    }

    // Fall back: find any embed_tokens key.
    weights
        .keys()
        .find(|k| k.ends_with("embed_tokens.weight"))
        .map(|k| {
            k.strip_suffix("embed_tokens.weight")
                .unwrap_or("")
                .to_string()
        })
        .unwrap_or_else(|| "model.".to_string())
}

/// Resolve a weight tensor by logical suffix (e.g. `layers.0.self_attn.q_proj`).
pub fn resolve_weight<'a>(
    weights: &'a HashMap<String, Tensor>,
    prefix: &str,
    suffix: &str,
) -> Result<&'a Tensor> {
    let key = format!("{prefix}{suffix}.weight");
    weights
        .get(&key)
        .or_else(|| weights.get(suffix))
        .with_context(|| format!("missing weight '{key}'"))
}

/// Resolve an optional bias tensor by logical suffix (e.g. `layers.0.self_attn.q_proj`).
/// Returns `None` when the model has no bias for that layer (e.g. Llama/Mistral).
pub fn resolve_bias<'a>(
    weights: &'a HashMap<String, Tensor>,
    prefix: &str,
    suffix: &str,
) -> Option<&'a Tensor> {
    let key = format!("{prefix}{suffix}.bias");
    weights
        .get(&key)
        .or_else(|| weights.get(&format!("{suffix}.bias")))
}

/// Resolve lm_head — may live outside the model prefix.
pub fn resolve_lm_head<'a>(
    weights: &'a HashMap<String, Tensor>,
    prefix: &str,
    tie_word_embeddings: bool,
    embed_key: &str,
) -> Result<&'a Tensor> {
    if tie_word_embeddings {
        return weights
            .get(embed_key)
            .with_context(|| format!("missing tied embedding weight '{embed_key}'"));
    }

    weights
        .get("lm_head.weight")
        .or_else(|| weights.get(&format!("{prefix}lm_head.weight")))
        // Tied embeddings: GGUF metadata has no `tie_word_embeddings` flag, so a
        // model that ties its output projection to the input embedding (SmolLM2,
        // Llama-3.2-1B/3B, Qwen small) simply omits `output.weight`. When no
        // explicit head exists, fall back to the embedding matrix.
        .or_else(|| weights.get(embed_key))
        .with_context(|| format!("missing lm_head.weight (and no '{embed_key}' to tie to)"))
}

pub fn tie_word_embeddings_from_config(raw: &serde_json::Value) -> bool {
    raw.get("tie_word_embeddings")
        .and_then(|v| v.as_bool())
        .or_else(|| {
            raw.get("text_config")
                .and_then(|tc| tc.get("tie_word_embeddings"))
                .and_then(|v| v.as_bool())
        })
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_text_model_prefix() {
        let mut w = HashMap::new();
        w.insert(
            "model.text_model.embed_tokens.weight".into(),
            Tensor::zeros(vec![1, 1], sapient_core::DType::F32).unwrap(),
        );
        assert_eq!(detect_weight_prefix(&w), "model.text_model.");
    }

    /// Writes a minimal BF16 safetensors file: one quantizable linear weight,
    /// one norm and one embedding table (the last two must stay BF16).
    fn write_bf16_checkpoint(path: &std::path::Path) {
        let tensors: [(&str, [usize; 2]); 3] = [
            ("model.layers.0.self_attn.q_proj.weight", [64, 96]),
            ("model.layers.0.input_layernorm.weight", [1, 64]),
            ("model.embed_tokens.weight", [40, 64]),
        ];
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (i, (name, shape)) in tensors.iter().enumerate() {
            let start = data.len();
            for j in 0..shape[0] * shape[1] {
                // Deterministic, varied values; BF16 = the top half of an f32.
                let v = ((j * 37 + i * 11) % 251) as f32 / 97.0 - 1.3;
                data.extend_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes());
            }
            header.insert(
                (*name).into(),
                serde_json::json!({ "dtype": "BF16", "shape": shape, "data_offsets": [start, data.len()] }),
            );
        }
        let header = serde_json::to_vec(&header).unwrap();
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(&header);
        file.extend_from_slice(&data);
        std::fs::write(path, file).unwrap();
    }

    #[test]
    fn quantized_load_is_byte_identical_to_load_then_quantize() {
        use crate::forward::common::{quantize_tensor_to_q8_0, should_quantize_online};
        let path = std::env::temp_dir().join(format!(
            "sapient-q8-load-{}.safetensors",
            std::process::id()
        ));
        write_bf16_checkpoint(&path);
        let paths = vec![path.clone()];

        // What the engines do today: load everything, then quantize.
        let reference: HashMap<String, Tensor> = load_hf_weights(&paths)
            .unwrap()
            .into_iter()
            .map(|(k, v)| {
                let v = if should_quantize_online(&k, &v) {
                    quantize_tensor_to_q8_0(v)
                } else {
                    v
                };
                (k, v)
            })
            .collect();
        let streamed = load_hf_weights_quantized(&paths).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(reference.len(), streamed.len());
        for (name, want) in &reference {
            let got = &streamed[name];
            assert_eq!(got.dtype(), want.dtype(), "{name}");
            assert_eq!(got.shape().dims(), want.shape().dims(), "{name}");
            assert_eq!(got.as_bytes(), want.as_bytes(), "{name}");
        }
        assert_eq!(
            streamed["model.layers.0.self_attn.q_proj.weight"].dtype(),
            sapient_core::DType::Q8_0
        );
        assert_eq!(
            streamed["model.embed_tokens.weight"].dtype(),
            sapient_core::DType::BF16
        );
        assert_eq!(
            streamed["model.layers.0.input_layernorm.weight"].dtype(),
            sapient_core::DType::BF16
        );
    }
}
