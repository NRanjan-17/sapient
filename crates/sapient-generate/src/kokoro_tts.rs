// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! `KokoroTts` — a real-time [`Tts`] backend powered by Kokoro-82M.
//!
//! Kokoro is a *non-autoregressive* StyleTTS2 + ISTFTNet model (see
//! [`sapient_models::KokoroModel`]): one forward pass turns a phoneme sequence +
//! a voice style vector into a 24 kHz waveform, with **no codec-token decode
//! loop**. That removes the autoregressive tokens-per-second ceiling that makes
//! Orpheus-3B ~0.18× real-time, so Kokoro is the path to glitch-free real-time
//! `sapient converse` voice replies.
//!
//! Text → phonemes uses the pure-Rust [`misaki_rs`] G2P (dictionary-first, the
//! same front-end Kokoro was trained with), built without the optional espeak-ng
//! fallback so the whole path stays FFI-free.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{anyhow, Context, Result};
use misaki_rs::language::Language;
use misaki_rs::G2P;
use sapient_hub::HubClient;
use sapient_models::{DecoderStreamInputs, KokoroModel, KOKORO_SAMPLE_RATE};

use crate::converse::Tts;

fn lang(british: bool) -> Language {
    if british {
        Language::EnglishGB
    } else {
        Language::EnglishUS
    }
}

/// The default Kokoro voice (American English, female — the arena-favourite
/// `af_heart`). Override per-construction with [`KokoroTts::with_voice`].
pub const DEFAULT_KOKORO_VOICE: &str = "af_heart";

/// The default Hugging Face repo holding the converted Kokoro-82M safetensors
/// (`config.json`, `model.safetensors`, `voices.safetensors`). Upstream
/// `hexgrad/Kokoro-82M` ships only a PyTorch pickle, so SAPIENT pulls this
/// converted mirror (`scripts/convert_kokoro_to_safetensors.py` output) — the
/// same pattern the SNAC codec uses for its `mlx-community` mirror.
pub const KOKORO_REPO: &str = "sai1974dev/kokoro-82m-safetensors";

/// Sub-directory (of the model repo, or of a local model dir) that holds the
/// G2P data: the gzip-compressed pronunciation dictionaries and tagger weights
/// listed in `misaki_rs::data::DATA_FILES`. They used to be compiled into the
/// binary (35.5 MB of JSON); they are now fetched with the model unless the
/// `embed-g2p` feature is on.
pub const KOKORO_G2P_DIR: &str = "g2p";

/// Point the G2P at `dir/g2p` when it holds the data files. Returns whether
/// the G2P data is now resolvable (from that directory or embedded).
fn use_g2p_dir(dir: &Path) -> bool {
    let g2p = dir.join(KOKORO_G2P_DIR);
    if misaki_rs::data::DATA_FILES
        .iter()
        .all(|f| g2p.join(f).is_file())
    {
        misaki_rs::data::set_data_dir(g2p);
    }
    misaki_rs::data::available()
}

/// Download the G2P data files from `repo` into the Hub cache and register them.
async fn download_g2p(hub: &HubClient, repo: &str) -> Result<()> {
    let names: Vec<String> = misaki_rs::data::DATA_FILES
        .iter()
        .map(|f| format!("{KOKORO_G2P_DIR}/{f}"))
        .collect();
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    let files = hub.download_files(repo, &refs).await.with_context(|| {
        format!(
            "downloading Kokoro pronunciation data from '{repo}' (needed once; \
             build with the `embed-g2p` feature for a fully offline binary)"
        )
    })?;
    let dir = files[0]
        .parent()
        .ok_or_else(|| anyhow!("kokoro: g2p download has no parent dir"))?;
    misaki_rs::data::set_data_dir(dir);
    Ok(())
}

/// A Kokoro-82M text-to-speech backend.
pub struct KokoroTts {
    model: KokoroModel,
    // misaki's G2P holds mutable internal state; the `Tts` trait takes `&self`,
    // so guard it. G2P is cheap relative to synthesis, so the lock is not a
    // throughput concern.
    g2p: Mutex<G2P>,
    british: bool,
    voice: String,
    speed: f32,
}

impl KokoroTts {
    /// Load from a directory of converted weights (`config.json`,
    /// `model.safetensors`, `voices.safetensors`) — the output of
    /// `scripts/convert_kokoro_to_safetensors.py`. American English by default.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let model = KokoroModel::from_dir(dir)
            .with_context(|| format!("load Kokoro model from {dir:?}"))?;
        if !use_g2p_dir(dir) {
            anyhow::bail!(
                "Kokoro pronunciation data not found: expected {:?} in {:?}. Use \
                 `KokoroTts::from_pretrained` (downloads it), copy the files there, or \
                 build with the `embed-g2p` feature",
                misaki_rs::data::DATA_FILES,
                dir.join(KOKORO_G2P_DIR)
            );
        }
        Ok(Self {
            model,
            g2p: Mutex::new(G2P::new(lang(false))),
            british: false,
            voice: DEFAULT_KOKORO_VOICE.to_string(),
            speed: 1.0,
        })
    }

    /// Load from the Hub: a local dir (`SAPIENT_KOKORO_DIR`) if set, else download
    /// `config.json` + `model.safetensors` + `voices.safetensors` from `repo`.
    pub async fn from_pretrained(repo: &str) -> Result<Self> {
        if let Ok(dir) = std::env::var("SAPIENT_KOKORO_DIR") {
            let dir = Path::new(&dir);
            // A local weights dir usually has no G2P data: fetch it from the
            // mirror unless it is already there or embedded.
            if !use_g2p_dir(dir) {
                download_g2p(&HubClient::new()?, repo).await?;
            }
            return Self::from_dir(dir);
        }
        let hub = HubClient::new()?;
        let files: Vec<PathBuf> = hub
            .download_files(
                repo,
                &["config.json", "model.safetensors", "voices.safetensors"],
            )
            .await
            .with_context(|| format!("downloading Kokoro-82M '{repo}'"))?;
        let dir = files[0]
            .parent()
            .ok_or_else(|| anyhow!("kokoro: download has no parent dir"))?;
        if !use_g2p_dir(dir) {
            download_g2p(&hub, repo).await?;
        }
        Self::from_dir(dir)
    }

    /// Load the default converted mirror ([`KOKORO_REPO`]).
    pub async fn from_default() -> Result<Self> {
        Self::from_pretrained(KOKORO_REPO).await
    }

    /// Select the voice (must be present in `voices.safetensors`).
    pub fn with_voice(mut self, voice: impl Into<String>) -> Self {
        self.voice = voice.into();
        self
    }

    /// Select British (`true`) vs American (`false`) English G2P.
    pub fn with_british(mut self, british: bool) -> Self {
        if british != self.british {
            self.g2p = Mutex::new(G2P::new(lang(british)));
            self.british = british;
        }
        self
    }

    /// Speaking-rate multiplier (1.0 = normal; >1 faster).
    pub fn with_speed(mut self, speed: f32) -> Self {
        self.speed = speed;
        self
    }

    /// The underlying Kokoro model (streaming probes/tools).
    pub fn model(&self) -> &sapient_models::forward::kokoro::KokoroModel {
        &self.model
    }

    /// The voice this backend will speak with.
    pub fn voice(&self) -> &str {
        &self.voice
    }

    /// Synthesize `text` and write it to a 24 kHz WAV at `path`. Returns the
    /// sample count.
    pub fn speak_to_wav(&self, text: &str, path: &Path) -> Result<usize> {
        let audio = self.synthesize(text)?;
        sapient_audio::write_wav(path, &audio, self.sample_rate())
            .with_context(|| format!("write {path:?}"))?;
        Ok(audio.len())
    }

    /// Synthesize `text` in an explicit `voice` and `speed`, overriding the
    /// instance defaults.
    ///
    /// Every voice embedding is already resident in the loaded model, so one
    /// cached engine can serve all of them — this is what lets the HTTP TTS
    /// route honor a per-request `voice` without reloading weights per caller.
    pub fn synthesize_as(&self, text: &str, voice: &str, speed: f32) -> Result<Vec<f32>> {
        let text = text.trim();
        if text.is_empty() {
            return Ok(Vec::new());
        }
        let phonemes = self.phonemize(text)?;
        self.model.synthesize(&phonemes, voice, speed)
    }

    /// Grapheme-to-phoneme for `text` (IPA string in Kokoro's inventory).
    pub fn phonemize(&self, text: &str) -> Result<String> {
        let (phonemes, _tokens) = self
            .g2p
            .lock()
            .unwrap()
            .g2p(text)
            .map_err(|e| anyhow!("kokoro G2P failed: {e:?}"))?;
        Ok(phonemes)
    }

    /// Streaming decoder-only path (duplex spike, gate 2): run the amortizable
    /// backbone once for `text`, returning the decoder inputs. Decode time-slices
    /// of it with [`Self::decode_prefix`].
    pub fn prepare_stream(&self, text: &str) -> Result<DecoderStreamInputs> {
        let phonemes = self.phonemize(text.trim())?;
        self.model
            .prepare_stream_phonemes(&phonemes, &self.voice, self.speed)
    }

    /// Decode only the first `frames` decoder time-steps of a prepared utterance
    /// (the convolutional ISTFTNet decoder, the ~80% cost), returning the prefix
    /// waveform. `frames` is clamped to the prepared length.
    pub fn decode_prefix(&self, inp: &DecoderStreamInputs, frames: usize) -> Result<Vec<f32>> {
        self.model.decode_prefix(inp, frames)
    }
}

impl Tts for KokoroTts {
    fn synthesize(&self, text: &str) -> Result<Vec<f32>> {
        let text = text.trim();
        if text.is_empty() {
            return Ok(Vec::new());
        }
        let phonemes = self.phonemize(text)?;
        self.model.synthesize(&phonemes, &self.voice, self.speed)
    }

    fn sample_rate(&self) -> u32 {
        KOKORO_SAMPLE_RATE
    }
}
