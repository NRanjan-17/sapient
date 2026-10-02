//! Pronunciation dictionaries and tagger weights.
//!
//! Vendored change (see VENDORED.md): upstream embeds 35.5 MB of JSON with
//! `include_str!`. Here every data file is gzip-compressed and resolved, in
//! order, from:
//!
//! 1. a directory registered with [`set_data_dir`] (files named `<name>.gz`,
//!    e.g. downloaded next to the TTS model), or
//! 2. the copy embedded in the binary — only with the `embed-dict` feature.
use crate::lexicon::PhonemeEntry;
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::RwLock;

/// File names (without the `.gz` suffix) of every data file a G2P needs.
pub const US_GOLD: &str = "us_gold.json";
pub const US_SILVER: &str = "us_silver.json";
pub const GB_GOLD: &str = "gb_gold.json";
pub const GB_SILVER: &str = "gb_silver.json";
pub const TAGGER_WEIGHTS: &str = "tagger_weights.json";

/// Every data file, as stored on disk / in a model repository.
pub const DATA_FILES: &[&str] = &[
    "us_gold.json.gz",
    "us_silver.json.gz",
    "gb_gold.json.gz",
    "gb_silver.json.gz",
    "tagger_weights.json.gz",
];

static DATA_DIR: RwLock<Option<PathBuf>> = RwLock::new(None);

/// Register the directory that holds the `*.json.gz` data files. Takes effect
/// for every G2P created afterwards.
pub fn set_data_dir(dir: impl Into<PathBuf>) {
    *DATA_DIR.write().unwrap() = Some(dir.into());
}

/// The registered data directory, if any.
pub fn data_dir() -> Option<PathBuf> {
    DATA_DIR.read().unwrap().clone()
}

/// Whether the dictionaries are compiled into this binary.
pub const fn embedded() -> bool {
    cfg!(feature = "embed-dict")
}

/// True when every data file can be resolved (embedded, or present in the
/// registered directory). Check this before `G2P::new`, which panics otherwise.
pub fn available() -> bool {
    embedded() || data_dir().is_some_and(|d| DATA_FILES.iter().all(|f| d.join(f).is_file()))
}

#[cfg(feature = "embed-dict")]
fn embedded_gz(name: &str) -> Option<&'static [u8]> {
    Some(match name {
        US_GOLD => include_bytes!("../data/us_gold.json.gz"),
        US_SILVER => include_bytes!("../data/us_silver.json.gz"),
        GB_GOLD => include_bytes!("../data/gb_gold.json.gz"),
        GB_SILVER => include_bytes!("../data/gb_silver.json.gz"),
        TAGGER_WEIGHTS => include_bytes!("resources/tagger/weights.json.gz"),
        _ => return None,
    })
}

#[cfg(not(feature = "embed-dict"))]
fn embedded_gz(_name: &str) -> Option<&'static [u8]> {
    None
}

fn gunzip(bytes: &[u8], what: &str) -> String {
    let mut out = String::new();
    flate2::read::GzDecoder::new(bytes)
        .read_to_string(&mut out)
        .unwrap_or_else(|e| panic!("misaki-rs: {what} is not valid gzip UTF-8: {e}"));
    out
}

fn read_external(dir: &Path, name: &str) -> Option<String> {
    let path = dir.join(format!("{name}.gz"));
    let bytes = std::fs::read(&path).ok()?;
    Some(gunzip(&bytes, &path.display().to_string()))
}

/// The decompressed text of one data file.
pub fn load_text(name: &str) -> String {
    if let Some(dir) = data_dir() {
        if let Some(text) = read_external(&dir, name) {
            return text;
        }
    }
    if let Some(gz) = embedded_gz(name) {
        return gunzip(gz, name);
    }
    panic!(
        "misaki-rs: data file `{name}.gz` not found — call data::set_data_dir() with a \
         directory containing {DATA_FILES:?}, or build with the `embed-dict` feature"
    );
}

fn load_dict(name: &str) -> HashMap<String, PhonemeEntry> {
    serde_json::from_str(&load_text(name)).unwrap_or_else(|e| panic!("Failed to parse {name}: {e}"))
}

pub fn load_us_gold() -> HashMap<String, PhonemeEntry> {
    load_dict(US_GOLD)
}

pub fn load_us_silver() -> HashMap<String, PhonemeEntry> {
    load_dict(US_SILVER)
}

pub fn load_gb_gold() -> HashMap<String, PhonemeEntry> {
    load_dict(GB_GOLD)
}

pub fn load_gb_silver() -> HashMap<String, PhonemeEntry> {
    load_dict(GB_SILVER)
}
