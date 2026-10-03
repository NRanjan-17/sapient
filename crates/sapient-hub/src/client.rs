// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! HuggingFace Hub API client.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use hf_hub::api::tokio::{Api, ApiBuilder, ApiRepo};
use tokio::sync::Semaphore;
use tracing::debug;

use crate::download::{configure_api_builder, max_parallel_downloads};
use crate::gguf::select_best_gguf;
use crate::model_info::ModelInfo;
use crate::resolver::ModelFiles;

// ── LoadOptions ───────────────────────────────────────────────────────────────

/// Options for model loading from the Hub.
#[derive(Debug, Clone)]
pub struct LoadOptions {
    /// HuggingFace access token. If `None`, reads `HF_TOKEN` env var, then
    /// `~/.cache/huggingface/token`.
    pub token: Option<String>,

    /// Preferred weight format, in priority order.
    /// Defaults to `["gguf", "safetensors", "bin"]`.
    pub formats: Vec<String>,

    /// If true, always re-download even if cached.
    pub force_download: bool,

    /// Maximum file size to download in bytes (0 = unlimited).
    pub max_size_bytes: u64,

    /// Model revision / branch (default: `"main"`).
    pub revision: String,

    /// Hide HuggingFace Hub progress bars (filenames, byte counts on stderr).
    pub quiet: bool,

    /// Parallel HTTP range downloads + concurrent shard fetches (recommended).
    ///
    /// Disable with `SAPIENT_FAST_DOWNLOAD=0`. Tune workers with
    /// `SAPIENT_HUB_MAX_PARALLEL` (default: min(CPU cores, 8)).
    pub fast_download: bool,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            token: None,
            formats: vec!["gguf".into(), "safetensors".into(), "bin".into()],
            force_download: false,
            max_size_bytes: 0,
            revision: "main".into(),
            quiet: false,
            fast_download: true,
        }
    }
}

// ── Cache location ───────────────────────────────────────────────────────────

/// The Hub cache directory, exactly where `HubClient` downloads to:
/// `$HF_HOME/hub` when `HF_HOME` is set (what `sapient-ffi`'s `set_cache_dir`
/// sets on iOS/Android), else `~/.cache/huggingface/hub`. Anything that looks
/// for downloaded files must use this; hard-coding the home-dir path watched
/// the wrong folder on iOS and had no answer on Android (no home directory).
pub fn hub_cache_dir() -> Option<PathBuf> {
    hub_cache_dir_from(
        std::env::var_os("HF_HOME").map(PathBuf::from),
        dirs::home_dir(),
    )
}

fn hub_cache_dir_from(hf_home: Option<PathBuf>, home: Option<PathBuf>) -> Option<PathBuf> {
    match hf_home {
        Some(hf_home) if !hf_home.as_os_str().is_empty() => Some(hf_home.join("hub")),
        _ => Some(home?.join(".cache/huggingface/hub")),
    }
}

/// The Hub's model-info endpoint WITH per-file sizes. Without `?blobs=true`
/// the API returns `siblings[].size = null`, which made `repo_total_bytes`
/// report 0 for every model (no download percentage anywhere).
fn model_api_url_with_sizes(repo_id: &str) -> String {
    format!("https://huggingface.co/api/models/{repo_id}?blobs=true")
}

/// Files of `repo_id`'s cached `main` snapshot (paths relative to it), or
/// empty if the repo was never downloaded. Layout:
/// `<hub>/models--org--name/refs/main` holds the commit hash, and
/// `snapshots/<hash>/` holds the files (symlinks into `blobs/`).
fn cached_repo_files(hub: &std::path::Path, repo_id: &str) -> Vec<String> {
    let repo = hub.join(format!("models--{}", repo_id.replace('/', "--")));
    let Ok(commit) = std::fs::read_to_string(repo.join("refs/main")) else {
        return Vec::new();
    };
    let snapshot = repo.join("snapshots").join(commit.trim());
    let mut files = Vec::new();
    let mut pending = vec![snapshot.clone()];
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
            let path = entry.path();
            if path.is_dir() {
                pending.push(path);
            } else if let Ok(relative) = path.strip_prefix(&snapshot) {
                files.push(relative.to_string_lossy().replace('\\', "/"));
            }
        }
    }
    files
}

/// Total size of the regular files under `path` (0 if it doesn't exist).
/// Symlinks are not followed, so a repo's `snapshots/` links don't double-count
/// its `blobs/`.
pub fn dir_bytes(path: &std::path::Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(path) else {
        return 0;
    };
    entries
        .flatten()
        .map(|entry| match entry.file_type() {
            Ok(t) if t.is_dir() => dir_bytes(&entry.path()),
            Ok(t) if t.is_file() => entry.metadata().map(|m| m.len()).unwrap_or(0),
            _ => 0,
        })
        .sum()
}

// ── HubClient ─────────────────────────────────────────────────────────────────

/// Client for the HuggingFace Hub REST API.
pub struct HubClient {
    api: Api,
    opts: LoadOptions,
}

impl HubClient {
    /// Create a new client, auto-reading the HF token from the environment.
    pub fn new() -> Result<Self> {
        Self::with_options(LoadOptions::default())
    }

    /// Create a new client with custom options.
    pub fn with_options(opts: LoadOptions) -> Result<Self> {
        let token = opts
            .token
            .clone()
            .or_else(|| std::env::var("HF_TOKEN").ok())
            .or_else(Self::read_cached_token);

        // from_env, NOT new: `new()` hard-codes `Cache::default()`, which
        // ignores HF_HOME and panics ("Cache directory cannot be found") on
        // platforms with no home dir — Android app processes. from_env()
        // honors HF_HOME, which is what `sapient-ffi`'s set_cache_dir and the
        // mobile docs promise. (iOS/desktop only worked because a home dir
        // exists there — HF_HOME was silently ignored.)
        let builder = ApiBuilder::from_env();
        let builder = if let Some(t) = token {
            builder.with_token(Some(t))
        } else {
            builder
        };
        let builder = configure_api_builder(builder, &opts);
        let api = builder
            .build()
            .context("Failed to build HF Hub API client")?;
        Ok(Self { api, opts })
    }

    /// Download a model by its HuggingFace model ID (e.g. `"meta-llama/Llama-3.2-1B"`).
    ///
    /// Returns the resolved local file paths — no Python, no git-lfs required.
    pub async fn download(&self, model_alias: &str) -> Result<ModelFiles> {
        let actual_repo = crate::registry::resolve_model_alias(model_alias)?;
        debug!("Downloading model: {model_alias} (resolved to {actual_repo})");
        let repo = self.api.model(actual_repo.clone());
        let mut files = match self.resolve_files(&repo, &actual_repo).await {
            Ok(f) => f,
            Err(e) => {
                if self.opts.fast_download {
                    eprintln!(
                        "Warning: Fast download failed ({}). Retrying in safe mode...",
                        e
                    );
                    let mut safe_opts = self.opts.clone();
                    safe_opts.fast_download = false;
                    let safe_client = Self::with_options(safe_opts)?;
                    let safe_repo = safe_client.api.model(actual_repo.clone());
                    safe_client.resolve_files(&safe_repo, &actual_repo).await?
                } else {
                    return Err(e);
                }
            }
        };
        files.model_id = model_alias.to_owned();
        Ok(files)
    }

    /// Download specific `files` from a HuggingFace repo by its **raw repo id**,
    /// bypassing the curated registry. Used for auxiliary assets that aren't
    /// chat models — e.g. the SNAC codec weights (`config.json` +
    /// `model.safetensors`) that the Orpheus TTS path decodes with. Returns the
    /// local cached paths in the same order as `files`.
    pub async fn download_files(&self, repo_id: &str, files: &[&str]) -> Result<Vec<PathBuf>> {
        let repo = self.api.model(repo_id.to_string());
        let mut out = Vec::with_capacity(files.len());
        for f in files {
            let path = repo
                .get(f)
                .await
                .with_context(|| format!("downloading `{f}` from `{repo_id}`"))?;
            out.push(path);
        }
        Ok(out)
    }

    /// Fetch model info / architecture type from the Hub (reads `config.json`).
    pub async fn model_info(&self, model_alias: &str) -> Result<ModelInfo> {
        let actual_repo = crate::registry::resolve_model_alias(model_alias)?;
        let repo = self.api.model(actual_repo);
        let config_path = repo
            .get("config.json")
            .await
            .context("Failed to fetch config.json")?;
        ModelInfo::from_config_file(&config_path)
    }

    /// Returns the expected download size (in bytes) for a model — the files
    /// `fetch_weights` will actually pull, not the whole repo. GGUF repos host
    /// 10+ quants but we download exactly one; summing every sibling used to
    /// make the progress bar report "8% · eta 3h" for a nearly-done pull (and
    /// its verify-phase heuristic, gated on ≥50%, could never fire).
    pub async fn repo_total_bytes(&self, model_alias: &str) -> Result<u64> {
        let actual_repo = crate::registry::resolve_model_alias(model_alias)?;
        let url = model_api_url_with_sizes(&actual_repo);
        let client = reqwest::Client::new();
        let mut req = client.get(&url);
        // Forward auth token if available
        let token = self
            .opts
            .token
            .clone()
            .or_else(|| std::env::var("HF_TOKEN").ok());
        if let Some(t) = token {
            req = req.bearer_auth(t);
        }
        let resp: serde_json::Value = req
            .send()
            .await
            .context("Failed to query HF API for model metadata")?
            .json()
            .await
            .context("Failed to parse HF API response")?;
        let files: Vec<(String, u64)> = resp["siblings"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|s| {
                Some((
                    s["rfilename"].as_str()?.to_string(),
                    s["size"].as_u64().unwrap_or(0),
                ))
            })
            .collect();
        let gguf_names: Vec<String> = files
            .iter()
            .map(|(n, _)| n.clone())
            .filter(|n| n.ends_with(".gguf"))
            .collect();
        let total = if let Some(best) = crate::gguf::select_best_gguf(&gguf_names) {
            // One selected GGUF + the small top-level metadata JSONs.
            files
                .iter()
                .filter(|(n, _)| n == best || (n.ends_with(".json") && !n.contains('/')))
                .map(|(_, sz)| sz)
                .sum()
        } else {
            // Safetensors path: HF repos often dual-ship legacy .bin/.pth
            // alongside .safetensors — count only the format we download.
            let has_st = files.iter().any(|(n, _)| n.ends_with(".safetensors"));
            files
                .iter()
                .filter(|(n, _)| {
                    !has_st
                        || !(n.ends_with(".bin")
                            || n.ends_with(".pth")
                            || n.ends_with(".h5")
                            || n.ends_with(".msgpack"))
                })
                .map(|(_, sz)| sz)
                .sum()
        };
        Ok(total)
    }

    /// Returns the on-disk blobs directory for a HuggingFace model, used to poll download progress.
    /// The path follows HF hub cache conventions: `<hub cache>/models--<org>--<name>/blobs/`,
    /// where the hub cache is [`hub_cache_dir`] (honours `HF_HOME`). Downloads in progress
    /// live here too, as `*.sync.part` files, so its size tracks bytes received.
    pub fn blobs_dir_for_model(model_alias: &str) -> Option<std::path::PathBuf> {
        let actual_repo = crate::registry::resolve_model_alias(model_alias).ok()?;
        let dir_name = format!("models--{}", actual_repo.replace('/', "--"));
        Some(hub_cache_dir()?.join(dir_name).join("blobs"))
    }

    // ── Internals ──────────────────────────────────────────────────────────────

    async fn resolve_files(&self, repo: &ApiRepo, model_id: &str) -> Result<ModelFiles> {
        let (
            config_result,
            tokenizer_path,
            tokenizer_config_path,
            generation_config_path,
            weight_paths,
        ) = tokio::join!(
            async { repo.get("config.json").await.ok() },
            async { repo.get("tokenizer.json").await.ok() },
            async { repo.get("tokenizer_config.json").await.ok() },
            // Optional: Whisper ships suppress-token lists here. Ignored if absent.
            async { repo.get("generation_config.json").await.ok() },
            self.fetch_weights(repo, model_id),
        );

        let weight_paths = weight_paths?;

        // GGUF-only repos (e.g. `…-GGUF`) have no config.json; that is acceptable
        // when the weights are a single GGUF file (ModelInfo comes from the GGUF
        // metadata). For safetensors repos a missing config.json is a hard error.
        let config_path = match config_result {
            Some(p) => {
                debug!("config.json cached for {model_id}");
                p
            }
            None => {
                let is_gguf_only = weight_paths
                    .iter()
                    .all(|p| p.extension().and_then(|e| e.to_str()) == Some("gguf"));
                if !is_gguf_only {
                    anyhow::bail!(
                        "config.json not found — is this a valid model repo? \
                         (GGUF-only repos are allowed to omit config.json)"
                    );
                }
                debug!("GGUF-only repo — skipping config.json for {model_id}");
                // Return a sentinel that the pipeline will ignore for GGUF loads.
                weight_paths[0].clone()
            }
        };

        Ok(ModelFiles {
            model_id: model_id.to_owned(),
            config_path,
            tokenizer_path,
            tokenizer_config_path,
            generation_config_path,
            weight_paths,
        })
    }

    async fn fetch_weights(&self, repo: &ApiRepo, model_id: &str) -> Result<Vec<PathBuf>> {
        let mut filenames: Vec<String> = match repo.info().await {
            Ok(info) => info.siblings.iter().map(|s| s.rfilename.clone()).collect(),
            // Offline (or the Hub is down): a model downloaded earlier is still
            // in the cache, and `repo.get` serves cached files without the
            // network, so choose among those instead of failing the load.
            Err(e) => {
                let cached = hub_cache_dir()
                    .map(|hub| cached_repo_files(&hub, model_id))
                    .unwrap_or_default();
                if cached.is_empty() {
                    return Err(e)
                        .context("Failed to fetch model file listing from HuggingFace Hub");
                }
                debug!(
                    "Hub unreachable ({e}); using {} cached files of {model_id}",
                    cached.len()
                );
                cached
            }
        };
        filenames.sort();

        for fmt in &self.opts.formats {
            match fmt.as_str() {
                "gguf" => {
                    if let Some(name) = select_best_gguf(&filenames) {
                        // Split GGUF (`-NNNNN-of-MMMMM.gguf`, e.g. GLM-4.5-Air Q4_K_M
                        // ≈ 63 GB) → download ALL shards. Sorted so weight_paths[0]
                        // is shard 1 (it carries the full metadata).
                        if let Some(shards) = crate::gguf::gguf_split_shards(name) {
                            debug!("Split GGUF: {} shards", shards.len());
                            let mut paths = if shards.len() > 1 && self.opts.fast_download {
                                self.download_files_parallel(model_id, &shards).await?
                            } else {
                                self.download_files_sequential(repo, &shards).await?
                            };
                            paths.sort();
                            return Ok(paths);
                        }
                        let path = repo
                            .get(name)
                            .await
                            .with_context(|| format!("Failed to download GGUF weights '{name}'"))?;
                        debug!("Found GGUF weights: {}", path.display());
                        return Ok(vec![path]);
                    }
                }
                "safetensors" => {
                    let shards: Vec<String> = filenames
                        .iter()
                        .filter(|n| n.ends_with(".safetensors"))
                        .cloned()
                        .collect();
                    if !shards.is_empty() {
                        let paths = if shards.len() > 1 && self.opts.fast_download {
                            self.download_files_parallel(model_id, &shards).await?
                        } else {
                            self.download_files_sequential(repo, &shards).await?
                        };
                        debug!("Found {} safetensors shard(s)", paths.len());
                        return Ok(paths);
                    }
                }
                "bin" => {
                    for candidate in &["pytorch_model.bin", "pytorch_model.bin.index.json"] {
                        if filenames.iter().any(|n| n == candidate) {
                            let path = repo.get("pytorch_model.bin").await.with_context(|| {
                                "Failed to download pytorch_model.bin".to_string()
                            })?;
                            debug!("Found PyTorch bin weights for model");
                            return Ok(vec![path]);
                        }
                    }
                }
                _ => {}
            }
        }

        anyhow::bail!(
            "No supported weight files found. Tried: {:?}",
            self.opts.formats
        )
    }

    async fn download_files_sequential(
        &self,
        repo: &ApiRepo,
        names: &[String],
    ) -> Result<Vec<PathBuf>> {
        let mut paths = Vec::with_capacity(names.len());
        for name in names {
            let mut retries = 5;
            let mut backoff = 2;
            let path = loop {
                match repo.get(name).await {
                    Ok(p) => break p,
                    Err(e) if retries > 0 => {
                        debug!(
                            "Retry downloading '{}' ({} retries left) due to: {}",
                            name, retries, e
                        );
                        retries -= 1;
                        tokio::time::sleep(std::time::Duration::from_secs(backoff)).await;
                        backoff = std::cmp::min(backoff * 2, 10);
                    }
                    Err(e) => return Err(anyhow::anyhow!("Failed to download '{}': {}", name, e)),
                }
            };
            paths.push(path);
        }
        Ok(paths)
    }

    async fn download_files_parallel(
        &self,
        model_id: &str,
        names: &[String],
    ) -> Result<Vec<PathBuf>> {
        let workers = max_parallel_downloads();
        let semaphore = Arc::new(Semaphore::new(workers));
        let mut handles = Vec::with_capacity(names.len());

        for name in names {
            let api = self.api.clone();
            let model_id = model_id.to_owned();
            let name = name.clone();
            let sem = semaphore.clone();
            handles.push(tokio::spawn(async move {
                let _permit = sem
                    .acquire()
                    .await
                    .map_err(|e| anyhow::anyhow!("download worker failed: {e}"))?;

                let mut retries = 5;
                let mut backoff = 2;
                loop {
                    match api.model(model_id.clone()).get(&name).await {
                        Ok(p) => return Ok(p),
                        Err(e) if retries > 0 => {
                            debug!(
                                "Retry parallel download '{}' ({} retries left) due to: {}",
                                name, retries, e
                            );
                            retries -= 1;
                            tokio::time::sleep(std::time::Duration::from_secs(backoff)).await;
                            backoff = std::cmp::min(backoff * 2, 10);
                        }
                        Err(e) => {
                            return Err(anyhow::anyhow!("Failed to download '{}': {}", name, e))
                        }
                    }
                }
            }));
        }

        let mut paths = Vec::with_capacity(handles.len());
        for handle in handles {
            paths.push(handle.await.context("parallel download task panicked")??);
        }
        paths.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
        Ok(paths)
    }

    fn read_cached_token() -> Option<String> {
        let path = dirs::home_dir()?.join(".cache/huggingface/token");
        std::fs::read_to_string(path)
            .ok()
            .map(|s| s.trim().to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cached_repo_files_lists_the_main_snapshot() {
        let hub = tempfile::tempdir().unwrap();
        let repo = hub.path().join("models--org--name");
        std::fs::create_dir_all(repo.join("refs")).unwrap();
        std::fs::write(repo.join("refs/main"), "abc123\n").unwrap();
        let snapshot = repo.join("snapshots/abc123");
        std::fs::create_dir_all(snapshot.join("sub")).unwrap();
        std::fs::write(snapshot.join("model-Q4_K_M.gguf"), b"x").unwrap();
        std::fs::write(snapshot.join("sub/extra.json"), b"{}").unwrap();
        // An older snapshot is ignored: only refs/main counts.
        std::fs::create_dir_all(repo.join("snapshots/old")).unwrap();
        std::fs::write(repo.join("snapshots/old/stale.gguf"), b"x").unwrap();

        let mut files = cached_repo_files(hub.path(), "org/name");
        files.sort();
        assert_eq!(files, vec!["model-Q4_K_M.gguf", "sub/extra.json"]);
        assert!(cached_repo_files(hub.path(), "org/missing").is_empty());
    }

    #[test]
    fn model_api_url_asks_for_file_sizes() {
        assert_eq!(
            model_api_url_with_sizes("unsloth/SmolLM2-135M-Instruct-GGUF"),
            "https://huggingface.co/api/models/unsloth/SmolLM2-135M-Instruct-GGUF?blobs=true"
        );
    }

    #[test]
    fn hub_cache_dir_honours_hf_home() {
        let home = Some(PathBuf::from("/home/u"));
        assert_eq!(
            hub_cache_dir_from(Some(PathBuf::from("/app/Caches/sapient")), home.clone()),
            Some(PathBuf::from("/app/Caches/sapient/hub"))
        );
        assert_eq!(
            hub_cache_dir_from(None, home.clone()),
            Some(PathBuf::from("/home/u/.cache/huggingface/hub"))
        );
        // An empty HF_HOME means unset, not the current directory.
        assert_eq!(
            hub_cache_dir_from(Some(PathBuf::new()), home),
            Some(PathBuf::from("/home/u/.cache/huggingface/hub"))
        );
        // Android: no home directory, but HF_HOME still works.
        assert_eq!(
            hub_cache_dir_from(Some(PathBuf::from("/data/app/cache")), None),
            Some(PathBuf::from("/data/app/cache/hub"))
        );
        assert_eq!(hub_cache_dir_from(None, None), None);
    }

    #[test]
    fn dir_bytes_counts_files_once_and_skips_symlinks() {
        let root = tempfile::tempdir().unwrap();
        let blobs = root.path().join("blobs");
        std::fs::create_dir_all(blobs.join("nested")).unwrap();
        std::fs::write(blobs.join("a"), vec![0u8; 1000]).unwrap();
        std::fs::write(blobs.join("nested/b.sync.part"), vec![0u8; 24]).unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(blobs.join("a"), root.path().join("link")).unwrap();
        assert_eq!(dir_bytes(&blobs), 1024);
        assert_eq!(
            dir_bytes(root.path()),
            1024,
            "the symlink is not counted again"
        );
        assert_eq!(dir_bytes(&root.path().join("missing")), 0);
    }
}
