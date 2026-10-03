// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

//! Generation benchmark core, shared by `sapient bench-llm` and the FFI
//! (`LlmSession::benchmark`) so the CLI and the mobile SDKs report the same
//! numbers with the same definitions (docs/BENCHMARKS.md):
//!
//! - greedy decoding, exact engine token count (no re-tokenization);
//! - **decode tok/s** = `(tokens − 1) / (t_last − t_first)`, prefill excluded;
//! - **TTFT** = prompt prefill + first token;
//! - **prefill tok/s** ≈ `prompt_tokens / TTFT` (includes the first decode
//!   step, so a slight under-estimate);
//! - warm-up runs are reported separately and excluded from every summary.

use std::time::{Duration, Instant};

use anyhow::Result;

use crate::pipeline::Pipeline;
use crate::sampler::SamplingStrategy;

/// One timed generation.
#[derive(Debug, Clone, PartialEq)]
pub struct BenchSample {
    /// Prompt prefill + first token, in milliseconds.
    pub ttft_ms: u64,
    /// Whole run, in milliseconds.
    pub elapsed_ms: u64,
    /// Tokens generated.
    pub tokens: usize,
    /// Decode-only throughput: `(tokens − 1) / (t_last − t_first)`.
    pub decode_tps: f64,
    /// `prompt_tokens / TTFT`.
    pub prefill_tps: f64,
    /// Generation ended on end-of-turn before reaching `max_tokens`.
    pub hit_eos: bool,
}

/// Means and range over the measured (non-warm-up) runs.
#[derive(Debug, Clone, PartialEq)]
pub struct BenchSummary {
    pub mean_ttft_ms: u64,
    pub mean_decode_tps: f64,
    pub min_decode_tps: f64,
    pub max_decode_tps: f64,
    pub mean_prefill_tps: f64,
}

/// Greedy generation of up to `max_tokens` tokens, timestamping every token
/// as the engine produces it. Runs synchronously on the calling thread (from
/// an async context, wrap it in `tokio::task::block_in_place`). Starts from a
/// cleared KV cache, so runs are independent of each other and of any chat.
pub fn bench_generate(
    pipeline: &Pipeline,
    prompt_ids: &[u32],
    eos_ids: &[u32],
    max_tokens: usize,
) -> Result<BenchSample> {
    let mut stamps: Vec<Instant> = Vec::with_capacity(max_tokens);
    let start = Instant::now();
    pipeline.generate_token_ids_streaming(
        prompt_ids,
        max_tokens,
        eos_ids,
        SamplingStrategy::Greedy,
        |_| {
            stamps.push(Instant::now());
            true
        },
    )?;
    let end = Instant::now();
    let offsets: Vec<Duration> = stamps.iter().map(|t| t.duration_since(start)).collect();
    Ok(sample_from_offsets(
        &offsets,
        end.duration_since(start),
        prompt_ids.len(),
        max_tokens,
    ))
}

/// The arithmetic behind [`bench_generate`]: `token_offsets` are each
/// token's arrival time measured from the start of the run.
pub fn sample_from_offsets(
    token_offsets: &[Duration],
    elapsed: Duration,
    prompt_tokens: usize,
    max_tokens: usize,
) -> BenchSample {
    let tokens = token_offsets.len();
    let ttft = token_offsets.first().copied().unwrap_or_default();
    let decode_tps = match (token_offsets.first(), token_offsets.last()) {
        (Some(first), Some(last)) if tokens >= 2 => {
            let span = last.saturating_sub(*first).as_secs_f64();
            if span > 0.0 {
                (tokens - 1) as f64 / span
            } else {
                0.0
            }
        }
        _ => 0.0,
    };
    let prefill_tps = if tokens > 0 && ttft.as_secs_f64() > 0.0 {
        prompt_tokens as f64 / ttft.as_secs_f64()
    } else {
        0.0
    };
    BenchSample {
        ttft_ms: ttft.as_millis() as u64,
        elapsed_ms: elapsed.as_millis() as u64,
        tokens,
        decode_tps,
        prefill_tps,
        hit_eos: tokens < max_tokens,
    }
}

/// Summary over measured runs; `None` when there are none.
pub fn summarize(runs: &[BenchSample]) -> Option<BenchSummary> {
    if runs.is_empty() {
        return None;
    }
    let n = runs.len() as f64;
    let decode = runs.iter().map(|r| r.decode_tps);
    Some(BenchSummary {
        mean_ttft_ms: runs.iter().map(|r| r.ttft_ms).sum::<u64>() / runs.len() as u64,
        mean_decode_tps: decode.clone().sum::<f64>() / n,
        min_decode_tps: decode.clone().fold(f64::INFINITY, f64::min),
        max_decode_tps: decode.fold(f64::NEG_INFINITY, f64::max),
        mean_prefill_tps: runs.iter().map(|r| r.prefill_tps).sum::<f64>() / n,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(v: u64) -> Duration {
        Duration::from_millis(v)
    }

    #[test]
    fn decode_rate_excludes_the_first_token() {
        // First token at 200 ms (prefill), then one every 50 ms.
        let offsets = [ms(200), ms(250), ms(300), ms(350), ms(400)];
        let s = sample_from_offsets(&offsets, ms(410), 40, 5);
        assert_eq!(s.ttft_ms, 200);
        assert_eq!(s.elapsed_ms, 410);
        assert_eq!(s.tokens, 5);
        // 4 tokens over 200 ms = 20 tok/s; prefill does not count.
        assert!((s.decode_tps - 20.0).abs() < 1e-9);
        // 40 prompt tokens before the first token at 200 ms.
        assert!((s.prefill_tps - 200.0).abs() < 1e-9);
        assert!(!s.hit_eos);
    }

    #[test]
    fn short_runs_have_no_decode_rate_and_flag_eos() {
        let one = sample_from_offsets(&[ms(120)], ms(130), 10, 64);
        assert_eq!(one.decode_tps, 0.0);
        assert!(one.hit_eos);
        let none = sample_from_offsets(&[], ms(5), 10, 64);
        assert_eq!((none.tokens, none.ttft_ms, none.prefill_tps), (0, 0, 0.0));
    }

    #[test]
    fn summary_is_mean_and_range_of_the_given_runs() {
        let run = |ttft_ms, decode_tps, prefill_tps| BenchSample {
            ttft_ms,
            elapsed_ms: 0,
            tokens: 10,
            decode_tps,
            prefill_tps,
            hit_eos: false,
        };
        let s = summarize(&[
            run(100, 10.0, 300.0),
            run(200, 20.0, 100.0),
            run(300, 30.0, 200.0),
        ])
        .unwrap();
        assert_eq!(s.mean_ttft_ms, 200);
        assert!((s.mean_decode_tps - 20.0).abs() < 1e-9);
        assert_eq!((s.min_decode_tps, s.max_decode_tps), (10.0, 30.0));
        assert!((s.mean_prefill_tps - 200.0).abs() < 1e-9);
        assert_eq!(summarize(&[]), None);
    }
}
