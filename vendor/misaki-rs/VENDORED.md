# Vendored: misaki-rs 0.3.0

Source: https://github.com/MicheleYin/misaki-rs (crates.io `misaki-rs` 0.3.0), MIT —
see `LICENSE`. Upstream README is kept as `README.upstream.md`.

## Why it is vendored

Upstream embeds its data with `include_str!`: four pronunciation dictionaries
(29.8 MB of JSON) and the POS-tagger weights (5.7 MB). That was 35.5 MB of a 52 MB
`sapient` binary, used only by Kokoro text-to-speech.

## Changes from upstream

- `data/*.json` and `src/resources/tagger/weights.json` are stored gzip-compressed
  (`gzip -9 -n`; 35.5 MB → 7.6 MB). Content is byte-identical after decompression.
- `src/data.rs` is rewritten: data resolves from a directory registered with
  `data::set_data_dir` (files named as in `data::DATA_FILES`), else from the embedded
  copy when the new `embed-dict` feature is on. `data::available()` reports whether a
  `G2P` can be built.
- `src/g2p.rs`: the tagger weights load through `data::load_text` (one line changed).
- `Cargo.toml`: the optional `espeak` feature and dependency are dropped; `flate2` added.
- Examples and upstream test data files are not included.

No phonemization logic is changed.

## Updating

Copy the new upstream `src/`, re-apply the two source changes above, and re-compress
the data files with `gzip -9 -n`. `tagger_weights.json.gz` in a data directory is
`src/resources/tagger/weights.json.gz` under its external name.
