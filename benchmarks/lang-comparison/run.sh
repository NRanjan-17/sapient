#!/usr/bin/env bash
# Single entry point for the Rust-vs-C++ study. `just bench-lang` wraps this.
#
#   ./run.sh [all|gate|asm|micro|e2e]
#
# The gate runs first and is fatal: no timing number is believed until the C++
# kernels are proven bit-identical to the Rust scalar oracles.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
STAGE="${1:-all}"
HARNESS="$HERE/rust/harness"
RESULTS="$HERE/results"
mkdir -p "$RESULTS/raw" "$RESULTS/asm"

banner() { printf '\n\033[1m── %s ──\033[0m\n' "$1"; }

need() { command -v "$1" >/dev/null || { echo "missing tool: $1"; exit 1; }; }
need cargo; need objdump; need python3

gate() {
  banner "correctness gate (bit-identity vs the Rust scalar oracles)"
  ( cd "$HARNESS" && cargo test --release --test bit_identity )
  ( cd "$HARNESS" && cargo test --release --test bit_identity -- --nocapture report_cpp \
      2>/dev/null | grep -E '^C\+\+' > "$RESULTS/raw/build_config.txt" ) || true
  cat "$RESULTS/raw/build_config.txt" 2>/dev/null || true
}

asm() {
  banner "assembly census + opcode diff (both arms from ONE linked binary)"
  ( cd "$HARNESS" && cargo bench --no-run >/dev/null 2>&1 )
  local bin
  bin=$(ls -t "$HARNESS"/target/release/deps/kernels-* | grep -v '\.d$' | head -1)
  "$HERE/asm/extract.sh" "$bin" "$RESULTS/asm" | tee "$RESULTS/raw/asm_census.txt"
}

micro() {
  banner "micro-benchmarks (Criterion, n=100/arm)"
  ( cd "$HARNESS" && caffeinate -dimsu cargo bench --bench kernels )
  ( cd "$HARNESS" && python3 "$HERE/harness/collect_criterion.py" \
        target/criterion "$RESULTS/raw/criterion.json" )
}

e2e() {
  banner "end-to-end A/B (real engine, Rust leaves vs C++ leaves)"
  "$HERE/harness/build_arms.sh"
  ( cd "$HERE/harness" && caffeinate -dimsu python3 ab_e2e.py \
      --blocks 4 --runs 3 --max-tokens 128 --threads 10 \
      --out "$RESULTS/raw/e2e_decode.json" )
}

profile() {
  banner "profile-share for modalities with no C++ arm (Whisper / Kokoro / SmolVLM)"
  ( cd "$HERE/../.." && caffeinate -dimsu python3 \
      benchmarks/lang-comparison/harness/profile_share.py --seconds 6 )
}

case "$STAGE" in
  gate)  gate ;;
  profile) profile ;;
  asm)   asm ;;
  micro) micro ;;
  e2e)   gate; e2e ;;          # never time without the gate
  all)   gate; asm; micro; e2e; profile ;;
  *)     echo "usage: $0 [all|gate|asm|micro|e2e|profile]"; exit 2 ;;
esac

banner "done — raw results in $RESULTS/raw"
