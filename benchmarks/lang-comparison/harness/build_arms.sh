#!/usr/bin/env bash
# Build the two end-to-end arms of the language study.
#
#   A  baseline  — the engine exactly as it ships (Rust leaf kernels)
#   C  cpp       — identical, except the two hottest K-quant GEMV leaves
#                  dispatch to the bit-identical C++ transliterations
#
# Everything else (spin pool, chunk geometry, activation quantisation, KV
# cache, tokenizer, sampler) is shared, so the only variable is the language of
# the leaf kernel.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT="$ROOT/benchmarks/lang-comparison/results/bin"
mkdir -p "$OUT"
cd "$ROOT"

echo "── arm A: baseline (default features, no C++ toolchain used) ──"
cargo build --release -p sapient-cli 2>&1 | tail -1
cp target/release/sapient "$OUT/sapient-rust"

echo "── arm B: unrolled-kernels (Rust, clang's two optimisations applied by hand) ──"
cargo build --release -p sapient-cli -p sapient-backends-cpu \
    --features sapient-backends-cpu/unrolled-kernels 2>&1 | tail -1
cp target/release/sapient "$OUT/sapient-rust-unrolled"

echo "── arm C: cpp-kernels ──"
cargo build --release -p sapient-cli -p sapient-backends-cpu \
    --features sapient-backends-cpu/cpp-kernels 2>&1 | tail -1
cp target/release/sapient "$OUT/sapient-cpp"

echo
echo "── verification: the swap must actually be live ──"
for arm in rust rust-unrolled cpp; do
  bin="$OUT/sapient-$arm"
  has_cpp=$(nm -U "$bin" 2>/dev/null | grep -c 'sapient_cpp_dot_q4_k_4rows_r4_q8k' || true)
  size=$(stat -f%z "$bin")
  printf "  %-16s size=%-10s C++ kernel symbol present: %s\n" "sapient-$arm" "$size" \
     "$([ "$has_cpp" -gt 0 ] && echo YES || echo no)"
done
# Arm A must NOT contain the C++ kernel; arm C must.
# NB: `grep -q` exits on first match, which SIGPIPEs `nm`; under `pipefail`
# that reads as a failed pipeline. Count into a variable instead.
a=$(nm -U "$OUT/sapient-rust" 2>/dev/null | grep -c 'sapient_cpp_dot' || true)
u=$(nm -U "$OUT/sapient-rust-unrolled" 2>/dev/null | grep -c 'unrolled_vectail' || true)
c=$(nm -U "$OUT/sapient-cpp"  2>/dev/null | grep -c 'sapient_cpp_dot' || true)
[ "$a" -eq 0 ] || { echo "FAIL: baseline contains the C++ kernel"; exit 1; }
[ "$c" -gt 0 ] || { echo "FAIL: cpp arm does not contain the C++ kernel"; exit 1; }
[ "$u" -gt 0 ] || { echo "FAIL: unrolled arm does not contain the unrolled kernel"; exit 1; }
echo "  OK: three arms differ exactly as intended"
echo "     baseline C++ syms=$a | unrolled kernel syms=$u | cpp C++ syms=$c"
