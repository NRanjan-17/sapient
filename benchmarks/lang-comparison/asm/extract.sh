#!/usr/bin/env bash
# Extract and census the Rust and C++ arms of a kernel FROM THE SAME LINKED
# BINARY, so both have seen the same linker, the same LTO settings and the same
# inlining scope. Comparing a standalone .o against a fully-linked binary would
# measure the build configuration, not the language.
#
# Usage: extract.sh <binary> <outdir>
set -euo pipefail
BIN="${1:?usage: extract.sh <binary> <outdir>}"
OUT="${2:?usage: extract.sh <binary> <outdir>}"
mkdir -p "$OUT"
HERE="$(cd "$(dirname "$0")" && pwd)"

for kern in q4_k q6_k; do
  rust_sym=$(nm -U "$BIN" | grep -oE "__RNv[A-Za-z0-9_]*dot_${kern}_4rows_r4_q8k_neon[A-Za-z0-9_]*" | head -1 || true)
  cpp_sym="_sapient_cpp_dot_${kern}_4rows_r4_q8k"
  [ -z "$rust_sym" ] && { echo "!! no rust symbol for $kern"; continue; }

  objdump -d --disassemble-symbols="$rust_sym" "$BIN" > "$OUT/${kern}.rust.asm"
  objdump -d --disassemble-symbols="$cpp_sym"  "$BIN" > "$OUT/${kern}.cpp.asm"

  echo "==================== $kern ===================="
  "$HERE/stats.sh" "$OUT/${kern}.rust.asm" "RUST  dot_${kern}_4rows_r4_q8k_neon"
  echo
  "$HERE/stats.sh" "$OUT/${kern}.cpp.asm"  "C++   sapient_cpp_dot_${kern}_4rows_r4_q8k"
  echo

  # Normalised opcode streams, for a structural diff that ignores addresses,
  # register allocation and literal offsets.
  for side in rust cpp; do
    # objdump layout is: "<addr>: <encoding>\t<mnemonic>\t<operands>".
    # Take the tab-delimited MNEMONIC field, not the first operand.
    grep -E '^[[:space:]]*[0-9a-f]+:[[:space:]]+[0-9a-f]{8}' "$OUT/${kern}.${side}.asm" \
      | awk -F'\t' '{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2}' \
      | grep -v '^$' \
      > "$OUT/${kern}.${side}.ops"
  done
  echo "--- opcode histogram delta (C++ minus Rust) ---"
  join -a1 -a2 -e0 -o 0,1.2,2.2 \
    <(sort "$OUT/${kern}.rust.ops" | uniq -c | awk '{print $2, $1}' | sort) \
    <(sort "$OUT/${kern}.cpp.ops"  | uniq -c | awk '{print $2, $1}' | sort) \
    | awk '{d=$3-$2; if (d!=0) printf "  %-12s rust=%-4s cpp=%-4s  %+d\n", $1, $2, $3, d}' \
    | sort -k4 -n
  echo
done
