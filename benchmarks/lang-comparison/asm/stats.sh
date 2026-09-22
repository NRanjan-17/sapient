#!/usr/bin/env bash
# Instruction census of a disassembled function. Handles both objdump layouts
# (leading-whitespace offsets from a .o, absolute addresses from a linked binary).
# Usage: stats.sh <file.asm> [label]
set -euo pipefail
f="$1"; label="${2:-$(basename "$f")}"
# An instruction line is "<hex addr>: <hex encoding> \t mnemonic ..."
insn() { grep -E '^[[:space:]]*[0-9a-f]+:[[:space:]]+[0-9a-f]{8}' "$f"; }
n()    { insn | grep -cE "$1" || true; }
printf '%-28s %s\n' "target:"        "$label"
printf '%-28s %s\n' "instructions:"  "$(insn | wc -l | tr -d ' ')"
printf '%-28s %s\n' "sdot:"          "$(n '\bsdot\b')"
printf '%-28s %s\n' "smmla:"         "$(n '\bsmmla\b')"
printf '%-28s %s\n' "stack refs [sp:" "$(n '\[sp')"
printf '%-28s %s\n' "calls (bl):"    "$(n '\bbl\b')"
printf '%-28s %s\n' "cond branches:" "$(n '\bb\.[a-z]+\b|\bcbn?z\b|\btbn?z\b')"
printf '%-28s %s\n' "loads (ldr/ldp):" "$(n '\bldr\b|\bldp\b|\bldur\b')"
printf '%-28s %s\n' "stores (str/stp):" "$(n '\bstr\b|\bstp\b|\bstur\b')"
