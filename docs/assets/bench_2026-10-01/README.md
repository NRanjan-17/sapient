# Raw data — 2026-10-01 M4 reproduction

Apple M4 (MacBook Pro, 16 GB). SAPIENT v0.6.0 release binaries (CPU build and
`-metal` build), llama.cpp b9860 (Homebrew), Ollama 0.12.6. Other desktop apps
were running (load average ~2.6), so absolute numbers sit a little below a
cool, idle machine.

- `r_qwen.txt`, `r_llama.txt` — verbatim output of `bench_m4.py` (SAPIENT and llama.cpp).
- `bench_m4.py` — the harness. SAPIENT decode tok/s = (400 − 16) / (T400 − T16) from
  `sapient chat <file.gguf> -p … -n N --raw` wall times (differencing removes
  load + prefill); llama.cpp = `llama-bench -p 0 -n 128 -r 3` (CPU `-t 4 -ngl 0`,
  Metal `-ngl 99`). Three interleaved rounds with 10 s rests.
- `bench_ollama.py` — Ollama `/api/generate`, `eval_count / eval_duration`, 4 runs, 128 tokens.
  Its output was (tok/s): `qwen2.5:1.5b` 76.1, 78.1, 73.2, 77.8 · `llama3.2:1b`
  63.1, 64.4, 64.0, 64.6.
- Thread check (Qwen file, `llama-bench -t 4,10 -ngl 0`): 55.9 ± 1.9 tok/s at 4
  threads, 17.0 ± 1.1 at 10.

The GGUF files were Ollama's own blobs (`qwen2.5:1.5b` = Q4_K_M, 986 MB;
`llama3.2:1b` = Q8_0, 1.32 GB), so all three engines read byte-identical files.
This run predates the fixed `bench-llm`; future runs should use
`sapient bench-llm <file.gguf> --json` and commit its output here.
