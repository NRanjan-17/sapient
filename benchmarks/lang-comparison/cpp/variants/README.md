# Optimisation attempts on the C++ Q4_K kernel — all NEGATIVE

Three hand-optimisations of `sapient_cpp_dot_q4_k_4rows_r4_q8k`, the kernel that
is 64.1% of decode time. Every one is **bit-identical** to the original (the
A/B driver compares raw f32 bit patterns, not approximate equality) and every
one is **slower**.

Kept as a record so nobody spends the afternoon re-deriving them.

| Variant | What it changes | Speed vs original |
|---|---|---:|
| `q4k_hoist_unroll.cpp` | hoist `get_scale_min_k4` into `SC[4][8]`/`MN[4][8]`; unroll the 4-row loop | **0.784x** (22% slower) |
| `q4k_vecacc_unroll.cpp` | the above **plus** `vmlaq_n_s32` vector accumulators — one horizontal reduction per super-block instead of 32 | **0.924x** (7.6% slower) |

## Reading the ablation

The vector-accumulator change on its own was worth **+18%** (0.784 -> 0.924).
Replacing 32 `vaddvq_s32` horizontal reductions per super-block with 4 is a real
win, and is worth keeping in mind for any future kernel.

It could not pay for the hoisting. Precomputing the scale/min pairs puts 64
bytes on the stack per super-block, and the unrolled 4-row body then needs those
registers back. clang was already keeping the unpack in flight; the
"optimisation" forced it through memory.

**The general lesson matches the three already in the main report** —
`get_unchecked` measured slower, cache-blocking was -7% on Thor, the x4 register
tile spilled on M4. The kernel sits at a local optimum and obvious
restructurings lose.

## Reproduce

```bash
cd cpp/variants
c++ -O3 -std=c++17 -flto -ffp-contract=off -I../include \
    -o ab ../src/q4k.cpp q4k_vecacc_unroll.cpp ab_bench.cpp
./ab 35 60000        # 35 super-blocks (a real qwen2.5-1.5b row), 60k iterations
```

Two traps the driver handles, both of which produced a *convincingly wrong*
answer first time:

1. **Random bytes are not valid weights.** Bytes 0..3 of each 144-byte block are
   the f16 `(d, dmin)`. Filled with random bytes they decode to NaN/Inf and every
   result is `nan` — which still "passes" a bit-identity check, because
   `nan == nan` bitwise. The driver writes real half-precision values there.
2. **LTO deletes the benchmark.** The call is pure with loop-invariant arguments,
   so clang hoists it out and times an empty loop — 60 000 iterations in 0.0002 s.
   The driver puts an `asm volatile("" ::: "memory")` barrier in each iteration.
