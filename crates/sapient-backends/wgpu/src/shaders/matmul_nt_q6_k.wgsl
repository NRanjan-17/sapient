// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

// Resident Q6_K linear projection: out[M,N] = x[M,K] @ dequant(W)[N,K]^T.
// W stays in ggml Q6_K super-blocks on the GPU, host-padded from 210 to 212 bytes
// per 256 weights (2 zero bytes so each block is word-aligned; nothing dequantized):
//   words 0..31  : ql[128] — lower 4 bits of the quants
//   words 32..47 : qh[64]  — upper 2 bits of the quants
//   words 48..51 : scales[16] — SIGNED int8, one per 16-weight group
//   word 52      : d (f16, low half) + 2 pad bytes
// Element e (h=e/128, g=(e%128)/32, l=e%32):
//   q  = ((ql[h*64+(g&1)*32+l] >> (g/2)*4) & 0xF) | (((qh[h*32+l] >> 2g) & 3) << 4)
//   val = d · sc[h*8 + 2g + l/16] · (q − 32)
// — exactly the (fixed) CPU dequantize_row_q6_K scale indexing: offsets +0/+2/+4/+6
// per 128-half with the l==16 split; getting this wrong is the historical
// token-salad bug, so change nothing here without the random-bit reference tests.
// ROWS output elements per workgroup, LANES threads each (was: one 256-lane workgroup per element); stride over the K/16 scale-groups,
// accumulating Σx·(q−32) per group and applying d·sc once. f32 accumulation.

struct P { m: u32, k: u32, n: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       x:   array<f32>;
@group(0) @binding(1) var<storage, read>       qb:  array<u32>; // 53 words / super-block
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform>             p:   P;

// Decode GEMV layout: ROWS output elements per workgroup, LANES threads per
// element. The old layout gave every output element a 256-thread workgroup:
// for a 1536-wide row most threads idled and every element paid an 8-step
// reduction. Swept on an Apple M4 GPU (Qwen2.5-1.5B Q4_K_M decode, ms/token):
// 256 lanes 57.6 · 64 lanes 36.5 · 32 lanes 34.8 · 16 lanes 27.4 · 8 lanes 27.2.
// Untuned on Vulkan/DX12 GPUs; ROWS must match GEMV_ROWS in resident.rs.
const LANES: u32 = 16u;
const ROWS: u32 = 16u; // LANES * ROWS = 256 = workgroup size
var<workgroup> partial: array<f32, 256>;

fn byte_of(word: u32, b: u32) -> u32 { return (word >> (b * 8u)) & 0xFFu; }

@compute @workgroup_size(256)
fn cs_main(@builtin(workgroup_id) wg: vec3<u32>,
           @builtin(local_invocation_id) lid: vec3<u32>,
           @builtin(num_workgroups) nwg: vec3<u32>) {
    let lane = lid.x % LANES;
    let idx = (wg.x + wg.y * nwg.x) * ROWS + lid.x / LANES;
    // No early return: every thread must reach the barriers below. Lanes past
    // the end compute a clamped (valid) element and skip the write.
    let in_range = idx < p.m * p.n;
    let idc = min(idx, p.m * p.n - 1u);
    let rm = idc / p.n;
    let rn = idc % p.n;
    let nblocks = p.k / 256u;
    let row_base = rn * nblocks * 53u;
    let xb = rm * p.k;
    let tid = lane;

    var acc = 0.0;
    var sg = tid;                 // 16-weight scale-group index within the row
    let nsg = p.k / 16u;
    loop {
        if (sg >= nsg) { break; }
        let b = sg / 16u;         // super-block
        let s = sg % 16u;         // scale index within it (0..15)
        let h = s >> 3u;          // 128-half
        let g = (s & 7u) >> 1u;   // 32-group within the half (0..3)
        let l16 = s & 1u;         // l==16 split
        let blk = row_base + b * 53u;

        let d = unpack2x16float(qb[blk + 52u]).x;
        // Signed int8 scale: byte s of the scales region (words 48..51).
        let sc_raw = byte_of(qb[blk + 48u + (s >> 2u)], s & 3u);
        let sc = f32(i32(sc_raw << 24u) >> 24u);

        let ql0 = (h * 64u + (g & 1u) * 32u + l16 * 16u) >> 2u; // ql word start
        let qh0 = (h * 32u + l16 * 16u) >> 2u;                  // qh word start
        let shift4 = (g >> 1u) * 4u;
        let qh_shift = g * 2u;
        let xoff = xb + b * 256u + h * 128u + g * 32u + l16 * 16u;

        // Vectorized dequant: assemble each 6-bit q in its byte lane (low nibble
        // from ql, two high bits from qh) and read all four as q/255 with one
        // unpack4x8unorm; Σx·(q−32) = 255·Σx·(q/255) − 32·Σx.
        var sum_q = 0.0; // Σ x·(q/255)
        var sum_x = 0.0; // Σ x
        for (var w = 0u; w < 4u; w = w + 1u) {
            let ln = (qb[blk + ql0 + w] >> shift4) & 0x0F0F0F0Fu;
            let hn = ((qb[blk + 32u + qh0 + w] >> qh_shift) & 0x03030303u) << 4u;
            let q6 = unpack4x8unorm(ln | hn);
            let xi = xoff + w * 4u;
            let xv = vec4<f32>(x[xi], x[xi + 1u], x[xi + 2u], x[xi + 3u]);
            sum_q = sum_q + dot(q6, xv);
            sum_x = sum_x + dot(xv, vec4<f32>(1.0));
        }
        acc = acc + d * sc * (255.0 * sum_q - 32.0 * sum_x);
        sg = sg + LANES;
    }
    partial[lid.x] = acc;
    workgroupBarrier();

    var s = LANES / 2u;
    loop {
        if (s == 0u) { break; }
        if (lane < s) { partial[lid.x] = partial[lid.x] + partial[lid.x + s]; }
        workgroupBarrier();
        s = s / 2u;
    }
    if (in_range && lane == 0u) { out[idx] = partial[lid.x]; }
}
