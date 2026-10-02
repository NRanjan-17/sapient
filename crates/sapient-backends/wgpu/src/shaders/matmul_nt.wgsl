// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)

// Resident linear projection: out[M,N] = x[M,K] @ w[N,K]^T  (w in HF [N,K] layout).
// ROWS output elements per workgroup, LANES threads each (was: one 256-lane workgroup per element); cooperatively reduce the K dot product
// (GEMV-style — ideal for batch-1 decode). f32 accumulation.
// Index is 2-D-tiled: idx = wg.x + wg.y*num_workgroups.x (handles N>65535, e.g. lm_head).

struct P { m: u32, k: u32, n: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       x:   array<f32>;
@group(0) @binding(1) var<storage, read>       w:   array<f32>;
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
    let xb = rm * p.k;
    let wb = rn * p.k;
    let tid = lane;

    var acc = 0.0;
    var i = tid;
    loop {
        if (i >= p.k) { break; }
        acc = acc + x[xb + i] * w[wb + i];
        i = i + LANES;
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
