#!/usr/bin/env python3
"""How far LeRobot's own default precision (bf16) moves SmolVLA's actions.

Runs `lerobot/smolvla_base` twice on CPU — all-f32 and LeRobot's default (VLM
and action expert in bf16) — over the same eight synthetic observations that
`smolvla_quantized_error_over_observations` (crates/sapient-models/tests/
smolvla_reference.rs) uses, and prints the max / RMS difference of the action
chunks. That is the yardstick Sapient's Q8_0 action error is compared with.

The observation generator mirrors the Rust test exactly (SplitMix64 stream,
procedural image, rotated instruction tokens).

Needs `pip install "lerobot[smolvla]"` (throwaway virtual environment).
Usage: python3 scripts/smolvla_bf16_yardstick.py
"""
import sys

import numpy as np
import torch

sys.path.insert(0, "scripts")
from gen_smolvla_fixture import TASK, test_image  # noqa: E402

N, SIZE = 8, 512
MASK = (1 << 64) - 1


class Stream:
    def __init__(self, seed: int = 0x5EED):
        self.s = seed

    def uniform(self) -> np.float32:
        self.s = (self.s + 0x9E3779B97F4A7C15) & MASK
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        return np.float32(((z ^ (z >> 31)) >> 40)) / np.float32(1 << 24)

    def normal(self) -> np.float32:
        acc = np.float32(0.0)
        for _ in range(12):
            acc = np.float32(acc + self.uniform())
        return np.float32(acc - np.float32(6.0))


def observations(base_lang: list[int]):
    rng = Stream()
    base = test_image().astype(np.float32).transpose(2, 0, 1) / np.float32(255.0) * 2.0 - 1.0
    out = []
    for o in range(N):
        px = np.clip(base * np.float32(0.6 + 0.05 * o), -1.0, 1.0).astype(np.float32)
        bx, by = 40 + (53 * o) % 300, 30 + (71 * o) % 300
        rgb = [(o * 37) % 256, (o * 91) % 256, 255 - (o * 29) % 256]
        for c in range(3):
            px[c, by : by + 120, bx : bx + 150] = np.float32(rgb[c]) / np.float32(255.0) * 2.0 - 1.0
        body = base_lang[:-1]
        lang = [body[(o + i) % len(body)] for i in range(4 + o)] + [base_lang[-1]]
        state = np.zeros(32, np.float32)
        for i in range(6):
            state[i] = rng.normal() * np.float32(0.7)
        noise = np.array([rng.normal() for _ in range(50 * 32)], np.float32).reshape(50, 32)
        out.append((px, lang, state, noise))
    return out


def main() -> None:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    def load(f32: bool):
        policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
        policy.to("cpu")
        if f32:
            policy.model.float()
        policy.eval()
        return policy.model

    def run(model, obs):
        tok = model.vlm_with_expert.processor.tokenizer
        pad = tok.pad_token_id
        res = []
        with torch.no_grad():
            for px, lang, state, noise in obs:
                ids = torch.full((1, 48), pad, dtype=torch.long)
                ids[0, : len(lang)] = torch.tensor(lang)
                mask = torch.zeros(1, 48, dtype=torch.bool)
                mask[0, : len(lang)] = True
                a = model.sample_actions(
                    [torch.from_numpy(px)[None]],
                    [torch.ones(1, dtype=torch.bool)],
                    ids,
                    mask,
                    torch.from_numpy(state)[None],
                    noise=torch.from_numpy(noise)[None],
                )
                res.append(a[0].float())
        return torch.stack(res)

    m = load(True)
    tok = m.vlm_with_expert.processor.tokenizer
    base_lang = tok([TASK + "\n"])["input_ids"][0]
    obs = observations(base_lang)
    ref = run(m, obs)
    del m
    d = run(load(False), obs) - ref
    per = d.pow(2).mean(dim=(1, 2)).sqrt()
    print(f"{N} observations · reference action RMS {ref.pow(2).mean().sqrt():.3f}")
    print(
        f"LeRobot bf16 default   max_err {d.abs().max():.3e}  rms {d.pow(2).mean().sqrt():.3e}  "
        f"per-observation rms [{' '.join(f'{v:.1e}' for v in per)}]"
    )
    print("first reference action row of observation 0:", [round(float(v), 4) for v in ref[0, 0, :6]])


if __name__ == "__main__":
    main()
