#!/usr/bin/env python3
"""Real robot observations for checking SmolVLA in Sapient.

Samples frames from a LeRobot SO-100 dataset (default `lerobot/svla_so100_pickplace`,
two cameras, 30 fps), and for each one stores what the Rust side needs to run the
same observation, plus LeRobot's own outputs on it:

  obs.{i}.top / obs.{i}.wrist   camera frames, 0..255 as f16 [480, 640, 3]
  obs.{i}.lang                  instruction token ids ("{task}\\n", no special tokens)
  obs.{i}.state                 robot state, normalized with the dataset's mean/std
  obs.{i}.noise                 flow-matching start noise [50, 32] (seeded)
  obs.{i}.gt                    the recorded next 50 actions, normalized the same way
  obs.{i}.lerobot_f32           LeRobot's action chunk, all f32 [50, 32]
  obs.{i}.lerobot_bf16          LeRobot's action chunk at its default precision

The Rust test `smolvla_real_observations` (crates/sapient-generate/tests/vla_e2e.rs)
compares Sapient's `exact` / `balanced` / `fast` modes against these.

Normalization note: `lerobot/smolvla_base` ships no state statistics for this robot,
so state and ground-truth actions are normalized with the DATASET's own statistics.
The base model was pretrained on SO-100 community data but not fine-tuned on this
dataset; the ground-truth comparison is an offline prediction error, not task success.

Integer tensors are stored as f32 (Sapient's safetensors loader reads floats only).
Needs `pip install "lerobot[smolvla]"` (throwaway virtual environment). Writes ~90 MB:

  python3 scripts/smolvla_dataset_eval.py --out ~/.cache/sapient-bench/smolvla_so100.safetensors
"""
import argparse
import os

import numpy as np
import torch
from safetensors.torch import save_file

CHUNK = 50


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default="lerobot/svla_so100_pickplace")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--per-episode", type=int, default=3)
    a = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.common.vla_utils import resize_with_pad
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    fps = 30
    ds = LeRobotDataset(
        a.dataset, delta_timestamps={"action": [i / fps for i in range(CHUNK)]}
    )
    stats = ds.meta.stats
    s_mean = torch.as_tensor(stats["observation.state"]["mean"]).float()
    s_std = torch.as_tensor(stats["observation.state"]["std"]).float()
    a_mean = torch.as_tensor(stats["action"]["mean"]).float()
    a_std = torch.as_tensor(stats["action"]["std"]).float()

    # Frames spread over episodes, away from episode ends (the chunk needs 50 future steps).
    ep_from = ds.meta.episodes["dataset_from_index"]
    ep_to = ds.meta.episodes["dataset_to_index"]
    n_ep = len(ep_from)
    episodes = np.linspace(0, n_ep - 1, a.episodes).round().astype(int)
    picks = []
    for e in episodes:
        lo, hi = int(ep_from[e]), int(ep_to[e]) - CHUNK
        for f in np.linspace(0.1, 0.8, a.per_episode):
            picks.append(int(lo + f * (hi - lo)))

    def load(f32: bool):
        policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
        policy.to("cpu")
        if f32:
            policy.model.float()
        policy.eval()
        return policy

    pol = load(True)
    tok = pol.model.vlm_with_expert.processor.tokenizer
    gen = torch.Generator().manual_seed(4321)
    out: dict[str, torch.Tensor] = {}
    inputs = []
    for i, idx in enumerate(picks):
        item = ds[idx]
        frames = {}
        for cam in ("top", "wrist"):
            img = item[f"observation.images.{cam}"]  # float [3, H, W] in [0, 1]
            u8 = (img * 255.0).round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0)
            frames[cam] = u8
            out[f"obs.{i}.{cam}"] = u8.half().contiguous()  # f16 holds 0..255 exactly
        task = item["task"]
        ids = tok([task + "\n"], add_special_tokens=False)["input_ids"][0][:48]
        state = (item["observation.state"].float() - s_mean) / s_std
        noise = torch.randn(CHUNK, 32, generator=gen)
        gt = (item["action"].float() - a_mean) / a_std  # [50, 6]
        out[f"obs.{i}.lang"] = torch.tensor(ids, dtype=torch.float32)
        out[f"obs.{i}.state"] = state
        out[f"obs.{i}.noise"] = noise
        out[f"obs.{i}.gt"] = gt
        inputs.append((frames, ids, state, noise))
        print(f"obs {i}: frame {idx}, task {task!r}, {len(ids)} tokens")

    def run(policy):
        m = policy.model
        pad = tok.pad_token_id
        res = []
        with torch.no_grad():
            for frames, ids, state, noise in inputs:
                imgs = []
                for cam in ("top", "wrist"):
                    img = frames[cam].permute(2, 0, 1)[None].float() / 255.0
                    img = resize_with_pad(img, 512, 512, pad_value=0) * 2.0 - 1.0
                    imgs.append(img)
                lang = torch.full((1, 48), pad, dtype=torch.long)
                lang[0, : len(ids)] = torch.tensor(ids)
                mask = torch.zeros(1, 48, dtype=torch.bool)
                mask[0, : len(ids)] = True
                st = torch.zeros(1, 32)
                st[0, : state.numel()] = state
                act = m.sample_actions(
                    imgs,
                    [torch.ones(1, dtype=torch.bool)] * 2,
                    lang,
                    mask,
                    st,
                    noise=noise[None],
                )
                res.append(act[0].float())
        return res

    for i, act in enumerate(run(pol)):
        out[f"obs.{i}.lerobot_f32"] = act
    del pol
    for i, act in enumerate(run(load(False))):
        out[f"obs.{i}.lerobot_bf16"] = act

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    save_file(
        {k: v.contiguous() for k, v in out.items()},
        a.out,
        metadata={"dataset": a.dataset, "observations": str(len(picks)), "frames": str(picks)},
    )

    # Summary (normalized units, first 6 action dims).
    f32 = torch.stack([out[f"obs.{i}.lerobot_f32"][:, :6] for i in range(len(picks))])
    bf16 = torch.stack([out[f"obs.{i}.lerobot_bf16"][:, :6] for i in range(len(picks))])
    gt = torch.stack([out[f"obs.{i}.gt"] for i in range(len(picks))])
    rms = lambda x: x.pow(2).mean().sqrt().item()  # noqa: E731
    print(f"{len(picks)} observations · ground-truth action RMS {rms(gt):.3f}")
    print(f"LeRobot bf16 vs f32      rms {rms(bf16 - f32):.3e}  max {(bf16 - f32).abs().max():.3e}")
    print(f"LeRobot f32 vs recorded  rms {rms(f32 - gt):.3e}")
    print(f"LeRobot bf16 vs recorded rms {rms(bf16 - gt):.3e}")


if __name__ == "__main__":
    main()
