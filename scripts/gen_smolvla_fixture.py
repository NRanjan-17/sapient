#!/usr/bin/env python3
"""Generate the SmolVLA reference fixture the Rust port is gated against.

Runs LeRobot's `lerobot/smolvla_base` in f32 on CPU on fixed inputs and saves the
inputs plus every intermediate the port has to reproduce:

  inputs        language token ids + mask, state (padded to 32), noise (50x32);
                the 512x512 camera image is procedural (`test_image`) and is
                rebuilt by the Rust test from the same formula, not stored
  vision        image embedding after SigLIP + connector            [64, 960]
  prefix        embed_prefix output, padding/attention masks, position ids
  prefix K/V    post-RoPE keys and raw values of VLM layers 0, 1 and 15
  prefix out    final-norm hidden states of the 16-layer prefix pass
  suffix        embed_suffix(noise, t=1.0)                         [50, 720]
  velocity      v_t of the first denoising step                    [50, 32]
  actions       the full 10-step Euler result (normalized space)   [50, 32]
  stats         state / action mean and std from the checkpoint's processors

Integer tensors are stored as f32 (Sapient's safetensors loader reads floats only).

One camera is used (the model accepts any subset of its three cameras), so the
prefix is 64 image tokens + 48 language tokens + 1 state token = 113 positions.

Needs LeRobot with its SmolVLA extra (`pip install "lerobot[smolvla]"`), ideally in a
throwaway virtual environment — it is NOT a dependency of this repository.

Usage:
  python3 scripts/gen_smolvla_fixture.py \
      --out crates/sapient-models/tests/fixtures/smolvla_base.safetensors
"""
import argparse
import json

import numpy as np
import torch
from safetensors.torch import load_file, save_file

TASK = "Pick up the red cube and place it in the box."
KV_LAYERS = (0, 1, 15)


def test_image() -> np.ndarray:
    """Deterministic 512x512 RGB test image (gradient + red block), u8.

    Already at the model's input size, so no resize is involved. Mirrored by
    `test_image` in `crates/sapient-models/tests/smolvla_reference.rs`.
    """
    size = 512
    y, x = np.mgrid[0:size, 0:size]
    img = np.stack([x * 255 // size, y * 255 // size, (x * 7 + y * 13) % 256], -1).astype(np.uint8)
    img[160:352, 144:368] = (217, 26, 26)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default="lerobot/smolvla_base")
    a = ap.parse_args()

    import lerobot
    import transformers
    from huggingface_hub import hf_hub_download
    from lerobot.policies.common.vla_utils import make_att_2d_masks, resize_with_pad
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    torch.manual_seed(0)
    policy = SmolVLAPolicy.from_pretrained(a.repo)
    policy.to("cpu")
    policy.model.float()  # the VLM loads as bf16; the reference is f32
    policy.eval()
    model = policy.model
    cfg = policy.config
    out: dict[str, torch.Tensor] = {}

    with torch.no_grad():
        # ── inputs ───────────────────────────────────────────────────────────
        raw = test_image()
        img = torch.from_numpy(raw).permute(2, 0, 1)[None].float() / 255.0  # [1, 3, 512, 512]
        w, h = cfg.resize_imgs_with_padding  # stored as (width, height)
        img = resize_with_pad(img, h, w, pad_value=0)
        img = img * 2.0 - 1.0
        images, img_masks = [img], [torch.ones(1, dtype=torch.bool)]

        tok = model.vlm_with_expert.processor.tokenizer
        enc = tok(
            [TASK + "\n"],  # smolvla_new_line_processor appends the newline
            max_length=cfg.tokenizer_max_length,
            padding="max_length",
            padding_side="right",
            truncation=True,
            return_tensors="pt",
        )
        lang_tokens, lang_masks = enc["input_ids"], enc["attention_mask"].bool()
        out["input.lang_tokens"] = lang_tokens[0].float()
        out["input.lang_mask"] = lang_masks[0].float()

        g = torch.Generator().manual_seed(1234)
        state = torch.zeros(1, cfg.max_state_dim)
        state[0, :6] = torch.randn(6, generator=g) * 0.5  # already in normalized space
        noise = torch.randn(1, cfg.chunk_size, cfg.max_action_dim, generator=g)
        out["input.state"] = state[0]
        out["input.noise"] = noise[0]

        # ── vision + prefix ──────────────────────────────────────────────────
        out["vision.image_embedding"] = model.vlm_with_expert.embed_image(img)[0].half()

        prefix_embs, prefix_pad, prefix_att = model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        att_2d = make_att_2d_masks(prefix_pad, prefix_att)
        pos = torch.cumsum(prefix_pad, dim=1) - 1
        out["prefix.embs"] = prefix_embs[0].half()
        out["prefix.pad_mask"] = prefix_pad[0].float()
        out["prefix.att_mask"] = prefix_att[0].float()
        out["prefix.att_2d"] = att_2d[0].float()
        out["prefix.position_ids"] = pos[0].float()

        (prefix_out, _), pkv = model.vlm_with_expert.forward(
            attention_mask=att_2d,
            position_ids=pos,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        out["prefix.out"] = prefix_out[0].half()
        for i in KV_LAYERS:
            # DynamicCache layout: [batch, kv_heads, seq, head_dim]
            out[f"prefix.kv.{i}.keys"] = pkv.layers[i].keys[0].half()
            out[f"prefix.kv.{i}.values"] = pkv.layers[i].values[0].half()

        # ── suffix, one velocity, full sample ────────────────────────────────
        t1 = torch.tensor([1.0], dtype=torch.float32)
        suffix_embs, _, _ = model.embed_suffix(noise, t1)
        out["suffix.embs_t1"] = suffix_embs[0]
        out["denoise.v_t1"] = model.denoise_step(
            prefix_pad_masks=prefix_pad, past_key_values=pkv, x_t=noise, timestep=t1
        )[0]
        out["actions.normalized"] = model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise
        )[0]

    # ── normalization statistics from the checkpoint's processors ───────────
    for fname, prefix in [
        ("policy_preprocessor_step_5_normalizer_processor.safetensors", "stats.pre."),
        ("policy_postprocessor_step_0_unnormalizer_processor.safetensors", "stats.post."),
    ]:
        for k, v in load_file(hf_hub_download(a.repo, fname)).items():
            if v.numel() <= 64:
                out[prefix + k] = v.float()

    meta = {
        "task": TASK,
        "repo": a.repo,
        "lerobot": lerobot.__version__,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "num_steps": str(cfg.num_steps),
        "kv_layers": json.dumps(KV_LAYERS),
        "note": "f32 CPU reference; tensors stored as f16 are marked by dtype",
    }
    save_file({k: v.contiguous() for k, v in out.items()}, a.out, metadata=meta)
    for k, v in sorted(out.items()):
        print(f"{k:34s} {str(v.dtype):14s} {tuple(v.shape)}")
    print("meta", meta)


if __name__ == "__main__":
    main()
