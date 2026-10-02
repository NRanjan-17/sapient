#!/usr/bin/env python3
"""Task success of a SmolVLA policy in the LIBERO simulator — LeRobot vs Sapient.

Runs LIBERO episodes through LeRobot's own environment wrapper and processors
(the exact observation pipeline `lerobot-eval` uses) and swaps only the policy:

  --backend lerobot   LeRobot's PyTorch SmolVLA, f32 on CPU (the reference)
  --backend sapient   `sapient serve` over HTTP (POST /v1/actions), at
                      --precision fast | balanced | exact

Both execute `--exec` actions of each 50-action chunk before asking for the
next one, so the comparison isolates the inference engine and its precision.
`--parity` instead compares one chunk from each backend on the same simulator
frame with the same start noise.

Each finished episode is appended to `--out` as one JSON line; re-running with
the same `--out` skips episodes already recorded.

Needs LeRobot with SmolVLA and LIBERO in a throwaway virtual environment. On
macOS LeRobot's `libero` extra is Linux-only; install `hf-libero==0.1.4` and
`robomimic==0.2.0` with `--no-deps` plus their other requirements (robosuite
1.4.0, bddl 1.0.1, mujoco<3.9, …) — the EGL probe they pull in does not build
on macOS and is not needed (rendering uses MUJOCO_GL=cgl).

Usage:
  sapient serve HuggingFaceVLA/smolvla_libero &
  python3 scripts/vla_sim_eval.py --backend sapient --precision fast \
      --suite libero_spatial --episodes 5 --out results.jsonl
  python3 scripts/vla_sim_eval.py --backend lerobot --suite libero_spatial \
      --episodes 5 --out results.jsonl
  python3 scripts/vla_sim_eval.py --parity --suite libero_spatial
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import time
import urllib.request

os.environ.setdefault("MUJOCO_GL", "cgl")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

IMAGE_KEYS = ("observation.images.image", "observation.images.image2")


def data_uri(img: torch.Tensor) -> str:
    """[3, H, W] float in [0, 1] (exact k/255 values) → PNG data URI."""
    u8 = (img.clamp(0, 1) * 255.0).round().to(torch.uint8).permute(1, 2, 0).numpy()
    buf = io.BytesIO()
    Image.fromarray(u8).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def sapient_chunk(url, model, precision, obs, task, seed=0, noise=None) -> np.ndarray:
    body = {
        "model": model,
        "task": task,
        "images": [data_uri(obs[k][0]) for k in IMAGE_KEYS],
        "state": obs["observation.state"][0].tolist(),
        "precision": precision,
        "seed": seed,
    }
    if noise is not None:
        body["noise"] = noise.tolist()
    req = urllib.request.Request(
        url + "/v1/actions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    return np.asarray(d["actions"], dtype=np.float32)  # [50, action_dim], robot units


class LeRobotPolicy:
    def __init__(self, repo):
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self.policy = SmolVLAPolicy.from_pretrained(repo)
        self.policy.to("cpu")
        self.policy.model.float()
        self.policy.eval()
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=repo,
            preprocessor_overrides={"device_processor": {"device": "cpu"}},
        )

    def chunk(self, obs, task, noise=None) -> np.ndarray:
        batch = dict(obs)
        batch["task"] = [task]
        batch = self.pre(batch)
        with torch.inference_mode():
            kw = {} if noise is None else {"noise": torch.from_numpy(noise)[None]}
            actions = self.policy.predict_action_chunk(batch, **kw)  # [1, 50, action_dim]
        actions = self.post(actions)
        return actions[0].float().cpu().numpy()


def make_libero(suite: str, task_id: int):
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env, make_env_pre_post_processors

    cfg = LiberoEnv(task=suite, task_ids=[task_id])
    envs = make_env(cfg, n_envs=1)
    env = envs[suite][task_id]
    env_pre, env_post = make_env_pre_post_processors(env_cfg=cfg, policy_cfg=None)
    return env, env_pre, env_post


def observe(env_pre, raw, task):
    from lerobot.envs.utils import preprocess_observation

    obs = preprocess_observation(raw)
    obs["task"] = [task]
    return env_pre(obs)


def run_episode(env, env_pre, env_post, get_chunk, n_exec, episode, task_id):
    raw, _ = env.reset(seed=[episode])
    task = list(env.call("task_description"))[0]
    max_steps = env.call("_max_episode_steps")[0]
    queue: list[np.ndarray] = []
    infer_s, n_infer, success, step = 0.0, 0, False, 0
    while step < max_steps:
        obs = observe(env_pre, raw, task)
        if not queue:
            # Common random numbers: every backend gets the same start noise
            # for the same (task, episode, chunk), so differences in outcome
            # come from the engine, not from sampling luck.
            rng = np.random.default_rng([task_id, episode, n_infer])
            noise = rng.standard_normal((50, 32)).astype(np.float32)
            t = time.perf_counter()
            chunk = get_chunk(obs, task, noise)
            infer_s += time.perf_counter() - t
            n_infer += 1
            queue = list(chunk[:n_exec])
        action = torch.from_numpy(queue.pop(0))[None]
        action = env_post({"action": action})["action"]
        raw, _, terminated, truncated, info = env.step(action.numpy())
        step += 1
        if "final_info" in info:
            fi = info["final_info"]
            success = bool(np.asarray(fi.get("is_success", [False]))[0]) if isinstance(fi, dict) else False
        elif "is_success" in info:
            success = bool(np.asarray(info["is_success"])[0])
        if success or bool(np.asarray(terminated)[0]) or bool(np.asarray(truncated)[0]):
            break
    return {"success": success, "steps": step, "inferences": n_infer, "infer_s": round(infer_s, 2), "task": task}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["lerobot", "sapient"], default="sapient")
    ap.add_argument("--precision", default="fast", help="sapient: fast | balanced | exact")
    ap.add_argument("--model", default="HuggingFaceVLA/smolvla_libero")
    ap.add_argument("--url", default="http://127.0.0.1:11435")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--tasks", default="0-9", help="task ids, e.g. 0-9 or 0,3,5")
    ap.add_argument("--episodes", type=int, default=5, help="init states per task (0..N-1)")
    ap.add_argument("--exec", type=int, default=10, help="actions executed per chunk")
    ap.add_argument("--out", default="vla_sim_results.jsonl")
    ap.add_argument("--parity", action="store_true", help="compare one chunk from both backends")
    a = ap.parse_args()

    if "-" in a.tasks:
        lo, hi = map(int, a.tasks.split("-"))
        task_ids = list(range(lo, hi + 1))
    else:
        task_ids = [int(t) for t in a.tasks.split(",")]

    if a.parity:
        env, env_pre, _ = make_libero(a.suite, task_ids[0])
        raw, _ = env.reset(seed=[0])
        task = list(env.call("task_description"))[0]
        obs = observe(env_pre, raw, task)
        noise = torch.randn(50, 32, generator=torch.Generator().manual_seed(7)).numpy()
        ref = LeRobotPolicy(a.model).chunk(obs, task, noise)
        print(f"task: {task}")
        for prec in ("exact", "balanced", "fast"):
            got = sapient_chunk(a.url, a.model, prec, obs, task, noise=noise)
            d = got - ref
            print(
                f"sapient {prec:8} vs LeRobot f32: max {np.abs(d).max():.3e}  "
                f"rms {np.sqrt((d**2).mean()):.3e}  (ref rms {np.sqrt((ref**2).mean()):.3f})"
            )
        return

    label = "lerobot-f32" if a.backend == "lerobot" else f"sapient-{a.precision}"
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line)
            if r["config"] == label and r["suite"] == a.suite and r["exec"] == a.exec:
                done.add((r["task_id"], r["episode"]))

    if a.backend == "lerobot":
        pol = LeRobotPolicy(a.model)
        get_chunk = lambda obs, task, noise: pol.chunk(obs, task, noise)  # noqa: E731
    else:
        get_chunk = lambda obs, task, noise: sapient_chunk(  # noqa: E731
            a.url, a.model, a.precision, obs, task, noise=noise
        )

    for tid in task_ids:
        if all((tid, ep) in done for ep in range(a.episodes)):
            continue
        # LeRobot's LIBERO env walks through the task's fixed init states, one
        # per reset — so episodes run in order, and recorded ones still reset.
        env, env_pre, env_post = make_libero(a.suite, tid)
        for ep in range(a.episodes):
            if (tid, ep) in done:
                env.reset(seed=[ep])
                continue
            t = time.perf_counter()
            r = run_episode(env, env_pre, env_post, get_chunk, a.exec, ep, tid)
            r.update(
                config=label,
                suite=a.suite,
                task_id=tid,
                episode=ep,
                exec=a.exec,
                wall_s=round(time.perf_counter() - t, 1),
            )
            with open(a.out, "a") as f:
                f.write(json.dumps(r) + "\n")
            print(
                f"{label} task {tid} ep {ep}: {'SUCCESS' if r['success'] else 'fail'} "
                f"in {r['steps']} steps, {r['inferences']} chunks, {r['wall_s']} s",
                flush=True,
            )
        env.close()


if __name__ == "__main__":
    main()
