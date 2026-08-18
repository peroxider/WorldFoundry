"""M3-T6 rollout eval: trained-VAE reconstruction → SSIM/PSNR/CLIP gate.

Why this exists
---------------
M3-T6's acceptance is production-grade frame quality from a *real*
decoder: ``SSIM >= 0.85`` and ``CLIP score >= 0.30`` (DEFAULT_THRESHOLDS
in :mod:`worldfoundry_inf.eval.rollout_eval`). This script loads a
trained :class:`VAEModel` checkpoint (from :mod:`scripts.train_vae`),
reconstructs synthetic frames through ``encode -> decode``, and scores
the reconstruction with :func:`compute_rollout_metrics`.

CLIP notes
----------
The CLIP backbone (``openai/clip-vit-base-patch32``) is fetched from
HuggingFace. On air-gapped Atlas, point ``HF_ENDPOINT`` at a mirror::

    HF_ENDPOINT=https://hf-mirror.com python scripts/eval_diffusion_rollout.py ...

When the CLIP model is unavailable the metric is ``null`` and the CLIP
clause is dropped from ``passed`` (per the eval module's contract), but
SSIM/PSNR still gate.

Usage
-----
::

    python scripts/eval_diffusion_rollout.py \\
        --ckpt models/vae_smoke.pt \\
        --out-dir bench_results/_diffusion_eval \\
        --n-frames 8 --img-size 32
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from worldfoundry_inf.diffusion.vae import VAEConfig, VAEModel
from worldfoundry_inf.eval.rollout_eval import compute_rollout_metrics


def _synthetic_batch(n_frames: int, img_size: int, seed: int = 0) -> torch.Tensor:
    """Procedural frames matching scripts/train_vae.py's data distribution."""
    rng = np.random.default_rng(seed)
    frames = np.zeros((n_frames, 3, img_size, img_size), dtype=np.float32)
    for i in range(n_frames):
        frames[i] = rng.uniform(0.0, 0.6, size=(3, 1, 1)).astype(np.float32)
        cy, cx = rng.uniform(0.2, 0.8, size=2) * img_size
        radius = img_size * rng.uniform(0.15, 0.35)
        yy, xx = np.mgrid[0:img_size, 0:img_size].astype(np.float32)
        d2 = (yy - cy) ** 2 + (xx - cx) ** 2
        blob = np.exp(-d2 / (2 * radius * radius)).astype(np.float32)
        color = rng.uniform(0.3, 1.0, size=(3, 1, 1)).astype(np.float32)
        frames[i] = frames[i] + blob[None] * color
    return torch.from_numpy(np.clip(frames, 0.0, 1.0))


def _load_model(ckpt: str | Path) -> VAEModel:
    payload = torch.load(ckpt, map_location="cpu")
    cfg_dict = payload.get("cfg", {})
    cfg = VAEConfig(
        in_channels=cfg_dict.get("in_channels", 3),
        latent_dim=cfg_dict.get("latent_dim", 4),
        base_channels=cfg_dict.get("base_channels", 32),
        downsample_steps=cfg_dict.get("downsample_steps", 3),
        groups=cfg_dict.get("groups", 8),
    )
    model = VAEModel(cfg)
    state = payload["state_dict"] if "state_dict" in payload else payload
    model.load_state_dict(state)
    model.eval()
    return model


def evaluate(args: argparse.Namespace) -> int:
    model = _load_model(args.ckpt)
    frames = _synthetic_batch(args.n_frames, args.img_size, seed=args.seed)
    with torch.no_grad():
        recon = model.reconstruct(frames)  # encode -> decode
    pred = recon.clamp(0.0, 1.0).numpy()
    ref = frames.numpy()

    metrics = compute_rollout_metrics(
        predicted_video=pred,
        reference_video=ref,
        clip_model_id=args.clip_model_id,
    )

    result = {
        "ckpt": str(args.ckpt),
        "n_frames": metrics.n_frames,
        "img_size": args.img_size,
        "ssim": metrics.ssim,
        "psnr_db": metrics.psnr_db,
        "clip_score": metrics.clip_score,
        "latent_frame_alignment": metrics.latent_frame_alignment,
        "passed": metrics.passed,
        "thresholds": metrics.thresholds,
        "notes": [
            "VAE encode->decode reconstruction on synthetic frames",
            "clip_score null when CLIP backbone unavailable (HF_ENDPOINT mirror)",
        ],
    }
    print(json.dumps(result, indent=2))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rollout_eval.json").write_text(json.dumps(result, indent=2))
    (out_dir / "summary.md").write_text(
        f"# M3-T6 rollout eval — {Path(args.ckpt).name}\n\n"
        f"```json\n{json.dumps(result, indent=2)}\n```\n"
    )
    print(f"[eval] wrote {out_dir / 'rollout_eval.json'}")

    ok = metrics.passed
    print(f"[eval] M3-T6 {'PASS' if ok else 'FAIL'}: ssim={metrics.ssim:.4f} "
          f"(>= {metrics.thresholds['ssim']}), "
          f"clip={metrics.clip_score if metrics.clip_score is not None else 'n/a'}"
          f"{'' if metrics.clip_score is None else ' >= ' + str(metrics.thresholds['clip_score'])}")
    return 0 if ok else 1


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M3-T6 rollout eval (VAE reconstruction gate)")
    p.add_argument("--ckpt", required=True, help="trained VAE checkpoint (.pt)")
    p.add_argument("--out-dir", default="bench_results/_diffusion_eval")
    p.add_argument("--n-frames", type=int, default=8)
    p.add_argument("--img-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--clip-model-id", default="openai/clip-vit-base-patch32")
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(evaluate(parse_args()))
