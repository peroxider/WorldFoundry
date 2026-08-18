"""Train the M3-T1 VAE on synthetic video frames (M3-T6 production-grade gate).

Why this exists
---------------
M3-T6's acceptance needs a *trained* decoder so ``encode -> decode``
reconstruction scores production-grade SSIM/CLIP on real frames. The
M3-T1 :class:`VAEModel` ships with a KL-weighted ``training_step``;
this script trains it on the same procedural synthetic video source the
orchestrator's ``render_phase`` produces, so the eval distribution
matches the training distribution.

Usage
-----
::

    python scripts/train_vae.py \\
        --out models/vae_smoke.pt \\
        --steps 200 --img-size 32 --batch 16 --lr 1e-3

The checkpoint payload is ``{"state_dict": ..., "cfg": {...},
"metrics": {...}}`` — loadable via
:meth:`VAEModel.from_pretrained`.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from worldfoundry_inf.diffusion.vae import VAEConfig, VAEModel


def _synthetic_batch(batch: int, img_size: int, seed: int = 0) -> torch.Tensor:
    """Procedural frames: random blobs / gradients (mirrors cv2 render source).

    Deterministic for a given ``seed`` (same distribution as
    ``scripts/eval_diffusion_rollout.py``).
    """
    rng = np.random.default_rng(seed)
    frames = np.zeros((batch, 3, img_size, img_size), dtype=np.float32)
    for i in range(batch):
        # Random color wash + a soft radial blob.
        base = rng.uniform(0.0, 0.6, size=(3, 1, 1)).astype(np.float32)
        frames[i] = base
        cy, cx = rng.uniform(0.2, 0.8, size=2) * img_size
        radius = img_size * rng.uniform(0.15, 0.35)
        yy, xx = np.mgrid[0:img_size, 0:img_size].astype(np.float32)
        d2 = (yy - cy) ** 2 + (xx - cx) ** 2
        blob = np.exp(-d2 / (2 * radius * radius)).astype(np.float32)
        color = rng.uniform(0.3, 1.0, size=(3, 1, 1)).astype(np.float32)
        frames[i] = frames[i] + blob[None] * color
    return torch.from_numpy(np.clip(frames, 0.0, 1.0))


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    cfg = VAEConfig(latent_dim=args.latent_dim)
    model = VAEModel(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    img = args.img_size

    history: list[dict[str, float]] = []
    best_recon = math.inf
    for step in range(1, args.steps + 1):
        x = _synthetic_batch(args.batch, img, seed=args.seed + step)
        opt.zero_grad()
        loss, metrics = model.training_step(x, kl_weight=args.kl_weight)
        loss.backward()
        opt.step()
        recon = float(metrics["recon"])
        kl = float(metrics["kl"])
        if recon < best_recon:
            best_recon = recon
        history.append({"step": step, "loss": float(loss.detach()), "recon": recon, "kl": kl})
        if step % args.log_every == 0 or step == args.steps:
            print(f"[vae-train] step={step}/{args.steps} loss={float(loss.detach()):.4f} "
                  f"recon={recon:.4f} kl={kl:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "cfg": {
            "in_channels": cfg.in_channels,
            "latent_dim": cfg.latent_dim,
            "base_channels": cfg.base_channels,
            "downsample_steps": cfg.downsample_steps,
            "groups": cfg.groups,
        },
        "metrics": {
            "final_recon_mse": history[-1]["recon"],
            "best_recon_mse": best_recon,
            "final_kl": history[-1]["kl"],
            "steps": args.steps,
            "img_size": img,
            "seed": args.seed,
            "final_loss": history[-1]["loss"],
        },
    }
    torch.save(payload, out)
    print(f"[vae-train] saved {out} (best_recon_mse={best_recon:.5f})")
    metrics_path = out.with_suffix(".json")
    metrics_path.write_text(json.dumps(payload["metrics"], indent=2))
    print(f"[vae-train] metrics -> {metrics_path}")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the M3-T1 VAE on synthetic frames")
    p.add_argument("--out", default="models/vae_smoke.pt")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--img-size", type=int, default=32)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--kl-weight", type=float, default=1e-4)
    p.add_argument("--latent-dim", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=25)
    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
