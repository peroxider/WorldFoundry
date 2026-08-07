"""Render -> Train -> Inference orchestrator for the CPU loop (RTI v1).

Drives a complete on-policy-style loop on CPU:

  R. **Render** — generate ``num_clips`` deterministic ``.mp4`` files via
     :func:`worldfoundry_train.data.synthetic_video.write_synthetic_clip`
     + a Vjepa2Dataset-compatible CSV manifest.

  T. **Train** — build a tiny ViT encoder + JEPA predictor + EMA target
     encoder, run ``num_steps`` optimizer steps on the synthetic clips,
     advance the EMA after each step, and save the trained checkpoint +
     pixel decoder to ``<out>/tiny_jepa.pt``.

  I. **Inference** — load the checkpoint, encode one full clip, mask
     half the tokens, ask the predictor for the masked-position
     predictions, splice them back into the sequence, decode the whole
     thing with the trained pixel decoder, and write a side-by-side PNG
     of the original vs. the predicted rendered frames. Also compute
     a cosine similarity between predicted and ground-truth frames as
     the quality probe.

Writes:
  <out>/videos/clip_*.mp4             synthetic clips
  <out>/manifest.csv                  Vjepa2Dataset manifest
  <out>/tiny_jepa.pt                  trained checkpoint
  <out>/render_vs_predicted.png       GT (top) vs. predicted (bottom) stack
  <out>/render_train_inference.json   summary (timings + recon cosine)

The whole loop runs in <60 s on a single CPU thread for the default
config (8 clips x 4 frames, 50 train steps).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path as _Path
# Allow `python scripts/render_train_inference_cpu.py` from repo root without
# requiring PYTHONPATH. We insert the parent of `scripts/` (the repo root)
# only if `worldfoundry_train` is not already importable.
try:
    import worldfoundry_train  # noqa: F401
except ImportError:
    sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from worldfoundry_train.data.dataset import Vjepa2Dataset
from worldfoundry_train.data.synthetic_manifest import (
    SyntheticManifestSpec,
    build_synthetic_manifest,
)
from worldfoundry_train.data.tubelet_mask import tubelet_mask
from worldfoundry_train.device import resolve_device, to_device
from worldfoundry_train.ema_compat import (
    TinyEMAUpdater,
    load_tiny_checkpoint,
    save_tiny_checkpoint,
)
from worldfoundry_train.jepa_objective import jepa_loss
from worldfoundry_train.model.tiny_jepa import (
    TinyEncoderConfig,
    TinyJEPAPredictor,
    TinyPixelDecoder,
    TinyVJEPA2Encoder,
)


@dataclass(frozen=True)
class OrchestratorConfig:
    """All knobs the orchestrator exposes; matches the RTI smoke defaults."""

    out_dir: Path
    num_clips: int = 8
    frames_per_clip: int = 4
    height: int = 64
    width: int = 64
    seed: int = 42
    encoder_embed_dim: int = 96
    encoder_depth: int = 4
    num_steps: int = 50
    learning_rate: float = 3.0e-4
    ema_momentum: float = 0.99
    mask_ratio: float = 0.5
    log_every_n: int = 10


# ---------------------------------------------------------------------------
# R. Render phase
# ---------------------------------------------------------------------------


def render_phase(cfg: OrchestratorConfig) -> tuple[Path, float]:
    """Write ``num_clips`` ``.mp4`` files + a CSV manifest."""
    t0 = time.perf_counter()
    spec = SyntheticManifestSpec(
        out_dir=cfg.out_dir,
        num_clips=cfg.num_clips,
        frames_per_clip=cfg.frames_per_clip,
        height=cfg.height,
        width=cfg.width,
        seed=cfg.seed,
    )
    manifest_path = build_synthetic_manifest(spec)
    elapsed = time.perf_counter() - t0
    return manifest_path, elapsed


# ---------------------------------------------------------------------------
# T. Train phase
# ---------------------------------------------------------------------------


def _build_model_stack(cfg: OrchestratorConfig) -> tuple[
    TinyVJEPA2Encoder, TinyJEPAPredictor, TinyPixelDecoder, TinyVJEPA2Encoder,
]:
    enc_cfg = TinyEncoderConfig(
        image_size=cfg.height, embed_dim=cfg.encoder_embed_dim, depth=cfg.encoder_depth,
    )
    online = TinyVJEPA2Encoder(enc_cfg)
    predictor = TinyJEPAPredictor(enc_cfg)
    decoder = TinyPixelDecoder(enc_cfg)
    target = TinyVJEPA2Encoder(enc_cfg)
    target.load_state_dict(online.state_dict())
    for p in target.parameters():
        p.requires_grad_(False)
    target.eval()
    return online, predictor, decoder, target


def _collate_one(item: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    """Single-sample collate (mirrors collate_vjepa2_batch's batch dim, B=1).

    ``item["mask"]`` is shape ``(T, Hg, Wg)``; the predictor expects
    ``(B, T*Hg*Wg)`` (matches the W22.8 collator's ``.reshape(-1)``).
    """
    return {
        "pixel_values_videos": torch.from_numpy(item["frames"]).unsqueeze(0),
        "bool_masked_pos": torch.from_numpy(item["mask"]).reshape(-1).unsqueeze(0),
    }


def train_phase(
    cfg: OrchestratorConfig,
    manifest_path: Path,
    device: str,
) -> tuple[Path, list[dict[str, float]], float, int]:
    """Train JEPA on the synthetic clips and save the checkpoint."""
    t0 = time.perf_counter()
    online, predictor, decoder, target = _build_model_stack(cfg)
    online = to_device(online, device)
    predictor = to_device(predictor, device)
    decoder = to_device(decoder, device)
    target = to_device(target, device)

    ema = TinyEMAUpdater(momentum=cfg.ema_momentum)

    optimizer = torch.optim.AdamW(
        list(online.parameters()) + list(predictor.parameters()) + list(decoder.parameters()),
        lr=cfg.learning_rate,
    )

    patch = online.config.patch_size
    Hg = cfg.height // patch
    Wg = cfg.width // patch

    def mask_fn(grid, rng):
        return tubelet_mask(
            grid,
            tubelet_size=(2, Hg, Wg),
            mask_ratio=cfg.mask_ratio,
            rng=rng,
        )

    dataset = Vjepa2Dataset(
        manifest_path=manifest_path,
        mask_fn=mask_fn,
        patch_size=patch,
        mask_count=1,
    )
    if len(dataset) == 0:
        raise ValueError("synthetic manifest is empty")

    history: list[dict[str, float]] = []
    rng = np.random.default_rng(cfg.seed + 1)
    online.train()
    predictor.train()
    decoder.train()

    for step in range(cfg.num_steps):
        idx = int(rng.integers(0, len(dataset)))
        sample = dataset[idx]
        batch = _collate_one(sample)
        batch = {k: v.to(device) for k, v in batch.items()}

        # Online encoder forward (trainable).
        h_online = online(batch["pixel_values_videos"]).last_hidden_state
        # Target encoder forward (no grad, EMA snapshot).
        with torch.no_grad():
            h_target = target(batch["pixel_values_videos"]).last_hidden_state
        # Predictor at the masked positions; returns (M, embed_dim).
        bool_mask = batch["bool_masked_pos"].bool()
        pred_at_mask = predictor(h_online, bool_mask)  # (M, embed_dim)
        # Target features at the same positions.
        target_at_mask = h_target[bool_mask]
        # W22-style JEPA loss (smooth_l1).
        loss_jepa = jepa_loss(pred_at_mask, target_at_mask, loss_type="smooth_l1", beta=1.0)
        # Tiny pixel-decoder reconstruction supervised against input frames.
        recon_pixels = decoder(h_online, num_frames=cfg.frames_per_clip)
        loss_recon = torch.nn.functional.l1_loss(
            recon_pixels, batch["pixel_values_videos"],
        )
        loss = loss_jepa + loss_recon

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(online.parameters()) + list(predictor.parameters()) + list(decoder.parameters()),
            max_norm=1.0,
        )
        optimizer.step()
        # EMA advance: lerp target <- momentum * target + (1 - momentum) * online
        ema.update(target, online)

        history.append({
            "step": step,
            "jepa_loss": float(loss_jepa.detach().cpu().item()),
            "recon_loss": float(loss_recon.detach().cpu().item()),
            "total_loss": float(loss.detach().cpu().item()),
        })
        if step == 0 or (step + 1) % cfg.log_every_n == 0 or step == cfg.num_steps - 1:
            print(
                f"[RTI][T] step={step + 1}/{cfg.num_steps} "
                f"jepa_loss={loss_jepa.item():+.4f} "
                f"recon_loss={loss_recon.item():+.4f} "
                f"total_loss={loss.item():+.4f}"
            )

    ckpt_path = cfg.out_dir / "tiny_jepa.pt"
    save_tiny_checkpoint(
        ckpt_path,
        encoder=online,
        predictor=predictor,
        decoder=decoder,
        optimizer=optimizer,
        step=len(history),
    )
    elapsed = time.perf_counter() - t0
    n_params = sum(p.numel() for p in online.parameters())
    return ckpt_path, history, elapsed, n_params


# ---------------------------------------------------------------------------
# I. Inference phase
# ---------------------------------------------------------------------------


def _save_side_by_side_png(
    frames_gt: np.ndarray,
    frames_pred: np.ndarray,
    out_path: Path,
) -> None:
    """Save a 2-row stack: GT frames on top, predicted frames on bottom.

    Each frame is a ``(H, W, 3)`` uint8 array (BGR). We stack ``T``
    frames horizontally and 2 rows vertically, separated by a 4-px white
    gutter.
    """
    T = frames_gt.shape[0]
    H, W = frames_gt.shape[1], frames_gt.shape[2]
    canvas = np.full((H * 2 + 4, W * T + (T - 1) * 2, 3), 255, dtype=np.uint8)
    for t in range(T):
        x_off = t * (W + 2)
        canvas[:H, x_off : x_off + W] = frames_gt[t]
        canvas[H + 4 :, x_off : x_off + W] = frames_pred[t]
    cv2.imwrite(str(out_path), canvas)


def _per_frame_cosine(
    gt_pixels: np.ndarray,
    pred_pixels: np.ndarray,
) -> float:
    """Mean cosine similarity between ``T`` corresponding (GT, predicted) pairs.

    Args:
        gt_pixels / pred_pixels: ``(T, C, H, W)`` float in [0, 1].
    """
    T = gt_pixels.shape[0]
    a = gt_pixels.reshape(T, -1)
    b = pred_pixels.reshape(T, -1)
    norms = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    # Avoid /0: zeros on both sides → cosine is 1 by convention (degenerate).
    sims = np.where(norms > 1e-8, (a * b).sum(axis=-1) / norms, 1.0)
    return float(sims.mean())


def inference_phase(
    cfg: OrchestratorConfig,
    ckpt_path: Path,
    device: str,
) -> dict[str, Any]:
    """Load checkpoint, run one inference rollout, save PNG + metrics."""
    t0 = time.perf_counter()
    bundle = load_tiny_checkpoint(ckpt_path, device=device)
    encoder = bundle["encoder"]
    predictor = bundle["predictor"]
    decoder = bundle["decoder"]
    encoder.eval()
    predictor.eval()
    decoder.eval()

    patch = encoder.config.patch_size
    Hg = cfg.height // patch
    Wg = cfg.width // patch

    def mask_fn(grid, rng):
        return tubelet_mask(
            grid,
            tubelet_size=(2, Hg, Wg),
            mask_ratio=cfg.mask_ratio,
            rng=rng,
        )
    manifest_path = cfg.out_dir / "manifest.csv"
    if not manifest_path.is_file():
        spec = SyntheticManifestSpec(
            out_dir=cfg.out_dir,
            num_clips=cfg.num_clips,
            frames_per_clip=cfg.frames_per_clip,
            height=cfg.height,
            width=cfg.width,
            seed=cfg.seed,
        )
        manifest_path = build_synthetic_manifest(spec)
    dataset = Vjepa2Dataset(
        manifest_path=manifest_path,
        mask_fn=mask_fn,
        patch_size=patch,
        mask_count=1,
    )
    sample = dataset[0]
    frames_gt = sample["frames"]  # (T, 3, H, W) float32 in [0, 1]

    with torch.no_grad():
        x = torch.from_numpy(frames_gt).unsqueeze(0).to(device)  # (1, T, 3, H, W)
        h = encoder(x).last_hidden_state  # (1, T*Hg*Wg, embed_dim)
        mask = torch.from_numpy(sample["mask"]).reshape(-1).unsqueeze(0).to(device).bool()
        predicted_at_masked = predictor(h, mask)  # (M, embed_dim)
        full_pred = h.clone()
        M_total = predicted_at_masked.shape[0]
        full_pred[mask] = predicted_at_masked
        pixels = decoder(full_pred, num_frames=cfg.frames_per_clip)
    pixels = pixels.squeeze(0).clamp(0.0, 1.0).cpu().numpy()  # (T, 3, H, W)
    # cv2.imwrite expects BGR uint8.
    frames_gt_bgr = (frames_gt.transpose(0, 2, 3, 1) * 255.0).round().astype(np.uint8)[:, :, :, ::-1]
    frames_pred_bgr = (pixels.transpose(0, 2, 3, 1) * 255.0).round().astype(np.uint8)[:, :, :, ::-1]

    out_png = cfg.out_dir / "render_vs_predicted.png"
    _save_side_by_side_png(frames_gt_bgr, frames_pred_bgr, out_png)

    recon_cosine = _per_frame_cosine(frames_gt, pixels)
    elapsed = time.perf_counter() - t0
    return {
        "elapsed_seconds": elapsed,
        "recon_cosine": recon_cosine,
        "png_path": str(out_png),
        "n_params": sum(p.numel() for p in encoder.parameters()),
        "n_predicted_tokens": int(M_total),
    }


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU render->train->inference orchestrator (RTI v1)")
    parser.add_argument("--out-dir", type=Path, default=Path("bench_results/render_train_inference"))
    parser.add_argument("--num-clips", type=int, default=8)
    parser.add_argument("--frames-per-clip", type=int, default=4)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--skip-render", action="store_true",
                        help="Reuse existing manifest (used by tests).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = OrchestratorConfig(
        out_dir=args.out_dir,
        num_clips=args.num_clips,
        frames_per_clip=args.frames_per_clip,
        height=args.height,
        width=args.width,
        seed=args.seed,
        num_steps=args.num_steps,
        learning_rate=args.learning_rate,
        mask_ratio=args.mask_ratio,
    )
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device()

    if args.skip_render:
        manifest_path = cfg.out_dir / "manifest.csv"
        if not manifest_path.is_file():
            print(f"[RTI] FATAL: --skip-render but manifest missing: {manifest_path}",
                  file=sys.stderr)
            return 2
        render_elapsed = 0.0
    else:
        manifest_path, render_elapsed = render_phase(cfg)
    print(f"[RTI][R] manifest={manifest_path} elapsed={render_elapsed:.2f}s")

    ckpt_path, history, train_elapsed, n_params = train_phase(cfg, manifest_path, device)
    print(f"[RTI][T] ckpt={ckpt_path} elapsed={train_elapsed:.2f}s "
          f"final_total_loss={history[-1]['total_loss']:+.4f} "
          f"encoder_params={n_params:,}")

    inference = inference_phase(cfg, ckpt_path, device)
    print(f"[RTI][I] recon_cosine={inference['recon_cosine']:+.4f} "
          f"png={inference['png_path']} elapsed={inference['elapsed_seconds']:.2f}s")

    summary = {
        "label": "RTI v1 CPU render->train->inference",
        "mode": "cpu",
        "config_serializable": {
            "out_dir": str(cfg.out_dir),
            "num_clips": cfg.num_clips,
            "frames_per_clip": cfg.frames_per_clip,
            "height": cfg.height,
            "width": cfg.width,
            "seed": cfg.seed,
            "num_steps": cfg.num_steps,
            "learning_rate": cfg.learning_rate,
            "mask_ratio": cfg.mask_ratio,
        },
        "render": {"elapsed_seconds": round(render_elapsed, 3)},
        "train": {
            "elapsed_seconds": round(train_elapsed, 3),
            "final_total_loss": round(history[-1]["total_loss"], 4),
            "final_jepa_loss": round(history[-1]["jepa_loss"], 4),
            "final_recon_loss": round(history[-1]["recon_loss"], 4),
            "n_encoder_params": n_params,
            "n_train_steps": len(history),
        },
        "inference": {
            "elapsed_seconds": round(inference["elapsed_seconds"], 3),
            "recon_cosine": round(inference["recon_cosine"], 4),
            "png_path": inference["png_path"],
            "n_predicted_tokens": inference["n_predicted_tokens"],
        },
        "notes": [
            "RTI v1 wires Render -> Train -> Inference on CPU without HF V-JEPA 2 1.2B download.",
            "TinyViT encoder (~325k params) + TinyJEPAPredictor (~96k params) + TinyPixelDecoder (~28k params).",
            "Real .mp4 files written via cv2.VideoWriter; loaded via existing Vjepa2Dataset -> decord.",
            "Cosine sim between GT frames and decoder-rendered prediction is the quality probe.",
            "EMA target encoder is honoured; momentum=0.99 (CPU-feasible constant schedule).",
        ],
    }
    summary_path = cfg.out_dir / "render_train_inference.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[RTI] summary: {summary_path}")
    print(f"[RTI] summary: {json.dumps(summary)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())