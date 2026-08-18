"""M2-T6 joint eval: M3-T7 Omni output + vllm-omni adapter evidence (P4 close).

Why this exists
---------------
P4's completion criterion: "vllm-omni 公开后一次性把 M3-T7 + M2-T6 跑完，
写入 ``bench_results/wXX_vllm_omni_joint/``". This script is that single
entry point:

* **M3-T7 (编排)**: :func:`build_omni_output` resolves the Omni
  multimodal output component (real vLLM path first, then the M3-T1
  VAE+DiT pipeline, then the ORT stand-in) — proving the diffusion
  latent-to-frame pipeline is wired as Omni's output component.
* **M2-T6 (eval / vision-encoder adapter point)**: the component's
  ``provider_summary()`` records ``vllm_omni_version`` (the public
  wheel, now released) + the real vllm package probe (``vllm_version``
  / ``llm_class_import_ok``) — the JEPA-latent-as-vision-encoder
  adapter evidence.
* **M3-T6 metrics**: encode→decode reconstruction + rollout are scored
  with :func:`compute_rollout_metrics` (SSIM/PSNR/CLIP) on synthetic
  frames, so the joint run also exercises the frame-quality gate.

Usage
-----
::

    python scripts/eval_vllm_omni_joint.py \\
        --out-dir bench_results/w46_vllm_omni_joint \\
        --n-frames 8 --img-size 32
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from worldfoundry_inf.eval.rollout_eval import compute_rollout_metrics
from worldfoundry_inf.omni import build_omni_output


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


def _resolve_img_size(comp: Any, fallback: int) -> int:
    """Detect the img_size the resolved pipeline actually accepts.

    ``RealVLLMOmniPipeline`` / ``VLLMOmniPipeline`` build their ORT
    inference worker with ``match_diffusion_defaults=True`` -> 64x64
    (matching ``OnnxDiffusionConfig``); ``VAEDiffusionPipeline``
    defaults to 32x32. Using a mismatched size fails ONNX with
    "Got invalid dimensions for input: pixel_values".
    """
    pipe = comp.pipeline
    if hasattr(pipe, "img_size"):
        return int(pipe.img_size)
    diffusion = getattr(pipe, "diffusion", None)
    cfg = getattr(diffusion, "cfg", None)
    if cfg is not None and hasattr(cfg, "img_size"):
        return int(cfg.img_size)
    return fallback


def run_joint_eval(args: argparse.Namespace) -> int:
    """Run the M3-T7 + M2-T6 joint eval and write the bench artifacts."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- M3-T7: build the Omni output component ---------------------
    comp = build_omni_output()
    summary = comp.provider_summary()
    print(f"[joint] Omni component -> {summary['pipeline_type']} "
          f"(vllm_omni={summary['vllm_omni_version'] or 'n/a'})")

    # ---- M2-T6: adapter evidence (vllm-omni wheel + real vllm probe) --
    extras = summary.get("extras", {})
    if summary["pipeline_type"] == "RealVLLMOmniPipeline":
        pipe = comp.pipeline
        probe = getattr(pipe, "probe", None)
        adapter = {
            "vllm_version": getattr(probe, "vllm_version", "unknown"),
            "llm_class_import_ok": bool(getattr(probe, "llm_class_import_ok", False)),
            "llm_class_path": getattr(probe, "llm_class_path", ""),
        }
    else:
        adapter = {"dispatch": extras.get("dispatch"), "pipeline_type": summary["pipeline_type"]}
    print(f"[joint] adapter: {json.dumps(adapter)}")

    # Resolve the pipeline's actual input resolution (64 for the ORT
    # stand-in / real-vllm pipeline, 32 for VAEDiffusionPipeline).
    img = _resolve_img_size(comp, args.img_size)

    # ---- M3-T6: frame-quality metrics on VAE reconstruction + rollout ----
    # The Omni component's encode/decode surface is NOT a symmetric
    # round-trip for VAEDiffusionPipeline (encode -> VAE latent dim=4,
    # decode <- JEPA condition dim=encoder_dim). Score reconstruction
    # through the underlying VAE's encode->decode path when available;
    # pipelines without a reconstruct path record null metrics.
    frames = _synthetic_batch(args.n_frames, img, seed=args.seed)
    recon = None
    if hasattr(comp.pipeline, "reconstruct"):
        with torch.no_grad():
            recon = comp.pipeline.reconstruct(frames)
    elif hasattr(comp.pipeline, "vae") and hasattr(comp.pipeline.vae, "reconstruct"):
        with torch.no_grad():
            recon = comp.pipeline.vae.reconstruct(frames)
    if recon is not None:
        pred = recon.clamp(0.0, 1.0).numpy()  # (T, 3, H, W)
        ref = frames.numpy()
        metrics = compute_rollout_metrics(predicted_video=pred, reference_video=ref,
                                          clip_model_id=args.clip_model_id)
        frame_metrics = {
            "ssim": metrics.ssim,
            "psnr_db": metrics.psnr_db,
            "clip_score": metrics.clip_score,
            "n_frames": metrics.n_frames,
            "passed": metrics.passed,
            "thresholds": metrics.thresholds,
        }
    else:
        frame_metrics = {
            "ssim": None, "psnr_db": None, "clip_score": None,
            "n_frames": args.n_frames, "passed": None, "thresholds": None,
            "note": "pipeline has no VAE reconstruct path; metrics skipped",
        }

    # ---- rollout smoke through the component surface ----
    # rollout requires (B, T_past+T_future, 3, H, W); prepend batch.
    past = _synthetic_batch(args.n_frames, img, seed=args.seed + 1).unsqueeze(0)
    with torch.no_grad():
        rolled = comp.rollout(past, actions=None, past_frames=2, future_frames=2)

    result = {
        "m3_t7": {
            "pipeline_type": summary["pipeline_type"],
            "pipeline_kind": summary.get("pipeline_kind"),
            "vllm_omni_version": summary["vllm_omni_version"],
            "extras": extras,
        },
        "m2_t6_adapter": adapter,
        "frame_metrics": frame_metrics,
        "rollout_shape": list(rolled.shape),
        "notes": [
            "P4 joint eval: M3-T7 Omni output + M2-T6 vllm-omni adapter + M3-T6 metrics",
            "clip_score null when CLIP backbone unavailable (HF_ENDPOINT mirror on Atlas)",
        ],
    }
    (out_dir / "joint_eval.json").write_text(json.dumps(result, indent=2))
    (out_dir / "summary.md").write_text(
        f"# W46 — vLLM-Omni joint eval (M3-T7 + M2-T6)\n\n"
        f"```json\n{json.dumps(result, indent=2)}\n```\n"
    )
    print(f"[joint] wrote {out_dir / 'joint_eval.json'}")
    print(f"[joint] ssim={metrics.ssim:.4f} psnr={metrics.psnr_db:.2f} "
          f"clip={metrics.clip_score if metrics.clip_score is not None else 'n/a'} "
          f"rollout={list(rolled.shape)}")
    return 0


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M2-T6 joint eval (M3-T7 + M2-T6 + M3-T6 metrics)")
    p.add_argument("--out-dir", default="bench_results/w46_vllm_omni_joint")
    p.add_argument("--n-frames", type=int, default=8)
    p.add_argument("--img-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--clip-model-id", default="openai/clip-vit-base-patch32")
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run_joint_eval(parse_args()))
