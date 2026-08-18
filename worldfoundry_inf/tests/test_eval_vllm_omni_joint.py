"""Unit tests for the M2-T6 joint eval script (M3-T7 + M2-T6 + M3-T6).

Covers the joint-eval entry point end-to-end on CPU: component build,
adapter evidence, frame metrics, rollout smoke, and the written
artifacts (joint_eval.json + summary.md).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from scripts.eval_vllm_omni_joint import (
    _synthetic_batch,
    parse_args,
    run_joint_eval,
)


def test_synthetic_batch_shape() -> None:
    x = _synthetic_batch(6, 32, seed=0)
    assert x.shape == (6, 3, 32, 32)
    assert x.min() >= 0.0 and x.max() <= 1.0


def test_joint_eval_writes_artifacts(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--out-dir", str(tmp_path),
            "--n-frames", "4",
            "--img-size", "32",
        ]
    )
    rc = run_joint_eval(args)
    assert rc == 0
    report = json.loads((tmp_path / "joint_eval.json").read_text())
    assert (tmp_path / "summary.md").exists()

    # M3-T7: Omni component resolved to a concrete pipeline.
    assert report["m3_t7"]["pipeline_type"] in (
        "RealVLLMOmniPipeline",
        "VAEDiffusionPipeline",
        "VLLMOmniPipeline",
    )
    assert "vllm_omni_version" in report["m3_t7"]

    # M2-T6: adapter evidence block present.
    assert "m2_t6_adapter" in report
    assert "dispatch" in report["m2_t6_adapter"] or "vllm_version" in report["m2_t6_adapter"]

    # M3-T6: frame metrics present; rollout smoke produced frames.
    fm = report["frame_metrics"]
    assert fm["ssim"] >= 0.0
    assert fm["psnr_db"] > 0.0
    assert report["rollout_shape"] == [1, 2, 3, 32, 32]


def test_joint_eval_reconstruction_computes_metrics(tmp_path: Path) -> None:
    """With the M3-T1 VAE injected, reconstruction metrics are computed."""
    from worldfoundry_inf.diffusion.backends.vae_diffusion import VAEDiffusionPipeline
    from worldfoundry_inf.omni import build_omni_output

    # Build + wrap the real VAE pipeline (random-init is fine — the joint
    # eval's M3-T6 section only asserts metrics are computed; the SSIM
    # quality gate lives in the M3-T6 suite with a trained VAE).
    pipe = VAEDiffusionPipeline(encoder_dim=64, img_size=32, num_steps=8, seed=0)
    import scripts.eval_vllm_omni_joint as joint

    orig = joint.build_omni_output
    joint.build_omni_output = lambda **kw: build_omni_output(pipeline=pipe, **kw)
    try:
        args = parse_args(["--out-dir", str(tmp_path), "--n-frames", "4", "--img-size", "32"])
        run_joint_eval(args)
    finally:
        joint.build_omni_output = orig
    report = json.loads((tmp_path / "joint_eval.json").read_text())
    assert report["m3_t7"]["pipeline_type"] == "VAEDiffusionPipeline"
    assert report["frame_metrics"]["ssim"] >= 0.0
    assert report["frame_metrics"]["psnr_db"] > 0.0


def test_joint_eval_rollout_smoke(tmp_path: Path) -> None:
    """rollout through the Omni component returns (1, T_future, 3, H, W)."""
    from worldfoundry_inf.omni import build_omni_output

    comp = build_omni_output()
    past = _synthetic_batch(4, 32, seed=1).unsqueeze(0)  # (B, T, 3, H, W)
    with torch.no_grad():
        out = comp.rollout(past, actions=None, past_frames=2, future_frames=2)
    assert out.shape == (1, 2, 3, 32, 32)
    assert torch.isfinite(out).all()
