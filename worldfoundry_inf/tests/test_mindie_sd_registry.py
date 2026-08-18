"""Tests for the M3-T8 mindie_sd backend wiring to VAEDiffusionPipeline.

Covers the registry-level dispatch contract: ``build("mindie_sd")`` now
returns the real v2 pipeline (:class:`VAEDiffusionPipeline`) instead of
the v1 CPU pixel-decoder stub, the ``patch_size`` v1-only kwarg is
dropped, and the orchestrator's ``frame_gen_phase`` runs it end-to-end
with the documented shapes.
"""
from __future__ import annotations

from worldfoundry_common.backends.inference import (
    _build_mindie_sd,
    _mindie_sd_available,
    registry,
)
from worldfoundry_inf.diffusion.backends.vae_diffusion import VAEDiffusionPipeline


def test_mindie_sd_available() -> None:
    """v1 probe stays True (in-tree diffusion scaffolding imports)."""
    assert _mindie_sd_available() is True
    available, _hint = registry.available("mindie_sd")
    assert available is True


def test_mindie_sd_build_returns_vae_diffusion_pipeline() -> None:
    """M3-T8: registry build returns the real v2 pipeline, not the stub."""
    pipe = registry.build("mindie_sd")
    assert isinstance(pipe, VAEDiffusionPipeline)


def test_mindie_sd_build_forwards_kwargs() -> None:
    """encoder_dim/img_size/num_steps reach the v2 pipeline."""
    pipe = _build_mindie_sd(
        encoder_dim=32, img_size=16, latent_dim=4, num_steps=4, seed=1
    )
    assert isinstance(pipe, VAEDiffusionPipeline)
    assert pipe.encoder_dim == 32
    assert pipe.img_size == 16
    assert pipe.num_steps == 4


def test_mindie_sd_build_drops_patch_size() -> None:
    """The orchestrator passes patch_size (v1-only); v2 drops it silently."""
    pipe = _build_mindie_sd(img_size=32, patch_size=4)
    assert isinstance(pipe, VAEDiffusionPipeline)
    assert pipe.img_size == 32


def test_mindie_sd_pipeline_available() -> None:
    """VAEDiffusionPipeline is pure PyTorch -> available on CPU hosts."""
    pipe = registry.build("mindie_sd")
    assert pipe.available() is True


def test_mindie_sd_pipeline_provider_summary_kind() -> None:
    """The dispatch fingerprint identifies the v2 pipeline."""
    pipe = registry.build("mindie_sd")
    summary = pipe.provider_summary()
    assert summary["engine"] == "vae_diffusion"
    assert summary["kind"] == "vae_diffusion"
    assert summary["available"] is True


def test_mindie_sd_pipeline_encode_decode_shapes() -> None:
    """Real architecture round-trip through the registry-built pipeline."""
    import torch

    pipe = registry.build("mindie_sd")  # default img_size=32
    video = torch.randn(1, 2, 3, 32, 32)
    latent = pipe.encode_latents(video)
    assert latent.shape == (1, 2 * 16, 4)
    frames = pipe.decode_latents(torch.randn(1, 16, 64), grid=(1, 4, 4))
    assert frames.shape == (1, 1, 3, 32, 32)
    assert torch.isfinite(frames).all()


def test_frame_gen_phase_mindie_sd_backend() -> None:
    """frame_gen_phase dispatches mindie_sd and records the fingerprint."""
    import torch

    from scripts.render_train_rl_cpu import V2Config, frame_gen_phase

    cfg = V2Config(out_dir=".", height=32, width=32)
    # T must equal past_frames + future_frames = 4 (rollout contract).
    predicted = torch.randn(4, 3, 32, 32)  # (T, 3, H, W) numpy-like 4-D
    out = frame_gen_phase(
        cfg,
        backend="mindie_sd",
        device="cpu",
        predicted_video=predicted.numpy(),
        past_frames=2,
        future_frames=2,
    )
    fg = out  # frame_gen_phase returns {backend, elapsed_seconds, frame_metrics, frame_gen_dispatched, n_rollout_frames}
    fgd = out["frame_gen_dispatched"]
    assert fg["backend"] == "mindie_sd"
    assert fgd["backend"] == "mindie_sd"
    assert fgd["dispatched"] is True
    # M3-T8 acceptance: engine type is the real v2 pipeline.
    assert fgd["engine_type"] == "VAEDiffusionPipeline"
    # Rollout output shape (B=1, T=future, 3, H, W).
    assert fg["n_rollout_frames"] == 2
    assert fg["frame_metrics"] is not None


def test_frame_gen_phase_none_backend_is_noop() -> None:
    """Default backend stays a no-op (pre-M3-T8 behavior preserved)."""
    from scripts.render_train_rl_cpu import V2Config, frame_gen_phase

    cfg = V2Config(out_dir=".")
    out = frame_gen_phase(cfg, backend="none", device="cpu")
    assert out["backend"] == "none"
    assert out["frame_gen_dispatched"]["dispatched"] is False
    assert out["elapsed_seconds"] == 0.0


def test_frame_gen_phase_unknown_backend_error() -> None:
    from scripts.render_train_rl_cpu import V2Config, frame_gen_phase

    cfg = V2Config(out_dir=".")
    out = frame_gen_phase(cfg, backend="bogus", device="cpu")
    assert "Unknown frame_gen_backend" in out["frame_gen_dispatched"]["dispatch_error"]
