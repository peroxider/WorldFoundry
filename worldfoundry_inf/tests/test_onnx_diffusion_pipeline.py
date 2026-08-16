"""Tests for the ONNX Runtime diffusion pipeline (M3-T4 followup).

The pipeline is the host-smoke non-PyTorch accelerated inference engine
for the world model's diffusion module. It builds a small DiT decoder
in pure PyTorch, exports it to ONNX once, and serves
:class:`onnxruntime.InferenceSession` for the
:class:`LatentToFramePipeline` surface.
"""
from __future__ import annotations

import pytest
import torch

from worldfoundry_inf.diffusion.backends.onnx_diffusion import (
    OnnxDiffusionConfig,
    OnnxDiffusionPipeline,
    is_onnx_diffusion_available,
)


pytestmark = pytest.mark.skipif(
    not is_onnx_diffusion_available(),
    reason="onnxruntime wheel not installed; onnx_diffusion backend unavailable",
)


def _tiny_cfg() -> OnnxDiffusionConfig:
    """Build a tiny DiT config that runs in <1 s on a stock CPU."""
    return OnnxDiffusionConfig(
        img_size=64, patch_size=8, embed_dim=32, depth=1, latent_dim=32,
        num_inference_steps=2,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_onnx_diffusion_builds_default_config() -> None:
    """Default construction picks :class:`OnnxDiffusionConfig` defaults."""
    pipe = OnnxDiffusionPipeline()
    summary = pipe.provider_summary()
    assert summary["engine"] == "onnxruntime"
    assert summary["img_size"] == 64
    assert summary["onnx_bytes"] > 0


def test_onnx_diffusion_builds_custom_config() -> None:
    """Custom decoder config is reflected in ``provider_summary``."""
    pipe = OnnxDiffusionPipeline(cfg=_tiny_cfg())
    summary = pipe.provider_summary()
    assert summary["img_size"] == 64
    assert summary["patch_size"] == 8
    assert summary["embed_dim"] == 32
    assert summary["depth"] == 1
    assert summary["num_inference_steps"] == 2


def test_onnx_diffusion_resolves_providers() -> None:
    """``session.get_providers()`` returns at least the requested CPU EP."""
    pipe = OnnxDiffusionPipeline(cfg=_tiny_cfg(), providers=["CPUExecutionProvider"])
    providers = pipe.session.get_providers()
    assert "CPUExecutionProvider" in providers
    assert pipe.provider_summary()["providers_resolved"] == providers


def test_onnx_diffusion_available_classmethod() -> None:
    """``available()`` returns True iff onnxruntime is importable."""
    assert OnnxDiffusionPipeline.available() is True


# ---------------------------------------------------------------------------
# Pipeline surface
# ---------------------------------------------------------------------------


def test_encode_latents_returns_expected_shape() -> None:
    """``encode_latents`` returns ``(B, N, latent_dim)``."""
    pipe = OnnxDiffusionPipeline(cfg=_tiny_cfg())
    video = torch.randn(2, 2, 3, 64, 64)
    latents = pipe.encode_latents(video)
    assert latents.shape == (2, 2 * 8 * 8, 32)


def test_encode_latents_rejects_non_5d_input() -> None:
    """Non-5-D video raises :class:`ValueError`."""
    pipe = OnnxDiffusionPipeline(cfg=_tiny_cfg())
    with pytest.raises(ValueError, match="B, T, 3, H, W"):
        pipe.encode_latents(torch.randn(2, 3, 64, 64))


def test_encode_latents_rejects_wrong_resolution() -> None:
    """Wrong image size raises :class:`ValueError`."""
    pipe = OnnxDiffusionPipeline(cfg=_tiny_cfg())
    with pytest.raises(ValueError, match="img_size"):
        pipe.encode_latents(torch.randn(2, 2, 3, 32, 32))


def test_decode_latents_returns_expected_shape() -> None:
    """``decode_latents`` returns ``(B, T, 3, H, W)``."""
    pipe = OnnxDiffusionPipeline(cfg=_tiny_cfg())
    latents = torch.randn(2, 2 * 8 * 8, 32)
    pixels = pipe.decode_latents(latents, (2, 8, 8))
    assert pixels.shape == (2, 2, 3, 64, 64)


def test_decode_latents_rejects_non_3d_input() -> None:
    """Non-3-D latent raises :class:`ValueError`."""
    pipe = OnnxDiffusionPipeline(cfg=_tiny_cfg())
    with pytest.raises(ValueError, match="B, N, latent_dim"):
        pipe.decode_latents(torch.randn(2, 8), (1, 8, 8))


def test_decode_latents_rejects_wrong_grid() -> None:
    """Grid (Hg, Wg) mismatch raises :class:`ValueError`."""
    pipe = OnnxDiffusionPipeline(cfg=_tiny_cfg())
    latents = torch.randn(1, 4 * 4, 32)
    with pytest.raises(ValueError, match=r"T\*Hg\*Wg"):
        pipe.decode_latents(latents, (4, 4, 4))


def test_rollout_returns_expected_shape() -> None:
    """``rollout`` returns ``(B, future_frames, 3, H, W)``."""
    pipe = OnnxDiffusionPipeline(cfg=_tiny_cfg())
    past_video = torch.randn(1, 4, 3, 64, 64)
    rollout = pipe.rollout(past_video, actions=None, past_frames=2, future_frames=2)
    assert rollout.shape == (1, 2, 3, 64, 64)


def test_rollout_rejects_wrong_total_frames() -> None:
    """Video length != past_frames + future_frames raises :class:`ValueError`."""
    pipe = OnnxDiffusionPipeline(cfg=_tiny_cfg())
    past_video = torch.randn(1, 5, 3, 64, 64)
    with pytest.raises(ValueError, match="past_frames"):
        pipe.rollout(past_video, actions=None, past_frames=2, future_frames=2)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_onnx_diffusion_is_deterministic_under_seed() -> None:
    """Two pipelines built from the same seed produce identical outputs."""
    torch.manual_seed(0)
    p1 = OnnxDiffusionPipeline(cfg=_tiny_cfg())
    torch.manual_seed(0)
    p2 = OnnxDiffusionPipeline(cfg=_tiny_cfg())
    v = torch.randn(1, 1, 3, 64, 64)
    l1 = p1.encode_latents(v)
    l2 = p2.encode_latents(v)
    assert torch.allclose(l1, l2)


# ---------------------------------------------------------------------------
# Backend registry integration
# ---------------------------------------------------------------------------


def test_inference_backend_registry_has_onnx_diffusion() -> None:
    """``registry.list()`` exposes the ``onnx_diffusion`` backend."""
    from worldfoundry_common.backends.inference import registry

    names = registry.list()
    assert "onnx_diffusion" in names


def test_inference_backend_builds_onnx_diffusion() -> None:
    """``make_engine("onnx_diffusion")`` returns a :class:`OnnxDiffusionPipeline`."""
    from worldfoundry_common.backends.inference import make_engine

    pipe = make_engine("onnx_diffusion")
    assert isinstance(pipe, OnnxDiffusionPipeline)
    assert pipe.session is not None


# ---------------------------------------------------------------------------
# Cross-backend contract parity
# ---------------------------------------------------------------------------


def test_decode_latents_matches_grid_contract() -> None:
    """``decode_latents`` produces ``(B, T, 3, H, W)`` for valid inputs.

    Cross-backend contract: all :class:`LatentToFramePipeline`
    implementations must produce the same shape for valid inputs.
    """
    pipe = OnnxDiffusionPipeline(cfg=_tiny_cfg())
    b = 1
    t = 1
    Hg, Wg = 8, 8
    latents = torch.randn(b, t * Hg * Wg, 32)
    out = pipe.decode_latents(latents, (t, Hg, Wg))
    assert out.shape == (b, t, 3, 64, 64)
    assert out.dtype == latents.dtype