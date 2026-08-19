"""Tests for the vLLM-Omni CPU stand-in pipeline (M2-T6 / M3 followup).

The pipeline binds a V-JEPA inference worker (ORT) and a diffusion
decoder (ORT) behind one surface so the orchestrator can dispatch to
``vllm_omni`` today without a missing backend. Tests cover construction
(defaults + custom cfgs + dual-config parity), the
:class:`LatentToFramePipeline` contract, the extra vLLM-Omni surface
(``encode_video`` / ``plan``), and the backend-registry integration.
"""
from __future__ import annotations

import pytest
import torch

from worldfoundry_inf.omni.pipeline import (
    VLLMOmniPipeline,
    is_vllm_omni_available,
)
from worldfoundry_inf.diffusion.pipeline import LatentToFramePipeline


pytestmark = pytest.mark.skipif(
    not is_vllm_omni_available(),
    reason="onnxruntime wheel not installed; vllm_omni stand-in unavailable",
)


def _tiny_inference_cfg():
    from worldfoundry_inf.workers.onnx_vllm_worker import TinyViTConfig

    return TinyViTConfig(
        img_size=64, patch_size=8, embed_dim=32, depth=1, num_heads=2,
    )


def _tiny_diffusion_cfg():
    from worldfoundry_inf.diffusion.backends.onnx_diffusion import (
        OnnxDiffusionConfig,
    )

    return OnnxDiffusionConfig(
        img_size=64, patch_size=8, embed_dim=32, depth=1,
        num_inference_steps=2, latent_dim=32,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_vllm_omni_default_construction_matches_diffusion_grid() -> None:
    """Default inference worker uses diffusion-compatible ``img_size``."""
    pipe = VLLMOmniPipeline()
    assert pipe.inference.cfg.img_size == pipe.diffusion.cfg.img_size == 64
    assert pipe.name == "vllm_omni"


def test_vllm_omni_custom_inference_and_diffusion_cfgs() -> None:
    """Custom cfgs are reflected on the bound workers."""
    pipe = VLLMOmniPipeline(
        inference_cfg=_tiny_inference_cfg(),
        diffusion_cfg=_tiny_diffusion_cfg(),
    )
    assert pipe.inference.cfg.img_size == 64
    assert pipe.inference.cfg.embed_dim == 32
    assert pipe.diffusion.cfg.img_size == 64
    assert pipe.diffusion.cfg.latent_dim == 32


def test_vllm_omni_explicit_workers_pass_through() -> None:
    """Caller-supplied workers are kept verbatim (no rebuild)."""
    from worldfoundry_inf.workers.onnx_vllm_worker import OnnxVJEPAWorker
    from worldfoundry_inf.diffusion.backends.onnx_diffusion import (
        OnnxDiffusionPipeline,
    )

    inf = OnnxVJEPAWorker(cfg=_tiny_inference_cfg())
    diff = OnnxDiffusionPipeline(cfg=_tiny_diffusion_cfg())
    pipe = VLLMOmniPipeline(inference_worker=inf, diffusion_pipeline=diff)
    assert pipe.inference is inf
    assert pipe.diffusion is diff


def test_vllm_omni_is_latent_to_frame_pipeline() -> None:
    """``VLLMOmniPipeline`` is a :class:`LatentToFramePipeline` subclass."""
    pipe = VLLMOmniPipeline()
    assert isinstance(pipe, LatentToFramePipeline)


def test_vllm_omni_available_classmethod() -> None:
    """``available()`` returns True iff onnxruntime is importable."""
    assert VLLMOmniPipeline.available() is True


# ---------------------------------------------------------------------------
# Pipeline surface
# ---------------------------------------------------------------------------


def test_encode_latents_returns_expected_shape() -> None:
    """``encode_latents`` returns ``(B, T*H*W, latent_dim)``."""
    pipe = VLLMOmniPipeline(
        inference_cfg=_tiny_inference_cfg(),
        diffusion_cfg=_tiny_diffusion_cfg(),
    )
    video = torch.randn(2, 2, 3, 64, 64)
    latents = pipe.encode_latents(video)
    # T=2, Hg=Wg=8 -> 128 tokens; latent_dim=32
    assert latents.shape == (2, 128, 32)


def test_decode_latents_matches_pipeline_grid() -> None:
    """``decode_latents`` produces ``(B, T, 3, H, W)`` for valid inputs."""
    pipe = VLLMOmniPipeline(
        inference_cfg=_tiny_inference_cfg(),
        diffusion_cfg=_tiny_diffusion_cfg(),
    )
    latents = torch.randn(1, 64, 32)
    pixels = pipe.decode_latents(latents, (1, 8, 8))
    assert pixels.shape == (1, 1, 3, 64, 64)


def test_roundtrip_encode_decode_preserves_dtype() -> None:
    """Roundtrip preserves the input dtype."""
    pipe = VLLMOmniPipeline(
        inference_cfg=_tiny_inference_cfg(),
        diffusion_cfg=_tiny_diffusion_cfg(),
    )
    video = torch.randn(1, 1, 3, 64, 64, dtype=torch.float32)
    latents = pipe.encode_latents(video)
    pixels = pipe.decode_latents(latents, (1, 8, 8))
    assert pixels.dtype == torch.float32


def test_rollout_returns_expected_shape() -> None:
    """``rollout`` returns ``(B, future_frames, 3, H, W)``."""
    pipe = VLLMOmniPipeline(
        inference_cfg=_tiny_inference_cfg(),
        diffusion_cfg=_tiny_diffusion_cfg(),
    )
    video = torch.randn(1, 3, 3, 64, 64)
    rollout = pipe.rollout(video, actions=None, past_frames=2, future_frames=1)
    assert rollout.shape == (1, 1, 3, 64, 64)


def test_rollout_ignores_actions() -> None:
    """Stand-in ignores actions (no native action head)."""
    pipe = VLLMOmniPipeline(
        inference_cfg=_tiny_inference_cfg(),
        diffusion_cfg=_tiny_diffusion_cfg(),
    )
    video = torch.randn(1, 2, 3, 64, 64)
    r1 = pipe.rollout(video, actions=None, past_frames=1, future_frames=1)
    r2 = pipe.rollout(video, actions=torch.zeros(1, 1, 7), past_frames=1, future_frames=1)
    assert torch.allclose(r1, r2)


# ---------------------------------------------------------------------------
# Extra vLLM-Omni surface
# ---------------------------------------------------------------------------


def test_encode_video_returns_per_frame_features() -> None:
    """``encode_video`` returns the inference worker's per-token features."""
    pipe = VLLMOmniPipeline(
        inference_cfg=_tiny_inference_cfg(),
        diffusion_cfg=_tiny_diffusion_cfg(),
    )
    video = torch.randn(1, 2, 3, 64, 64)
    feats = pipe.encode_video(video)
    # T=2, Hg=Wg=8 -> 128 tokens, embed_dim=32
    assert feats.shape == (1, 128, 32)


def test_plan_returns_features_and_actions() -> None:
    """``plan`` returns ``(features, actions)`` matching the input."""
    pipe = VLLMOmniPipeline(
        inference_cfg=_tiny_inference_cfg(),
        diffusion_cfg=_tiny_diffusion_cfg(),
    )
    video = torch.randn(1, 1, 3, 64, 64)
    actions = [3, 1, 4]
    feats, returned_actions = pipe.plan(video, actions)
    assert feats.shape[0] == 1
    assert returned_actions == [3, 1, 4]


def test_provider_summary_exposes_both_engines() -> None:
    """``provider_summary`` reports ``engine=vllm_omni`` + nested engines."""
    pipe = VLLMOmniPipeline(
        inference_cfg=_tiny_inference_cfg(),
        diffusion_cfg=_tiny_diffusion_cfg(),
    )
    summary = pipe.provider_summary()
    assert summary["engine"] == "vllm_omni"
    assert summary["inference"]["engine"] == "onnxruntime"
    assert summary["diffusion"]["engine"] == "onnxruntime"


# ---------------------------------------------------------------------------
# Backend registry integration
# ---------------------------------------------------------------------------


def test_inference_backend_registry_has_vllm_omni() -> None:
    """``registry.list()`` exposes the ``vllm_omni`` backend."""
    from worldfoundry_common.backends.inference import registry

    assert "vllm_omni" in registry.list()


def test_inference_backend_builds_vllm_omni() -> None:
    """``make_engine("vllm_omni")`` returns a :class:`VLLMOmniPipeline`."""
    from worldfoundry_common.backends.inference import make_engine

    pipe = make_engine("vllm_omni")
    assert isinstance(pipe, VLLMOmniPipeline)
    assert pipe.inference is not None
    assert pipe.diffusion is not None


def test_diffusers_backend_is_unavailable_when_wheel_missing() -> None:
    """``registry.available('diffusers')`` returns ``(False, hint)`` without the wheel."""
    from worldfoundry_common.backends.inference import registry

    ok, hint = registry.available("diffusers")
    # diffusers may be installed somewhere in CI; treat both branches.
    if not ok:
        assert isinstance(hint, str) and "pip install" in hint
    else:  # pragma: no cover -- depends on env
        assert hint is None


def test_diffusers_backend_raises_when_wheel_missing() -> None:
    """``make_engine('diffusers')`` raises an actionable ImportError when missing."""
    from worldfoundry_common.backends.inference import registry, make_engine

    if registry.available("diffusers")[0]:
        pytest.skip("diffusers wheel installed in this env")
    with pytest.raises(ImportError, match="pip install"):
        make_engine("diffusers")
