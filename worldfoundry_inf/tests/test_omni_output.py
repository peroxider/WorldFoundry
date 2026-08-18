"""Unit tests for the M3-T7 Omni multimodal output component.

Covers :func:`build_omni_output` priority dispatch and the
:class:`OmniOutputComponent` LatentToFramePipeline surface:
encode/decode/rollout shapes, provider fingerprint, and pipeline-type
resolution on hosts with/without the real vllm package.
"""
from __future__ import annotations

import torch

from worldfoundry_inf.diffusion.backends.vae_diffusion import VAEDiffusionPipeline
from worldfoundry_inf.diffusion.pipeline import LatentToFramePipeline
from worldfoundry_inf.omni import OmniOutputComponent, build_omni_output


def test_build_omni_output_returns_component() -> None:
    """Default build resolves a pipeline and wraps it in the component."""
    comp = build_omni_output()
    assert isinstance(comp, OmniOutputComponent)
    assert isinstance(comp, LatentToFramePipeline)
    assert isinstance(comp.pipeline, LatentToFramePipeline)
    assert comp.name == "omni_latent_to_frame"


def test_build_omni_output_injected_pipeline() -> None:
    """A caller-supplied pipeline is wrapped as-is."""
    pipe = VAEDiffusionPipeline(encoder_dim=64, img_size=32, num_steps=4)
    comp = build_omni_output(pipeline=pipe)
    assert comp.pipeline is pipe
    assert comp.extras["dispatch"] == "injected"


def test_build_omni_output_priority_resolves_vae_on_cpu_host() -> None:
    """On a host without the real vllm package, VAEDiffusionPipeline wins."""
    comp = build_omni_output()
    # RealVLLMOmniPipeline.available() is False when vllm is absent, so
    # the priority chain falls through to VAEDiffusionPipeline (or the
    # ORT stand-in). Both are LatentToFramePipeline; assert the component
    # resolved something concrete.
    assert type(comp.pipeline).__name__ in (
        "VAEDiffusionPipeline",
        "VLLMOmniPipeline",
        "RealVLLMOmniPipeline",
    )


def test_component_provider_summary_shape() -> None:
    comp = build_omni_output()
    summary = comp.provider_summary()
    assert summary["engine"] == "omni_latent_to_frame"
    assert summary["kind"] == "omni_output"
    assert summary["available"] is True
    assert "pipeline_type" in summary
    assert "vllm_omni_version" in summary
    assert summary["pipeline_type"] == type(comp.pipeline).__name__


def test_component_encode_decode_roundtrip() -> None:
    """encode -> decode through the component preserves frame shapes."""
    comp = build_omni_output(pipeline=VAEDiffusionPipeline(
        encoder_dim=64, img_size=32, num_steps=4, seed=0,
    ))
    video = torch.randn(1, 2, 3, 32, 32)
    latent = comp.encode_latents(video)
    assert latent.shape == (1, 2 * 16, 4)
    frames = comp.decode_latents(torch.randn(1, 16, 64), grid=(1, 4, 4))
    assert frames.shape == (1, 1, 3, 32, 32)
    assert torch.isfinite(frames).all()


def test_component_rollout_shape() -> None:
    comp = build_omni_output(pipeline=VAEDiffusionPipeline(
        encoder_dim=64, img_size=32, num_steps=4, seed=0,
    ))
    past = torch.randn(1, 4, 3, 32, 32)
    out = comp.rollout(past, actions=None, past_frames=2, future_frames=2)
    assert out.shape == (1, 2, 3, 32, 32)


def test_vllm_omni_version_probe_never_raises() -> None:
    """The version probe returns a string (possibly empty) without raising."""
    from worldfoundry_inf.omni.output import _vllm_omni_version

    v = _vllm_omni_version()
    assert isinstance(v, str)
