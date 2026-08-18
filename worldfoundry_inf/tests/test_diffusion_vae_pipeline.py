"""Integration tests for the M3-T1 VAE+DiT+scheduler pipeline wiring.

Verifies that the three real building blocks compose into a working
:class:`LatentToFramePipeline` implementation: shape contracts through
encode → decode / rollout, determinism, and the registry-level
``available()`` probe.
"""
from __future__ import annotations

import pytest
import torch

from worldfoundry_inf.diffusion.backends.vae_diffusion import VAEDiffusionPipeline
from worldfoundry_inf.diffusion.pipeline import LatentToFramePipeline


@pytest.fixture()
def pipeline() -> VAEDiffusionPipeline:
    # Small CI-sized pipeline: 32x32 frames, 8x VAE compression -> 4x4
    # latent grid, latent_dim=4, encoder_dim=64, 8 sampling steps.
    return VAEDiffusionPipeline(
        encoder_dim=64,
        img_size=32,
        latent_dim=4,
        num_steps=8,
        scheduler="ddim",
        seed=0,
    )


def test_pipeline_is_latent_to_frame(pipeline: VAEDiffusionPipeline) -> None:
    assert isinstance(pipeline, LatentToFramePipeline)
    assert pipeline.name == "vae_diffusion"
    assert pipeline.available() is True


def test_pipeline_encode_shape(pipeline: VAEDiffusionPipeline) -> None:
    """(B, T, 3, 32, 32) -> (B, T*16, latent_dim) — 4x4 latent grid."""
    video = torch.randn(2, 3, 3, 32, 32)
    latent = pipeline.encode_latents(video)
    assert latent.shape == (2, 3 * 16, 4)


def test_pipeline_decode_shape(pipeline: VAEDiffusionPipeline) -> None:
    """(B, N, encoder_dim) + grid -> (B, T, 3, 32, 32)."""
    latent = torch.randn(2, 2 * 16, 64)
    frames = pipeline.decode_latents(latent, grid=(2, 4, 4))
    assert frames.shape == (2, 2, 3, 32, 32)
    assert torch.isfinite(frames).all()


def test_pipeline_decode_deterministic(pipeline: VAEDiffusionPipeline) -> None:
    """Same condition + same scheduler state -> identical decode (DDIM eta=0)."""
    latent = torch.randn(1, 16, 64)
    out1 = pipeline.decode_latents(latent, grid=(1, 4, 4))
    out2 = pipeline.decode_latents(latent, grid=(1, 4, 4))
    assert torch.equal(out1, out2)


def test_pipeline_rollout_shape(pipeline: VAEDiffusionPipeline) -> None:
    """past 2 + future 2 -> (B, 2, 3, 32, 32)."""
    past_video = torch.randn(2, 4, 3, 32, 32)
    out = pipeline.rollout(past_video, actions=None, past_frames=2, future_frames=2)
    assert out.shape == (2, 2, 3, 32, 32)
    assert torch.isfinite(out).all()


def test_pipeline_rollout_bad_total_raises(pipeline: VAEDiffusionPipeline) -> None:
    with pytest.raises(ValueError, match="past_frames\\+future_frames"):
        pipeline.rollout(torch.randn(1, 5, 3, 32, 32), None, 2, 2)


def test_pipeline_encode_bad_ndim_raises(pipeline: VAEDiffusionPipeline) -> None:
    with pytest.raises(ValueError, match=r"video must be \(B, T, 3, H, W\)"):
        pipeline.encode_latents(torch.randn(2, 3, 32, 32))


def test_pipeline_decode_bad_dim_raises(pipeline: VAEDiffusionPipeline) -> None:
    with pytest.raises(ValueError, match="encoder_dim"):
        pipeline.decode_latents(torch.randn(1, 16, 32), grid=(1, 4, 4))


def test_pipeline_flow_scheduler_works() -> None:
    """Flow-Matching scheduler composes the same way."""
    pipe = VAEDiffusionPipeline(
        encoder_dim=64, img_size=32, latent_dim=4, num_steps=4, scheduler="flow"
    )
    latent = torch.randn(1, 16, 64)
    frames = pipe.decode_latents(latent, grid=(1, 4, 4))
    assert frames.shape == (1, 1, 3, 32, 32)


def test_pipeline_bad_scheduler_raises() -> None:
    with pytest.raises(ValueError, match="scheduler"):
        VAEDiffusionPipeline(scheduler="nope")


def test_pipeline_provider_summary(pipeline: VAEDiffusionPipeline) -> None:
    summary = pipeline.provider_summary()
    assert summary["engine"] == "vae_diffusion"
    assert summary["kind"] == "vae_diffusion"
    assert summary["available"] is True
    extras = summary["extras"]
    assert extras["encoder_dim"] == 64
    assert extras["img_size"] == 32
    assert extras["latent_dim"] == 4
    assert extras["num_steps"] == 8
    assert extras["scheduler"] == "ddim"


def test_pipeline_state_dicts_are_loadable(pipeline: VAEDiffusionPipeline) -> None:
    """The composed sub-model state_dicts round-trip (checkpoint portability)."""
    state = {
        "vae": pipeline.vae.state_dict(),
        "dit": pipeline.dit.state_dict(),
    }
    pipe2 = VAEDiffusionPipeline(
        encoder_dim=64, img_size=32, latent_dim=4, num_steps=8, seed=0
    )
    pipe2.vae.load_state_dict(state["vae"])
    pipe2.dit.load_state_dict(state["dit"])
    latent = torch.randn(1, 16, 64)
    assert torch.equal(
        pipeline.decode_latents(latent, grid=(1, 4, 4)),
        pipe2.decode_latents(latent, grid=(1, 4, 4)),
    )
