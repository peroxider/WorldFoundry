"""Unit tests for :mod:`worldfoundry_inf.diffusion` (M3-T1).

Pure-Python / CPU tests; no Atlas hardware or MindIE-SD needed.

Coverage:
* :class:`LatentToFramePipeline` is properly abstract (cannot be
  instantiated directly; subclasses must implement the three methods).
* :class:`PipelineNotAvailableError` subclasses :class:`NotImplementedError`
  so the API endpoint's existing 503 mapping keeps working.
* :class:`CPUPixelDecoderPipeline` shape/dtype round-trip:
  - ``encode_latents`` produces ``(B, N, encoder_dim)`` from a video.
  - ``decode_latents`` produces ``(B, T, 3, H, W)`` from the matching
    ``(B, N, encoder_dim)`` latent.
  - ``rollout`` produces ``(B, future_frames, 3, H, W)`` and ignores
    ``actions``.
  - Two pipelines with the same seed produce identical latents
    (determinism contract for v1).
* :class:`NPUDiffusionPipeline` (``available()``) returns ``False`` and
  any constructor call raises :class:`PipelineNotAvailableError`.
"""
from __future__ import annotations

import pytest
import torch

from worldfoundry_inf.diffusion import LatentToFramePipeline, PipelineNotAvailableError
from worldfoundry_inf.diffusion.backends.cpu import CPUPixelDecoderPipeline
from worldfoundry_inf.diffusion.backends.npu import NPUDiffusionPipeline


# --------------------------------------------------------------------------- #
# ABC contract
# --------------------------------------------------------------------------- #
def test_latent_to_frame_pipeline_is_abstract() -> None:
    """The ABC cannot be instantiated directly — it has 3 abstract methods."""
    with pytest.raises(TypeError):
        LatentToFramePipeline()  # type: ignore[abstract]


def test_subclass_must_implement_all_methods() -> None:
    """A subclass missing any abstract method still cannot be instantiated."""

    class HalfPipeline(LatentToFramePipeline):
        name = "half"

        def encode_latents(self, video: torch.Tensor) -> torch.Tensor:
            return video

        # decode_latents and rollout not implemented.

    with pytest.raises(TypeError):
        HalfPipeline()  # type: ignore[abstract]


def test_pipeline_not_available_is_not_implemented_error() -> None:
    """PipelineNotAvailableError subclasses NotImplementedError.

    This lets the API endpoint keep using ``except NotImplementedError``
    to map missing-backend cases to HTTP 503.
    """
    assert issubclass(PipelineNotAvailableError, NotImplementedError)


# --------------------------------------------------------------------------- #
# CPU pixel decoder — shape / dtype round-trip
# --------------------------------------------------------------------------- #
def _make_video(batch: int = 1, frames: int = 2, img: int = 196) -> torch.Tensor:
    """Build a normalized float tensor shaped (B, T, 3, H, W)."""
    return torch.randn(batch, frames, 3, img, img, dtype=torch.float32)


def test_cpu_pixel_decoder_available_always_true() -> None:
    """CPU backend is always available (no NPU dependency)."""
    assert CPUPixelDecoderPipeline.available() is True


def test_cpu_pixel_decoder_encode_latents_shape() -> None:
    """encode_latents emits (B, N, encoder_dim) where N = T * Hg * Wg."""
    pipe = CPUPixelDecoderPipeline(encoder_dim=64, img_size=196, patch_size=14)
    video = _make_video(batch=1, frames=2, img=196)
    latents = pipe.encode_latents(video)
    Hg = Wg = 196 // 14  # 14
    n = 2 * Hg * Wg  # 392
    assert latents.shape == (1, n, 64)
    assert latents.dtype == torch.float32


def test_cpu_pixel_decoder_decode_latents_shape() -> None:
    """decode_latents emits (B, T, 3, H, W) matching the (T, Hg, Wg) grid."""
    pipe = CPUPixelDecoderPipeline(encoder_dim=64, img_size=196, patch_size=14)
    Hg = Wg = 196 // 14  # 14
    T = 2
    n = T * Hg * Wg
    latents = torch.randn(1, n, 64, dtype=torch.float32)
    frames = pipe.decode_latents(latents, (T, Hg, Wg))
    assert frames.shape == (1, T, 3, 196, 196)
    assert frames.dtype == torch.float32


def test_cpu_pixel_decoder_rollout_shape_and_action_ignored() -> None:
    """rollout emits (B, future_frames, 3, H, W) and accepts None actions."""
    pipe = CPUPixelDecoderPipeline(encoder_dim=64, img_size=196, patch_size=14)
    past_frames = 3
    future_frames = 4
    total = past_frames + future_frames
    video = _make_video(batch=1, frames=total, img=196)
    # With actions=None and actions=tensor both shapes match.
    out_none = pipe.rollout(video, actions=None, past_frames=past_frames, future_frames=future_frames)
    out_act = pipe.rollout(
        video,
        actions=torch.randn(1, past_frames, 6, dtype=torch.float32),
        past_frames=past_frames,
        future_frames=future_frames,
    )
    assert out_none.shape == (1, future_frames, 3, 196, 196)
    assert out_act.shape == out_none.shape
    assert out_none.dtype == torch.float32


def test_cpu_pixel_decoder_deterministic_under_same_seed() -> None:
    """Two pipelines with the same seed produce identical latents (v1 contract)."""
    video = _make_video(batch=1, frames=2, img=196)
    a = CPUPixelDecoderPipeline(encoder_dim=64, img_size=196, patch_size=14, seed=7).encode_latents(video)
    b = CPUPixelDecoderPipeline(encoder_dim=64, img_size=196, patch_size=14, seed=7).encode_latents(video)
    assert torch.equal(a, b)


def test_cpu_pixel_decoder_different_seeds_diverge() -> None:
    """Sanity check: different seeds produce different latents (the projection
    matrix actually depends on the seed)."""
    video = _make_video(batch=1, frames=2, img=196)
    a = CPUPixelDecoderPipeline(encoder_dim=64, img_size=196, patch_size=14, seed=1).encode_latents(video)
    b = CPUPixelDecoderPipeline(encoder_dim=64, img_size=196, patch_size=14, seed=2).encode_latents(video)
    assert not torch.equal(a, b)


def test_cpu_pixel_decoder_rejects_wrong_grid() -> None:
    """decode_latents raises ValueError when the grid doesn't match N."""
    pipe = CPUPixelDecoderPipeline(encoder_dim=64, img_size=196, patch_size=14)
    Hg = Wg = 196 // 14
    # N=64 but grid (T, Hg, Wg) = (2, Hg, Wg) requires N = 2*Hg*Wg.
    bad_latents = torch.randn(1, 64, 64)
    with pytest.raises(ValueError, match="N="):
        pipe.decode_latents(bad_latents, (2, Hg, Wg))


def test_cpu_pixel_decoder_rejects_wrong_video_rank() -> None:
    """encode_latents requires (B, T, 3, H, W) rank-5 input."""
    pipe = CPUPixelDecoderPipeline()
    with pytest.raises(ValueError, match="(B, T, 3, H, W)"):
        pipe.encode_latents(torch.randn(1, 3, 196, 196))


def test_cpu_pixel_decoder_rejects_wrong_video_spatial() -> None:
    """encode_latents requires H/W == img_size."""
    pipe = CPUPixelDecoderPipeline(img_size=196, patch_size=14)
    with pytest.raises(ValueError, match="img_size"):
        pipe.encode_latents(torch.randn(1, 1, 3, 224, 224))


def test_cpu_pixel_decoder_rejects_rollout_length_mismatch() -> None:
    """rollout requires past_video.shape[1] == past_frames + future_frames."""
    pipe = CPUPixelDecoderPipeline()
    video = torch.randn(1, 5, 3, 196, 196)
    with pytest.raises(ValueError, match="past_frames"):
        pipe.rollout(video, actions=None, past_frames=3, future_frames=4)


# --------------------------------------------------------------------------- #
# NPU diffusion — stub behavior
# --------------------------------------------------------------------------- #
def test_npu_diffusion_available_is_false() -> None:
    """The NPU stub reports unavailable until MindIE-SD lands (M3-T4)."""
    assert NPUDiffusionPipeline.available() is False


def test_npu_diffusion_init_raises_pipeline_not_available() -> None:
    """Constructing the NPU stub raises the documented error."""
    with pytest.raises(PipelineNotAvailableError, match="not implemented"):
        NPUDiffusionPipeline()


def test_npu_diffusion_subclasses_abstract_pipeline() -> None:
    """NPUDiffusionPipeline IS a LatentToFramePipeline (registry contract)."""
    assert issubclass(NPUDiffusionPipeline, LatentToFramePipeline)