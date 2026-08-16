"""``LatentToFramePipeline`` ABC + interface contract.

Per TODO M3-T2 / spec ``docs/superpowers/specs/2026-08-06-w40-frame-gen-adr.md``
§5.2: the diffusion-frame-gen pipeline exposes three methods:

* :meth:`encode_latents` — ``(B, T, 3, H, W)`` video → ``(B, N, encoder_dim)``
  latent. Reuses the V-JEPA 2 online encoder; in v1 we plug the existing
  ``VJEPA2Worker.encode`` (torch backend) or the
  ``VJEPA2NPUWorker.encode`` (vllm backend). In v2 this stays the same —
  only the diffusion module changes.
* :meth:`decode_latents` — ``(B, N, encoder_dim)`` latent → ``(B, T, 3, H, W)``
  pixel frames. v1 uses a lightweight pixel decoder (CPU);
  v2 uses latent-conditional diffusion + DiT (NPU/MindIE-SD).
* :meth:`rollout` — autoregressive ``past_video`` + ``actions`` →
  ``future_video`` pixel frames. v1 returns pixel frames directly;
  v2 uses the diffusion sampler to roll out H frames conditioned on
  JEPA latents + actions.

The interface is frozen so the API endpoint can dispatch uniformly
between v1 (CPU pixel decoder) and v2 (NPU diffusion) implementations
without changing call sites. v1 implementations must be deterministic
given a fixed seed; v2 implementations may be stochastic (the diffusion
sampler carries its own RNG).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class PipelineNotAvailableError(NotImplementedError):
    """Raised when a backend is selected but its dependencies are missing.

    Subclasses :class:`NotImplementedError` so existing callers that
    catch ``NotImplementedError`` (e.g. the API endpoint's 503 mapping
    in :mod:`worldfoundry_inf.vllm_ascend_worker`) keep working, while
    letting new code raise a more specific error type.
    """


class LatentToFramePipeline(ABC):
    """Abstract base class for latent → pixel-frame pipelines.

    Concrete subclasses implement :meth:`encode_latents`,
    :meth:`decode_latents`, and :meth:`rollout`. The :meth:`available`
    class method lets the backend registry probe whether the pipeline
    is usable in the current environment (deps installed, hardware
    present, etc.) without raising.
    """

    name: str = ""

    @classmethod
    def available(cls) -> bool:
        """Whether this pipeline can be built in the current environment.

        Default is ``True``; subclasses override to inspect deps + HW.
        """
        return True

    @abstractmethod
    def encode_latents(self, video: torch.Tensor) -> torch.Tensor:
        """Encode ``(B, T, 3, H, W)`` video to ``(B, N, encoder_dim)``.

        ``N = T * Hg * Wg`` matches the V-JEPA 2 encoder's patch grid.

        Args:
            video: ``(B, T, 3, H, W)`` float tensor (normalized).

        Returns:
            ``(B, N, encoder_dim)`` latent features.
        """
        raise NotImplementedError

    @abstractmethod
    def decode_latents(
        self,
        latent: torch.Tensor,
        grid: tuple[int, int, int],
    ) -> torch.Tensor:
        """Decode ``(B, N, encoder_dim)`` latent to ``(B, T, 3, H, W)``.

        Args:
            latent: ``(B, N, encoder_dim)`` features.
            grid: ``(T, Hg, Wg)`` matching the ``N = T*Hg*Wg`` patches.

        Returns:
            ``(B, T, 3, H, W)`` pixel frames.
        """
        raise NotImplementedError

    @abstractmethod
    def rollout(
        self,
        past_video: torch.Tensor,
        actions: torch.Tensor | None,
        past_frames: int,
        future_frames: int,
    ) -> torch.Tensor:
        """Roll out ``future_frames`` pixel frames conditioned on
        ``past_frames`` past frames and (optionally) per-frame actions.

        Args:
            past_video: ``(B, T_past + T_future, 3, H, W)`` historical
                frames; the pipeline consumes ``T_past`` and rolls out
                ``T_future``.
            actions: ``(B, T_past, action_dim)`` per-frame action
                sequence, or ``None`` for action-free rollout.
            past_frames: ``T_past`` (must match ``past_video``.
                ``shape[1] - future_frames``).
            future_frames: ``T_future``.

        Returns:
            ``(B, T_future, 3, H, W)`` predicted pixel frames.
        """
        raise NotImplementedError


__all__ = ["LatentToFramePipeline", "PipelineNotAvailableError"]