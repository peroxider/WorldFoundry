"""v2 NPU diffusion pipeline (Route B from the W40 frame-gen ADR).

Per TODO M3-T4 / spec
``docs/superpowers/specs/2026-08-06-w40-frame-gen-adr.md`` §0/§5.2: this
is the v2 path — latent-conditional diffusion + DiT served by
MindIE-SD on Atlas NPU. The real model is **deferred** until W44+
when the MindIE-SD real compile link lands and the W8A8 precision
regression + ≥6 NPU-month data/compute evaluation clear the gate.

Until then, this module ships as a **stub** that mirrors the
``worldfoundry_inf.workers.vllm_ascend_worker`` NPU-fallback pattern:

* :meth:`NPUDiffusionPipeline.available` returns ``False`` — there is
  no MindIE-SD runtime in this build.
* :meth:`NPUDiffusionPipeline.__init__` raises
  :class:`PipelineNotAvailableError` (which subclasses
  :class:`NotImplementedError`) so the backend registry's
  ``build()`` returns a clean, actionable error and the API endpoint's
  503 mapping continues to work.

When M3-T4 lands, replace the body of :meth:`available` with a real
probe (``importlib.util.find_spec("mindie_sd")`` + Atlas device
detection) and the bodies of ``encode_latents`` / ``decode_latents``
/ ``rollout`` with MindIE-SD calls.
"""
from __future__ import annotations

import torch

from worldfoundry_inf.diffusion.pipeline import (
    LatentToFramePipeline,
    PipelineNotAvailableError,
)


class NPUDiffusionPipeline(LatentToFramePipeline):
    """v2 NPU diffusion stub — not implemented until MindIE-SD lands.

    The interface is frozen so the API endpoint can dispatch uniformly
    between this stub (NPU path) and :class:`CPUPixelDecoderPipeline`
    (v1 fallback) without changing call sites. Until the real backend
    lands, every method raises
    :class:`PipelineNotAvailableError`.
    """

    name = "npu_diffusion"

    def __init__(self, *args: object, **kwargs: object) -> None:
        # Always raise at construction so backend registry surfaces
        # the failure early. Callers that probe ``available()`` first
        # never reach this branch.
        raise PipelineNotAvailableError(
            "NPU diffusion pipeline is not implemented yet. "
            "Tracked under TODO M3-T4 — depends on MindIE-SD real-link "
            "landing in W44+. Use CPUPixelDecoderPipeline for v1 routing."
        )

    @classmethod
    def available(cls) -> bool:
        """``False`` until MindIE-SD lands on Atlas NPU (TODO M3-T4)."""
        return False

    def encode_latents(self, video: torch.Tensor) -> torch.Tensor:  # pragma: no cover - unreachable
        raise PipelineNotAvailableError(
            "NPU diffusion encode_latents is not implemented (TODO M3-T4)."
        )

    def decode_latents(
        self,
        latent: torch.Tensor,
        grid: tuple[int, int, int],
    ) -> torch.Tensor:  # pragma: no cover - unreachable
        raise PipelineNotAvailableError(
            "NPU diffusion decode_latents is not implemented (TODO M3-T4)."
        )

    def rollout(
        self,
        past_video: torch.Tensor,
        actions: torch.Tensor | None,
        past_frames: int,
        future_frames: int,
    ) -> torch.Tensor:  # pragma: no cover - unreachable
        raise PipelineNotAvailableError(
            "NPU diffusion rollout is not implemented (TODO M3-T4)."
        )


__all__ = ["NPUDiffusionPipeline"]