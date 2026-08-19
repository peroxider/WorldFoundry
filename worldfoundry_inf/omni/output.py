"""Omni multimodal output component: LatentToFramePipeline registration (M3-T7).

Why this exists
---------------
M3-T7 registers the diffusion :class:`LatentToFramePipeline` as
vLLM-Omni's **multimodal output component** — the piece that turns a
JEPA latent (or past video) into future pixel frames on the same
serving endpoint that produces the JEPA features. Until the public
``vllm-omni`` wheel shipped, this seam was a stub; now that
``vllm_omni`` 0.18.0 is installed (aligned with the container's vllm
0.18.0), :class:`OmniOutputComponent` wires the real package:

* :func:`build_omni_output` — priority dispatch:
    1. :class:`RealVLLMOmniPipeline` (real ``vllm`` package importable)
       — records ``kind="real_vllm_omni"`` + ``vllm_version``.
    2. :class:`VAEDiffusionPipeline` (real VAE + DiT + scheduler,
       M3-T1) — the production diffusion decoder, pure PyTorch.
    3. :class:`VLLMOmniPipeline` (ORT stand-in) — fallback.

Interface contract
------------------
:class:`OmniOutputComponent` exposes the frozen
:class:`LatentToFramePipeline` surface (``encode_latents`` /
``decode_latents`` / ``rollout``) plus a ``provider_summary()``
fingerprint so the serving endpoint can record WHICH pipeline class
served the output (real vllm-omni vs vae-diffusion vs ORT stand-in).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from worldfoundry_inf.diffusion.pipeline import LatentToFramePipeline


@dataclass
class OmniOutputComponent(LatentToFramePipeline):
    """vLLM-Omni multimodal output component (M3-T7).

    Wraps a :class:`LatentToFramePipeline` (the diffusion decoder) with
    the component identity the Omni serving layer needs: a name, a
    build-time ``vllm_omni`` version probe, and a provider fingerprint
    so the endpoint's response carries which engine produced the
    future frames.

    Args:
        pipeline: the wrapped latent-to-frame pipeline.
        vllm_omni_version: ``vllm_omni`` package version captured at
            build time ("" when the wheel is absent).
        extras: additional dispatch metadata.
    """

    name = "omni_latent_to_frame"
    pipeline: LatentToFramePipeline = field(default_factory=dict)  # type: ignore[assignment]
    vllm_omni_version: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # LatentToFramePipeline surface — delegate to the wrapped pipeline
    # ------------------------------------------------------------------ #
    def encode_latents(self, video: torch.Tensor) -> torch.Tensor:
        return self.pipeline.encode_latents(video)

    def decode_latents(
        self,
        latent: torch.Tensor,
        grid: tuple[int, int, int],
    ) -> torch.Tensor:
        return self.pipeline.decode_latents(latent, grid)

    def rollout(
        self,
        past_video: torch.Tensor,
        actions: torch.Tensor | None,
        past_frames: int,
        future_frames: int,
    ) -> torch.Tensor:
        return self.pipeline.rollout(
            past_video=past_video,
            actions=actions,
            past_frames=past_frames,
            future_frames=future_frames,
        )

    # ------------------------------------------------------------------ #
    # Provider fingerprint
    # ------------------------------------------------------------------ #
    def provider_summary(self) -> dict[str, Any]:
        """M3-T7 dispatch fingerprint: which engine served the output."""
        inner = (
            self.pipeline.provider_summary()
            if hasattr(self.pipeline, "provider_summary")
            else {}
        )
        return {
            "engine": "omni_latent_to_frame",
            "kind": "omni_output",
            "available": bool(type(self).available()),
            "pipeline_type": type(self.pipeline).__name__,
            "pipeline_kind": inner.get("kind"),
            "vllm_omni_version": self.vllm_omni_version,
            "extras": dict(self.extras),
        }


def _vllm_omni_version() -> str:
    """Resolve the installed vllm-omni package version ("" when absent)."""
    try:
        from importlib import metadata as importlib_metadata

        return importlib_metadata.version("vllm-omni")
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return ""


def _build_priority_pipeline() -> tuple[LatentToFramePipeline, str, dict[str, Any]]:
    """Build the highest-priority pipeline + provenance metadata.

    Priority (M3-T7 / M2-T6 P1):
      1. Real vllm package path (RealVLLMOmniPipeline) — records the
         real vllm package import + LLM probe.
      2. VAEDiffusionPipeline (M3-T1 production diffusion decoder).
      3. ORT stand-in (VLLMOmniPipeline) — fallback.

    Returns:
        ``(pipeline, vllm_omni_version, extras)``.
    """
    try:
        from worldfoundry_inf.omni.pipeline_real import RealVLLMOmniPipeline

        if RealVLLMOmniPipeline.available():
            pipe = RealVLLMOmniPipeline()
            extras = {
                "dispatch": "real_vllm",
                "vllm_version": getattr(pipe.probe, "vllm_version", ""),
            }
            return pipe, _vllm_omni_version(), extras
    except ImportError:  # pragma: no cover -- defensive
        pass

    try:
        from worldfoundry_inf.diffusion.backends.vae_diffusion import (
            VAEDiffusionPipeline,
        )

        pipe = VAEDiffusionPipeline()
        return pipe, _vllm_omni_version(), {"dispatch": "vae_diffusion"}
    except Exception:  # pragma: no cover -- defensive
        pass

    from worldfoundry_inf.omni.pipeline import VLLMOmniPipeline

    return VLLMOmniPipeline(), _vllm_omni_version(), {"dispatch": "ort_standin"}


def build_omni_output(**kwargs: Any) -> OmniOutputComponent:
    """Build the Omni multimodal output component (M3-T7 entry point).

    Args:
        **kwargs: forwarded to the priority pipeline build; may include
            ``pipeline`` to inject a custom :class:`LatentToFramePipeline`
            directly.

    Returns:
        :class:`OmniOutputComponent` wrapping the resolved pipeline.
    """
    pipeline = kwargs.pop("pipeline", None)
    if pipeline is None:
        pipeline, vo_version, extras = _build_priority_pipeline()
    else:
        vo_version = _vllm_omni_version()
        extras = {"dispatch": "injected"}
    return OmniOutputComponent(
        pipeline=pipeline,
        vllm_omni_version=vo_version,
        extras=extras,
    )


__all__ = ["OmniOutputComponent", "build_omni_output"]
