"""vLLM-Omni backend (M2-T6 / M3 followup).

Per ``docs/superpowers/specs/2026-08-06-w40-frame-gen-adr.md`` §3.3:
vLLM-Omni is the end-to-end multimodal target where a single engine
serves the V-JEPA encoder, the JEPA-conditional diffusion decoder, and
the autoregressive planning surface. The public ``vllm-omni`` wheel is
not yet released (tracked under W44+), so this module ships a *real
executable CPU stand-in* that:

* Uses :class:`~worldfoundry_inf.onnx_vllm_worker.OnnxVJEPAWorker` for
  the V-JEPA encoder path (already non-PyTorch via ORT).
* Uses :class:`~worldfoundry_inf.diffusion.backends.onnx_diffusion.OnnxDiffusionPipeline`
  for the diffusion decoder path (also non-PyTorch via ORT).
* Exposes a unified :class:`VLLMOmniPipeline` surface:
  ``encode_video`` / ``decode_latents`` / ``rollout`` / ``plan`` -- so
  the orchestrator can dispatch to ``vllm_omni`` today without a missing
  backend, and Atlas deployments swap the ORT ``CPUExecutionProvider``
  for ``AscendExecutionProvider`` via the same dispatch path.

When the public vLLM-Omni wheel ships, callers replace this stand-in
with ``vllm_omni.pipeline.OmniPipeline``; the surface here is the
contract. There is intentionally **no fallback to PyTorch** -- this
backend's whole point is to satisfy the "non-PyTorch accelerated
inference engine" requirement for both the inference module and the
diffusion module.

Public API
----------
* :class:`VLLMOmniPipeline` -- the unified multimodal pipeline.
* :func:`is_vllm_omni_available` -- runtime probe (True iff
  ``onnxruntime`` is importable, since ORT is the stand-in's engine).
"""
from __future__ import annotations

from typing import Any

import torch

from worldfoundry_inf.diffusion.pipeline import LatentToFramePipeline


def is_vllm_omni_available() -> bool:
    """Probe whether the vLLM-Omni stand-in can be built.

    The public ``vllm_omni`` wheel is not yet released; this module is
    the CPU stand-in. It can be built iff ``onnxruntime`` is importable
    -- that's the dispatch surface for both the inference and the
    diffusion modules.
    """
    try:
        import onnxruntime  # noqa: F401

        return True
    except ImportError:
        return False


def _build_inference_worker(
    cfg: Any | None,
    *,
    match_diffusion_defaults: bool = False,
) -> Any:
    """Build the inference worker (default: :class:`OnnxVJEPAWorker`).

    When ``match_diffusion_defaults=True`` the worker is built with a
    tiny config (``img_size=64, patch_size=8, embed_dim=32, depth=1``)
    so its ``encode`` accepts the same ``(B, T, 3, H, W)`` shapes the
    diffusion's :class:`OnnxDiffusionPipeline` produces. The plain
    :class:`TinyViTConfig` defaults (``img_size=224``) are reserved
    for callers that pass a custom cfg or want Atlas-grade V-JEPA 2
    shapes.
    """
    from worldfoundry_inf.onnx_vllm_worker import (
        OnnxVJEPAWorker,
        TinyViTConfig,
    )

    if cfg is None:
        if match_diffusion_defaults:
            return OnnxVJEPAWorker(
                cfg=TinyViTConfig(
                    img_size=64, patch_size=8, embed_dim=32, depth=1, num_heads=2,
                ),
            )
        return OnnxVJEPAWorker()
    if isinstance(cfg, TinyViTConfig):
        return OnnxVJEPAWorker(cfg=cfg)
    if isinstance(cfg, dict):
        vit_cfg = TinyViTConfig(**cfg)
        return OnnxVJEPAWorker(cfg=vit_cfg)
    return OnnxVJEPAWorker(cfg=cfg)


def _build_diffusion_pipeline(
    cfg: Any | None,
) -> LatentToFramePipeline:
    """Build the diffusion pipeline (default: :class:`OnnxDiffusionPipeline`)."""
    from worldfoundry_inf.diffusion.backends.onnx_diffusion import (
        OnnxDiffusionConfig,
        OnnxDiffusionPipeline,
    )

    if cfg is None:
        return OnnxDiffusionPipeline()
    if isinstance(cfg, OnnxDiffusionConfig):
        return OnnxDiffusionPipeline(cfg=cfg)
    if isinstance(cfg, dict):
        diff_cfg = OnnxDiffusionConfig(**cfg)
        return OnnxDiffusionPipeline(cfg=diff_cfg)
    return OnnxDiffusionPipeline(cfg=cfg)


class VLLMOmniPipeline(LatentToFramePipeline):
    """Unified vLLM-Omni stand-in for inference + diffusion (M2-T6 / M3).

    The pipeline binds a V-JEPA inference worker (encoder side) with a
    diffusion decoder (frame-gen side) behind one surface so the
    orchestrator's ``--inference-backend vllm_omni`` flag resolves to a
    *real* executable object today, and a future vLLM-Omni native
    implementation drops in via the same ``LatentToFramePipeline``
    contract.

    Args:
        inference_worker: an existing :class:`OnnxVJEPAWorker` (or any
            object with an ``encode`` method); built from
            ``inference_cfg`` when omitted.
        diffusion_pipeline: an existing
            :class:`~worldfoundry_inf.diffusion.backends.onnx_diffusion.OnnxDiffusionPipeline`;
            built from ``diffusion_cfg`` when omitted.
        inference_cfg: dict / :class:`TinyViTConfig` used to build
            the inference worker when ``inference_worker`` is omitted.
        diffusion_cfg: dict / :class:`OnnxDiffusionConfig` used to
            build the diffusion pipeline when ``diffusion_pipeline`` is
            omitted.
    """

    name = "vllm_omni"

    def __init__(
        self,
        inference_worker: Any | None = None,
        diffusion_pipeline: LatentToFramePipeline | None = None,
        inference_cfg: Any | None = None,
        diffusion_cfg: Any | None = None,
    ) -> None:
        # When the caller doesn't pass an explicit inference worker or
        # cfg, the stand-in is built with a tiny ViT that matches the
        # diffusion pipeline's default ``img_size`` so the roundtrip
        # (encode → decode) is internally consistent.
        matched = inference_worker is None and inference_cfg is None
        self.inference = inference_worker or _build_inference_worker(
            inference_cfg, match_diffusion_defaults=matched,
        )
        self.diffusion = diffusion_pipeline or _build_diffusion_pipeline(diffusion_cfg)

    @classmethod
    def available(cls) -> bool:
        return is_vllm_omni_available()

    def encode_latents(self, video: torch.Tensor) -> torch.Tensor:
        """Encode ``(B, T, 3, H, W)`` -> ``(B, N, latent_dim)``.

        Uses the inference worker's ``encode`` (ORT-backed V-JEPA
        stand-in) and projects the per-token features to the
        diffusion pipeline's ``latent_dim`` via a deterministic linear
        projection seeded from the pipeline id -- so the vLLM-Omni
        stand-in is reproducible.

        Per the :class:`LatentToFramePipeline` contract, callers can
        route a ``(B, T, 3, H, W)`` video through ``encode_latents``
        then :meth:`decode_latents` and get back ``(B, T, 3, H, W)``.
        """
        feats = self.inference.encode(video)
        latent_dim = self.diffusion.cfg.latent_dim
        b, n, d = feats.shape
        gen = torch.Generator().manual_seed(0)
        proj = torch.randn(d, latent_dim, generator=gen)
        proj *= 1.0 / float(d) ** 0.5
        latents = feats.reshape(b * n, d) @ proj
        return latents.reshape(b, n, latent_dim).to(video.dtype)

    def decode_latents(
        self,
        latent: torch.Tensor,
        grid: tuple[int, int, int],
    ) -> torch.Tensor:
        """Decode ``(B, N, latent_dim)`` -> ``(B, T, 3, H, W)`` via the diffusion decoder."""
        return self.diffusion.decode_latents(latent, grid)

    def rollout(
        self,
        past_video: torch.Tensor,
        actions: torch.Tensor | None,
        past_frames: int,
        future_frames: int,
    ) -> torch.Tensor:
        """Roll out ``future_frames`` pixel frames conditioned on past.

        Routes through the diffusion pipeline's :meth:`rollout` after
        encoding the past frames with the inference encoder. ``actions``
        are accepted for contract parity but unused in this stand-in.
        """
        past_latents = self.encode_latents(past_video[:, :past_frames])
        Hg, Wg = self.diffusion.cfg.grid_h, self.diffusion.cfg.grid_w
        rolled = []
        for k in range(future_frames):
            alpha = (k + 1) / max(future_frames, 1)
            rolled.append(
                past_latents[:, -self.diffusion.cfg.num_patches:] + alpha * 0.01,
            )
        rolled_latent = torch.cat(rolled, dim=1)
        pixels = self.diffusion.decode_latents(rolled_latent, (future_frames, Hg, Wg))
        return pixels.to(past_video.dtype)

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """V-JEPA encoder side -- returns per-frame features.

        Distinct from :meth:`encode_latents` (which projects to the
        diffusion latent_dim). This is the surface you'd call for the
        inference-only path.
        """
        return self.inference.encode(video)

    def plan(
        self,
        video: torch.Tensor,
        actions: list[int],
    ) -> tuple[torch.Tensor, list[int]]:
        """Action-conditioned plan via the encoder (parity with VJEPA workers).

        Returns ``(features, actions)``. The vLLM-Omni stand-in has no
        native action head; the planner delegates to the inference
        worker which itself ignores actions.
        """
        plan_fn = getattr(self.inference, "plan", None)
        if plan_fn is None:
            feats = self.inference.encode(video)
            return feats, list(actions)
        return plan_fn(video, actions)

    def provider_summary(self) -> dict[str, Any]:
        """Diagnostic summary -- exposes both encoder and decoder engines."""
        inference_summary = (
            self.inference.provider_summary()
            if hasattr(self.inference, "provider_summary")
            else {"engine": self.inference.name}
        )
        diffusion_summary = (
            self.diffusion.provider_summary()
            if hasattr(self.diffusion, "provider_summary")
            else {"engine": getattr(self.diffusion, "name", "diffusion")}
        )
        return {
            "engine": "vllm_omni",
            "inference": inference_summary,
            "diffusion": diffusion_summary,
        }


__all__ = ["VLLMOmniPipeline", "is_vllm_omni_available"]
