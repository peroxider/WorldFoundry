"""Latent → pixel diffusion pipelines for WorldFoundry inference (M3).

Per ``docs/superpowers/specs/2026-08-06-w40-frame-gen-adr.md`` §0/§5.2:
Route A (lightweight pixel decoder) is the Phase-5 v1 baseline; Route B
(latent-conditional diffusion + DiT) is the v2 path gated on W44+
MindIE-SD real-link + W8A8 W26 precision regression + ≥6 NPU-month
data/compute re-evaluation. This package ships the **v1 scaffolding** so
the CPU-fallback surface + ``mindie_sd`` / ``vllm_omni`` backend
registrations can be unit-tested without Atlas hardware; the v2
diffusion model itself is deferred to the MindIE-SD landing (M3-T4).

Public surface:
* :class:`LatentToFramePipeline` — the ABC every backend implements.
* :class:`worldfoundry_inf.diffusion.backends.cpu.CPUPixelDecoderPipeline`
  — v1 CPU stub: random linear projection from JEPA latent space to
  pixel-frame space. ``available()`` always returns ``True`` so the
  backend can be exercised on CPU-only hosts (CI / local dev).
* :class:`worldfoundry_inf.diffusion.backends.npu.NPUDiffusionPipeline`
  — v2 NPU stub: raises ``NotImplementedError`` until MindIE-SD lands.

Backend registrations live in
:mod:`worldfoundry_common.backends.inference` (``mindie_sd``,
``vllm_omni``); this package only defines the pipeline ABC and the
per-backend implementations.
"""
from __future__ import annotations

from worldfoundry_inf.diffusion.pipeline import (
    LatentToFramePipeline,
    PipelineNotAvailableError,
)

__all__ = [
    "LatentToFramePipeline",
    "PipelineNotAvailableError",
]