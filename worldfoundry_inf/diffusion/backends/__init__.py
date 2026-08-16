"""Backend implementations of :class:`LatentToFramePipeline`.

* :mod:`worldfoundry_inf.diffusion.backends.cpu` — v1 CPU stub
  (:class:`CPUPixelDecoderPipeline`, fixed random projection), always
  available; used for CI / local dev and for the ``mindie_sd``
  backend's CPU fallback.
* :mod:`worldfoundry_inf.diffusion.backends.npu` — v2 NPU stub
  (:class:`NPUDiffusionPipeline`); raises
  :class:`PipelineNotAvailableError` until MindIE-SD lands in M3-T4.
"""
from __future__ import annotations

from worldfoundry_inf.diffusion.backends.cpu import CPUPixelDecoderPipeline
from worldfoundry_inf.diffusion.backends.npu import NPUDiffusionPipeline

__all__ = ["CPUPixelDecoderPipeline", "NPUDiffusionPipeline"]