"""Backend implementations of :class:`LatentToFramePipeline`.

* :mod:`worldfoundry_inf.diffusion.backends.cpu` — v1 CPU stub
  (:class:`CPUPixelDecoderPipeline`, fixed random projection), always
  available; used for CI / local dev and for the ``mindie_sd``
  backend's CPU fallback.
* :mod:`worldfoundry_inf.diffusion.backends.npu` — v2 NPU stub
  (:class:`NPUDiffusionPipeline`); raises
  :class:`PipelineNotAvailableError` until MindIE-SD lands in M3-T4.
* :mod:`worldfoundry_inf.diffusion.backends.onnx_diffusion` — M3-T4
  followup non-PyTorch accelerated inference engine
  (:class:`OnnxDiffusionPipeline`); backed by
  :class:`onnxruntime.InferenceSession` so the diffusion module can
  run on a stock CPU today and on Atlas NPU via
  ``AscendExecutionProvider``.
"""
from __future__ import annotations

from worldfoundry_inf.diffusion.backends.cpu import CPUPixelDecoderPipeline
from worldfoundry_inf.diffusion.backends.npu import NPUDiffusionPipeline
from worldfoundry_inf.diffusion.backends.onnx_diffusion import (
    OnnxDiffusionConfig,
    OnnxDiffusionPipeline,
)

__all__ = [
    "CPUPixelDecoderPipeline",
    "NPUDiffusionPipeline",
    "OnnxDiffusionConfig",
    "OnnxDiffusionPipeline",
]