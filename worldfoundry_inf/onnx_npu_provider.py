"""Thin forwarder — ONNX NPU provider moved to worldfoundry_inf.workers.

Historical path: ``worldfoundry_inf.workers.onnx_npu_provider``. The worker
modules were consolidated under :mod:`worldfoundry_inf.workers` (package
root keeps only service entry points). New code should import from
``worldfoundry_inf.workers.onnx_npu_provider``.
"""
from worldfoundry_inf.workers.onnx_npu_provider import *  # noqa: F401,F403
from worldfoundry_inf.workers.onnx_npu_provider import __all__  # noqa: F401
