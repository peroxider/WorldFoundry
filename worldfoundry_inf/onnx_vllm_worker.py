"""Thin forwarder — ONNX VJEPA2 worker moved to worldfoundry_inf.workers.

Historical path: ``worldfoundry_inf.workers.onnx_vllm_worker``. The worker
modules were consolidated under :mod:`worldfoundry_inf.workers` (package
root keeps only service entry points). New code should import from
``worldfoundry_inf.workers.onnx_vllm_worker``.
"""
from worldfoundry_inf.workers.onnx_vllm_worker import *  # noqa: F401,F403
from worldfoundry_inf.workers.onnx_vllm_worker import __all__  # noqa: F401
