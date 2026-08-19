"""Inference worker implementations (worldfoundry_inf.workers).

M3-T8 restructuring: the four inference worker modules moved from the
``worldfoundry_inf`` package root into this subpackage so the package
root only keeps service/orchestration entry points (``server.py``,
``scheduler.py``, ``batcher.py`` ...). Top-level thin forwarders in
``worldfoundry_inf/`` re-export these for backward compatibility; new
code should import from :mod:`worldfoundry_inf.workers`.

Modules:
* :mod:`~worldfoundry_inf.workers.vjepa2_worker` — torch VJEPA2 worker.
* :mod:`~worldfoundry_inf.workers.vllm_ascend_worker` — vLLM-Ascend worker.
* :mod:`~worldfoundry_inf.workers.onnx_vllm_worker` — ORT VJEPA2 worker.
* :mod:`~worldfoundry_inf.workers.onnx_npu_provider` — ORT NPU provider.
"""
