"""Tests for the Atlas-NPU-aware ORT provider selector (M2/M3 followup).

The helper lives at :mod:`worldfoundry_inf.workers.onnx_npu_provider` and
decides which ORT execution providers a session should register:

* Atlas box with the ``onnxruntime-ascend`` wheel installed ->
  ``["AscendExecutionProvider", "CPUExecutionProvider"]``.
* Anything else (host smoke, CI, dev box) ->
  ``["CPUExecutionProvider"]``.

These tests cover the contract: explicit overrides win, auto-selection
never returns an empty list, and the host smoke path keeps returning
the CPU provider so CI stays deterministic.

GateGuard fact-force declaration
-------------------------------
* Importers (test-only, no new deps): pytest (existing dev dep);
  ``worldfoundry_inf.workers.onnx_npu_provider`` (new); existing
  ``worldfoundry_inf.workers.onnx_vllm_worker`` and
  ``worldfoundry_inf.diffusion.backends.onnx_diffusion``.
* Callers: none -- pytest-only.
* Affected API (test surface):
    - ``is_ascend_provider_available() -> bool``
    - ``default_providers(requested=None) -> list[str]``
    - ``OnnxVJEPAWorker(providers=None).providers == default_providers()``
    - ``OnnxDiffusionPipeline(providers=None).providers == default_providers()``
* Data schemas: ``bool`` / non-empty ``list[str]``.
* Safety: no destructive ops, no modifications to existing files.
* Verbatim user instruction: "持续执行开发直到世界模型训练基于RLinf执行，
  推理基于vllm(或者vllm-omni)，世界模型的扩散模块也采用加速的推理引擎
  (非pytorch)".
"""
from __future__ import annotations

import pytest

from worldfoundry_inf.workers.onnx_npu_provider import (
    default_providers,
    is_ascend_provider_available,
)


# ---------------------------------------------------------------------------
# is_ascend_provider_available
# ---------------------------------------------------------------------------


def test_is_ascend_provider_available_returns_bool() -> None:
    """Probe returns a plain ``bool`` (no exceptions on a stock host)."""
    result = is_ascend_provider_available()
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# default_providers -- explicit override
# ---------------------------------------------------------------------------


def test_default_providers_explicit_list_wins() -> None:
    """An explicit ``requested`` list is returned verbatim."""
    assert default_providers(["CPUExecutionProvider"]) == ["CPUExecutionProvider"]


def test_default_providers_explicit_list_can_be_empty_ignore() -> None:
    """An empty explicit list falls back to auto-selection (not None)."""
    out = default_providers([])
    # We treat ``[]`` as "no preference" and let auto-selection run.
    assert isinstance(out, list)
    assert len(out) >= 1


def test_default_providers_explicit_tuple_wins() -> None:
    """A tuple request is normalized to a list (still iterable)."""
    out = default_providers(("CPUExecutionProvider",))
    assert out == ["CPUExecutionProvider"]


def test_default_providers_returns_nonempty_list() -> None:
    """Auto-selection always returns a non-empty list of providers."""
    out = default_providers(None)
    assert isinstance(out, list)
    assert len(out) >= 1
    assert all(isinstance(p, str) for p in out)


def test_default_providers_host_smoke_returns_cpu() -> None:
    """On a CPU-only host (no Ascend EP) the helper returns CPU."""
    if is_ascend_provider_available():
        pytest.skip("AscendExecutionProvider available in this env; "
                    "the host-smoke contract is not exercised here")
    assert default_providers(None) == ["CPUExecutionProvider"]


def test_default_providers_preserves_order_when_explicit() -> None:
    """Order of the explicit list is preserved (ORT picks first match)."""
    out = default_providers(["CPUExecutionProvider", "AscendExecutionProvider"])
    assert out == ["CPUExecutionProvider", "AscendExecutionProvider"]


# ---------------------------------------------------------------------------
# Integration: wired into OnnxVJEPAWorker / OnnxDiffusionPipeline
# ---------------------------------------------------------------------------


def test_onnx_vllm_worker_uses_default_providers_when_none_passed() -> None:
    """``OnnxVJEPAWorker(providers=None)`` calls ``default_providers()``."""
    pytest.importorskip("onnxruntime")
    from worldfoundry_inf.workers.onnx_npu_provider import default_providers
    from worldfoundry_inf.workers.onnx_vllm_worker import OnnxVJEPAWorker, TinyViTConfig

    worker = OnnxVJEPAWorker(cfg=TinyViTConfig(img_size=64, patch_size=8))
    assert worker.providers == default_providers()


def test_onnx_diffusion_uses_default_providers_when_none_passed() -> None:
    """``OnnxDiffusionPipeline(providers=None)`` calls ``default_providers()``."""
    pytest.importorskip("onnxruntime")
    from worldfoundry_inf.workers.onnx_npu_provider import default_providers
    from worldfoundry_inf.diffusion.backends.onnx_diffusion import (
        OnnxDiffusionPipeline,
    )

    pipe = OnnxDiffusionPipeline()
    assert pipe.providers == default_providers()


def test_onnx_vllm_worker_respects_explicit_providers() -> None:
    """``OnnxVJEPAWorker(providers=[...])`` does NOT override the caller's list."""
    pytest.importorskip("onnxruntime")
    from worldfoundry_inf.workers.onnx_vllm_worker import OnnxVJEPAWorker, TinyViTConfig

    worker = OnnxVJEPAWorker(
        cfg=TinyViTConfig(img_size=64, patch_size=8),
        providers=["CPUExecutionProvider"],
    )
    assert worker.providers == ["CPUExecutionProvider"]