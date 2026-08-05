"""vLLM-Ascend Hardware Pluggable registration entry points.

This module is the target of two entry points declared in ``pyproject.toml``:

- ``vllm.platform_plugins`` -> ``worldfoundry_inf = worldfoundry_inf._vllm_ascend_plugin:register``
- ``vllm.general_plugins``   -> ``vjepa2_model = worldfoundry_inf._vllm_ascend_plugin:register_model``

When vLLM boots, it loads every entry point and invokes its target. ``register``
tells vLLM which platform class to use (we reuse vllm-ascend's existing
``NPUPlatform`` — we don't ship our own); ``register_model`` registers the
``VJEPA2Model`` architecture with vLLM's ``ModelRegistry`` so the engine knows
how to dispatch inference for it via :class:`VJEPA2NPUWorker`.

Both functions are designed to fail soft when vLLM is not installed (the
common case on CPU-only dev hosts): they log + return rather than raising,
so importing this module never breaks the rest of ``worldfoundry_inf``.
"""
from __future__ import annotations

import logging

__all__ = ["register", "register_model"]

_log = logging.getLogger(__name__)


def _vllm_model_registry():
    """Lazy import of vLLM's ModelRegistry. Returns None if vLLM is absent."""
    try:
        from vllm.model_executor.models.registry import ModelRegistry  # type: ignore[import-not-found]
        return ModelRegistry
    except ImportError:
        return None


def register() -> str:
    """Return the vLLM platform class path for the Ascend NPU.

    Called by vLLM when it loads the ``vllm.platform_plugins`` entry point.
    We delegate to vllm-ascend's existing platform implementation rather
    than ship our own — that's the established convention per vllm-ascend
    0.18.0's own ``register()``.
    """
    return "vllm_ascend.platform.NPUPlatform"


def register_model() -> None:
    """Register the VJEPA2Model architecture with vLLM's ModelRegistry.

    Called by vLLM when it loads the ``vllm.general_plugins`` entry point.
    On CPU-only hosts (no vLLM installed), this is a no-op + log so the
    plugin loader doesn't fail.
    """
    registry = _vllm_model_registry()
    if registry is None:
        _log.info(
            "vllm not installed; skipping VJEPA2Model registration. "
            "This is expected on CPU-only dev hosts."
        )
        return

    # Lazy import of the worker so the entry-point loader doesn't pull in
    # torch_npu / transformers until vLLM is actually initializing.
    from worldfoundry_inf.vllm_ascend_worker import VJEPA2NPUWorker

    try:
        registry.register_model("VJEPA2Model", VJEPA2NPUWorker)
        _log.info("Registered VJEPA2Model -> VJEPA2NPUWorker with vLLM ModelRegistry")
    except Exception as exc:  # pragma: no cover — defensive
        _log.warning("VJEPA2Model registration failed: %s", exc)


def _patch_quantized_loader() -> None:
    """Hook for W11-W12 W8A8 quantized models. Stub for now.

    When a V-JEPA 2 model has been quantized via the W11-W12 stack, vLLM's
    default model loader won't know how to read the quantized weights.
    This function will register a custom loader with vLLM-Ascend to
    intercept VJEPA2Model loads and route them through msmodelslim's
    W8A8 unpacker. Deferred to W19b.
    """
    return None  # pragma: no cover — placeholder for W19b