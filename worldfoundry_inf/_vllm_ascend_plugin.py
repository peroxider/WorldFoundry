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
    """Register a W8A8-aware weight loader with vLLM's ModelRegistry (W26).

    When a V-JEPA 2 model has been quantized via the W11-W12 stack, vLLM's
    default HF loader reads ``model.safetensors`` as bf16 and never
    consults :mod:`worldfoundry_inf.quantize`, so int8 packed weights
    would land in the inference graph as raw bytes. This hook registers
    a per-arch weight iterator that detects ``quant_meta.json`` via
    :func:`worldfoundry_inf.quantize.is_w8a8_quantized` and, when present,
    routes int8 weights through :func:`worldfoundry_inf.quantize.unpack_linear`.

    On CPU-only dev hosts (no vLLM installed) the import of vLLM's
    registry fails and the hook is a no-op + log, mirroring the fail-soft
    pattern of :func:`register_model`.
    """
    registry = _vllm_model_registry()
    if registry is None:
        _log.info(
            "vllm not installed; skipping W8A8 quantized loader hook. "
            "This is expected on CPU-only dev hosts."
        )
        return

    # Lazy imports keep the entry-point loader from pulling in our
    # inference-side utilities + torch until a VJEPA2Model load starts.
    from pathlib import Path
    import torch
    from worldfoundry_inf import quantize as _quantize

    def _w8a8_loader(weights_iterator, model_dir):
        """Iterate ``(name, tensor)`` pairs, unpacking int8 weights in-place.

        Non-quantized architectures (no ``quant_meta.json``) and unknown
        weight dtypes fall through to the default bf16 iterator — the
        loader is the same function, so vLLM treats the model identically
        to a non-quantized one.
        """
        is_quant = False
        try:
            is_quant = _quantize.is_w8a8_quantized(Path(model_dir))
        except (OSError, ValueError) as exc:
            _log.warning(
                "Failed to read quant_meta.json at %s: %s; "
                "falling back to default bf16 loader.",
                model_dir, exc,
            )
        scales: dict = (
            _load_w8a8_scales(Path(model_dir)) if is_quant else {}
        )
        for name, tensor in weights_iterator:
            if is_quant and tensor.dtype == torch.int8 and not name.endswith("_scale"):
                scale_name = name.rsplit(".", 1)[0] + ".weight_scale"
                if scale_name in scales:
                    packed = {"weight": tensor, "scale": scales[scale_name]}
                    yield name, _quantize.unpack_linear(packed)
                    continue
            yield name, tensor

    try:
        registry.register_model_loader(  # type: ignore[attr-defined]
            "VJEPA2Model", _w8a8_loader
        )
        _log.info("Registered W8A8 loader hook for VJEPA2Model.")
    except AttributeError:
        # Older vLLM without register_model_loader: best-effort fallback is
        # to attach the loader via VJEPA2NPUWorker (handled in
        # vllm_ascend_worker.VJEPA2NPUWorker.load_model when this lands).
        _log.info(
            "vLLM ModelRegistry lacks register_model_loader; "
            "W8A8 hook deferred to VJEPA2NPUWorker."
        )
    except Exception as exc:  # pragma: no cover — defensive
        _log.warning("W8A8 loader hook failed: %s", exc)


def _load_w8a8_scales(model_dir):
    """Read per-channel scales from the safetensors index (best-effort).

    Returns an empty dict on any error (no safetensors lib, no index,
    malformed sidecar). A missing scale forces the loader to pass int8
    tensors through unmodified, which the inference graph will reject
    (fail-fast per G4) — by design: a partial W8A8 load is worse than
    no load.
    """
    try:
        from safetensors import safe_open  # type: ignore[import-not-found]
    except ImportError:
        return {}
    scales: dict = {}
    try:
        for sf_path in model_dir.glob("*.safetensors"):
            with safe_open(str(sf_path), framework="pt") as f:
                for key in f.keys():
                    if key.endswith(".weight_scale"):
                        scales[key] = f.get_tensor(key)
    except Exception as exc:  # pragma: no cover — defensive
        _log.warning("Failed to read W8A8 scales from %s: %s", model_dir, exc)
    return scales