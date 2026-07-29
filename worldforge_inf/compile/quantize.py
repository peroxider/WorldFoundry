"""Quantization facade — dispatches by quant_type to registered backend.

Phase-1 API surface preserved verbatim (G3). The actual quantization work
lives in worldforge_inf.compile.quantize_backends.*.
"""
from __future__ import annotations

from pathlib import Path

from worldforge_inf.compile.quantize_backends import (
    SUPPORTED_QUANT_TYPES,
    _REGISTRY,
    backend_available,
    get_backend,
)

__all__ = ["quantize_model", "SUPPORTED_QUANT_TYPES"]


# Per-spec §5: W4A8 is acknowledged in SUPPORTED_QUANT_TYPES so callers can
# probe it, but no backend is registered yet — request must surface a clear
# deferral ValueError (not the generic ImportError that missing-deps uses).
_DEFERRED_QUANT_TYPES = {
    "w4a8": "W4A8 deferred to future sub-project.",
}


def quantize_model(
    model_path: Path,
    output_path: Path,
    quant_type: str = "w8a8",
    calibration_dataset: str | None = None,
) -> Path:
    """Quantize a model directory and write the result to ``output_path``.

    Returns the output directory path.

    Raises:
        ValueError: ``quant_type`` is unsupported, or the requested quant
            type has no registered backend (e.g. W4A8 is deferred to a
            future sub-project — see spec §5).
        ImportError: backend is registered but its optional dependency is
            not installed (actionable install hint, G4).
        Any exception propagated from the backend (e.g. ``FileNotFoundError``
            for missing model files).
    """
    if quant_type not in SUPPORTED_QUANT_TYPES:
        raise ValueError(
            f"Unsupported quant_type={quant_type!r}. "
            f"Supported: {SUPPORTED_QUANT_TYPES}"
        )

    # Spec §5: distinguish "no backend registered" (deferred / future) from
    # "backend registered but optional dep missing". Check the registry first
    # so deferred types like W4A8 raise ValueError (with deferral wording)
    # rather than ImportError.
    if quant_type not in _REGISTRY:
        deferral = _DEFERRED_QUANT_TYPES.get(quant_type, "")
        msg = f"No backend registered for quant_type={quant_type!r}."
        if deferral:
            msg += f" ({deferral})"
        raise ValueError(msg)

    available, hint = backend_available(quant_type)
    if not available:
        raise ImportError(hint)

    backend = get_backend(quant_type)
    return backend.quantize(Path(model_path), Path(output_path), calibration_dataset)
