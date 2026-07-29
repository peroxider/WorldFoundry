"""Quantization facade — dispatches by quant_type to registered backend.

Phase-1 API surface preserved verbatim (G3). The actual quantization work
lives in worldforge_inf.compile.quantize_backends.*.
"""
from __future__ import annotations

from pathlib import Path

from worldforge_inf.compile.quantize_backends import (
    SUPPORTED_QUANT_TYPES,
    backend_available,
    get_backend,
)

__all__ = ["quantize_model", "SUPPORTED_QUANT_TYPES"]


def quantize_model(
    model_path: Path,
    output_path: Path,
    quant_type: str = "w8a8",
    calibration_dataset: str | None = None,
) -> Path:
    """Quantize a model directory and write the result to ``output_path``.

    Returns the output directory path. Raises ``ValueError`` on unsupported
    ``quant_type``, ``ImportError`` if the backend's dependencies are not
    installed (with an actionable install hint), or any exception propagated
    from the backend (e.g. ``FileNotFoundError`` for missing model files).
    """
    if quant_type not in SUPPORTED_QUANT_TYPES:
        raise ValueError(
            f"Unsupported quant_type={quant_type!r}. "
            f"Supported: {SUPPORTED_QUANT_TYPES}"
        )

    available, hint = backend_available(quant_type)
    if not available:
        raise ImportError(hint)

    backend = get_backend(quant_type)
    return backend.quantize(Path(model_path), Path(output_path), calibration_dataset)
