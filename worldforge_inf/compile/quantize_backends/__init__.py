"""Backend registry for quantization implementations.

Subclasses register themselves via the @register decorator; get_backend()
is the only dispatch path used by worldforge_inf.compile.quantize.

W11 only ships the W8A8 backend (msModelSlim). Future W4A8 backend can
land here without disturbing the facade.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Protocol


class QuantBackend(Protocol):
    """Contract every quantization backend must satisfy."""

    def quantize(
        self,
        model_path: Path,
        output_path: Path,
        calibration_dataset: str | None,
    ) -> Path: ...


_REGISTRY: dict[str, type[QuantBackend]] = {}


def register(quant_type: str) -> Callable[[type[QuantBackend]], type[QuantBackend]]:
    """Class decorator that registers a backend under ``quant_type``."""

    def deco(cls: type[QuantBackend]) -> type[QuantBackend]:
        _REGISTRY[quant_type] = cls
        return cls

    return deco


def get_backend(quant_type: str) -> QuantBackend:
    """Return a fresh instance of the backend registered under ``quant_type``.

    Raises:
        ValueError: no backend is registered for ``quant_type``.
    """
    if quant_type not in _REGISTRY:
        raise ValueError(
            f"No backend registered for quant_type={quant_type!r}. "
            f"Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[quant_type]()


_BACKEND_IMPORT_ERROR: Exception | None = None


def backend_available(quant_type: str) -> tuple[bool, str | None]:
    """Return ``(available, hint)`` for ``quant_type``.

    ``hint`` is non-None when the backend is unavailable due to a missing
    dependency (e.g. msModelSlim not installed). The facade uses this to
    emit an ImportError with an actionable install instruction (G4).
    """
    if _BACKEND_IMPORT_ERROR is not None:
        return False, (
            "W8A8 quantization requires msmodelslim. "
            "Install with: pip install worldforge[quant]\n"
            f"Underlying error: {_BACKEND_IMPORT_ERROR}"
        )
    if quant_type not in _REGISTRY:
        return False, (
            f"No backend registered for quant_type={quant_type!r}. "
            f"Available: {sorted(_REGISTRY)}"
        )
    return True, None


# Trigger backend module imports so they self-register. msModelSlim backend
# (Task 3) imports msmodelslim at module top; if missing, capture the error
# so backend_available() can surface it later.
try:
    from . import msmodelslim  # noqa: F401  (W8A8Backend self-registers via @register)
except ImportError as exc:
    _BACKEND_IMPORT_ERROR = exc


# Public list of quant types the facade accepts. Backends register themselves
# at import time; the facade only dispatches to types listed here.
SUPPORTED_QUANT_TYPES: tuple[str, ...] = ("w8a8", "w4a8")  # w4a8 deferred to future sub-project