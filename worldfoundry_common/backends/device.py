"""Device axis — CPU / NPU switching backend registry.

This formalises the CPU<->NPU seam that :mod:`worldfoundry_train.device`
already implements imperatively. The ``device`` axis exposes ``cpu`` / ``npu``
as explicit backends plus an ``auto`` resolver, so callers can select the
device through the same registry API used by every other axis.
"""
from __future__ import annotations

from worldfoundry_common import npu_env
from worldfoundry_common.backends.registry import Backend, BackendRegistry

registry = BackendRegistry("device")

_NPU_HINT = (
    "Run inside the Ascend NPU base image (torch_npu 2.9.0 + CANN 8.5.1). "
    "Set WORLDFORGE_DEVICE=cpu to force CPU on NPU hardware."
)


def _build_npu() -> str:
    if not npu_env.is_npu_available():
        raise RuntimeError(f"NPU backend unavailable: {_NPU_HINT}")
    return "npu"


def _build_auto() -> str:
    return "npu" if npu_env.is_npu_available() else "cpu"


registry.register(
    "cpu",
    Backend("cpu", "PyTorch CPU backend.", build=lambda: "cpu"),
)
registry.register(
    "npu",
    Backend(
        "npu",
        "Ascend NPU backend (torch_npu).",
        build=_build_npu,
        available=npu_env.is_npu_available,
        hint=_NPU_HINT,
    ),
)
registry.register(
    "auto",
    Backend(
        "auto",
        "Auto-select cpu or npu via is_npu_available().",
        build=_build_auto,
    ),
)

__all__ = ["registry"]
