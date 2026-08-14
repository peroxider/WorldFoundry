"""Backend registry: maps a backend name -> ordered stage classes.

This is the NPU extension point. The ``cpu`` backend is pre-registered
with the CPU stage implementations. An ``npu`` backend can register a
different set of stage classes (same ``name`` + ``run(ctx)`` contract)
without touching the CPU path or the pipeline code.
"""
from __future__ import annotations

from typing import Type

from worldfoundry_train.recipe.stages import CPU_STAGES, Stage

# backend name -> {stage_name: stage_class}
_REGISTRY: dict[str, dict[str, Type[Stage]]] = {
    "cpu": dict(CPU_STAGES),
}


def register_backend(name: str, stages: dict[str, Type[Stage]]) -> None:
    """Register (or replace) a backend's stage map.

    Args:
        name: backend key, e.g. ``"cpu"`` or ``"npu"``.
        stages: ``{stage_name: Stage subclass}`` for that backend.
    """
    _REGISTRY[name] = dict(stages)


def get_backend_stages(name: str) -> dict[str, Type[Stage]]:
    """Return the stage map for a backend.

    Raises:
        KeyError: the backend is not registered.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown backend {name!r}; registered backends: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def list_backends() -> list[str]:
    return sorted(_REGISTRY)


__all__ = ["register_backend", "get_backend_stages", "list_backends"]
