"""Unified multi-axis backend facade for the WorldFoundry pluggable layer.

Five pluggable axes, each with its own :class:`BackendRegistry`:

* ``device``    — CPU / NPU runtime selection.
* ``training``  — RL training framework (``self`` / ``verl`` / ``rlinf``).
* ``inference`` — inference engine (``torch`` / ``vllm`` / ``sglang``).
* ``model``     — world-model encoder layer (``self`` / ``transformers``).
* ``sim_env``   — simulation environment (``mock`` / ``mujoco`` / ``droid`` / …).

Use :func:`get_backend` to inspect a single backend, :func:`list_backends` to
enumerate an axis, :func:`resolve` to validate a whole config map, and
:func:`build_backend` to construct a backend's artifact. Concrete backends are
lazily imported inside their ``build`` so selecting a backend never imports an
optional dependency until it is actually constructed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from worldfoundry_common.backends.registry import Backend, BackendRegistry
from worldfoundry_common.backends import device, inference, model, sim_env, training

# Importing the axis modules above triggers their backend registration.

_AXES: dict[str, BackendRegistry] = {
    "device": device.registry,
    "training": training.registry,
    "inference": inference.registry,
    "model": model.registry,
    "sim_env": sim_env.registry,
}

AXES: tuple[str, ...] = tuple(sorted(_AXES))


def _registry(axis: str) -> BackendRegistry:
    if axis not in _AXES:
        raise KeyError(f"unknown backend axis {axis!r}; available: {AXES}")
    return _AXES[axis]


def get_backend(axis: str, name: str) -> Backend:
    """Return the :class:`Backend` record for ``name`` in ``axis``."""
    return _registry(axis).get(name)


def list_backends(axis: str) -> list[str]:
    """Return the sorted backend names registered under ``axis``."""
    return _registry(axis).list()


def backend_available(axis: str, name: str) -> tuple[bool, str | None]:
    """Return ``(available, hint)`` for a backend in ``axis``."""
    return _registry(axis).available(name)


def build_backend(axis: str, name: str, **kwargs: Any) -> Any:
    """Construct the artifact for a backend in ``axis``, forwarding ``kwargs``."""
    return _registry(axis).build(name, **kwargs)


def resolve(config: dict[str, str]) -> dict[str, Backend]:
    """Validate a ``{axis: backend_name}`` map and return the backend records.

    Raises:
        KeyError: an axis or backend name is unknown.
    """
    return {axis: _registry(axis).get(name) for axis, name in config.items()}


@dataclass
class Stack:
    """A lazily-built, full-stack composition selected from one config map.

    Holds the resolved :class:`Backend` record for each axis and exposes one
    builder per axis. Nothing is imported/constructed until a builder is
    called, so composing a stack is cheap and dependency-free.
    """

    backends: dict[str, Backend]

    def device(self) -> str:
        """Resolve the device string (``"cpu"`` / ``"npu"``)."""
        return self.backends["device"].build()

    def model(self, **kwargs: Any) -> Any:
        """Build the world-model encoder module."""
        return self.backends["model"].build(**kwargs)

    def env(self, **kwargs: Any) -> Any:
        """Return an env *factory* (no-arg callable producing an Env)."""
        return self.backends["sim_env"].build(**kwargs)

    def engine(self, **kwargs: Any) -> Any:
        """Build the inference engine."""
        return self.backends["inference"].build(**kwargs)

    def training(self) -> Any:
        """Build the training runner (or parallel strategy)."""
        return self.backends["training"].build()


def compose(config: dict[str, str]) -> Stack:
    """Compose a full stack from a ``{axis: backend_name}`` map.

    Example: ``compose({"device": "cpu", "model": "tiny", "sim_env": "mock",
    "inference": "torch", "training": "self"})``. All names are validated
    eagerly; artifacts are built lazily via :class:`Stack`.

    Raises:
        KeyError: an axis or backend name is unknown.
    """
    return Stack(resolve(config))


# Default backend per axis. ``resolve_config`` fills missing axes from here so
# a caller only needs to spell out the axes it wants to switch.
BACKEND_DEFAULTS: dict[str, str] = {
    "device": "auto",
    "training": "self",
    "inference": "torch",
    "model": "self",
    "sim_env": "mock",
}


def resolve_config(config: dict[str, Any] | None = None) -> Stack:
    """Resolve a ``{axis: backend_name}`` map into a validated :class:`Stack`.

    Axes omitted from ``config`` fall back to :data:`BACKEND_DEFAULTS`, so a
    one-axis map like ``{"device": "npu"}`` is enough to switch a single axis
    while keeping the rest at their defaults. Backend names are validated
    eagerly; artifacts stay lazily built (see :class:`Stack`).

    Args:
        config: ``{axis: backend_name}`` map. ``None``/empty yields the
            default stack. ``None`` values for an axis are ignored.

    Raises:
        KeyError: an axis or backend name is unknown.
    """
    mapping = dict(BACKEND_DEFAULTS)
    if config:
        mapping.update({k: v for k, v in config.items() if v is not None})
    return compose(mapping)


def load_stack(path: str | Path, *, key: str = "backends") -> tuple[Stack, dict[str, Any]]:
    """Load a YAML config file and resolve its ``backends:`` block.

    Convenience for startup entries: reads the ``{axis: backend_name}`` map
    under the top-level ``backends:`` key of ``path``, resolves it into a
    :class:`Stack` (filling defaults for missing axes), and returns it
    alongside the full YAML dict so callers can keep using the rest of the
    file unchanged.

    Args:
        path: YAML file to read.
        key: top-level YAML key holding the backend map (default ``"backends"``).

    Returns:
        ``(stack, full_config)``. A missing/empty ``backends`` block yields the
        default stack.
    """
    import yaml

    full = yaml.safe_load(Path(path).read_text()) or {}
    block = full.get(key) if isinstance(full, dict) else None
    return resolve_config(block if isinstance(block, dict) else None), full


__all__ = [
    "AXES",
    "BACKEND_DEFAULTS",
    "Backend",
    "Stack",
    "compose",
    "get_backend",
    "list_backends",
    "backend_available",
    "build_backend",
    "resolve",
    "resolve_config",
    "load_stack",
]
