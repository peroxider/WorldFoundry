"""Generic multi-axis backend registry for the WorldFoundry pluggable layer.

Each of the five pluggable axes (``device`` / ``training`` / ``inference`` /
``model`` / ``sim_env``) owns a :class:`BackendRegistry`. A registry maps a
backend *name* to a :class:`Backend` record describing how to build that
backend's artifact, whether it is available in the current environment, and
(if unavailable) an actionable activation hint.

Design intent — same shape as ``worldfoundry_inf.compile.quantize_backends``
(Protocol + registry + dispatch), generalised so every axis shares one
mechanism. Concrete backends register themselves lazily so no optional
dependency is ever imported at package import time; availability is probed
on demand via ``available()`` and ``build()``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class Backend:
    """A registered backend for one pluggable axis.

    Attributes:
        name: unique key within its axis (e.g. ``"mujoco"``).
        description: one-line human summary of what the backend provides.
        build: ``build(**kwargs) -> artifact`` — constructs the backend's
            artifact. Must lazily import optional dependencies and raise an
            actionable error when they are missing (G4).
        available: ``available() -> bool`` — probes whether this backend can
            run in the current environment without raising. Defaults to
            always-available; override for optional/guarded backends.
        hint: install/activation instruction surfaced when ``available()`` is
            False (e.g. ``"pip install -e '.[inf]'"``).
    """

    name: str
    description: str
    build: Callable[..., Any]
    available: Callable[[], bool] = field(default=lambda: True)
    hint: str | None = None


class BackendRegistry:
    """Name -> :class:`Backend` map for a single pluggable axis."""

    def __init__(self, axis: str) -> None:
        self.axis = axis
        self._backends: dict[str, Backend] = {}

    def register(self, name: str, backend: Backend) -> None:
        """Register (or replace) a backend under ``name``."""
        self._backends[name] = backend

    def get(self, name: str) -> Backend:
        """Return the :class:`Backend` registered under ``name``.

        Raises:
            KeyError: no backend is registered for ``name``.
        """
        if name not in self._backends:
            raise KeyError(
                f"unknown {self.axis} backend {name!r}; "
                f"available: {sorted(self._backends)}"
            )
        return self._backends[name]

    def list(self) -> list[str]:
        return sorted(self._backends)

    def available(self, name: str) -> tuple[bool, str | None]:
        """Return ``(available, hint)`` for ``name``.

        Raises:
            KeyError: no backend is registered for ``name``.
        """
        backend = self.get(name)
        if backend.available():
            return True, None
        return False, backend.hint

    def build(self, name: str, **kwargs: Any) -> Any:
        """Build the artifact for ``name``, forwarding ``kwargs``.

        Raises:
            KeyError: no backend is registered for ``name``.
        """
        return self.get(name).build(**kwargs)


__all__ = ["Backend", "BackendRegistry"]
