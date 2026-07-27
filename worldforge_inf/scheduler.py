"""Adapter-aware router for VJEPA2Worker instances.

Per spec §4.5: each LoRA adapter is compiled to an independent OM, and
requests carrying `adapter_name` route to the matching worker.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from worldforge_common.exceptions import AdapterNotFoundError


class AdapterRouter:
    """Routes inference requests to the right VJEPA2Worker by adapter name."""

    def __init__(self) -> None:
        self._workers: dict[str, "object"] = {}

    def register(self, adapter_name: str, worker) -> None:
        self._workers[adapter_name] = worker

    def route(self, adapter_name: Optional[str]) -> object:
        """Return the worker for `adapter_name`. Raises AdapterNotFoundError if
        `adapter_name` is None or not registered. Per spec §4.7: no silent default substitution."""
        if adapter_name is None or adapter_name not in self._workers:
            raise AdapterNotFoundError(
                f"Adapter {adapter_name!r} not registered. "
                f"Available: {list(self._workers.keys())}. "
                f"Per spec §4.7: no silent default substitution."
            )
        return self._workers[adapter_name]

    def available_adapters(self) -> list[str]:
        return list(self._workers.keys())