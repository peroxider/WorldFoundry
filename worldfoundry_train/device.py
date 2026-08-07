"""Trainer-side device resolution + module migration (W20).

Thin glue between :mod:`worldfoundry_common.npu_env` (which owns the
"is there an NPU?" predicate) and the trainer orchestrator. The trainer
itself never imports ``torch.npu`` directly — it asks this module for
the device string, then calls :func:`to_device` on every model that
needs to land there.

Per W20 plan §"CPU-only prep": the default device on a CPU-only machine
is ``"cpu"``; on a real Atlas box the predicate flips and the default
becomes ``"npu"``. Callers can pin either with the ``WORLDFORGE_DEVICE``
environment variable, which is checked before :func:`is_npu_available`
so CI / smoke scripts can force the CPU path on hardware that happens
to expose an NPU.

Out of scope here: distributed rank / world_size — that already lives in
:func:`worldfoundry_train.parallel.ddp.setup_ddp` /
:func:`worldfoundry_train.parallel.fsdp.setup_fsdp` and is consumed
upstream of this module. We deliberately do NOT add a
``worldfoundry_train/npu_env.py`` because :mod:`worldfoundry_common.npu_env`
already owns the environment predicates (G8 — no duplicate modules).
"""
from __future__ import annotations

import os

from torch import nn

from worldfoundry_common import npu_env


_DEVICE_ENV = "WORLDFORGE_DEVICE"
_VALID_DEVICES = ("cpu", "npu")


# Re-exported as a module-level name so tests can monkey-patch
# ``worldfoundry_train.device.is_npu_available`` (the canonical
# ``worldfoundry_common.npu_env.is_npu_available`` is also patchable; both
# reach the same function because this module imports it via the module
# attribute lookup below).
def is_npu_available() -> bool:
    """Proxy to :func:`worldfoundry_common.npu_env.is_npu_available`.

    Going through this proxy keeps the trainer's contract single-sourced:
    every code path that decides between ``"cpu"`` and ``"npu"`` calls
    into this module, and tests can swap the predicate here without
    having to know which submodule imported it.
    """
    return npu_env.is_npu_available()


def resolve_device(*, override: str | None = None) -> str:
    """Return the device string the trainer should use.

    Precedence:
        1. ``override`` argument (used by tests and by callers that have
           already resolved the value).
        2. ``WORLDFORGE_DEVICE`` environment variable.
        3. :func:`is_npu_available` (True -> ``"npu"``, False -> ``"cpu"``).

    Raises:
        ValueError: the resolved value is not one of ``{"cpu", "npu"}``.
            This is a fail-fast gate (G4) so a typo on the CLI / in YAML
            cannot silently fall through to ``"cpu"`` on an Atlas box
            and skip the migration.
    """
    candidate = override if override is not None else os.environ.get(_DEVICE_ENV)
    if candidate is None:
        candidate = "npu" if is_npu_available() else "cpu"
    if candidate not in _VALID_DEVICES:
        raise ValueError(
            f"unknown device {candidate!r}; expected one of {_VALID_DEVICES} "
            f"(or unset {_DEVICE_ENV} for auto-detection via is_npu_available())"
        )
    return candidate


def to_device(model: nn.Module, device: str) -> nn.Module:
    """Call ``model.to(device)`` and return ``model`` (chainable).

    Wrapping is intentional: it gives tests a single seam to monkey-patch
    (``worldfoundry_train.device.to_device``) so they can assert the
    trainer called the migration without actually requiring ``torch_npu``
    to be installed on the test host.
    """
    if device not in _VALID_DEVICES:
        raise ValueError(
            f"to_device called with unknown device {device!r}; "
            f"expected one of {_VALID_DEVICES}"
        )
    model.to(device)
    return model


def is_npu_target(device: str) -> bool:
    """True iff ``device`` is the NPU backend (``"npu"``)."""
    return device == "npu"


__all__ = ["resolve_device", "to_device", "is_npu_target", "is_npu_available"]
