"""Training-framework axis — which RL training stack drives the loop.

Native (self) backends — these build WorldFoundry's own PPO trainer or the
parallel-distribution strategy it runs under:

* ``self``     — self-contained PPO trainer (``worldfoundry_rl.distributed_train``).
* ``pytorch``  — alias of ``self`` (the native PyTorch PPO trainer).
* ``fsdp``     — PyTorch native FSDP parallel strategy (``FSDPStrategy``).
* ``mindspeed``— MindSpeed-MM parallel strategy (``MMStrategy``).

External framework backends — runner adapters that wrap each framework's
training entrypoint behind the
:class:`~worldfoundry_common.adapters.base.TrainingRunner` contract (lazy
imports; building one never imports the framework):

* ``verl``   — ``VerlRunner`` (wraps verl 0.5.0's ``verl.trainer.main_ppo.run_ppo``,
  with Ascend NPU defaults: ``trainer.device=npu`` + ``vllm-ascend`` rollout).
* ``rlinf``  — ``RLinfRunner`` (wraps RLinf's ``EmbodiedRunner``).
"""
from __future__ import annotations

from typing import Any, Callable

from worldfoundry_common.backends.registry import Backend, BackendRegistry

registry = BackendRegistry("training")


def _import_self() -> Callable[..., Any]:
    """Lazy import the self-contained PPO trainer's distributed entry point."""
    from worldfoundry_rl.distributed_train import run_distributed  # noqa: F401

    return run_distributed


def _build_fsdp(**kwargs: Any) -> Any:
    from worldfoundry_train.parallel.fsdp import FSDPStrategy

    return FSDPStrategy(**kwargs)


def _build_mindspeed(**kwargs: Any) -> Any:
    from worldfoundry_train.parallel.mindspeed import MMStrategy

    return MMStrategy(**kwargs)


def _verl_available() -> bool:
    try:
        import verl  # noqa: F401

        return True
    except ImportError:
        return False


def _build_verl(**kwargs: Any) -> Any:
    """Build the verl training runner adapter (``VerlRunner``)."""
    from worldfoundry_common.adapters.verl import VerlRunner

    return VerlRunner(**kwargs)


def _rlinf_available() -> bool:
    try:
        import rlinf  # noqa: F401

        return True
    except ImportError:
        return False


def _build_rlinf(**kwargs: Any) -> Any:
    """Build the RLinf training runner adapter (``RLinfRunner``)."""
    from worldfoundry_common.adapters.rlinf import RLinfRunner

    return RLinfRunner(**kwargs)


registry.register(
    "self",
    Backend(
        "self",
        "Self-contained PPO trainer (worldfoundry_rl).",
        build=_import_self,
    ),
)
registry.register(
    "pytorch",
    Backend(
        "pytorch",
        "Alias for the 'self' native PyTorch PPO trainer.",
        build=_import_self,
    ),
)
registry.register(
    "fsdp",
    Backend(
        "fsdp",
        "PyTorch native FSDP parallel strategy (FSDPStrategy).",
        build=_build_fsdp,
    ),
)
registry.register(
    "mindspeed",
    Backend(
        "mindspeed",
        "MindSpeed-MM parallel strategy (MMStrategy).",
        build=_build_mindspeed,
    ),
)
registry.register(
    "verl",
    Backend(
        "verl",
        "verl RL framework runner adapter (VerlRunner).",
        build=_build_verl,
        available=_verl_available,
        hint="pip install -e '.[rl]'",
    ),
)
registry.register(
    "rlinf",
    Backend(
        "rlinf",
        "RLinf embodied-RL framework runner adapter (RLinfRunner).",
        build=_build_rlinf,
        available=_rlinf_available,
        hint="RLinf is vendored under spikes/w35-3-rlinf; install it to use this backend.",
    ),
)

__all__ = ["registry"]
