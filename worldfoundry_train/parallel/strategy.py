"""DistributedStrategy interface + profile-based selector (Phase-2 W9)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import ContextManager

from torch import nn
from torch.optim import Optimizer


class DistributedStrategy(ABC):
    """Abstract base for parallel training strategies.

    Three responsibilities: parallelize a model, route backward through the
    parallel runtime, and synchronize optimizer steps.
    """

    @abstractmethod
    def wrap(self, model: nn.Module) -> nn.Module: ...

    @abstractmethod
    def backward(self, loss: "torch.Tensor") -> None:
        """Run loss backward via the parallel runtime. FSDP = passthrough;
        MM = engine.backward(loss)."""

    @abstractmethod
    def step(self, optimizer: Optimizer) -> None: ...

    def no_sync(self, model: nn.Module) -> ContextManager[None]:
        """Context manager suppressing cross-rank gradient sync.

        Used by the training loop for gradient-accumulation micro-batches
        that do NOT close an accumulation window: DDP/FSDP would otherwise
        all-reduce / reduce-scatter gradients on every backward. The
        default is a null context (no suppression — always correct, just
        extra sync traffic); strategies with native sync suppression
        (DDP.no_sync, FSDP.no_sync) override this.
        """
        return nullcontext()


def select_strategy(profile: str) -> DistributedStrategy:
    """Map WORLDFORGE_PROFILE to a strategy instance.

    - "train"           -> FSDPStrategy (Phase-1 default)
    - "train-mmspeed"   -> MMStrategy  (Phase-2 W10, GO path only)

    Raises:
        ValueError: unknown profile
        ImportError: train-mmspeed selected but mindspeed-mm not installed
    """
    if profile == "train":
        from worldfoundry_train.parallel.fsdp import FSDPStrategy
        return FSDPStrategy()

    if profile == "train-mmspeed":
        try:
            from worldfoundry_train.parallel.mindspeed import MMStrategy
        except ImportError as exc:
            raise ImportError(
                "MindSpeed-MM not installed. Run: pip install -e '.[mindspeed]'"
            ) from exc
        return MMStrategy()

    raise ValueError(f"unknown WORLDFORGE_PROFILE: {profile!r}")