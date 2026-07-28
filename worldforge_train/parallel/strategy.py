"""DistributedStrategy interface + profile-based selector (Phase-2 W9)."""
from __future__ import annotations

from abc import ABC, abstractmethod

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


def select_strategy(profile: str) -> DistributedStrategy:
    """Map WORLDFORGE_PROFILE to a strategy instance.

    - "train"           -> FSDPStrategy (Phase-1 default)
    - "train-mmspeed"   -> MMStrategy  (Phase-2 W10, GO path only)

    Raises:
        ValueError: unknown profile
        ImportError: train-mmspeed selected but mindspeed-mm not installed
    """
    if profile == "train":
        from worldforge_train.parallel.fsdp import FSDPStrategy
        return FSDPStrategy()

    if profile == "train-mmspeed":
        try:
            from worldforge_train.parallel.mindspeed import MMStrategy
        except ImportError as exc:
            raise ImportError(
                "MindSpeed-MM not installed. Run: pip install -e '.[mindspeed]'"
            ) from exc
        return MMStrategy()

    raise ValueError(f"unknown WORLDFORGE_PROFILE: {profile!r}")