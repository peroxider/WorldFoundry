"""MMStrategy stub (Phase-2 W10 real impl only if W9 = GO)."""
from __future__ import annotations

from torch import nn
from torch.optim import Optimizer

from worldforge_train.parallel.strategy import DistributedStrategy

_MM_STUB_MSG = (
    "MMStrategy not implemented. W9 = NO-GO; "
    "see docs/superpowers/specs/W9_EVAL_REPORT.md for re-evaluation guidance."
)


class MMStrategy(DistributedStrategy):
    """Stub — replaced by real implementation in Task 8 (GO path)."""

    def wrap(self, model: nn.Module) -> nn.Module:
        raise NotImplementedError(_MM_STUB_MSG)

    def backward(self, loss: "torch.Tensor") -> None:
        raise NotImplementedError(_MM_STUB_MSG)

    def step(self, optimizer: Optimizer) -> None:
        raise NotImplementedError(_MM_STUB_MSG)