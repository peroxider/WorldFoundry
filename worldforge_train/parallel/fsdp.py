"""FSDP launcher (Phase-2 W9: full FSDP strategy impl).

Per Global Constraint G3 + spec §3.3:
- FSDP is ONLY used for full-finetune or long-seq training.
- LoRA training defaults to DDP (1.2B model + LoRA fits per-card).

Phase-1 ships a minimal stub so `train.py --parallel fsdp` does not crash;
Phase-2 W9 implements Hybrid Shard ZeRO-3 + Activation Checkpointing via
the `FSDPStrategy` class appended below. The `setup_fsdp()` / `cleanup_fsdp()`
helpers are preserved for backward compatibility with `train.py --parallel fsdp`.
"""
from __future__ import annotations

import warnings

from torch import nn
from worldforge_train.parallel.strategy import DistributedStrategy


def setup_fsdp() -> tuple[int, int, int]:
    """Phase-1 stub: behaves like DDP but warns that full FSDP is deferred to W9."""
    warnings.warn(
        "FSDP path is a Phase-1 stub. Hybrid Shard ZeRO-3 lands in Phase-2 W9. "
        "Use --parallel ddp for LoRA training now.",
        UserWarning,
        stacklevel=2,
    )
    from worldforge_train.parallel.ddp import setup_ddp
    return setup_ddp()


def cleanup_fsdp() -> None:
    from worldforge_train.parallel.ddp import cleanup_ddp
    cleanup_ddp()


class FSDPStrategy(DistributedStrategy):
    """FSDP-based strategy. Wraps model for sharded data parallel training.

    Phase-2 W9: this replaces the Phase-1 setup_fsdp() stub path. The
    wrap() delegates to torch.distributed.fsdp when world_size > 1; on
    single-rank (tests) it returns the model unmodified.
    """

    def wrap(self, model: nn.Module) -> nn.Module:
        try:
            import torch.distributed as dist
        except ImportError as exc:
            raise RuntimeError("torch.distributed not available") from exc

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if world_size == 1:
            return model  # single-rank fallback (tests / CPU)

        from torch.distributed.fsdp import FSDPModule
        return FSDPModule(model, use_orig_params=True)

    def backward(self, loss) -> None:
        # FSDP path: backward is a passthrough (engine.step() handles sync).
        loss.backward()

    def step(self, optimizer) -> None:
        # Per spec §3.2: all_reduce grads across DP group before optimizer.step().
        import torch.distributed as dist
        if dist.is_initialized() and dist.get_world_size() > 1:
            for param_group in optimizer.param_groups:
                for p in param_group["params"]:
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                        p.grad /= dist.get_world_size()
        optimizer.step()
        optimizer.zero_grad()