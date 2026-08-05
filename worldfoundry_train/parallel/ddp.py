"""DDP launcher + strategy for V-JEPA 2 LoRA training (Phase-1 default per Global Constraint G3).

Per spec §3.3 LoRA training defaults to DDP (1.2B model + LoRA fits
per-card); FSDP is reserved for full-finetune / long-seq. ``DDPStrategy``
implements the ``DistributedStrategy`` contract on top of
``torch.nn.parallel.DistributedDataParallel``; single-rank keeps the model
unwrapped so CPU unit tests need no process group.
"""
from __future__ import annotations

import torch
import os
from contextlib import nullcontext
from typing import ContextManager

from torch import nn
from torch.optim import Optimizer

from worldfoundry_common.npu_env import is_npu_available
from worldfoundry_train.parallel.strategy import DistributedStrategy


def setup_ddp() -> tuple[int, int, int]:
    """Initialize torch.distributed for DDP.

    Returns (world_size, local_rank, global_rank).
    """
    import torch
    import torch.distributed as dist

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if is_npu_available():
            backend = "hccl"
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        elif hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.set_device(local_rank)

    return world_size, local_rank, global_rank


def cleanup_ddp() -> None:
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()


class DDPStrategy(DistributedStrategy):
    """DDP-based strategy.

    On multi-rank the model is wrapped by
    ``torch.nn.parallel.DistributedDataParallel``; gradient averaging is
    then performed by DDP's backward hooks, so ``step`` is a plain
    ``optimizer.step() + zero_grad()`` with NO manual all-reduce (unlike
    the legacy single-rank fallback in ``FSDPStrategy.step``). On
    single-rank the model is returned unchanged so CPU unit tests can run
    without ``dist.init_process_group``.

    ``device_ids`` is left as None: correct for CPU/gloo modules, and for
    NPU the tensors are already placed on the target device upstream
    (``setup_ddp`` does ``torch.npu.set_device``).
    """

    def __init__(self) -> None:
        self._wrapped: nn.Module | None = None

    def wrap(self, model: nn.Module) -> nn.Module:
        import torch.distributed as dist

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if world_size == 1:
            self._wrapped = model
            return model

        from torch.nn.parallel import DistributedDataParallel

        wrapped = DistributedDataParallel(model)
        self._wrapped = wrapped
        return wrapped

    def no_sync(self, model: nn.Module) -> ContextManager[None]:
        """Suppress DDP gradient all-reduce for accumulation micro-steps.

        Only real ``DistributedDataParallel`` modules support ``no_sync``;
        anything else (single-rank unwrapped model) has no cross-rank sync
        to suppress, so a null context is returned.
        """
        from torch.nn.parallel import DistributedDataParallel

        if isinstance(model, DistributedDataParallel):
            return model.no_sync()
        return nullcontext()

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    def step(self, optimizer: Optimizer) -> None:
        optimizer.step()
        optimizer.zero_grad()
