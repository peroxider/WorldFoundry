"""DDP launcher for V-JEPA 2 LoRA training (Phase-1 default per Global Constraint G3)."""
from __future__ import annotations

import os

from worldforge_common.npu_env import is_npu_available


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
