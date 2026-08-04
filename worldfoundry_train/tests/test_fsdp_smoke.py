import importlib
import os

import pytest


def test_fsdp_module_imports():
    mod = importlib.import_module("worldforge_train.parallel.fsdp")
    assert hasattr(mod, "setup_fsdp")
    assert hasattr(mod, "cleanup_fsdp")


def test_setup_fsdp_returns_ranks_without_initializing_single_rank():
    """On a single-rank host setup_fsdp must return (1, 0, 0) without init."""
    from worldforge_train.parallel.fsdp import cleanup_fsdp, setup_fsdp
    import torch.distributed as dist

    old_world_size = os.environ.pop("WORLD_SIZE", None)
    try:
        world_size, local_rank, global_rank = setup_fsdp()
        assert (world_size, local_rank, global_rank) == (1, 0, 0)
        assert not dist.is_initialized()
    finally:
        cleanup_fsdp()
        if old_world_size is not None:
            os.environ["WORLD_SIZE"] = old_world_size
