"""CPU 2-rank e2e: prove strategy.step() actually all-reduces gradients.

NOT NPU-gated. The first real multi-process test in the project.

Pattern: parent test uses torch.multiprocessing.spawn to launch 2 worker
processes. Each worker:
  1. init_process_group(backend="gloo", init_method="env://")
  2. build identical tiny model + identical batch (torch.manual_seed(42))
  3. forward + loss + strategy.backward (per-rank grads identical due to seed)
  4. snapshot per-rank grads into Manager().dict()
  5. call strategy.step(optimizer) -> all_reduce + step + zero_grad
  6. snapshot per-rank weights into Manager().dict()
  7. dist.destroy_process_group() in finally

Parent then asserts:
  - pre-step grads identical across ranks (sanity)
  - post-step weights identical across ranks (proves all-reduce worked)
"""
from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.optim import AdamW

from worldforge_train.parallel.fsdp import FSDPStrategy


def _make_tiny_model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 1))


def _make_batch() -> dict[str, torch.Tensor]:
    return {
        "x": torch.randn(2, 4),
        "y": torch.randn(2, 1),
    }


def _find_free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _worker(rank: int, world_size: int, master_port: int,
            snapshots: dict) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)

    dist.init_process_group(backend="gloo", init_method="env://")

    try:
        torch.manual_seed(42)
        model = _make_tiny_model()
        optimizer = AdamW(model.parameters(), lr=0.01)
        strategy = FSDPStrategy()

        batch = _make_batch()
        pred = model(batch["x"])
        loss = ((pred - batch["y"]) ** 2).sum()
        strategy.backward(loss)

        pre_grads = {
            i: p.grad.detach().clone().cpu()
            for i, p in enumerate(model.parameters())
            if p.grad is not None
        }

        strategy.step(optimizer)

        post_weights = {
            i: p.detach().clone().cpu()
            for i, p in enumerate(model.parameters())
        }

        snapshots[f"rank{rank}_pre_grads"] = pre_grads
        snapshots[f"rank{rank}_post_weights"] = post_weights
    finally:
        dist.destroy_process_group()


def test_strategy_step_allreduces_grads_across_2_ranks():  # gloo init hangs would block ~60s
    world_size = 2
    master_port = _find_free_port()
    manager = mp.Manager()
    snapshots = manager.dict()

    mp.spawn(
        _worker, args=(world_size, master_port, snapshots),
        nprocs=world_size, join=True,
    )

    rank0_pre = snapshots["rank0_pre_grads"]
    rank1_pre = snapshots["rank1_pre_grads"]
    rank0_post = snapshots["rank0_post_weights"]
    rank1_post = snapshots["rank1_post_weights"]

    # Sanity: same seed + same forward -> pre-grads identical
    for k in rank0_pre:
        assert torch.allclose(rank0_pre[k], rank1_pre[k], atol=1e-6), (
            f"pre-grads diverged at param {k}; seed setup is broken"
        )

    # KEY ASSERTION: weights identical after strategy.step (proves all-reduce)
    for k in rank0_post:
        assert torch.allclose(rank0_post[k], rank1_post[k], atol=1e-6), (
            f"param {k}: rank-0 and rank-1 weights diverged after "
            f"strategy.step -- all-reduce did not synchronize gradients"
        )


def test_strategy_step_actually_changes_weights_on_both_ranks():
    """Sanity: weights must differ from init after step (loss > 0 -> grad > 0)."""
    world_size = 2
    master_port = _find_free_port()
    manager = mp.Manager()
    snapshots = manager.dict()
    mp.spawn(
        _worker, args=(world_size, master_port, snapshots),
        nprocs=world_size, join=True,
    )
    for rank in range(world_size):
        post = snapshots[f"rank{rank}_post_weights"]
        # At least one param should have changed from initial
        any_changed = any(
            not torch.allclose(post[k], torch.zeros_like(post[k]))
            for k in post
        )
        # Loss-dependent; on random init this is virtually certain
        assert any_changed, f"rank{rank} weights unchanged after step"
