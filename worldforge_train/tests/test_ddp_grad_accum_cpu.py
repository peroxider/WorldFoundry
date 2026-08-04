"""CPU 2-rank e2e: DDPStrategy + gradient accumulation keeps ranks in sync.

NOT NPU-gated. Mirrors the mp.spawn pattern of test_fsdp_allreduce_cpu.py.

Each worker:
  1. init_process_group(backend="gloo", init_method="env://")
  2. build identical tiny model + identical batches (torch.manual_seed(42))
  3. DDPStrategy.wrap -> real DistributedDataParallel (world_size=2)
  4. run_training_loop with grad_accum=2 over 4 micro-batches
     (intermediate micro-batches run under DDP.no_sync)
  5. snapshot weights into Manager().dict()

Parent asserts post-training weights are identical across ranks (proves
gradient all-reduce fired on window-closing backwards) and that weights
actually moved from init.
"""
from __future__ import annotations

import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.optim import AdamW

from worldforge_train.parallel.ddp import DDPStrategy
from worldforge_train.training_loop import run_training_loop


def _make_tiny_model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 1))


def _make_batches(n: int) -> list[dict[str, torch.Tensor]]:
    return [{"x": torch.randn(2, 4), "y": torch.randn(2, 1)} for _ in range(n)]


def _loss_fn(model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return ((model(batch["x"]) - batch["y"]) ** 2).sum()


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
        initial = {i: p.detach().clone() for i, p in enumerate(model.parameters())}

        batches = _make_batches(4)  # same seed -> identical across ranks

        strategy = DDPStrategy()
        wrapped = strategy.wrap(model)
        optimizer = AdamW(wrapped.parameters(), lr=0.01)

        losses = run_training_loop(
            wrapped, iter(batches), num_steps=4,
            strategy=strategy, optimizer=optimizer, loss_fn=_loss_fn,
            grad_accum=2,
        )
        assert len(losses) == 2  # 4 micro-batches / grad_accum=2

        snapshots[f"rank{rank}_initial"] = initial
        snapshots[f"rank{rank}_post_weights"] = {
            i: p.detach().clone().cpu() for i, p in enumerate(model.parameters())
        }
    finally:
        dist.destroy_process_group()


def test_ddp_grad_accum_keeps_2_ranks_in_sync():
    world_size = 2
    master_port = _find_free_port()
    manager = mp.Manager()
    snapshots = manager.dict()

    mp.spawn(
        _worker, args=(world_size, master_port, snapshots),
        nprocs=world_size, join=True,
    )

    rank0_post = snapshots["rank0_post_weights"]
    rank1_post = snapshots["rank1_post_weights"]

    for k in rank0_post:
        assert torch.allclose(rank0_post[k], rank1_post[k], atol=1e-6), (
            f"param {k}: ranks diverged after accumulated DDP training -- "
            f"gradient sync on window-closing backward is broken"
        )

    rank0_initial = snapshots["rank0_initial"]
    any_changed = any(
        not torch.allclose(rank0_post[k], rank0_initial[k]) for k in rank0_post
    )
    assert any_changed, "weights unchanged after accumulated training"
