"""CPU 2-rank gloo smoke: JEPA + true FSDP (W22.6, design §9.1
test_fsdp_jepa_smoke).

Pattern mirrors test_fsdp_allreduce_cpu.py (mp.spawn, Manager().dict()
snapshots). Each rank:
  1. init gloo process group
  2. build identical tiny encoder + JEPAObjective (same seed)
  3. wrap encoder with REAL FSDP (FULL_SHARD, use_orig_params=True)
  4. bind_distributed + PredictorGradSyncStrategy
  5. run 2 optimizer steps with RANK-DEPENDENT batches (so per-rank grads
     genuinely differ: predictor weights would diverge without the explicit
     all_reduce, and the EMA would lerp toward shard garbage without the
     FULL_STATE_DICT gather)
  6. snapshot predictor weights + fp32 EMA master + target params

Parent asserts rank-0 == rank-1 for all snapshots (proves predictor grad
sync + gather-then-update EMA consistency), finite losses, and that the
EMA master actually moved from init.
"""
from __future__ import annotations

import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from worldforge_train.jepa_objective import (
    JEPAObjective, PredictorGradSyncStrategy,
)
from worldforge_train.model.predictor import JEPAPredictor

N, D = 16, 8
GRID = (1, 4, 4)


class _TinyEncoder(nn.Module):
    """Minimal encoder with the VJEPA2Model call shape; all params trainable
    (uniform requires_grad keeps the single-unit FSDP wrap happy). Returns a
    real ``BaseModelOutput`` (dataclass + OrderedDict, exactly what HF
    models return) — FSDP's output/input traversal requires dataclass or
    built-in container outputs; a bare SimpleNamespace or dict subclass
    breaks its pre-backward hook registration / kwargs reconstruction."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, D)
        self.block = nn.Linear(D, D)

    def forward(self, *, pixel_values_videos, skip_predictor=False):
        from transformers.modeling_outputs import BaseModelOutput

        b = pixel_values_videos.shape[0]
        x = pixel_values_videos.reshape(b, 3, N).permute(0, 2, 1)  # (B,N,3)
        h = self.block(torch.relu(self.proj(x)))  # (B,N,D)
        return BaseModelOutput(last_hidden_state=h)


def _find_free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _batch(seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        "pixel_values_videos": torch.randn(1, 3, N, generator=g),
        "bool_masked_pos": torch.rand(1, N, generator=g) < 0.5,
    }


def _worker(rank: int, world_size: int, master_port: int, snapshots) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    dist.init_process_group(backend="gloo", init_method="env://")

    try:
        torch.manual_seed(42)  # identical init on both ranks
        online = _TinyEncoder()
        predictor = JEPAPredictor(
            num_patches=N, grid_size=GRID, encoder_dim=D,
            predictor_dim=D, depth=1, num_heads=2,
        )
        obj = JEPAObjective.from_online_model(
            online, predictor,
            ema_momentum_start=0.5, ema_momentum_end=0.5,
            ema_schedule="constant",
        )
        ema_master_init = [b.clone() for b in obj.ema.state_dict()["master"]]

        # Real FSDP wrap. torch 2.10 FSDP1 requires an explicit CPU device_id
        # when no accelerator is present (FSDPStrategy.wrap targets NPU where
        # one always exists); the JEPA<->FSDP interaction under test here is
        # identical either way.
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        model = FSDP(online, use_orig_params=True, device_id=torch.device("cpu"))
        obj.bind_distributed(model)

        from worldforge_train.parallel.strategy import DistributedStrategy

        class _Inner(DistributedStrategy):
            """Minimal inner strategy: FSDP reduces encoder grads in backward
            itself, so step is a plain optimizer step."""

            def wrap(self, m):
                return m

            def backward(self, loss):
                loss.backward()

            def step(self, optimizer):
                optimizer.step()
                optimizer.zero_grad()

        sync_strategy = PredictorGradSyncStrategy(_Inner(), obj.predictor)

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad]
            + list(obj.predictor.parameters()),
            lr=1e-2,
        )

        losses = []
        for step in range(2):
            # Rank-dependent batch: per-rank grads genuinely differ.
            batch = _batch(seed=1000 + 10 * step + rank)
            loss = obj.loss_fn(model, batch)
            sync_strategy.backward(loss)
            sync_strategy.step(optimizer)
            obj.after_step()  # EMA: gather-then-update under FSDP
            losses.append(float(loss.detach()))

        snapshots[f"rank{rank}_losses"] = losses
        snapshots[f"rank{rank}_predictor"] = {
            n: p.detach().cpu().clone()
            for n, p in obj.predictor.named_parameters()
        }
        snapshots[f"rank{rank}_ema_master"] = [
            b.cpu().clone() for b in obj.ema.state_dict()["master"]
        ]
        snapshots[f"rank{rank}_ema_master_init"] = ema_master_init
        snapshots[f"rank{rank}_target"] = {
            n: p.detach().cpu().clone()
            for n, p in obj.target_encoder.named_parameters()
        }
    finally:
        dist.destroy_process_group()


def test_jepa_fsdp_gather_ema_and_predictor_sync_2_ranks():
    world_size = 2
    master_port = _find_free_port()
    manager = mp.Manager()
    snapshots = manager.dict()
    mp.spawn(
        _worker, args=(world_size, master_port, snapshots),
        nprocs=world_size, join=True,
    )

    for rank in range(world_size):
        for v in snapshots[f"rank{rank}_losses"]:
            assert v == v and v >= 0.0

    # ① Predictor weights identical across ranks -> the explicit all_reduce
    # in PredictorGradSyncStrategy ran (rank-dependent batches would
    # otherwise diverge them).
    for n, p0 in snapshots["rank0_predictor"].items():
        p1 = snapshots["rank1_predictor"][n]
        assert torch.allclose(p0, p1, atol=1e-6), f"predictor {n} diverged"

    # ② EMA fp32 masters identical across ranks -> the FULL_STATE_DICT
    # gather fed every rank the same full online weights before lerp.
    for i, m0 in enumerate(snapshots["rank0_ema_master"]):
        m1 = snapshots["rank1_ema_master"][i]
        assert torch.allclose(m0, m1, atol=1e-6), f"ema master {i} diverged"

    # ③ EMA actually moved from init on both ranks.
    moved = any(
        not torch.equal(m, m0)
        for m, m0 in zip(
            snapshots["rank0_ema_master"], snapshots["rank0_ema_master_init"]
        )
    )
    assert moved, "EMA master unchanged after 2 updates"

    # ④ Target encoder params identical across ranks (deterministic EMA).
    for n, t0 in snapshots["rank0_target"].items():
        t1 = snapshots["rank1_target"][n]
        assert torch.allclose(t0, t1, atol=1e-6), f"target {n} diverged"
