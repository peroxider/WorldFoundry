"""Unit tests for W22.7 pilot early-stop verdict logic
(worldfoundry_train/eval/pilot_checks.py) — synthetic series triggering each
branch: GO, memory-leak NO-GO, throughput-degradation warning, NaN NO-GO,
EMA-stuck NO-GO, memory-gauge skip. Plus the trainer step_callback
passthrough used to collect the series.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from worldfoundry_train.eval.pilot_checks import GO, NO_GO, evaluate_pilot

N = 200


def _stable_memory() -> list[float]:
    rng = np.random.default_rng(0)
    return list(4000.0 + rng.normal(0, 5, N))  # flat 4GB ± noise


def _stable_times() -> list[float]:
    rng = np.random.default_rng(1)
    return list(0.5 + rng.normal(0, 0.01, N))


def _decreasing_losses() -> list[float]:
    return list(np.linspace(5.0, 3.0, N))


def test_go_all_checks_pass():
    v = evaluate_pilot(
        arm="b",
        loss_series=_decreasing_losses(),
        step_time_series=_stable_times(),
        memory_series_mb=_stable_memory(),
        memory_source="npu_peak",
        timing={"target_forward_s": 20.0, "ema_update_s": 1.0,
                "ema_gather_s": 0.5},
        ema_moved=True,
    )
    assert v["verdict"] == GO
    statuses = {c["name"]: c["status"] for c in v["checks"]}
    assert statuses["numeric_health"] == "pass"
    assert statuses["loss_decreasing"] == "pass"
    assert statuses["memory_growth"] == "pass"
    assert statuses["throughput_stability"] == "pass"
    assert statuses["ema_movement"] == "pass"
    # Component breakdown landed in throughput evidence.
    tp = next(c for c in v["checks"] if c["name"] == "throughput_stability")
    assert tp["evidence"]["target_forward_fraction"] > 0
    assert tp["evidence"]["ema_gather_s"] == 0.5


def test_memory_leak_is_no_go():
    mem = list(np.linspace(4000.0, 6000.0, N))  # steady +50% climb
    v = evaluate_pilot(
        arm="a",
        loss_series=_decreasing_losses(),
        step_time_series=_stable_times(),
        memory_series_mb=mem,
        memory_source="npu_peak",
    )
    assert v["verdict"] == NO_GO
    mem_check = next(c for c in v["checks"] if c["name"] == "memory_growth")
    assert mem_check["status"] == "fail"
    assert mem_check["evidence"]["relative_growth"] > 0.10
    assert mem_check["action"]  # actionable hint present


def test_memory_gauge_missing_is_skipped_not_blocking():
    v = evaluate_pilot(
        arm="a",
        loss_series=_decreasing_losses(),
        step_time_series=_stable_times(),
        memory_series_mb=None,
        memory_source="none",
    )
    assert v["verdict"] == GO
    mem_check = next(c for c in v["checks"] if c["name"] == "memory_growth")
    assert mem_check["status"] == "skipped"


def test_throughput_degradation_is_warning_not_no_go():
    times = [0.5] * 100 + [1.0] * 100  # last-50 median 2x first-50
    v = evaluate_pilot(
        arm="a",
        loss_series=_decreasing_losses(),
        step_time_series=times,
        memory_series_mb=_stable_memory(),
    )
    assert v["verdict"] == GO  # warning does not block
    tp = next(c for c in v["checks"] if c["name"] == "throughput_stability")
    assert tp["status"] == "warn"
    assert tp["action"]


def test_nan_loss_is_no_go():
    losses = _decreasing_losses()
    losses[150] = float("nan")
    v = evaluate_pilot(
        arm="a", loss_series=losses,
        step_time_series=_stable_times(), memory_series_mb=_stable_memory(),
    )
    assert v["verdict"] == NO_GO
    nh = next(c for c in v["checks"] if c["name"] == "numeric_health")
    assert nh["status"] == "fail"


def test_flat_loss_is_warning_not_no_go():
    v = evaluate_pilot(
        arm="a", loss_series=[3.0] * N,
        step_time_series=_stable_times(), memory_series_mb=_stable_memory(),
    )
    assert v["verdict"] == GO
    ld = next(c for c in v["checks"] if c["name"] == "loss_decreasing")
    assert ld["status"] == "warn"


def test_ema_stuck_is_no_go_for_arm_b_only():
    stuck = dict(
        loss_series=_decreasing_losses(),
        step_time_series=_stable_times(),
        memory_series_mb=_stable_memory(),
        ema_moved=False,
    )
    vb = evaluate_pilot(arm="b", **stuck)
    assert vb["verdict"] == NO_GO
    em = next(c for c in vb["checks"] if c["name"] == "ema_movement")
    assert em["status"] == "fail"
    # Arm A has no EMA check at all.
    va = evaluate_pilot(arm="a", **{k: v for k, v in stuck.items()
                                    if k != "ema_moved"})
    assert va["verdict"] == GO
    assert all(c["name"] != "ema_movement" for c in va["checks"])


# ---------------------------------------------------------------------------
# trainer step_callback passthrough (pilot series collection point)
# ---------------------------------------------------------------------------

class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(3 * 16 * 16, 4)

    def forward(self, *, pixel_values_videos, bool_masked_pos=None):
        from types import SimpleNamespace
        return SimpleNamespace(
            last_hidden_state=self.q_proj(pixel_values_videos.flatten(2))
        )


class _FakeDS:
    def __init__(self, n=8):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng(i + 1)
        return {
            "frames": rng.random((1, 3, 16, 16), dtype=np.float32),
            "mask": rng.integers(0, 2, size=(1, 1, 1), dtype=bool),
        }


def test_trainer_step_callback_fires_per_optimizer_step(monkeypatch):
    import worldfoundry_train.trainer as trainer_mod
    from worldfoundry_train.parallel.strategy import DistributedStrategy

    class _Noop(DistributedStrategy):
        def wrap(self, model):
            return model

        def backward(self, loss):
            loss.backward()

        def step(self, optimizer):
            optimizer.step()
            optimizer.zero_grad()

    monkeypatch.setattr(trainer_mod, "_build_dataset", lambda cfg: _FakeDS())
    cfg = {
        "data": {"manifest": "/x", "mask_tubelet_size": [1, 14, 14],
                 "mask_ratio": 0.5},
        "lora": {"r": 4, "alpha": 8, "dropout": 0.0, "bias": "none"},
        "training": {"per_device_batch": 1, "epochs": 1, "lr": 1e-4,
                     "log_every_n": 100, "grad_accum": 2},
    }
    batch = {
        "pixel_values_videos": torch.zeros(1, 1, 3, 16, 16),
        "bool_masked_pos": torch.zeros(1, 1, dtype=torch.bool),
    }

    def loss_fn(m, b):
        return m(
            pixel_values_videos=b["pixel_values_videos"]
        ).last_hidden_state.sum()

    calls: list[tuple[int, float]] = []
    losses = trainer_mod.run_training(
        cfg, _Noop(), base_model=_TinyModel(), num_steps=4,
        batch_fn=lambda i: batch, loss_fn=loss_fn,
        logger=lambda s, loss: None,
        step_callback=lambda s, loss: calls.append((s, loss)),
    )
    # 4 micro-batches, grad_accum=2 -> 2 optimizer steps, 1-indexed.
    assert [s for s, _ in calls] == [1, 2]
    assert len(losses) == 2
