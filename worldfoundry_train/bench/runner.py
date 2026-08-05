"""Single-profile benchmark runner (Phase-2 W9).

Runs N steps (warmup W + measurement M), emits JSON with clips_per_s and
loss_curve[0..M]. Designed for single-rank testing; in real NPU eval the
runner is invoked once per rank and rank-0 writes the JSON.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from worldfoundry_train.parallel.strategy import select_strategy


class _TinyBenchModel(torch.nn.Module):
    """Tiny model for bench smoke testing. Accepts forward(x=...)."""
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).sum()


def _build_tiny_model() -> torch.nn.Module:
    """Tiny bench model for smoke testing (matches test's TinyModel shape)."""
    return _TinyBenchModel()


def _fixed_batch() -> dict[str, torch.Tensor]:
    """Single-rank fixed batch (real eval uses real dataloader)."""
    return {"x": torch.zeros(1, 4)}


def run_bench(
    profile: str,
    steps: int,
    warmup: int,
    tag: str,
    output_dir: Path,
) -> Path:
    """Run bench and write <tag>.json. Returns the output path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    strategy = select_strategy(profile)
    model = strategy.wrap(_build_tiny_model())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    batch = _fixed_batch()

    measurement_steps = steps - warmup
    if measurement_steps <= 0:
        raise ValueError(f"steps ({steps}) must exceed warmup ({warmup})")

    loss_curve: list[float] = []

    # Warmup (not measured)
    for _ in range(warmup):
        out = model(**batch)
        loss = out.sum()
        strategy.backward(loss)
        strategy.step(optimizer)

    # Measurement
    t0 = time.monotonic()
    for _ in range(measurement_steps):
        out = model(**batch)
        loss = out.sum()
        strategy.backward(loss)
        strategy.step(optimizer)
        loss_curve.append(float(loss.detach()))
    elapsed = time.monotonic() - t0

    # clips_per_s: 1 batch * 1 clip / elapsed seconds
    # (real eval uses real batch_size; this stub uses 1 for simplicity)
    clips_per_s = measurement_steps / elapsed if elapsed > 0 else 0.0

    payload: dict[str, Any] = {
        "profile": profile,
        "tag": tag,
        "world_size": 1,
        "warmup_steps": warmup,
        "measurement_steps": measurement_steps,
        "clips_per_s": clips_per_s,
        "loss_curve": loss_curve,
    }

    out_path = output_dir / f"{tag}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path
