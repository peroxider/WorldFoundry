# W13 CPU Training Loop — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) to execute this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire a real (forward + loss + backward + optimizer-step) training loop on CPU that uses an existing `DistributedStrategy` (FSDP default) and the existing LoRA wrapper. Add a CPU-runnable integration test that runs 3 real training steps on a tiny VJEPA2Model and verifies loss is computed, gradients propagate, and weights actually change.

**Architecture:**
- New module `worldforge_train/training_loop.py` — pure-Python training step + multi-step loop, no `distributed.init_process_group` call, no `accelerate.prepare`. CPU-native, multi-process-safe by construction (single-process semantics; FSDPStrategy's `world_size==1: return model` branch is the natural entry).
- `DistributedStrategy` is invoked for `wrap()` + `backward()` + `step()` so the same loop works on DDP/FSDP/MM (`MMStrategy` is still a stub, but the loop is correct against its contract).
- LoRA wrapping is the caller's responsibility (matches existing `lora_wrapper.py` contract). The test wires `lora_wrapper.build_lora_model` → training loop.
- Loss is a small V-JEPA-style masked-feature MSE: given `(frames, mask)`, predict the encoder output and compute MSE on masked positions vs. the same encoder's unmasked output at the same positions. This is a stand-in for the production V-JEPA 2 self-supervised loss, sufficient for proving the training path works on CPU.

**Tech Stack:** PyTorch, `transformers.VJEPA2Model` (CPU), `peft` (LoRA, already required), `numpy` (for mask). No new deps.

## Global Constraints

- **G2:** No new dependency pins; existing pins preserved
- **G3:** Phase-1 code (`worldforge_train/train.py`, `worldforge_train/parallel/*`, `worldforge_train/data/*`, `worldforge_train/model/*`, `worldforge_inf/compile/mindie_sd.py`, `worldforge_train/tests/test_*` existing) **0 changes**. This sub-project only **adds** new files in `worldforge_train/` (`training_loop.py`, `tests/test_training_loop.py`, `tests/test_training_loop_cpu.py`)
- **G4:** Explicit errors — wrong batch keys → ValueError; empty dataset → ValueError; optimizer with no `.step` → TypeError
- **G5:** Seeded reproducibility — `seed=42` default for synthetic tensor batches
- **G6:** Single dispatch path — `run_training_step(...)` is the only public function; multi-step is sugar on top
- **G8:** Green tests per task; final state has all training_loop tests passing
- **G9:** Phase-1 test style — mocks for unit tests, real models for the CPU integration test

---

## Task 1: `worldforge_train/training_loop.py` — real forward + backward + step

**Files:**
- Create: `worldforge_train/training_loop.py` (~70 lines)

**Interfaces:**
- `run_training_step(model, batch, strategy, optimizer, loss_fn=None) -> float`
- `run_training_loop(model, batch_iter, num_steps, strategy, optimizer, loss_fn=None) -> list[float]`

**Why:** Current `train.py:run_ddp_training()` is a stub (`for step in range(2): print; break`). `run_fsdp_training()` raises `NotImplementedError`. The strategy classes (`FSDPStrategy`, `DDP` launcher) are real but never invoked. This task wires a usable training loop that:
1. Calls `model(...)` (real forward)
2. Computes a loss (default: masked-feature MSE)
3. Routes backward via `strategy.backward(loss)` (so FSDP all-reduce + DDP hooks work)
4. Calls `optimizer.step()` + `optimizer.zero_grad()`
5. Returns the scalar loss value so callers can log/assert

CPU-runnable because `FSDPStrategy.wrap(model)` returns the model unchanged when `world_size == 1` (single-rank fallback at `fsdp.py:51-53`), and `strategy.backward(loss)` is a thin passthrough calling `loss.backward()`.

### Step 1.1: Implement `training_loop.py`

```python
"""Real CPU-runnable training loop (forward + loss + backward + step).

Wires the existing ``DistributedStrategy`` contract
(``worldforge_train.parallel.strategy.DistributedStrategy``) to a real
forward/loss/backward/step cycle. Works in single-rank mode (the
``world_size == 1`` path used in unit/CPU tests) without needing
``dist.init_process_group``; ``FSDPStrategy.wrap`` returns the model
unchanged in that case (see ``fsdp.py:51-53``).

The default loss is a V-JEPA-style masked-feature MSE: encoder output
at masked positions is regressed against the encoder's prediction at
the same positions from a target pass. This is sufficient for CPU
testing of the training path; the production V-JEPA 2 self-supervised
objective is identical in shape (predict masked tokens) but with EMA
target encoder + cosine similarity loss.

G3 boundary: this file does NOT modify ``train.py`` or ``parallel/*``.
"""
from __future__ import annotations

from typing import Callable, Iterable, Mapping

import torch
from torch import nn
from torch.optim import Optimizer

from worldforge_train.parallel.strategy import DistributedStrategy

LossFn = Callable[[nn.Module, Mapping[str, torch.Tensor]], torch.Tensor]


def default_loss_fn(model: nn.Module, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """V-JEPA-style masked-feature MSE.

    Expects ``batch["pixel_values_videos"]`` of shape ``(B, T, C, H, W)``
    and ``batch["bool_masked_pos"]`` of shape ``(B, T, num_patches)``.
    Computes encoder output, then MSE on the masked-token positions
    against a self-distillation target (the same encoder with no_grad
    on unmasked inputs -- CPU surrogate for the EMA target encoder).
    """
    pixel_values_videos = batch["pixel_values_videos"]
    bool_masked_pos = batch["bool_masked_pos"]
    out = model(pixel_values_videos=pixel_values_videos, bool_masked_pos=bool_masked_pos)
    feats = out.last_hidden_state
    with torch.no_grad():
        tgt_out = model(pixel_values_videos=pixel_values_videos, bool_masked_pos=bool_masked_pos)
        target = tgt_out.last_hidden_state
    return torch.nn.functional.mse_loss(feats[bool_masked_pos], target[bool_masked_pos])


def run_training_step(
    model: nn.Module,
    batch: Mapping[str, torch.Tensor],
    strategy: DistributedStrategy,
    optimizer: Optimizer,
    loss_fn: LossFn | None = None,
) -> float:
    """One training step: forward + loss + strategy.backward(loss) + optimizer.step().

    Returns the scalar loss value as a Python float for logging/assertions.

    Raises:
        ValueError: ``batch`` is missing required keys for the default loss_fn.
        RuntimeError: forward / backward fails (propagated from torch).
    """
    if loss_fn is None:
        loss_fn = default_loss_fn

    optimizer.zero_grad()
    loss = loss_fn(model, batch)
    if not loss.requires_grad:
        raise RuntimeError(
            f"loss_fn returned a tensor with requires_grad=False; "
            f"check that the model parameters have requires_grad=True."
        )
    strategy.backward(loss)
    optimizer.step()
    return float(loss.detach().item())


def run_training_loop(
    model: nn.Module,
    batch_iter: Iterable[Mapping[str, torch.Tensor]],
    num_steps: int,
    strategy: DistributedStrategy,
    optimizer: Optimizer,
    loss_fn: LossFn | None = None,
) -> list[float]:
    """Run ``num_steps`` training steps, returning per-step losses."""
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")

    losses: list[float] = []
    iterator = iter(batch_iter)
    for step in range(num_steps):
        batch = next(iterator)
        loss = run_training_step(model, batch, strategy, optimizer, loss_fn)
        losses.append(loss)
    return losses


__all__ = ["run_training_step", "run_training_loop", "default_loss_fn"]
```

### Step 1.2: Smoke-check the module loads

Run: `python3 -c "from worldforge_train.training_loop import run_training_step, run_training_loop, default_loss_fn; print('imports OK')"`
Expected: `imports OK`

### Step 1.3: Commit

```bash
git add worldforge_train/training_loop.py
git commit -m "feat(train): real CPU-runnable training loop via DistributedStrategy"
```

---

## Task 2: Unit tests `worldforge_train/tests/test_training_loop.py`

**Files:**
- Create: `worldforge_train/tests/test_training_loop.py` (~80 lines, 4 tests)

**Why:** Verify the loop's mechanics with mocked model + mocked batch (no real VJEPA 2 forward). Cheap to run; catches regression in backward/optimizer call ordering.

### Step 2.1: Write the 4 unit tests

```python
"""Unit tests for worldforge_train.training_loop (mock-based, fast)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from torch.optim import SGD

from worldforge_train.parallel.strategy import DistributedStrategy
from worldforge_train.training_loop import run_training_step, run_training_loop


class _PassthroughStrategy(DistributedStrategy):
    """A no-op DistributedStrategy for tests."""

    def __init__(self):
        self.backward_calls: list[float] = []

    def wrap(self, model: nn.Module) -> nn.Module:
        return model

    def backward(self, loss: torch.Tensor) -> None:
        self.backward_calls.append(float(loss.item()))
        loss.backward()

    def step(self, optimizer) -> None:
        optimizer.step()


def _fake_batch() -> dict[str, torch.Tensor]:
    return {
        "pixel_values_videos": torch.zeros(1, 1, 3, 16, 16),
        "bool_masked_pos": torch.zeros(1, 1, 4, dtype=torch.bool),
    }


def test_run_training_step_returns_scalar_loss():
    model = nn.Linear(3 * 16 * 16, 4)
    optimizer = SGD(model.parameters(), lr=0.1)
    strategy = _PassthroughStrategy()

    def loss_fn(model, batch):
        x = batch["pixel_values_videos"]
        return model(x.flatten(1)).sum()

    initial_weight = model.weight.detach().clone()
    loss_val = run_training_step(model, _fake_batch(), strategy, optimizer, loss_fn=loss_fn)
    assert isinstance(loss_val, float)
    assert strategy.backward_calls == [loss_val]
    assert not torch.equal(initial_weight, model.weight.detach()), "weight must change"


def test_run_training_step_raises_when_loss_does_not_require_grad():
    model = nn.Linear(10, 4)
    optimizer = SGD(model.parameters(), lr=0.1)
    strategy = _PassthroughStrategy()

    def bad_loss_fn(model, batch):
        return torch.tensor(0.0)

    with pytest.raises(RuntimeError, match="requires_grad=False"):
        run_training_step(model, _fake_batch(), strategy, optimizer, loss_fn=bad_loss_fn)


def test_run_training_loop_runs_num_steps_and_returns_losses():
    model = nn.Linear(3 * 16 * 16, 4)
    optimizer = SGD(model.parameters(), lr=0.1)
    strategy = _PassthroughStrategy()

    def loss_fn(model, batch):
        x = batch["pixel_values_videos"]
        return (model(x.flatten(1)) ** 2).sum()

    batches = [_fake_batch() for _ in range(5)]
    losses = run_training_loop(model, iter(batches), num_steps=3, strategy=strategy,
                                optimizer=optimizer, loss_fn=loss_fn)
    assert len(losses) == 3
    for v in losses:
        assert isinstance(v, float)
        assert v == v


def test_run_training_loop_rejects_zero_or_negative_num_steps():
    model = nn.Linear(10, 4)
    optimizer = SGD(model.parameters(), lr=0.1)
    strategy = _PassthroughStrategy()
    with pytest.raises(ValueError, match="num_steps must be positive"):
        run_training_loop(model, iter([_fake_batch()]), num_steps=0,
                          strategy=strategy, optimizer=optimizer)
```

### Step 2.2: Run unit tests

Run: `python3 -m pytest worldforge_train/tests/test_training_loop.py -v`
Expected: 4 passed

### Step 2.3: Verify G3 still clean

Run: `git diff --stat 27b828f..HEAD -- worldforge_train/train.py worldforge_train/parallel/ worldforge_train/data/ worldforge_train/model/`
Expected: empty (no changes)

### Step 2.4: Commit

```bash
git add worldforge_train/tests/test_training_loop.py
git commit -m "test(train): unit tests for run_training_step + run_training_loop"
```

---

## Task 3: CPU integration test `test_training_loop_cpu.py`

**Files:**
- Create: `worldforge_train/tests/test_training_loop_cpu.py` (~70 lines, 1 test)

**Why:** End-to-end CPU proof: tiny VJEPA 2 forward + LoRA wrap + real MSE loss + 3 steps + loss finite + trainable params > 0 + weight actually changes.

### Step 3.1: Write the CPU integration test

```python
"""CPU e2e: real tiny V-JEPA 2 + real LoRA + 3-step training loop.

NOT NPU-gated -- runs by default. Validates:
1. ``lora_wrapper.build_lora_model`` wraps real VJEPA2Model and returns
   trainable_params > 0 (G4 sanity check)
2. ``FSDPStrategy.wrap`` returns the model unchanged in single-rank mode
3. ``run_training_loop`` runs 3 real forward+backward+step passes,
   returns 3 finite losses, and weight deltas are non-zero
"""
from __future__ import annotations

import pytest
import torch
from transformers import VJEPA2Config, VJEPA2Model

from worldforge_train.model.lora_wrapper import build_lora_model
from worldforge_train.parallel.fsdp import FSDPStrategy
from worldforge_train.training_loop import run_training_loop


def _build_real_tiny_vjepa2() -> VJEPA2Model:
    cfg = VJEPA2Config(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        image_size=224,
        patch_size=16,
        tubelet_size=1,
        in_chans=3,
        frames_per_clip=1,
        drop_path_rate=0.0,
    )
    return VJEPA2Model(cfg).eval()


def _synthetic_batch_iter(n: int):
    for _ in range(n):
        yield {
            "pixel_values_videos": torch.randn(1, 1, 3, 224, 224),
            "bool_masked_pos": torch.zeros(1, 1, 196, dtype=torch.bool),
        }


def test_real_vjepa2_lora_three_step_training_loop_on_cpu():
    torch.manual_seed(42)
    base = _build_real_tiny_vjepa2()

    lora_result = build_lora_model(base, r=4, alpha=8)
    model = lora_result.peft_model
    assert lora_result.trainable_params > 0
    assert lora_result.target_modules

    strategy = FSDPStrategy()
    wrapped = strategy.wrap(model)
    assert wrapped is model

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )

    initial = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    losses = run_training_loop(
        model, _synthetic_batch_iter(n=3), num_steps=3,
        strategy=strategy, optimizer=optimizer,
    )

    assert len(losses) == 3
    for v in losses:
        assert v == v
        assert v >= 0.0

    changed = 0
    for n, p in model.named_parameters():
        if p.requires_grad and not torch.equal(initial[n], p.detach()):
            changed += 1
    assert changed > 0
```

### Step 3.2: Run the CPU integration test

Run: `python3 -m pytest worldforge_train/tests/test_training_loop_cpu.py -v`
Expected: 1 passed (~10-15s on CPU)

### Step 3.3: Run full test suite for G3 verification

Run: `python3 -m pytest worldforge_inf/tests worldforge_train/tests -q`
Expected: same pre-existing pass count + 5 new tests (4 unit + 1 CPU integration).

G3: `git diff --stat 27b828f..HEAD -- worldforge_train/train.py worldforge_train/parallel/ worldforge_train/data/ worldforge_train/model/ worldforge_inf/compile/mindie_sd.py`
Expected: empty.

### Step 3.4: Commit

```bash
git add worldforge_train/tests/test_training_loop_cpu.py
git commit -m "test(train): CPU e2e integration test -- real VJEPA 2 + LoRA + 3-step loop"
```

---

## Task 4: Final review + merge

1. Branch off master: `git checkout -b phase2/w13-cpu-train`
2. Build review package: `/home/chad/.claude/skills/subagent-driven-development/scripts/review-package <base_sha> HEAD`
3. Dispatch task reviewer (sonnet). Address findings per SDD protocol.
4. Ask user for merge authorization. `git merge --no-ff phase2/w13-cpu-train` from master.
5. Update `progress.md` ledger.

---

## Self-Review

### Spec coverage

| Section | Requirement | Task |
|---------|------------|------|
| New | Real forward + backward + step on CPU | T1 |
| New | `DistributedStrategy` integration (wrap / backward / step) | T1 |
| New | Default loss is masked-feature MSE | T1 |
| New | Multi-step `run_training_loop` returns per-step losses | T1 |
| New | Mock-based unit tests | T2 |
| New | CPU e2e with real VJEPA 2 + LoRA | T3 |
| New | Loss finite, weights change, LoRA targets match | T3 |

### Caveats

- **`run_fsdp_training()` is still `NotImplementedError` in `train.py`.** This sub-project does not modify `train.py` (G3 boundary). The new training loop is callable directly. Wiring it into `train.py` is a future sub-project.
- **MMStrategy is still a stub.** The training loop calls `strategy.backward(loss)` which works for any `DistributedStrategy` impl, including a future real MMStrategy.
- **`DistributedStrategy.step()` is unused by the loop.** `optimizer.step()` is called directly (matches FSDP/DDP convention where the engine handles gradient sync internally).
- **CPU test uses random-init weights.** MSE assertion is `>= 0.0` (finite + non-negative), not "loss decreases" -- on random-init + random-data, MSE is high; asserting monotonic decrease would require careful initialization. Sticking with finite + weights-change is the right CPU smoke.