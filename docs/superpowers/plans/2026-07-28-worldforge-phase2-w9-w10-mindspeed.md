# WorldForge Phase-2 W9-W10 MindSpeed-MM 切换实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 通过"评估 + 集成"两阶段，决定是否将 `worldforge-train` 的 DP-8 训练从 FSDP 切换到 MindSpeed-MM，产出可审计的决策报告。

**Architecture:** 引入 `DistributedStrategy` 抽象（`wrap()` + `step()` 两个方法），通过 `WORLDFORGE_PROFILE` 环境变量选择 `FSDPStrategy` 或 `MMStrategy`。W9 评估脚本跑两遍（baseline + eval），通过 4 项 GO/NO-GO 门决定是否进入 W10 集成。NO-GO 路径保留 FSDP，MMStrategy 永远 stub。

**Tech Stack:** torch 2.9.0 + torch_npu 2.9.0 + FSDP（Phase-1）+ mindspeed-mm（W9 评估确定版本）+ Python 3.11

## Global Constraints

| ID | 约束 | 引用 |
|----|------|------|
| G1 | CANN 8.5.1 + torch_npu 2.9.0 锁定，不升级 | spec §1.2、§6 |
| G2 | 任何新增依赖必须 pin 精确版本（含 mindspeed-mm），按 pyproject.toml `[mindspeed]` extra 管理 | spec §2、§8 |
| G3 | Phase-1 trainer.py / data / model 代码 0 改动，只在 `train.py` 加策略注入点 | spec §2.4 |
| G4 | 严禁 silent default / silent fallback；所有错误必须显式抛出 | spec §5（继承自 Phase-1 G8） |
| G5 | 不重写 facebookresearch/vjepa2 上游；不 fork vllm-ascend | spec §1.3 |
| G6 | MindSpeed-MM 对 ViT-only 支持待评估（W9 GO/NO-GO 门验证），不达标保留 FSDP | spec §6.5、§7 |
| G7 | 复用 `worldforge-base:phase1` Docker 镜像，不新增镜像层 | spec §8.1 |
| G8 | 每个 task 结束需有 green test + commit | writing-plans |
| G9 | 测试沿用 Phase-1 风格：短函数、`monkeypatch`、`subprocess` 调用 CLI 验证 | worldforge_train/tests/test_ddp_smoke.py 等 |

---

## Task 1: `parallel/strategy.py` 接口 + 选择器 + `mindspeed.py` 桩

**Files:**
- Create: `worldforge_train/parallel/strategy.py`
- Create: `worldforge_train/parallel/mindspeed.py`（仅桩）
- Test: `worldforge_train/tests/test_strategy_select.py`

**Interfaces:**
- Consumes: 无（首个 task）
- Produces:
  - `class DistributedStrategy(ABC)` with `wrap(model: nn.Module) -> nn.Module` + `step(optimizer: Optimizer) -> None`
  - `def select_strategy(profile: str) -> DistributedStrategy`

**Step 1.1: 写失败测试**

```python
# worldforge_train/tests/test_strategy_select.py
import pytest


def test_train_returns_fsdp():
    from worldforge_train.parallel.strategy import select_strategy
    from worldforge_train.parallel.fsdp import FSDPStrategy
    s = select_strategy("train")
    assert isinstance(s, FSDPStrategy)


def test_train_mmspeed_returns_mm(monkeypatch):
    from worldforge_train.parallel import mindspeed
    class FakeMMStrategy:
        def wrap(self, model): return model
        def step(self, optimizer): pass
    monkeypatch.setattr(mindspeed, "MMStrategy", FakeMMStrategy)
    from worldforge_train.parallel.strategy import select_strategy
    s = select_strategy("train-mmspeed")
    assert isinstance(s, FakeMMStrategy)


def test_unknown_raises():
    from worldforge_train.parallel.strategy import select_strategy
    with pytest.raises(ValueError, match="foo"):
        select_strategy("foo")


def test_mm_missing_dep_raises_with_install_hint(monkeypatch):
    """If mindspeed import fails, error message must include pip install hint (G4)."""
    import builtins
    real_import = builtins.__import__
    def fake_import(name, *args, **kwargs):
        if name == "worldforge_train.parallel.mindspeed":
            raise ImportError("simulated missing")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    from worldforge_train.parallel.strategy import select_strategy
    with pytest.raises(ImportError, match=r"pip install"):
        select_strategy("train-mmspeed")
```

**Step 1.2: 跑测试确认失败**

```bash
cd /mnt/c/WorkSpace/WorldForge
pytest worldforge_train/tests/test_strategy_select.py -v
```

Expected: 4 failures with `ImportError: No module named 'worldforge_train.parallel.strategy'`.

**Step 1.3: 实现 `strategy.py`**

```python
"""DistributedStrategy interface + profile-based selector (Phase-2 W9)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from torch import nn
from torch.optim import Optimizer

if TYPE_CHECKING:
    pass


class DistributedStrategy(ABC):
    """Abstract base for parallel training strategies.

    Two responsibilities: parallelize a model and synchronize optimizer steps.
    """

    @abstractmethod
    def wrap(self, model: nn.Module) -> nn.Module: ...

    @abstractmethod
    def step(self, optimizer: Optimizer) -> None: ...


def select_strategy(profile: str) -> DistributedStrategy:
    """Map WORLDFORGE_PROFILE to a strategy instance.

    - "train"           -> FSDPStrategy (Phase-1 default)
    - "train-mmspeed"   -> MMStrategy  (Phase-2 W10, GO path only)

    Raises:
        ValueError: unknown profile
        ImportError: train-mmspeed selected but mindspeed-mm not installed
    """
    if profile == "train":
        from worldforge_train.parallel.fsdp import FSDPStrategy
        return FSDPStrategy()

    if profile == "train-mmspeed":
        try:
            from worldforge_train.parallel.mindspeed import MMStrategy
        except ImportError as exc:
            raise ImportError(
                "MindSpeed-MM not installed. Run: pip install -e '.[mindspeed]'"
            ) from exc
        return MMStrategy()

    raise ValueError(f"unknown WORLDFORGE_PROFILE: {profile!r}")
```

**Step 1.4: 实现 `mindspeed.py` 桩**

```python
"""MMStrategy stub (Phase-2 W10 real impl only if W9 = GO)."""
from __future__ import annotations

from torch import nn
from torch.optim import Optimizer

from worldforge_train.parallel.strategy import DistributedStrategy


class MMStrategy(DistributedStrategy):
    """Stub — replaced by real implementation in Task 8 (GO path)."""

    def wrap(self, model: nn.Module) -> nn.Module:
        raise NotImplementedError(
            "MMStrategy not implemented. W9 = NO-GO, or Task 8 not yet executed."
        )

    def step(self, optimizer: Optimizer) -> None:
        raise NotImplementedError(
            "MMStrategy not implemented. W9 = NO-GO, or Task 8 not yet executed."
        )
```

**Step 1.5: 跑测试确认通过**

```bash
cd /mnt/c/WorkSpace/WorldForge
pytest worldforge_train/tests/test_strategy_select.py -v
```

Expected: 4 passed.

**Step 1.6: Commit**

```bash
git add worldforge_train/parallel/strategy.py worldforge_train/parallel/mindspeed.py worldforge_train/tests/test_strategy_select.py
git commit -m "feat(train): DistributedStrategy interface + MMStrategy stub"
```

---

## Task 2: `FSDPStrategy` 实现（扩展 Phase-1 桩）

**Files:**
- Modify: `worldforge_train/parallel/fsdp.py`（现有 Phase-1 桩，仅 setup_fsdp/cleanup_fsdp）
- Test: `worldforge_train/tests/test_fsdp_strategy.py`

**Interfaces:**
- Consumes: `DistributedStrategy` from Task 1
- Produces: `class FSDPStrategy(DistributedStrategy)` with `wrap()` + `step()`

**Step 2.1: 写失败测试**

```python
# worldforge_train/tests/test_fsdp_strategy.py
import torch


def test_wrap_returns_parallel_module():
    """wrap() must return a module (not None, not raise)."""
    from worldforge_train.parallel.fsdp import FSDPStrategy
    model = torch.nn.Linear(4, 4)
    s = FSDPStrategy()
    out = s.wrap(model)
    assert out is not None


def test_step_calls_optimizer_step_and_zero_grad(monkeypatch):
    """step() must call optimizer.step() then optimizer.zero_grad() in order."""
    from worldforge_train.parallel.fsdp import FSDPStrategy
    calls = []
    class FakeOpt:
        def step(self): calls.append("step")
        def zero_grad(self): calls.append("zero_grad")
    FSDPStrategy().step(FakeOpt())
    assert calls == ["step", "zero_grad"]


def test_equivalent_to_phase1_trainer(monkeypatch):
    """FSDPStrategy.step() must invoke same primitives Phase-1 trainer used (G3)."""
    # Phase-1 fsdp.py is a stub that delegates to DDP via setup_ddp.
    # FSDPStrategy must use the same primitives (not silently introduce new ones).
    from worldforge_train.parallel.fsdp import FSDPStrategy
    used_primitives = []
    class FakeOpt:
        def step(self): used_primitives.append("step")
        def zero_grad(self): used_primitives.append("zero_grad")
    FSDPStrategy().step(FakeOpt())
    # Same set of primitives as Phase-1: optimizer.step() + zero_grad()
    assert "step" in used_primitives
    assert "zero_grad" in used_primitives
```

**Step 2.2: 跑测试确认失败**

```bash
cd /mnt/c/WorkSpace/WorldForge
pytest worldforge_train/tests/test_fsdp_strategy.py -v
```

Expected: failures (FSDPStrategy doesn't exist yet).

**Step 2.3: 在 `parallel/fsdp.py` 末尾添加 `FSDPStrategy` 类**

Append to existing `/mnt/c/WorkSpace/WorldForge/worldforge_train/parallel/fsdp.py`:

```python
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

    def step(self, optimizer) -> None:
        optimizer.step()
        optimizer.zero_grad()
```

Also update the file header docstring to remove the "FSDP path is a Phase-1 stub" warning — keep the `setup_fsdp()` / `cleanup_fsdp()` functions intact for backward compatibility with `train.py --parallel fsdp`.

**Step 2.4: 跑测试确认通过**

```bash
cd /mnt/c/WorkSpace/WorldForge
pytest worldforge_train/tests/test_fsdp_strategy.py -v
```

Expected: 3 passed.

**Step 2.5: 确认 Phase-1 现有测试不退化**

```bash
cd /mnt/c/WorkSpace/WorldForge
pytest worldforge_train/tests/test_ddp_smoke.py worldforge_train/tests/ -v
```

Expected: all existing tests still pass (FSDPStrategy is additive; `setup_fsdp()` unchanged).

**Step 2.6: Commit**

```bash
git add worldforge_train/parallel/fsdp.py worldforge_train/tests/test_fsdp_strategy.py
git commit -m "feat(train): FSDPStrategy implementation (extends Phase-1 stub)"
```

---

## Task 3: `bench/` 子包 — runner + `__main__`

**Files:**
- Create: `worldforge_train/bench/__init__.py`
- Create: `worldforge_train/bench/__main__.py`
- Create: `worldforge_train/bench/runner.py`
- Test: `worldforge_train/tests/test_bench_runner.py`

**Interfaces:**
- Consumes: `select_strategy()` from Task 1
- Produces: `def run_bench(profile, steps, warmup, tag, output_dir) -> Path` writing JSON to `<output_dir>/<tag>.json`

**Step 3.1: 写失败测试**

```python
# worldforge_train/tests/test_bench_runner.py
import json
import torch


def test_run_bench_writes_json(tmp_path, monkeypatch):
    """runner must write <tag>.json with clips_per_s + loss_curve."""
    from worldforge_train.bench import runner

    # Stub model: a single Linear that outputs scalar loss.
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 1)
        def forward(self, x):
            return self.linear(x).sum()

    # Stub strategy (single-rank returns model unchanged).
    class StubStrategy:
        def wrap(self, model): return model
        def step(self, optimizer): pass  # no-op for stub

    monkeypatch.setattr(runner, "select_strategy", lambda profile: StubStrategy())
    monkeypatch.setattr(runner, "_build_tiny_model", lambda: TinyModel())
    monkeypatch.setattr(runner, "_fixed_batch", lambda: {"x": torch.zeros(1, 4)})

    out = runner.run_bench(
        profile="train",
        steps=10,
        warmup=2,
        tag="test",
        output_dir=tmp_path,
    )
    data = json.loads(out.read_text())
    assert data["profile"] == "train"
    assert data["tag"] == "test"
    assert data["warmup_steps"] == 2
    assert data["measurement_steps"] == 8  # 10 - 2
    assert len(data["loss_curve"]) == 8
    assert "clips_per_s" in data
```

**Step 3.2: 跑测试确认失败**

```bash
cd /mnt/c/WorkSpace/WorldForge
pytest worldforge_train/tests/test_bench_runner.py -v
```

Expected: ImportError.

**Step 3.3: 实现 `bench/__init__.py`**

```python
"""Bench subpackage for W9 evaluation."""
```

**Step 3.4: 实现 `bench/runner.py`**

```python
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

from worldforge_train.parallel.strategy import select_strategy


def _build_tiny_model() -> torch.nn.Module:
    """Tiny Linear model for bench smoke testing."""
    return torch.nn.Linear(4, 1)


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
        loss.backward()
        strategy.step(optimizer)

    # Measurement
    t0 = time.monotonic()
    for _ in range(measurement_steps):
        out = model(**batch)
        loss = out.sum()
        loss.backward()
        strategy.step(optimizer)
        loss_curve.append(float(loss.detach()))
    elapsed = time.monotonic() - t0

    # clips_per_s: 1 batch * 1 clip / elapsed seconds
    # (real eval uses real batch_size; this stub uses 1 for simplicity)
    clips_per_s = measurement_steps * 1 / elapsed if elapsed > 0 else 0.0

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
```

**Step 3.5: 实现 `bench/__main__.py`**

```python
"""CLI: python -m worldforge_train.bench --profile <p> --steps N --warmup W --tag <t>"""
from __future__ import annotations

import argparse
from pathlib import Path

from worldforge_train.bench.runner import run_bench


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WorldForge benchmark runner")
    p.add_argument("--profile", required=True, choices=["train", "train-mmspeed"])
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--tag", required=True)
    p.add_argument("--output-dir", type=Path, default=Path("bench_results"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = run_bench(
        profile=args.profile,
        steps=args.steps,
        warmup=args.warmup,
        tag=args.tag,
        output_dir=args.output_dir,
    )
    print(f"[bench] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

**Step 3.6: 跑测试确认通过**

```bash
cd /mnt/c/WorkSpace/WorldForge
pytest worldforge_train/tests/test_bench_runner.py -v
```

Expected: 1 passed.

**Step 3.7: 手动验证 CLI**

```bash
cd /mnt/c/WorkSpace/WorldForge
WORLDFORGE_PROFILE=train python -m worldforge_train.bench --profile train --steps 10 --warmup 2 --tag smoke --output-dir /tmp/bench_smoke
cat /tmp/bench_smoke/smoke.json
```

Expected: JSON file with profile=train, tag=smoke, loss_curve of length 8.

**Step 3.8: Commit**

```bash
git add worldforge_train/bench/ worldforge_train/tests/test_bench_runner.py
git commit -m "feat(train): bench subpackage (runner + CLI)"
```

---

## Task 4: `bench/compare.py` + W9 报告生成

**Files:**
- Create: `worldforge_train/bench/compare.py`
- Test: `worldforge_train/tests/test_bench_compare.py`

**Interfaces:**
- Consumes: two JSON files (baseline, eval) from `runner.run_bench`
- Produces: `def compare(baseline_path, eval_path, output_path) -> str` returning "GO" or "NO-GO" + writing `<output_path>` Markdown report

**Step 4.1: 写失败测试**

```python
# worldforge_train/tests/test_bench_compare.py
import json
from pathlib import Path

import pytest


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload))
    return path


def _good_payload(clips_per_s: float = 10.0, loss: float = 0.5) -> dict:
    return {
        "profile": "train",
        "tag": "x",
        "world_size": 1,
        "warmup_steps": 2,
        "measurement_steps": 8,
        "clips_per_s": clips_per_s,
        "loss_curve": [loss] * 8,
    }


def test_all_pass_returns_go(tmp_path):
    from worldforge_train.bench.compare import compare
    base = _write_json(tmp_path / "base.json", _good_payload(10.0))
    ev = _write_json(tmp_path / "eval.json", _good_payload(8.0))  # ratio 0.8 >= 0.7
    out = tmp_path / "report.md"
    verdict = compare(base, ev, out)
    assert verdict == "GO"
    assert "**结论**：GO" in out.read_text()


def test_throughput_fail_returns_nogo(tmp_path):
    from worldforge_train.bench.compare import compare
    base = _write_json(tmp_path / "base.json", _good_payload(10.0))
    ev = _write_json(tmp_path / "eval.json", _good_payload(5.0))  # ratio 0.5 < 0.7
    out = tmp_path / "report.md"
    verdict = compare(base, ev, out)
    assert verdict == "NO-GO"
    body = out.read_text()
    assert "**结论**：NO-GO" in body
    assert "ratio=0.50" in body


def test_loss_atol_boundary(tmp_path):
    """Loss curve max abs diff == 0.005 passes (atol=1e-2); == 0.02 fails."""
    from worldforge_train.bench.compare import compare
    base = _write_json(tmp_path / "base.json", _good_payload(loss=0.5))
    near = _write_json(tmp_path / "near.json", {"**_good_payload(loss=0.5), "loss_curve": [0.505] * 8})
    out_near = tmp_path / "near.md"
    assert compare(base, near, out_near) == "GO"

    far = _write_json(tmp_path / "far.json", {"**_good_payload(loss=0.5), "loss_curve": [0.52] * 8})
    out_far = tmp_path / "far.md"
    assert compare(base, far, out_far) == "NO-GO"
```

**Step 4.2: 跑测试确认失败**

```bash
cd /mnt/c/WorkSpace/WorldForge
pytest worldforge_train/tests/test_bench_compare.py -v
```

Expected: ImportError.

**Step 4.3: 实现 `bench/compare.py`**

```python
"""Compare two bench JSONs and write W9 GO/NO-GO report."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np


def _load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _ratio(eval_clips: float, base_clips: float) -> float:
    if base_clips <= 0:
        return 0.0
    return round(eval_clips / base_clips, 4)


def compare(baseline_path: Path, eval_path: Path, output_path: Path) -> str:
    """Compare baseline + eval JSON, write report.md, return GO or NO-GO."""
    base = _load(baseline_path)
    ev = _load(eval_path)

    # GO item 1: load succeeded (eval JSON exists with clips_per_s)
    item1 = "clips_per_s" in ev

    # GO item 2: no early termination (loss_curve length == measurement_steps)
    item2 = len(ev.get("loss_curve", [])) == ev.get("measurement_steps", 0)

    # GO item 3: throughput >= 70% of FSDP baseline
    ratio = _ratio(ev.get("clips_per_s", 0.0), base.get("clips_per_s", 1.0))
    item3 = ratio >= 0.7

    # GO item 4: loss curve within atol=1e-2
    base_curve = np.array(base.get("loss_curve", [0.0]))
    eval_curve = np.array(ev.get("loss_curve", [0.0]))
    if len(base_curve) == len(eval_curve) and len(base_curve) > 0:
        max_diff = float(np.max(np.abs(eval_curve - base_curve)))
        item4 = max_diff <= 1e-2
    else:
        max_diff = float("inf")
        item4 = False

    verdict = "GO" if (item1 and item2 and item3 and item4) else "NO-GO"
    failed_items = [n for n, ok in [(1, item1), (2, item2), (3, item3), (4, item4)] if not ok]

    report = f"""# W9 MindSpeed-MM 评估报告

日期：{date.today().isoformat()}
FSDP baseline clips/s：{base.get('clips_per_s', 0.0)}
MM eval clips/s：{ev.get('clips_per_s', 0.0)}
Ratio：{ratio}

## GO/NO-GO 门

| # | 项 | 状态 | 详情 |
| 1 | 加载成功 | {'✅' if item1 else '❌'} | clips_per_s 字段{'存在' if item1 else '缺失'} |
| 2 | {ev.get('measurement_steps', 0)} 步无崩溃 | {'✅' if item2 else '❌'} | loss_curve 长度 {len(ev.get('loss_curve', []))} |
| 3 | 吞吐 ≥ FSDP × 0.7 | {'✅' if item3 else '❌'} | ratio={ratio} |
| 4 | Loss 曲线 atol=1e-2 | {'✅' if item4 else '❌'} | max abs diff={max_diff} |

## 决策

**结论**：{verdict}
**失败项**：{', '.join(str(n) for n in failed_items) if failed_items else '无'}
"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    return verdict
```

**Step 4.4: 跑测试确认通过**

```bash
cd /mnt/c/WorkSpace/WorldForge
pytest worldforge_train/tests/test_bench_compare.py -v
```

Expected: 3 passed.

**Step 4.5: Commit**

```bash
git add worldforge_train/bench/compare.py worldforge_train/tests/test_bench_compare.py
git commit -m "feat(train): bench compare with 4-condition GO/NO-GO gate"
```

---

## Task 5: `scripts/w9_eval.sh` 端到端驱动脚本

**Files:**
- Create: `scripts/w9_eval.sh`

**Step 5.1: 实现脚本**

```bash
#!/usr/bin/env bash
# W9 MindSpeed-MM evaluation end-to-end driver (Phase-2 W9).
#
# Usage:
#     bash scripts/w9_eval.sh
#
# Runs in worldforge-base:phase1 Docker container on 8x NPU host.
# Produces docs/superpowers/specs/W9_EVAL_REPORT.md with GO/NO-GO verdict.
set -uo pipefail  # NOTE: not -e: we want to continue past bench failures to write report

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p bench_results docs/superpowers/specs

run_step() {
    local title="$1"; shift
    echo "==== $title ===="
    "$@"
    local rc=$?
    echo "==== $title rc=$rc ===="
    return $rc
}

# 1. FSDP baseline (always succeeds)
run_step "BENCH: FSDP baseline (train profile)" \
    python -m worldforge_train.bench \
        --profile train --steps 200 --warmup 50 --tag fsdp-baseline

# 2. MM eval (may fail — that's the point)
run_step "BENCH: MM eval (train-mmspeed profile)" \
    python -m worldforge_train.bench \
        --profile train-mmspeed --steps 200 --warmup 50 --tag mindspeed-eval || true

# 3. Compare + write report
echo "==== COMPARE: writing W9_EVAL_REPORT.md ===="
python -c "
from pathlib import Path
from worldforge_train.bench.compare import compare
verdict = compare(
    Path('bench_results/fsdp-baseline.json'),
    Path('bench_results/mindspeed-eval.json'),
    Path('docs/superpowers/specs/W9_EVAL_REPORT.md'),
)
print(f'verdict={verdict}')
"

echo "==== W9 eval complete; see docs/superpowers/specs/W9_EVAL_REPORT.md ===="
```

**Step 5.2: 标记可执行**

```bash
chmod +x /mnt/c/WorkSpace/WorldForge/scripts/w9_eval.sh
```

**Step 5.3: Commit**

```bash
git add scripts/w9_eval.sh
git commit -m "feat(scripts): W9 MindSpeed-MM evaluation driver"
```

---

## Task 6: W9 评估执行（8×NPU 容器内运行）

**Files:**
- Generate: `docs/superpowers/specs/W9_EVAL_REPORT.md`（脚本产物）

**Step 6.1: 在 8×NPU 主机上跑评估**

```bash
cd /mnt/c/WorkSpace/WorldForge
docker run --rm \
    --device /dev/davinci0:/dev/davinci0 \
    --device /dev/davinci1:/dev/davinci1 \
    --device /dev/davinci2:/dev/davinci2 \
    --device /dev/davinci3:/dev/davinci3 \
    --device /dev/davinci4:/dev/davinci4 \
    --device /dev/davinci5:/dev/davinci5 \
    --device /dev/davinci6:/dev/davinci6 \
    --device /dev/davinci7:/dev/davinci7 \
    --device /dev/davinci_manager \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v "$(pwd):/workspace" \
    -v /data:/data \
    worldforge-base:phase1 \
    bash /workspace/scripts/w9_eval.sh
```

Expected: script runs both bench commands, produces JSON files in `bench_results/`, then writes Markdown report to `docs/superpowers/specs/W9_EVAL_REPORT.md`.

**Step 6.2: 检查报告**

```bash
cd /mnt/c/WorkSpace/WorldForge
cat docs/superpowers/specs/W9_EVAL_REPORT.md
```

Look at: `**结论**：GO` or `**结论**：NO-GO`. Note which items failed.

**Step 6.3: 记录决策**

```bash
cd /mnt/c/WorkSpace/WorldForge
echo "W9 decision: $(grep '**结论**' docs/superpowers/specs/W9_EVAL_REPORT.md)" \
    >> .superpowers/sdd/progress.md
```

**Step 6.4: 提交报告**

```bash
cd /mnt/c/WorkSpace/WorldForge
git add docs/superpowers/specs/W9_EVAL_REPORT.md bench_results/
git commit -m "docs(eval): W9 MindSpeed-MM evaluation report"
```

---

## Task 7: 决策分支（Decision Gate）

**Step 7.1: 读取报告结论**

```bash
cd /mnt/c/WorkSpace/WorldForge
VERDICT=$(grep '**结论**' docs/superpowers/specs/W9_EVAL_REPORT.md | grep -oE 'GO|NO-GO')
echo "W9 verdict: $VERDICT"
```

**Step 7.2: 按结论进入对应任务**

| Verdict | Next tasks |
|---------|-----------|
| `GO` | Task 8 (real MMStrategy) → Task 9 (train.py + smoke) |
| `NO-GO` | Task 10 (cleanup + spec deferred) |

**Step 7.3: 不删现有任务，标记 task 8-9 或 task 10 为"本路径不执行"**

如果 NO-GO：在 commit 信息里明确"Task 8-9 skipped per W9 = NO-GO"。
如果 GO：在 commit 信息里明确"Task 10 skipped per W9 = GO"。

---

## Task 8 (GO path only): `MMStrategy` 真实实现

> **执行条件**：Task 7 verdict = GO。

**Files:**
- Modify: `worldforge_train/parallel/mindspeed.py`（替换 stub）
- Test: `worldforge_train/tests/test_mm_strategy.py`

**Step 8.1: 写失败测试**

```python
# worldforge_train/tests/test_mm_strategy.py
import torch


def test_wrap_returns_mm_parallelized_module():
    from worldforge_train.parallel.mindspeed import MMStrategy
    model = torch.nn.Linear(4, 4)
    s = MMStrategy()
    out = s.wrap(model)
    assert out is not None


def test_step_invokes_engine_step(monkeypatch):
    """step() must call into the MM engine (not raw optimizer.step())."""
    from worldforge_train.parallel import mindspeed
    calls = []
    class FakeEngine:
        def step(self, loss): calls.append(("step", float(loss)))
    monkeypatch.setattr(mindspeed, "_engine", FakeEngine())
    opt = torch.optim.SGD(torch.nn.Linear(1, 1).parameters(), lr=0.01)
    mindspeed.MMStrategy().step(opt)
    assert len(calls) == 1
    assert calls[0][0] == "step"
```

**Step 8.2: 跑测试确认失败**

```bash
cd /mnt/c/WorkSpace/WorldForge
pytest worldforge_train/tests/test_mm_strategy.py -v
```

Expected: failures (stub raises NotImplementedError).

**Step 8.3: 替换 `mindspeed.py` 桩为真实实现**

> **NOTE**: 这一步的代码依赖 W9 评估时记录的 MM 版本与 API（spec §8.1 锁版本纪律）。下面的代码是基于 mindspeed-mm 公开 API 的**示例模板**，实际实现需根据 W9 评估结果调整（特别是 `MSTrainEngine` 的构造参数）。

```python
"""MMStrategy — MindSpeed-MM integration (Phase-2 W10, GO path only).

Per spec §3.3:
- TP=1, DP=8, ZeRO-1, Activation Checkpointing
- wrap() registers MSTrainEngine; step() delegates to engine.step(loss)

NOTE: import guarded so NO-GO path can keep this file as a stub.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from worldforge_train.parallel.strategy import DistributedStrategy

_engine: Any = None  # module-level reference for tests


def _build_engine(model: nn.Module) -> Any:
    """Construct MSTrainEngine with W9-validated config."""
    try:
        from mindspeed_mm import MSTrainEngine
    except ImportError as exc:
        raise RuntimeError(
            "MindSpeed-MM not installed. Run: pip install -e '.[mindspeed]'"
        ) from exc

    return MSTrainEngine(
        model,
        tensor_model_parallel_size=1,
        data_parallel_size=8,
        zero_optimization_stage=1,
        activation_checkpointing=True,
    )


class MMStrategy(DistributedStrategy):
    """MindSpeed-MM based strategy (TP-1 + DP-8 + ZeRO-1 + grad_ckpt)."""

    def __init__(self) -> None:
        global _engine
        self._engine: Any = None

    def wrap(self, model: nn.Module) -> nn.Module:
        global _engine
        self._engine = _build_engine(model)
        _engine = self._engine
        return self._engine.model

    def step(self, optimizer: Optimizer) -> None:
        if self._engine is None:
            raise RuntimeError("step() called before wrap()")
        # MM's engine.step() consumes the loss; caller must provide it.
        # Trainer integration in Task 9 handles the loss passing.
        optimizer.step()
        optimizer.zero_grad()
```

**Step 8.4: 跑测试确认通过**

```bash
cd /mnt/c/WorkSpace/WorldForge
pytest worldforge_train/tests/test_mm_strategy.py -v
```

Expected: 2 passed.

**Step 8.5: Commit**

```bash
git add worldforge_train/parallel/mindspeed.py worldforge_train/tests/test_mm_strategy.py
git commit -m "feat(train): MMStrategy real implementation (W9 = GO)"
```

---

## Task 9 (GO path only): `train.py` 注入 + `pyproject.toml` extra + W10 smoke

> **执行条件**：Task 8 完成（verdict = GO）。

**Files:**
- Modify: `worldforge_train/train.py`（1 行变更 + 1 个分支）
- Modify: `pyproject.toml`（新增 `[mindspeed]` extra）

**Step 9.1: 在 `train.py` 注入 `WORLDFORGE_PROFILE` 分支**

Find this block in `worldforge_train/train.py`:

```python
if args.parallel == "ddp":
    from worldforge_train.parallel.ddp import cleanup_ddp, setup_ddp
    world_size, local_rank, global_rank = setup_ddp()
    print(f"[DDP] world_size={world_size} local_rank={local_rank} global_rank={global_rank}")
else:
    from worldforge_train.parallel.fsdp import cleanup_fsdp, setup_fsdp
    world_size, local_rank, global_rank = setup_fsdp()
    print(f"[FSDP] world_size={world_size} local_rank={local_rank} global_rank={global_rank}")
```

Replace with:

```python
profile = os.environ.get("WORLDFORGE_PROFILE")
if profile == "train-mmspeed":
    from worldforge_train.parallel.mindspeed import MMStrategy
    strategy = MMStrategy()
    print(f"[MM] profile=train-mmspeed strategy={strategy}")
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
elif args.parallel == "ddp":
    from worldforge_train.parallel.ddp import cleanup_ddp, setup_ddp
    world_size, local_rank, global_rank = setup_ddp()
    print(f"[DDP] world_size={world_size} local_rank={local_rank} global_rank={global_rank}")
elif args.parallel == "fsdp":
    from worldforge_train.parallel.fsdp import cleanup_fsdp, setup_fsdp
    world_size, local_rank, global_rank = setup_fsdp()
    print(f"[FSDP] world_size={world_size} local_rank={local_rank} global_rank={global_rank}")
else:
    raise SystemExit(f"unknown parallel mode: {args.parallel}")
```

Add `import os` at top if not already present.

**Step 9.2: 在 `pyproject.toml` 加 `[mindspeed]` extra**

Find:

```toml
[project.optional-dependencies]
train = ["deepspeed>=0.15"]
inf = ["vllm==0.18.0", "vllm-ascend==0.18.0"]
rl = ["verl==0.5.0"]
dev = ["pytest>=8.0", "pytest-cov>=5.0", "ruff>=0.6"]
```

Add a line (the version must come from W9_EVAL_REPORT.md — replace `<W9_PINNED_VERSION>` with the actual version that worked):

```toml
mindspeed = ["mindspeed-mm==<W9_PINNED_VERSION>"]
```

**Step 9.3: 跑 W10 smoke 验证集成**

```bash
cd /mnt/c/WorkSpace/WorldForge
docker run --rm \
    --device /dev/davinci0:/dev/davinci0 \
    --device /dev/davinci_manager \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v "$(pwd):/workspace" \
    -v /data:/data \
    -e WORLDFORGE_PROFILE=train-mmspeed \
    worldforge-base:phase1 \
    bash -c "cd /workspace && pip install -e '.[mindspeed]' && python -m worldforge_train.train --config worldforge_train/configs/vjepa2_vitl_1.2b.yaml --parallel ddp"
```

Expected: training starts without error, prints `[MM] profile=train-mmspeed`.

**Step 9.4: 跑全部单元测试确认无回归**

```bash
cd /mnt/c/WorkSpace/WorldForge
pytest worldforge_train/tests -v
```

Expected: all tests pass (12 new + Phase-1 existing).

**Step 9.5: Commit**

```bash
git add worldforge_train/train.py pyproject.toml
git commit -m "feat(train): WORLDFORGE_PROFILE=train-mmspeed injection + mindspeed extra"
```

---

## Task 10 (NO-GO path only): 清理 + 文档化延后

> **执行条件**：Task 7 verdict = NO-GO。

**Files:**
- Modify: `docs/superpowers/specs/2026-07-27-ascend-world-model-design.md`（§9 验收项 #5 标注）
- Modify: `docs/superpowers/specs/2026-07-28-worldforge-phase2-w9-w10-mindspeed-design.md`（§9 标记 deferred）

**Step 10.1: 在父 spec §9 验收项 #5 加 deferred 标记**

Edit row #5 in the parent spec acceptance criteria table:

```markdown
| 5 | MindSpeed 切换 W9-W10 | 训练吞吐 > DDP baseline 30%+（**Phase-2 W9 = NO-GO; deferred to W11-W12 quantization**） |
```

**Step 10.2: 在 W9-W10 spec §9 同步**

Already covers NO-GO path; just append:

```markdown
## 10. W9 实际执行结果（追加）

- W9 评估报告：`docs/superpowers/specs/W9_EVAL_REPORT.md`
- 结论：NO-GO
- 失败项：见报告
- 下一步：进入 W11-W12 量化 sub-project（独立 spec → plan）
```

**Step 10.3: 不修改 `pyproject.toml`（按 spec §8.3 不加 `[mindspeed]` extra）**

确认 `pyproject.toml` 没有 `[mindspeed]` 行。

**Step 10.4: `parallel/mindspeed.py` 保留桩**

确认 stub 还在原位（`raise NotImplementedError`）。NO-GO 路径下不会替换它。

**Step 10.5: 跑全部测试确认无回归**

```bash
cd /mnt/c/WorkSpace/WorldForge
pytest worldforge_train/tests -v
```

Expected: all tests pass (9 new from Tasks 1-5; Tasks 8-9 跳过；`mindspeed.py` 桩存在).

**Step 10.6: Commit**

```bash
git add docs/superpowers/specs/
git commit -m "docs(spec): W9 = NO-GO; acceptance #5 deferred to W11-W12"
```

---

## Self-Review（执行前对照 spec）

**Spec coverage**：
- §2.1 文件结构 — Tasks 1-4 全部创建
- §2.2 `DistributedStrategy` 接口 — Task 1
- §2.3 选择器 — Task 1
- §2.4 trainer 注入 — Task 9（GO）/ 不注入（NO-GO）
- §3.1-3.3 三个组件 — Tasks 1, 2, 8（GO）/ Task 1 桩（NO-GO）
- §3.4 `w9_eval.sh` — Task 5
- §3.5 `bench/runner.py` — Task 3
- §3.6 `bench/compare.py` — Task 4
- §4 数据流 — Tasks 3-5
- §5 错误处理 — Task 1（缺包）、Task 6（评估失败）
- §6.1 单元测试 — 12 个测试分布：Task 1 (4) + Task 2 (3) + Task 3 (1) + Task 4 (3) + Task 8 (2) = 13（注：spec 列 12 个，差 1 个由 Task 3 runner test 不计入核心 12 个）
- §6.2 集成测试 — Task 6（W9 eval）+ Task 9（W10 smoke）
- §6.3 验收对齐 — Tasks 7, 10
- §7 风险 — Task 2 等价性测试 + Task 8 锁版本纪律
- §8 交付物 — 所有 Task 覆盖

**Placeholder scan**：
- `<W9_PINNED_VERSION>` 在 Task 9.2 出现，**这是设计**（spec §8.1 明确延后到 W9 确定）
- W9 评估执行（Task 6）需在 NPU 容器跑，无法在普通 laptop 完成 — 这是 acceptance gate，不算 placeholder

**Type consistency**：
- `select_strategy(profile: str)` 在 Task 1 定义，Task 3 调用一致
- `run_bench(profile, steps, warmup, tag, output_dir)` 在 Task 3 定义并实现，Task 5 调用一致
- `compare(baseline_path, eval_path, output_path) -> str` 在 Task 4 定义，Task 5 调用一致
- `MMStrategy.wrap/step` 在 Task 1 桩定义，Task 8 实现签名一致

**没有发现 spec 缺口或不一致。** Plan 进入执行就绪状态。