# WorldForge Phase-2 W9-W10 MindSpeed-MM 切换设计

**作者**：MiniMax-M3（与 chad 共同设计）
**日期**：2026-07-28
**状态**：✅ 用户已批准（2026-07-28 brainstorming 流程通过）
**父级设计**：`docs/superpowers/specs/2026-07-27-ascend-world-model-design.md` §6.4 W9-W10
**关系**：Phase-2 的第一个 sub-project；Phase-1（已合并）的前置

---

## 1. 目标与边界

### 1.1 一句话目标
通过"评估 + 集成"两阶段，决定是否将 `worldforge-train` 的 DP-8 训练从 FSDP 切换到 MindSpeed-MM（TP-1 + DP-8 + ZeRO-1 + Activation Checkpointing），并产出可审计的决策报告。

### 1.2 W9 vs W10 职责切分

| 周次 | 产出 | GO 条件 |
|------|------|---------|
| W9 | `W9_EVAL_REPORT.md`（GO 或 NO-GO 路径均产出） | 4 项 GO/NO-GO 门任一不过即 NO-GO |
| W10 | `parallel/mindspeed.py` + trainer 一行注入 + W10 smoke | 仅在 W9 = GO 时执行；NO-GO 时不启动 |

### 1.3 非目标（Out of scope）

- 对 `facebookresearch/vjepa2` 上游代码的任何改动（spec §1.3）
- MindSpeed-MM 配置调优（TP>1、CP、SP 等高级选项）
- Loss 曲线等价性的形式化证明
- 性能基准报告的人类可读 PR description
- W11-W12 量化 / W13-W14 RL / W15-W16 仿真（独立 sub-project）

---

## 2. 架构

### 2.1 文件结构

```
worldforge_train/
├── parallel/                      ← 新增
│   ├── __init__.py
│   ├── strategy.py                # DistributedStrategy 接口 + select_strategy()
│   ├── fsdp.py                    # FSDPStrategy：封装 Phase-1 T3 已用的 FSDP
│   └── mindspeed.py               # MMStrategy：仅 W10、仅 GO 路径实现
└── (现有 trainer.py / data / model / worldlogger 不改逻辑，仅 trainer.py 引入 strategy 注入)

scripts/
├── w9_eval.sh                     # W9 端到端评估脚本（生成 W9_EVAL_REPORT.md）
└── (现有 phase1_smoke.sh / phase1_ac_repro.sh 不动)

worldforge_train/bench/            # 新增（评估子包）
├── __init__.py
├── __main__.py                    # python -m worldforge_train.bench 入口
├── runner.py                      # 单 profile 200 步跑批 + warmup
└── compare.py                     # 两份 JSON 对比 + 写 W9_EVAL_REPORT.md

docs/superpowers/specs/
└── W9_EVAL_REPORT.md              # 评估产物（GO 或 NO-GO 都写）
```

### 2.2 接口契约：`DistributedStrategy`

```python
from abc import ABC, abstractmethod
from torch import nn
from torch.optim import Optimizer

class DistributedStrategy(ABC):
    @abstractmethod
    def wrap(self, model: nn.Module) -> nn.Module:
        """并行化模型。FSDP 返回 FSDPModule；MM 返回 MM MSTrainEngine 包装的 module。"""

    @abstractmethod
    def step(self, optimizer: Optimizer) -> None:
        """同步梯度 + optimizer.step() + zero_grad()。"""
```

### 2.3 选择器

```python
def select_strategy(profile: str) -> DistributedStrategy:
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

### 2.4 与现有 Trainer 的注入点

`worldforge_train/trainer.py` 仅新增一行（Phase-1 T3 已有结构，注入点位置在 `init_distributed()` 之后、`for batch in dataloader` 之前）：

```python
self.strategy = select_strategy(profile=os.environ["WORLDFORGE_PROFILE"])
self.model = self.strategy.wrap(self.model)
# 现有 step 循环中的 optimizer.step() 替换为 self.strategy.step(optimizer)
```

**约束**：`trainer.py` 其他代码 0 修改。

---

## 3. 组件

### 3.1 `parallel/strategy.py`（接口 + 选择器）

- **依赖**：`torch`、`torch_npu`（已有）；不依赖 FSDP 或 MM 具体实现
- **错误模式**：
  - 未知 profile → `ValueError("unknown WORLDFORGE_PROFILE: 'foo'")`
  - MM profile 但 `mindspeed` 包未安装 → `ImportError` 含安装指引

### 3.2 `parallel/fsdp.py`（FSDP 封装）

- **行为**：
  - `wrap(model)`：调用 `torch.distributed.fsdp.FSDPModule(model)`，配置 `use_orig_params=True`
  - `step(optimizer)`：`all_reduce(grad, group=dp_group)` → `optimizer.step()` → `optimizer.zero_grad()`
- **依赖**：`torch.distributed.fsdp`（Phase-1 已有）
- **等价性要求**：与 Phase-1 T3 trainer 直接调用 FSDP 的行为**完全一致**（GO/NO-GO 路径的参考基线；任何 FSDPStrategy 与 Phase-1 trainer 行为不一致都视为 bug）

### 3.3 `parallel/mindspeed.py`（MM 封装，W10 新增，仅 GO 路径实现）

- **行为**：
  - `wrap(model)`：构造 `MSTrainEngine(model, config=...)`，配置 `tensor_model_parallel_size=1, data_parallel_size=8, zero_optimization.stage=1, activation_checkpointing=True`
  - `step(optimizer)`：`engine.step(loss)`，由 MM 内部完成梯度同步 + optimizer 步进
- **依赖**：`mindspeed-mm`（可选依赖，`pyproject.toml` 的 `[mindspeed]` extra）；`torch_npu`（已有）
- **错误处理**：
  - `wrap()` 失败 → 必须抛出明确的"V-JEPA 2 submodule not recognized by MindSpeed-MM" 错误信息（不暴露 MM 内部 traceback 噪音）
  - 运行时 OOM/deadlock → 透传异常，由 `w9_eval.sh` 捕获

### 3.4 `scripts/w9_eval.sh`

- **入口**：`bash scripts/w9_eval.sh`
- **行为**：
  1. `docker run worldforge-base:phase1` 启动容器（复用 Phase-1 镜像，pin V-JEPA 2 commit 已就位）
  2. 运行两次 `python -m worldforge_train.bench --profile <p> --steps 200 --warmup 50 --tag <t>`：
     - `train` → tag `fsdp-baseline` → 写 `bench_results/fsdp-baseline.json`
     - `train-mmspeed` → tag `mindspeed-eval` → 写 `bench_results/mindspeed-eval.json`
  3. 运行 `python -m worldforge_train.bench.compare fsdp-baseline.json mindspeed-eval.json` → 生成 `docs/superpowers/specs/W9_EVAL_REPORT.md`
- **错误处理**：
  - MM 未安装 → 报告 "GO 项 1: FAIL — mindspeed-mm not installed"，不中断脚本
  - 任一 profile bench 命令退出码非 0 → 标记对应 GO 项 FAIL，stderr 末尾 50 行写入报告
- **失败路径**：4 项 GO 项全 FAIL 或部分 FAIL → 报告结论 = NO-GO；任一项 PASS → 报告结论 = GO

### 3.5 `worldforge_train/bench/runner.py`（单 profile 200 步跑批）

- **职责**：给定 profile + tag，跑 200 步（warmup 50 + measurement 150），输出 JSON 报告
- **JSON 字段**：
  ```json
  {
    "profile": "train",
    "tag": "fsdp-baseline",
    "world_size": 8,
    "warmup_steps": 50,
    "measurement_steps": 150,
    "clips_per_s": 12.34,
    "loss_curve": [0.693, 0.512, ...]   // 长度 = measurement_steps
  }
  ```
- **clips_per_s 定义**：单节点吞吐 = measurement_steps × batch_size × 视频片段数 / measurement 阶段总秒数。固定 batch_size=1、clip_len=16 帧。

### 3.6 `worldforge_train/bench/compare.py`（两 JSON 对比 + 写报告）

- **输入**：两个 JSON 路径（baseline, eval）
- **算法**：
  - GO 项 1 (load)：eval JSON 必须存在 + 含 `clips_per_s` 字段
  - GO 项 2 (no-crash)：eval JSON 必须存在 + `loss_curve` 长度 == 150（无提前终止）
  - GO 项 3 (throughput)：`eval.clips_per_s >= baseline.clips_per_s * 0.7`
  - GO 项 4 (loss)：`np.allclose(eval.loss_curve, baseline.loss_curve, atol=1e-2)`
- **输出**：`docs/superpowers/specs/W9_EVAL_REPORT.md`，结构：
  ```markdown
  # W9 MindSpeed-MM 评估报告
  日期：2026-XX-XX
  FSDP baseline clips/s：X1
  MM eval clips/s：X2
  Ratio：X2/X1

  ## GO/NO-GO 门
  | # | 项 | 状态 | 详情 |
  | 1 | 加载成功 | ✅/❌ | ... |
  | 2 | 100 步无崩溃 | ✅/❌ | ... |
  | 3 | 吞吐 ≥ FSDP × 0.7 | ✅/❌ | ratio=X.XX |
  | 4 | Loss 曲线 atol=1e-2 | ✅/❌ | max abs diff=X.XX |

  ## 决策
  **结论**：GO / NO-GO
  **理由**：...（4 项 PASS/FAIL 汇总）
  ```

---

## 4. 数据流

### 4.1 单次训练迭代（FSDP 或 MM 通用）

```
for batch in dataloader:           # Phase-1 T5 已实现
    batch = batch.to(f"npu:{local_rank}")
    loss = model(batch)
    loss.backward()
    strategy.step(optimizer)        # ← 新钩子（FSDP 手动同步 / MM engine.step）
    worldlogger.log(loss=loss.item(), step=step)
```

### 4.2 W9 评估数据流

```
bash scripts/w9_eval.sh
  │
  ├── docker run worldforge-base:phase1   # 复用 Phase-1 镜像（已 pin V-JEPA 2 commit）
  │
  ├── bench --profile train --tag fsdp-baseline
  │     → bench_results/fsdp-baseline.json
  │
  ├── bench --profile train-mmspeed --tag mindspeed-eval
  │     → bench_results/mindspeed-eval.json
  │
  └── compare fsdp-baseline.json mindspeed-eval.json
        → docs/superpowers/specs/W9_EVAL_REPORT.md
```

### 4.3 W10 集成数据流（仅 GO 路径）

```
WORLDFORGE_PROFILE=train-mmspeed python -m worldforge_train.train ...
  → init_distributed()
  → select_strategy("train-mmspeed") → MMStrategy()
  → strategy.wrap(model) → MM 并行化后的 model
  → 训练循环（strategy.step() 由 MM 处理）
```

---

## 5. 错误处理

| 场景 | 处理 | 用户感知 |
|------|------|----------|
| 未知 `WORLDFORGE_PROFILE` | `select_strategy` 抛 `ValueError` | 启动失败，错误信息含 profile 名 |
| `train-mmspeed` 但 MM 未装 | `select_strategy` 抛 `ImportError` 含 `pip install -e ".[mindspeed]"` 指引 | 启动失败 |
| MM 加载 V-JEPA 2 失败（`wrap()` 抛异常） | `w9_eval.sh` 捕获 → GO 项 1 标记 FAIL → 报告 stderr 末 50 行 | W9 报告 NO-GO；W10 不启动 |
| MM 训练中途崩（NPU error、deadlock） | `w9_eval.sh` 捕获退出码非 0 → GO 项 2 标记 FAIL | 同上 |
| MM 吞吐 < FSDP × 0.7 | `compare.py` 计算 ratio → GO 项 3 标记 FAIL | 同上 |
| MM loss 偏差 > atol=1e-2 | `np.allclose` 返回 False → GO 项 4 标记 FAIL + max abs diff | 同上 |
| W10 集成后 `train-mmspeed` 训练崩 | Trainer 抛异常；WorldLogger 记 stack trace；用户 unset profile 回 FSDP | profile 切换即恢复，无须重新部署 |

**原则**：所有错误处理遵循 Phase-1 G8（**不静默回退、不静默默认值**）。W9 任一项 FAIL 必须显式记录；W10 集成失败必须显式让用户切回 FSDP。

---

## 6. 测试

### 6.1 单元测试（无 NPU 可跑）

| 测试文件 | 覆盖 | 断言 |
|----------|------|------|
| `tests/test_strategy_select.py::test_train_returns_fsdp` | B1 选择器 | `select_strategy("train")` 返回 `FSDPStrategy` 实例 |
| `tests/test_strategy_select.py::test_train_mmspeed_returns_mm` | B1 选择器 | `select_strategy("train-mmspeed")` 返回 `MMStrategy`（monkeypatch MM 入口） |
| `tests/test_strategy_select.py::test_unknown_raises` | B1 错误 | `select_strategy("foo")` 抛 `ValueError`，信息含 profile 名 |
| `tests/test_strategy_select.py::test_mm_missing_dep_raises` | B1 错误 | MM profile + 缺包（`sys.modules['mindspeed'] = None`）→ `ImportError` 含 `pip install` |
| `tests/test_fsdp_strategy.py::test_wrap_calls_fsdp_module` | B2 接口 | mock `FSDPModule`，验证 `FSDPStrategy.wrap()` 调用 |
| `tests/test_fsdp_strategy.py::test_step_all_reduces_and_steps` | B2 接口 | mock optimizer + dp group，验证 `all_reduce` → `step` → `zero_grad` 顺序 |
| `tests/test_mm_strategy.py::test_wrap_configures_tp_dp_zero_grad_ckpt` | B3 接口 | mock `MSTrainEngine`，验证 config 传入参数为 `tp=1, dp=8, zero=1, grad_ckpt=True` |
| `tests/test_mm_strategy.py::test_wrap_failure_surfaces_clear_error` | B3 错误 | MM `wrap()` 抛 `ValueError` 含 "V-JEPA 2" 关键字 |
| `tests/test_bench_compare.py::test_all_pass_writes_go` | B3.6 | 4 项全 PASS 假数据 → 报告含 "**结论**：GO" |
| `tests/test_bench_compare.py::test_throughput_fail_writes_nogo` | B3.6 | throughput FAIL → 报告含 "**结论**：NO-GO" + ratio 数值 |
| `tests/test_bench_compare.py::test_loss_curve_atol_boundary` | B3.6 | max abs diff = 0.005 → PASS；= 0.02 → FAIL |

### 6.2 集成测试（需 8×NPU 容器）

| 测试 | 覆盖 | 断言 |
|------|------|------|
| `bash scripts/w9_eval.sh` 端到端 | B4 端到端 | 4 项 GO 项产出 + `W9_EVAL_REPORT.md` 存在 |
| W10 smoke（GO 路径专属） | W10 集成 | `WORLDFORGE_PROFILE=train-mmspeed` 跑 100 步训练不崩 + loss 单调下降（与 Phase-1 smoke 同等断言） |

### 6.3 与 spec §9 验收项 #5 对齐

- **GO 路径**：W10 完成后实测 `train-mmspeed` 吞吐 > FSDP 30%+，对应验收项 #5 "MindSpeed 切换 W9-W10：训练吞吐 > DDP baseline 30%+"
- **NO-GO 路径**：`W9_EVAL_REPORT.md` 记录 4 项 FAIL 原因 + 决策结论"保留 FSDP"+ 留待 W11-W12 量化补足性能。验收项 #5 标为 "deferred to W11-W12"。

---

## 7. 风险与缓解

| 风险 | 等级 | 缓解 |
|------|------|------|
| MindSpeed-MM 不支持 V-JEPA 2 ViT-only（spec §6.5 已点名） | 高 | W9 GO/NO-GO 门 4 项条件显式检测；NO-GO 不集成 |
| W9 评估脚本在 NPU 容器内运行失败（driver/HCCL 问题） | 中 | 复用 Phase-1 `worldforge-base:phase1` 镜像（已验证）；失败时退出码非 0 → 报告写 "无法评估：容器启动失败"，W10 推迟 |
| FSDPStrategy 与 Phase-1 trainer 直接调用 FSDP 行为不一致（导致 W9 baseline 数字不可比） | 中 | FSDPStrategy 实现须与 Phase-1 T3 完全等价；引入对比单元测试 `test_fsdp_strategy_equivalent_to_phase1_trainer`（mock 同一输入验证两路径 loss 数值相同） |
| MM 版本与 CANN 8.5.1 + torch_npu 2.9.0 矩阵错位（spec §2 锁定） | 中 | `[mindspeed]` extra 在 `pyproject.toml` 锁版本；与 Phase-1 同样的 G2 锁版本纪律 |
| MM 集成后 W10 smoke 失败（吞吐达标但 loss 不收敛） | 低 | W10 smoke 复用 Phase-1 smoke 同样的 loss 单调下降断言 |

---

## 8. 依赖与交付

### 8.1 依赖

- **Phase-1 已就位**：`worldforge-train`（T1-T3）、WorldLogger、DataLoader、LoRA wrapper
- **新增依赖**（`pyproject.toml` 的 `[mindspeed]` extra）：
  - `mindspeed-mm==<待 W9 评估确定>`（锁版本，对照 Phase-1 G2 纪律）
- **Docker**：复用 `worldforge-base:phase1` 镜像；不新增镜像层

### 8.2 交付物

1. `worldforge_train/parallel/` 3 个文件
2. `worldforge_train/trainer.py` 一行注入
3. `worldforge_train/bench/` 子包（runner + compare）
4. `scripts/w9_eval.sh`
5. `docs/superpowers/specs/W9_EVAL_REPORT.md`（GO 或 NO-GO 均产出）
6. `pyproject.toml` 新增 `[mindspeed]` extra
7. 单元测试 11 个（覆盖选择器 / FSDPStrategy / MMStrategy / compare.py）

### 8.3 NO-GO 路径下的最小交付

- 上述 1、3、4、5（不含 2、不含 mindspeed.py 真实实现）
- `parallel/mindspeed.py` 仅留 stub（`raise NotImplementedError("W9 = NO-GO; MMStrategy not implemented")`）
- `pyproject.toml` 不加 `[mindspeed]` extra

---

## 9. 验收

W9-W10 sub-project 完成对应 spec §9 验收项 #5：
- **GO**：MM 集成后训练吞吐 > FSDP baseline 30%+
- **NO-GO**：`W9_EVAL_REPORT.md` 记录决策，保留 FSDP，验收项 #5 标 deferred

W9-W10 完成后，下一个 Phase-2 sub-project 是 **W11-W12 量化**（独立 spec → plan 流程）。