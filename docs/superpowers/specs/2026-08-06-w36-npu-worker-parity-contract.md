# W36 — NPU worker 同构契约 + NotImplementedError 收口门

**作者**: chad + Claude
**状态**: 🟡 设计文档（pre-Atlas gate）
**日期**: 2026-08-06
**关联**: roadmap G11（"NPU worker 未上机验证"）；为 phase-4 候选 #2 (W26)、
#3 (W21)、#4 (W27) 提供实施门。
**closes**: G11 的契约层；G11 的实现层由 W26/W27 自己各关（本文档为门槛）。

---

## 0. 范围

本文档是 **W36 = "CPU worker ↔ NPU worker 同构契约 + NotImplementedError
收口验收门"** 的预 Atlas 设计文档。**仅交付契约，不交付任何 NPU 实现**。
读者：phase-4 W20/W21/W26/W27 实施者，以及 W37 集成者。落地时建议保留
"W36" 这个 internal alias，但实际文件命名不复用项目 canonical W36 =
Reward 接口（G9）那个标签。

## 1. 背景

截至 `c9e7197`（W32）phase-3 收口，
`worldfoundry_inf/vllm_ascend_worker.py` 在 4 个方法上还挂着
`NotImplementedError`，等 Atlas 硬件窗口：

| Method | 行号锚点 | 当前文本 | 阻塞 |
|--------|----------|----------|------|
| `predict_masked_batch` | 178-181 | "W25 NPU batched masked prediction deferred to Atlas pilot" | W20 (`.to("npu")`) |
| `predict_temporal_batch` | 188-192 | "W25 NPU batched temporal prediction deferred to Atlas pilot" | W20 + W24 训练 ckpt |
| `predict_temporal` | 219-222 | "W24 NPU temporal-mode path deferred to Atlas pilot" | W20 + W24 ckpt |
| `predict_action_conditioned` | 239-243 | "W27 NPU action-conditioned path deferred to Atlas pilot" | W20 + W27 v2 ckpt |

CPU worker (`worldfoundry_inf/vjepa2_worker.py`) 已经全部实现，包括 W33
的 `predict_action_conditioned_batch`（NPU 当前**没有对应 stub**）。W36
先把契约画清楚，避免 W26/W27 实施时再发现 NPU 端缺方法签名而回头补。

W28 编排设计（`docs/superpowers/specs/2026-08-06-w28-train-inference-orchestration-design.md`）
定义 vRAM 配额、scheduler manifest、inferencer 锁；W37 集成时引用本契约。

W22 addendum（`docs/superpowers/specs/2026-08-05-w22-design-review-addendum.md`）
§1.1 提到 "HF 静默吞 `bool_masked_pos`" 一致性问题，本契约不去改 HF，
而是要求 CPU ↔ NPU 在同一 HF 静默语义下表现完全相同——避免 NPU 端因为
不同代码路径"恰好"消费 `bool_masked_pos` 而产生一方对、一方错的隐性
bug。

## 2. 同构契约

### 2.1 方法签名对照

| Method | CPU 现状 (`vjepa2_worker.py`) | NPU 现状 (`vllm_ascend_worker.py`) | 差异 | 优先级 |
|--------|------------------------------|------------------------------------|------|--------|
| `__init__(model_path, adapter_name, dtype, predictor_ckpt, action_predictor_ckpt, action_dim, past_frames, future_frames, patch_size, image_size)` | ✓ | ✓ (属性持久化) | NPU `__init__` 当真 base 时调 `super().__init__()`；CPU 不调 | — |
| `init_device()` | n/a | ✓ (delegating) | CPU worker 不接 vLLM HPA | — |
| `load_model()` | (通过 `_load_hf_model` 在 `__init__` 隐式) | ✓ (HF eager) | NPU 走 super + `from_pretrained`；CPU 走 `__init__` 内 eager | 文档化即可，不强求字面对齐 |
| `encode(video)` | ✓ | ✓ | 同（HF `last_hidden_state`） | — |
| `predict(masked, mask)` | ✓ | ✓ | 同；HF 静默吞 `bool_masked_pos` 见 W22 addendum §1.1 | — |
| `plan(video, actions)` | passthrough | passthrough | 同；真实实现属 roadmap 之后 phase-2 W4 parity | low |
| `predict_masked_batch(videos, mask_ratios)` | ✓ (W25) | stub | **NIE**; W36 §3.1 验收 | 高 |
| `predict_temporal_batch(videos, p, f)` | ✓ (W25+W24) | stub | **NIE**; W36 §3.2 | 高 |
| `predict_temporal(video, p, f)` | ✓ (W24) | stub | **NIE**; W36 §3.3 | 高 |
| `predict_action_conditioned(video, acts, p, f)` | ✓ (W27 v2) | stub | **NIE**; W36 §3.4 | 高 |
| `predict_action_conditioned_batch(videos, acts, p, f)` | ✓ (W33) | **缺 stub** | 新增占位 stub | 中 |

**新增 `predict_action_conditioned_batch` NPU stub 的原因**: CPU 端 W33
落地时把它加到了 `VJEPA2Worker`；NPU 端目前没有对应 stub。W36 契约必须
先把 stub 占位放好，否则 FastAPI dispatch 落到 NIE 时 stacktrace 与 CPU
不一致，违反 §0 "字节级同构"。

### 2.2 错误契约

| 错误类别 | 应抛 | 谁用 | 是否跨 CPU/NPU 一致 |
|----------|------|------|---------------------|
| 功能未上机 | `NotImplementedError` | stub | 跨 backend 一致 |
| API 契约违反 | `RuntimeError`（或 `ValueError`） | shape / dtype / 不一致 ckpt | 跨 backend 一致 |
| ckpt 文件不存在 | `FileNotFoundError` | loader | 一致 |
| ckpt 内容损坏 | `ValueError` | loader | 一致 |
| hccl init 失败 | `NotImplementedError`（**NPU 特有**） | NPU `__init__` 检测 `hccl.is_available()` → False | NPU 端日志记 "NPU backend unreachable"，API 上抛 503 |
| NPU OOM | `torch.cuda.OutOfMemoryError`（cann 复用命名空间） | runtime | NPU 端不吞；FastAPI → 503 + `Retry-After` |
| stale weight manifest | `ValueError` | NPU `load_model` 比对 hash | NPU 端；server → 409 Conflict |
| scheduler 拒绝 inference lock | `RuntimeError` | NPU `__init__` 末发起锁（W28 §D7-D9） | NPU 端；server → 503 "scheduler denied" |

**FastAPI 错误映射**（`worldfoundry_inf/server.py`）:

| 异常 | HTTP 状态 | 错误码 |
|------|-----------|--------|
| `NotImplementedError` | 503 | `feature_unavailable` |
| `RuntimeError` | 400 | `bad_request`（带 structured detail） |
| `ValueError` | 400 | `validation_failed` |
| `FileNotFoundError` | 404 | `ckpt_not_found` |
| `torch.cuda.OutOfMemoryError` | 503 | `npu_oom`（含 `Retry-After: 30`） |
| stale manifest `ValueError` | 409 | `stale_manifest` |
| scheduler 锁失败 `RuntimeError` | 503 | `scheduler_busy` |

CPU worker 不会触发最后两条；抛其他任何未捕获异常 → 500 with redacted
detail（NPU 上细节不外泄）。**§2.2 错误契约跨 backend 完全一致是优先项，
允许 FastAPI 例外行（hccl / OOM / stale / scheduler）在 NPU 不触发时不存在。**

## 3. NotImplementedError 收口验收门（pre-Atlas gate）

每个 stub 在 master 上替换为真实实现前**必须**通过：

### 3.1 `predict_masked_batch` 收口门

1. **CPU ↔ NPU 对照 fixture**（在 `tests/test_npu_cpu_parity.py` 新）：
   同 `model_path` + 无 `predictor_ckpt`，CPU 与 NPU worker 各跑一次
   `predict_masked_batch(videos, mask_ratios)`，输出 cosine per-token ≥
   0.999（bf16 噪声预算）。
2. **Atlas 单卡 smoke** (`scripts/w36_atlas_smoke_masked_batch.sh`):
   - `bench_results/atlas_baseline/masked_batch.json` 写明 clip/sec。
   - 阈值：单卡 clip/sec ≥ `bench_results/w22_7/*` 同口径中位数的 80%。
3. **shape / dtype 单元测试** (`tests/test_vllm_ascend_worker.py` 新增):
   forward shape `(B, N_tokens, hidden_dim)`、bf16 路径、batch 不一致抛
   `ValueError`。CPU 上 stub base class 跑（不依赖真 NPU）。

### 3.2 `predict_temporal_batch` / `predict_temporal` 收口门

1. **CPU ↔ NPU 对照 fixture**: 必须存在 W24 训练出的 JEPA ckpt（phase-3
   候选 #2 W20 完成后由 W24 training 跑出来）；同一 ckpt，CPU 与 NPU
   `predict_temporal` 输出 cosine ≥ 0.999。
   - **冷启动 parity**: ckpt 缺失时该验收门将 `pytest.importorskip` 跳过；
     仅当 `WORLDFOUNDRY_JEPA_CKPT` 环境变量指向真实 ckpt 时跑通。
2. **Atlas 单卡 smoke** (`scripts/w36_atlas_smoke_temporal.sh`):
   `bench_results/atlas_baseline/temporal.json`；阈值同上。
3. **shape / dtype 单元测试**: 单 / batch forward shape 一致；
   `video.shape[1] < past + future` 抛 `RuntimeError`（与 CPU 完全一致）。

### 3.3 `predict_action_conditioned` 收口门

1. **CPU ↔ NPU 对照 fixture**: 需 W27 v2 ckpt；同 ckpt，CPU 与 NPU
   `predict_action_conditioned(video, actions, p, f)` 输出 cosine ≥ 0.999。
   - 冷启动 parity: ckpt 缺失时 `pytest.importorskip` 跳过。
2. **Atlas 单卡 smoke** (`scripts/w36_atlas_smoke_action.sh`):
   `bench_results/atlas_baseline/action.json`；阈值同上。
3. **shape / dtype 单元测试**: action shape `(B, K, action_dim)` 严格校验；
   action_dim 不匹配抛 `RuntimeError`（与 CPU 完全一致）。
4. **NPU 必须新增 `predict_action_conditioned_batch` 占位 stub**:
   - 接受与 CPU 同形 list 输入；
   - 不在 W36 内部做真实实现；
   - 必须抛 `NotImplementedError("W36 batch action-mode NPU path:
     see W36 §3.3 for收口门")` 以保持契约对等。

### 3.4 通用门槛（所有方法）

1. **bf16 一致性**: NPU 端输出 dtype 必须与 CPU 端相同（`torch.bfloat16`
   或 `torch.float16`，视 HF 权重 dtype）。
2. **GRAD 隔离**: 所有方法都用 `@torch.no_grad()`；NPU 不开 grad。
3. **OOM 路径**: 在 batch 大到 NPU OOM 时不吞 `OutOfMemoryError`，
   上抛到 FastAPI。

## 4. Adapter-name dispatch (LoRA on NPU)

W22/W24/W25 训练时 LoRA 在 backbone；NPU 推理侧的 adapter 加载必须存在：

- vllm-ascend 0.18.0 的 `LoRARequest` API 在 vllm 0.10 上稳定；
  `VJEPA2NPUWorker.load_model` 之后必须接受 `(adapter_name, adapter_path)`
  作为可选 second-stage load（vLLM 通过 `add_lora` API 调）。
- CPU worker 不调 vLLM 的 `add_lora`（CPU 走 manual LoRA merge + state dict）；
  NPU worker 必须把 LoRA 权重的合并/切换委托给 vLLM 的 LoRA runtime。
- **契约**: `worldfoundry_inf/server.py` 的 CLI `--adapter-name` 在两
  个 backend 上位置一致；NPU server 启动时打印 `add_lora` 注册结果。

CPU worker 的 LoRA 路径不在 W36 范围；phase-3 完成后已独立 ship（W22
W24 W25），NPU 路径属 phase-4（roadmap 之后 W37+ 实施）。

## 5. W8A8 量化交互（W26 解锁前置）

`load_model()` 在 NPU worker 上必须：

1. 检查 `_vllm_ascend_plugin._patch_quantized_loader` 是否已 patch；
   若未 patch 且 ckpt 是量化权重 → `ValueError` 早失败（不让 fallback
   到 FP32 静默精度漂移）。
2. patch 过的路径对权重做 NZ 格式 cast + HCSP 调度（参考 W26 design，
   phase-4 候选 #2）。
3. CPU worker 不强制 patch；但**当 CLI 显式 `--use-quantized-ckpt` 时
   CPU 也走 patch 路径**，保证 parity。

## 6. 失败模式 + W28 协调

- **hccl backend 不可用**: NPU worker `__init__` 检测 `hccl.is_available()`
  → False 时不调 `super().__init__()`（仅留属性持久化 + stub base），
  server 端 worker 选 `WORLDFOUNDRY_BACKEND=cpu` 降级；info log 输出
  "hccl unavailable, falling back to CPU worker"。
- **OOM**: vllm-ascend 0.18.0 在 NPU OOM 时抛 `torch.cuda.OutOfMemoryError`
  （cann 复用 cuda 命名空间），NPU worker 不吞；FastAPI → 503 with
  `Retry-After: 30` header。
- **stale weight manifest**（W28 §D5）: adapter 路由用 ckpt manifest 的
  hash 验证；NPU worker 在 `load_model` 内做 hash 比对，不匹配 →
  `ValueError`，server 返回 409 Conflict。
- **scheduler 锁**（W28 §D7-D9）: W28 §D10 `disable training on 1-card`，
  NPU worker 在 `__init__` 末尾发起 "inferencer 锁" 请求；锁失败 →
  `RuntimeError` → API 503 "scheduler denied inference lock"。

## 7. 验收门槛（仅文档评审）

W36 文档作为设计交付，本身的验收门（不是 NPU 实施门）：

1. `git diff --check` 干净（仅 docs 文件）。
2. `python -m pytest --collect-only worldfoundry_inf/tests/` 不出现因本
   文档新增的失败（本文档不动测试）。
3. 用户评审通过；`docs/dev/next-steps-phase4-atlas.md` Tracking 表更新
   "W36 parity contract, done" 行 + commit hash。

## 8. 风险

- **门槛值 "cosine ≥ 0.999" 是试验性数字**，需在 W24 训练跑出第一份
  ckpt 之后校准。门槛值允许在第一次 Atlas 验证 smoke 后做一次 release
  note 修订。
- **W36 不在 phase-4 候选 #1/#6/#14 主线上**；建议在 phase-4 doc Tracking
  表里单独标注 "pre-W20, blocks W20/W21/W26/W27" 来显式绑定，避免候选
  计数漂移。
- **`predict_action_conditioned_batch` NPU stub**: 这是一个**纯占位**，
  不应被误读为已实现；如果未来 W37 集成阶段才发现该方法必须提前收口，
  该 stub 需要被替换为真实实现；§3.3.4 已明确 stub 的错误信息。

## 9. 关联文件索引

| 主题 | 位置 |
|------|------|
| NPU worker (主契约对象) | `worldfoundry_inf/vllm_ascend_worker.py` |
| CPU worker (契约 mirror) | `worldfoundry_inf/vjepa2_worker.py` |
| Server CLI + dispatch | `worldfoundry_inf/server.py` |
| vLLM plugin 装配 | `worldfoundry_inf/_vllm_ascend_plugin.py` |
| NPU env | `worldfoundry_common/npu_env.py` |
| W24 ckpt 加载 | `worldfoundry_train/model/jepa_inference.py` |
| W27 v2 ckpt 加载 | `worldfoundry_train/model/action_jepa_inference.py` |
| FSDP+HCCL 路径 | `worldfoundry_train/parallel/fsdp.py` (预留给 W21) |
| W22 训练主 spec | `docs/superpowers/specs/2026-08-03-w22-jepa-predictor-ema-design.md` |
| W22 设计评审 addendum | `docs/superpowers/specs/2026-08-05-w22-design-review-addendum.md` |
| W28 编排设计 | `docs/superpowers/specs/2026-08-06-w28-train-inference-orchestration-design.md` |
| W35 RL 框架 ADR | `docs/superpowers/specs/2026-08-04-w35-rl-framework-adr.md` |
| **Phase-4 plan** (本契约 entry) | `docs/dev/next-steps-phase4-atlas.md` |
