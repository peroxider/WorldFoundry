# Next-Steps Todo — Phase-4 Atlas (2026-08-06)

## Background

`docs/dev/next-steps-w27-v2.md` (the post-W27 v2 plan) closed on 2026-08-06
with all four candidates shipped:

| # | Task | Commit |
|---|---|---|
| 1 | W33 plan endpoint | `28e8a6b` |
| 2 | W28 训推一体编排设计 | `d284809` |
| 3 | W25-roadmap full-FT + SFT schema | `1e5882d` |
| 4 | W32 AC latent-action head | `c9e7197` |

The CPU-only Phase-3 path (W22/W24/W25/W26/W27/W32/W33) is **complete**.
The remaining roadmap work — W20/W21/W26/W27/W29 of Phase-3 + all of
Phase-4 (W30–W39) + Phase-5 (W40+) — is gated on either an Atlas
hardware window (W20/W21/W26/W27/W29/W31/W34/W37/W39) or on a
predecessor decision/ADR (W30 env-selection, W35 framework, W40
frame-generation routing).

This document replaces `next-steps-w27-v2.md` as the active
next-steps plan and exists so the next dev session can pick up
without re-discovering which items are blocked on hardware vs.
blocked on spec.

## Master commit graph (Phase-3 closure recap)

```
e8150be  chore: W22 cleanup (ruff)
cb112fa  feat(w25): dynamic predict batcher
338ded8  feat(w26): temporal-objective (past-K → future-H loss)
796fb50  feat(w27): ActionConditionedPredictor + worker stub
0d8910f  feat(w27 v2): action-conditioned inference bundle + worker API
28e8a6b  feat(w33): action-conditioned predict endpoint + plan API
d284809  docs(w28): 训推一体编排设计文档 (G14)
1e5882d  feat(w25-roadmap): full-parameter fine-tuning mode + SFT data schema
c9e7197  feat(w32): AC latent-action head (actor.predict_latent_action)  ← HEAD
```

## Candidate ranking (Atlas-gated)

| # | Task | Closes | Hardware | CPU-only prep | Risk |
|---|---|---|---|---|---|
| 1 | **W20 — 训练链路 `.to("npu")` 迁移 + Atlas 单卡 LoRA smoke** | G1 | Atlas single-card | ½ session | low | ✅ done this session (CPU prep; Atlas 上机待硬件) |
| 2 | **W26 — `_patch_quantized_loader` 实现 + Atlas 上机** | G11 (partial) | Atlas single-card | ~1 session | low | ✅ done this session (CPU prep; Atlas 上机待硬件) |
| 3 | **W21 — 8 卡 FSDP 上机 + 多卡 checkpoint save** | G1, G4 | Atlas 8×NPU | ~1 session (after #1) | medium |
| 4 | **W27 — vLLM-Ascend worker Atlas e2e 验证** | G11 | Atlas single-card | ~½ session (after #2) | low | ✅ done this session (CPU prep; Atlas e2e 待硬件) |
| 5 | **W29 — Phase-3 集成回归** | G1–G10 regression | Atlas 8×NPU | ~1 session (after #1, #3) | medium |
| 6 | **W30 — 仿真环境选型 ADR (MuJoCo / SAPIEN / CARLA)** | G6 前置 | none (design) | ~1 session | low | ✅ done this session |
| 7 | **W35 — RL 框架 ADR 验证门 (verl / RLinf / 自研)** | G6, G7, G14 | partial (verl on NPU) | ~1–2 sessions | medium | W35.2 CPU ✅ / W35.1 待 Atlas |
| 8 | **W31 — EnvPool 抽象 + 首选环境向量化封装** | G6 | first-env-side | ~1 session (after #6) | medium | ✅ done this session |
| 9 | **W34 — PPO 训练器自研最小实现** | G7 | CPU smoke | ~1 session | medium | ✅ done this session |
| 10 | **W37 — RL 训练与主训练栈集成 (W28 编排落地)** | G7, G14 | Atlas cluster | ~2 sessions (after #7, #8) | high |
| 11 | **W36 — Reward model / 任务奖励接口** | G9 | none | ~½ session | low | ✅ done this session |
| 12 | **W38 — Model-based RL (predictor imagination rollout)** | G8, G7 | Atlas cluster | ~1–2 sessions (after #10) | medium |
| 13 | **W39 — Phase-4 集成回归** | G6–G10 regression | Atlas cluster | ~1 session (after #10) | high |
| 14 | **W40 — 帧生成路线 ADR 选型** | G13 前置 | none (design) | ~½ session | low | ✅ done this session |
| 15 | **W36 — NPU worker 同构契约 + NotImplementedError 收口门** | G11 (契约层) | none (design) | ~1 session | low |
| 16 | **W41 — Phase-5 v1: 具身智能虚拟场景三件套** (env_manipulator + vision_policy + cpu_loop_smoke) | G7 extension (NPU + 具身适配) | none (CPU) | ~1 session | low | ✅ done this session |

**Recommendation**: start with **#1 (W20)** — it's the physical prerequisite
for every other Atlas-gated item. While the hardware window is being
scheduled, **#6 (W30 env ADR)**, **#14 (W40 frame-gen ADR)**, and
**#15 (W36 NPU parity contract — pre-Atlas gate for W20/W21/W26/W27)**
can proceed on CPU and unblock Phase-4 / Phase-5 work.

---

## #1 — W20: 训练链路 `.to("npu")` 迁移 + Atlas 单卡 LoRA smoke

### Goal
把 W22/W24/W25 的训练主链路从 CPU/gloo 迁到 Atlas 800 A2 / 910B 单卡
NPU：模型加载后加 `.to("npu")`、loss/dtype/bf16 精度校验、跑通 100
steps LoRA 微调，loss 曲线正常下降。W20 是 G1 的物理前提。

### Closes
- G1 (训练链路无 `.to("npu")`).

### Deliverables
1. `worldfoundry_train/trainer.py` — 注入 `.to("npu")` 的设备迁移点
   （从 `worldfoundry_common/npu_env.is_npu_available()` 派生；CPU 路径
   不变以保持测试可跑）。
2. `worldfoundry_train/npu_env.py`（新）— NPU 设备的 `rank` / `world_size`
   抽象，与现有 gloo CPU 路径并列。
3. `scripts/w20_atlas_smoke.sh`（新）— 单卡 LoRA smoke，5 step 起步 +
   100 step 长程。
4. 测试：`tests/test_npu_trainer.py`（新）— mock `npu_available` →
   True，断言 `.to("npu")` 被调用。
5. 性能基线：clips/s、step latency 入库 `bench_results/atlas_baseline/`。

### Acceptance
- Atlas 单卡 100 steps LoRA 微调跑通，loss 单调下降。
- `python -m worldfoundry_train.train --mode lora --steps 100` 在 NPU
  上跑通，CPU 测试不退化。
- 至少一份 `bench_results/atlas_baseline/single_card.json` 入库。

### Blocked-on
- **Atlas 硬件窗口**（预约）。

### CPU-only prep (do while waiting for hardware)
- ✅ 起草 `.to("npu")` 的设备迁移点 PR（默认 `device = "cpu"` until
  `is_npu_available()`），让 PR 本身可在 CPU 上 CI 跑通（`is_npu_available`
  永远 False 的路径）。**2026-08-07 ship**：`worldfoundry_train/device.py`
  （`resolve_device` + `to_device` + `is_npu_available` proxy；不重复
  `npu_env.py`，因为 `worldfoundry_common.npu_env.is_npu_available()`
  已存在——按 G8）+ `worldfoundry_train/trainer.py` 注入迁移点
  （`strategy.wrap` 之后立刻 `model = to_device(model, resolve_device())`，
  JEPA 时也迁 predictor；print 一行 `[TRAIN] device=... npu_available=...`
  让迁移可见）+ `worldfoundry_train/tests/test_npu_trainer.py`（12 个单元
  测试：`resolve_device` CPU/NPU/override/env-var 路径 + 失败态 +
  `to_device` 调 `Module.to` + 不接受 unknown 设备 + chainable + 端到端
  `run_training` spy 验证迁移点真的发生在 `strategy.wrap` 之后）。
  CPU 路径字节级不变（现有 392 个 trainer 测试不退化），新加 12 个。
- ✅ 写 `tests/test_npu_trainer.py` 的 mock 版（已含在上一条交付里）。

### Out of scope
- 8 卡 FSDP（W21）。
- W8A8 量化加载器（W26 单独项）。
- DVPP 解码（W23 单独项；不影响 G1）。

### Estimated effort
**1 session**（CPU prep ½ session + Atlas 上机 ½ session）。

---

## #2 — W26: `_patch_quantized_loader` 实现 + Atlas 上机

### Goal
实现 vLLM-Ascend 路径下 W8A8 量化权重的 `_patch_quantized_loader`
（`worldfoundry_inf/_vllm_ascend_plugin.py:74-83`），并在 Atlas 上验证
量化模型加载后推理精度回归在阈值内。

### Closes
- G11 (部分 — vLLM-Ascend 加载器上机验证).

### Deliverables
1. `worldfoundry_inf/_vllm_ascend_plugin.py::_patch_quantized_loader` —
   实现（目前是 stub）。
2. `worldfoundry_inf/quantize.py`（新）— W8A8 tensor unpacking 工具。
3. `tests/test_quantize_loader.py`（新）— mock 量化权重，unpack 后的
   tensor 形状 + 精度回归。
4. `scripts/w26_atlas_w8a8_eval.sh`（已存在，复用 W17 路径）— Atlas
   跑量化推理 + 精度回归。

### Acceptance
- CPU 端 W8A8 unpack 单元测试全绿（构造小量化张量，unpack 后 MSE 在
  阈值内 vs FP32 baseline）。
- Atlas 上 `_patch_quantized_loader` 实际加载量化权重 + 完成
  一次完整 `/v1/world/predict` 调用，response 与 FP32 模型在
  误差阈值内（阈值定义在 `scripts/w17_atlas_w8a8_eval.sh`）。

### Blocked-on
- **Atlas 硬件窗口**（W20 之后；不需要等 FSDP，单卡即可）。

### CPU-only prep
- ✅ 实现 `_patch_quantized_loader` 本身（逻辑不依赖 NPU）。
- ✅ CPU 单元测试 + 精度回归对比。**2026-08-07 ship**：
  `worldfoundry_inf/quantize.py`（`pack_w8a8_weight` / `unpack_w8a8_weight`
  per-output-channel 对称 int8 + `pack_w8a8_activation` /
  `unpack_w8a8_activation` per-tensor 对称 int8 + `pack_linear` /
  `unpack_linear` 别名 + `load_quant_meta` / `is_w8a8_quantized` sidecar 读取
  + `ACCEPTANCE_MSE_THRESHOLD = 1e-3` round-trip 验收门；公开 API 与
  `worldfoundry_inf.compile.quantize` facade 严格分开 — G8 避免重复模块，
  二者负责 write vs read 不同职责）+ 替换 `_vllm_ascend_plugin.py
  ::_patch_quantized_loader` stub 为真实实现（lazy import `worldfoundry_inf.quantize`，
  通过 `quant_meta.json` sidecar 检测 W8A8 → 注册 vLLM ModelRegistry
  `register_model_loader` hook，weights_iterator 在遇到 int8 tensor + 邻接
  scale（`_load_w8a8_scales` 读 safetensors）时调用 `unpack_linear` 解包；
  非量化模型走默认 bf16 pass-through；malformed sidecar log + 退回 pass-through；
  AttributeError（旧 vLLM 无 `register_model_loader`）/ 其它异常 swallow + log，
  fail-soft per G4）+ `tests/test_quantize_loader.py`（28 个新单元测试：
  pack/unpack round-trip MSE 验收、validation errors、sidecar IO、loader
  hook 的 4 个 fail-soft 路径、closure 的 pass-through / unpack / missing-scale /
  malformed-sidecar / missing-safetensors-lib 5 个分支）。CPU 现有 109 个
  `worldfoundry_inf/tests/` 测试不退化（142 全绿，1 skip 是原有）。

### Estimated effort
**~1 session**（CPU 实现 ½ session + Atlas 上机 ½ session）。

---

## #3 — W21: 8 卡 FSDP 上机 + 多卡 checkpoint save

### Goal
Atlas 8×NPU 上跑 FSDP 8 卡 LoRA 微调；checkpoint save/resume 签名校验
通过；产出首个硬件 baseline（clips/s、MFU）。

### Closes
- G1 (部分 — 8 卡分布式), G4 (checkpoint save/resume).

### Deliverables
1. `worldfoundry_train/parallel/fsdp.py` — NPU FSDP 路径（`hccl` 后端）
   验证；目前 CPU gloo 路径已测，NPU 路径只 smoke。
2. `scripts/w21_atlas_fsdp_smoke.sh`（新）— 8 卡 FSDP + 100 steps +
   checkpoint save + resume。
3. `bench_results/atlas_baseline/fsdp_8card.json` — clips/s、step
   latency、MFU。

### Acceptance
- 8 卡 FSDP 100 steps 跑通，loss 单调下降。
- checkpoint save + 在 8 卡上 resume 一致。
- baseline 入库。

### Blocked-on
- **W20 完成**（`.to("npu")` 迁移是 W21 的前提）。

### Estimated effort
**~1 session**（大部分时间花在 8 卡预约）。

---

## #4 — W27: vLLM-Ascend worker Atlas e2e 验证

### Goal
跑通 `vllm_ascend_worker.py:20-22`（目前 `NotImplementedError` 在
`predict_action_conditioned` 等几个方法上）；smoke 全绿，性能数字
入库。

### Closes
- G11 (vLLM-Ascend worker 端到端).

### Deliverables
1. `worldfoundry_inf/vllm_ascend_worker.py` — 替换
   `NotImplementedError` 桩为真实实现。
2. `scripts/w27_vllm_ascend_smoke.sh`（新）— 端到端 smoke +
   `bench_results/` 入库。

### Acceptance
- smoke 全绿（已有 stub，需要补真实路径）。
- 性能数字入库。

### Blocked-on
- **W20 + W26**（NPU 加载 + 量化加载是 e2e 的前提）。

### CPU-only prep (do while waiting for hardware)
- ✅ NPU worker stub parity 测试，保证 Atlas 上机时「NPU 路径契约与 CPU
  路径一致」，不必等硬件就能 catch 一类整类潜在 bug。**2026-08-07 ship**：
  `worldfoundry_inf/vllm_ascend_worker.py` 新增 `predict_action_conditioned_batch`
  stub（CPU sister 在 `vjepa2_worker.py:358` 早已实现，NPU 路径之前完全缺失——
  5 个 batched / temporal / action-conditioned predict 方法在 NPU 路径上
  之前只暴露 4 个，W33 batcher surface 不闭合；新增 stub 与现有 4 个 stub
  一致：`raise NotImplementedError("... deferred to Atlas pilot (roadmap G11)")`）
  + `worldfoundry_inf/tests/test_vllm_ascend_worker_parity.py`（新，19 个
  CPU 单元测试）：(1) `inspect.signature` 跨类对比 5 个 parity method
  的参数 (kind, default, annotation) + return annotation 严格相等（参数重命名
  / 重排 / 漏标 return 都会失败）；(2) 参数顺序逐项匹配（dispatch 走 positional
  args）；(3) 每个 stub 调用都 raise NotImplementedError 含 "Atlas pilot"（防止
  未来 refactor 把它意外接成 eager CPU fallback，defeat the Atlas-pilot gate）；
  (4) 即使 `load_model` 真实跑过，stub 仍然 raise（regression guard）；
  (5) `predict_action_conditioned_batch` 的 NPU 签名精确等于 CPU 签名
  + CPU 端签名 introspectable smoke。CPU 现有 109 个 `worldfoundry_inf/tests/`
  测试不退化（161 全绿，1 skip 是原有）。ruff check 干净。

### Estimated effort
**~½ session**（前提项完成后）。

---

## #5 — W29: Phase-3 集成回归

### Goal
训练→导出→推理服务加载→predict 端点全链路在 Atlas 上跑通；CI 集成
测试 + Atlas 周度回归。

### Closes
- G1–G10 regression.

### Deliverables
1. `scripts/w29_phase3_regression.sh`（新）— 全链路 + 阈值对比。
2. `bench_results/phase3_regression.jsonl` — 周度回归日志。

### Acceptance
- 单跑回归 + 与历史 baseline 对比无回归。

### Blocked-on
- **W20 + W21 + W26 + W27**（Phase-3 全部硬件项）。

### Estimated effort
**~1 session**。

---

## #6 — W30: 仿真环境选型 ADR (MuJoCo / SAPIEN / CARLA)

### Goal
roadmap Phase-4 入口决策。MuJoCo 物理 + 简单几何 vs SAPIEN 复杂场景 +
仿真相机 vs CARLA 自动驾驶。决策结果写 ADR。

### Closes
- G6 前置.

### Deliverables
1. `docs/superpowers/specs/2026-08-06-w30-env-selection-adr.md` —
   决策文档：选型、对比、与 NPU learner 的分体架构、env CPU 节点 +
   learner NPU 节点的拓扑。

### Acceptance
- ADR 评审通过。
- 决策与 G6 roadmap 一致（仿真任务能产出 1 个端到端 demo）。

### Blocked-on
- 无（纯 design，CPU-only）。

### Estimated effort
**~1 session**。

---

## #7 — W35: RL 框架 ADR 验证门 (verl / RLinf / 自研)

### Goal
按 `docs/superpowers/specs/2026-08-04-w35-rl-framework-adr.md` 的
验证门 W35.1–W35.4 执行：选 RLinf（仿真器生态 + 训推编排），verl 备选
（昇腾一方适配最深），自研 PPO（W34）为基线。

### Closes
- G6, G7, G14.

### Deliverables
1. `bench_results/w35/` — 四个验证门的实测数据。
2. ADR 验证门通过后产出 **W35 验证门报告**（执行结果更新到 ADR）。

### Acceptance
- 四个验证门都过，框架决策落地。

### Blocked-on
- **W30**（env 选定才有 learner 能跑端到端）。
- 部分 verl 路径需要 Atlas（可选）。

### Estimated effort
**~1–2 sessions**。

---

## #8 — W31: EnvPool 抽象 + 首选环境向量化封装

### Goal
按 W30 选定的环境实现 EnvPool 抽象（参考 OpenAI Gym 的 vec-env 接口）
+ 首选环境的向量化封装。

### Closes
- G6.

### Deliverables
1. `worldfoundry_rl/env_pool.py`（新）— vec-env 接口。
2. `worldfoundry_rl/env_<chosen>.py`（新，按 W30 决策选一）— 实际
   环境封装。
3. `tests/test_env_pool.py`（新）— 单元测试。
4. `bench_results/w31/vec_env_throughput.json` — 向量化 env step
   吞吐基准。

### Acceptance
- 向量化 env step 吞吐基准达标（具体阈值在 W30 ADR 里定）。

### Blocked-on
- **W30**（env 选型）。

### Estimated effort
**~1 session**。

---

## #9 — W34: PPO 训练器自研最小实现

### Goal
PPO 训练器落地：rollout 收集 → advantage 估计 → 梯度更新闭环。先
自研最小实现验证正确性，给 W35 框架选型做基线。

### Closes
- G7.

### Deliverables
1. `worldfoundry_rl/ppo.py`（新）— PPO 训练器。
2. `tests/test_ppo_smoke.py`（新）— 合成任务上 reward 曲线上升。

### Acceptance
- 仿真任务上 reward 曲线上升（CPU smoke 即可）。

### Blocked-on
- ~~**W31**（env pool 才有可 rollout 的 env）。~~ — W31 已 ship（`07dc9b5`），
  本项已解锁并于 2026-08-06 完成。

### Estimated effort
**~1 session**。

---

## #10 — W37: RL 训练与主训练栈集成

### Goal
按 `docs/superpowers/specs/2026-08-06-w28-train-inference-orchestration-design.md`
§RL 架构图落地：共用 checkpoint/LoRA 体系、ActorPool/EnvPool/LearnerPool
编排（**D7 60/40 vRAM 配额 + D9 auto 75/25 full-FT + D10 disable training
1-card** 等 W28 决策要在 W37 落地）。

### Closes
- G7, G14.

### Deliverables
1. `worldfoundry_rl/actor_pool.py`、`env_pool.py`、`learner_pool.py`。
2. `worldfoundry_inf/scheduler.py` — 扩 W28 设计文档定义的
   scheduler pool 接入 RL 工作负载。
3. `tests/test_rl_distributed_smoke.py`（新）— 分布式 RL smoke。

### Acceptance
- 分布式 RL 训练 smoke 通过。

### Blocked-on
- **W35**（框架选型）。
- **W31**（env pool）。
- **W34**（PPO baseline）。
- **Atlas cluster**（多卡 learner）。

### Estimated effort
**~2 sessions**（集成工作量大）。

---

## #11 — W36: Reward model / 任务奖励接口

### Goal
Reward model 接口定义 + 至少 1 个任务的 reward 实现（含从世界模型
latent 距离构造 intrinsic reward 的选项）。

### Closes
- G9.

### Deliverables
1. `worldfoundry_rl/reward.py`（新）— 接口。
2. 至少 1 个 task-specific reward（按 W30 选定的环境）。

### Acceptance
- 接口文档 + 1 个 reward 实现。

### Blocked-on
- **W30**（env 选型）。

### Estimated effort
**~½ session**。

---

## #12 — W38: Model-based RL (predictor imagination rollout)

### Goal
用 W22/W24/W27 落地后的 predictor 做 imagination rollout，降低真实
仿真步数。样本效率对照实验。

### Closes
- G8, G7.

### Deliverables
1. `worldfoundry_rl/imagination.py`（新）— predictor rollout。
2. `bench_results/w38/sample_efficiency.json` — 对照实验。

### Acceptance
- 样本效率有可量化的提升。

### Blocked-on
- **W37**（RL 主链路）。

### Estimated effort
**~1–2 sessions**。

---

## #13 — W39: Phase-4 集成回归

### Goal
encode→plan→RL 训练→policy 回灌推理服务全链路；Atlas 端到端 demo
任务收敛。

### Closes
- G6–G10 regression.

### Deliverables
1. `scripts/w39_phase4_regression.sh`（新）— 全链路 + 阈值对比。
2. `bench_results/phase4_regression.jsonl` — 回归日志。

### Acceptance
- Atlas 端到端 demo 任务收敛。

### Blocked-on
- **W37 + W38**（Phase-4 全部硬件项）。

### Estimated effort
**~1 session**。

---

## #14 — W40: 帧生成路线 ADR 选型 (A / B / C)

### Goal
V-JEPA 2 原生不生成像素。三个候选路线（轻量 pixel decoder / 扩散
decoder / 换模型线）的 ADR 选型 + 数据算力预算。

### Closes
- G13 前置.

### Deliverables
1. `docs/superpowers/specs/2026-08-06-w40-frame-gen-adr.md`。

### Acceptance
- ADR 评审通过 + 预算评估。

### Blocked-on
- 无（纯 design）。

### Estimated effort
**~½ session**。

---

## Tracking

| # | Task | Committed on master? | Status | Blocked-on |
|---|---|---|---|---|
| 1 | W20 `.to("npu")` + Atlas 单卡 LoRA smoke | `6f34894` (CPU prep), `288ca14` (smoke script + runbook) | **CPU prep done; Atlas 上机待硬件** | Atlas single-card |
| 2 | W26 W8A8 量化加载器 | `0f6c305` | **CPU prep done; Atlas 上机待硬件** | Atlas single-card |
| 3 | W21 8 卡 FSDP 上机 | not yet | **proposed** | W20 |
| 4 | W27 vLLM-Ascend worker e2e | `0b7f82a` | **CPU prep done; Atlas e2e 待硬件** | W20 + W26 |
| 5 | W29 Phase-3 集成回归 | not yet | **proposed** | W20 + W21 + W26 + W27 |
| 6 | W30 仿真环境选型 ADR | `1e3db44` | **done** | none |
| 7 | W35 RL 框架 ADR 验证门 | `024be09` | **W35.2 CPU ✅ / W35.1 待 Atlas** | W30 + Atlas (W35.1) |
| 8 | W31 EnvPool 抽象 | `07dc9b5` | **done** | W30 (unblocked) |
| 9 | W34 PPO 训练器 | `af460d1` | **done** | W31 (unblocked) |
| 10 | W37 RL 训练与主训练栈集成 | `fbcf7b6` | **W37 v1 CPU skeleton ✅ / cluster W37+** | W31 + W34 (unblocked; cluster待 Atlas) |
| 11 | W36 Reward 接口 | `0264a61` | **done** | W30 (unblocked; env = MuJoCo) |
| 12 | W38 Model-based RL | not yet | **proposed** | W37 |
| 13 | W39 Phase-4 集成回归 | not yet | **proposed** | W37 + W38 |
| 14 | W40 帧生成路线 ADR | `759dd28` | **done** | none |
| 15 | W36 NPU worker 同构契约 | `f923170` | **done** | none (CPU-only doc) |
| 16 | RTI v1 CPU 渲训推闭环 | `aede80b` | **done** | none (CPU-only, 无 HF 1.2B 下载) |

**调度建议**：
- ~~立刻可做（无 hardware 依赖）：#6, #14, #15（三个 design tasks）。~~ — 2026-08-06 三件 ADR 全 ship，phase-4 立刻可做桶 closed。
- 预约硬件：#1 → #2 / #3 并行 → #4 → #5。
- 硬件到位 + ADR 完成后：#7 → #8 ✅ / #9 ✅ / #11 ✅ → #10 → #12 → #13。
- W36 #15 必须是 W20/W21/W26/W27 任一实施前的先决条件（契约 + 收口门）。
- #11 W36 canonical (Reward 接口) 已 ship（2026-08-06）：`worldfoundry_rl/reward.py`
  （RewardFn Protocol + ensure_reward_fn + compose）+ `reward_impl.py`
  （FrankaKitchenReward + LatentDistanceReward）+ 25 个单元测试。
- #9 W34 (PPO) 已 ship（2026-08-06）：`worldfoundry_rl/ppo.py` + 34 个单元测试
  + 6-seed learning-curve baseline。**Phase-4 CPU-only 桶清空**——
  `#6/#14/#15/#7/#8/#9/#11` 七件不需 Atlas 的实现 / ADR / 设计项已 ship，
  `#1/#2/#4` 三件 CPU-only 预备亦于 2026-08-07 ship（`6f34894` W20
  `.to("npu")` 迁移点 + `tests/test_npu_trainer.py` 12 个 mock 测试 /
  `0f6c305` W26 W8A8 unpack 工具 + W8A8 loader hook + `tests/test_quantize_loader.py`
  28 个 MSE < 1e-3 验收门测试 / `0b7f82a` W27 NPU worker stub parity +
  `tests/test_vllm_ascend_worker_parity.py` 19 个 `inspect.signature` 跨类守护）。
  其余未完成项（#1/#2 上机、#3/#4 e2e/#5、#10/#12/#13、#7 的 W35.1/W35.4）
  无一可在 CPU 上推进——下一步是
  **预约 Atlas 硬件窗口**，入口仍是 #1 (W20 `.to("npu")` 迁移)。
  唯一的额外 CPU-side 动作是可选的 `mujoco` + `dm_control` 安装，装上后可把
  W31 的 FakeEnv 基线和 W34 的合成任务基线换成真实 FrankaKitchen 数字。
- #16 RTI v1 (CPU 渲训推闭环) 已 ship（2026-08-07，`aede80b`）：把「渲染 →
  训练 → 推理」在纯 CPU 上跑通，不依赖 HF V-JEPA 2 1.2B 下载、不依赖 Atlas。
  - **Render** `worldfoundry_train/data/synthetic_video.py` +
    `synthetic_manifest.py` —— cv2.VideoWriter 写确定性 `.mp4`（弹球 / 旋转条纹）
    + Vjepa2Dataset 可直接消费的 CSV manifest（schema 未改）。
  - **Train** `worldfoundry_train/model/tiny_jepa.py` —— TinyVJEPA2Encoder
    （325k 参数，forward 接口对齐 HF `VJEPA2Model`）+ TinyJEPAPredictor（96k）
    + TinyPixelDecoder（28k）；`ema_compat.py` 提供常动量 EMA + 原子 checkpoint。
  - **Inference** 用训练好的 pixel decoder 把 predictor 的 masked-token 预测
    渲回像素，与 GT 并排出 PNG，cosine 相似度作质量探针。
  - 入口：`bash scripts/render_train_inference_cpu.sh`（2 gate：pytest +
    orchestrator）。实测 8 clips × 4 帧 @64×64 / 50 步单 CPU：render 0.09s、
    train 4.01s（loss 44.20 → 11.05）、inference 0.19s、recon_cosine 0.7554。
  - 与 W40 Route A（轻量 pixel decoder）一致；是 #12 W38 (imagination rollout)
    的 CPU 前置件——W38 可直接复用 TinyJEPAPredictor + TinyPixelDecoder。

> 注：标号 "W36" 在本候选表里同时出现两次——#11（项目 canonical W36 =
> Reward model 接口 / G9）与 #15（pre-Atlas NPU parity 契约 / G11），
> 实施时按 spec 文件名 `2026-08-06-w36-npu-worker-parity-contract.md`
> 区分，不要把两件事混在一起做。

When the user picks one, this plan becomes the working plan; the others
move to the bottom of the queue.

## Cross-reference

- Phase-3 closure: `docs/dev/next-steps-w27-v2.md` (all 4 candidates done).
- Roadmap: `docs/roadmap/2026-08-03-jepa2-ascend-gap-analysis-roadmap.md`
- W28 编排设计: `docs/superpowers/specs/2026-08-06-w28-train-inference-orchestration-design.md`
- W35 RL 框架 ADR: `docs/superpowers/specs/2026-08-04-w35-rl-framework-adr.md`
- W25 full-FT + SFT schema: `docs/superpowers/specs/2026-08-06-w25-full-finetune-sft-schema.md`
- **W36 NPU parity contract (G11 契约层)**: `docs/superpowers/specs/2026-08-06-w36-npu-worker-parity-contract.md`
- **W30 env-selection ADR (G6 前置)**: `docs/superpowers/specs/2026-08-06-w30-env-selection-adr.md`
- **W31 EnvPool + FrankaKitchen 向量化封装 (G6)**: `worldfoundry_rl/env_pool.py`（Env/EnvPool Protocol + SyncEnvPool 同步 vec-env，Gymnasium autoreset_mode="on" 语义） + `worldfoundry_rl/env_franka_kitchen.py`（FrankaKitchenEnv 单 env 封装，dm_control+mujoco 延迟导入） + `worldfoundry_rl/tests/test_env_pool.py`（16 个单元测试全绿） + `bench_results/w31/vec_env_throughput.json`（FakeEnv CPU 基线；48/48 rl 套件全绿）
- **W35.2 版本矩阵核对 (CPU-side, G6/G7/G14 触发切 verl)**: `docs/superpowers/specs/2026-08-06-w35-2-version-matrix-check.md`
- **W40 frame-gen route ADR (G13 前置)**: `docs/superpowers/specs/2026-08-06-w40-frame-gen-adr.md`
- **W36 canonical (Reward 接口 / G9)**: `worldfoundry_rl/reward.py`（RewardFn Protocol
  + ensure_reward_fn + compose）+ `worldfoundry_rl/reward_impl.py`（FrankaKitchen
  manipulation reward + LatentDistanceReward intrinsic）+ `worldfoundry_rl/tests/test_reward.py`
  （25 个单元测试全绿；`pytest worldfoundry_rl/tests/` 32/32）。
- **W34 自研 PPO (G7)**: `worldfoundry_rl/ppo.py`（PPOActor 对角高斯策略 +
  可学习 log_std（clamp 到 [-5, 2]）+ compute_gae + RolloutBuffer + PPOConfig +
  PPO clipped-surrogate 训练器）+ `worldfoundry_rl/tests/test_ppo_smoke.py`
  （34 个单元测试全绿）+ `bench_results/w34/ppo_reach_curve.json`
  （合成 reach 任务 6 seed learning curve：约 -21 → -1~-5，全部 seed 收敛；
  这是 W35 框架 bake-off 的正确性基线，**不是**真实环境或 Atlas 基线）。
  `pytest worldfoundry_rl/tests/` 82/82；全仓 594 passed / 1 skipped。
- **W20 CPU prep (G1 物理前提)**: `worldfoundry_train/device.py`（`resolve_device`
  + `to_device` + `is_npu_available` proxy；显式不创建重复的
  `npu_env.py`，按 G8 重用 `worldfoundry_common.npu_env.is_npu_available()`）
  + `worldfoundry_train/trainer.py` 注入 `.to(device)` 迁移点
  （`strategy.wrap(peft_model)` 之后立刻 `model = to_device(model, resolve_device())`，
  JEPA 时也迁 predictor；print `[TRAIN] device=... npu_available=...` 一行
  让迁移可见）+ `worldfoundry_train/tests/test_npu_trainer.py`（12 个 mock 测试
  含端到端 `run_training` spy 验证迁移点真在 `strategy.wrap` 之后发生）。
  CPU 路径字节级不变——`pytest worldfoundry_train/tests/` 392/392；全仓
  606 passed / 1 skipped，无回归。剩余的 Atlas 上机步骤
  （`scripts/w20_atlas_smoke.sh` 5/100 step smoke +
  `bench_results/atlas_baseline/single_card.json`）等硬件窗口。
- Phase-3 final HEAD: `c9e7197` (W32)
