# W28 — 训推一体编排设计文档（同集群训练/推理混部 + 权重热更新 + vRAM 配额）

**作者**：chad（设计）+ Claude（执行记录）
**日期**：2026-08-06
**状态**：🟡 设计评审稿（Proposed）；W28 v1 不写代码，仅评审通过 + ADR 决策记录
**关联**：
- 路线图：`docs/roadmap/2026-08-03-jepa2-ascend-gap-analysis-roadmap.md` Phase-3 W28 / Phase-4 W37
- 上一份设计：`docs/superpowers/specs/2026-08-03-w22-jepa-predictor-ema-design.md`（JEPA predictor + EMA target encoder）
- RL 框架 ADR：`docs/superpowers/specs/2026-08-04-w35-rl-framework-adr.md`（RLinf 首选，verl 备选；本设计需在 RLinf/verl 各自训推编排之上做一层适配）
- 总体架构：`docs/superpowers/specs/2026-07-27-ascend-world-model-design.md:198-201,307-336`（RL ActorPool/EnvPool/LearnerPool 架构图）
- 上游已落地（CPU 闭环）：W22（`0d8910f` 前序 commit `e8150be`）、W24（`dcf4adc`）、W25（`cb112fa`）、W26（`338ded8`）、W27（`796fb50`）、W27 v2（`0d8910f`）、W33（`28e8a6b`）

---

## 0. TL;DR

W28 解决**G14（训推一体资源编排）**——具体三件事：

1. **拓扑**：在 Atlas 910B×8 + 仿真节点（GPU/CPU）的混合集群上，如何把"训练占用 NPU"和"推理占用 NPU"互不打架地排进同一个 vRAM 池。
2. **权重热更新**：训练侧每 N 步产出新 checkpoint，推理侧如何在**不中断在线服务**的前提下换权重，且保证正在处理的请求不读到撕裂状态。
3. **vRAM 配额**：单卡同时被训练（LoRA+grad optimizer state+activation）和推理（encoder bf16 + temporal/AC predictor weights + KV cache）双向抢内存时的硬性分配比例与 OOM 兜底语义。

**决策（见 §11 ADR 决策日志）：** **采用 B 方案"调度器中介的共享 NPU 池 + 双 manifest 轮询权重热更新 + 静态 60/40 vRAM 配额 + 抢占式 OOM 回退"** 为基线方案，理由详见 §4 / §5 / §6。

---

## 1. 背景与问题陈述

### 1.1 已落地的上游能力（W22 → W33）

到本设计稿起草为止，CPU 路径上训练与推理的对接已成型，但**还没有在同一进程/集群中让两者共同跑起来**的工程实现。具体已交付：

| 能力 | 实现位置 | 状态 |
|---|---|---|
| JEPA predictor + EMA target encoder | `worldfoundry_train/model/{predictor,ema}.py`（W22） | ✅ CPU 单测全过 |
| TemporalPredictor（warm-start from W22 trunk） | `worldfoundry_train/model/temporal_predictor.py`（W24） | ✅ |
| ActionConditionedPredictor（AC + JEPAPredictor 双模式） | `worldfoundry_train/model/action_predictor.py`（W27） | ✅ |
| W22/W27 v2 inference-side loader | `worldfoundry_train/model/{jepa,action_jepa}_inference.py`（W24 / W27 v2） | ✅ |
| Per-adapter `PredictBatcher`（W25 动态批处理） | `worldfoundry_inf/predict_batcher.py` | ✅ |
| AdapterRouter + HMAC FastAPI | `worldfoundry_inf/scheduler.py` + `api/{encode,predict,plan}.py` | ✅ |
| `VJEPA2Worker` 持有 `bundle` (W22) + `action_bundle` (W27 v2) | `worldfoundry_inf/vjepa2_worker.py:101-115` | ✅ |
| `/v1/world/predict` 双模 → 三模（masked/temporal/action，W33） | `worldfoundry_inf/api/predict.py` | ✅ |
| `/v1/world/plan` action-conditioned（W33） | `worldfoundry_inf/api/plan.py` | ✅ |

**结论**：CPU 闭环上的算法与 API 已就绪，**剩下的是把它们放到共享的 NPU 池中并被调度起来**。

### 1.2 现状缺口

读码确认 `worldfoundry_inf/server.py:43-50` 与 `worldfoundry_train/trainer.py:129-144` 的现状：

- 推理进程启动时 `VJEPA2Worker.__init__(...)` 一次性 in-place 把 `online_encoder` 加载完成并冻结，**之后再无重新加载入口**；要换权重必须重启 worker 进程 → 在线服务中断。
- 训练侧 `worldfoundry_train/checkpoint.py:save_checkpoint` 每 N 步原子写 `.pt`（tmp + rename），但**没有"通知推理端"的机制**；推理端只能通过 `--predictor-ckpt` / `--action-predictor-ckpt` CLI 在启动时绑定。
- `AdapterRouter._workers: dict[str, object]` 是进程内 dict，**没有外部 registry / 共享存储**——多个推理副本之间无法共享 adapter 注册表，也无法感知集群级 adapter 变更。
- `PredictBatcher` 是 in-process 的 thread-safe queue（`predict_batcher.py:55-69`），不会跨进程/节点工作。
- 全库搜索 `vRAM / memory budget / OOM quota` 零命中——**没有任何 vRAM 分配与抢占策略的设计**。

### 1.3 问题的硬性约束

| # | 约束 | 来源 |
|---|---|---|
| C1 | 训练/推理必须运行于昇腾 NPU（Atlas 800 A2 / 910B×8；CANN 8.5.1 + torch_npu 2.9.0 + vllm-ascend 0.18.0） | `worldfoundry_common/npu_env.py:15-21` |
| C2 | 推理 API 必须保证 HMAC 鉴权 + 不中断在线请求（FastAPI 既有契约） | `worldfoundry_inf/server.py:36` `_check_auth`；spec §4.3 |
| C3 | LoRA 微调是常规形态（少量可训练参数 + 冻结 backbone），全参微调（G5）规模更大 | `worldfoundry_train/model/lora_wrapper.py:17-20`；W25-roadmap |
| C4 | 仿真环境大概率仍是 GPU/CPU（IsaacLab/ManiSkill GPU/Genesis 物理依赖 NVIDIA），形成"GPU 仿真 + NPU 训练"异构混部 | ADR-W35 §3.B |
| C5 | 推理端权重是 W22 JEPA ckpt 或 W27 v2 AC ckpt（FSDP/LoRA/full 均有），且推理端是冻结参数 | W22/W24/W27 v2 ckpt fingerprint |
| C6 | W37（RLinf 集成）需要训练栈与推理服务共存，编排接口必须先于集成实现落地 | 路线图 W37 + ADR-W35 §4 |

---

## 2. 设计目标与非目标

### 2.1 目标

| # | 目标 | 验收标准 |
|---|------|----------|
| G1 | 选型基线拓扑方案（split cluster / shared NPU pool / scheduler-mediated），含理由 | §4 决策矩阵 + §11 ADR 决策日志 |
| G2 | 权重热更新机制：训练侧 checkpoint 完成 → 推理侧在**不重启、不中断正在处理请求**的前提下换权重 | §5 流程图 + 时序图；明确"在飞请求 + 新请求"的衔接语义 |
| G3 | vRAM 配额策略：单卡训练 + 推理共存时的硬性分配 + OOM 兜底 | §6 数值表（per-card）+ 抢占式回退流程 |
| G4 | 失败模式显式枚举：stale 权重、checkpoint save 期间训练需独占 NPU、推理侧权重加载失败回滚、OOM 抢占 | §7 至少 5 种 failure mode + 检测 + 缓解 |
| G5 | 与 W22/W24/W25/W26/W27/W33 的集成点显式枚举 | §8 至少 6 条集成点（每条指向具体文件/方法） |
| G6 | 列出 W37 实施前置条件 | §9 至少 5 条 prerequisite（含 G1 决策依赖项） |

### 2.2 非目标（明确不做）

- **不实现**：W28 v1 是设计文档，**不写代码**。所有 NPU 拓扑实现、manifest server、vRAM 配额调度器都属 W37+ 实施范围。
- **不替换 vLLM-Ascend 插件**：`worldfoundry_inf/_vllm_ascend_plugin.py` 与 `vllm_ascend_worker.py` 保持现状；本设计与既有 vllm-ascend worker 兼容（推理 worker 只是被动接收 manifest 通知）。
- **不引入新依赖栈**：W28 不指定 RLinf/verl，编排层面对两者的接口差异通过 W37 集成设计吸收。
- **不做跨集群联邦**：本期限定单集群（同 Atlas 域内）；跨集群 sync 留作 W38+ 选项。
- **不重做 HMAC 契约**：C2 不变；权重热更新不绕过 HMAC，请求身份保持调用者绑定。
- **不假设 910B×8 永远满配**：拓扑选项必须可在 4 卡、2 卡、1 卡配置下退化；不允许"必须 8 卡"的硬依赖。

---

## 3. 总体架构

```
                   ┌───────────────────────────────────────────────────────────────┐
                   │                  Atlas 800 A2 / 910B×N Cluster                   │
                   │                                                                │
                   │   ┌────────────┐         ┌─────────────┐      ┌────────────┐  │
                   │   │ Trainer    │         │ Scheduler   │      │ Inference  │  │
                   │   │ (NPU rank  │ ─save──►│ (manifest   │ ─pub─│ Pool       │  │
                   │   │  0..N-1)   │         │  server)    │      │ (M 个副本, │  │
                   │   │            │ ◄─ack── │             │      │  K×NPU)    │  │
                   │   └────────────┘         └─────────────┘      └─────┬──────┘  │
                   │         │                                              │         │
                   │         │ NPU device lock (per-card)                   │         │
                   │         ▼                                              ▼         │
                   │   ┌────────────┐                                   ┌─────────┐ │
                   │   │  NPU Card  │  ◄── 60% training / 40% inference quota ──►│ │ │
                   │   │  i (910B)  │       (static, see §6)              │         │ │
                   │   └────────────┘                                   └─────────┘ │
                   │                                                                │
                   │   ┌──────────────────────────────────────────────────────────┐  │
                   │   │ Simulator Pool (异构, C4)                               │  │
                   │   │  GPU/CPU IsaacLab / ManiSkill / MuJoCo (W30/W31 选型)   │  │
                   │   └──────────────────────────────────────────────────────────┘  │
                   └───────────────────────────────────────────────────────────────┘
```

**关键设计原则**：

1. **Scheduler 中介**：训练与推理不直接通信；所有"权重已就绪"信号通过 manifest server 流转。
2. **Manifest 即真相**：scheduler 维护一个不可变的 manifest 版本链（按 commit hash + step monotonic），推理端按版本号轮询（默认 5s）或主动订阅（push 通道）。
3. **Per-card quota 而非 per-process**：vRAM 配额按物理卡分配；同一卡上"训练 + 推理"是允许的（§6 详述），但合计超出配额时由 scheduler 抢占。
4. **推理副本无状态**：除 adapter 字典（`AdapterRouter`）外，推理 worker 不持久化任何状态；切换副本/重启不影响 manifest 订阅。

---

## 4. 拓扑方案对比

### 4.1 三个候选

**A. 拆分集群（split cluster）**：训练组与推理组在物理上分开（甚至跨 Atlas 域）。训练 checkpoint 通过共享存储推送给推理集群，推理集群独立 load。

**B. 调度器中介的共享 NPU 池（scheduler-mediated shared pool）**：所有 NPU 属一个资源池；scheduler（cgroup + 自研 registry）按 quota 把卡分给训练任务与推理副本；权重走 scheduler 转发的 manifest。

**C. 共享 NPU 物理混部（shared-NPU mixed）**：训练与推理**在同一进程内**通过 vLLM-Ascend 的 colocate 模式共存（如同卡同时跑 FSDP trainer + vLLM engine）。无独立 scheduler，依赖 vLLM 自身的 GPU memory pool。

### 4.2 决策矩阵

| 维度 | A. split cluster | B. scheduler pool | C. shared-NPU mixed |
|---|---|---|---|
| 隔离性 | 🟢 完全物理隔离 | 🟡 配额级隔离 | 🔴 同卡共存，依赖软件层 |
| 资源利用率 | 🔴 推理集群常空闲；训练集群常空闲（负载错峰不均） | 🟢 共享池自动错峰 | 🟡 同卡约束下利用率高 |
| 权重延迟 | 🔴 跨集群推送需分钟级（共享存储+网络） | 🟡 池内推送秒级（manifest 轮询 ≤5s） | 🟢 同进程纳秒级 |
| 实施成本 | 🟢 现有 `--predictor-ckpt` 启动参数几乎不变 | 🟡 需自研/集成 scheduler（约 1–2 sprint） | 🔴 需深度改 vLLM-Ascend 插件，跨厂商代码路径风险 |
| RLinf/verl 兼容 | 🟡 RLinf/verl 都假设同集群调度，需在 W37 做跨集群适配 | 🟢 RLinf/verl 都是 Ray/SGLang 调度模型，原生兼容 | 🟡 RLinf 文档明列支持 colocate，但昇腾路径未公开验证 |
| 4 卡/2 卡/1 卡退化 | 🟢 任意切分即可 | 🟢 配额按比例缩放 | 🔴 单卡时训练+推理相互饿死 |
| 故障爆炸半径 | 🟢 推理集群崩溃不影响训练 | 🟡 scheduler 单点故障需做 HA | 🔴 同进程崩溃两边都倒 |
| 与 W22/W33 API 兼容 | 🟢 零改动 | 🟢 零改动（manifest 是新增 channel） | 🟡 需在 worker init 加 lock |

### 4.3 推荐

**B 方案**——调度器中介的共享 NPU 池（scheduler-mediated shared pool）。理由：

1. 资源利用率显著高于 A（同 Atlas 池内错峰，跨集群跨域网络代价不必支付）。
2. 实施成本可控：scheduler 是单一服务，可独立做 HA；与 vLLM-Ascend 插件无侵入。
3. 与 RLinf/verl 的 Ray 调度模型天然兼容，W37 集成无需改框架侧。
4. 单卡配置下仍可工作（训练占 60%、推理占 40%，剩余 0% 给 OOM 缓冲；详见 §6）。
5. A 方案保留为集群规模扩大后的回退路径（路线图 §5 风险表注 1："Atlas 硬件窗口不可控"——若需跨域拉集群，A 是兜底）。

### 4.4 B 方案的"调度器"实现形态（决策范围外但需明确）

scheduler 的实现不属 W28 范围（属 W37），但 W28 决策时需对其形态有预设以验证可行性。候选：

- **Ray cluster**（RLinf/verl 都用 Ray）；复用其 PlacementGroup + ResourceManager；W37 集成时吸收
- **Kubernetes + Volcano/PyTorch Operator**：与 vLLM-Ascend K8s 部署兼容；优势是 HA 与跨域
- **自研轻量 scheduler**：贴合项目需求但增加维护成本

W37 应**优先选 Ray**（理由：W35 ADR 已确定 RLinf/verl 都是 Ray 框架，复用其调度层即可；K8s 路径作为 Phase-5/集群规模扩大后的备选）。

---

## 5. 权重热更新机制

### 5.1 设计原则

1. **零中断**：正在处理的 HTTP 请求必须在同一 worker 进程内得到完整响应，不被权重切换打断。
2. **版本单调**：manifest 版本号单调递增（step + commit hash），推理端拒绝任何 ≤current 的更新（除非显式 force-rollback）。
3. **原子加载**：单 adapter 内的 `online_encoder` + `target_encoder` + predictor weights 必须在**同一原子事务内**完成切换；切换中途读到不一致状态的请求应被新事务完整丢弃并重试。
4. **可回滚**：scheduler 保留最近 K=10 个 manifest 版本；推理侧权重加载失败时回退上一版本。

### 5.2 推送机制（决策）：双通道轮询（polling）+ manifest 版本比对

**轮询（默认）**：推理 worker 每 5s（可配置 `WORLDFOUNDRY_MANIFEST_POLL_INTERVAL_MS`）向 scheduler 发起 `GET /v1/manifests?adapter=<name>`，scheduler 返回 `[{adapter, version, ckpt_url, manifest_url}, ...]`。worker 比对本地 `current_version[adapter]` 与返回中的 `version`：若更大则下载 ckpt 并走 §5.3 的"在飞请求衔接"流程。

**Push 通道（可选）**：scheduler 维护 WebSocket 长连接 (`/v1/manifests/ws`)，新 manifest 产生时主动推送；worker 在 5s 窗口内未收到 push 则降级到 polling（健康检查）。

**Push vs Polling 取舍**：纯 push 在弱网/防火墙环境易断连；纯 polling 延迟粒度受轮询间隔限制。**双通道 fallback** 取两者之长：push 是低延迟路径，polling 是可靠基线。

### 5.3 在飞请求衔接流程

```python
# worldfoundry_inf/worker.py (W37+) — 伪代码，W28 不实现

async def on_new_manifest(adapter: str, new_ckpt_url: str, new_version: int):
    # Step 1: snapshot 当前 bundle 引用
    old_bundle = self.bundles[adapter]
    new_bundle = await load_jepa_inference_model(new_ckpt_url, ...)
    new_bundle.eval()  # 冻结

    # Step 2: 在飞请求数 ≥1 则等待 ≤ grace_period（默认 30s）
    in_flight = await self.inflight_count(adapter)
    deadline = time.monotonic() + 30.0
    while in_flight > 0 and time.monotonic() < deadline:
        await asyncio.sleep(0.05)
        in_flight = await self.inflight_count(adapter)
    # 超出 grace_period 的请求进入第 3 步的 fast-fail 通道

    # Step 3: 原子替换 AdapterRouter 中的引用
    self.router._workers[adapter] = new_worker_using(new_bundle)
    old_bundle = None  # GC

    # Step 4: 更新 manifest 版本
    self.manifest_versions[adapter] = new_version
```

**关键设计**：

- AdapterRouter 持有的是**新 bundle 的 worker 对象**而非 raw tensor（避免推理中读取已 GC 的 `online_encoder` 引用）。
- in-flight 请求的 worker 闭包仍持有 `old_bundle` 引用，所以 `old_bundle` 在 GC 引用计数归零前不会被释放。
- grace_period（默认 30s）后仍 stuck 的请求由 FastAPI middleware 识别为"权重切换期"并返回 503 + `Retry-After: 1`；客户端按 HTTP 标准重试（spec §4.3 既有 503 语义已为此预留）。

### 5.4 Manifest Schema

```yaml
# scheduler 发布的不可变 manifest（W37+ 定义格式，W28 仅为契约预览）
apiVersion: worldfoundry/v1
kind: AdapterManifest
metadata:
  adapter: "default"
  version: 42              # monotonic，commit hash 派生
  step: 12345              # 训练 step
  timestamp: 2026-08-06T12:34:56Z
  parent_version: 41       # 回滚链
spec:
  predictor_ckpt_url: "s3://worldfoundry-ckpts/w22/step_12345.pt"
  action_predictor_ckpt_url: null    # null = 推理端不切换 AC bundle
  lora_signature: {"r": 8, "alpha": 16}    # null = 全参
  jepa_signature:
    encoder_dim: 1024
    pred_dim: 384
    pred_depth: 12
    pred_heads: 12
  signature_sha256: "abc123..."   # 训练侧 save_checkpoint 时写入，推理侧 load 前校验
```

推理端在 `load_jepa_inference_model` 完成后比对 `signature_sha256` 与 ckpt 内容 hash；不一致则拒绝加载并上报 scheduler（避免 W24/W27 v2 已记录的 ckpt 解析缺陷被默默接受）。

---

## 6. vRAM 配额策略

### 6.1 静态配额基线

**默认（910B 单卡，bf16）**：

| 用途 | 占比 | 估算内存 | 说明 |
|---|---|---|---|
| 推理 (encoder bf16 + target bf16 + predictor + action_predictor) | **40%** | ~9 GB | ViT-L encoder ≈ 2.4 GB × 2（online+target） ≈ 4.8 GB；temporal_predictor ≈ 1.2 GB；action_predictor ≈ 1.2 GB；KV cache + activations ≈ 1.5 GB |
| 训练 (LoRA adapters + optimizer state + activation ckpt buffer) | **60%** | ~13 GB | LoRA +32 adapter = ~10 MB；AdamW state ≈ 80 MB（bf16 × 2 + fp32 master × 2）；activation checkpointing 重计算峰值 ≈ 8 GB |
| OOM 缓冲 / KV cache spike | **0%（共享池）** | 剩余 ~3 GB | 见 §6.3 OOM 兜底 |

> 数字取自 W22 设计文档 §1.1 + W22 design review addendum §1.2（FSDP gather 瞬态 ≈ 2.4GB/rank/EMA step）；本表为 W28 决策草案，正式测量留待 W20/W21 Atlas 上机。

### 6.2 配额调整规则

- **全参微调模式**（W25-roadmap）：训练份额由 60% 上调至 75%，推理下调至 25%；scheduler 检测到 manifest 中 `lora_signature == null` 时自动应用。
- **AC + temporal 双 bundle 共存**：推理份额不变（40%），但 scheduler 监控 `predict_action_conditioned` 调用占比；若 >70% 流量走 action 路径，scheduler 可提示运维上调 action bundle 预算（不自动切换，避免服务抖动）。
- **RL 训练模式（W34+ PPO / RLinf rollout）**：训练份额含 rollout collector 的临时 KV cache，可上调至 70%；推理份额下调至 30%。W37 集成时由 RLinf Ray PlacementGroup 接管分配，本表仅作 FSDP-only 形态的参考。

### 6.3 OOM 兜底语义

```python
# worldfoundry_inf/worker.py (W37+) — 伪代码
# 触发条件：torch.npu.OutOfMemoryError caught at forward()

async def handle_oom(adapter: str):
    # 1) 通知 scheduler 抢占：发 POST /v1/quota/evict
    await scheduler.evict(adapter, reason="oom", token=self.auth_token)
    # 2) 退避 1s 后重试 1 次（fast path 缓解瞬时波动）
    try:
        return await self._forward_with_current_bundle(...)
    except OutOfMemoryError:
        pass
    # 3) 仍 OOM：返回 503 + Retry-After，触发 manifest 回滚
    await self.rollback_to_parent_version(adapter)
    raise HTTPException(503, "inference OOM; rolled back to previous version")
```

**关键语义**：

- OOM 是**推理端主动驱逐**而非 scheduler 单方面拉黑；驱逐后的 adapter 进入 `quarantine` 状态（默认 60s），期间 scheduler 不再向其路由请求。
- 重试 1 次是 fast path；超过则触发 manifest 回滚（§5.4 parent_version 字段）。
- 全参微调期间的 OOM 阈值比 LoRA 模式更敏感（vRAM 占用更大），scheduler 在 manifest 中标注 `high_memory=true` 时主动预留更多缓冲。

---

## 7. 失败模式（Failure Modes）

| # | 失败模式 | 检测信号 | 缓解策略 | 责任方 |
|---|---|---|---|---|
| F1 | **推理 worker 持有 stale 权重**（manifest 已更新但 worker 未拉到） | scheduler 推送后 30s 仍未见 worker ACK | scheduler 主动 webhook 通知 worker；超过 grace_period 标记 worker `stale=true` | scheduler + worker |
| F2 | **训练 save_checkpoint 期间需独占 NPU**（避免 trainer 与 inference 同时写同一文件） | scheduler 收到 `trainer.lock_request` 信号 | 训练侧进入 `save_window`（默认 30s，期间冻结 EMA 更新）；推理侧此期间只读，不切换 manifest | scheduler 协调 |
| F3 | **推理侧权重加载失败**（ckpt 损坏 / signature 不匹配 / shape 不匹配） | `load_jepa_inference_model` 抛 ValueError / RuntimeError | worker 自动回滚至 `parent_version`；上报 scheduler 标记该 manifest 为 `bad` | worker |
| F4 | **OOM 抢占**（同卡训练 + 推理合计超 vRAM 配额） | `torch.npu.OutOfMemoryError` | 推理端驱逐（§6.3）；scheduler 把训练任务排队到空闲卡 | worker + scheduler |
| F5 | **scheduler 单点故障** | scheduler 心跳丢失 > 30s | 推理端降级到本地缓存的 manifest（最新已知版本），不再接受更新；训练侧继续训练但 manifest 写入降级到本地待发队列 | scheduler HA |
| F6 | **LoRA adapter 热切换期间 online/target 不一致** | manifest 中 lora_signature 变化但 predictor weights 未跟随 | scheduler 拒绝发布此类 manifest（语义不合法校验）；worker 加载时若发现不一致 raise | scheduler |
| F7 | **跨 manifest 切换时 in-flight 请求超时** | grace_period > 30s 仍有未完成请求 | FastAPI middleware 发送 503 + Retry-After: 1；客户端标准重试 | worker + client |
| F8 | **集群缩减（N 卡 → M<N 卡）期间推理副本被强制驱逐** | scheduler 下发 `evict_immediate` 信号 | 推理 worker 进入 graceful shutdown（关闭前等 ≤5s in-flight 请求）；evict 副本由剩余副本接管 | scheduler |

---

## 8. 与已落地能力的集成点

W28 决策的实施面（W37+）需在以下既有代码点上做**最小侵入**扩展：

| # | 集成点 | 现有位置 | 扩展方向 |
|---|---|---|---|
| I1 | `VJEPA2Worker.__init__` 在启动时绑定 ckpt | `worldfoundry_inf/vjepa2_worker.py:79-115` | 增加 `manifest_watchdog` 后台 task 订阅 scheduler；保持现有 `--predictor-ckpt` / `--action-predictor-ckpt` 作为启动默认值 |
| I2 | `AdapterRouter._workers` 是 in-process dict | `worldfoundry_inf/scheduler.py:13-33` | 增加 `swap(adapter_name, new_worker)` 原子方法；现有 `register()` 不变 |
| I3 | `PredictBatcher` 是 in-process queue | `worldfoundry_inf/predict_batcher.py:36-69` | 增加 `drain(adapter_name)` 方法：权重切换时排空该 adapter 的 in-flight 请求后再返回 |
| I4 | `load_jepa_inference_model` 返回 `JEPInferenceBundle` | `worldfoundry_train/model/jepa_inference.py:42-65` + `worldfoundry_train/model/action_jepa_inference.py` | 已支持 in-place load（W22/W27 v2 已就位）；增加 `signature_sha256` 校验（§5.4 manifest 字段） |
| I5 | `worldfoundry_train/checkpoint.save_checkpoint` 原子写 `.pt` | `worldfoundry_train/checkpoint.py` | 写完后调用 `scheduler.notify_new_manifest(...)`（W37+）；现有 atomic write 行为不变 |
| I6 | FastAPI HMAC `_check_auth` | `worldfoundry_inf/api/{encode,predict,plan}.py` | 不变；503 语义已为 F7 预留 |
| I7 | `vllm_ascend_worker.py:20-22` 的 `NotImplementedError` 占位 | `worldfoundry_inf/vllm_ascend_worker.py` | 增加 `manifest_watchdog` 镜像；NPU 路径上线后此 stub 转正 |
| I8 | `worldfoundry_train/parallel/fsdp.py:42-180` FSDP 路径 | 已落地 | W21 集成时同时验证 `save_window` 期间 FSDP all-reduce 不被抢占 |
| I9 | `worldfoundry_inf/compile/quantize_backends/msmodelslim.py` 量化加载 | W11 落地 | manifest 中 `quant_signature` 字段决定是否走 msmodelslim 加载路径；保持 W11 既有加载器契约 |

**最小侵入原则**：W28 决策的实施**优先扩展现有 entry-point**（I1/I2/I3/I4/I5），不引入新并行框架；NPU 拓扑实施（scheduler 进程）作为 W37 的独立子模块。

---

## 9. W37 实施前置条件（Prerequisites）

W37（训推一体集成 / RLinf 对接）开始前必须满足：

1. ✅ **W28 本设计评审通过**（本 ADR 被 chad approve；§11 决策日志填写完整）
2. 🟡 **W22.7 对照实验结果可读**（真 JEPA 目标与代理 loss 的对照差异有结论；影响 W37 scheduler 对"权重质量"的判定策略——若差异过小则 trainer 推断该 manifest 为 `low_quality` 并延迟发布）
3. 🟡 **W20/W21 Atlas 上机 smoke 通过**（FSDP 端到端训练 + checkpoint save/load 上机验证；不依赖硬件的设计无法在 W37 验证 vRAM 数字）
4. 🟡 **RLinf 验证门 W35.1–W35.4 通过**（影响 W37 是否需要切 verl 备选；ADR-W35 §6）
5. 🟡 **集群 NPU 卡数确定**（影响默认配额：8 卡用 60/40 拆分，4 卡用 50/50，1 卡下不建议混部——W37 应在 1 卡配置下禁用训练任务）
6. 🟡 **FastAPI 503 + Retry-After 语义在 spec §4.3 中显式化**（W28 §5.3 / §6.3 复用该语义；spec 现有 503 文本需 chad 评审）
7. 🟡 **scheduler 实现形态决策**（§4.4 候选三选一：Ray / K8s / 自研；W37 sprint 0 决策）

---

## 10. 风险与开放问题

### 10.1 风险

| # | 风险 | 严重度 | 缓解 |
|---|---|---|---|
| R1 | Atlas 910B 实测 vRAM 数字与 §6.1 估算差距大（特别是 W22 design review addendum §1.2 已记录的 FSDP gather 瞬态 2.4GB） | 🟠 | W20/W21 上机后实测调整；§6.1 表头注明"估算" |
| R2 | scheduler 单点故障（F5）影响范围大 | 🟠 | scheduler 必须 HA；W37 实施时优先选 Ray（其调度器自身高可用） |
| R3 | LoRA adapter 热切换期间 online/target 不一致（F6） | 🟠 | scheduler manifest 发布前必须做语义合法性校验（I9）；manifest 字段禁止部分更新 |
| R4 | 1 卡配置下 60/40 拆分后两边都跑不动 | 🟡 | W37 在 1 卡配置下禁用训练任务，仅保留推理（spec §4.7 "no silent default" 原则延伸） |
| R5 | 跨 manifest 切换的 grace_period（30s）可能太长，影响推理 SLA | 🟡 | W37 实施时实测调整；可暴露 `WORLDFOUNDRY_WEIGHT_SWITCH_GRACE_S` 环境变量 |

### 10.2 开放问题（需 chad 在评审中回答）

1. **是否同意 B 方案为基线**（§4.3 推荐）？
2. **scheduler 实现是否优先 Ray**（§4.4 候选三选一）？
3. **grace_period 默认值是否 30s**（§5.3）？
4. **manifest 版本保留 K=10 是否足够**（§5.1）？还是需要更长链？
5. **W37 是否在 1 卡配置下禁用训练**（§R4）？
6. **全参微调时是否自动调整 vRAM 配额到 75/25**（§6.2）？还是仅给出建议，由运维手动调整？
7. **scheduler 是否需要支持训练侧"预约 save window"**（F2）？还是 trainer 自适应？
8. **跨集群场景（A 方案回退）何时启用**？是 Atlas 卡数 ≥ 16 还是 ≥ 32？

---

## 11. ADR 决策日志

> ADR-W28：训推一体编排选型（拓扑 + 权重热更新 + vRAM 配额）

| # | 决策项 | 选定方案 | 替代方案 | 理由（简） |
|---|---|---|---|---|
| D1 | 拓扑基线 | **B. scheduler-mediated shared pool** | A. split cluster / C. shared-NPU mixed | §4.2 矩阵（资源利用率、RLinf/verl 兼容、单卡退化）综合最优 |
| D2 | scheduler 实现（实施层，W37 决） | **Ray cluster**（倾向，非终决） | K8s+Volcano / 自研 | W35 ADR 已选 RLinf/verl（均为 Ray）；W37 复用其调度层 |
| D3 | 权重推送机制 | **双通道：push + polling fallback** | 纯 push / 纯 polling | 弱网友好性 + 低延迟兼顾；§5.2 |
| D4 | 轮询间隔 | **5s（可配置 `WORLDFOUNDRY_MANIFEST_POLL_INTERVAL_MS`）** | 1s / 30s | 训练 checkpoint 频率通常 ≥30s 一次，5s 粒度足够；运维可调 |
| D5 | manifest 版本保留 | **K=10**（scheduler 端） | K=3 / K=30 | 平衡回滚能力与存储；W37 实施时再调 |
| D6 | 在飞请求 grace_period | **30s**（可配置） | 10s / 60s | 推理 P99 < 1s；30s 足以覆盖；超时进 F7 |
| D7 | vRAM 默认配额 | **训练 60% / 推理 40%** | 50/50 / 70/30 | 训练侧（LoRA + AdamW + activation ckpt）天然比推理侧（冻结 bf16）需要更多 vRAM；§6.1 表 |
| D8 | OOM 兜底 | **推理端主动驱逐 + manifest 回滚** | scheduler 单方面拉黑 | 推理端持有完整上下文，回滚原子性更易保证；§6.3 |
| D9 | 全参微调配额 | **自动调整 75/25**（基于 `lora_signature == null`） | 仅建议，不自动 | 全参微调是 W25-roadmap 主线；自动避免手动运维出错 |
| D10 | 1 卡配置策略 | **W37 禁用训练任务**（仅推理） | 1 卡下 50/50 强跑 | R4：1 卡下两边都跑不动 |
| D11 | scheduler HA 策略 | **Ray 调度器自带 HA**（D2 派生） | 自建主备 | 减少自研成本；W37 实施时验证 |
| D12 | manifest schema 版本 | **worldfoundry/v1（本文 §5.4 草案）** | 不指定 | W37 实施前冻结；chad 评审后写入 `docs/superpowers/specs/manifest-schema.md` |

**状态**：🟡 Proposed（2026-08-06 起草，待 chad 评审 + open questions 答复）
**下一步**：chad 评审 §10.2 开放问题 → D1–D12 任一项否决即重开评审 → 通过则 freeze 为 Accepted 状态并启动 W37 sprint 0

---

## 附录 A：与现有设计文档的交叉引用

| 文档 | 关系 | 章节引用 |
|---|---|---|
| `docs/roadmap/2026-08-03-jepa2-ascend-gap-analysis-roadmap.md` | 上游路线图 | §3 G14；§4 Phase-3 W28 / Phase-4 W37 |
| `docs/superpowers/specs/2026-08-03-w22-jepa-predictor-ema-design.md` | 算法层 | §3 三模块架构；§5.3 EMA 更新语义（训推权重同步的基础） |
| `docs/superpowers/specs/2026-08-05-w22-design-review-addendum.md` | W22 post-impl 评审 | §1.2 FSDP gather 瞬态（影响 §6.1 vRAM 数字） |
| `docs/superpowers/specs/2026-08-04-w35-rl-framework-adr.md` | RL 框架选型 | §3.B（RLinf 异构混部）/ §4（训推编排影响）/ §6（验证门） |
| `docs/superpowers/specs/2026-07-27-ascend-world-model-design.md` | 总体架构图 | §198-201、§307-336（ActorPool/EnvPool/LearnerPool 编排参考） |
| `docs/superpowers/specs/2026-08-04-w35-3-rlinf-custom-policy-spike-plan.md` | RLinf 接入 spike | §3（VJEPA2Policy 数据通路，与 W28 权重同步通道同源） |
| `docs/superpowers/specs/2026-08-05-w22-8-multi-mask-pipeline-design.md` | W22.8 多 mask | 与 W28 决策无直接耦合；引用作为训练侧 checkpoint 内容演化的参考 |
| `docs/dev/next-steps-w27-v2.md` | 当前 next-step 队列 | §1 #2 W28（本设计） / §1 #3 W25-roadmap / §1 #4 W32 |

---

## 附录 B：术语表

| 术语 | 含义 |
|---|---|
| Manifest | scheduler 发布的不可变描述文件，标识某一时刻某 adapter 的权重版本与 ckpt 位置（§5.4 schema） |
| In-flight 请求 | 已进入 FastAPI 路由但尚未返回响应的 HTTP 请求；权重切换时需其完成或被 graceful 驱逐 |
| Grace period | 权重切换期间允许 in-flight 请求完成的最长等待时间（§5.3 / D6） |
| Quota | 单卡 vRAM 在训练 / 推理之间的硬性比例分配（§6） |
| Quarantine | adapter 在 OOM 驱逐后进入的冷却状态，期间 scheduler 不向其路由请求（§6.3） |
| Save window | 训练侧进入 `save_checkpoint` 的短暂窗口，期间训练冻结 EMA 更新、推理只读不切换（§7 F2） |
| Parent version | manifest 链中的前驱版本；用于权重加载失败时回滚（§5.4 / §7 F3） |

---

## 附录 C：W28 v1 评审 checklist（供 chad 使用）

- [ ] §4.3 推荐 B 方案的理由是否充分？
- [ ] §5.2 双通道（push + polling fallback）是否接受？
- [ ] §5.3 在飞请求衔接流程（30s grace period）是否合理？
- [ ] §5.4 manifest schema 草案是否需要补充字段？
- [ ] §6.1 vRAM 数字（60/40 拆分）是否需要等 W20/W21 上机后再冻结？
- [ ] §6.3 OOM 兜底（推理端主动驱逐 + manifest 回滚）是否接受？
- [ ] §7 失败模式 F1–F8 是否有遗漏？
- [ ] §8 集成点 I1–I9 是否覆盖完整？
- [ ] §9 W37 前置条件 7 条是否完整？
- [ ] §10.2 开放问题 8 条是否需要 chad 直接决策（vs 留给 W37 实施期）？
- [ ] §11 ADR 决策日志 D1–D12 是否有需要修改/补充？