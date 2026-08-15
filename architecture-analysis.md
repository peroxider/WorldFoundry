# WorldFoundry 架构分析

> 分析时间：2026-08-11  
> 项目根目录：`C:\WorkSpace\WorldFoundry`  
> 版本：0.1.0 (Phase-1 ~ Phase-5 演进中)

---

## 一、项目定位

**WorldFoundry** 是一个面向 **Ascend NPU** 的 **世界模型（World Model）训练 + 推理平台**，基于 Meta 的 **V-JEPA 2**（Video Joint Embedding Predictive Architecture）架构。核心能力覆盖：

- **训练**：自监督视频表征学习（JEPA 目标、LoRA/全量微调、DDP/FSDP 分布式）
- **推理**：FastAPI 服务化部署，支持 masked / temporal / action-conditioned 三种预测模式
- **RL**：PPO 策略优化，支持与仿真环境（Franka Kitchen / MuJoCo / DROID）闭环
- **量化**：W8A8 量化（msmodelslim / torchao）与编译（MindIE-SD）

---

## 二、顶层包结构（Monorepo 四包划分）

```
WorldFoundry/
├── worldfoundry_common/      # 共享层：契约、注册表、异常、NPU 环境
├── worldfoundry_train/       # 训练层：数据集、模型、训练循环、并行策略
├── worldfoundry_inf/         # 推理层：FastAPI 服务、Worker、批处理、量化
├── worldfoundry_rl/          # RL 层：环境、PPO、策略网络、分布式训练
```

四包之间形成 **单向依赖 DAG**：

```
worldfoundry_common  ←──────  被所有人依赖（契约层）
       ↑
worldfoundry_train  ←──────  worldfoundry_inf 需要加载训练 checkpoint
       ↑                           ↑
       └────────────────── worldfoundry_rl 需要训练层的 device/checkpoint
```

**依赖方向原则**：上层可依赖下层，同层不循环。`common` 作为最底层，**禁止反向依赖**。

---

## 三、逐包架构深度解析

### 3.1 worldfoundry_common — 共享契约层

| 模块 | 职责 |
|------|------|
| `exceptions.py` | 领域异常层次：`WorldFoundryError` → `VersionMismatchError`, `ModelNotFoundError`, `OMCompileError` 等 |
| `model_registry.py` | 模型注册表（`MODEL_REGISTRY`），当前注册 `vjepa2-vitl-1.2b`，支持本地路径 / HF Hub 回退 |
| `npu_env.py` | NPU 环境管理：版本锁定矩阵（torch 2.9.0 + torch_npu 2.9.0 + transformers 4.55.0）、设备检测、`hccl` 分布式初始化 |
| `metrics.py` | Prometheus / OpenTelemetry 指标收集 |
| `adapters/base.py` | **协议层**：`InferenceWorker`（encode/predict/plan）、`RolloutWorker`（generate_sequences）、`TrainingRunner`（run(config)）—— 外部框架（verl / rlinf / sglang）的接入契约 |
| `adapters/{verl,rlinf,sglang}.py` | 外部框架适配器的懒加载实现 |
| `backends/` | **五轴后端插件系统**：device / training / inference / model / sim_env，基于 `BackendRegistry` 的统一注册-发现-构建机制 |

**关键设计：五轴后端插件系统**

```python
# worldfoundry_common/backends/__init__.py
_AXES = {
    "device":     device.registry,      # cpu / npu / auto
    "training":   training.registry,    # self / verl / rlinf
    "inference":  inference.registry,   # torch / vllm / sglang
    "model":      model.registry,       # self / transformers
    "sim_env":    sim_env.registry,     # mock / mujoco / droid
}
```

每个 backend 延迟导入（lazy import），保证未安装的可选依赖不会污染导入路径。`Stack.compose()` 支持从 YAML 配置一键解析全栈。

---

### 3.2 worldfoundry_train — 训练引擎层

#### 3.2.1 目录结构

```
worldfoundry_train/
├── data/               # 数据管道：dataset, collator, tubelet_mask, synthetic_video, video_source
├── model/              # 模型定义：tiny_jepa, predictor, temporal_predictor, action_predictor, ema, lora_wrapper, hf_vjepa_shim
├── parallel/           # 分布式策略：ddp.py, fsdp.py, mindspeed.py, strategy.py (抽象基类)
├── recipe/             # RTI (Render-Train-Inference) 流水线：config, stages, pipeline, backends
├── eval/               # 评估：jepa_probing, pilot_checks
├── bench/              # 基准测试：runner, compare
├── train.py            # 训练 CLI 入口（YAML 驱动）
├── trainer.py          # 训练编排：数据集 → LoRA → 优化器 → 训练循环
├── training_loop.py    # 核心训练循环：前向 + 反向 + 梯度累积 + 分布式 step
├── jepa_objective.py   # JEPA 训练目标：Predictor + EMA Target Encoder + Loss
├── checkpoint.py       # Checkpoint 保存/加载（含 LoRA signature 验证）
├── device.py           # 设备解析（W20 NPU 迁移层）
└── quantize_bridge.py  # 训练-推理量化桥接
```

#### 3.2.2 核心架构模式

**A. 策略模式 — DistributedStrategy**

```python
class DistributedStrategy(ABC):
    def wrap(self, model: nn.Module) -> nn.Module: ...
    def backward(self, loss: torch.Tensor) -> None: ...
    def step(self, optimizer: Optimizer) -> None: ...
    def no_sync(self, model) -> ContextManager: ...  # 梯度累积用
```

实现：`DDPStrategy` / `FSDPStrategy` / `MMStrategy`（MindSpeed-MM）。训练循环通过 `strategy.backward(loss)` 和 `strategy.step(optimizer)` 与具体并行实现解耦。

**B. 装饰器模式 — PredictorGradSyncStrategy**

```python
class PredictorGradSyncStrategy(DistributedStrategy):
    """包装任意 Strategy，在 step() 前对 Predictor 的梯度做显式 all_reduce。
    Predictor 位于 DDP/FSDP wrap 之外（~21M 参数，每 rank 全量副本），
    因此需要手动梯度同步。"""
```

**C. JEPA 训练目标 — JEPAObjective**

```
online_encoder(video) ──→ h_ctx ──→ predictor(h_ctx, mask) ──→ pred
                                    ↓
target_encoder(video) ──→ h_tgt ────────────────┘
                                    ↓
                              loss = smooth_l1(pred, h_tgt[mask])
                                    ↓
                            after_step() → EMA.update()
```

关键约束（设计 §5.5）：
- `target_encoder` 在 `strategy.wrap()` **之前** deepcopy，避免被 FSDP 包裹
- EMA 更新发生在 **optimizer step 之后**（`post_step_hook`），而非每个 micro-batch
- 多 mask 支持（W22.8）：`bool_masked_pos_k` 形状 `(B, K, N)`，K 个 mask 独立采样取平均

**D. 训练循环 — 梯度累积语义**

```
for step in range(num_steps):          # num_steps 计 micro-batch
    is_sync = (step+1) % grad_accum == 0 or last_step
    with strategy.no_sync() if not is_sync else nullcontext():
        loss = loss_fn(model, batch)
        strategy.backward(loss / grad_accum)
    if is_sync:
        strategy.step(optimizer)
        post_step_hook()               # EMA 更新
```

返回的 `losses` 长度 = `ceil(num_steps / grad_accum)`，每个元素是一个 accumulation window 的均值损失。

**E. Recipe 流水线（RTI）**

```python
@dataclass(frozen=True)
class RecipeConfig:
    backend: str = "cpu"
    stages: tuple[str, ...] = ("render", "train", "inference", "policy", "outer_loop")
    # ... 所有超参扁平化

def run_recipe(cfg: RecipeConfig) -> dict:
    device = cfg.resolve_device()          # "cpu" or "npu"
    stage_map = get_backend_stages(cfg.backend)
    ctx = RecipeContext(cfg, device)
    for stage_name in cfg.stages:
        stage_map[stage_name]().run(ctx)   # 每阶段读写 ctx.results
    return summary
```

RTI 流水线将 Render → Train → Inference → Policy → Outer Loop 串成可配置、可跳过阶段（如 `skip_render=True`）的声明式管道。

---

### 3.3 worldfoundry_inf — 推理服务层

#### 3.3.1 服务架构

```
FastAPI App (server.py)
    ├── /health                  # 健康检查（NPU 可用性、adapter 列表、batcher 配置）
    ├── /v1/world/encode         # 视频 → dense features
    ├── /v1/world/predict        # 三种模式：masked / temporal / action
    └── /v1/world/plan           # 动作规划（W33）

Worker: VJEPA2Worker
    ├── encode(video)            # 基础编码
    ├── predict(masked_video, mask)         # masked mode
    ├── predict_temporal(video, K, H)       # W22 temporal mode
    └── predict_action_conditioned(video, actions, K, H)  # W27 v2 action mode

Batcher: PredictBatcher (W25)
    └── 动态批处理：max_batch + max_latency_ms 双重触发
```

#### 3.3.2 VJEPA2Worker 状态机

```
model = load_hf_vjepa2(model_path)          # 基础编码器

可选加载（互不影响）：
├── predictor_ckpt → bundle (JEPInferenceBundle)
│   ├── online encoder (训练 checkpoint 权重)
│   ├── target encoder (EMA 快照)
│   ├── jepa_predictor
│   └── temporal_predictor
└── action_predictor_ckpt → action_bundle
    ├── online encoder
    ├── target encoder
    └── action_predictor
```

两种 checkpoint 独立加载，worker 可以同时支持 W22 temporal + W27 action 两种模式。

#### 3.3.3 推理批处理（PredictBatcher）

- 延迟聚合：请求进入队列，等待 `max_batch` 填满或 `max_latency_ms` 超时
- 三种 batch 方法：`predict_masked_batch` / `predict_temporal_batch` / `predict_action_conditioned_batch`
- 批量内要求 uniform shape（同 T/C/H/W），mixed-shape 抛 `ValueError`

#### 3.3.4 量化编译

```
worldfoundry_inf/compile/
├── quantize.py           # 量化 facade：W8A8 / W8A16 / W4A16 dispatch
├── mindie_sd.py          # MindIE-SD 编译入口（Ascend OM 模型）
└── quantize_backends/
    ├── fake.py           # 假量化 backend（CPU 测试）
    └── msmodelslim.py    # 华为 msmodelslim backend（真量化）
```

---

### 3.4 worldfoundry_rl — 强化学习层

#### 3.4.1 模块职责

| 模块 | 职责 |
|------|------|
| `ppo.py` | **自研 PPO 训练器**（W34）：GAE + Clipped Surrogate + Value Loss + Entropy，作为 verl/RLinf 的 correctness baseline |
| `vision_policy.py` | **可插拔编码器的 Actor-Critic**（W41）：支持 state_mlp / TinyVJEPA2Encoder / HF VJEPA2 + LoRA 三种编码器后端 |
| `env_*.py` | 环境封装：FrankaKitchen / MuJoCo Manipulator / DROID / Mock |
| `env_pool.py` | 向量化环境池：`SyncEnvPool`（多进程）|
| `actor_pool.py` / `learner_pool.py` | 分布式 Actor-Learner 分离架构（W37+，尚未完整实现）|
| `train.py` | RL 训练 CLI 入口（YAML 驱动，镜像 train.py 表面）|

#### 3.4.2 VisionPolicy 编码器策略

```python
VisionPolicy
├── from_state_dim()          # 2-layer MLP encoder（CPU smoke 默认）
├── from_image_obs()          # TinyVJEPA2Encoder + mean-pooling
└── from_hf_image_obs()       # HF VJEPA2Model + LoRA + mean-pooling
```

所有 factory 统一返回 `feature_dim` 维特征，送入共享的 policy_head / value_head。编码器可 freeze（默认）或端到端微调。

#### 3.4.3 PPO 架构

```
PPOActor ( vision_policy.py )
    ├── backbone (encoder)  → features
    ├── policy_head         → action_mean
    ├── value_head          → value
    └── log_std (可学习参数，state-independent diagonal Gaussian)

PPO ( ppo.py )
    ├── collect_rollout(pool) → RolloutBuffer (T steps × N envs)
    ├── compute_gae()         → advantages / returns
    └── update(buffer)        → minibatch PPO epochs
```

---

## 四、关键技术决策与约束

### 4.1 版本锁定矩阵（Global Constraint G2）

| 包 | 锁定版本 |
|---|---|
| torch | 2.9.0 |
| torch_npu | 2.9.0 |
| transformers | 4.55.0 |
| peft | 0.18.0 |
| accelerate | 1.10.0 |
| vllm / vllm-ascend | 0.18.0 |

运行时通过 `npu_env.check_versions(strict=True)` 校验，不匹配抛 `VersionMismatchError`。

### 4.2 设备抽象层（W20）

```python
# worldfoundry_train/device.py
def resolve_device() -> str:
    """优先顺序：WORLDFORGE_DEVICE 环境变量 → NPU 自动检测 → CPU"""

def to_device(module, device: str):
    """统一迁移：CPU 上无操作，NPU 上 .to("npu")"""
```

所有模型构造后调用 `to_device(model, resolve_device())`，保证 **单一代码路径同时支持 CPU 测试和 NPU 生产**。

### 4.3 延迟导入（Lazy Import）原则

- 所有外部框架适配器（verl / rlinf / sglang）在方法内部懒加载
- vllm-ascend 插件仅在入口点触发注册
- `worldfoundry_inf` 可选依赖 `[inf]` 不安装时，基础包仍可导入

### 4.4 CPU RTI Smoke（~3.5s 端到端验证）

```bash
bash scripts/render_train_rl_cpu.sh
# Phase: Render → Train → Inference → PPO → Outer Loop
```

使用 `TinyVJEPA2Encoder`（~80K 参数）替代 1.2B ViT-L，在单 CPU 线程上验证全链路正确性，无需 HF 下载或 NPU。

---

## 五、数据流全景

### 5.1 训练数据流

```
视频文件 / 合成数据
    ↓
Vjepa2Dataset (manifest.csv 驱动)
    ↓
collate_vjepa2_batch() + tubelet_mask()
    ↓
(B, T, C, H, W) pixel_values_videos
(B, T, Hg, Wg) bool_masked_pos        ← tubelet mask
    ↓
model(pixel_values_videos, bool_masked_pos)  [with skip_predictor=True]
    ↓
last_hidden_state (B, N_tokens, hidden_dim)
    ↓
JEPAPredictor(context_tokens, mask) → pred (M, encoder_dim)
target_encoder(video) → h_tgt[mask]   (stop-gradient)
    ↓
jepa_loss(pred, target) = smooth_l1 / cosine / l1
    ↓
backward + DDP/FSDP all_reduce + optimizer.step
    ↓
EMAUpdater.after_step()  (每 optimizer step，非 micro-batch)
```

### 5.2 推理数据流

```
POST /v1/world/predict
    ↓
base64 video → np.array (T, 64, 64, 3) → torch.Tensor (1, T, 3, 64, 64)
    ↓
mode="masked":   worker.predict(video, mask) → features
mode="temporal": worker.predict_temporal(video, K, H) → (future_features, confidence)
mode="action":   worker.predict_action_conditioned(video, actions, K, H) → (future_features, confidence)
    ↓
JSON response
```

### 5.3 RL 数据流

```
EnvPool (N parallel envs)
    ↓
PPO.collect_rollout()  →  RolloutBuffer (T steps × N envs)
    ↓
compute_gae(rewards, values, dones) → advantages, returns
    ↓
PPO.update()  →  minibatch SGD over epochs_per_update epochs
    ↓
policy weights updated → next rollout
```

---

## 六、测试策略

| 包 | 测试目录 | 测试数量级 | 关键测试 |
|---|---|---|---|
| common | `tests/` | ~6 模块 | 适配器契约、后端注册表、NPU 环境、异常 |
| train | `tests/` | ~40+ 模块 | DDP/FSDP smoke、JEPA 目标、EMA、checkpoint、梯度累积、recipe 流水线 |
| inf | `tests/` | ~20+ 模块 | API endpoint、batching、量化 backend、安全、插件注册 |
| rl | `tests/` | ~10+ 模块 | PPO smoke、环境池、策略网络、分布式训练 |

测试设计原则：
- **CPU 可运行**：所有核心测试在 CPU 上通过，无需 NPU
- **NPU 专属测试**：`test_*_npu.py` 命名隔离，pytest 条件跳过
- **Smoke 测试**：`<1s` 的端到端验证，保证回归快速反馈

---

## 七、当前演进阶段

| 阶段 | 状态 | 关键交付 |
|---|---|---|
| Phase-1 | ✅ 完成 | CPU 训练 + 推理 smoke、LoRA、FSDP |
| Phase-2 W9-W10 | ✅ 完成 | MindSpeed-MM 集成、FSDP 扩展 |
| Phase-2 W11-W12 | ✅ 完成 | W8A8 量化、msmodelslim backend |
| Phase-3 W22 | ✅ 完成 | JEPA Predictor + EMA、Temporal Prediction、Multi-mask |
| Phase-3 W24-W27 | ✅ 完成 | Action-conditioned predictor、Inference bundle |
| Phase-4 W33-W35 | ✅ 完成 | Plan endpoint、PPO baseline、RL framework ADR |
| Phase-5 W37-W41 | 🔄 进行中 | VisionPolicy、多步 latent rollout、端到端 RL 闭环 |

---

## 八、架构优势与风险

### 优势
1. **清晰的单向分层**：common → train → inf/rl，无循环依赖
2. **协议先行**：`InferenceWorker` / `RolloutWorker` / `TrainingRunner` 协议让外部框架（verl / vllm / sglang）可插拔替换
3. **CPU-NPU 同构**：单一代码路径，设备解析层统一封装
4. **声明式配置**：YAML 驱动 + Recipe 流水线，支持阶段跳过和复用
5. **严格的版本锁定**：避免依赖漂移导致的 NPU 兼容性问题

### 风险
1. **包边界泄漏风险**：`worldfoundry_inf` 直接导入 `worldfoundry_train.model.jepa_inference`，若 train 包膨胀可能导致 inf 启动变慢
2. **FSDP + EMA 内存**：target encoder 全量复制在 FSDP 外，大模型时显存峰值高（设计 §8.3 已标记与 cpu_offload 互斥）
3. **Predictor 梯度同步**：`PredictorGradSyncStrategy` 手动 all_reduce 是分布式瓶颈点，大规模集群需验证带宽
4. **多 mask 计算开销**：K>1 时 predictor 前向 K 次，虽 encoder 共享但 predictor 成为新瓶颈
