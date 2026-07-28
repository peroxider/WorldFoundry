# WorldForge — 昇腾 NPU 世界模型训练 + 推理平台设计

**作者**：MiniMax-M3（与 chad 共同设计）
**日期**：2026-07-27
**状态**：✅ 用户已批准，进入实施计划阶段

---

## 1. 上下文与决策摘要

### 1.1 项目定位

构建一套**面向昇腾 NPU**（Atlas 800 A2 / 910B × 8）的**世界模型（World Model）训练框架 + 推理引擎软件平台**，基线模型为 **Meta V-JEPA 2 / V-JEPA 2.1 / V-JEPA 2-AC**。

世界模型方向：具身 / 自动驾驶（感知 + 预测）。

### 1.2 已锁定的关键决策

| 维度 | 决策 | 理由 |
|------|------|------|
| 基线模型 | V-JEPA 2 (1.2B) + V-JEPA 2.1 + V-JEPA 2-AC | 参数量小、社区热门、HF transformers 原生支持、AC 范式天然对应 RL 后训练 |
| 硬件 | Atlas 800 A2 / 910B × 8 (单节点，RoCE v2 + HCCL) | 1.2B 全参微调显存/算力充足，AC 训练推理对算力要求中等 |
| 与 MIS 关系 | 完全无关；复用 vLLM / vLLM-Omni / SGLang 推理生态 | MIS 是 LLM 推理微服务；V-JEPA 2 是非生成式 ViT，自建推理服务 |
| 训练范围 | 下游微调（SFT / LoRA）+ 后训练 RL | 不做从零预训练；复用官方权重 |
| 架构演进 | Phase-1 出原型 (torch_npu + FSDP)，关键模块后续走 MindSpeed | 务实演进：先出活，再优化热点 |

### 1.3 风险偏好

- **不重写上游**：训练代码尽量少改 facebookresearch/vjepa2；推理走 vLLM-Ascend 插件机制而非 fork
- **渐进式升级**：原型 → 性能优化 → 通用化；每个阶段可独立验收
- **上游回馈**：所有对 facebookresearch/vjepa2、vllm-ascend、verl 的改进通过 PR 回馈

---

## 2. 整体架构

### 2.1 系统拓扑

```
┌──────────────────────────────────────────────────────────────────────────┐
│              WorldForge — 昇腾世界模型训练+推理平台                        │
│         (基于 V-JEPA 2 / V-JEPA 2.1 / V-JEPA 2-AC)                        │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────┐  ┌──────────────────┐  ┌──────────────────────┐    │
│  │ 训练子系统        │  │ RL 后训练子系统   │  │ 推理引擎              │    │
│  │ WorldForge-Train │  │ WorldForge-RL    │  │ WorldForge-Inf       │    │
│  │                  │  │                  │  │                      │    │
│  │ • Data Pipeline  │  │ • verl/trl +     │  │ • vLLM-Ascend        │    │
│  │   (DVPP decode,  │  │   torch_npu      │  │   插件挂载 V-JEPA 2  │    │
│  │   masked view)   │  │ • V-JEPA 2-AC    │  │ • 自研 batch service │    │
│  │ • FSDP /         │  │   动作条件化      │  │ • MindIE-SD 编译     │    │
│  │  MindSpeed-MM*   │  │ • 仿真环境集成    │  │ • W8A8/W4A8 量化    │    │
│  │ • LoRA / SFT /   │  │   (MuJoCo/SAPIEN │  │                      │    │
│  │   Full-FT        │  │    /CARLA-AD)    │  │                      │    │
│  │ • DDP/FSDP/      │  │                  │  │                      │    │
│  │   ZeRO-1/2/3     │  │                  │  │                      │    │
│  └────────┬─────────┘  └─────────┬────────┘  └──────────┬───────────┘    │
│           │                    │                      │                │
│           └────────────────────┼──────────────────────┘                │
│                                │                                        │
│                    ┌───────────▼──────────┐                             │
│                    │   共享层 (Common)     │                             │
│                    │ • 模型仓库 (OM ckpt)  │                             │
│                    │ • Ascend NPU 驱动栈    │                             │
│                    │   (CANN 8.5.1 + torch │                             │
│                    │    _npu 2.9.0 + HCCL)  │                            │
│                    │ • 评价/可视化         │                             │
│                    │ • 模型配置/Schema     │                             │
│                    └────────────────────────┘                             │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘

外部对齐:
  facebookresearch/vjepa2 (上游 PyTorch)
  transformers.VJEPA2Model (HF 上游)
  vllm-ascend / vllm-omni / verl / MindSpeed-LLM+MM (昇腾生态, ViT支持待评估)
```

### 2.2 设计原则

1. **三子系统独立部署**：训练、RL、推理各自可独立演进，避免相互阻塞
2. **共享底座锁定硬件依赖**：CANN 8.5.1 + torch_npu 2.9.0 + HCCL 一处升级
3. **上游兼容**：训练代码尽量少改 facebookresearch/vjepa2，PR 回馈社区
4. **接口契约化**：训练产出 OM/PT 权重 → 推理直接消费，模型仓库即接口

---

## 3. 训练子系统（WorldForge-Train）

### 3.1 数据流水线

```
原始视频 (mp4/webm)
   │  Ascend DVPP 硬件解码 (CANN 内置) + decord 兜底
   ▼
帧序列 (T×H×W×3, uint8) → resize → center-crop → normalize
   │
   │  V-JEPA 2 多视图采样 (Tube/Grid mask generator, 复用 facebookresearch/vjepa2)
   ▼
Masked Tokens + Context Tokens (transformers 兼容张量)
   │
   ▼
HuggingFace datasets.ArrowDataset (内存映射到 NVMe SSD)
```

**关键技术选择**：

- **解码**：优先 **Ascend DVPP**（CANN 硬件解码）+ decord 兜底；DVPP 单卡可达 300+ fps 解码
- **采样**：复用 `facebookresearch/vjepa2` 的 `tubelet_masking.py`（Tube/Grid 双模式）
- **存储**：HF datasets + Arrow 格式，分片并行读取
- **数据源**：VideoMix22M（公开）+ Droid（机器人）+ 内部 AD 数据集（接口预留）

### 3.2 模型包装（LoRA 微调）

```python
from transformers import VJEPA2Model
from peft import LoraConfig, get_peft_model

base = VJEPA2Model.from_pretrained(
    "facebook/vjepa2-vitl-1.2b",
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",  # Ascend SDPA 后端
)

# ⚠️ Phase-1 W2 必做：先打印确认实际模块名
#   for name, module in base.named_modules():
#       print(name, type(module).__name__)
# 然后再填写 target_modules。V-JEPA 2 基于 ViT-Large，但 HF 实现可能使用
# fused qkv (单个 qkv_proj) 或分离的 q/k/v_proj，需要实际验证。
peft_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "mlp.fc1", "mlp.fc2"],   # 候选默认值，Phase-1 验证
    lora_dropout=0.05, bias="none",
    modules_to_save=["predictor.head"],  # 预测头全参
)
model = get_peft_model(base, peft_config)

# 启动 sanity check：确认 LoRA 实际匹配到参数
model.print_trainable_parameters()
# 期望: trainable params > 0 且 ~30M 量级
```

### 3.3 分布式训练（双阶段演进）

| 阶段 | 框架 | 并行策略 | 适用场景 | 切换条件 |
|------|------|----------|----------|----------|
| **Phase-1 (原型)** | torch_npu + FSDP (PyTorch 2.9 native) | FSDP-2 + Hybrid Shard (ZeRO-3) 或 DDP（LoRA 时优先） | LoRA 微调、轻量续训 | 跑通即启 Phase-2 |
| **Phase-2 (优化)** | MindSpeed-MM (候选) + torch_npu | TP-1 + DP-8 + ZeRO-1 + Activation Checkpointing | 全参微调、长序列训练 | LoRA 收敛后切换（**W9 前先评估 MindSpeed-MM 对 ViT-only 模型支持；若不支持回退 MindSpeed-LLM 或继续 FSDP**） |

**Phase-1 启动命令**：

```bash
# 默认走 DDP（LoRA + 1.2B 模型大小 + 8×64GB 不需要分片）
torchrun --nproc_per_node=8 \
  worldforge_train/train.py \
  --model facebook/vjepa2-vitl-1.2b \
  --dataset /data/videomix22m \
  --parallel ddp \
  --bf16 --gradient_checkpointing \
  --lora_r 16 --lora_alpha 32 \
  --per_device_batch 2 --grad_accum 8
# 全参微调时切换到 FSDP（启用 ZeRO-3）：
#   --parallel fsdp --fsdp_shard_size 8 --fsdp_full_shard
```

**为什么默认 DDP**：1.2B bf16 ≈ 2.4GB；+ AdamW 状态 ~9.6GB；+ 梯度 ~2.4GB；每卡 ~15GB，远低于 64GB 单卡显存。FSDP-8 分片对该模型收益有限（额外通信开销可能反超）。**全参微调**或**长序列**才值得启用 FSDP。

### 3.4 检查点 & 模型仓库

- 训练产出：PEFT adapter (LoRA) + merged bf16 checkpoint → 转 OM 格式（MindIE-SD 编译入口）
- 模型仓库目录结构（**契约化**）：

```
models/
  vjepa2-vitl-1.2b/
    ├── config.json              # HF transformers
    ├── pytorch_model.bin        # 训练产出
    ├── model.om                 # 推理编译产物
    ├── lora_adapters/
    │   ├── embodied-v1/         # 具身任务 LoRA
    │   └── driving-v1/          # AD LoRA
    └── eval_metrics.json
```

### 3.5 评价指标

- **视频理解**：Perception Test、EgoSchema、IntentBench
- **动作预测**：Something-Something V2
- **具身规划**：Droid 验证集（V-JEPA 2-AC 原生指标）
- **自定义 AD**：与仿真环境接口，输出端到端成功率

---

## 4. 推理引擎（WorldForge-Inf）

### 4.1 核心问题：V-JEPA 不是 LLM

V-JEPA 2 是**非自回归 ViT 编码器**（一次 forward 输出全部特征），传统 vLLM 的 PagedAttention/KV-cache 调度不适用。但可以**复用 vLLM-Ascend 的 Hardware Pluggable 机制**：注册为 `VJEPA2Worker`，复用其算子库、量化工具、HCCL 通信。

### 4.2 推理服务架构

```
┌────────────────────────────────────────────────────────────┐
│  WorldForge-Inf Service (gRPC :50051 / HTTP :8080)          │
├────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  REST/gRPC   │  │  Batcher     │  │  Scheduler   │    │
│  │  Frontend    │─▶│  (video      │─▶│  (HCCL aware) │    │
│  │  OpenAI-     │  │  stream      │  │              │    │
│  │  compatible  │  │  buffer)     │  │              │    │
│  └──────────────┘  └──────────────┘  └──────┬───────┘    │
│                                              │            │
│                              ┌───────────────▼────────┐  │
│                              │  VJEPA2WorkerPool       │  │
│                              │  (8×910B, TP=1)         │  │
│                              │  ┌──────────────────┐   │  │
│                              │  │ MindIE-SD        │   │  │
│                              │  │ compiled OM      │   │  │
│                              │  │ (per LoRA        │   │  │
│                              │  │  adapter)        │   │  │
│                              │  └──────────────────┘   │  │
│                              └────────────────────────┘  │
│                                                              │
│  共享 vLLM-Ascend Operator Library:                          │
│   FlashAttention/SFA · TopK · RMSNorm · RoPE · Quant        │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 三个推理模式

| 模式 | 输入 | 输出 | 场景 |
|------|------|------|------|
| **encode** | video tensor | dense features `[B, N_tokens, D]` | 视频分类/检索 |
| **predict** | masked video | predicted features + recon loss | 物理一致性评估 |
| **plan** | (state, action_history) | next-frame prediction | 具身/AD 决策 |

### 4.4 量化与编译

```python
from vllm_ascend.quantization import ModelSlimConfig

quant_cfg = ModelSlimConfig(
    quant_type="w8a8",           # Phase-1: 简单 W8A8
    # quant_type="w4a8",         # Phase-2: 极致压缩
    calibration_dataset=videomix22m_calib_512,
)
model_om = compile_with_mindie_sd(
    model="models/vjepa2-vitl-1.2b",
    output="models/vjepa2-vitl-1.2b/model.om",
    config=quant_cfg,
    target_hardware="Atlas 800 A2 / 910B",
)
```

### 4.5 LoRA 热加载

由于 V-JEPA 2 推理开销相对小，**支持多 LoRA 适配器在线热切换**：
- 每个 LoRA adapter 在启动时编译为独立 OM（小模型，单 LoRA ~30MB）
- 请求携带 `adapter_name` 字段，调度到对应 OM
- 无需重启服务即可切换 embodied-v1 / driving-v1

### 4.6 接口契约

```
POST /v1/world/encode
  body: { video: base64_or_url, adapter?: "embodied-v1", quant?: "w8a8" }
  resp: { features: float32[binary], shape: [N_tokens, D] }

POST /v1/world/predict
  body: { video, mask_ratio: 0.7, ... }
  resp: { predicted_features, loss_score }

POST /v1/world/plan  (V-JEPA 2-AC)
  body: { video, actions: [a_1..a_t] }
  resp: { next_features, planned_actions }
```

OpenAI 兼容层只是薄壳，**核心数据格式对齐 HF transformers.VJEPA2Model 的输入输出**。

### 4.7 安全 / 鉴权 / 监控

- **认证**：Phase-1 部署在内网，HMAC 签名 token（共享密钥 + timestamp + 路径）；Phase-2 引入 mTLS（双向证书）
- **限速**：每 adapter 每分钟 token 桶限速；推理 QPS > 1000 时返回 429
- **输入校验**：video URL 白名单 / 文件大小 ≤ 200MB / 时长 ≤ 60s
- **可观测**：Prometheus 指标（`worldforge_rps`, `worldforge_latency_ms`, `worldforge_om_load_errors`）+ OpenTelemetry trace
- **失败处理**：
  - OM 加载失败 → 降级到原始 bf16 + 标记降级状态
  - 单 batch DVPP 解码失败 → 回退到 decord
  - LoRA 不存在 → 拒绝请求（4xx），不允许 silent 默认

---

## 5. RL 后训练子系统（WorldForge-RL）

### 5.1 三种 RL 范式（在 V-JEPA 2 上的应用）

| 范式 | 描述 | V-JEPA 2 对应 | 难度 |
|------|------|---------------|------|
| **AC（Action-Conditioned）** | 学习 latent action 表示 | V-JEPA 2-AC（Meta 已开源 62h Droid 训练） | 中 |
| **WM-as-Sim（DreamerV3）** | 用世界模型当 simulator，policy 在 latent 中学 | actor + critic 在 V-JEPA 2 features 上 | 高 |
| **RFT（Reward Fine-Tuning）** | 任务级奖励反传到 encoder | 端到端 PPO/GRPO with reward model | 中 |

**MVP 选 AC → WM-as-Sim → RFT 渐进**。Phase-1 先把 V-JEPA 2-AC 跑通（Meta 已有完整脚本）。

### 5.2 架构（基于 verl + torch_npu）

```
┌──────────────────────────────────────────────────────────────────┐
│                       WorldForge-RL                               │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ ActorPool    │    │ EnvPool      │    │ LearnerPool  │       │
│  │ (4×910B)     │    │ (CPU+GPU)    │    │ (4×910B)     │       │
│  │              │    │              │    │              │       │
│  │ • V-JEPA 2   │───▶│ • MuJoCo/    │───▶│ • LoRA       │       │
│  │   forward    │    │   SAPIEN     │    │   update     │       │
│  │   rollout    │    │ • CARLA-AD   │    │ • PPO/GRPO   │       │
│  │ • action     │    │ • Droid sim  │    │ • KL reg     │       │
│  │   sampling   │    │              │    │              │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         ▲                                        │               │
│         │            ┌──────────────┐            │               │
│         └────────────│ RewardModel  │◀───────────┘               │
│                      │ (1×910B)     │                            │
│                      │ • task success│                           │
│                      │ • physics    │                            │
│                      │   consistency│                            │
│                      └──────────────┘                            │
│                                                                  │
│  verl RayActor + torch_npu worker (已 Ascend 适配)               │
└──────────────────────────────────────────────────────────────────┘
```

### 5.3 V-JEPA 2-AC 训练流（Phase-1 MVP）

> ⚠️ **范围说明**：Meta 的 V-JEPA 2-AC 是基于 **1M 小时预训练 + 62h Droid 微调**得到的；W7-W8 仅复现 **AC 微调阶段**（latent action 训练），完整 80% 指标对齐不在 Phase-1 范围。

```python
# 基于 facebookresearch/vjepa2 官方 evals/ 目录的 AC 训练脚本改编
# （Phase-1 W7 任务：先打开该仓库 evals/ 目录确认实际脚本名再 fill in）
from worldforge_rl import VJEPAActor, DroidEnv, PPOTrainer

actor = VJEPAActor.from_pretrained(
    "facebook/vjepa2-vitl-1.2b",   # 起点：原 V-JEPA 2 权重
    load_lora_adapter="optional/vjepa2-ac-init",
)
env = DroidEnv(task="pick_place", max_steps=50)
trainer = PPOTrainer(
    actor=actor,
    env=env,
    reward_fn=task_success_reward,
    kl_coef=0.1,
    clip_range=0.2,
    npu_device="npu:0..7",
)

trainer.fit(epochs=100, batch_size=32)
```

### 5.4 关键依赖

- **verl**（[volcengine/verl](https://github.com/volcengine/verl)）：已合并 Ascend NPU 后端 PR，torch_npu worker 支持 PPO/GRPO/REINFORCE++
- **MuJoCo / SAPIEN**：CPU 仿真（机器人任务）
- **CARLA / DriveArena**（可选）：自动驾驶仿真
- **wandb / tensorboard**：训练可视化

### 5.5 安全护栏

- **KL 散度约束**：policy 不能离原 V-JEPA 2 太远（防 reward hacking）
- **Reward shaping**：物理一致性奖励（不违反牛顿定律）+ 任务完成奖励
- **回滚机制**：eval loss 退化 >5% 自动回 checkpoint

---

## 6. 部署拓扑与开发路线图

### 6.1 硬件拓扑（Atlas 800 A2 / 910B×8）

```
┌─────────────────────────────────────────────────────────────────┐
│  Atlas 800 A2 单节点                                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐│
│  │NPU 0│ │NPU 1│ │NPU 2│ │NPU 3│ │NPU 4│ │NPU 5│ │NPU 6│ │NPU 7││
│  │64GB │ │64GB │ │64GB │ │64GB │ │64GB │ │64GB │ │64GB │ │64GB ││
│  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘│
│     └──────┴───┬────┴───┬────┴───┬────┴───┬────┴───┬────┴─────┘│
│           RoCE v2 (200Gbps) × HCCL 全互联                         │
│                                                                  │
│  CPU: 2× Kunpeng 920 (192 cores)                                │
│  Mem: 1TB DDR4                                                  │
│  NVMe: 8× 3.84TB (≈30TB raw, 视频数据集常用)                    │
└─────────────────────────────────────────────────────────────────┘

8 卡算力 (bf16):
  • Dense: ~1.4 PFlops (FP16)
  • V-JEPA 2 (1.2B) 全参微调 batch=2×8=16：单步 ~2-3s
  • LoRA (r=16)：单步 ~1-1.5s
  • DVPP 解码：300+ fps/卡
```

### 6.2 软件栈固定（CANN 8.5.1 + torch 2.9.0 + vllm-ascend 0.18.0）

> ⚠️ **版本矩阵**（按 [vLLM-Ascend Versioning Policy](https://docs.vllm.ai/projects/ascend/en/latest/community/versioning_policy.html)）：
> - vllm-ascend v0.18.0 对应 CANN 8.5.1 + torch 2.9.0
> - vllm-ascend v0.22.x 对应 CANN 9.0.0 + torch 2.10.0
> - vllm-ascend v0.23.0rc1 对应 CANN ≥ 8.5.0 (HDK 25.5.1+) + torch ≥ 2.10.0
>
> **Phase-1 起始基线**：CANN 8.5.1 + torch 2.9.0 + vllm-ascend 0.18.0（保守兼容）。Phase-2 评估升级到 v0.22.1+。

```yaml
# docker/worldforge-base/Dockerfile (基于 ascend-docker-image/ascend-pytorch)
FROM ascend-pytorch:2.9.0-cann8.5.1-ubuntu22.04
RUN pip install \
    transformers==4.55.0 \
    peft==0.18.0 \
    vllm==0.18.0 vllm-ascend==0.18.0 \
    verl==0.5.0 \
    decord==0.6.0 av==14.0.0 \
    datasets==3.6.0 accelerate==1.10.0
# 装 HF vjepa2 主干
RUN git clone https://github.com/facebookresearch/vjepa2 /opt/vjepa2
```

### 6.3 节点资源分配（训练任务时）

| NPU | 训练子系统 | 推理子系统 | RL 子系统 |
|-----|-----------|-----------|----------|
| 0   | Trainer rank 0 | -        | Actor 0 |
| 1   | Trainer rank 1 | -        | Actor 1 |
| 2   | Trainer rank 2 | -        | Actor 2 |
| 3   | Trainer rank 3 | -        | Actor 3 |
| 4   | Trainer rank 4 | Worker 0 | Learner 0 |
| 5   | Trainer rank 5 | Worker 1 | Learner 1 |
| 6   | Trainer rank 6 | Worker 2 | RewardModel |
| 7   | Trainer rank 7 | Worker 3 | Reference |

通过 `WORLDFORGE_PROFILE` 环境变量切换配置（`train` / `serve` / `rl` / `train+serve`）。

### 6.4 开发路线图（3 阶段 24 周）

```
Phase-1 (W1-W8) ─ 原型闭环
  ├─ W1-W2: 训练子系统 Phase-1 (FSDP + LoRA)
  ├─ W3-W4: 推理引擎 vLLM-Ascend 插件挂载 + MindIE-SD 编译
  ├─ W5-W6: 端到端 smoke test（VideoMix22M 子集训练 → 推理编码 → 评价）
  └─ W7-W8: V-JEPA 2-AC 复现（Meta 62h Droid）

  交付物：
    • worldforge-train (FSDP LoRA 跑通)
    • worldforge-inf (vLLM-Ascend + V-JEPA 2 Worker)
    • V-JEPA 2-AC 复现报告

Phase-2 (W9-W16) ─ 性能优化 + RL 闭环
  ├─ W9-W10: 评估并切换 MindSpeed-MM (TP+DP 混合并行)
  ├─ W11-W12: 量化 (W8A8 → W4A8) + 编译优化
  ├─ W13-W14: WorldForge-RL 完整 verl 集成
  └─ W15-W16: 自定义具身任务验证（自建仿真环境）

  交付物：
    • worldforge-train Phase-2 (MindSpeed-MM 提速 30%+, 不达标则保留 FSDP)
    • worldforge-rl (PPO/GRPO 跑通)
    • 性能 benchmark 报告

Phase-3 (W17-W24) ─ 通用化 & 上游回馈
  ├─ W17-W18: V-JEPA 2.1 升级（引入 Dense Predictive Loss 等）
  ├─ W19-W20: 自动 AD benchmark（与 CARLA 等对接）
  ├─ W21-W22: PR 回馈 facebookresearch/vjepa2 + vllm-ascend + verl
  └─ W23-W24: 文档化 & 培训材料

  交付物：
    • V-JEPA 2.1 训练全链路
    • 3+ 个 upstream PR
    • 用户手册 + API 文档
```

### 6.5 风险 & 缓解

| 风险 | 等级 | 缓解 |
|------|------|------|
| V-JEPA 2 在 Ascend NPU 上算子覆盖不全（如 SDPA 后端） | 高 | Phase-1 充分测试；准备回退到 eager + Ascend 实现 |
| LoRA target_modules 与 V-JEPA 2 实际模块名不匹配 → 静默零训练 | 高 | Phase-1 W2 强制要求先 `print_trainable_parameters()`，匹配为 0 则拒绝启动 |
| DVPP 与 HF datasets 数据格式不兼容 | 中 | 抽象 `VideoSource` 接口，DVPP/decord 双后端 |
| DVPP 硬件故障导致 batch 中断 | 中 | 4.7 失败处理：单 batch DVPP 失败回退 decord，整卡失败剔除该 NPU |
| vLLM-Ascend 注册 V-JEPA 2 需 fork 上游 | 中 | 通过 Hardware Pluggable 插件机制避免 fork；如必须 fork，开源回馈 |
| MindSpeed-LLM 对 ViT-only 模型支持薄弱 | 中 | 先 FSDP 跑通，MindSpeed 仅在瓶颈模块启用；若不支持回退 FSDP |
| vllm-ascend 与 CANN/torch 版本矩阵错位 | 中 | 锁定 vllm-ascend v0.18.0 + CANN 8.5.1 + torch 2.9.0；升级前查 Versioning Policy |
| verl Ascend 后端稳定性 | 中 | 沿用 verl 官方推荐 torch_npu worker；遇 bug 提 PR |
| W-JEPA 2-AC 复现无法在 W7-W8 达到 Meta 指标 | 中 | 范围收敛到 AC 微调阶段；不承诺完整 80% 复现 |
| gRPC 服务无鉴权导致数据泄露 | 中 | Phase-1 引入 HMAC token；Phase-2 引入 mTLS |

---

## 7. 实施计划范围（注意）

本设计文档是 **6 个月（3 阶段）蓝图**。首次进入实施计划阶段（`writing-plans` skill）时，**仅针对 Phase-1 (W1-W8 原型闭环)** 输出详细步骤；Phase-2 (W9-W16) 与 Phase-3 (W17-W24) 在 Phase-1 验收通过后，作为独立实施计划陆续展开，避免单次实施计划过大。

---

## 8. 与现有昇腾组件的关系

| 已有组件 | 是否复用 | 复用方式 |
|---------|---------|---------|
| ascend-docker-image/ascend-pytorch | ✅ 复用 | 基础镜像（含 CANN + torch_npu） |
| ascend-docker-image/mindie | ❌ 不复用 | MindIE 偏向 LLM 推理，V-JEPA 走 vLLM-Ascend 插件 |
| ascend-docker-image/ascendbase-toolkit | ✅ 复用 | Ascend 基础工具链 |
| MindInferenceService (MIS) | ❌ 不复用 | 用户明确 MIS 是 LLM 服务，与本项目解耦 |
| MultimodalSDK | ⚠️ 部分复用 | 若含视频解码/预处理组件可借鉴，不强制依赖 |
| AgentSDK | ⚠️ 间接 | 本项目产出可作为 AgentSDK 的一种世界模型能力 |

---

## 9. 验收标准

| # | 验收项 | 度量（具体定义） |
|---|--------|------|
| 1 | 训练子系统 W1-W2 跑通 | DDP LoRA 在 VideoMix22M 1000 子集上 100 步内 train loss 单调下降（前 50 步 ≤ 起动 +0.5），`model.print_trainable_parameters()` 输出 > 0 |
| 2 | 推理引擎 W3-W4 跑通 | encode 模式 P99 latency < 200ms（batch=1，bf16，8 卡），稳态 RPS > 50 |
| 3 | 端到端 smoke test W5-W6 | 训练→OM 编译→推理→评价闭环：训练 checkpoint 输出 bf16 features 与推理 OM 输出 bf16 features 在 `allclose(atol=1e-3)` 内通过 |
| 4 | V-JEPA 2-AC 复现 W7-W8 | **仅复现 AC 微调阶段**：latent action prediction loss 在 Droid 验证集上 ≤ Meta 公开值 × 1.2 倍（不要求完整 80% 复现） |
| 5 | MindSpeed 切换 W9-W10 | 训练吞吐 > DDP baseline 30%+（**Phase-2 W9 = NO-GO; deferred to W11-W12 quantization**） |
| 6 | W8A8 量化 W11-W12 | 模型 OM 大小压缩 ~50%，下游特征 cosine similarity 与 bf16 baseline ≥ 0.98 |
| 7 | WorldForge-RL PPO W13-W14 | Droid pick_place 任务成功率 ≥ 50%（100 episode 平均，meta 公开值 ~75%） |
| 8 | 自定义具身任务 W15-W16 | 与 DDP LoRA baseline 在 eval suite 上 ±2% 内 |
| 9 | V-JEPA 2.1 升级 W17-W18 | Dense Predictive Loss 在 VideoMix22M 1k 子集上 100 步内 loss 单调下降 |
| 10 | AD benchmark W19-W20 | CARLA Town01 直行场景闭环跑通：100 episode 平均 episode_return ≥ Meta GAIA-1 公开 baseline × 0.9 |
| 11 | Upstream PR W21-W22 | ≥3 个 PR merged（vjepa2 / vllm-ascend / verl 各 ≥1） |
| 12 | 文档化 W23-W24 | 用户手册 ≥ 50 页 + 完整 OpenAPI 文档 + 至少 1 段培训视频 |

---

## 10. 参考资料

### 10.1 上游代码与权重

- [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) — V-JEPA 2 / V-JEPA 2.1 / V-JEPA 2-AC 官方 PyTorch 代码
- [HuggingFace V-JEPA 2 docs](https://huggingface.co/docs/transformers/en/model_doc/vjepa2) — transformers.VJEPA2Model 集成
- [Introducing V-JEPA 2](https://ai.meta.com/research/vjepa) — Meta 官方介绍
- [V-JEPA 2 博客](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks) — 基准测试与基准

### 10.2 昇腾生态

- [vLLM Ascend Roadmap Q2 2025](https://github.com/vllm-project/vllm-ascend/issues/448) — vLLM-Ascend 路线图
- [vLLM-Omni NPU 2026 Q1 Roadmap](https://github.com/vllm-project/vllm-omni/issues/886) — Q1 已完成基础 NPU 适配
- [vLLM-Omni NPU 2026 Q2 Roadmap](https://github.com/vllm-project/vllm-omni/issues/2223) — Q2 计划 MindIE-SD 编译后端集成
- [SGLang NPU 2026 Q1 Roadmap](https://github.com/sgl-project/sglang/issues/13664) — SGLang 昇腾路线图
- [MindSpeed-LLM](https://gitee.com/ascend/MindSpeed-LLM) — 昇腾分布式 LLM 训练框架
- [MindSpore Official](https://www.mindspore.cn/en) — 原生深度学习框架
- [volcengine/verl](https://github.com/volcengine/verl) — RLHF/PPO 框架（含 Ascend NPU 后端）