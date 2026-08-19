# JEPA2 × 昇腾 NPU 训推一体：差距分析与分阶段实施路线图

> 文档日期：2026-08-03
> 基线代码：main @ f212646（`feat(train): 实现完整的FSDP分布式训练支持和检查点功能`）
> 目标形态：JEPA2 世界模型在昇腾 NPU（Atlas 800 A2 / 910B×8）上训推一体
> 训练域：LoRA/全参微调 + 仿真环境强化学习
> 推理域：隐变量推理 + 实际画面帧推理

---

## 0. 摘要（TL;DR）

WorldFoundry 当前是 **"昇腾上的 V-JEPA 2 编码器微调 + 特征服务平台" Phase-1 原型**（部分 Phase-2 W9–W19 已落地）。昇腾工程骨架扎实（版本锁定、HCCL、vLLM-Ascend 插件、量化链路），微调在 CPU 路径完整闭环；但对照"JEPA2 训推一体"四象限目标：

| 能力象限 | 现状评级 | 一句话 |
|---|---|---|
| 昇腾 NPU 底座 | 🟡 骨架就绪，链路未闭合 | 分布式原语齐备，但训练主链路无 `.to("npu")`，端到端未上机 |
| 训练 · 微调 | 🟡 CPU 闭环，数学目标是代理 | 训练循环真实；损失是自蒸馏代理，缺 predictor + EMA target encoder |
| 训练 · 仿真 RL | 🔴 纯桩 | 113 行占位代码，无环境、无算法、无梯度更新，基本从零 |
| 推理 · 隐变量 | 🟡 API 骨架完整，语义浅 | encode 可用；predict/plan 是浅层代理或 passthrough |
| 推理 · 画面帧 | 🔴 设计中不存在 | 无解码器/VAE/扩散/渲染路径，需超出 V-JEPA 2 原生设计的新增投入 |

**总体判断：距离完整目标约 2~3 个 Phase 的工作量。** 其中"帧生成"不是渐进补齐，而是一条新模型线（latent→pixel decoder），需要单独立项决策。

---

## 1. 目标形态定义

为避免范围漂移，先定义"JEPA2 × 昇腾训推一体"的完整能力矩阵：

```
训练侧                          推理侧
├─ T1 微调                       ├─ I1 隐变量编码（encode 当前帧 → latent）
│   ├─ T1a LoRA 微调             ├─ I2 隐变量预测（过去帧 latent → 未来帧 latent）
│   ├─ T1b 全参微调              │   ├─ I2a 无动作条件（predictor）
│   └─ T1c JEPA 目标训练          │   └─ I2b 动作条件（AC predictor / plan）
│       （encoder-predictor      └─ I3 画面帧推理（latent → pixel frames）
│        + EMA target encoder）       ├─ I3a 单帧解码
└─ T2 仿真环境 RL 训练                └─ I3b 时序帧生成（rollout 渲染）
    ├─ T2a 仿真环境接入（MuJoCo/SAPIEN/CARLA）
    ├─ T2b RL 算法（PPO/GRPO，verl 集成）
    ├─ T2c latent-action head（世界模型作为 policy 底座）
    └─ T2d reward model / 任务奖励

横切：H1 NPU 端到端训练链路 · H2 NPU 推理上机验证 · H3 训推一体资源编排（同集群训练/推理混部）
```

---

## 2. 现状盘点（按代码证据）

### 2.1 昇腾 NPU 底座 —— 🟡 骨架就绪

| 组件 | 状态 | 证据 |
|---|---|---|
| 版本锁定 | ✅ CANN 8.5.1 + torch_npu 2.9.0 + vllm-ascend 0.18.0 + transformers 4.55.0 | `worldfoundry_common/npu_env.py:15-21`、`docker/worldfoundry-base/Dockerfile:1-4` |
| Docker 镜像 | ✅ Ascend 基础镜像 | `docker/worldfoundry-base/Dockerfile:4` |
| 设备探测/分布式 | ✅ `is_npu_available()` / `init_distributed(backend="hccl")`，CPU 降级 gloo + warning | `worldfoundry_common/npu_env.py:63-103` |
| FSDP/DDP | ✅ HCCL 分支 + `torch.npu.set_device` | `worldfoundry_train/parallel/fsdp.py:42-50`、`parallel/ddp.py:27-29` |
| vLLM-Ascend 插件 | ⚠️ 结构完整，Atlas 上机验证 deferred（W19b） | `worldfoundry_inf/_vllm_ascend_plugin.py:37-71`、`vllm_ascend_worker.py:20-22` |
| 量化 | ⚠️ msmodelslim W8A8 后端 + W17 验证脚本齐备，需真实硬件执行；W4A8 deferred | `worldfoundry_inf/compile/quantize_backends/msmodelslim.py`、`scripts/archive/w17/w17_atlas_w8a8_eval.sh` |
| 数据解码 | ⚠️ DVPP 硬件解码为 stub，`_dvpp_available()` 恒 False，走 decord CPU 兜底 | `worldfoundry_train/data/video_source.py:16,38-44` |
| MindSpeed | ❌ W9 评估 NO-GO，`MMStrategy` 为 NotImplementedError 桩 | `docs/superpowers/specs/W9_EVAL_REPORT.md:5`、`worldfoundry_train/parallel/mindspeed.py:12-25` |
| MindIE-SD 编译 | ❌ `compile_to_om` 只写占位 .om | `worldfoundry_inf/compile/mindie_sd.py:1-8` |

**关键缺口（H1）**：`worldfoundry_train/trainer.py:129-144` 中模型经 `AutoModel.from_pretrained` 加载后**没有任何 `.to("npu")` 设备迁移**；现有训练测试全部走 CPU/gloo（trainer docstring 自述 "CPU-runnable"）。NPU 分布式原语齐了，端到端"模型上 NPU 训练"链路未闭合。

### 2.2 训练 · 微调 —— 🟡 CPU 闭环

已实现（真实代码，非 stub）：

- 完整训练循环：forward + loss + backward + step（`worldfoundry_train/training_loop.py:51-106`）
- LoRA：PEFT 包装 + G4 防静默零训练校验（匹配 0 模块/0 可训练参数直接 raise）（`worldfoundry_train/model/lora_wrapper.py:31-76`），target = q/k/v/o_proj + mlp.fc1/fc2（`lora_wrapper.py:17-20`）
- 分布式：DDP 默认；FSDP 完整（FULL_SHARD/HYBRID、bf16 混合精度、activation checkpointing、CPU offload）（`worldfoundry_train/parallel/fsdp.py:85-180`）
- Checkpoint：原子保存（tmp+rename），含 optimizer state/RNG/step/LoRA 签名，支持 resume + 签名校验（`worldfoundry_train/checkpoint.py`、`worldfoundry_train/trainer.py:146-211`）

缺口：

- **T1c JEPA 目标是数学代理**：损失为 masked-feature MSE，target 来自同一 encoder 的 no_grad 二次前向（自蒸馏代理）；docstring 明确生产目标应为 "EMA target encoder + cosine similarity"（`worldfoundry_train/training_loop.py:32-48`）。**没有独立 predictor 模块**，损失在数学上接近恒等回归，无法学到有意义的预测表征。
- **T1b 无全参微调入口**，只有 LoRA；无 SFT 数据格式定义。
- 多卡 FSDP checkpoint 保存 deferred 到 Phase-2（`worldfoundry_train/checkpoint.py:7`）。
- `bench_results/fsdp-baseline.json` 自述 "local smoke, not hardware baseline"（4332 clips/s 为 CPU 数字，非 Atlas 实测）。

### 2.3 训练 · 仿真 RL —— 🔴 纯桩

`worldfoundry_rl/` 三文件合计 113 行：

- `train_ac.py:37-43`："训练循环"仅遍历样本数并 print，**无梯度更新、无 PPO、无 reward**
- `env_droid.py:1-6`：`DroidEnv` 为 CSV 离线回放，自述 "no live simulator；MuJoCo/SAPIEN lands in Phase-2 W14"
- `actor.py:29-40`：`VJEPAActor.predict_latent_action` 为 passthrough，直接返回 encoder 特征（"full latent-action head lands in Phase-2 W7"）
- verl 0.5.0 已声明依赖（`pyproject.toml:29`）且 Dockerfile 已安装，但**全库零 import、零集成**
- 设计文档中 MuJoCo/SAPIEN/CARLA、PPO/GRPO、RewardModel、ActorPool/EnvPool/LearnerPool 均停留在架构图（`docs/superpowers/specs/2026-07-27-ascend-world-model-design.md:307-336`）

**结论：T2a–T2d 全部为空白，是四象限中差距最大的一块。**

### 2.4 推理 · 隐变量 —— 🟡 API 骨架完整

FastAPI 服务三端点，均有 HMAC 鉴权、adapter 路由、批处理骨架：

| 端点 | 现状 | 证据 |
|---|---|---|
| `POST /v1/world/encode` | ✅ 真实可用（video → dense features） | `worldfoundry_inf/api/encode.py`、`vjepa2_worker.py:38-48` |
| `POST /v1/world/predict` | ⚠️ 浅层代理：输出 encoder `last_hidden_state` 的 masked 重建；`loss_score = (pred**2).mean()` 形式分 | `worldfoundry_inf/api/predict.py` |
| `POST /v1/world/plan` | ⚠️ 纯 passthrough：返回 encode 特征 + 原样回传输入 actions；真实 AC predictor deferred 到 Phase-2 W4 | `vjepa2_worker.py:56-66`、`vllm_ascend_worker.py:164-172` |

缺口：

- **I2a/I2b 缺失**：无"给定过去帧预测未来帧 latent"的时序预测路径（predictor 模型本体不存在）
- vLLM-Ascend worker 未上机验证；W8A8 量化加载器 `_patch_quantized_loader` 为 stub（`_vllm_ascend_plugin.py:74-83`）

### 2.5 推理 · 画面帧 —— 🔴 设计中不存在

- 全库无解码器/VAE/扩散/渲染模型代码；三端点返回均为 features/actions，无像素输出路径
- V-JEPA 2 的设计哲学即"latent 空间预测、不生成像素"，项目忠实于此
- `mindie_sd.py` 提及 diffusion compile backend，但仅指工具链来源且为占位 stub（`worldfoundry_inf/compile/mindie_sd.py:1-8`）

**I3a/I3b 需要新增整条 latent→pixel decoder 模型线（如冻结 encoder + 训练 pixel decoder / 扩散解码器），属超出当前设计范围的新增投入，需单独技术选型决策。**

---

## 3. 差距清单（Gap Register）

| # | 差距 | 象限 | 严重度 | 现有证据 | 依赖 |
|---|---|---|---|---|---|
| G1 | 训练链路无 `.to("npu")`，Atlas 端到端训练未验证 | H1 | 🔴 阻断性 | `trainer.py:129-144` | Atlas 硬件窗口 |
| G2 | 损失为自蒸馏代理，缺 predictor + EMA target encoder | T1c | 🔴 核心算法 | `training_loop.py:32-48` | — |
| G3 | DVPP 硬件解码 stub，数据流水线只能 CPU decord | H1 | 🟠 性能 | `video_source.py:38-44` | CANN DVPP API |
| G4 | 多卡 FSDP checkpoint 保存 deferred | T1 | 🟠 | `checkpoint.py:7` | G1 |
| G5 | 无全参微调入口 / SFT 数据格式 | T1b | 🟡 | — | — |
| G6 | 无仿真环境接入（MuJoCo/SAPIEN/CARLA） | T2a | 🔴 从零 | `env_droid.py:1-6` | 环境选型决策 |
| G7 | 无 PPO/GRPO 训练器；verl 未集成 | T2b | 🔴 从零 | `train_ac.py:37-43`、`pyproject.toml:29` | G6 |
| G8 | 无 latent-action head / AC 模型 | T2c, I2b | 🔴 | `actor.py:29-40` | G2 |
| G9 | 无 reward model / 任务奖励定义 | T2d | 🟠 | — | G6 |
| G10 | predict/plan 端点为代理/passthrough，无时序 latent 预测 | I2 | 🔴 核心算法 | `vjepa2_worker.py:56-66` | G2 |
| G11 | vLLM-Ascend worker + W8A8 加载器未上机验证 | H2 | 🟠 | `vllm_ascend_worker.py:20-22`、`_vllm_ascend_plugin.py:74-83` | Atlas 硬件窗口 |
| G12 | MindIE-SD OM 编译为占位 | H2 | 🟡 | `mindie_sd.py:1-8` | — |
| G13 | 无 latent→pixel 解码器 / 帧生成路径 | I3 | 🔴 从零 + 超出现设计 | 全库无 decoder 代码 | **技术选型决策** |
| G14 | 训推一体资源编排（同集群混部、权重热更新）未设计 | H3 | 🟡 | — | G1, G11 |

严重度：🔴 阻断目标达成 / 🟠 显著影响性能或可用性 / 🟡 工程完善项

---

## 4. 分阶段实施路线图

延续项目现有 W 编号风格（W9–W19 已占用），建议新增 **Phase-3（W20–W29）** 与 **Phase-4（W30–W39）** 两个阶段。帧生成单列为 **Phase-5（W40+）**，需先完成立项决策。

### Phase-3：NPU 端到端闭环 + 真 JEPA 目标（W20–W29）

> 主题：让"真模型在真硬件上真训练"，并把损失函数从代理换成真 JEPA 目标。

| W# | 任务 | 覆盖 Gap | 验收标准 |
|---|---|---|---|
| W20 | 训练链路 NPU 迁移：`.to("npu")`、bf16 精度校验、HCCL 端到端 smoke | G1 | Atlas 单卡 LoRA 微调跑通 100 steps，loss 曲线正常下降 |
| W21 | 8 卡 FSDP 上机 + 多卡 checkpoint 保存 | G1, G4 | Atlas 8×NPU FSDP 跑通；checkpoint save/resume 签名校验通过；产出首个硬件 baseline（clips/s、MFU） |
| W22 | **JEPA predictor 模块实现**（独立 predictor 网络 + EMA target encoder + cosine/MSE 组合损失），替换自蒸馏代理 | G2 | 单测覆盖；与代理损失对照实验，下游 probing 指标提升 |
| W23 | DVPP 硬件视频解码落地，decord 降级为兜底 | G3 | DVPP 路径吞吐 ≥ CPU 路径 2×；`_dvpp_available()` 真实探测 |
| W24 | predict 端点接入真 predictor：过去 K 帧 latent → 未来 H 帧 latent | G10（I2a 部分） | `/v1/world/predict` 输出未来 latent + 预测置信度；单测 + API 契约测试 |
| W25 | 全参微调入口 + SFT 数据格式定义 | G5 | `full` / `lora` 双模式训练配置；SFT 数据 schema 文档化 |
| W26 | W8A8 量化加载器 `_patch_quantized_loader` 实现 + Atlas 上机 | G11（部分） | 量化模型加载推理，精度回归在阈值内 |
| W27 | vLLM-Ascend worker Atlas e2e 验证（W19b 收尾） | G11 | smoke 全绿，性能数字入库 `bench_results/` |
| W28 | 训推一体编排设计文档：同集群训练/推理混部、显存配额、权重热更新机制 | G14 | 设计文档评审通过 |
| W29 | Phase-3 集成回归：训练→导出→推理服务加载→predict 端点全链路 | G1–G10 回归 | CI 集成测试 + Atlas 周度回归 |

**Phase-3 出口判据**：Atlas 上 8 卡 FSDP 微调（真 JEPA 目标）产出 checkpoint，推理服务加载该 checkpoint 后 predict 端点输出真实未来 latent。

### Phase-4：仿真环境 RL 训练（W30–W39）

> 主题：从零建立"世界模型作为 policy 底座"的 RL 训练能力。

| W# | 任务 | 覆盖 Gap | 验收标准 |
|---|---|---|---|
| W30 | 仿真环境选型决策（MuJoCo vs SAPIEN vs CARLA）+ 仿真在 NPU 集群的部署形态（CPU 仿真节点 + NPU  learner 的分体架构） | G6 前置 | ADR 评审通过 |
| W31 | 环境接入层：EnvPool 抽象 + 首选环境的向量化封装（对接 `worldfoundry_rl`） | G6 | 向量化 env step 吞吐基准达标 |
| W32 | AC latent-action head 实现（替换 actor passthrough）：encoder latent + action → 下一 latent | G8（T2c 侧） | 单测 + 离线数据预训练收敛 |
| W33 | plan 端点接入 AC predictor（Phase-2 W4 遗留项收口） | G8, G10（I2b） | `/v1/world/plan` 返回动作条件下的真实预测 latent |
| W34 | PPO 训练器落地：rollout 收集 → advantage 估计 → 梯度更新闭环（先自研最小实现，验证正确性） | G7 | 仿真任务上 reward 曲线上升 |
| W35 | RL 框架选型验证（verl vs RLinf vs 自研，按 ADR-W35 执行验证门 W35.1–W35.4）：首选 RLinf（仿真器生态 + 训推编排），verl 为备选（昇腾一方适配最深），自研 PPO（W34）为基线 | G6, G7, G14 | 验证门通过，框架决策生效；详见 `docs/superpowers/specs/2026-08-04-w35-rl-framework-adr.md` |
| W36 | Reward model / 任务奖励接口定义（含从世界模型 latent 距离构造 intrinsic reward 的选项） | G9 | reward 接口文档 + 至少 1 个任务的 reward 实现 |
| W37 | RL 训练与主训练栈集成：共用 checkpoint/LoRA 体系、ActorPool/EnvPool/LearnerPool 编排（按设计文档 §RL 架构图落地） | G7, G14 | 分布式 RL 训练 smoke 通过 |
| W38 | 世界模型辅助 RL（model-based）：用 predictor 做 imagination rollout 降低真实仿真步数 | G8, G7 | 样本效率对照实验报告 |
| W39 | Phase-4 集成回归：encode→plan→RL 训练→policy 回灌推理服务全链路 | G6–G10 回归 | Atlas 端到端 demo 任务收敛 |

**Phase-4 出口判据**：在至少 1 个仿真任务上，以 V-JEPA 2 encoder + AC predictor 为底座的 RL 训练收敛，policy 可通过推理服务对外提供 plan 能力。

### Phase-5：画面帧推理（W40+，需先立项决策）

> ⚠️ **前置决策点**：V-JEPA 2 原生不生成像素。帧生成需新增 latent→pixel 解码模型线，三个候选路线需在 W40 完成选型：
>
> - **路线 A**：冻结 encoder，训练轻量 pixel decoder（最近邻检索 + CNN/Transformer 上采样）——成本最低，画质有限
> - **路线 B**：latent 条件扩散解码器（latent → 扩散生成帧）——画质最高，训练/推理成本最大，可复用 MindIE-SD 工具链（G12 随之收口）
> - **路线 C**：换用/增挂原生像素预测世界模型（如 Genie 系）作为姊妹模型线——偏离当前代码基线最远

| W# | 任务 | 覆盖 Gap | 验收标准 |
|---|---|---|---|
| W40 | 帧生成路线 ADR 选型（A/B/C）+ 数据与算力预算评估 | G13 前置 | ADR 评审通过 |
| W41 | 解码器模型实现与训练数据管线（视频帧 ↔ latent 配对数据） | G13 | 单帧解码 PSNR/SSIM 达路线目标 |
| W42 | `/v1/world/render` 端点：latent → 帧图序列（HMAC/adapter/批处理复用现有骨架） | G13 | API 契约测试通过 |
| W43 | 时序 rollout 渲染：predictor 递推 latent → decoder 逐帧渲染 | G13, G2 | 长时序渲染 demo，漂移评估报告 |
| W44 | （路线 B 专属）MindIE-SD OM 编译链路收口 + NPU 推理优化 | G12, G13 | 帧生成 NPU 延迟达标 |

### 里程碑视图

```
现在 ──────────────────────────────────────────────────────────►
     Phase-3 (W20–W29)            Phase-4 (W30–W39)        Phase-5 (W40+)
     NPU 闭环 + 真 JEPA 目标        仿真 RL 从零到收敛        帧生成（先选型）
     ├─ M3.1 Atlas 单卡训练跑通     ├─ M4.1 环境接入+AC head  ├─ M5.1 路线 ADR
     ├─ M3.2 8卡FSDP+真predictor   ├─ M4.2 PPO 闭环收敛      ├─ M5.2 单帧解码达标
     └─ M3.3 训→推全链路回归       └─ M4.3 训推一体RL demo   └─ M5.3 rollout 渲染
```

---

## 5. 风险与依赖

| 风险 | 影响 | 缓解 |
|---|---|---|
| Atlas 硬件窗口不可控 | G1/G11/G21 全部上机任务阻塞 | Phase-3 排序时 W20/W21 优先预约硬件；CPU 路径保持可测 |
| MindSpeed 已 NO-GO | 大规模并行只能靠 FSDP 自研路径 | W21 把 FSDP 硬件 baseline 做实；保留 Megatron-NPU 备选评估 |
| RL 从零开始，工作量被低估 | Phase-4 延期 | W30/W31 选型与接入先行；W34 先自研最小 PPO 验证正确性再迁 verl |
| 帧生成超出 V-JEPA 2 原生设计 | Phase-5 范围膨胀 | 强制 W40 ADR 决策点；路线 A 作为最小可行方案兜底 |
| predictor 训练效果不及自蒸馏代理 | G2 改了反而更差 | W22 要求对照实验 + probing 指标验收，不达标可回退开关 |
| 训推一体混部的显存/调度冲突 | 训练与推理互相干扰 | W28 设计先行，显存配额与权重热更新机制评审后再进 W37 |

---

## 6. 建议的立即行动（本周）

1. **W20 启动**：预约 Atlas 硬件窗口，训练链路 `.to("npu")` 迁移 + 单卡 smoke —— 这是所有后续工作的物理前提
2. **W22 启动**：JEPA predictor + EMA target encoder 设计评审（纯算法，不依赖硬件）
3. **W40 预研**：帧生成路线 A/B/C 的成本-画质对比材料准备，提前排 ADR 评审
4. **W30 预研**：仿真环境选型调研（MuJoCo/SAPIEN/CARLA 在国产算力集群的部署案例）

---

## 附录 A：证据文件索引

- 项目定位：`README.md:1-5`、`pyproject.toml:8`
- NPU 环境：`worldfoundry_common/npu_env.py:15-21,63-103`
- 训练：`worldfoundry_train/trainer.py:129-211`、`training_loop.py:32-106`、`model/lora_wrapper.py:17-76`、`parallel/fsdp.py:42-180`、`checkpoint.py`
- RL：`worldfoundry_rl/train_ac.py:37-43`、`env_droid.py:1-6`、`actor.py:29-40`
- 推理：`worldfoundry_inf/vjepa2_worker.py:38-66`、`vllm_ascend_worker.py:20-22,148,164-172`、`_vllm_ascend_plugin.py:37-83`、`api/encode.py`、`api/predict.py`
- 编译/量化：`worldfoundry_inf/compile/mindie_sd.py:1-8`、`compile/quantize_backends/msmodelslim.py`
- 数据：`worldfoundry_train/data/video_source.py:16,38-44`
- 设计文档：`docs/superpowers/specs/2026-07-27-ascend-world-model-design.md:198-201,307-336`、`docs/superpowers/specs/W9_EVAL_REPORT.md:5`
