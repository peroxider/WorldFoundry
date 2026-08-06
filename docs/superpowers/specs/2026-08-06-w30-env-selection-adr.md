# ADR-W30：Phase-4 仿真环境选型 —— MuJoCo vs SAPIEN vs CARLA

> 状态：提议（Proposed）
> 日期：2026-08-06
> 关联: `docs/roadmap/2026-08-03-jepa2-ascend-gap-analysis-roadmap.md`
> (Phase-4 W30–W39)、`docs/superpowers/specs/2026-08-04-w35-rl-framework-adr.md`
> (Phase-4 RL 框架底座)、`docs/superpowers/specs/2026-08-06-w28-train-inference-orchestration-design.md`
> (训推一体编排)。
> 决策范围: G6 ("无仿真环境接入") 前置；为 W31 EnvPool 抽象选型；为 W35
> 框架底座对接提供环境基线。

---

## 1. 背景与问题

WorldFoundry Phase-4 的目标是 "以 V-JEPA 2 encoder + AC predictor 为
policy 底座的仿真环境 RL 训练能力"。roadmap §0 概览中 T2a (仿真环境接
入) 是四象限差距最大的一块，且唯一前置决策点就是**选什么环境**。

当前 `worldfoundry_rl/env_droid.py:1-6` 仅 CSV 离线回放（自述 "no live
simulator；MuJoCo/SAPIEN lands in Phase-2 W14"）；Phase-2 W14 在 Phase-3
收口时已并入 W30 决策。RLinf（首选 Phase-4 RL 框架底座，见 W35 ADR §4）
已经把 14+ 仿真器接入即用——但**首选仿真器选型**必须先于 W35 框架集成，
否则 W31 EnvPool 抽象会绑死在错的接口上。

本决策回答一个问题：**Phase-4 v1 的 baseline 仿真环境是 MuJoCo、SAPIEN
还是 CARLA，其他候选（如 ManiSkill/LIBERO/IsaacLab）如何与 baseline
协处？**

### 硬性约束

| # | 约束 | 来源 |
|---|---|---|
| C1 | 训练在昇腾 NPU（Atlas 800 A2 / 910B×8；CANN 8.5.1 + torch_npu 2.9.0） | `worldfoundry_common/npu_env.py:15-21` |
| C2 | Policy 输入是 RGB 帧序列（V-JEPA 2 encoder，W22） | `worldfoundry_train/model/predictor.py`、W22 设计文档 |
| C3 | 仿真节点与 learner 节点分体（CPU sim + NPU learner） | W28 §RL 架构图；W35 ADR §6 W35.4 |
| C4 | 必须支持 headless 运行（集群节点无显示） | 部署形态 |
| C5 | 必须有可向量化接口（一个 CPU 节点并行 ≥ 4 env / process） | W31 EnvPool 抽象 |
| C6 | License 必须与项目分发兼容（Apache-2.0 / MIT / BSD-3 优先） | `pyproject.toml` 现有声明 |
| C7 | W35 RLinf（首选）已封装 14+ 仿真器可即用，不能引入 baseline 与 RLinf 不兼容的方案 | W35 ADR §1 决策 #2 |

### 关键生态事实（2026-08 调研）

**MuJoCo (DeepMind, 2017–) 与 MJX (2023–):**

- 物理引擎与仿真器，被 OpenAI Gym、DeepMind Control Suite 沿用；2022 起
  源码以 **Apache-2.0** 释出，二进制走官方下载协议（免学术注册）
- MJX (JAX + GPU) 提供批量并行，CPU 路径上 `mujoco-py` /
  `dm_control` 也支持批 rollout
- 任务谱：Adroit hand、FrankaKitchen、Hopper/HalfCheetah/Walker2d 等；
  classic-control 是世界模型 RL 的标准起步点
- 渲染：官方 `MuJoCo.RenderContextOffscreen` 支持 headless RGB-D
  输出；分辨率与相机内参可控
- 接入 RLinf：MuJoCo 已为 RLinf EnvPool 默认后端

**SAPIEN (UCSD/Rutgers, 2020–):**

- 关节物体 + 物理级接触仿真；底层 PyBullet + UE4 渲染
- 任务谱：FractalBlocks、PartNet Manipulation、ArticulatedObject；
  适合 RLBench 类桌面机械臂操作任务
- License：SAPIEN 自身 **BSD-3-Clause**；UE4 部分（SimReady / PartNet
  资产生态）分免费包，**商业分发需要授权**（实操中大多研究场景限定
  非商业使用）
- 渲染：基于 UE4，提供高保真 RGB-D + 法线 + 语义分割
- 接入 RLinf：已 wrap（ManiSkill / LIBERO 均是 SAPIEN 系后端）

**CARLA (Carla.org, 2017–):**

- 自动驾驶仿真（不是操作/导航 RL）
- 渲染：UE4/UE5，多相机 RGB + LiDAR + semantic seg
- License：服务端 **MIT**；内容资产 (town/weather) 走额外许可
- 部署：**GPU-heavy**（UE4 实例要求每 env 一个 GPU 或多 GPU 共享；
  headless 时也需要 GPU）
- 任务谱：CarlaLeaderboard、Route + Town scenarios、自动驾驶 end-to-end
- 与 WorldFoundry policy 焦点错位：WorldFoundry 是 first-person
  camera manipulation 的 world-model-as-policy，CARLA 是 third-person +
  multi-camera 驾驶规划，policy 输入形态差异较大

**其他候选（W31 + Phase-4 v2 可考虑）:**

- **ManiSkill (CUHK)**: SAPIEN 系桌面操作 benchmark；已 RLinf wrap；
  作为 baseline 的 *任务套件* 不另作仿真器选型
- **LIBERO (Univ Washington)**: Lifelong robot learning benchmark；
  SAPIEN 系；同上
- **IsaacLab (NVIDIA 2024–)**: GPU-only（NVIDIA 物理 + cuIEC）；
  Atlas CPU 节点拒绝（违反 C3）；**Phase-5+ 评估，不再 v1 baseline 之列**
- **Genesis (Stanford / MIT 2024–)**: GPU-only differentiable physics；
  同 C3 否决
- **BEHAVIOR-1K (Stanford 2022–)**: 长程 household tasks；CPU/GPU 均可；
  Phase-4 v2 候选，不与 v1 baseline 冲突

### 任务焦点与 policy 输入对齐

WorldFoundry 的 policy 焦点 = "V-JEPA 2 encoder 处理 RGB 帧序列 → latent
→ AC predictor 给出 action"。这与**第一人称 / ego-centric 操作型 RL**
（manipulation, navigation）天然契合，与**驾驶规划**（multi-camera
bird-view + LiDAR）存在形态错位。基于此，CARLA 不应作为 Phase-4 v1 的
baseline。

## 2. 决策驱动因素（权重）

| 驱动因素 | 权重 | 说明 |
|---|---|---|
| D1 第一人称操作焦点匹配 | ★★★ | C2 + policy 焦点；错位 → W38 imagination rollout 数据形态需重构 |
| D2 CPU 节点 headless 可行 | ★★★ | C3 + C4；IsaacLab/Genesis 直接出局 |
| D3 向量化吞吐（每节点 ≥ 4 env） | ★★★ | C5；rollout 吞吐直接决定 W37 learner 训练曲线收敛速度 |
| D4 渲染保真度（RGB-D + 多视角） | ★★ | C2 + W38 imagination rollout 需要多帧采样；不是 trivial 但也不是核心 |
| D5 RLinf 接入成熟度 | ★★ | C7；W35 已锁 RLinf；不另起 wrapper 节约整 Phase-4 工程量 |
| D6 License 与分发兼容 | ★★ | C6；BSD-3/Apache-2.0/MIT 优先 |
| D7 学习曲线与调试成本 | ★ | 团队上手时间 |

## 3. 方案对比

| 维度 | A. MuJoCo | B. SAPIEN | C. CARLA |
|------|-----------|-----------|----------|
| D1 焦点匹配 | 🟢 first-person 操作；Adroit / FrankaKitchen benchmark 完整 | 🟢 first-person 桌面操作；PartNet / FractalBlocks | 🔴 驾驶，错位 |
| D2 CPU headless | 🟢 `mujoco-py` + `dm_control`；MJX 是可选 GPU 加速但 CPU 完全可用 | 🟡 BSD SAPIEN 部分 CPU 运行，但 UE4 渲染需 GPU/X11/Egl；headless Linux 需 Xvfb/Egl config | 🔴 UE4 实例 GPU 强制；不可能纯 CPU sim 节点 |
| D3 向量化吞吐 | 🟢 MJX 32–64 env/CPU core；`dm_control` 批 rollout | 🟡 单实例并行难；通常 1 env/process | 🔴 每 town 1 实例，节点级 GPU 共享 |
| D4 渲染保真 | 🟡 中（off-screen render 可，但与 RGB 真实域差距较大） | 🟢 UE4 高保真 RGB-D + semantic + normal | 🟢 UE5 多相机 + LiDAR + semantic seg |
| D5 RLinf 接入 | 🟢 **EnvPool 默认后端**；社区任务全集 | 🟢 SAPIEN 系（ManiSkill / LIBERO）已 wrap | 🔴 RLinf 不 wrap CARLA（需要额外项目，社区先例在 Waymax / nuPlan） |
| D6 License | 🟢 Apache-2.0（2022+ 源码释出）+ 二进制下载 | 🟡 **BSD-3 source 自身免费；UE4 资产生态商业分发受限** | 🟡 MIT 服务端；town 资产自带商业许可 |
| D7 调试成本 | 🟢 团队已有 scipy/HF 背景，调试门槛低 | 🟡 UE4 集成复杂度高 | 🟡 UE5 集成 + 多相机接口 |
| 估算投入 (Phase-4 内) | 低: W31 接入 ~½ session; EnvPool + FrankaKitchen demo ~1 session | 中: SAPIEN CPU 渲染管线约 1 session; demo 1 session | 高: GPU 异构节点管理 + 多 env/topology 重构 + Phase-4 v1 不必要 |

## 4. 决策

**Phase-4 v1 baseline = MuJoCo (CPU 路径)**，理由分层：

1. **D1 焦点决定** — MuJoCo 任务谱（Adroit / FrankaKitchen / dm_control
   manipulation）完全匹配 WorldFoundry 的 first-person 操作型 policy；
   CARLA 与 WorldFoundry 的 first-person-单相机焦点形态错位。
2. **D2/D3 拓扑决定** — W28 §RL 架构图定义 "CPU sim + NPU learner" 分体
   形态；MuJoCo (CPU) 在该拓扑里天然适配；CARLA (UE4 GPU-heavy) 必
   须引入第二类 sim 节点 (GPU sim)，超出 W28 设计基线（**不强制否
   决，但 Phase-4 v1 不必承担集群形态扩张成本**）。
3. **D5 RLinf 集成决定** — W35 ADR §4 已锁 RLinf；MuJoCo 是 RLinf
   EnvPool 的**默认后端**，无需额外 wrapper；SAPIEN 需走 ManiSkill/
   LIBERO wrapper（额外集成单位，但**已在 RLinf 生态内**）；CARLA 与
   RLinf 不兼容（社区先例 Waymax/nuPlan，不在 RLinf v0.3 范围）。
4. **D6/D7 收敛** — MuJoCo source 2022 后是 Apache-2.0，分发零障碍；
   SAPIEN source BSD-3 但 UE4 资产生态商用受限，**保守起见不入 baseline**。

### baseline 配套选型

- **首选任务套件**：FrankaKitchen（dm_control 系；4 sub-tasks + 7
  compound；操作 benchmark 基线）
- **次选任务套件**：Adroit Hand（4 sub-tasks；dexterous 复杂操作）
- **Phase-4 v2 扩展**：SAPIEN 系的 ManiSkill / LIBERO 入 v2，作为长
  horizon / 大场景实验；不与 v1 baseline 冲突
- **CARLA 留在 Phase-4 之后的 driving 题材料储备**；不在 Phase-4 v1
  baseline 之列

### 仿真集群分体架构

```
┌───────────────────────────────────────┐    gRPC (rollout data)   ┌────────────────────────────────────────┐
│  CPU sim 节点 × N                     │   ─────────────────────► │   NPU learner 节点 (Atlas 800 A2 / 910B×8)│
│                                       │                          │                                        │
│  - MuJoCo CPU 仿真实例 (mj_env.so)    │   ◄───────────────────── │  - worldfoundry_rl PPO/GRPO loop         │
│  - 多 env/process (向量化 ≥ 4)        │   gRPC (policy weights)  │  - V-JEPA 2 encoder + AC predictor       │
│  - W31 EnvPool 包装 `gym.Env` 接口    │                          │  - worldfoundry_inf FastAPI (推理服务化)  │
│  - 渲染: offscreen RGB (224×224)      │                          │  - W36 reward model 接口                 │
└───────────────────────────────────────┘                          └────────────────────────────────────────┘
```

- **sim → learner 流量**: rollout (state, action, reward, next_state)
  经 `worldfoundry_rl/env_pool.py` (W31) 序列化为 Protobuf or msgpack
  通过 gRPC 推送到 learner；单 frame 大小 ~ 数十 KB（RGB 224×224 ≈
  150KB / 帧的 compressed tensor）。
- **learner → sim 流量**: policy weights 经 `worldfoundry_inf/` server
  的 HMAC + adapter 名签后下发；sim 节点 reload adapter 后再开新批次
  rollout（避免每 step 跨节点，详见 W28 §weight hot-update）。
- **scale**: 单 CPU 节点 ≥ 32 parallel env（MuJoCo CPU 单 instance
  256–1024 env 已实证），NPU learner 节点 1× 即可消费；Phase-4 v1
  起步 = 1 sim 节点 + 1 learner 节点。

## 5. 影响与后果

**正面**:

- G6 / G7 / G14 三个 Gap 与 W35 RLinf 决策收敛，sim 选型不引入新框架分歧
- cluster 形态保持 W28 设计基线（CPU sim + NPU learner），不扩张为
  GPU-sim 三类节点
- RLinf MuJoCo 路径已被社区深度试用，减少 Phase-4 调试循环数
- FrankaKitchen + Adroit 任务套件足以支持 W38 imagination rollout 对
  照实验（4+ sub-tasks × 2 套件 = ≥ 8 任务）

**负面 / 接受成本**:

- 不优先 CARLA / driving 题：若 roadmap 后期转向自动驾驶，需要 Phase-5
  重新选型，并付出新工程量
- MuJoCo 渲染保真度低于 SAPIEN/Genesis；W38 imagination rollout 的
  visual fidelity 上限受限于此。如有商业级画质需求需 Phase-4 v2+ 评估
- SAPIEN 受 UE4 资产生态 license 限制，本 ADR 暂不放进 baseline；
  若团队确有 SAPIEN 需求，需 W31 之前先做 license 论证

**对路线图的影响**:

- W31 EnvPool 抽象的接口签名**以 MuJoCo API 为 baseline 抽出**，SAPIEN
  / ManiSkill / LIBERO 各做 adapter；CARLA 等 Phase-4 v2+ 任务加入
- W34 self-PPO smoke 改用 MuJoCo + 一个简易 FrankaKitchen / CartPole
  级任务（MuJoCo CPU 路径 5–10 分钟即可跑通）
- W35.4 "GPU/CPU sim + NPU learner 异构混部" 验证门在 Phase-4 v1
  退化为 **"CPU sim + NPU learner" 单一验证**，因为本 ADR 选 CPU sim
- W37 learner 调度只需管理 NPU 节点，sim 节点由独立进程池维系

## 6. 验证门（W30 执行清单）

| 步骤 | 内容 | 通过标准 |
|---|---|---|
| W30.1 | MuJoCo CPU 路径 smoke：单 Python process 内 4 parallel env 跑 FrankaKitchen (dm_control) 1000 steps | step 吞吐 ≥ 200 step/s (单 CPU core) |
| W30.2 | W31 EnvPool 接口框架：基于 MuJoCo 抽象 `reset/step/render/close`，CPU 单节点 32 env batched | batch step 吞吐 ≥ 4× 单 env；throughput 入 `bench_results/w30/vec_env_muJoCo.json` |
| W30.3 | NPU learner smoke: W34 self-PPO + MuJoCo + FrankaKitchen 跑 1000 steps，policy 权重从 learner 回灌到 sim 节点 | reward 曲线单调上升；rollout 路径 RLinf 不必经 |
| W30.4 | W36 reward 接口用 MuJoCo FrankaKitchen task-specific reward 接入 | reward signal 数值合理（与 dm_control 官方 RL 训练对照 ± 5%） |

(W30 本身不直接执行 W31/W34/W36；本表列出后续任务依赖 MuJoCo 决策的
关键 gate，便于 Phase-4 实施期间回归验收。)

## 7. 开放问题

1. **MuJoCo 二进制下载协议的合规性**：Apache-2.0 source 释放后，官
   方二进制仍走 https://mujoco.org/ 下载；CI 镜像需要 pre-download；
   是否引入 pre-baked Docker layer (W31+ 准备)
2. **FrankaKitchen 的 State vector 形态**：dm_control 任务的 state
   通常是 proprioceptive + object poses；WorldFoundry 的 VJEPA policy
   默认 input 仅 RGB。如 W34 smoke 走 state-only，需明确 policy 同
   时消费 RGB（"visual policy"）还是 state-only（"state policy"）
3. **SAPIEN 商业 license 论证**：若 Phase-4 v2 需要 SAPIEN，需提交
   UE4 资产生态 license 论证文档，**不与本 ADR 决策冲突但需预留 paper trail**
4. **CARLA 是否降级为 "Phase-5 候选评估"**：若 roadmap 调整方向，转
   driving 题，本 ADR 决策需重审

## 附录：信息来源

- MuJoCo / MJX: <https://mujoco.org/> ; DeepMind Control Suite GitHub repo; OpenAI Gym MuJoCo env set
- SAPIEN: <https://sapien.ucsd.edu/> ; UCSD CVS lab GitHub
- CARLA: <https://carla.org/> ; CarlaLeaderboard GitHub
- ManiSkill (SAPIEN 系 / RLinf wrap): <https://github.com/haosulab/ManiSkill>
- LIBERO (SAPIEN 系 / RLinf wrap): <https://github.com/Lifelong-Robot-Learning/LIBERO>
- IsaacLab: <https://isaac-sim.github.io/IsaacLab/> (Phase-5+ 评估)
- Genesis: <https://genesis-embodied-ai.github.io/> (Phase-5+ 评估)
- BEHAVIOR-1K: <https://behavior.stanford.edu/> (Phase-4 v2 候选)
- RLinf 仿真器生态: W35 ADR §1 `RLinf` 段; <https://github.com/RLinf/RLinf>
- W28 编排设计: `docs/superpowers/specs/2026-08-06-w28-train-inference-orchestration-design.md`
- W35 框架 ADR: `docs/superpowers/specs/2026-08-04-w35-rl-framework-adr.md`
- W36 NPU parity contract: `docs/superpowers/specs/2026-08-06-w36-npu-worker-parity-contract.md`
- Roadmap G6 来源: `docs/roadmap/2026-08-03-jepa2-ascend-gap-analysis-roadmap.md:131, 173`
