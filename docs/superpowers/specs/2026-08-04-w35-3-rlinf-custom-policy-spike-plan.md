# W35.3 Spike 实现方案：V-JEPA 2 Policy 接入 RLinf Custom Model 通道

> 日期：2026-08-04
> 类型：Spike（技术验证，限时限产，代码不进主干）
> 关联：`docs/superpowers/specs/2026-08-04-w35-rl-framework-adr.md` §6 验证门 W35.3
> 目标：验证"自定义 ViT 级模型（V-JEPA 2 encoder + action head）能通过 RLinf 官方 custom model 注册通道接入 PPO 训练闭环"，为 ADR-W35 的 RLinf 首选决策提供第三项通过证据

---

## 1. Spike 目标与判定标准

### 1.1 要回答的问题（按优先级）

| # | 问题 | 对应 ADR 否决项 |
|---|---|---|
| Q1 | RLinf v0.3 的 custom model 注册通道（PR #920）能否承载 **ViT 级模型**（而非仅有先例的 MLP/CNN-Policy）？ | 「自定义模型通道无法承载 ViT 级模型」 |
| Q2 | 非标准 observation 预处理（视频帧序列 → encoder → latent → action）能否嵌入 RLinf 的 rollout/actor 数据通路？ | 同上 |
| Q3 | actor 训练与 rollout 推理之间的**权重同步**对自定义模型是否开箱可用（v0.3 权重同步支持 broadcast/增量/分桶/仅同步可训练参数）？ | 同上（隐性否决） |
| Q4 | 接入的工程当量是否 ≤ ADR 预估（2–3 周）？ | 投入超限则重估 |

### 1.2 通过标准（Definition of Done）

1. `VJEPA2Policy`（V-JEPA 2 encoder + 线性 action head + value head）注册进 RLinf 模型注册表，config 可指定加载
2. 在 EmbodiChain CartPole（或 gymnasium CartPole 兜底）上 PPO 训练 **reward 曲线明确上升**（≥ MLP 基线的收敛趋势，不要求同等样本效率）
3. CPU 环境可复现（spike 不依赖 Atlas 硬件窗口；NPU 验证属于 W35.1 范围）
4. 产出《接入成本评估》：实际改动点清单、接口文档缺失记录、工时对表 ADR 预估

### 1.3 非目标

- ❌ 不验证 NPU 路径（W35.1/W35.2 的职责）
- ❌ 不接入 W22 的真 JEPA predictor / EMA 权重（spike 用冻结 encoder + 随机初始化 head 即可）
- ❌ 不追求 reward 收敛到 SOTA；不优化吞吐
- ❌ 不碰 AC predictor / plan 链路（W32/W33 的范畴）

## 2. 已核实的接口事实（方案的地基）

来自 RLinf 仓库与官方文档（2026-08-04 核实）：

- **自定义模型注册机制存在**：v0.3 changelog 含 `feat: support custom model registration`（PR #920）与 `fix: add model registery in standalone scripts`（PR #1100）——注册通道是一等公民且在被维护
- **对照基线存在**：EmbodiChain CartPole 官方配方 `examples/embodiment/config/embodichain_ppo_cart_pole.yaml`（MLP + PPO），关键配置字段 `actor.model.obs_dim` / `actor.model.action_dim` / `actor.model.policy_setup`；e2e 测试位于 `tests/e2e_tests/embodied/`
- **运行拓扑**：Ray workers 三件套——actor（训练）、rollout（推理）、env（仿真）；EmbodiChain 走 Gym 风格接口，observation 由 `state_keys` 拼接为 `states` 张量
- **CI 先例**：EmbodiChain CartPole 已被 embodied e2e 覆盖，spike 的通过标准可对照官方 e2e 判定方式
- Docker 镜像 `rlinf/rlinf:agentic-rlinf0.3-embodichain` 开箱；CPU 路径需在 spike 第一步实测（官方配方标注 1 节点 4 GPU，CPU 可行性本身是待验证项，见风险 R1）

## 3. 总体设计

### 3.1 模型侧：`VJEPA2Policy`（新增，spike 代码）

```
observation (图像帧序列 或 状态向量)
    │
    ▼
┌─────────────────────────────────────────┐
│ ObservationAdapter                      │  ← 两种模式，见 §3.2
│  A) state → 线性投影 → 伪 token 序列      │
│  B) 渲染帧 → VJEPA2 视频预处理 → patches │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ VJEPA2Model encoder（冻结）              │  ← HF transformers，随机小配置（§3.3）
│  输出 last_hidden_state → mean pooling   │
└─────────────────────────────────────────┘
    │
    ├──────────────► Linear action head → 动作分布（CartPole: Gaussian/Tanh 2 维连续）
    └──────────────► Linear value head   → V(s)（PPO critic）
```

### 3.2 观测适配的两种模式（关键设计决策）

CartPole 原生观测是 4 维状态向量，而 V-JEPA 2 是视频编码器——**这个错配必须在 spike 内正面处理**，因为它正是 WorldFoundry 未来接入的真实形态：

| 模式 | 做法 | 验证什么 | 成本 |
|---|---|---|---|
| **A. 状态投影模式**（先做） | state → Linear(4→D) →  reshape 为伪 token 序列（或跳过 encoder，先只验证注册+训练闭环） | Q1/Q3：注册、权重同步、PPO 闭环 | 低，1–2 天 |
| **B. 渲染帧模式**（后做） | EmbodiChain/gymnasium 渲染 RGB 帧，堆叠 K=2 帧 → V-JEPA 2 预处理 → encoder → latent → head | Q2：真实数据通路（图像观测 → encoder） | 中，2–3 天 |

模式 A 先打通"注册 + PPO + 权重同步"的机制问题；模式 B 才回答真实场景的数据通路问题。**两个模式都过，Q1–Q3 才算全部回答。**

### 3.3 encoder 规格：随机初始化的小配置，不下载 1.2B 权重

- spike 用 `transformers.VJEPA2Config` 构造**小规模随机初始化** encoder（如 hidden=384、layers=4），而非 `facebook/vjepa2-vitl-1.2b`
- 理由：spike 验证的是**接入机制**不是表征质量；1.2B 在 CPU 上前向过慢会拖垮 PPO 迭代速度；随机权重下 reward 仍能上升（CartPole 足够简单，随机特征 + 可训练 head 可学）
- 若模式 B 收敛不佳，允许 fallback：encoder 也解冻参与训练（此时验证的仍是机制，不是最终架构）

### 3.4 与 RLinf 的集成点（待 spike 第一步读码确认精确接口）

基于 PR #920 的注册机制，预期改动面：

| 集成点 | 预期动作 | 确认来源 |
|---|---|---|
| 模型注册表 | 注册 `VJEPA2Policy` 类（模型名 → 类映射） | PR #920 diff、`rlinf/` 下 model registry 模块 |
| actor worker | 确认自定义模型的 forward 契约（输入字段、返回分布/值的协议） | MLP-Policy 实现（EmbodiChain 配方所用类） |
| rollout worker | 确认自定义模型走哪个推理路径（MLP 先例应不经 vLLM/SGLang，直接 actor 权重推理——待核实） | rollout worker 代码 + 权重同步机制（PR #997/#1051） |
| 配置 YAML | 仿 `embodichain_ppo_cart_pole.yaml` 新增 spike 配置：`actor.model` 指向注册名，obs/action 维度对齐 | 官方配方文件 |
| 观测通路 | 模式 B 需确认 env worker 返回图像观测的字段约定（EmbodiChain 是否支持 RGB 观测，否则用 gymnasium 兜底） | EmbodiChain gym_config + RLinf env worker 协议 |

## 4. 实施步骤（预估 5 个工作日，纯 CPU）

| 步骤 | 内容 | 产出 | 预估 |
|---|---|---|---|
| S1 | 环境就绪：RLinf v0.3 clone + Docker/venv；**复现官方 MLP+PPO CartPole 基线**（含 CPU 可行性实测，GPU 不可得时降级验证）；读码确认 §3.4 全部集成点 | 基线 reward 曲线 + 《集成点确认纪要》（修正本文档 §3.4） | 1 天 |
| S2 | 模式 A：`VJEPA2Policy` 骨架（state 投影 + 双 head）+ 注册 + spike YAML | PPO 闭环跑通（不要求收敛） | 1 天 |
| S3 | 模式 A 收敛验证：调参至 reward 曲线明确上升；验证 actor→rollout 权重同步生效（不同步则 reward 恒为初始水平） | 模式 A 通过证据 | 1 天 |
| S4 | 模式 B：渲染帧通路 + 小配置 VJEPA2 encoder；跑通并观察收敛趋势（允许冻结失败则解冻） | 模式 B 通过证据 | 1.5 天 |
| S5 | 收尾：《接入成本评估》（实际改动点、工时对表、接口文档缺口、NPU 迁移预判）→ 回填 ADR-W35 §6 W35.3 结论 | 评估报告 | 0.5 天 |

**回退预案**：若 S1 发现官方配方强依赖 GPU 且 CPU 不可行 → 改用 gymnasium CartPole（经典版）自建最小 env worker 接入，验证目标不变但需额外 1 天，并在评估报告中标注 RLinf CPU 开发体验缺口。

## 5. 风险与开放问题

| # | 风险/开放问题 | 应对 |
|---|---|---|
| R1 | 官方 EmbodiChain 配方标注 4 GPU，CPU 路径未文档化 | S1 第一步实测；回退预案见 §4 |
| R2 | 自定义模型的 rollout 推理路径不明（是否必须经 vLLM/SGLang——若是，ViT encoder 接入成本显著上升，可能触及否决项） | S1 读码确认；若强制走推理引擎，评估走 SGLang 自定义模型路径的当量并如实记录 |
| R3 | 权重同步机制（broadcast/增量/分桶）可能对自定义模型的 state_dict 结构有假设 | S3 用"reward 是否随训练更新"作为黑盒判据；必要时显式打印 rollout 权重哈希 |
| R4 | PR #920 注册机制文档化程度未知，可能需要直接读源码+提 issue | S1 预留；接口缺口写入评估报告（本身即 spike 交付物） |
| R5 | EmbodiChain 不支持 RGB 观测 | gymnasium CartPole-v1 渲染模式兜底（`render_mode="rgb_array"`） |
| R6 | spike 结论依赖 RLinf 版本：锁定 v0.3 release tag，不用 main 分支 | S1 记录精确 commit hash |

## 6. 交付物清单

1. spike 代码（独立目录/分支，**不进 WorldFoundry 主干**；含 `VJEPA2Policy`、spike YAML、运行脚本）
2. 基线与两种模式的 reward 曲线截图/数据
3. 《集成点确认纪要》（S1 产出，修正本文档 §3.4 的推测部分）
4. 《接入成本评估》（S5 产出，回填 ADR-W35 §6 W35.3 行结论：通过 / 触发否决项）

## 7. 与验证门其他步骤的协同

- W35.3 全程 CPU 单机，**可与 W35.1（Atlas NPU smoke）、W35.2（版本矩阵核对）完全并行**
- 若 W35.3 通过而 W35.1 否决 → 仍切 verl，但本 spike 产出的接口认知不浪费（verl 自定义 worker 设计哲学类似）
- spike 中发现的观测通路设计（§3.2 模式 B）将直接复用于 W32（AC latent-action head）的接口设计

## 附录：信息来源

- RLinf v0.3 release notes（PR #920 custom model registration、#1100 registry fix）：https://github.com/RLinf/RLinf/releases
- EmbodiChain CartPole 官方配方文档：https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/embodichain.html
- RLinf 安装与镜像：`rlinf/rlinf:agentic-rlinf0.3-embodichain`（见同上文档 Installation 节）
