# ADR-W35：仿真 RL 训练框架选型 —— verl vs RLinf vs 自研

> 状态：提议（Proposed）
> 日期：2026-08-04
> 关联：`docs/roadmap/2026-08-03-jepa2-ascend-gap-analysis-roadmap.md`（Phase-4 W30–W39）、`docs/superpowers/specs/2026-08-03-w22-jepa-predictor-ema-design.md`
> 决策范围：Phase-4 仿真环境 RL 训练（G6/G7/G14）采用哪个 RL 框架底座

---

## 1. 背景与问题

WorldFoundry Phase-4 需要在昇腾 NPU 上建立"以 V-JEPA 2 encoder + AC predictor 为 policy 底座"的仿真环境 RL 训练能力。当前 `worldfoundry_rl/` 为纯桩（113 行，无梯度更新、无环境、无算法；`train_ac.py:37-43`），`pyproject.toml:29` 声明了 `verl==0.5.0` 依赖但全库零集成。

本决策回答一个问题：**Phase-4 的 RL 训练底座是自研、集成 verl，还是集成 RLinf？**

### 硬性约束

| # | 约束 | 来源 |
|---|---|---|
| C1 | 训练/推理必须运行于昇腾 NPU（Atlas 800 A2 / 910B×8；CANN 8.5.1 + torch_npu 2.9.0 + vllm-ascend 0.18.0） | `worldfoundry_common/npu_env.py:15-21` |
| C2 | Policy 底座是自定义架构（HF `VJEPA2Model` encoder + 自研 AC predictor，W22/W32），不是标准 LLM/VLA | `worldfoundry_rl/actor.py:29-40`、W22 设计文档 |
| C3 | 需要真实仿真环境接入（MuJoCo/SAPIEN/CARLA 级，G6） | 路线图 W30/W31 |
| C4 | 需要 PPO 起步、GRPO 可扩展（G7） | 路线图 W34/W35 |
| C5 | 训推一体编排（rollout 推理与 actor 训练的协同调度，G14） | 路线图 W28/W37 |
| C6 | JEPA 隐变量路线（latent 预测 + 规划）无现成框架先例，任何方案都需自研该部分 | W22/W33 |

### 关键生态事实（2026-08 调研，均经核实）

**verl 的昇腾支持是一方深度适配，且版本矩阵与 WorldFoundry 完全对齐：**
- 昇腾团队原生合入 verl 社区：支持 **verl FSDP2 原生后端** + 新增 **MindSpeed-MM NPU 加速后端**（昇腾社区公告，2026-04）
- verl 官方文档设 **Ascend 专章**（`verl.readthedocs.io/en/latest/ascend_tutorial/`），支持硬件含 Atlas 900 A2 POD / Atlas 800T A3 / 950 系列；2025-12 起 NPU 设备自动识别（[verl Ascend Install Guidance](https://verl.readthedocs.io/en/latest/ascend_tutorial/get_start/install_guidance.html)，2026-05-20 更新）
- **版本对齐**：verl 昇腾配套为 vllm-ascend 0.18.0 + torch_npu 2.9.0——与 WorldFoundry 锁定矩阵逐项一致
- **Fully Async 特性已在昇腾验证可用**（main + v0.7.1 分支，性能提升 1.6–2.0×，2026-05）；Qwen2.5-32B DAPO、Qwen3.5 GRPO 有公开 NPU 调优实践
- 局限：以 LLM/VLM reasoning RL 为主战场；多模态 omni RL 在昇腾仍处 RFC 阶段（[verl-omni #245](https://github.com/verl-project/verl-omni/issues/245)）；无具身仿真器生态

**RLinf 是具身 RL 基础设施，仿真器生态是决定性优势，但昇腾支持极新：**
- 14+ 仿真器开箱（ManiSkill/LIBERO/RoboTwin/IsaacLab/BEHAVIOR/Genesis…）、15+ RL 算法、VLA/世界模型（OpenSora、Wan）接入先例、WoVR（世界模型作为 VLA 后训练模拟器）论文与实现
- 架构契合 C5：混合执行模式（rollout/训练细粒度流水）吞吐达同类框架 2.434×；v0.3 含 Reward Model 组件、SGLang 推理服务化、解耦环境执行组件
- **昇腾支持为 v0.3（2026-07）新增特性**（"全面支持 Ascend CANN/torch-npu"），华为云 CloudRobo 合作确认"原生支持昇腾 NPU 后端"；但无公开 NPU 端到端性能/收敛报告，成熟度未经验证
- 局限：GPU 仿真器（IsaacLab/ManiSkill GPU/Genesis）物理依赖 NVIDIA——仿真节点大概率仍需 GPU/CPU，形成"GPU 仿真 + NPU 训练"异构混部；Custom Model 通道仅有 MLP/CNN-Policy 先例

**自研**：W34 已排期最小 PPO 自研实现（正确性验证基线）；完整自研需重做调度、环境接入、算法验证体系。

## 2. 决策驱动因素（权重）

| 驱动因素 | 权重 | 说明 |
|---|---|---|
| D1 昇腾 NPU 支持成熟度 | ★★★ | C1 是硬约束；未验证的支持等于不支持 |
| D2 仿真环境覆盖 | ★★★ | 直接决定 G6 工作量；自建 EnvPool 是 Phase-4 最大工程量 |
| D3 自定义模型（JEPA policy）接入成本 | ★★★ | C2/C6 意味着无论选谁都要写自定义 worker，接口友好度是关键 |
| D4 训推一体编排能力 | ★★ | 对应 G14/C5 |
| D5 与现有代码/依赖的兼容性 | ★★ | 版本矩阵、pyproject 既有声明 |
| D6 社区可持续性与风险分散 | ★ | 锁定单一框架的退出成本 |

## 3. 方案对比

| 维度 | A. verl | B. RLinf | C. 自研 |
|---|---|---|---|
| D1 昇腾成熟度 | 🟢 **一方适配、文档/镜像/调优实践齐全**；版本矩阵与项目逐项对齐；Fully Async 已验证 | 🟡 官方宣称支持（v0.3，2026-07），华为云合作背书；**无公开 NPU 性能报告** | 🟢 完全可控（但一切自己验证） |
| D2 仿真环境 | 🔴 无具身仿真器生态，G6 仍需自建 | 🟢 **14+ 仿真器开箱**，直接消解 G6 大半工作量；WoVR 提供"世界模型×RL"范式先例 | 🔴 全部自建（W30/W31 原估算不变） |
| D3 JEPA policy 接入 | 🟡 worker 模型接入面向 LLM/VLM 设计；video-latent policy 需自定义 actor/rollout worker | 🟡 自定义模型通道存在但先例浅（MLP/CNN）；VLA 接入模式可借鉴 | 🟢 原生设计 |
| D4 训推编排 | 🟢 Fully Async、训推混合流水线（FSDP2+vLLM 天然协同） | 🟢 混合执行模式 2.434×、解耦环境执行、推理服务化 | 🔴 自建（W28 设计文档需落地为代码） |
| D5 兼容性 | 🟢 已在 `pyproject.toml:29` 声明依赖；vllm-ascend 版本对齐 | 🟡 需新增依赖栈（Ray/SGLang/Megatron 可选）；与现有 FastAPI 推理服务并存需集成设计 | 🟢 零新依赖 |
| D6 社区 | 🟢 字节/火山引擎背书，昇腾一方合入 | 🟡 清华/无问芯穹，4.4k stars 增长快，但项目年轻（2025-08 开源） | — |
| 估算投入（Phase-4 内） | 中：框架集成省，但 G6 环境接入仍全量 | **低-中**：若昇腾支持验证通过，G6/G7/G14 同时受益 | 高：全量自建 |

## 4. 决策

**采用"RLinf 为主、verl 为备、自研最小 PPO 为验证基线"的三层策略：**

1. **W34 不变**：自研最小 PPO 闭环先行（正确性验证 + 不依赖外部框架的回退路径）。这是任何方案下的公共前置。
2. **W35 改造为本选型流程**：先执行验证门（见 §6），**首选 RLinf** 作为 Phase-4 RL 底座——决定性理由是 D2（仿真器生态直接消解 G6，这是 Phase-4 最大工程量）与 D4（混合执行调度直接回答 G14）。
3. **verl 保留为备选触发路径**：若验证门发现 RLinf 昇腾支持不成立（§6 任一否决项触发），切换 verl——其昇腾适配最深（一方合入、版本对齐、Fully Async 已验证），代价是 G6 环境接入回到自建，Phase-4 工期按原估算。
4. **不选纯自研**：D2/D4 的自建成本远超框架集成成本；且 C6 已保证核心差异化（JEPA predictor/plan）在任何方案下都是自研，框架选型不影响项目的技术壁垒。

## 5. 影响与后果

**正面：**
- G6（环境接入）、G7（RL 算法）、G14（训推编排）三个 Gap 由同一框架决策收敛，Phase-4 工期结构从"串行自建"变为"集成+自研核心"
- 保留 verl 备选使昇腾风险有对冲；W34 自研 PPO 保证终极回退
- RLinf 的 WoVR/Wan 先例为 Phase-5（帧生成路线 B）提供额外验证路径

**负面 / 需接受的成本：**
- 引入 RLinf 依赖栈（Ray 调度、SGLang/vLLM 推理后端），与现有 FastAPI 推理服务形成两套 serving 体系，需在 W37 做集成设计
- "GPU 仿真 + NPU 训练"异构混部成为标准部署形态，集群规划需在 W30 ADR 中明确
- JEPA policy（encoder + AC predictor）接入 RLinf 需自研自定义 worker（预计 Phase-4 内 2–3 周当量），无先例可循

**对路线图的变更：**
- W35 由"verl 集成评估与迁移"改为"RL 框架选型验证（本 ADR §6）"
- W30 ADR 增加议题：异构混部集群形态（仿真节点 GPU/CPU 选型）
- W37 范围扩大：RLinf（或 verl）与 worldfoundry 训练栈/推理服务的集成编排

## 6. 验证门（W35 执行清单）

| 步骤 | 内容 | 通过标准 | 否决项（触发切 verl） |
|---|---|---|---|
| W35.1 | RLinf 昇腾环境 smoke：Atlas 单卡跑通官方 NPU quickstart（ManiSkill CPU 模式 + MLP-Policy PPO） | 训练收敛曲线正常，无 NPU 算子报错 | 官方 NPU 路径无法跑通且无社区响应 |
| W35.2 | 版本矩阵核对：RLinf v0.3 依赖的 torch_npu/CANN/vllm-ascend 与项目锁定（torch_npu 2.9.0 / CANN 8.5.1 / vllm-ascend 0.18.0）兼容 | 依赖冲突可解（含降级方案评估） | 需要 torch_npu 主版本变更 |
| W35.3 | 自定义模型 spike：将 V-JEPA 2 encoder + 线性 action head 封装为 RLinf custom policy，CPU 跑通 CartPole 级任务 | 接入路径打通，接口成本 ≤ 预估 | 自定义模型通道无法承载 ViT 级模型 |

> ✅ **W35.3 已完成（2026-08-04），判定通过**：官方 `RLINF_EXT_MODULE` 注册通道承载 ViT 级自定义模型无缺口；状态投影与渲染帧双模式均实证打通 rollout→actor 数据通路（模式 A reward 17.7→274.2）；actor→rollout 权重同步对自定义模型自动生效；实际工时 ~1 天，JEPA policy 正式接入修正估算 **3–5 人日**（远低于 2–3 周预估）。证据与 14 项接入缺口分类：`spikes/w35-3-rlinf/integration-cost-assessment.md`、`spikes/w35-3-rlinf/integration-points-memo.md`。 |
| W35.4 | 异构混部验证：GPU/CPU 仿真节点 + NPU 训练节点的 rollout-learner 通信 | 吞吐不低于同框架纯 GPU 基线的 60% | 通信架构不支持跨硬件混部 |

验证门预计 2–3 周（可压缩在 Atlas 硬件窗口内与 W20/W21 并行）。

## 7. 开放问题

1. RLinf 的昇腾支持是否覆盖 FSDP2 + vllm-ascend 组合，还是仅 torch_npu 原生路径？（W35.1/W35.2 核实）
2. RLinf 世界模型接入（OpenSora/Wan）在昇腾上是否已验证，还是仅 CUDA 路径？（影响 Phase-5 路线 B 先例的有效性）
3. verl 备选路径下，是否引入 stable-worldmodel 补环境层？（仅备选触发时展开）
4. RLinf 的 License（Apache-2.0）与项目分发模式兼容性确认（形式核查，预计无问题）

## 附录：信息来源

- verl Ascend 官方文档：https://verl.readthedocs.io/en/latest/ascend_tutorial/get_start/install_guidance.html（2026-05-20）
- 昇腾 × verl 合作公告：https://www.hiascend.com/activities/dynamic-news/677（2026-04-17）
- verl Fully Async 昇腾上线报道（2026-05-15）；Qwen2.5-32B DAPO 昇腾调优实践：https://www.hiascend.com/developer/techArticles/20260126-1
- RLinf 仓库与 README：https://github.com/RLinf/RLinf（v0.3，2026-07）
- RLinf v0.3 发布报道（昇腾支持声明）：https://www.qbitai.com/2026/07/451379.html（2026-07-16）
- 华为云 × RLinf 合作：https://bbs.huaweicloud.com/forum/thread-0212721509141483595-1-1.html（2026-05-25）
- WoVR 论文：arXiv:2602.13977
- verl-omni 昇腾 RFC：https://github.com/verl-project/verl-omni/issues/245（2026-07-08）
