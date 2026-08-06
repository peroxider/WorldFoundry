# W35.2 — RLinf v0.3 版本矩阵核对（CPU-side 分析）

> 日期：2026-08-06
> 类型：W35 ADR 验证门 CPU-only 部分（姊妹文档 `docs/superpowers/specs/2026-08-04-w35-rl-framework-adr.md` §6 W35.2 行）
> 输入：`spikes/w35-3-rlinf/pip-freeze.txt`（RLinf v0.3.0 / WSL2 CPU 实测锁定） + `pyproject.toml`（项目锁文件）
> 输出：版本冲突识别、降级方案评估、W35.2 CPU 侧初步判定；NPU 端最终确认属于 W35.1

---

## 1. 背景与判定标准

### 1.1 验证门定义（W35 ADR §6 W35.2 行复述）

> **W35.2** 版本矩阵核对：RLinf v0.3 依赖的 torch_npu/CANN/vllm-ascend 与项目锁定（torch_npu 2.9.0 / CANN 8.5.1 / vllm-ascend 0.18.0）兼容
> **通过标准**：依赖冲突可解（含降级方案评估）
> **否决项**：需要 torch_npu 主版本变更

### 1.2 本文档的 CPU-side 范围

- ✅ 可在 CPU 侧完成：依赖清单对照、版本号冲突识别、降级路径文档化、官方/社区已知问题检索
- ❌ 需 NPU 实测：实际 torch_npu 2.9.0 + RLinf 0.3.0 + CANN 8.5.1 三件套的端到端跑通 → W35.1 范围

---

## 2. 项目锁文件 vs RLinf v0.3 实际依赖

### 2.1 项目侧（`pyproject.toml`）

| 包 | 版本 | 来源 |
|---|---|---|
| `torch_npu` | `==2.9.0` | `pyproject.toml:12` |
| `vllm` | `==0.18.0` | `pyproject.toml:29`（`inf` extra） |
| `vllm-ascend` | `==0.18.0` | `pyproject.toml:29` |
| `verl` | `==0.5.0` | `pyproject.toml:30`（`rl` extra） |
| CANN toolkit | `8.5.1` | W28 §3 复述（与 vllm-ascend 0.18.0 同矩阵） |
| Python | `>=3.10` | `pyproject.toml` |

### 2.2 RLinf v0.3 实际拉入（spike pip-freeze 实测，2026-08-04）

| 包 | 版本 | 备注 |
|---|---|---|
| `rlinf` | `0.3.0` | pip 公开 wheel，与 GitHub `v0.3` tag `0505431` 一致 |
| `torch` | `2.10.0+cpu` | **PyTorch 主线**；NPU 侧需替换为 `torch_npu` 同主版本 |
| `ray` | `2.56.1` | 调度底座 |
| `huggingface_hub` | `1.26.0` | HF 模型中心客户端（pip-freeze 中**未含** `transformers` 自身——spike 使用 HF 模型通过 `huggingface_hub` + HF transformers 间接装载，spike 文档中提及的 transformers 4.57.6 未在 pip-freeze 中可见，需 W35.1 实测时复核） |
| `accelerate` | `1.14.0` | 与 transformers 配合（间接依赖，pip-freeze 可见） |
| `hydra-core` | `1.3.4` | RLinf 配置系统（取代 OmegaConf 直接依赖） |
| `omegaconf` | `2.3.1` | Hydra 间接依赖 |
| `torchdata` | `0.11.0` | 数据加载 |
| `triton` | `3.7.1` | 编译器内核（NPU 侧通常禁用） |

> **注**：vllm-ascend / Megatron / DeepSpeed / FlashAttention 等均**不在 RLinf 默认依赖**——RLinf v0.3 的 actor 走 HF `transformers` + FSDP；rollout 走 huggingface worker（不经推理引擎）——证据见 `spikes/w35-3-rlinf/integration-points-memo.md` c 节（"无 vLLM/SGLang 依赖 ⇒ VJEPA2 encoder 的前向（torch 原生）可直接在 rollout worker 内执行"）。

---

## 3. 冲突识别

### 3.1 主冲突：**torch 主版本不一致**

| 维度 | 项目侧 | RLinf v0.3 实际 | 风险 |
|---|---|---|---|
| PyTorch 主版本 | `torch_npu 2.9.0`（PyTorch 2.9.x + Ascend NPU 补丁） | `torch 2.10.0+cpu`（PyTorch 2.10.x） | **minor version 不匹配** |

**判定**：W35 ADR §6 否决项「需要 torch_npu 主版本变更」中"主版本"的精确含义存在歧义：

- **严解**（minor 变更即触发）→ **否决项触发**，应切 verl
- **宽解**（仅 major version 变更触发）→ **未触发**，继续评估降级方案

按 W35 ADR 文字"W35.2 否决项：需要 torch_npu 主版本变更"，结合代码评审惯例与 PyTorch 生态实践（minor version 跳变通常引入 FSDP / torch.compile / inductor 默认行为变化），本节取**严解**作为安全解读 —— 但需在 W35.1 NPU 实测后定夺。

### 3.2 次要冲突 / 风险

| 项 | 风险 | 备注 |
|---|---|---|
| `huggingface_hub` 1.26.0 vs V-JEPA 2 引入者 | 低 | 项目侧 W22 已用 `transformers.VJEPA2Model`（`huggingface_hub` ≥ 1.0 即可装载）；spike pip-freeze 仅有 `huggingface_hub`，未直接列出 `transformers` 自身，但 spike 文档（`spikes/w35-3-rlinf/PROGRESS.md` 环境事实行）声明 transformers 4.57.6，需 W35.1 实测时核对实际安装版本与 `transformers>=4.45` 兼容 |
| `accelerate` 1.14.0 vs 项目侧 | 低 | 项目未显式 pin accelerate；RLinf 自带版本一般可共存 |
| `hydra-core` 1.3.4 vs 项目侧 | 低 | 项目使用 dataclass 风格 YAML，未引入 hydra；可作为独立依赖共存 |
| `ray` 2.56.1 vs vLLM-Ascend | **中** | vLLM-Ascend 0.18.0 内部用 Ray 做 worker 协调；RLinf 也用 Ray。两套 Ray 进程不冲突，但端口需规划（已在 W28 §4.4 B 方案"调度器"实现形态提及） |
| `triton` 3.7.1 vs CANN | **中** | Triton 仅在 CUDA 路径使用；torch_npu 2.9.0 下 triton 通常禁用或被替换为 Ascend 图引擎。需 W35.1 实测确认无导入报错 |
| CANN 8.5.1 vs RLinf | **低** | RLinf v0.3 官方支持声明仅 2026-07（qbitai 报道），未列具体 CANN 版本——但 RLinf 与 CANN 不直接耦合（中间隔 torch_npu），所以风险由 torch_npu 决定 |

### 3.3 不存在冲突的项

| 包 | 说明 |
|---|---|
| `vllm-ascend 0.18.0` | RLinf 默认不使用；verl 才用 vllm-ascend |
| `verl 0.5.0` | 与 RLinf 同在 `rl` extra 内不现实（项目当前 OR 关系）；选择 RLinf 意味着 verl 不再使用 |
| `gymnasium` `0.29.1` | 项目侧未显式 pin；RLinf spike 用了 gymnasium CartPole-v1 兜底 |

---

## 4. 降级方案评估（按"严解否决项触发"假设）

### 方案 A：升级项目 torch_npu 到 2.10.0

| 步骤 | 工作量 | 风险 |
|---|---|---|
| 1. 验证 torch_npu 2.10.0 + CANN 8.5.1 + vllm-ascend 0.18.0 三方兼容矩阵 | **高**：CANN 8.5.1 适配 torch_npu 2.10 的 wheel 是否发布未核实 | 高（vllm-ascend 0.18.0 通常 pin 特定 torch_npu 版本） |
| 2. `pyproject.toml` 改 pin torch_npu 2.10.0 | 低 | — |
| 3. W20/W21/W26/W27 重新适配 | 中（如果原 .to("npu") / FSDP / 量化加载器对 torch_npu minor 敏感） | 中 |
| 4. Atlas smoke 重跑 | 中（Atlas 时间窗口消耗） | — |

**判定**：方案 A 等价于"为 RLinf 改写整个项目依赖底座"，工期与回归成本远超 RLinf 自身收益（RLinf 是 G6/G7/G14 三件合一的项目，但如果它让 Phase-3 / W20-W29 全部回归测试，就是净亏）。

### 方案 B：降级 RLinf 到支持 torch_npu 2.9.0 的旧版本

| 步骤 | 工作量 | 风险 |
|---|---|---|
| 1. 查 RLinf v0.2.x release notes 是否声明 torch_npu 2.9 兼容 | 中（公开 release notes） | 不确定（RLinf v0.3 才宣布 NPU 支持） |
| 2. 若 v0.2.x 兼容，pip install rlinf==0.2.x 并复跑 spike 的 CartPole 基线 | 中 | RLinf v0.3 才引入 `RLINF_EXT_MODULE` 通道 + 权重同步成熟方案；v0.2.x 可能缺乏自定义模型通道（W35.3 通过条件之一） |
| 3. 若 v0.2.x 不支持，**本方案不可行** | — | — |

**判定**：方案 B 取决于 RLinf v0.2.x 是否携带 `RLINF_EXT_MODULE`（PR #920）。若缺失则 W35.3 验证不成立；若存在但 API 形式变化，则 spike 成果部分失效。需进一步核实。

### 方案 C：patch RLinf 源码以支持 torch_npu 2.9.0

| 步骤 | 工作量 | 风险 |
|---|---|---|
| 1. 在 RLinf 源码层 monkeypatch torch 版本检查（如有） | 低 | 可能不存在版本检查 |
| 2. 修复 torch 2.10 → 2.9 之间的 API 差异（`torch.compile` 默认 / `torch.distributed.fsdp` 状态等） | 高 | 不可知差异数量 |
| 3. 在 NPU 上跑 CartPole + VJEPA2Policy 双模式 spike | 中（与 W35.1 合并） | — |

**判定**：方案 C 工作量与不确定性最高；只有当方案 A/B 均不可行时才考虑。

### 方案 D：切 verl（ADR 既有备选路径）

- verl 已在 `pyproject.toml:30` 锁定，与项目 torch_npu 2.9.0 + vllm-ascend 0.18.0 + CANN 8.5.1 **同矩阵**
- ADR §4 已写明："verl 保留为备选触发路径：若验证门发现 RLinf 昇腾支持不成立（§6 任一否决项触发），切换 verl"
- **代价**：G6（仿真环境接入）回到自建路径（W31 范围不变），但失去 RLinf 的 14+ 仿真器生态；D4 训推编排优势也消失（verl 的 FSDP2 + vLLM-Ascend 协同仍可用，但失去 RLinf 的混合执行模式 2.434× 加速）
- **工期影响**：Phase-4 工期回到原估算（ADR §5 估算"中"），不再享有"低-中"红利

---

## 5. W35.2 CPU 侧初步判定

### 5.1 主冲突结论

**torch_npu 2.9.0 (项目侧) vs torch 2.10.0 (RLinf v0.3 实际) 构成 minor version 不匹配**。按严解（minor 变更触发否决项）解读，**W35.2 否决项被触发**；按宽解（仅 major 变更触发），未触发但需走方案 A 验证。

### 5.2 推荐路径

**接受否决项触发，切 verl**。理由：

1. **零额外工期风险**：verl 已在项目锁文件内，与 torch_npu 2.9.0 / vllm-ascend 0.18.0 同矩阵，CPU-only 即可确认无需 NPU 实测（**W35.2 CPU 部分已可定案**）
2. **W35.3 spike 投入不浪费**：模式 A/B 双通、CartPole 收敛、自定义模型通道验证等 spike 资产仍可作为 verl 自定义 worker 设计的参考输入（ADR §7.3 已埋伏笔："verl 备选路径下，是否引入 stable-worldmodel 补环境层？"）
3. **G6/G14 失去的 RLinf 红利由 verl 的 FSDP2 + Fully Async + vllm-ascend 部分弥补**（ADR §3 D4 行已记载：verl FSDP2+vLLM 天然协同）
4. **方案 A 工期与回归成本不可接受**（见 §4 方案 A 判定）
5. **方案 B/C 验证成本高，结果不确定**

### 5.3 NPU 侧最终确认（W35.1 兜底）

即使 CPU 侧建议切 verl，**W35.1 仍需执行一次**作为 "RLinf 不在 NPU 上跑通" 的最终判定记录：

- 目的：将本判定从"CPU-only 推断"升级为"NPU 实测结果"，避免未来被质疑
- 工作量：单卡 Atlas 启动 RLinf docker 镜像 + ManiSkill CPU 模式 PPO + 报错即终止
- 兜底结论：若 W35.1 因 torch_npu 版本问题开箱即崩 → 验证 CPU 推断；W35.2 判定升级为 "CPU + NPU 双确认，切 verl 决议成立"

---

## 6. 后续动作

1. **更新 W35 ADR §6 W35.2 行**：写入本文档 hash 与 CPU 侧判定；状态从"待执行"改为"CPU 侧完成，建议触发否决项切 verl"
2. **更新 phase-4 计划文档** `docs/dev/next-steps-phase4-atlas.md`：
   - 候选 #7 标注 "W35.2 CPU 侧完成 / W35.1 待 Atlas"
   - 跟踪表 #7 行更新 hash
3. **W35 ADR §4 决策路径调整**：将"RLinf 为主"改为"verl 为 RLinf 备选"的执行顺序——这等同于执行 ADR §4 第 3 项的备选触发路径
4. **W37 范围影响**：原计划"W37 = RLinf 与 worldfoundry 集成" → 现"verl 与 worldfoundry 集成"，接口契约基本不变（custom worker 设计哲学类似，spike 资产可复用）
5. **W35.1（Atlas smoke）排入硬件窗口优先级**：作为对 RLinf 路径的"封口验证"，避免未来重新评估

---

## 7. 信息来源

- 项目锁：`pyproject.toml:12,29-30`
- RLinf v0.3 实际依赖：`spikes/w35-3-rlinf/pip-freeze.txt`
- W35 ADR：`docs/superpowers/specs/2026-08-04-w35-rl-framework-adr.md`
- W35.3 spike 集成点纪要：`spikes/w35-3-rlinf/integration-points-memo.md`
- W35.3 spike 成本评估：`spikes/w35-3-rlinf/integration-cost-assessment.md`
- W28 编排设计（D7/D9/D10 决策）：`docs/superpowers/specs/2026-08-06-w28-train-inference-orchestration-design.md`