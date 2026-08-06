# ADR-W40：帧生成路线选型 —— 轻量 decoder / 扩散 decoder / 换模型线

> 状态：提议（Proposed）
> 日期：2026-08-06
> 关联：`docs/roadmap/2026-08-03-jepa2-ascend-gap-analysis-roadmap.md`
> §Phase-5 帧生成前置决策点、`docs/superpowers/specs/2026-08-06-w36-npu-worker-parity-contract.md`
> §3.4（fast API 推理契约）、`docs/superpowers/specs/2026-08-06-w30-env-selection-adr.md`
> §5（W38 imagination rollout 需要 decoder 供视觉下游消费）。
> 决策范围：G13 ("无 latent→pixel 解码器 / 帧生成路径") 前置；
> 影响 W41（decoder 实现）、W42（`/v1/world/render` 端点）、W43（时序
> rollout 渲染）以及路线 B 专属的 W44（MindIE-SD 编译链路收口）。

---

## 0. TL;DR

**路线 A（轻量 pixel decoder）作为 Phase-5 v1 baseline**；路线 B（latent
条件扩散 decoder）作为 v2 升级路径，不阻塞 v1 上线；路线 C（换模型线）
作为 no-go 不予考虑。决策理由 = A 与 V-JEPA 2 当前代码基线距离最短、
训练/推理成本可控、可在 Atlas NPU 上直接复用 W36 同构契约；B 留给 W44
之后的 Phase-5 v2，前提是 MindIE-SD 真链路跑通 + 数据算力预算可承担。

---

## 1. 背景与问题

V-JEPA 2 原生不生成像素——其设计哲学即 "latent 空间预测、不生成像素"
（roadmap §2.5 I3）。WorldFoundry 4 象限之一是 **画面帧推理**（I3a/b）：
给定 latent 还原 / 生成 RGB 帧图。**当前全库无 latent→pixel 解码器、
无 pixel-frame 推理路径**（G13 阻断性差距，roadmap G13 表）。

roadmap §Phase-5 列出三个候选路线：
- **A**：冻结 encoder，训练轻量 pixel decoder（最近邻检索 + CNN/Transformer 上采样）——成本最低，画质有限
- **B**：latent 条件扩散解码器（latent → 扩散生成帧）——画质最高，训练/推理成本最大，可复用 MindIE-SD 工具链（G12 随之收口）
- **C**：换用/增挂原生像素预测世界模型（如 Genie 系）作为姊妹模型线——偏离当前代码基线最远

本决策回答：**Phase-5 v1 baseline decoder 走哪条？B / C 是否进 v1，
还是推迟？数据算力预算是否可承担？**

### 关键事实

- **当前 MindIE-SD 链路仅为占位**（`worldfoundry_inf/compile/mindie_sd.py:7-8`
  自述 "Phase-1 stub ... Real MindIE-SD integration lands in Phase-2 W3"
  —— Phase-2 W3 现在相位化为 Phase-5 W44）。`compile_to_om` 当前只
  写一个带 checksum 的 stub .om JSON 文件，不做真实 MindIE-SD 调用
  （同文件 50-65 行）。
- **VAE / diffusion model 代码全库无**：现有 `worldfoundry_inf/compile/`
  目录只有 `mindie_sd.py`（stub）+ `quantize.py`（W8A8）+ `__init__.py`；
  无 diffusion / VAE / decoder 模型代码。
- **W36 NPU 同构契约**：路线 A 的 decoder 走 torch nn.Module 即可走
  vjepa2_worker 的 cpu/npu 对照（与 W22 JEPA predictor 同形）；路线 B
  需要 vllm-omni / MindIE-SD OM compile 走 vLLM 引擎（vjepa2_worker
  的 HPA 路径），引入新的部署形态。
- **W38 imagination rollout**：W30 ADR §5 已约束 "MuJoCo 渲染保真
  度低于 SAPIEN/Genesis；W38 imagination rollout 的 visual fidelity
  上限受限于此"——但 W38 在 env 侧，已经是 last-mile 瓶颈。Decoder 的
  视觉保真度上限则在**模型侧**，与本 ADR 选型交叉。

### 与 roadmap §5 风险的对照

roadmap §5 列了 "帧生成超出 V-JEPA 2 原生设计 — Phase-5 范围膨胀 —
强制 W40 ADR 决策点；路线 A 作为最小可行方案兜底"。本 ADR 的 v1 默认
即依此选择 A，避免 Phase-5 范围膨胀。

---

## 2. 硬性约束

| # | 约束 | 来源 |
|---|---|---|
| C1 | Pixel 来自 V-JEPA 2 encoder latent（条件信号） | V-JEPA 2 原生设计；roadmap §1 目标形态 I3 |
| C2 | 推理必须能跑在 Atlas NPU（W36 同构契约） | W36 §3.4 通用门槛；`worldfoundry_common/npu_env.py:15-21` |
| C3 | 模型尺寸 + 推理延迟在 demo 任务内可接受（W39 端到端目标） | roadmap §4 Phase-4 出口判据 |
| C4 | 训练数据管线与 W22 + W27 视频流共基线 | W25-roadmap SFT schema；SFT 数据格式 schema |
| C5 | 验收指标可量化（PSNR / SSIM / FVD on real-domain） | W41 验收 |

---

## 3. 决策驱动因素

| 驱动因素 | 权重 | 说明 |
|---|---|---|
| D1 视觉保真度（real-domain PSNR / FVD） | ★★ | W38 imagination rollout + W39 demo 的视觉下限 |
| D2 训练成本（数据量 + NPU 日） | ★★★ | C4 + 算力预算；超过 ~6 NPU 月需要 Phase-5 重新评审 |
| D3 NPU 推理延迟（FPS sustained） | ★★★ | C3 + W39 demo |
| D4 与 V-JEPA 2 encoder 兼容度（冻结 encoder，单独 decoder） | ★★ | 避免重新预训练 encoder |
| D5 MindIE-SD / 现有工具链复用 | ★ | G12 收口是 nice-to-have，不是 v1 阻塞 |
| D6 部署复杂度（runtime deps / 权重大小） | ★★ | C6（项目分发许可证兼容）已隐含 |

---

## 4. 方案对比

| 维度 | A. 轻量 pixel decoder | B. Latent 条件扩散 decoder | C. 换 / 增挂模型线 |
|------|-----------------------|----------------------------|--------------------|
| D1 视觉保真 | 🟡 受限（最近邻 + CNN/Transformer 上采样；实域 PSNR ~ 22-25 dB；FVD 显著差于 B） | 🟢 高（latent DiT / Stable Diffusion 系；实域 PSNR ~ 28+ dB；FVD 显著低） | 🟢 高（取决于选定 sister 模型） |
| D2 训练成本 | 🟢 低：1×NPU 月级（数据：V-JEPA 2 训练视频帧 ↔ encoder latent 配对） | 🟠 高：≥ 6×NPU 月（视频帧 ↔ latent 数据集 ≥ 1M clips；diffusion model 300M-1B 参数） | 🔴 高（同 B + 完整预训练 4-8 周） |
| D3 NPU 推理延迟 | 🟢 单帧 < 30 ms（224×224）— 单次 CNN/Transformer 前向 | 🟠 单帧 ≥ 200 ms（4-step DDIM，NPU 上 MindIE-SD OM 编译后） | 🟠 取决于模型，单帧 100-300 ms |
| D4 兼容 V-JEPA 2 encoder | 🟢 冻结 encoder，latent 作为 conditioning | 🟢 latent 作为 diffusion 条件信号（LDM / DiT 风格）；V-JEPA 2 latent 直接接 LDM 类模型 | 🔴 架构错位；encoder 需重新训练或绕过；V-JEPA 2 设计哲学被打断 |
| D5 MindIE-SD 复用 | ❌（不上 diffusion） | 🟢 完整复用 W44 MindIE-SD 编译链路；同步关闭 G12 | ❌（同） |
| D6 部署复杂度 | 🟢 单一 .pt 文件，~500MB-1GB（轻量 decoder）；runtime = `worldfoundry_inf/render_worker.py` 复用 W36 worker | 🟡 多文件（diffusion + VAE + encoder）；需要 MindIE-SD OM compile 链路；runtime = vllm-omni engine | 🔴 新增 whole model line 部署栈 |
| 与 W25-roadmap SFT 兼容 | 🟢 复用 SFT 数据 schema（视频帧 + latent 配对） | 🟢 同一 schema | 🔴 不同 schema |
| W36 NPU 同构契约 | 🟢 直接复用 vjepa2_worker pattern | 🟡 走 vllm-omni / W37+ 新 worker 类 | 🟡 取决于 sister 模型 |
| 数据量估算 | 🟢 ~10K 视频 clips（百万帧） | 🟠 ≥ 100K 视频 clips（千万帧级） | 🟠 / 🔴 取决于 sister 模型 |
| **Phase-5 v1 推荐度** | **推荐** 🟢 | 推迟到 v2 | no-go |

---

## 5. 决策

**Phase-5 v1 = 路线 A（轻量 pixel decoder）**；**路线 B 在 Phase-5 v2 评估**（前提：W44 MindIE-SD 真链路可用 + 数据算力预算批准）；**路线 C 不予考虑**。

### 5.1 决策理由（按权重分层）

1. **D2 训练成本决定**（★★★）：A 的 1×NPU 月级数据算力预算可控，与 W22
   / W24 / W25-roadmap 已 ship 的训练栈同步运行；B 需 ≥ 6×NPU 月，
   **超过 Phase-5 v1 预算上限**——把 B 推到 v2，等 Atlas 资源富余 + W44
   链路验证后单独评审。
2. **D3 NPU 推理延迟决定**（★★★）：A 单帧 < 30 ms 可以撑起实时 / 接
   近实时的 demo（W39 frame-in-frame-out）；B ≥ 200 ms 难以做实时
   demo，更接近 "生成式" 而非 "实时" 用例。
3. **D4 兼容决定**（★★）：A 与 V-JEPA 2 现状天然兼容——冻结 encoder，
   latent 作为 conditioning 是 LDM/DiT 系范式外的"轻量版本"。C 强
   制重新设计 encoder，违反 "最小破坏" 原则。
4. **D1 视觉保真度接受成本**：A 在 real-domain PSNR ~ 22-25 dB，对 W38
   imagination rollout 的视觉评估足够（imagination rollout 重在
   **latent 距离度量**，非像素对比——可接受）。
5. **D5 MindIE-SD 复用**：本 ADR 不收口 G12，因为 v1 不上 diffusion；
   G12 的 W44 关闭依赖 B-v2 的具体上线时机。
6. **D6 部署复杂度**：A 的运行时形态完全复用 W36 同构契约（vjepa2_worker
   加一个 `decode(latent)` 方法）；B 需要新建 vllm-omni worker
   类，扩大 W36 契约。

### 5.2 与路线 B 的因果绑定

路线 B 进 v2 必须满足**前置条件**（W41+ v2 启动前一次性评估）：
- B1: MindIE-SD 真编译链路跑通（W44 关闭 G12；产出真 `.om` 而非占位）
- B2: W26 W8A8 路径在 diffusion model 上做精度回归通过（bf16 ↔ W8A8
  误差在阈值内）
- B3: 数据算力预算重新评审通过（≥ 6×NPU 月，且不挤占 Phase-4 RL 训练）

### 5.3 与路线 C 的否定理由

C（换/增挂 sister 模型）的核心问题是 **架构错位**：
- V-JEPA 2 训练完成的 encoder + EMA + predictor 体系（W22+W24+W27+v2
  chain）全部失效或退化为"姊妹模型之一"；
- WorldFoundry 当前不在生产状态，引入 sister 模型是 **第四象限的
  Phase-6+ 项目** 而非 Phase-5 路线；
- roadmap §5 已将 sister 模型系列为 Phase-5+ 评估，不阻塞本 ADR。

### 5.4 Decoder 数据约定

v1 decoder 训练数据 = {视频帧，encoder latent} 配对（与 W25-roadmap
SFT schema 子集兼容）。数据规模：
- 输入：~10K 视频 clips（百万帧）+ V-JEPA 2 encoder 离线 batch 推
  出的 latent
- 标签：原始 RGB 帧
- Split：train 90% / val 5% / test 5%（视频级别不重叠）
- 数据管线：复用 `worldfoundry_train/data/video_source.py` + W22 ckpt
  加载，仅新增 latent-cache 中间层

---

## 6. 实现路径

```
W40 (本 ADR)
  ↓ approve
W41: 视频帧 ↔ latent 配对数据管线 + 轻量 decoder 实现
  ├─ model/latent_decoder.py (新): Decoder = ConvTranspose 上采样 + 残差块
  ├─ data/latent_video_pair.py (新): JSONL manifest schema
  ├─ tests/test_latent_decoder.py (新): forward shape / dtype / 有限性
  ├─ scripts/w41_decoder_smoke.sh (新): 单 clip 5-step training + checkpoint
  └─ 验收: PSNR ≥ 20 dB on held-out frames (route A target floor)
W42: /v1/world/render 端点 (latent → frames; HMAC / adapter / batch 复用现有骨架)
  ├─ worldfoundry_inf/api/render.py (新): FastAPI 端点
  ├─ worldfoundry_inf/render_worker.py (新): CPU worker (W36 同构对镜)
  └─ 验收: API 契约测试 + smoke
W43: 时序 rollout 渲染 (predictor 递推 latent → decoder 逐帧渲染)
  ├─ worldfoundry_inf/rollout_render.py (新): rollout + render 闭环
  └─ 验收: 16-frame rollout render < 500 ms (route A target)
W44 (option, B-v2 only): MindIE-SD OM 编译链路收口 + NPU 推理优化
   (not in v1; activated only if B-v2 trigger condition met)
W45+: 把 W38 imagination rollout demo 接入 v1 decoder
```

---

## 7. 验证门（W40 + W41–W43 执行清单）

| 步骤 | 内容 | 通过标准 |
|---|---|---|
| W40.1 | ADR 评审通过（本文档） | chad sign-off；track `1e3db44`-style commit |
| W41.1 | 轻量 decoder forward shape | `(B, N_patches, encoder_dim)` → `(B, T, 3, H, W)`；bf16 dtype preservation |
| W41.2 | 单 clip 5-step training smoke | loss 单调下降；checkpoint 字节加载 |
| W41.3 | held-out frame PSNR | PSNR ≥ 20 dB on V-JEPA 2 训练集分布外视频（route A target floor） |
| W42.1 | `/v1/world/render` API 契约 | 与 `/encode` / `/predict` / `/plan` 共 HMAC；batch 字段一致 |
| W43.1 | 16-frame rollout render | 单帧渲染 < 30 ms（route A）；rollout 总 < 500 ms |
| W43.2 | 长时序 drift 评估 | L2 drift on encoder latent consistency per-step ≤ baseline + 5% |

### B-v2 触发条件（v1 上线后才评估）

- 路线 B1: MindIE-SD 真链路 + W44 验证通过
- 路线 B2: B-v2 数据算力预算批准
- 路线 B3: V-JEPA 2 latent 作为 diffusion 条件的 conditioning 兼容性报告（W41/W43 latent cache 可复用）

---

## 8. 影响与连锁效应

**正面**:
- G13 入口路径就位（route A 立即可行；不需要 MindIE-SD 真链路）
- 训练成本可控（≤ 1 NPU 月，可与 Phase-4 RL 训练共享 Atlas 窗口）
- W36 同构契约只新增 decoder 一个方法（`decode(latent)`），不像 B
  那样引入 vllm-omni 新 stack
- SFT 数据 schema 兼容性达成（v1 decoder 训练数据 = W25-roadmap
  schema 子集 + latent cache 中间层）

**负面 / 接受成本**:
- Route A 视觉保真度上限受限于 CNN/Transformer 上采样的真实域表
  现；商业级 VR / 长程精细任务不在 v1 支持范围
- MindIE-SD 链路（G12）的真实落地被推迟到 B-v2；项目维护一个
  占位 stub 跨 Phase-5
- V-JEPA 2 论文中"未来帧图像展示"的视觉 demo 需 V1 内部 commit 失
  望门槛上接受

**对路线图的影响**:
- W41 必须在 W22 training 先跑出 ckpt 之后启动（C4 数据依赖）
- W42 `/v1/world/render` 端点新增，与 W33 `/v1/world/plan` 并列
- W43 长程 rollout 受 W44 onboarding 延后，但 v1 deliverable 不依赖
- W44 收口时机落到 Phase-5 v2 评审窗口；不是 v1 阻塞项

---

## 9. 开放问题

1. **Route A 的具体 decoder 架构**：本 ADR 不锁定具体超参（W41 评估
   时定）。初步选项：
   - A1: ConvTranspose + ResBlock stack（最快出原型）
   - A2: ViT-style patch-decode + bilinear upsample（更适合 long-tail）
   - A3: nearest-neighbor patch retrieval + tiny refinement CNN（最低
     参数量，但视觉上限更低）
2. **Decoder 是否需要时序建模**：本 ADR 默认 decoder 单帧独立（latent
   + patch grid → RGB frame）；W43 时序 rollout 由 predictor 提供 latent
   序列，decoder 仅做单帧去噪。**不引入 temporal-aware decoder**，原因
   = 增加训练复杂度且不显著改善 PSNR（待 W41 数据验证）。
3. **B-v2 数据集规模**：真要上 B 时，10K → ≥ 100K clips 扩 10×，
   数据采集时间 ≥ 4 周；需提前规划 SLA
4. **C 是否真不予考虑**：若 V-JEPA 2 latent 不适合作为 diffusion 条件
   （B 跑通时发现），C 可能需要重新评估。这是"事后清单"，不是本 ADR 阻塞项
5. **MindIE-SD 占位 stub**：是否在 Phase-5 v1 期间予以 deprecation（删除
   而非保留 G12 入口），还是保留至 B-v2。决策：保留至 B-v2（减少项
   目整理工作量）

---

## 10. ADR 决策日志

| 日期 | 决策 | 状态 |
|---|---|---|
| 2026-08-06 | Phase-5 v1 = Route A 轻量 pixel decoder | 🟡 提议（本 ADR） |
| 2026-08-06 | Phase-5 v2 = Route B latent-conditional diffusion（pending W44 + 算力预算） | 🟡 提议 |
| 2026-08-06 | Phase-5 C = no-go | 🟡 提议 |

---

## 附录：信息来源

- Roadmap G13 来源: `docs/roadmap/2026-08-03-jepa2-ascend-gap-analysis-roadmap.md:137, 196-200`
- Roadmap Phase-5 路线 A/B/C 段: 同文件 §4 Phase-5 块引文
- MindIE-SD stub: `worldfoundry_inf/compile/mindie_sd.py:1-8, 50-65`
- W36 NPU 同构契约: `docs/superpowers/specs/2026-08-06-w36-npu-worker-parity-contract.md` §3.4 通用门槛
- W30 env-selection ADR (imagination rollout visual 约束): `docs/superpowers/specs/2026-08-06-w30-env-selection-adr.md` §5
- W22 JEPA 设计: `docs/superpowers/specs/2026-08-03-w22-jepa-predictor-ema-design.md`
- W25-roadmap SFT 数据 schema: `docs/superpowers/specs/2026-08-06-w25-full-finetune-sft-schema.md`
- V-JEPA 2 native pixel generation 设计: roadmap §2.5 I3
- Stable Diffusion / Latent Diffusion Model (LDM) 范式: Rombach et al. 2022 (CVPR)
- DiT (Diffusion Transformer): Peebles & Xie 2023 (ICCV)
- Latent Diffusion 在 NPU 上的现状 (CANN / MindIE-SD): 2025-2026 工具链 roadmap
