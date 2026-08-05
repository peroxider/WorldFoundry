# W22 — 设计评审 addendum（2026-08-05）

**作者**：chad（设计评审）+ Claude（执行记录）
**基础文档**：`docs/superpowers/specs/2026-08-03-w22-jepa-predictor-ema-design.md`
**状态**：🟡 增量规范 — 与基础 spec 并行生效；每条均给出代码锚点或验收门槛
**关联**：W22.7（对照实验）、W22.6（FSDP 路径）、Phase-3 W20（Atlas 硬件窗口）

---

## 0. 背景

W22 主 spec（2026-08-03）由 Kimi 起草，2026-08-04 落地 W22.1–W22.6（167 → 213 tests）。本文是 **post-implementation 设计评审** 的输出：记录评审中发现的关键语义偏差、可执行的实验设计修订、以及落地时引入的小型改进。所有修订遵循"必须处理 / 应该处理 / 可延后"三级分级。

---

## 1. 关键语义偏差（评审结论）

### 1.1 偏差 #1 — HF 不消费 `bool_masked_pos` 导致 context/target 信息不对称弱于官方配方

**事实**：HF `VJEPA2Model.forward` 在 transformers 5.1.0 中通过 `**kwargs` 静默吞掉 `bool_masked_pos`（基础 spec §12-① 已记录，由 `tests/test_jepa_objective.py::test_hf_encoder_ignores_bool_masked_pos` 固化）。当前实现下：
- context encoder 看完整视频（仅靠 LoRA dropout 提供扰动）
- target encoder 看完整视频（deterministic，无 dropout）

不对称性退化为 "dropout 开/关 + EMA 权重差"，**弱于官方配法的"只可见 vs 全可见"信息不对称**。

**结论**：基础 spec 闭合了"True=mask vs True=predict"的歧义，但留下了**更深层的语义偏差**。

### 1.2 偏差 #2 — FSDP gather 瞬态开销比设计估的高

**事实**：FSDP1 `FULL_STATE_DICT` 会在每 rank 瞬态物化全模型（ViT-L bf16 ≈ 2.4GB/rank/EMA step），而非设计估计的 ~10MB LoRA-only allgather（基础 spec §12-② 已记录，`ema.py:156-161` 注释固化）。

**结论**：v1 接受开销换取正确性，缓解杆为 `ema_update_every > 1`。

### 1.3 偏差 #3 — LoRA-only EMA 信号天然偏弱

**事实**：backbone 冻结，LoRA adapters (~10MB) 是唯一被训练的 encoder 权重；$m=0.996$ 时时间常数仅 250 步；LoRA 漂移本身小 + EMA 平滑 → target ≈ online 差距有限。

**结论**：对照实验需选择**对信号强度敏感**的探针指标（见 §3）。

---

## 2. 实验设计修订（W22.7）

基础 spec §9.2 定义了 A/B 双臂对照实验。本 addendum **新增 C 臂** 并修订验收门槛。

### 2.1 新增 C 臂 — `mask_ratio=0` 健全性检查

| 臂 | 配置 | 期望结果 |
|----|------|----------|
| A（基线） | `jepa.enabled: false` | 训练 loss 曲线 |
| B（JEPA） | `jepa.enabled: true` | 训练 loss + 下游 probing |
| **C（健全性）** | **`jepa.enabled: true` + `data.mask_ratio: 0`** | **recon_cosine 应显著低于 B 臂（≈随机水平）；probing 指标应与 A 臂无显著差异** |

**目的**：
- C 臂充当"零遮挡 sanity check"：证明 mask 信号被真正学到。
- 若 C 臂的 `recon_cosine` 与 B 臂差距不显著，说明 mask 信号弱，**问题不在算法而在 HF 架构层面（偏差 #1）**，需启动上游 patch 方案。
- C 臂同时作为 B 臂的统计对照：剥离"encoder 拟合到训练分布"的贡献。

**实现**：复用 W22.7 的 `make_subset` / `run_arm` 骨架，仅新增一份 YAML 配置 `worldfoundry_train/configs/w22_7_arm_c_mask0.yaml`，并在 `scripts/w22_7_ab_experiment.sh` 中加入 `c` 臂循环。

### 2.2 修订验收门槛 — 差值优于绝对值

基础 spec §9.2 末尾给出的 `cosine ≥ 0.6` 是**任意绝对值**，对算法贡献的灵敏度不足。修订为：

| 指标 | 旧门槛（绝对值） | 新门槛（差值或相对） |
|------|-----------------|---------------------|
| masked_reconstruction_cosine | B 臂 ≥ 0.6 | **B − A ≥ 0.2**（绝对差）；且 **B − C ≥ 0.15**（证明 mask 信号有效） |
| linear_probe_accuracy | B > A | B − A 的 95% CI 下界 > 0（bootstrap） |
| token_variance（有效秩） | B 不衰减 | B / A 在 step 200 时 ≥ 0.85（防坍塌） |
| knn_hit_rate | B > A | B − A ≥ 0.1 |

**理由**：绝对值与初始化（偏差 §3.4，见 §4）、特征尺度（bias 等）耦合；差值直接反映算法相对增益。

### 2.3 NPU profile 报告门槛（偏差 #2 落实）

W22.7 在 Atlas 上跑时，**`throughput.json` 必须包含**：

| 字段 | 测量方式 | 门槛 |
|------|----------|------|
| `ema_gather_peak_mb` | `torch.npu.max_memory_allocated()` 在 EMA step 期间峰值 | 报告值；若 > 4GB 必须给出 `ema_update_every` 上调方案 |
| `ema_gather_s_avg` | `WORLDFORGE_JEPA_TIMING=1` 下 timing dict | < 50ms @ 8 卡 HCCL（设计估；超过则需切 `update_every=4`） |
| `target_forward_fraction` | `target_forward_s / loss_fn_s` | < 0.4（理论 ~25-33%） |
| `context_forward_s` | 同上 | 基线对照 |

**用途**：决定 W22.7 之后是否启动 FSDP2 局部 gather 替换（v2 优化项）。

---

## 3. M4 — step 0-100 初始化效应日志约定

基础 spec §4.1 + `predictor.py:101-103` 末层 MLP zero-init：训练初期 predictor 输出接近 0，**loss 起始 ≈ 0 是初始化效应而非收敛**。

**约定**（强制在 W22.7 报告中体现）：

1. `throughput.json` 增加字段 `warmup_window_loss: [step 0..100 的 loss 序列]`。
2. 实验报告 §结果 中必须有一节 "warm-up window analysis"：
   - 标注 B 臂 step 0-100 loss 曲线与 A 臂的差异来源（初始化 vs 算法贡献）
   - 提供 step 200 / 500 / 1000 / 2000 四个对照点的 loss 数值表
3. 若 B 臂 step 50 处的 loss < A 臂 step 50 处的 loss，但 step 1000 反过来，**不能解读为"B 臂长期更优"**——需分别报告初始化窗口后（即 step ≥ 200）的曲线对比。

---

## 4. checkpoint.py 修订 — 偏差 #4（已落地）

### 4.1 背景

基础 spec §7.4 定义 `JEPASignature` 校验架构字段；超参（`momentum_end`, `loss.beta`）被刻意排除并以 warning 处理。评审认为该决定**风险偏高**：
- `momentum_end` 改变 cosine schedule 终点 → 后续每个 EMA lerp 的 (1−m) 全部漂移
- `loss.beta` 改变 smooth-L1 Huber 阈值 → 梯度幅度不连续跳变
- 长程训练（>10k 步）跨 resume 时静默漂移代价远大于强制用户对齐

### 4.2 修订方案（已落地）

| 修订 | 文件 | 行号锚点 |
|------|------|----------|
| 新增 `JEPAHyperparams` dataclass | `worldfoundry_train/checkpoint.py` | `JEPAHyperparams` |
| 新增 `validate_jepa_hyperparams()` 函数 | `worldfoundry_train/checkpoint.py` | `validate_jepa_hyperparams` |
| `JEPAObjective.hyperparams()` 暴露当前值 | `worldfoundry_train/jepa_objective.py` | `JEPAObjective.hyperparams` |
| `JEPAObjective.state_dict()` 写入新键 | `worldfoundry_train/jepa_objective.py` | `state_dict` |
| `save_checkpoint` 序列化新键 | `worldfoundry_train/checkpoint.py` | `save_checkpoint` (jepa_objective 分支) |
| `load_checkpoint` ERROR 级校验新键 | `worldfoundry_train/checkpoint.py` | `load_checkpoint` (jepa_objective 分支) |

**向后兼容**：
- 旧 ckpt（无 `jepa_hyperparams` 键）→ `validate_jepa_hyperparams(None, current)` 直接返回 no-op
- 中间版本 ckpt（有键但缺字段）→ 每缺失一个字段 emit `UserWarning` 但继续；已存在但漂移的字段仍 ERROR
- 容差：1e-9（浮点序列化噪声）

### 4.3 测试覆盖

`tests/test_jepa_checkpoint.py` 新增 4 个测试：
- `test_jepa_hyperparam_mismatch_raises_value_error` — 单元覆盖 match/tolerance/no-op
- `test_jepa_hyperparam_missing_key_warns_only` — 单字段缺失走 warn 分支
- `test_load_checkpoint_with_mismatched_momentum_end_raises` — 端到端走 `save/load` 链路
- `test_load_checkpoint_legacy_partial_hyperparams_warns` — 模拟中间版本 ckpt

---

## 5. 实施层改进（已落地）

### 5.1 M1 — predictor 三轴插值测试覆盖

**文件**：`worldfoundry_train/tests/test_predictor.py`
**新增**：
- `test_t_axis_trilinear_interpolation_finite` — grid (1,4,4) → (3,4,4) 数值稳定性
- `test_h_w_axis_trilinear_interpolation_finite` — grid (1,4,4) → (1,6,8)
- `test_grid_size_unchanged_skips_interpolation` — fast-path 健全性

V-JEPA 2 典型 T=8/16，但 trilinear 在小 T 上可能退化（interpolate 退化到 nearest），测试至少保证有限值与正确形状。

### 5.2 M2 — predictor dtype cache

**文件**：`worldfoundry_train/model/predictor.py`
**修改**：`forward()` 中将 `mask_token` 与 `pos_embed` 在 `torch.where` / `add` 之前一次性 cast 到 `context_feats.dtype`，避免 bf16 输入触发 fp32 隐式提升（双倍激活内存 + O(N) dtype-promotion ops）。
**测试**：`test_bf16_input_keeps_predictor_output_in_bf16`（要求 predictor 自身先 `.to(bfloat16)`，模拟生产路径）。

### 5.3 M3 — EMA 形状漂移保护

**文件**：`worldfoundry_train/model/ema.py`
**修改**：`update()` 入口增加 shape 一致性断言，shape 不匹配时立即 `RuntimeError` 而非静默错误 lerp。
**触发场景**：未来 LoRA rank 改变、误绑 EMAUpdater 到不同子集、checkpoint resume 时 online/target 不对称。

### 5.4 M5 — `W22_7_PILOT_STEPS` 模块级 fallback

**文件**：`scripts/w22_7_run_arm.py`
**修改**：原内联字面量 `"200" if backend == "atlas" else "40"` 提到模块级常量 `DEFAULT_PILOT_STEPS_ATLAS / _DRY`，避免 docstring 与 main() 漂移。

### 5.5 M6 — `PredictorGradSyncStrategy` decorator 解释

**文件**：`worldfoundry_train/jepa_objective.py`
**修改**：扩展 docstring 解释"为何是 decorator 而非新 strategy 子类"——与 `parallel/` 目录的 DDP/FSDP 风格协调，以及未来新增 sync 需求时的可扩展性。

---

## 6. 验收门槛修订汇总

| 项目 | 旧 | 新 |
|------|----|----|
| cosine 验收门槛 | B ≥ 0.6（绝对） | B − A ≥ 0.2；B − C ≥ 0.15 |
| warm-up 窗口报告 | 未规定 | 强制 step 0-100 单独报告 + step 200/500/1000/2000 对照表 |
| NPU profile | 选填 | 必填 gather 内存峰值 + 时延 + target_forward 占比 |
| 旧 ckpt 兼容性 | n/a | warn-and-proceed，缺字段仅 warning；漂移字段仍 ERROR |

---

## 7. 可延后项（不在本次评审范围）

| 项 | 原因 |
|----|------|
| FSDP2 局部 gather 替换 | 待 NPU profile 数据 |
| target encoder backbone 共享引用 | 节省 2.4GB/卡；v2 优化 |
| 3D position embedding RoPE 化 | 长时序鲁棒性；W23+ 评估 |
| HF `bool_masked_pos` patch 或 masked_fill | 上游侵入或语义偏置；独立 ADR |

---

## 8. 关联文件索引

| 主题 | 位置 |
|------|------|
| Hyperparam dataclass + 校验 | `worldfoundry_train/checkpoint.py` (JEPAHyperparams, validate_jepa_hyperparams) |
| Hyperparam 暴露 | `worldfoundry_train/jepa_objective.py` (JEPAObjective.hyperparams) |
| Hyperparam 序列化 | `worldfoundry_train/checkpoint.py` (save_checkpoint, load_checkpoint) |
| Predictor dtype cache | `worldfoundry_train/model/predictor.py` (forward) |
| EMA 形状守卫 | `worldfoundry_train/model/ema.py` (update) |
| Pilot step fallback | `scripts/w22_7_run_arm.py` (DEFAULT_PILOT_STEPS_ATLAS / _DRY) |
| Decorator 解释 | `worldfoundry_train/jepa_objective.py` (PredictorGradSyncStrategy docstring) |
| 新单测 | `worldfoundry_train/tests/test_predictor.py` (4 项), `worldfoundry_train/tests/test_jepa_checkpoint.py` (4 项) |