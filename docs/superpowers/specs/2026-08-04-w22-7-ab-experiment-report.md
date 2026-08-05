# W22.7 A/B/C 对照实验报告 — 代理损失 vs 真 JEPA vs 零遮挡健全性

**状态**：🟡 骨架已生成，数字待 Atlas 硬件窗口（带 TBD 处即未填）
**依据**：设计文档 §9.2（`2026-08-03-w22-jepa-predictor-ema-design.md`）
**修订**：2026-08-05 addendum（`2026-08-05-w22-design-review-addendum.md`）— 验收门槛改为差值，新增 NPU profile 必填项
**数据来源**：⚠️ 当前数字来自 CPU dry-run 自检（随机初始化小模型 + 合成噪声，仅验证管线，不构成实验结论）

- Arm A：`jepa.enabled: false`（自蒸馏代理损失基线）
- Arm B：`jepa.enabled: true`，`data.mask_ratio: 0.75`（predictor + EMA target encoder，smooth-L1，cosine 0.996→1.0）
- Arm C：`jepa.enabled: true`，`data.mask_ratio: 0.0`（零遮挡健全性检查 — 证明 mask 信号被学到）
- 数据：固定子集 manifest（三臂完全相同样本+mask 序列）；同步数/batch/grad_accum/lr

## 0. Pilot 早停检查（正式跑前置门禁）

| 臂 | 判定 | 失败项 | 警告项 |
|----|------|--------|--------|
| A | ✅ GO（30 步）| — | loss_decreasing |
| B | ✅ GO（30 步）| — | — |
| C | TBD（pilot 未跑）| — | — |

> 证据数值见 `bench_results/w22_7/{a,b,c}/pilot/verdict.json`；NO-GO 时 .sh 已中止正式跑。

## 1. Probing 指标 + 修订后的 Δ-table 验收（review §2.2）

| 指标 | Arm A | Arm B | Arm C | Δ(B−A) | Δ(B−C) | 验收门槛（差值） | 判定 |
|------|-------|-------|-------|--------|--------|------------------|------|
| 掩码重建 cosine | -0.0155（≈随机水平）| 0.0189 | 0.0189 | 0.0344 | 0.0000 | B−A ≥ 0.2；B−C ≥ 0.15 | ❌ FAIL / ❌ FAIL |
| token 方差 ratio (step 12) | — / — | ratio=1.0000 (A=0.9948) | — | — | — | B/A ≥ 0.85（step ≤ 200）| ✅ PASS |
| 有效秩 | 61.62 | 61.62 | 61.62 | — | — | 参考 | TBD |
| 线性 probing（Δ bootstrap 95% CI）| 1.000 | 1.000 | — | mean=0.0000 ci=[0.0000, 0.0000] | — | ci_low > 0 | ❌ FAIL |
| k-NN 一致性 | 1.000 | 1.000 | — | mean=0.0000 | — | B−A mean ≥ 0.1 | ❌ FAIL |

> Bootstrap CI 由 `_paired_delta_series + _bootstrap_ci`（n_boot=2000, seed=20260805）算出；空集时阈值回到 TBD。

## 2. 吞吐与开销

| 项 | Arm A | Arm B | Arm C |
|----|-------|-------|-------|
| 总 wall time (s) | 11.1 | 6.1 | 5.7 |
| 平均 optimizer step (s) | 0.925134 | 0.508914 | 0.473913 |
| target 双前向占比 | — | 1.8% | — |
| EMA update（含 FSDP gather）占比 | — | 0.1% | — |
| EMA gather 耗时 (s，FSDP 路径）| — | TBD | — |

**决策点**：若 EMA gather 占比显著，切 `jepa.ema.update_every > 1`（§12-② 缓解杆）后重测本表。

## 3. NPU profile 报告（review §2.3，必填）

| 字段 | 测量方式 | Arm B 当前值 | 门槛 / 动作 |
|------|----------|--------------|-------------|
| `ema_gather_peak_mb` | `torch.npu.max_memory_allocated()` EMA 期间峰值 | TBD | 4096 MB 预算；超阈 → 推荐 `update_every` ≥ 2 |
| `ema_gather_s_avg` | `timing.ema_gather_s / timing.ema_updates` | TBD | < 50 ms（8-card HCCL 设计估）|
| `target_forward_fraction` | `target_forward_s / loss_fn_s` | 1.8% | < 0.40 |
| `context_forward_s` | 同上 | 0.0966 | 基线对照 |

**决策提示**：无动作 — 当前所有 NPU profile 项在预算内。

## 4. warm-up window 分析（review §3 M4 强制项）

实验报告中必须包含 step 0–100 的 loss 序列，以及 step 200/500/1000/2000 的对照表
（详见 throughput.json 的 `loss_curve` 与 metrics.jsonl 的 `step`）。
B 臂 step 0–100 期间 predictor 末层 MLP zero-init（`predictor.py:101-103`），起始 loss ≈ 0 不是收敛，
结论需基于 step ≥ 200 的曲线对比。

## 5. 曲线数据

逐 checkpoint step 的指标历史见 `bench_results/w22_7/arm_{a,b,c}/metrics.jsonl`；
loss 曲线见 `bench_results/w22_7/arm_{a,b,c}/throughput.json` 的 `loss_curve` 字段。

## 6. 结论

TBD — 待 Atlas 正式跑完成后填写。所有 Δ-table 验收 + NPU profile 项满足后才能标记 GO。
