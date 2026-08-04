# W22.7 A/B 对照实验报告 — 代理损失 vs 真 JEPA 目标

**状态**：🟡 骨架已生成，数字待 Atlas 硬件窗口（带 TBD 处即未填）
**依据**：设计文档 §9.2（`2026-08-03-w22-jepa-predictor-ema-design.md`）
**数据来源**：⚠️ 当前数字来自 CPU dry-run 自检（随机初始化小模型 + 合成噪声，仅验证管线，不构成实验结论）

- Arm A：`jepa.enabled: false`（自蒸馏代理损失基线）
- Arm B：`jepa.enabled: true`（predictor + EMA target encoder，smooth-L1，cosine 0.996→1.0）
- 数据：固定子集 manifest（两臂完全相同样本+mask 序列）；同步数/batch/grad_accum/lr

## 0. Pilot 早停检查（正式跑前置门禁）

| 臂 | 判定 | 失败项 | 警告项 |
|----|------|--------|--------|
| A | ✅ GO（30 步）| — | loss_decreasing |
| B | ✅ GO（30 步）| — | — |

> 证据数值见 `bench_results/w22_7/{a,b}/pilot/verdict.json`；NO-GO 时 .sh 已中止正式跑。

## 1. Probing 指标（最终 checkpoint step）

| 指标 | Arm A | Arm B | 验收门槛 | 判定 |
|------|-------|-------|----------|------|
| 掩码重建 cosine | -0.0155（≈随机水平）| 0.0189 | ≥ 0.6 | ❌ FAIL |
| token 方差（坍塌检查）| 0.994846 | 0.994829 | B 臂不衰减（≥首点的 50%）| ✅ PASS |
| 有效秩 | 61.62 | 61.62 | B > A（参考）| TBD |
| 线性 probing 准确率 | 1.000 | 1.000 | B 显著优于 A（至少一项下游指标）| TBD |
| k-NN 一致性 | 1.000 | 1.000 | 参考 | TBD |

## 2. 吞吐与开销（§12-②③ 补充实测项）

| 项 | Arm A | Arm B |
|----|-------|-------|
| 总 wall time (s) | 11.1 | 6.1 |
| 平均 optimizer step (s) | 0.925134 | 0.508914 |
| target 双前向占比 | — | 1.8% |
| EMA update（含 FSDP gather）占比 | — | 0.1% |
| EMA gather 耗时 (s，FSDP 路径）| — | TBD |

**决策点**：若 EMA gather 占比显著，切 `jepa.ema.update_every > 1`（§12-② 缓解杆）后重测本表。

## 3. 曲线数据

逐 checkpoint step 的指标历史见 `bench_results/w22_7/arm_{a,b}/metrics.jsonl`；
loss 曲线见 `bench_results/w22_7/arm_{a,b}/throughput.json` 的 `loss_curve` 字段。

## 4. 结论

TBD — 待 Atlas 正式跑完成后填写（GO/NO-GO：B 臂是否满足全部验收门槛）。
