# W22 — JEPA Predictor + EMA Target Encoder 设计文档

**作者**：Kimi（ML 系统架构师，与 chad 共同设计）
**日期**：2026-08-03
**状态**：🟢 已实现（W22.1–W22.6，2026-08-04 落地，213 tests 全绿）；W22.7 对照实验待 Atlas 硬件窗口
**关联**：W16（checkpoint）、W9/W10（FSDP/MindSpeed）、Phase-2 训练路线；上游对齐 facebookresearch/vjepa2

---

## 1. 背景与问题陈述

### 1.1 现状（已读码核实）

当前训练循环 `worldfoundry_train/training_loop.py:32-48` 的默认损失为：

```python
# worldfoundry_train/training_loop.py:41-48
out  = model(pixel_values_videos=x, bool_masked_pos=mask)      # 前向 ①（有梯度）
feats = out.last_hidden_state
with torch.no_grad():
    tgt_out = model(pixel_values_videos=x, bool_masked_pos=mask)  # 前向 ②（无梯度）
    target = tgt_out.last_hidden_state
return F.mse_loss(feats[bool_masked_pos], target[bool_masked_pos])
```

docstring 自述这是 "CPU surrogate for the EMA target encoder"（`training_loop.py:39`），生产目标应为 "EMA target encoder + cosine similarity loss"（`training_loop.py:14-15`）。全库搜索确认**没有任何独立 predictor 模块**。

### 1.2 代理损失的数学缺陷

设在线 encoder 为 $f_\theta$，输入 $x$，掩码集合 $\mathcal{M}$。当前损失为：

$$
\mathcal{L}_{\text{proxy}} = \big\| f_\theta(x, \mathcal{M})[\mathcal{M}] - \mathrm{sg}[f_\theta](x, \mathcal{M})[\mathcal{M}] \big\|_2^2
$$

注意两个前向的**输入完全相同**（同一 $x$、同一 $\mathcal{M}$），网络也相同。两者唯一的差异来源是训练模式下 LoRA dropout（`configs/vjepa2_vitl_1.2b.yaml:9`，`dropout: 0.05`）带来的随机扰动。因此：

1. **接近恒等回归**：$\mathbb{E}[\mathcal{L}_{\text{proxy}}] \approx \mathrm{Var}_{\text{dropout}}[f_\theta(x)]$。梯度信号的唯一内容是"让 encoder 输出对 dropout 不敏感"，而不是"从可见上下文预测被遮挡内容"。模型**学不到任何预测性表征**，最优解是输出趋近于确定性常数的退化方向——恰好与 JEPA 目标相反。
2. **无上下文/目标不对称性**：真 JEPA 的关键结构是 context encoder 只能看见可见 token，而 target 来自**完整视频**的编码。当前实现两次前向都传入 `bool_masked_pos`，target 端同样缺失被遮挡信息，不存在"从部分预测整体"的任务结构。
3. **无 predictor 瓶颈**：没有容量受限的预测头迫使表征保留对补全有用的信息；encoder 输出直接被回归，等价于自编码恒等映射。
4. **无 EMA 去耦**：即便引入了不对称性，target 与 online 共享同一组权重且同步更新，MSE 在无量纲不对称机制下存在平凡坍塌解（输出常数）。EMA 是 I-JEPA/V-JEPA 系列防止表征坍塌的核心机制之一。
5. **MSE 对未归一化特征尺度敏感**：ViT-L 隐层维度 1024，特征范数随训练漂移，MSE 的绝对数值不可比、不可监控，无法像 cosine 那样提供天然的"对齐度"读数。

结论：当前损失仅能验证训练路径（前向/反向/优化器/FSDP/checkpoint）可跑通，**不构成自监督世界模型目标**。W22 用真 V-JEPA 2 目标替换它。

### 1.3 与背景描述不符之处（读码发现，供评审注意）

- `training_loop.py:36-37` docstring 声称 `bool_masked_pos` 形状为 `(B, T, num_patches)`，实际 `collator.py:36-39` 产出的是展平的 `(B, T*Hg*Wg)`。布尔索引语义上恰好兼容（`(B,N,D)[(B,N)]`），但 docstring 描述有误，本设计按 `(B, N)` 展平格式为准。
- 背景称"target 来自同一 encoder 的 no_grad 二次前向（自蒸馏代理）"——实际更弱：二次前向连 mask 都一样（`training_loop.py:46`），是恒等回归而非自蒸馏。
- `configs/vjepa2_vitl_1.2b.yaml:13` 的 `training.parallel: ddp` 字段目前**未被 trainer 消费**（`trainer.py` 未读取该字段，`make_strategy` 只认 `"train"/"fsdp"`，见 `trainer.py:38-42`）。本设计的 FSDP 兼容方案需在真实 FSDP profile 下另行验证。
- `training_loop.py` 未实现梯度累积（`grad_accum: 8` 在 yaml 中存在但循环里无对应逻辑）。EMA 更新绑定在 `run_training_step` 之后，即当前语义下"每个 micro-batch 后"。

---

## 2. 设计目标与非目标

### 2.1 目标

| # | 目标 | 验收标准 |
|---|------|----------|
| G1 | 引入独立 predictor Transformer 模块 | 新模块 `worldfoundry_train/model/predictor.py`，CPU 单测可跑通前向/反向 |
| G2 | 引入 EMA target encoder（momentum schedule + FSDP 安全更新） | EMA 更新后 target 与 online 权重按公式精确收敛；FSDP 多卡 smoke 测试通过 |
| G3 | 真 JEPA 损失替换代理损失，保留旧损失为 feature flag | `jepa.enabled: false` 时行为与现状逐位一致 |
| G4 | checkpoint 兼容：predictor/EMA 状态可保存、恢复、签名校验 | resume 后 loss 曲线连续；签名不匹配时按 G4 约束 raise |
| G5 | 与 LoRA/FSDP/现有训练入口零破坏集成 | 现有 24 个测试文件全部通过；新增测试覆盖新路径 |

### 2.2 非目标

- **不做 action-conditioned predictor**（V-JEPA 2-AC）：属 RL 子系统 / `vjepa2_worker.py:57` 的 `plan()` Phase-2 W4 范围。
- **不改数据管线**：multiblock mask 的多 block 采样器（`tubelet_mask.py:15` 的 `multiblock_mask` 已存在但未被 trainer 使用）接入留作 W23+；本期继续用 `tubelet_mask`，仅在损失侧支持多 mask。
- **不做从零预训练**：仍是 LoRA 微调范式（spec §3.2），EMA 范围相应限定（见 §5.3）。
- **不改推理侧**：`worldfoundry_inf/vjepa2_worker.py` 不变；predictor 是训练期模块，不进推理图。
- **不追求 V-JEPA 2 官方完整训练配方**（多 crop、多 mask、长 warmup、大规模 batch）：本期对齐**结构**，超参对齐留待对照实验后调优。

---

## 3. 总体架构

```
                    ┌───────────────────────── 训练期三模块 ─────────────────────────┐
                    │                                                              │
  pixel_values_videos (B,T,C,H,W)                                                  │
        │                                                                          │
        ├───────────────────────────┐        ┌──────────────────────────────┐      │
        ▼                           │        ▼                              │      │
 ┌──────────────┐                   │  ┌────────────────┐                   │      │
 │ context       │  bool_masked_pos │  │ target encoder │  完整视频          │      │
 │ encoder       │◄── (B, N)        │  │ (EMA 权重,     │  无 mask, no_grad  │      │
 │ (online:      │                  │  │  requires_grad │                  │      │
 │  backbone+LoRA│                  │  │  =False)       │                  │      │
 └──────┬───────┘                  │  └───────┬────────┘                   │      │
        │ h_ctx (B, N, 1024)       │          │ h_tgt (B, N, 1024)          │      │
        │ 可见位置=上下文编码        │          │ stop-grad                   │      │
        │ 被 mask 位置=mask token   │          │                             │      │
        ▼                          │          ▼                             │      │
 ┌──────────────┐                  │   ┌────────────┐                       │      │
 │ predictor    │                  │   │ 取 masked   │                      │      │
 │ (12×384      │                  │   │ 位置 target │                      │      │
 │  Transformer │                  │   └─────┬──────┘                       │      │
 │  + mask token│                  │         │ ŷ (M, 1024)                  │      │
 │  + 时空位置编码│                 │         ▼                              │      │
 └──────┬───────┘                  │  ┌────────────┐                        │      │
        │ ŷ_pred (M, 1024)         │  │ smooth-L1  │ ← 或 cosine（配置）      │      │
        └───────────────┬──────────┴─►│   loss     │                        │      │
                        │             └─────┬──────┘                        │      │
                        │                   │ loss                          │      │
                        ▼                   ▼                               │      │
                  backward → optimizer.step()                               │      │
                        │                                                   │      │
                        ▼ post_step_hook                                    │      │
                  EMA update: θ_tgt ← m·θ_tgt + (1-m)·θ_online              │      │
                  (仅作用于 encoder 可训练参数 = LoRA adapters)              │      │
                    └──────────────────────────────────────────────────────┘
```

三模块职责：

- **context encoder（online）**：现有 `peft_model`（frozen backbone + LoRA），是唯一接收梯度的 encoder。前向时带 `bool_masked_pos`，与 HF `VJEPA2Model` 接口一致（`training_loop.py:43`）。
- **predictor**：新增模块，消费 context 编码，在 masked 位置重建 target 表征。接收梯度。
- **target encoder**：online encoder 的 EMA 副本，**整体 `requires_grad_(False)`**，前向完整视频（不传 mask），输出经 stop-gradient 作为回归目标。结构上与 online 同构（backbone + 同配置 LoRA），但独立持有权重。

---

## 4. Predictor 模块详细设计

### 4.1 网络结构选型

参考 V-JEPA 2 官方实现（facebookresearch/vjepa2，ViT-L 预训练配置）：predictor 是一个**窄而深的 Transformer**，宽度远小于 encoder（1024 → 384），以容量瓶颈防止 predictor 退化为恒等映射。

| 超参 | 取值 | 依据 |
|------|------|------|
| 层数 depth | 12 | 官方 vjepa2 vitl 配置 `pred_depth=12`（I-JEPA 为 6，V-JEPA 系列加深以建模时空依赖） |
| 宽度 embed_dim | 384 | 官方 `pred_embed_dim=384`；约为 encoder 宽度的 3/8，形成表征瓶颈 |
| 注意力头数 | 12 | 官方 vitl predictor 配置；每头 32 维 |
| MLP ratio | 4.0 | 与 ViT 惯例一致 |
| 参数量估算 | ≈ 12 × (4·384² + 8·384²) ≈ 21M | 约为 encoder 的 1.7%，训练/显存开销可忽略 |
| qkv_bias | True | 官方默认 |
| 初始化 | trunc_normal(std=0.02)；末层 block 输出投影 zero-init | 保证训练初期 predictor 输出接近 0，EMA target 侧主导早期梯度尺度，稳定 warmup |

实现选择：**手写轻量 `nn.TransformerEncoder`-风格模块**而非依赖 HF 内部类。理由：HF `VJEPA2Model` 的 predictor 子结构未以独立可复用模块暴露；项目约束"训练代码尽量少改 facebookresearch/vjepa2"（spec §1.3 风险偏好），自建模块零侵入上游。

新文件 `worldfoundry_train/model/predictor.py`：

```python
class JEPAPredictor(nn.Module):
    def __init__(
        self,
        num_patches: int,          # N = T*Hg*Wg，构建期由首个 batch 或 cfg 推断
        encoder_dim: int = 1024,   # ViT-L hidden size
        predictor_dim: int = 384,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
    ) -> None: ...

    def forward(
        self,
        context_feats: torch.Tensor,     # (B, N, 1024)，online encoder 全序列输出
        bool_masked_pos: torch.Tensor,   # (B, N) bool，True = 被 mask
    ) -> torch.Tensor:                    # returns (M_total, 1024)，仅 masked 位置的预测
        ...
```

### 4.2 Mask token 与时空位置编码

- **可学习 mask token**：`nn.Parameter(torch.zeros(1, 1, predictor_dim))`，trunc_normal(0.02) 初始化。前向时把 context 序列中 masked 位置的 token **替换**为 mask token（而非丢弃重排），保持序列定长 `N`，与现有 `(B, N)` mask 格式零转换对接。
- **时空位置编码**：采用**可学习 3D 位置嵌入** `pos_embed: (1, N, predictor_dim)`，按 `(t, h, w)` 网格序展开，与 collator 的展平序一致（`collator.py:36` 用 `m.reshape(-1)`，C-order，即 `(T, Hg, Wg)` 行优先）。为支持可变分辨率/时长，位置嵌入构建为 `(1, T_max*Hg_max*Wg_max, D)` 参数 + 前向时按实际网格做 3D 线性插值（`F.interpolate(mode="trilinear")`）裁剪。v1 简化：固定网格由 cfg `data` 段推导，可变网格走插值路径并加单测。
- **输入投影**：`proj_in: Linear(1024 → 384)`；**输出投影**：`proj_out: Linear(384 → 1024)`。预测在 1024 维空间与 target 对齐。

### 4.3 前向数据流（结合项目实际 batch 格式）

```
batch["pixel_values_videos"] : (B, T, C, H, W)      # collator.py:38
batch["bool_masked_pos"]     : (B, N) bool           # collator.py:39，N = T*Hg*Wg

1) h_ctx = online_encoder(pixel_values_videos=x, bool_masked_pos=mask).last_hidden_state
                                                         # (B, N, 1024)  有梯度
2) p = predictor(h_ctx, mask)                            # (M, 1024)，M = mask.sum()
   内部:
     x = proj_in(h_ctx)                                  # (B, N, 384)
     x = torch.where(mask[..., None], mask_token.expand(B,N,-1), x)
     x = x + pos_embed                                   # 时空位置编码
     x = blocks(x)                                       # 12 × Transformer
     x = proj_out(x)                                     # (B, N, 1024)
     return x[mask]                                      # (M, 1024)
3) with torch.no_grad():
     h_tgt = target_encoder(pixel_values_videos=x).last_hidden_state   # (B, N, 1024)
     y = h_tgt[mask]                                     # (M, 1024)
4) loss = jepa_loss(p, y)
```

注意第 3 步 target 前向**不传 `bool_masked_pos`**——target encoder 看完整视频。这正是当前代理损失缺失的不对称性（§1.2-2）。

---

## 5. EMA Target Encoder 详细设计

### 5.1 EMA 更新公式

$$
\theta_{\text{tgt}}^{(s)} \leftarrow m_s \cdot \theta_{\text{tgt}}^{(s-1)} + (1 - m_s) \cdot \theta_{\text{online}}^{(s)}
$$

其中 $s$ 为优化器步数。EMA **只作用于 online encoder 中被训练的参数**（见 §5.3），逐参数 `torch._foreach_lerp_` 批量执行（NPU 上走 `_foreach` fused kernel，避免逐参数 Python 循环开销）。

### 5.2 Momentum schedule

采用 I-JEPA/V-JEPA 惯例的余弦 warmup：

$$
m_s = m_{\text{end}} - (m_{\text{end}} - m_{\text{start}}) \cdot \frac{1 + \cos(\pi \cdot s / S)}{2}
$$

- $m_{\text{start}} = 0.996$，$m_{\text{end}} = 1.0$，$S$ = `ema_schedule_steps`（默认 = 总训练步数）。
- $m \to 1.0$ 时 EMA 时间常数 $1/(1-m) \to \infty$，target 冻结，训练后期提供稳定回归目标。
- schedule 按 **optimizer step** 计数（不是 micro-batch）；由于当前循环未实现梯度累积（§1.3-4），两者暂等价，但代码中计数器必须挂在 `post_step_hook` 上，为将来 grad_accum 落地留出正确语义。

### 5.3 与 LoRA 的交互：EMA 更新什么？

**决策：EMA 作用于 LoRA adapter 权重；backbone 全量权重不参与 EMA。**

理由：

1. LoRA 微调范式下 backbone 冻结（`lora_wrapper.py:61` 后 backbone 参数 `requires_grad=False`），**全库唯一随训练变化的 encoder 权重就是 LoRA adapters**（`trainer.py:141-142` 日志亦只报告 LoRA trainable）。对 frozen backbone 做 EMA 是恒等操作，徒增 2.4GB 全量拷贝的写放大。
2. target encoder 结构 = frozen backbone（与 online 逐位相同，初始化时 `deepcopy` 或直接共享只读引用）+ EMA 后的 LoRA adapters。前向行为 = "合并了平滑 LoRA 的完整模型"，与官方"EMA 全量 encoder"在**函数空间**等价——因为 backbone 相同，差异仅来自 LoRA，而 LoRA 差异被 EMA 平滑。
3. **不合并 LoRA 再 EMA**（`merge_and_unload`）：合并会物化一份新的全量 bf16 权重（+2.4GB）且使 target 与 online 的 backbone 解耦，破坏"backbone 逐位相同"这一等价前提，无收益。
4. 前瞻兼容：若未来开启全量微调（`lora.r=0` / full-FT profile），EMA 作用域改为"所有 `requires_grad=True` 的 encoder 参数"，同一 `EMAUpdater` 代码路径天然覆盖——它按参数名集合操作，不假设 LoRA 结构。

target encoder 显式持有独立的 LoRA adapter 副本（~10MB），backbone 部分 v1 采用 `deepcopy`（+2.4GB bf16，见 §8.2），v2 优化项为与 online 共享 frozen backbone 引用（两个 PEFT adapter 挂同一 base）。

### 5.4 FSDP 分片状态下的 EMA（难点）

问题：FSDP `FULL_SHARD`（`fsdp.py:138`）下 LoRA 参数也被分片，每个 rank 只持有 1/W 的 flattened shard（`use_orig_params=True` 保留逻辑视图，但物理存储仍分片）。直接 `param.data.lerp_` 只在 shard 上操作，语义错误。

**方案（v1，简单正确）：gather-then-update。**

- 复用 `checkpoint.py:79-110` `_trainable_state_dict` 的 FULL_STATE_DICT gather 机制，但变体为 `rank0_only=False, offload_to_cpu=False`（LoRA 全量仅 ~10MB，allgather 开销可忽略）。
- 每个 post-step：
  1. `gather_full_trainable_params(model)` → 每 rank 得到完整 LoRA 权重字典（bf16，NPU 显存内）；
  2. 在每 rank 本地对 target encoder 的 LoRA 参数执行 `torch._foreach_lerp_(tgt_params, online_params, 1 - m_s)`；
  3. target encoder **不进 FSDP wrap**（§8.1），其 LoRA 参数在每 rank 全量复制，更新天然一致（输入相同 → 全 rank EMA 结果逐位相同，无需额外广播）。
- 通信开销：每步 allgather ~10MB bf16。在 HCCL 上 < 1ms 量级；相对 1.2B 模型一次 fwd+bwd 可忽略。若未来步频敏感，可降频 EMA（每 k 步更新，$m$ 相应幂次补偿：$m' = m^k$），配置项预留 `ema_update_every: 1`。
- 单 rank / DDP 路径：参数本就全量，跳过 gather，直接 lerp（同一 `EMAUpdater.update()` 内部按 `dist.is_initialized() and is_fsdp(model)` 分派，判断逻辑复用 `checkpoint.py:70-76` `_is_fsdp_module`）。
- **顺序约束**：EMA 必须在 `optimizer.step()` 之后（否则平滑的是旧权重），由 `post_step_hook` 保证（§7.1）。

### 5.5 Target encoder 的数值细节

- 构建期：`target_encoder = copy.deepcopy(peft_model)`（在 FSDP wrap **之前**），随后 `target_encoder.requires_grad_(False)`、`.eval()`（关闭 LoRA dropout——target 必须确定性，这是 §1.2-1 的 dropout 噪声在 target 侧的正确归宿：彻底消除）。
- target 前向始终在 `torch.no_grad()` 下进行。
- dtype 与 online 一致（bf16）。EMA lerp 在 bf16 上做会引入舍入漂移，尤其 $m \to 1.0$ 时 $(1-m)$ 极小。**EMA 状态以 fp32 副本持有**，update 时：fp32 lerp → 写回 bf16 target 参数。fp32 LoRA 副本仅 ~20MB，代价可忽略。

---

## 6. 损失函数设计

### 6.1 smooth-L1 vs cosine 的选择

**决策：默认 smooth-L1（Huber, β=1.0），cosine 保留为配置选项。**

| 维度 | smooth-L1（默认） | cosine |
|------|-------------------|--------|
| 上游惯例 | I-JEPA / V-JEPA / V-JEPA 2 官方均用 L1 族回归未归一化特征 | DINO/BYOL 系惯例 |
| 与 EMA 协同 | EMA + L1 是 V-JEPA 验证过的防坍塌组合 | cosine 只对齐方向，放任范数漂移；范数自由度需额外机制约束 |
| 监控可读性 | 绝对数值随特征尺度漂移，需配合 target 范数日志解读 | 有界 [-1,1]，天然可当 probing 指标 |
| 异常值鲁棒 | β 内二次、外线性，优于纯 L2 | 本身对尺度不敏感 |

当前 docstring 提到的 "cosine similarity"（`training_loop.py:15`）**不作为默认**：在 V-JEPA 官方范式中 cosine 未经验证，而 L1 族 + EMA 是已复现组合。cosine 作为 `loss.type: cosine` 选项保留，供对照实验使用。

### 6.2 归一化与多 mask

- **loss 归一化**：`F.smooth_l1_loss(p, y, beta=β, reduction="sum") / M`，按本 rank 的 masked token 数 M 归一；分布式下 loss 已由 FSDP 梯度 allreduce 平均，不重复缩放。
- **target 侧不归一化**（保持与官方一致）；训练日志额外输出 `||y||_2` 均值作为特征尺度监控。
- **多 mask 策略**：接口层面 `jepa_loss_fn` 接受 `masks: list[Tensor]`，对 K 个独立采样的 mask 各算一次 loss 取平均（对应官方多 block 预测）。v1 默认 K=1 复用 `batch["bool_masked_pos"]`；K>1 依赖多 mask 数据管线（非目标 §2.2），代码路径预留并加单测。

---

## 7. 与现有代码的集成方案

### 7.1 `training_loop.py` 改造点

| 位置 | 改造 |
|------|------|
| `training_loop.py:29` `LossFn` | 契约不变：`loss_fn(model, batch) -> Tensor`。JEPA 损失以闭包注入（predictor/target/EMA 捕获在闭包内），零侵入现有签名 |
| `training_loop.py:51` `run_training_step` | 新增可选参数 `post_step_hook: Callable[[], None] | None = None`，在 `strategy.step(optimizer)`（`:76`）之后调用。EMA 更新挂这里 |
| `training_loop.py:80` `run_training_loop` | 透传 `post_step_hook` 参数 |
| `training_loop.py:32` `default_loss_fn` | **不改**，作为回退路径保留（§10.3） |
| 新增 `jepa.py`（建议放 `worldfoundry_train/jepa_objective.py`） | `class JEPAObjective`：持有 predictor、target_encoder、ema_updater；方法 `loss_fn(model, batch) -> Tensor`（符合 LossFn 契约）与 `after_step() -> None`（EMA + momentum 步进）。单一聚合点，避免闭包散变量 |

### 7.2 `trainer.py` 改造点

| 位置 | 改造 |
|------|------|
| `trainer.py:61-65` `_build_optimizer` | 参数组从 `[p for p in model.parameters() if p.requires_grad]` 扩展为 **LoRA 组 + predictor 组**，predictor 可用独立 lr 倍率（配置 `jepa.predictor_lr_scale`，默认 1.0）。注意必须在 predictor 构建**之后**调用 |
| `trainer.py:132-139` LoRA 构建后 | 插入：若 `cfg["jepa"]["enabled"]`，构建 predictor → `deepcopy` target encoder（**在 `strategy.wrap` 之前**，`:140`）→ 构建 `JEPAObjective` → 替换 `loss_fn` 与 `post_step_hook` |
| `trainer.py:155-158` resume 分支 | 扩展 `load_checkpoint` 调用，恢复 predictor/EMA 状态与 momentum step（§7.4） |
| `trainer.py:168-173` `_step_cb` / `:205-210` final save | 传入扩展后的 checkpoint payload |
| `trainer.py:38-42` `make_strategy` | 顺手修复但不阻塞：`training.parallel` 字段当前未被消费（§1.3-3），本期不动，仅在文档中标注 |

### 7.3 配置文件新增字段

```yaml
# worldfoundry_train/configs/vjepa2_vitl_1.2b.yaml 追加：

jepa:
  enabled: true                # false = 回退到旧代理损失（feature flag, §10.3）

  predictor:
    embed_dim: 384             # 官方 vitl pred_embed_dim
    depth: 12                  # 官方 pred_depth
    num_heads: 12
    mlp_ratio: 4.0
    lr_scale: 1.0              # 相对 training.lr 的倍率

  ema:
    momentum_start: 0.996      # I-JEPA/V-JEPA 惯例
    momentum_end: 1.0
    schedule: cosine           # cosine | constant
    schedule_steps: null       # null = 等于总训练步数
    update_every: 1            # 每 k 个 optimizer step 更新一次（m 取 m^k 补偿）

  loss:
    type: smooth_l1            # smooth_l1 | l1 | cosine
    beta: 1.0                  # smooth_l1 的 Huber β
    num_masks: 1               # 多 mask 预留（§6.2）；>1 需数据管线支持（W23+）
```

### 7.4 Checkpoint 格式变更

向后兼容策略：**新增键全部可选**；旧 checkpoint（W16 格式）在 `jepa.enabled=false` 时可原样加载；`jepa.enabled=true` 加载无 JEPA 键的旧 checkpoint 时 raise `KeyError` 并明确提示"该 checkpoint 不含 JEPA 状态，请关闭 jepa.enabled 或从头训练"。

`checkpoint.py` 变更点：

| 位置 | 变更 |
|------|------|
| `checkpoint.py:60-67` `_REQUIRED_KEYS` | 不变；新增 `_OPTIONAL_JEPA_KEYS = ("predictor_state", "ema_state", "ema_step", "jepa_signature")` |
| `checkpoint.py:121` `save_checkpoint` | 新增可选参数 `jepa_objective: JEPAObjective | None = None`；存在时序列化：`predictor_state`（predictor.state_dict，~85MB fp32 / ~42MB bf16）、`ema_state`（target encoder 的 LoRA fp32 EMA 副本，~20MB）、`ema_step`（momentum schedule 计数器，int） |
| `checkpoint.py:165` `load_checkpoint` | 新增可选参数 `jepa_objective`；存在时恢复 predictor 权重、EMA 状态、`ema_step`，并对 `jepa_signature` 走与 `validate_lora_signature`（`:233`）同构的校验 |
| 新增 `JEPASignature` dataclass | 字段：`pred_dim, pred_depth, pred_heads, encoder_dim, ema_schedule, loss_type`。架构字段不匹配 → `ValueError`（对齐 `validate_lora_signature:255` 语义）；不含架构语义的超参（momentum 端点、β）不匹配 → 仅 warning |

predictor 保存注意：predictor **不参与 FSDP**（§8.1），其 state_dict 直接取即可，无需 FULL_STATE_DICT 路径。

---

## 8. FSDP / 分布式兼容性

### 8.1 Predictor 是否参与 FSDP wrap

**决策：不参与。** Predictor 仅 ~21M 参数，每 rank 全量持有，走普通 DDP 式梯度同步。实现上：

- predictor 在 `strategy.wrap(peft_model)`（`trainer.py:140`）之外独立存在；
- 梯度同步：多 rank 时在 `post_step_hook` **之前**（backward 之后、step 之前需要对 predictor 梯度 allreduce——挂在 `strategy.step` 路径旁，复用 `fsdp.py:238-245` 已有的手动 all_reduce 分支模式：当 encoder 未被真 FSDP wrap（单 rank / DDP profile）时 predictor 梯度随该分支一起 allreduce；当真 FSDP 时，为 predictor 参数显式 `dist.all_reduce`（~21M×bf16 ≈ 42MB，一步一次，可接受）。
- 简化 v1：predictor 参数注册进与 LoRA 相同的 optimizer（`trainer.py:61` 改造后自然覆盖），梯度同步作为 `_sync_predictor_grads(model_is_fsdp)` 工具函数置于 `jepa_objective.py`。

### 8.2 EMA target encoder 显存开销

ViT-L 1.2B bf16 全量副本：

$$
1.2 \times 10^9 \times 2\ \text{B} = 2.4\ \text{GB}
$$

叠加项：fp32 EMA LoRA 副本 ~20MB、target 前向激活（no_grad、可开 checkpointing 或降分辨率推理，bf16 batch=2 时 < 2GB 量级）、predictor ~85MB(fp32 master) + 激活。**合计每卡新增 ≈ 4.5–5.5GB**。Atlas 800 A2 单卡 64GB HBM，LoRA 训练基线占用远低于此，余量充足。

进一步优化（非 v1 范围）：target 与 online 共享 frozen backbone 引用 → 省 2.4GB；target 前向用 activation checkpointing。

### 8.3 NPU 注意事项

- `_foreach_lerp_` 在 torch_npu 2.9.0 已有 fused 实现；若单测发现不支持，降级为 `for` 循环 `lerp_`（20MB 参数，开销仍可忽略）——代码中加 `try/except` 降级并 warning。
- HCCL allgather/allreduce 张量需 contiguous 且在 NPU 上；EMA gather 的 `offload_to_cpu=False` 路径需注意 FSDP 在 CPU-offload 配置（`fsdp.py:179` `cpu_offload`）下与 NPU 的组合未验证——v1 声明 `cpu_offload: true` 与 `jepa.enabled: true` 互斥，配置校验时报错。
- target 前向的 no_grad 在 torch_npu 上无特殊行为；但需确认 `VJEPA2Model` 在 `torch_dtype=bf16` + NPU 下两次前向（online + target）的算子图编译开销（MindIE/torch.compile 缓存按形状命中，形状一致则无重复编译）。

---

## 9. 验证方案

### 9.1 单元测试清单（新增 `worldfoundry_train/tests/`）

| 测试文件 | 用例 |
|----------|------|
| `test_predictor.py` | ① 前向形状：`(B,N,1024)`+`(B,N)` mask → `(M,1024)`；② mask 全 False 时 M=0 不报错（返回空张量或跳过 loss）；③ mask 全 True 边界；④ 反向梯度到达 mask_token/pos_embed/proj；⑤ 可变网格 3D 插值路径形状正确 |
| `test_ema_updater.py` | ① 单步更新数值精确：`m=0.5` 时 `tgt = 0.5*tgt + 0.5*online`；② cosine schedule 端点：`m_0=0.996`、`m_S=1.0`、单调递增；③ `update_every=k` 时等效 `m^k`；④ fp32 EMA 状态与 bf16 写回一致性；⑤ deepcopy 后修改 online 不串改 target |
| `test_jepa_loss.py` | ① smooth_l1/l1/cosine 三分支数值；② 按 M 归一化正确性（构造已知 M）；③ target 无梯度（`h_tgt.requires_grad == False`）；④ online 前向带 mask、target 前向不带 mask（mock model 断言调用参数） |
| `test_jepa_checkpoint.py` | ① save/load 后 predictor 权重逐位一致；② `ema_step` 恢复后 momentum 接续正确；③ `JEPASignature` 不匹配 → ValueError；④ 旧格式 ckpt + `jepa.enabled=true` → KeyError 提示；⑤ 旧格式 ckpt + `enabled=false` → 兼容加载 |
| `test_training_loop_jepa.py` | ① `post_step_hook` 在 optimizer.step 之后被调用（顺序 mock 断言）；② `enabled=true` 端到端 3 步 CPU smoke：loss 有限、EMA 权重发生变化、predictor 权重发生变化 |
| `test_fsdp_jepa_smoke.py` | 多 rank gloo：② gather 路径下各 rank EMA 结果一致；③ predictor 梯度 allreduce 后各 rank 一致 |

### 9.2 对照实验设计（代理损失 vs 真 JEPA）

固定数据子集（VideoMix22M 抽样 ~10k 视频）、相同步数（如 2k 步）、相同 LoRA 配置，跑 A/B：

| 臂 | 配置 | 观察指标 |
|----|------|----------|
| A（基线） | `jepa.enabled: false` | 训练 loss 曲线 |
| B（JEPA） | `jepa.enabled: true`（smooth-L1, EMA cosine 0.996→1.0） | 训练 loss + 下游 probing |

**另需实测**（W22.6 实现回填，见 §12-②③）：FSDP gather 每步引入的同步点开销（FULL_STATE_DICT 瞬态全模型物化，决定是否切 `ema_update_every>1`）；target 双前向的吞吐损失（理论 -25~33%，对应 §11.1 风险表第 2 行）。

**Probing 对比指标**（冻结 encoder，线性/轻量探针）：

1. **掩码重建余弦相似度**：held-out 视频上，predictor 输出与 target encoder 输出的 cosine（B 臂应显著高于 A 臂的随机水平）；
2. **线性 probing 下游任务**：动作/场景分类线性头准确率（SSv2 子集或项目 eval 管线已有任务，见 `worldfoundry_inf/eval/`）；
3. **表征坍塌检查**：encoder 输出的 token 间方差 / 有效秩（effective rank）。A 臂预期方差持续走低（趋坍塌），B 臂应保持稳定——这是判别"学到东西 vs 恒等退化"的最直接信号；
4. **k-NN 检索一致性**：同视频不同时段的表征 k-NN 命中率。

验收门槛（建议，评审时定稿）：B 臂在指标 3 上 token 方差不衰减、在指标 1 上 cosine ≥ 0.6（2k 步时），且至少一项下游 probing 指标显著优于 A。

### 9.3 回退开关（feature flag）

- `jepa.enabled: false` 时，`trainer.py` 不构建 predictor/target/EMA，`loss_fn` 落回 `default_loss_fn`（`training_loop.py:32`），`post_step_hook=None`，checkpoint 不含 JEPA 键——**与现状逐位等价**，现有测试零改动通过。
- 运行时环境变量 `WORLDFORGE_JEPA=off` 覆盖配置（便于线上快速回退，优先级：env > yaml）。

---

## 10. 分步实施计划

| 子任务 | 内容 | 可独立评审点 | 依赖 | 状态 |
|--------|------|--------------|------|------|
| W22.1 | `model/predictor.py`：JEPAPredictor 模块 + 单测（§9.1 test_predictor） | 纯新增文件，零侵入；API 冻结 | 无 | ✅ 完成 |
| W22.2 | `model/ema.py`：EMAUpdater（schedule、fp32 状态、单 rank lerp）+ 单测 | 纯新增文件 | 无 | ✅ 完成 |
| W22.3 | `jepa_objective.py`：JEPAObjective 聚合（loss_fn + after_step + target 构建）+ 单测 | 串起 W22.1/2；CPU smoke | W22.1, W22.2 | ✅ 完成 |
| W22.4 | `training_loop.py`：`post_step_hook` 透传 + `trainer.py` 集成 + YAML 配置 + feature flag | 现有测试全绿 + 新增集成测试 | W22.3 | ✅ 完成 |
| W22.5 | `checkpoint.py`：JEPA 状态 + JEPASignature + 兼容性测试 | 旧 ckpt 兼容矩阵全绿 | W22.4 | ✅ 完成 |
| W22.6 | FSDP 路径：gather-then-update EMA + predictor 梯度同步 + gloo 多 rank 测试 | 多 rank smoke | W22.5 | ✅ 完成（两处实现偏差见 §12-②③） |
| W22.7 | 对照实验（§9.2）+ 实验报告 | 报告归档 `docs/superpowers/specs/` | W22.6，需要 NPU 资源 | ⏳ 待 Atlas 硬件窗口 |

每步独立提交；W22.1–W22.3 纯新增可并行评审，W22.4 是第一个触及现有文件的 PR，单独过 G3 边界检查。

---

## 11. 风险与开放问题

### 11.1 风险

| 风险 | 等级 | 缓解 |
|------|------|------|
| HF `VJEPA2Model` 带 `bool_masked_pos` 的前向语义与预期不符（True 究竟表示"屏蔽该 token"还是"预测该 token"需在真实模型上验证；当前代码两种读法都自洽） | ~~高~~ **已关闭** | W22.3 实证定案：该 kwarg 在 transformers 5.1.0 中被 `**kwargs` 静默吞掉，语义问题不存在（§12-①） |
| 2.4GB target 副本 + 双前向的吞吐损失（每步 +1 次 encoder 前向，理论吞吐降 ~25–33%） | 中 | target 前向 no_grad 无反向图；必要时 target 降帧采样（配置预留）；实测写入 W22.7 报告 |
| FSDP gather-then-update 每步 allgather 引入同步点，与 `BACKWARD_PRE` prefetch（`fsdp.py:176`）交叠行为未验证 | 中 | W22.6 gloo 先行；NPU 上 profile 后视情况切 `ema_update_every>1` |
| LoRA-only 训练下 EMA 信号弱（adapter 仅 ~10MB 权重在动，target≈online 的差距主要来自 EMA 滞后，早期 $m=0.996$ 时时间常数仅 250 步） | 中 | 对照实验指标 3（方差/有效秩）直接监控；若坍塌仍发生，备选：predictor 侧加 stop-grad 不对称强化或提高 mask_ratio |
| bf16 EMA 舍入漂移 | 低 | fp32 EMA master（§5.5）已内建 |
| 位置编码 3D 插值在分辨率变化时引入系统性偏差 | 低 | v1 固定网格为主路径；插值路径单独单测 |

### 11.2 开放问题

1. **`bool_masked_pos` 在 HF 前向中的确切语义**（见风险表第 1 行）——**已解决（§12-①）**：kwarg 被静默忽略，masking 全部由 JEPAPredictor 内部承担，无需翻转。
2. **predictor 是否共享 encoder 的 activation checkpointing 策略**：predictor 小（21M），默认不开；若 batch 放大导致激活紧张再议。
3. **多 mask（K>1）与 multiblock 采样器的接入时机**：`multiblock_mask`（`tubelet_mask.py:15`）已实现但未接线，建议 W23 独立工作项，与官方配方对齐一并做。
4. **`training.parallel: ddp` 字段悬而未用**（§1.3-3）：DDP profile 是否要在 W22 顺带补，或另开工作项？本期建议不动，仅记录。
5. **EMA 是否覆盖 predictor**（V-JEPA 官方不覆盖；某些变体对 predictor 也做 EMA 用于推理蒸馏）——本期不做，若 AC/推理侧未来需要 predictor 的稳定副本再议。

---

## 12. 实施偏差记录（2026-08-04 实现回填）

W22.1–W22.6 已落地（167 基线 + 46 新增 = 213 tests 全绿）。以下三处实测结论/实现偏差替代设计稿中的原始假设，均为有意决策并已在代码注释中固化：

### ① `bool_masked_pos` 被 HF `**kwargs` 静默吞掉（风险 #1 关闭，比预期更彻底）

实证（随机初始化小 `VJEPA2Model`，transformers 5.1.0；固化测试 `tests/test_jepa_objective.py::test_hf_encoder_ignores_bool_masked_pos`）：`VJEPA2Model.forward` 的签名是 `(pixel_values_videos, context_mask, target_mask, skip_predictor, ..., **kwargs)`——**根本没有 `bool_masked_pos` 参数**，它被 `**kwargs` 吞掉。不带 mask、带 mask A、带 mask B 的三次前向 `last_hidden_state` 逐位相同（`torch.equal`）。

结论与影响：

- "True=屏蔽 vs True=预测"的问题在 HF 层面不存在，predictor 输入构造**无需翻转 mask**；masking 完全由 `JEPAPredictor` 内部的 mask-token 替换承担（late masking，即 §4.3 step 2 的自包含数据流）。
- `JEPAObjective.loss_fn` 两次前向**均不传** `bool_masked_pos`，且传 `skip_predictor=True` 跳过 HF 内置（frozen）predictor 的死算力。
- 推论：现行 `default_loss_fn` 比 §1.2 描述的还要弱——mask 连前向都不影响，只挑选回归位置，是纯粹的恒等回归。
- context/target 不对称性改由"predictor 内 mask-token 替换 + EMA 权重差 + LoRA dropout 开/关"承担；encoder 级"只看可见 token"的真不对称在 HF 架构下不可达（需 patch 上游，超范围）。

### ② FSDP gather 的瞬态开销大于设计估计（§5.4 修正）

设计估"每步 allgather ~10MB bf16"。实现发现 FSDP1 的 FULL_STATE_DICT gather 会在**每 rank 瞬态物化全模型**（含 frozen backbone，ViT-L bf16 ~2.4GB/rank/EMA step），随后才按名过滤 LoRA 权重；FSDP1 无部分参数 gather API。v1 按设计指定机制保正确性（`EMAUpdater._online_f32`，`worldfoundry_train/model/ema.py`），缓解杆为 `ema_update_every > 1`，是否启用由 W22.7 的 NPU profile 决定。

### ③ predictor 梯度同步实现为策略装饰器（§8.1 挂点修正）

设计建议 `_sync_predictor_grads` "挂在 `strategy.step` 路径旁"。实际 `training_loop` 在 backward 之后、`strategy.step` 之前没有注入点，故实现为 `PredictorGradSyncStrategy`（`worldfoundry_train/jepa_objective.py`）：装饰任意 inner strategy，在 `step()` 内对 predictor 梯度显式 `all_reduce(SUM)/world_size` 后委托。对 DDP 与 FSDP 两种 profile 统一成立（predictor 在两种 wrap 之外），单 rank 纯透传；`parallel/*` 零改动。grad_accum 下只在关窗 step 同步，且 predictor 梯度与 encoder 同样经 `1/grad_accum` 缩放 + 尾部 rescale，语义为窗口均值。

附带实测约束（仅影响测试桩，生产 HF 路径天然满足）：torch 2.10 的 FSDP1 在无 accelerator 时需显式 `device_id=torch.device("cpu")`；且 FSDP 的 pre-backward hook 注册要求 forward 返回 dataclass/内建容器（HF `ModelOutput` 满足，裸 `SimpleNamespace` 会导致 hook 不注册、post-backward 状态断言失败）。

---

## 附录 A：关键文件与行号索引

| 主题 | 位置 |
|------|------|
| 现行代理损失 | `worldfoundry_train/training_loop.py:32-48` |
| LossFn 契约 | `worldfoundry_train/training_loop.py:29` |
| step/loop 改造点 | `worldfoundry_train/training_loop.py:51-77, 80-106` |
| 训练编排改造点 | `worldfoundry_train/trainer.py:61-65, 129-144, 155-173, 205-210` |
| LoRA 构建 | `worldfoundry_train/model/lora_wrapper.py:31-76` |
| FSDP wrap / 手动 allreduce 分支 | `worldfoundry_train/parallel/fsdp.py:120-187, 232-248` |
| checkpoint 改造点 | `worldfoundry_train/checkpoint.py:60-110, 121-162, 165-230, 233-259` |
| batch 数据格式 | `worldfoundry_train/data/collator.py:11-40` |
| mask 采样器 | `worldfoundry_train/data/tubelet_mask.py:15-55, 80-108` |
| 推理侧（不变） | `worldfoundry_inf/vjepa2_worker.py:38-65` |
