# W22.8 设计稿：多 mask 数据管线（multi-mask pipeline）

- 日期：2026-08-05
- 状态：草案（待实现）
- 前置：W22 JEPA predictor + EMA（`2026-08-03-w22-jepa-predictor-ema-design.md`）、
  W22 设计评审补遗（`2026-08-05-w22-design-review-addendum.md`）
- 编号说明：**本工作项编号为 W22.8，不是 W23**。路线图
  （`docs/roadmap/2026-08-03-jepa2-ascend-gap-analysis-roadmap.md:157`）的
  W23 槽位已分配给「DVPP 硬件视频解码落地」。代码与 W22 设计稿里散落的
  `W23+` 字样（`jepa_objective.py:278`、`trainer.py:141-142`、
  `configs/vjepa2_vitl_1.2b.yaml:51`、W22 设计稿 §2.2/§6.2/§7.3/§11.2）
  含义是「W22 之后」，非路线图 W23。落地时一并把这些字样改指 W22.8。

---

## §1 背景

W22 已经把「对 K 个独立采样 mask 取平均损失」的计算内核写完了——
`JEPAObjective._loss_for_masks(h_ctx, h_tgt, masks)`
（`worldfoundry_train/jepa_objective.py:270-298`）接受 `list[torch.Tensor]`，
逐 mask 调 predictor、逐 mask 算 loss，最后 `torch.stack(losses).mean()`，
并已处理「全 False mask 跳过」与「K 个 mask 全空时返回挂图零」两个边界。

**缺的是数据侧**，且 trainer 目前**主动拒绝** K>1：

| 层 | 现状 | 位置 |
| --- | --- | --- |
| config | `jepa.loss.num_masks != 1` → `ValueError` | `trainer.py:138-142` |
| dataset | 每样本调一次 `mask_fn`，返回 `{"frames", "mask"}` | `data/dataset.py:76-77` |
| collator | B 个 `mask` reshape 成 `(B, N)` → `bool_masked_pos` | `data/collator.py:35-36` |
| objective | `loss_fn` 用 `[mask]` 包成单元素 list | `jepa_objective.py:256` |

W22 设计稿 §6.2 原文：「v1 默认 K=1 复用 `batch["bool_masked_pos"]`；
K>1 依赖多 mask 数据管线（非目标 §2.2），代码路径预留并加单测。」
W22.8 就是把这条预留路径接通，并把 `trainer.py:139` 的拒绝改成支持。

**为什么需要 K>1**：原始 I-JEPA / V-JEPA 配方对每个 context 采样多个
（典型 4 个）target block，让 predictor 在同一份 context 表征上被多次
不同条件地考问。这是 JEPA 系列抑制表征坍缩的主要正则手段之一：
K=1 时 predictor 可对该 mask 位置学出捷径；K>1 时同一 `h_ctx` 必须同时
支撑 K 组不同预测，捷径被压制。W22.7 A/B/C 实验度量的
`token_variance` / `effective_rank` 正是坍缩指标，K>1 预期能进一步拉开
B 与 A 的差距。

---

## §2 目标与非目标

### §2.1 目标（W22.8 v1）

1. `jepa.loss.num_masks > 1` 不再抛错，而是驱动数据管线产出 K 个 mask。
2. `Vjepa2Dataset` 支持 `mask_count: int = 1`；K>1 时对同一样本采样 K 个
   mask，采样确定性可复现。
3. `collate_vjepa2_batch` 支持多 mask 样本，产出 `bool_masked_pos_k`
   形状 `(B, K, N)`。
4. `JEPAObjective.loss_fn` 优先消费 `bool_masked_pos_k`，拆成 K 个
   `(B, N)` 交给已有的 `_loss_for_masks`。
5. **K=1 路径完全向后兼容**：batch key `bool_masked_pos`、样本 key `mask`、
   W22.7 三臂实验脚本、以及除 §5.2 列出的 1 个测试外的全部现有测试不变。

### §2.2 非目标（推迟）

- **接入 `multiblock_mask`**（`data/tubelet_mask.py:15` 已实现但从未被
  trainer 使用）。W22 开放问题 #3 建议「多 mask 与 multiblock 采样器一并
  做」，但两件事的失败模式不同：multiblock 换的是**单个 mask 的形状分布**，
  多 mask 换的是**每步 mask 的个数**。捆在一起做，实验一旦变差无法归因。
  v1 只做后者，K 个 mask 全部来自当前的 `tubelet_mask`、同一组超参、
  仅 rng 种子不同。multiblock 接入另开工作项。
- **context / target 分离采样**（I-JEPA 式 context block 也随机化）。v1 的
  context 恒为完整帧（`h_ctx` 是无 mask 前向），只有 target 位置随机。
- **K 随训练步动态变化**（curriculum）。
- **K>4 的显存优化**。v1 的 K 个 predictor 前向串行，显存峰值随 K 增长；
  先量 K=2 在 Atlas 上的实测开销再谈 K=4。**v1 推荐 K=2**。
- **异构 mask 组**（不同 mask 用不同 `mask_fn` / 不同 tubelet 尺寸）。

---

## §3 数据管线改动

### §3.1 `Vjepa2Dataset`（`worldfoundry_train/data/dataset.py`）

新增构造参数：

```python
def __init__(
    self,
    manifest_path: Path,
    mask_fn: MaskFn,
    patch_size: int = 14,
    mask_count: int = 1,      # NEW
) -> None:
```

校验：`mask_count < 1` → `ValueError`，与既有 `patch_size <= 0` 校验同风格，
在 `__init__` 早失败，不拖到 `__getitem__`。

`__getitem__` 行为：

```python
grid = (T, H // self.patch_size, W // self.patch_size)
masks = [
    self.mask_fn(grid, np.random.default_rng(idx * self.mask_count + i))
    for i in range(self.mask_count)
]
out = {"frames": frames, "mask": masks[0]}
if self.mask_count > 1:
    out["masks"] = masks
return out
```

三个要点：

1. **`mask` 键永远存在**，值是第 0 个 mask。任何只认 `mask` 的旧 consumer
   （`trainer.py:279` 的 `ds[0]["mask"].shape` grid 推断、`test_dataset.py`
   的 9 个测试、`dataset.py:_self_test`）零改动。
2. **`masks` 键仅在 K>1 时出现**。collator 用 `"masks" in sample` 做无歧义
   分派，K=1 时不必构造单元素 list 再拆掉。
3. **种子公式 `idx * mask_count + i`**：保证「同一 idx 的 K 个 mask 互不
   相同」且「跨 epoch 可复现」。K=1 时该式退化为 `idx`，与 W22 现有行为
   **逐位一致**——必须有测试钉死（§5.1 #1），否则 W22.7 已跑出的 A/B
   baseline 曲线会因 mask 序列漂移而无法与 W22.8 结果对比。

### §3.2 `collate_vjepa2_batch`（`worldfoundry_train/data/collator.py`）

返回值扩展：

```
{"pixel_values_videos": (B, T, C, H, W) float32,
 "bool_masked_pos":     (B, N) bool,       # 恒存在 = 第 0 个 mask
 "bool_masked_pos_k":   (B, K, N) bool}    # 仅当样本带 "masks"
```

实现约束：

- `bool_masked_pos` 继续由 `s["mask"]` 生成，语义不变（= 第 0 个 mask）。
  这保证 `training_loop.py:43-49` 的 proxy loss、
  `worldfoundry_inf/vjepa2_worker.py:53` 与 `vllm_ascend_worker.py:160` 的
  推理路径、`test_collator.py` 的 12 个测试全部不受影响。
- **K 一致性校验**：所有样本的 `len(s["masks"])` 必须相同，否则
  `ValueError`，消息带 `sample {i}` 与两侧 K 值，与既有 frames shape
  不一致的报错风格对齐。
- **混合批次校验**：一批里部分样本有 `masks`、部分没有 → `ValueError`。
  静默降级到 K=1 会让 K>1 配置悄悄失效，属典型静默失败。
- 每个 mask 用 `.reshape(-1)` 打平，与现有 C-order 布局一致
  （`model/predictor.py:39` 依赖该布局做位置编码索引）。

### §3.3 `trainer`（`worldfoundry_train/trainer.py`）

**（a）解除拒绝**（`trainer.py:138-142`）。现有代码：

```python
num_masks = int(loss_cfg.get("num_masks", 1))
if num_masks != 1:
    raise ValueError(
        f"jepa.loss.num_masks={num_masks} requires the multi-mask data "
        f"pipeline (W23+, design §2.2 non-goal); v1 supports num_masks=1."
    )
```

改为下界校验：`num_masks < 1` → `ValueError`。上界不设硬限，由 §7 风险 #2
的显存监控兜底。

**（b）透传到 dataset**（`trainer.py:70-78` 的 `_build_dataset`）。
`num_masks` 位于 `cfg["jepa"]["loss"]`，而 `_build_dataset(cfg)` 当前只读
`cfg["data"]`；且 `jepa` 段在 `jepa.enabled=false` 时可能整段缺失。故：

```python
def _build_dataset(cfg: dict) -> Vjepa2Dataset:
    ...
    num_masks = int(
        cfg.get("jepa", {}).get("loss", {}).get("num_masks", 1)
    )
    return Vjepa2Dataset(
        manifest_path=cfg["data"]["manifest"],
        mask_fn=mask_fn,
        mask_count=num_masks,
    )
```

全程用 `.get(..., 默认)` 链：现有 7 处测试 cfg dict
（`test_trainer.py:80`、`test_trainer_jepa.py:83`、`test_pilot_checks.py:188`、
`test_jepa_checkpoint.py:328`、`test_grad_accum.py:298`、
`test_train_cli.py:74`、`test_quantize_bridge_cli.py:29`）中有些没有 `jepa`
段、有些没有 `loss` 段，任何一环写成必填都会一次性打挂它们。

**不新增 `data.mask_count` 配置路径**：`jepa.loss.num_masks` 是 W22 设计稿
§7.3 已经预留并写进 3 个 YAML 的键，再开一个平行开关会产生「两个 K」的
歧义。`mask_count` 只作为 `Vjepa2Dataset` 的构造参数名存在。

---

## §4 目标函数改动

### §4.1 `JEPAObjective.loss_fn`（`jepa_objective.py:235-268`）

唯一改动是 mask 的取法：

```python
x = batch["pixel_values_videos"]
if "bool_masked_pos_k" in batch:
    mk = batch["bool_masked_pos_k"]                   # (B, K, N)
    masks = [mk[:, i] for i in range(mk.shape[1])]    # K × (B, N)
else:
    masks = [batch["bool_masked_pos"]]                # W22 K=1 路径
...
loss = self._loss_for_masks(h_ctx, h_tgt, masks)
```

- 两次 encoder 前向（online + target）**不随 K 增加**：`h_ctx` / `h_tgt` 与
  mask 无关（HF `VJEPA2Model` 压根不消费 `bool_masked_pos`，见
  `jepa_objective.py:14-18` 的实测结论 + `test_jepa_objective.py:175` 的
  回归测试）。K 只放大 predictor 前向与 loss 的开销，不放大 backbone
  开销——这是 K>1 在 NPU 上可行的前提。
- `timing` 的 `context_forward_s` / `target_forward_s` / `loss_fn_s` 三个
  累加器语义不变；K 的开销自然体现在 `loss_fn_s` 与
  `target_forward_fraction` 的比值变化上，W22.7 报告不用改。
- `_loss_for_masks` **函数体一行都不用改**——它早就是 K-general 的。
  只需把它 docstring 里「v1 uses K=1 … reserved for W23+」的措辞更新。

### §4.2 `mk[:, i]` 而非 `mk.unbind(1)`

切片返回 view，不额外分配；`unbind` 语义等价但可读性差。两者都保持
`(B, N)`；contiguous 与否对 `predictor(h_ctx, mask)` 无影响（内部先
`mask.bool()` 再做 boolean index）。

---

## §5 验证

### §5.1 新增测试 `worldfoundry_train/tests/test_w22_8_multi_mask.py`

**dataset 层**

1. `mask_count=1` 时样本**无** `masks` 键，且 `mask` 与 W22 行为逐位相同
   （对照 `mask_fn(grid, default_rng(idx))` 直接计算）。← 向后兼容锚点
2. `mask_count=3` → `len(sample["masks"]) == 3`，且 `sample["mask"]` 与
   `masks[0]` 等值。
3. K 个 mask **互不相同**（至少存在一对不等）。
4. 同一 idx 重复取两次 → K 个 mask 完全一致（确定性）。
5. `mask_count=0` / `-1` → `ValueError`。

**collator 层**

6. K=1 样本 → 输出**无** `bool_masked_pos_k` 键，`bool_masked_pos` 形状
   `(B, N)`。
7. K=2 样本 → `bool_masked_pos_k.shape == (B, 2, N)`、dtype bool；且
   `bool_masked_pos_k[:, 0]` 等于 `bool_masked_pos`。
8. 样本间 K 不一致 → `ValueError`。
9. 混合批次（部分带 `masks`）→ `ValueError`。

**objective 层**

10. `loss_fn` 收到带 `bool_masked_pos_k` 的 batch → predictor 被调 K 次、
    encoder 各被调 1 次（call-counting fake 模型断言）。← 钉死「K 不放大
    backbone」这条 §4.1 的性能前提
11. K 个 mask 相同时，K>1 的 loss ≈ K=1 的 loss（`pytest.approx`）。
    ← 证明是 mean 而非 sum
12. K>1 且全部 mask 为空 → 返回挂图零，`loss.backward()` 不炸。
13. 无 `bool_masked_pos_k` 的 batch → 仍走 K=1 路径（W22 契约）。

**trainer 层**

14. cfg 无 `jepa` 段 → dataset 以 `mask_count=1` 构造（默认值兜底）。
15. cfg 有 `jepa.loss.num_masks=2` → 透传为 `mask_count=2`。
16. `num_masks=0` → `ValueError`。

### §5.2 必须修改的现有测试（唯一一处）

`worldfoundry_train/tests/test_trainer_jepa.py:217`
`test_jepa_rejects_num_masks_above_one`——该测试断言 `num_masks=2` 抛
`ValueError`，正是 W22.8 要解除的限制。改为
`test_jepa_rejects_num_masks_below_one`（断言 `num_masks=0` 抛错），
并把文件头 docstring 第 7 行的「num_masks>1 rejection」改为
「num_masks<1 rejection」。**这是本工作项唯一允许改动的既有测试**；
任何其他既有测试需要改动，都说明向后兼容性被破坏了，应先停下来复查。

### §5.3 文案同步

把以下位置的 `W23+` 改指 W22.8，并更新已过时的「不支持」措辞：
`jepa_objective.py:278`、`trainer.py:141-142`（随代码一起改写）、
`configs/vjepa2_vitl_1.2b.yaml:51`、W22 设计稿 §2.2/§6.2/§7.3/§11.2。

---

## §6 验收门槛

1. `PYTHONPATH=. pytest worldfoundry_train worldfoundry_rl` 全绿；
   既有测试改动数 = 1（§5.2 那一个），其余 259 个一行不改。
2. `ruff check` 干净。
3. `_loss_for_masks` 函数体 diff 为空（仅 docstring 可改），证明改动收敛
   在数据侧。
4. CPU dry-run：`PYTHONPATH=. python3 scripts/w22_7_run_arm.py --arm b`
   在 `num_masks=2` 下跑通 40 步，loss 曲线趋势与 K=1 同量级——**只要求
   不发散，不要求更优**。K>1 的收益需要 Atlas pilot 200 步 + 探针指标才
   谈得上，CPU dry-run 的 40 步没有统计效力。
5. 记录 K=2 相对 K=1 的 `avg_step_s` 比值，作为 §7 风险 #6 的实测输入。

---

## §7 风险与对策

| # | 风险 | 对策 |
| --- | --- | --- |
| 1 | 种子公式改变导致 K=1 的 mask 序列漂移，W22.7 baseline 失效 | §5.1 #1 逐位对照 `default_rng(idx)`；`idx * 1 + 0 == idx` 恒成立 |
| 2 | 显存随 K 增长，Atlas 上 OOM | v1 只推荐 K=2；predictor 前向串行，峰值由单 mask 的 `h_tgt[mask]` 决定，可用 `mask_ratio` 下调对冲；`timing.ema_gather_peak_mb`（评审 §2.3 已落地）继续监控 |
| 3 | collator 静默降级到 K=1，K>1 配置形同虚设 | §3.2 混合批次显式 `ValueError`；§5.1 #9 钉死 |
| 4 | `bool_masked_pos_k` 被只认 `bool_masked_pos` 的下游误用 | 保留 `bool_masked_pos` 且语义不变（第 0 个 mask）；推理路径不经过 collator |
| 5 | K 个 mask 高度重叠，正则效果打折 | v1 不做互斥采样；§5.1 #3 只保证「不全同」。重叠度作为后续调优项，由 `token_variance` 指标反馈 |
| 6 | grad_accum × K 让单步过长，pilot 200 步跑不完 | backbone 开销不变、只有 predictor 翻倍；W22.7 的 `avg_step_s` 直接暴露，§6 #5 在 CPU dry-run 阶段先测 |
| 7 | 解除 `num_masks` 上限后，用户配 K=16 直接 OOM | v1 不设硬上限但在 YAML 注释与 §2.2 写明推荐 K=2；OOM 由风险 #2 的显存监控暴露，而非静默截断 |

---

## §8 落地顺序

1. **dataset**：`mask_count` 参数 + `masks` 键 + 校验 → 测试 #1-5。
2. **collator**：`bool_masked_pos_k` + 两条校验 → 测试 #6-9。
3. **objective**：`loss_fn` 分派 → 测试 #10-13。
4. **trainer**：解除拒绝 + 透传 → 测试 #14-16 + 改 §5.2 那一个既有测试。
5. 文案同步（§5.3）。
6. 全量 pytest + ruff + CPU dry-run → 单个 commit
   `feat(w22.8): multi-mask data pipeline (K>1 target sampling)`。

每一步保持仓库可测试状态；`_loss_for_masks` 函数体全程不动。

---

## §9 关联文件索引

| 文件 | 改动 |
| --- | --- |
| `worldfoundry_train/data/dataset.py` | `mask_count` 参数、`masks` 键、校验 |
| `worldfoundry_train/data/collator.py` | `bool_masked_pos_k` 输出、K 一致性/混合批次校验 |
| `worldfoundry_train/jepa_objective.py` | `loss_fn` mask 分派；`_loss_for_masks` 仅改 docstring |
| `worldfoundry_train/trainer.py` | `num_masks` 拒绝改下界校验；`_build_dataset` 透传 |
| `worldfoundry_train/tests/test_w22_8_multi_mask.py` | 新增，16 项 |
| `worldfoundry_train/tests/test_trainer_jepa.py` | **唯一改动的既有测试**（§5.2） |
| `worldfoundry_train/configs/vjepa2_vitl_1.2b.yaml` | `num_masks` 注释更新 |
| `worldfoundry_train/data/tubelet_mask.py` | 不动（K 个 mask 复用同一 `mask_fn`；`multiblock_mask` 接入见 §2.2 非目标） |
| `worldfoundry_train/model/predictor.py` | 不动（逐 mask 调用，接口已是 `(B, N)`） |
