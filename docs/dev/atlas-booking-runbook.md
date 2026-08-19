# Atlas 硬件窗口 Runbook（W20 → W5 → W39）

> 用途：Atlas 8×NPU 预约到位后，从「窗口开放」到「W20 baseline 入库」
> 的端到端操作清单。本 runbook 配套 `scripts/archive/w20/w20_atlas_smoke.sh`（W20
> 交付 #3，CPU-only 起草、syntax-clean，端到端需 NPU）。每一步都标注
>「如果失败怎么办」——不在 Atlas 上浪费时间做考古。

## 0. 预约到位时确认的事

| 项 | 取值 / 操作 | 备注 |
|---|---|---|
| 硬件 | Atlas 800 A2 / 910B ×8（8 卡 FSDP 用）/ ×1（W20 单卡） | W20 用 1 卡即可 |
| 容器 | `worldfoundry-base:phase1` | 与 W17 一致；含 torch_npu + vllm-ascend |
| 驱动 | `/usr/local/Ascend/driver` 需 RO 挂载 | W17 已验证 |
| 数据卷 | `/data`（videomix22m manifest + 权重） | W22 已有 subset manifest 路径 |
| 项目卷 | `$ROOT` 挂到容器的 `/workspace` | 与 W17 一致 |
| 预约时长 | W20 5-step smoke ≤ 10 min；100-step ≤ 30 min；预留 buffer | 1 卡预约即可 |

## 1. 启动前环境检查（容器内）

逐条确认，任一失败即不进入下一步：

```bash
# 1.1 NPU 可见
python -c "from worldfoundry_common import npu_env; assert npu_env.is_npu_available(), 'no NPU'"
python -c "import torch; print(torch.npu.device_count())"  # 期望 ≥1
# 1.2 关键包版本（与 worldfoundry-common 版本矩阵一致）
python -c "import torch_npu; print(torch_npu.__version__)"   # 期望 2.9.0
python -c "import vllm_ascend; print(vllm_ascend.__version__)"  # 期望 0.18.0
python -c "import transformers; print(transformers.__version__)"  # 期望 4.55.0
# 1.3 项目能 import（无 half-baked path）
python -c "import worldfoundry_train; from worldfoundry_train.device import resolve_device; print(resolve_device())"
# 1.4 数据 manifest 存在（如果走 atlas 模式而非 dry）
ls -la /data/videomix22m/manifest.csv          # 期望存在
```

预期输出：
- `resolve_device()` 返回 `"npu"`（即 `is_npu_available()` 为 True 且无 `WORLDFORGE_DEVICE` override）
- `torch.npu.device_count() == 1`（或目标卡数）
- transformers 4.55.x，torch_npu 2.9.x，vllm_ascend 0.18.x

**失败处置**：
- NPU 不可见 → 检查 `/dev/davinci_manager` 挂载是否正确；驱动版本是否与 torch_npu 匹配
- 版本错位 → 走 `worldfoundry-common` 矩阵文档重新对齐容器
- `resolve_device()` 返回 `cpu` → 检查 `is_npu_available()` 链路；G4 fail-fast 设计就是为了不静默退到 CPU

## 2. W20 单卡 LoRA smoke（5 步门槛 + 100 步基线）

```bash
# 2.1 直接跑（容器外调度，Docker 包装已内置）
bash scripts/archive/w20/w20_atlas_smoke.sh
# 或：已经进容器内
WORLDFORGE_NO_DOCKER=1 bash scripts/archive/w20/w20_atlas_smoke.sh
# 或：只跑 5 步门槛（不在 Atlas 上烧 100 步时间）
bash scripts/archive/w20/w20_atlas_smoke.sh --steps 5
```

预期产物：
- `bench_results/w20-preflight.log` — 预检 stderr
- `bench_results/w20-smoke-5.log` — 5 步 LoRA stderr
- `bench_results/w20-baseline-100.log` — 100 步 LoRA stderr
- `bench_results/atlas_baseline/single_card.json` — 结构化吞吐量基线

### 2.2 验收门

| 检查 | 通过条件 |
|---|---|
| Pre-flight rc | `0` |
| 5-step smoke rc | `0`，且 log 含 `[TRAIN] device=npu npu_available=True` 一行 |
| 5-step loss 下降 | 末步 loss < 首步 loss（任意单调即可；不要求显著） |
| 100-step baseline rc | `0` |
| `single_card.json` | `loss_decreased == true`、`avg_step_s` 合理（数量级 ~1s / step） |

### 2.3 失败处置

**5-step smoke 失败** → **不**进入 100-step。常见原因：

| 现象 | 处置 |
|---|---|
| `RuntimeError: load_model() must be called before ...` | 检查 `worldfoundry_train/device.py:to_device` 是否真的在 `strategy.wrap()` 之后被调；查看 trainer.py wrap 步骤的 diff |
| `torch_npu` import 失败 | 容器内重装；检查 `WORLDFORGE_NO_DOCKER=1` 是否生效 |
| OOM | 把 `--per-device-batch 1` 显式传给 YAML（或临时改 `configs/vjepa2_vitl_1.2b.yaml`）；不要现场改 lora 维度 |
| dtype 不匹配（bf16 vs fp32） | 看 trainer.py 的 dtype 分支；W22 已加 dtype cache |
| loss NaN | 检查 EMA momentum_end=1.0 在前几步的数值；W22 shape guard 应该已经 catch |

**100-step 失败（5-step 已 PASS）**：
- 多半是 scheduler / grad-accum 边界问题；保留 `w20-baseline-100.log` 末尾 60 行供事后分析
- `single_card.json` 不会被写入（脚本用 `set -uo pipefail`，失败则停）

### 2.4 拿到 baseline 后

把 `bench_results/atlas_baseline/single_card.json` 的数字抄进
`docs/dev/next-steps-phase4-atlas.md` 跟踪表 #1 行的 Status 列：

```
| 1 | W20 `.to("npu")` + Atlas 单卡 LoRA smoke | `<commit>` | **Atlas 上机 ✅; baseline: avg=<X>s/step loss=<Y>→<Z>** | Atlas single-card |
```

## 3. 进入 Phase-4 后面的工作

W20 PASS 后，按 `docs/dev/next-steps-phase4-atlas.md` 调度建议：

```
W20 ✅ → W26 (#2) 并行上机 → W21 (#3) 8卡 FSDP → W27 (#4) vLLM e2e
        → W29 (#5) Phase-3 集成回归
```

每一步的 runbook 在该步骤交付时再写（本文件是 W20 专用；不要把所有
W 的步骤挤进来——单文件聚焦单步骤，避免 runbook 自己 drift）。

## 4. 关闭 / 收尾

窗口结束时：
1. 把所有 `bench_results/w20*.log` 拉回本地仓库（git add + commit）
2. 把 `bench_results/atlas_baseline/single_card.json` 提交
3. 更新 `docs/dev/next-steps-phase4-atlas.md` 跟踪表 + CPU prep 行的
   status（"Atlas 上机待硬件" → "Atlas 上机 ✅; baseline: ..."）
4. 如果有意外 / 失败，把 stack trace + log 末尾 60 行贴到 runbook 末尾的
   「Lessons learned」一节（目前为空，下次窗口前补）
5. **运行 [`scripts/verify_goal_evidence.py`](../../scripts/verify_goal_evidence.py)**
   并把生成的 `goal_verdict.json` 一起 commit 上回（如果使用了
   `--keep-out-dir` 或显式 `--out-dir`）。这是 3 轴目标（RLinf +
   vLLM-Omni + ORT 扩散）的通用验证门 — Atlas 部署后必须 3/3 PASS，
   否则说明 worldfoundry 的 RLinf / 推理 / 扩散任一管线在 Atlas
   容器内出现回归。回归测试在
   [`worldfoundry_inf/tests/test_verify_goal_evidence.py`](../../worldfoundry_inf/tests/test_verify_goal_evidence.py)
   里覆盖（包括 diffusion 静默 fallback 到 PyTorch eager 的 critical
   回归守卫）。

## Out of Scope

- **W21 8 卡 FSDP** — 单独 runbook；多卡预约是另一个时长档（≥2h）。
- **DVPP 解码（W23）** — 不影响 G1；如果 W20 上发现视频解码是热点再补 runbook。
- **W26 W8A8 量化加载器** — 单独 runbook（与 W20 并行，不在 W20 路径里）。
- **W27 vLLM-Ascend e2e** — 需要 `worldfoundry_inf` 服务起来；单独 runbook。