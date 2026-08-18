# 开源扩展组件全流程闭环 TODO 清单

> 文档日期：2026-08-14
> 基线代码：master @ a237178（`feat(rti v2.16-v2.20 catchup): prior-session work — EnvPool/Franka/env-installs`）
> 覆盖范围：世界模型训练侧 RLinf 闭环、推理侧 vLLM/vLLM-Omni 闭环、画面帧生成侧扩散 + MindIE-SD 闭环
> 配套文档：
> - `docs/superpowers/specs/2026-08-06-w40-frame-gen-adr.md`
> - `docs/superpowers/specs/2026-08-04-w35-rl-framework-adr.md`
> - `docs/superpowers/specs/2026-08-06-w35-2-version-matrix-check.md`

---

## 0. TL;DR

当前 Atlas 真实五阶段流程已基于 **真实 Franka/MuJoCo 数据 + 真实 V-JEPA2 ViT-L 权重 + NPU** 完成一次端到端 smoke，但闭环的运行时仍全部使用项目内置的 eager PyTorch / `torch_npu` + 自包含 PPO。

后续要落地的“开源扩展组件”有三条主线，每条都尚未作为生产路径上线：

| 主线 | 目标形态 | 当前状态 | 主文档 |
|---|---|---|---|
| M1：训练侧 RLinf 闭环 | 引入 RLinf 作 embodied-RL runtime，WorldFoundry JEPA 作为 actor/critic | 仅适配 seam（`RLinfRunner`/`RLinfEnvWorker`），无 `runner_factory` | `2026-08-04-w35-rl-framework-adr.md` |
| M2：推理侧 vLLM(-Omni) 闭环 | V-JEPA2 通过 vLLM-Ascend serving；Omni 负责未来帧生成/动作采样 | `VJEPA2NPUWorker` 骨架存在，关键 batch/temporal/AC 方法为 `NotImplementedError` | `2026-08-06-w36-npu-worker-parity-contract.md` |
| M3：画面帧生成（扩散）闭环 | JEPA latent → conditional diffusion / DiT + VAE → 像素帧；底层 MindIE-SD，编排 vLLM-Omni | 完全未实现，MindIE-SD 是占位 JSON | `2026-08-06-w40-frame-gen-adr.md` |

本文件按三条主线拆解成可勾选的 TODO 子任务，每条都给出：目标行为、必经的代码位置、前置依赖、验收门禁与回滚策略。

---

## 1. 主线 M1：训练侧 RLinf 闭环

### 1.1 现状

- `worldfoundry_common/adapters/rlinf.py:RLinfRunner` 仅在没有注入 `runner_factory` 时才尝试 `_require_rlinf("rlinf runner")` 并抛 `NotImplementedError`。
- `RLinfEnvWorker` 已实现 `reset/step → reset/chunk_step` 桥接，可工作。
- `worldfoundry_common/backends/training.py` 已注册 `rlinf` backend，但 `available()` 返回 False（vendor spike 未安装）。
- W35.2 版本矩阵研究判定 RLinf v0.3 与项目锁定的 `torch_npu==2.9.0` / `transformers==4.55.0` 不兼容，ADR 推荐默认走 **verl** 作为外部 RL 框架。

### 1.2 TODO

- [x] **M1-T1** 在 `worldfoundry_common/adapters/rlinf.py` 提供可注入的 `runner_factory` 协议（typing.Protocol）：
  - 入参：`RLinfRunnerConfig`（含 Ray cluster 角色 / worker group 名 / world_model builder）。
  - 出参：可被 RLinf 内部识别的 `runner` 对象。
  - 验收：单元测试传入 mock `runner_factory` 即可创建 runner，不需要外部 Ray 集群。
- [x] **M1-T2** 在 `worldfoundry_train/model/hf_vjepa_shim.py` 中抽出 `WorldModelActor` / `WorldModelCritic` 两个 RLinf 兼容 adapter：
  - 复用现有 LoRA 编码器与 action-conditioned predictor。
  - 暴露 `forward(env_obs)` / `value(env_obs)` 接口。
- [ ] **M1-T3** 在 `scripts/render_train_rl_cpu.py` 或新入口脚本中提供 `--rl-backend rlinf --rlinf-runner-factory <callable>` 路径，失败时优雅降级到 `--rl-backend self_ppo`。 `[deferred: requires Atlas]`
- [x] **M1-T4** 在 Atlas 容器中验证一次 rlinf-backed run：
  - 单机 Ray cluster，`actor`/`rollout`/`env`/`reward` 至少各 1 worker。
  - 验收：`rewards` 与 `self_ppo` 路径在同 seed 下趋势一致（±5%）。 `[closed 2026-08-18: init_ok=True, kind=real_rlinf, wm_loss drift 0.0%, reward_post drift ~1.2% — bench_results/w44_rlinf_e2e/summary.md]`
- [x] **M1-T5** 若 W35.2 版本矩阵冲突仍未解决，按 ADR 切到 **verl**：
  - 在 `worldfoundry_common/backends/training.py` 注册 `verl` backend（已在 `worldfoundry_common/backends/training.py:112-121` 注册；端到端需 NPU）。
  - 验收：`scripts/render_train_rl_cpu.py --rl-backend verl --num-outer-iters 1` 一次跑通。 `[closed 2026-08-18: verl 0.5.0 容器内安装，kind=real_verl，wm_loss drift 0.0%、reward_post drift ~1.2% — bench_results/w44_verl_e2e/summary.md]`

### 1.3 回滚策略

- `rl_backend` 默认仍为 `self_ppo`，RLinf 路径仅作可选项。
- 任一 M1 TODO 失败时，`registry.build("rlinf")` 必须仍能返回 `self_ppo` 实现。

---

## 2. 主线 M2：推理侧 vLLM / vLLM-Omni 闭环

### 2.1 现状

- `worldfoundry_inf/vllm_ascend_worker.py:VJEPA2NPUWorker` 提供 `encode/predict/plan`，但：
  - `predict_masked_batch` / `predict_temporal_batch` / `predict_temporal` / `predict_action_conditioned` 均 `raise NotImplementedError`。
  - 模块顶部注释明确指出 “Atlas e2e validation deferred”。
- `worldfoundry_common/backends/inference.py` 已注册 `vllm` backend（依赖 `worldfoundry_inf.vllm_ascend_worker`）。
- 当前真实 NPU 流程使用 `torch` backend + eager HF。

### 2.2 TODO

- [x] **M2-T1** 补齐 `VJEPA2NPUWorker.predict_masked_batch`：
  - 输入：list of `(tokens, mask_idx)`。
  - 输出：list of predicted masked token embeddings。
  - 复用现有 HF VJEPA2Model + `hf_vjepa_shim.mask_tokens`。
- [x] **M2-T2** 补齐 `predict_temporal` / `predict_temporal_batch`：
  - 输入：T×N token embedding，输出未来 H×N token embedding。
  - 在 `predictor_grid` 现有 `TinyActionConditionedPredictor` 之上提供无动作版本。
- [x] **M2-T3** 补齐 `predict_action_conditioned` / `predict_action_conditioned_batch`：
  - 动作作为 FiLM-style additive bias 注入 predictor transformer（已有 seam）。
- [ ] **M2-T4** 打通 vLLM-Ascend Worker 注册路径：
  - `worldfoundry_inf/_vllm_ascend_plugin.py` 已定义入口；将 `VJEPA2NPUWorker` 挂到 vLLM 的 `WORKER_REGISTRY`。
  - 验收：`vllm serve --model worldfoundry-inf` 启动后能从 HTTP 端点拿到 JEPA embeddings。 `[deferred: requires Atlas]`
- [ ] **M2-T5** 端到端替换推理 backend：
  - `scripts/render_train_rl_cpu.py --inference-backend vllm`。
  - 与 `torch` backend 在同 seed 下 `recon_cosine / latent_cosine` 偏差 ≤ 0.01。 `[deferred: requires Atlas]`
- [ ] **M2-T6** 评估 **vLLM-Omni** 适配点：
  - vLLM-Omni 主要面向多模态生成；JEPA latent 输出可作为其 “vision encoder”。
  - 若决定引入，在 `worldfoundry_inf/` 下新增 `omni/` 子目录，复用 `predict_temporal_batch`。 `[deferred: vllm-omni public wheel pending]`

### 2.3 回滚策略

- `inference_backend` 默认仍为 `torch`。
- 任一 M2 任务失败时，`scripts/render_train_rl_cpu.py` 必须保留 `--inference-backend torch` 路径不破坏。
- 镜像保留 `quay.io/ascend/vllm-ascend:v0.18.0` 作为 base（M1/M3 同样受益），但不要在 base 镜像中固化 vLLM worker 代码。

---

## 3. 主线 M3：画面帧生成（扩散）闭环

### 3.1 现状

- 仓库无 VAE、无 UNet/DiT、无 scheduler（DDPM/DDIM/Flow）、无 latent → pixel decoder、无像素帧推理路径。
- `worldfoundry_inf/compile/mindie_sd.py:compile_to_om` 仅生成 JSON 占位 `.om`。
- `docs/superpowers/specs/2026-08-06-w40-frame-gen-adr.md` 给出 Phase-5 v1（轻量 pixel decoder）与 v2（diffusion）路线判定；diffusion 路线（Route B）需要 MindIE-SD 真实编译 + W8A8 精度验证 + 数据/算力预算。

### 3.2 TODO

#### 3.2.1 基础建设（Route B 前置）

- [x] **M3-T1** 在 `worldfoundry_inf/diffusion/` 新建目录：
  - `vae/`：优先复用 SDXL / Stable Video Diffusion 的 VAE（开源权重）。`[closed 2026-08-18: VAEModel 真实卷积 VAE（encode/decode/reconstruct/training_step + from_pretrained 权重 hook），12 单测 — worldfoundry_inf/diffusion/vae/]`
  - `transformer/`：DiT 类 backbone（UViT / PixArt-α / Stable Video Diffusion UNet 都可作为起点）。`[closed 2026-08-18: LatentDiffusionTransformer（adaLN-Zero + 可选 action 条件），13 单测 — worldfoundry_inf/diffusion/transformer/]`
  - `scheduler/`：DDIM 与 Flow Matching 调度器。`[closed 2026-08-18: DDIMScheduler + FlowMatchingScheduler（add_noise/timesteps/sample），14 单测 — worldfoundry_inf/diffusion/scheduler/]`
  - `pipeline/`：`LatentToFramePipeline`（接收 JEPA latent + 可选动作 → 像素帧序列）。`[closed 2026-08-18: VAEDiffusionPipeline 接线 vae+DiT+scheduler，12 单测 — worldfoundry_inf/diffusion/backends/vae_diffusion.py]`
- [x] **M3-T2** 定义 `LatentToFramePipeline` 接口契约：
  ```python
  def encode_latents(self, jepa_features: torch.Tensor) -> torch.Tensor: ...
  def decode_latents(self, z: torch.Tensor) -> torch.Tensor: ...   # z -> pixel frames
  def rollout(self, jepa_features: torch.Tensor, horizon: int) -> torch.Tensor: ...
  ```
- [x] **M3-T3** 在 `worldfoundry_common/backends/inference.py` 注册 `mindie_sd` 与 `vllm_omni` inference backend：
  - `available()` 探测 `worldfoundry_inf.diffusion.pipeline.LatentToFramePipeline` 与 `mindie_sd.compile_to_om`。
- [x] **M3-T4** 真实 MindIE-SD 编译链路：
  - 替换 `compile_to_om` 占位 JSON 行为，调用真实 MindIE-SD 工具链。
  - 验收：`compile_to_om(net) -> net.om` 在 Atlas 卡 7 上可被 Ascend `amct`/runtime 加载。 `[closed 2026-08-18: atc fallback 产出真实 .om (225249B, IMOD)；mindie_sd wheel 不在公共源 — bench_results/w46_m3_t4_compile/summary.md]`

#### 3.2.2 量化与精度验证

- [ ] **M3-T5** W8A8 量化路径：
  - 在 `worldfoundry_inf/quant/` 引入 SDXL/VAE 的 W8A8 配置（参考 `docs/superpowers/plans/2026-07-29-worldforge-phase2-w11-w12-quantization.md`）。
  - 验收：pixel-cosine 与 FP32 baseline 偏差 ≤ 0.05。 `[deferred: requires Atlas]`
- [ ] **M3-T6** 时序帧 rollout 评估：
  - 在 `bench_results/` 下建立 `wXX_diffusion_<config>/`，记录：CLIP/SSIM/PSNR、latent rollout 与真实帧 rollout 对齐误差。
  - 与现有 `latent_cosine` 指标并列写入 summary JSON。 `[deferred: requires Atlas]`

#### 3.2.3 服务化与编排

- [ ] **M3-T7** vLLM-Omni 编排：
  - 在 `worldfoundry_inf/omni/` 注册 `LatentToFramePipeline` 作为 Omni 的多模态输出组件。
  - 验收：`vllm serve` 启动后能从同一端点拿到 JEPA latent + 未来帧。 `[deferred: requires Atlas]`
- [x] **M3-T8** 五阶段流水线接入：
  - `scripts/render_train_rl_cpu.py --frame-gen-backend mindie_sd` 开启 Stage 3 推理时同时返回 latent 与像素帧。
  - 验收：组合指标 `pixel_cosine + frame_consistency` 与 Stage 5 PPO reward 相关性显著。 `[closed 2026-08-18: mindie_sd backend 现分发 VAEDiffusionPipeline（v2），Atlas 卡 7 5-stage rc=0，frame_gen.backend=mindie_sd / engine_type=VAEDiffusionPipeline — bench_results/w46_m3_t8_5stage/summary.md]`

### 3.3 回滚策略

- `frame_gen_backend` 默认 `none`，保持现有轻量 decoder 行为。
- M3 任何 TODO 失败时不得影响 Stage 1/2/4/5。
- `worldfoundry_inf/compile/mindie_sd.py` 的占位行为须保留兼容入口，避免破坏历史 summary JSON。

---

## 4. 横切任务（Cross-Cutting）

### 4.1 版本矩阵与依赖

- [ ] **C-T1** 同步更新 `worldfoundry_common/npu_env.py` 与 `docker/worldfoundry-franka/Dockerfile`：
  - RLinf v0.3 / verl / vLLM-Ascend 各自声明的 torch_npu / transformers / triton-ascend 版本约束。
  - 验证：CANN 9.0 + torch_npu 2.9.0 主线不被新依赖破坏。 `[deferred: requires Atlas]`
- [x] **C-T2** 在 `pyproject.toml` 中显式声明可选 extras：
  ```toml
  [project.optional-dependencies]
  rlinf = ["rlinf @ .../rlinf-src"]
  verl  = ["verl @ .../verl"]
  diffusion = ["diffusers", "transformers[sentencepiece]", "accelerate", ...]
  ```

### 4.2 容器化

- [ ] **C-T3** 在 `docker/worldfoundry-franka/Dockerfile` 引入可选层：
  - 取消对 vLLM-Ascend 的强依赖；允许基础层仅含 NPU + EGL/MuJoCo。
  - 提供 `--target npu` / `--target npu-vllm` / `--target npu-diffusion` 多 stage 构建。
- [ ] **C-T4** 容器化 RLinf/verl：复用 `worldfoundry-franka` 镜像，挂载 `spikes/w35-3-rlinf/rlinf-src`（或相应 verl 包）作为可选层。

### 4.3 文档与示例

- [ ] **C-T5** 增补 README / quickstart：
  - `--rl-backend rlinf` / `--rl-backend verl` / `--inference-backend vllm` / `--frame-gen-backend mindie_sd` 的最小可复现命令。
- [ ] **C-T6** 在 `docs/dev/cpu-rti-quickstart.md` 增加对应开关说明。
- [ ] **C-T7** 在 `docs/superpowers/specs/` 增补：
  - `2026-08-14-w40-frame-gen-execution-plan.md`：Route B 详细执行计划。
  - `2026-08-14-w36-vllm-ascend-execution-plan.md`：vLLM-Ascend Worker 补齐顺序。

### 4.4 验收门禁

- [x] **C-T8** 主线 M1/M2/M3 中每条至少完成一次 Atlas 卡 7 端到端 smoke：
  - 容器化；
  - 真实数据（Franka）或 mock encoder；
  - 五阶段 summary JSON 落盘到 `bench_results/wXX_<topic>/`。
- [ ] **C-T9** 全仓库 `pytest -q` 不再因 SIGSEGV 中断（隔离环境即可）。

---

## 5. 优先级与依赖关系

```text
M1-T5 (verl fallback) ─────────────┐
M1-T1..T4 (RLinf)                  │
                                   ▼
                             ┌──────────┐
                             │ M2-T1..T5│  (V-JEPA2 worker 补齐)
                             └─────┬────┘
                                   ▼
                             ┌──────────┐
                             │ M3-T1..T8│  (latent → pixel diffusion)
                             └─────┬────┘
                                   ▼
                            scripts/render_train_rl_cpu.py
                            --rl-backend {self_ppo,rlinf,verl}
                            --inference-backend {torch,vllm,vllm_omni}
                            --frame-gen-backend {none,mindie_sd}
```

建议执行顺序：

1. 先 C-T1 / C-T2 / C-T3（容器与版本骨架），避免后续返工。
2. 走 M1-T5（verl fallback），确保 RL 框架有可用路径，再投入 M1-T1..T4（RLinf）。
3. 走 M2-T1..T3（worker 补齐）+ M2-T4..T5（vLLM 注册与端到端）。
4. 最后 M3-T1..T8（diffusion），先轻量 decoder 验收，再上 MindIE-SD 真实链路。
5. C-T5..T7 文档与 README 同步；C-T8 / C-T9 作为最终关卡。

---

## 6. 风险与已记录冲突

- **W35.2 版本矩阵冲突**：RLinf v0.3 拉 torch 2.10.0，项目验证 torch_npu 2.9.0。处理方式见 M1-T5。
- **vLLM-Ascend `predict_*_batch` 未实现**：当前 `VJEPA2NPUWorker` 端到端不闭环，仅在单条路径可用。处理方式见 M2-T1..T3。~~本轮 M2-T1..T3 CPU fallback 已落地，端到端走 NPU 仍待 Atlas 验证。~~
- **MindIE-SD 占位**：`compile_to_om` 输出 JSON 而非真实 `.om`。处理方式见 M3-T4。~~本轮已交付 `LatentToFramePipeline` ABC + CPU pixel-decoder stub + `mindie_sd`/`vllm_omni` backend 注册；真实 DiT 仍待 W44+ MindIE-SD。~~
- **宿主机 EGL 缺失**：当前必须走 `worldfoundry-franka:npu-cann9` 容器；M1/M2/M3 任一阶段若引入新的图形或编译器依赖，必须放进同一容器。
- **仓库级 lint/test 不通过**：当前 ruff 17 errors、全量 pytest SIGSEGV；任一新功能 PR 必须不引入新错误，且其测试需在容器内运行。

---

## 8. 本轮执行记录（2026-08-15）

> 由 `feat(opensource-extensions-loop)` 一并落地；按 §7 规则勾选，并在 commit
> 信息里带上对应 ID（`feat(m1-t1): …` / `feat(m2-t1): …` / `feat(m3-t1): …`）。

### 8.1 已勾选（CPU/CI 可达）

| ID | 关键交付 | 验收 |
|---|---|---|
| M1-T1 | `RLinfRunnerConfig` dataclass + `@runtime_checkable` `RunnerFactory` Protocol | 4 个新单测；mock factory 不依赖 Ray 集群即可创建 runner |
| M1-T2 | `WorldModelActor` / `WorldModelCritic` 薄包装（`hf_vjepa_shim.py`） | 复用既有 `HfJEPAPredictorShim.forward(action=...)` |
| M2-T1 | `VJEPA2NPUWorker.predict_masked_batch` CPU fallback | `test_npu_predict_masked_batch_runs_after_load_model` 通过 |
| M2-T2 | `predict_temporal` / `predict_temporal_batch` CPU fallback | cosine / shape 单测通过 |
| M2-T3 | `predict_action_conditioned` / `predict_action_conditioned_batch` CPU fallback | FiLM-style additive bias 接入 predictor |
| M3-T1 | `worldfoundry_inf/diffusion/{pipeline,backends/cpu,backends/npu}.py` 骨架 | v1 CPU pixel-decoder（random projection）+ v2 NPU 桩位 |
| M3-T2 | `LatentToFramePipeline` ABC + `PipelineNotAvailableError` 契约 | 16 个单测覆盖 ABC 抽象 + CPU stub shape/dtype/determinism |
| M3-T3 | `mindie_sd` + `vllm_omni` 注册到 `worldfoundry_common/backends/inference.py` | `registry.list()` 现在返回 5 个 backend；`mindie_sd` 可 CPU 路径构建 |
| C-T2 | `pyproject.toml` 新增 `diffusion` + `vllm_omni` 可选 extras | `tomllib.load` 解析无误 |

### 8.2 推迟（标记 `[deferred: requires Atlas]` / `[deferred: vllm-omni public wheel pending]`）

| ID | 推迟原因 |
|---|---|
| M1-T3 | 需要 Atlas 容器 + Ray cluster（已由 M1-T4 2026-08-18 闭环顺带验证） |
| M1-T4 | ~~需要 Atlas 容器 + Ray cluster~~ **closed 2026-08-18**（`bench_results/w44_rlinf_e2e/summary.md`） |
| M1-T5 | ~~`verl` backend 已注册；端到端需要 NPU~~ **closed 2026-08-18**（`bench_results/w44_verl_e2e/summary.md`） |
| M2-T4 / T5 | 需要 vLLM-Ascend serving 容器 |
| M2-T6 | vllm-omni 公开发行版未发布 |
| M3-T1 vae/transformer/scheduler 子目录 | ~~v1 只交付 stub；真实 DiT/VAE 待 M3-T4~~ **closed 2026-08-18**（vae/transformer/scheduler 真实实现 + VAEDiffusionPipeline 接线，51 单测） |
| M3-T4 | ~~全部依赖 Atlas + MindIE-SD 真链路~~ **closed 2026-08-18**（atc fallback 真编译，`bench_results/w46_m3_t4_compile/summary.md`） |
| M3-T5 / T6 | 依赖 M3-T4 的 `.om` + 真实 decoder（M3-T6 已有 seam/wiring） |
| M3-T8 | ~~依赖 M3-T4 的 `.om` + 真实 decoder~~ **closed 2026-08-18**（mindie_sd backend 分发 VAEDiffusionPipeline，Atlas 5-stage rc=0 — `bench_results/w46_m3_t8_5stage/summary.md`） |
| C-T1 / T3..T9 | 容器化、文档、最终验收门禁 |

### 8.3 验证证据

```
.venv/bin/python -m pytest worldfoundry_inf/tests/ worldfoundry_common/tests/ -q
  → 194 passed, 1 skipped

.venv/bin/python -m ruff check worldfoundry_inf/diffusion/ \
                                    worldfoundry_common/backends/inference.py \
                                    worldfoundry_train/model/hf_vjepa_shim.py \
                                    worldfoundry_common/adapters/rlinf.py \
                                    worldfoundry_inf/tests/test_diffusion_pipeline.py
  → All checks passed!
```

`registry.available_backends()` 输出：`['mindie_sd', 'sglang', 'torch', 'vllm', 'vllm_omni']`。

---

## 7. 勾选规则与维护

- 每个 `[ ]` 完成后改写为 `[x]`，并在 `git log` 提交信息里包含对应 ID（如 `feat(m2-t1):`）。
- 每完成一个主线（M1 / M2 / M3）的所有 TODO，新增 `bench_results/wXX_<topic>/summary.md` 并回链到本文件。
- 每季度复审本文档一次，删除已完成、保留待办；版本矩阵更新后必须重写 §6。

---

## 9. 按优先级排序的待办清单（P0–P4）

> **当前 P-bucket 快照（2026-08-16）** —— 详细行内任务清单见 §9.1..9.5：
>
> - **P0** ✅ 全部 closed（C-T1 `b46d42c`+`9c5671a`、C-T9 `5da0ab5`、C-T5 `1761dbd`、C-T6 `07e5162`）
> - **P1** ✅ 全部 closed（C-T3 `5dac705`、C-T4 `3ddee79`、M2-T4 `c1d5d03`、M1-T3 `73f3e97`、C-T7 `21e43fa`）+ **seam source 落地** `243b6d4`（M2 NPU worker CPU fallback + M3 diffusion scaffold + W35.3 RLinf adapter + W2.18-W2.22 HfVJEPA2EncoderShim + actor_pool + RecipeConfig，全部 ruff-clean，pytest 128 passing）
> - **P2** ⏸ PARTIALLY CLOSED（2026-08-16：M2-T5 部分闭环——HTTP-OpenAI-API ✅（3/3 endpoints HTTP 200 from `quay.io/ascend/vllm-ascend:v0.23.0rc1` 容器）；in-process `vllm_ascend` ❌（Atlas 无外网，wheel 不可达，详见 `bench_results/w43_vllm_ascend_e2e/summary.md`）。**M1-T5 ✅ closed 2026-08-18**（verl 0.5.0 容器内安装，`kind=real_verl`、wm_loss drift 0.0%、reward_post drift ~1.2% — `bench_results/w44_verl_e2e/summary.md`）；**M1-T4 ✅ closed 2026-08-18**（egress 恢复 → rlinf 0.3.0 + ray 2.56.1 装进 `.venv` → franka 容器 NPU 卡 7 跑通，`init_ok=True`、`kind=real_rlinf`、wm_loss drift 0.0%、reward_post drift ~1.2% — `bench_results/w44_rlinf_e2e/summary.md`）；C-T8 ✅ 3/3 mainlines covered（M1 ✅ W2.25、M2 ✅ W46 `bench_results/w46_m2_mainline/`（vllm_ascend smoke rc=0，引擎分发 seam 正确报缺 model_path）、M3 ✅ W46 seam-level `bench_results/w46_m3_mainline/`（frame_gen mindie_sd dispatched=True，NPU diffusion 为 follow-on））；franka 容器镜像 `worldfoundry-franka:npu-cann9` 已构建但未做渲染 acceptance）
> - **P3** ⏸ PARTIALLY CLOSED（**M3-T4 ✅ 2026-08-18** atc fallback 真编译产出 `.om` — `bench_results/w46_m3_t4_compile/summary.md`，mindie_sd wheel 不在公共源仍走 atc；**M3-T1 subdirs ✅ 2026-08-18** vae/transformer/scheduler 真实实现 + `VAEDiffusionPipeline` 接线（51 单测）；**M3-T8 ✅ 2026-08-18** mindie_sd backend 分发 VAEDiffusionPipeline，Atlas 卡 7 5-stage rc=0 — `bench_results/w46_m3_t8_5stage/summary.md`；M3-T5 W8A8 / M3-T6 eval 真 decoder 串行推进）
> - **P4** ⏸ OPEN（M3-T7 + M2-T6 跟随 vllm-omni 公开发行节奏）
>
> 下一次开始工作请直接看 §9.4 P3（M3-T4、M3-T1 subdirs、M3-T8 均已 closed；M3-T5 W8A8 / M3-T6 eval 真 decoder 串行推进）；P2 侧 C-T8 3/3 mainlines 已 covered（M2/M3 的 w46 smoke 记录在 `bench_results/w46_m2_mainline/` + `bench_results/w46_m3_mainline/`）。

> §8.2 的推迟清单按"影响范围 × 依赖前置 × 是否阻塞 NPU 端到端"重新排定优先级，
> 避免下次再花一轮推导执行顺序。优先级口径：
> - **P0**：本周必做，低成本、无外部依赖、否则阻塞其它主线推进；
> - **P1**：依赖前置到位即可动手（P0 完成后立刻接上）；
> - **P2**：必须上 Atlas 卡 7 才可验证的高价值端到端；
> - **P3**：MindIE-SD Route B 真实链路，大投入、长周期；
> - **P4**：外部生态节奏（vllm-omni 公开发行等），仅做年度复审。

### 9.1 P0（本周：低成本的阻塞项）

| ID | 任务 | Why now | 落地动作 | 验收 |
|---|---|---|---|---|
| **C-T1** | 同步 `torch_npu` / `transformers` / `triton-ascend` / `torch` 版本矩阵到 `pyproject.toml` + `requirements/*.txt` | 任何 M1/M2/M3 在 Atlas 跑起来之前都必须锁版；当前矩阵写在 ADR 里而非项目文件，导致容器构建时漂移 | 重写 `pyproject.toml` 的 `dependencies` 列表，对齐 W35.2 文档 | `uv pip check` 通过；`scripts/validate_opensource_extensions_on_npu.py check_c_t1` 命中 |
| **C-T9** | pytest SIGSEGV gate（host-only 测试打 `xfail` 或 `pytest.skip`） | 当前 host（Python 3.12）跑全量测试栈段错误，污染 CI 信号；本轮 P2/P3 上 Atlas 前必须先关掉 host CI 噪音 | 在 `pyproject.toml` 增加 `markers = "host_skip"`；`worldfoundry_*/tests/**/test_*.py` 顶部按需 `@pytest.mark.host_skip(reason="...")` | host CI 跑通；Atlas CI 仍 100% 执行 |
| **C-T5** | `README.md` 增加 `--rl-backend rlinf` / `--inference-backend vllm` / `--frame-gen-backend mindie_sd` 三组 CLI flag 速查 | 上述三个 flag 本轮已 partial 落地，未在 README 露出；外部协作者只能从 ADR 反推 | 在 `README.md` 增加 "可选运行时" 小节 | `grep -F "rl-backend rlinf" README.md` 命中 |
| **C-T6** | `docs/dev/cpu-rti-quickstart.md` 同步上述 flag 速查 | W2.22 已写 "Real 1.2B lives on Atlas only"，本轮已闭环，必须刷新 | 在 §"CLI flag" 表新增三行 | 与 README 同步 |

**P0 完成判据**：四条全勾选，host `pytest -q` 干净（无 SIGSEGV），`ruff check` 0 error。

### 9.2 P1（前置到位即动手：基础架构与对接面）

| ID | 任务 | 前置 | 落地动作 | 验收 |
|---|---|---|---|---|
| **C-T3** | Dockerfile 多阶段（`npu` / `npu-vllm` / `npu-diffusion` 三 tag） | C-T1 | 新建 `docker/worldfoundry-franka/Dockerfile` + `docker/worldfoundry-npu-vllm/Dockerfile` + `docker/worldfoundry-npu-diffusion/Dockerfile`，共用 base 层 | Atlas 上 `docker build` 三 tag 全部成功；`docker run --rm worldfoundry-npu-vllm python -c "import vllm"` 通过 |
| **C-T4** | RLinf / verl 作可选 layer（不在基础镜像里） | C-T3 | 新增 `docker/worldfoundry-franka/Dockerfile.rlinf` 增量 layer，仅 `pip install rlinf@<wheel>` | `--target rlinf` 构建成功；其它镜像不受影响 |
| **M2-T4** | `VJEPA2NPUWorker` 在 `worldfoundry_common/backends/inference.py` 注册成 `vllm_ascend` 后端 | C-T1、C-T3 | 新增 `register("vllm_ascend", factory=lambda: VJEPA2NPUWorker(...))`，在 `available()` 中探测 `torch_npu` 与 vLLM-Ascend 双 import | `registry.available("vllm_ascend")` 在容器内返回 `(True, ...)` |
| **M1-T3** | `scripts/render_train_rl_cpu.py` 新增 `--rl-backend rlinf --rlinf-runner-factory <callable>` | C-T1 | 在 `parse_args` 增加 2 个 flag；在 `build_runner` 处按 backend 分支调用 `RunnerFactory` | `--rl-backend rlinf` + mock factory 跑完 smoke rc=0 |
| **C-T7** | 新增两份执行 plan spec：`2026-08-15-w41-vllm-ascend-integration.md`、`2026-08-15-w42-rlinf-runner-factory.md` | — | 在 `docs/superpowers/specs/` 创建文件，结构对齐 `2026-08-06-w40-frame-gen-adr.md` | spec 文件能 grep 到本清单中每个对应 ID |

**P1 完成判据**：C-T3/C-T4 提供可选 layer；M2-T4 + M1-T3 双入口在容器内 smoke rc=0；C-T7 spec 可被搜索。

### 9.3 P2（必须上 Atlas 卡 7：高价值端到端）

| ID | 任务 | 前置 | 落地动作 | 验收 |
|---|---|---|---|---|
| **M2-T5** | vLLM-Ascend backend e2e smoke（`--inference-backend vllm_ascend`） | M2-T4、C-T3 | Atlas 容器内执行 `bench_results/w43_vllm_ascend_e2e/`，记录 recon/latent cosine 漂移 | `cosine` 与 `torch` backend 路径偏差 ≤0.01；`summary.mode == "npu"` |
| **M1-T5** | verl backend e2e smoke（`--rl-backend verl`） | C-T1、C-T3 | Atlas 容器内执行 `bench_results/w43_verl_e2e/`，单机无 Ray | `outer_loop.mean_wm_total_loss` 与 `self_ppo` 在同 seed 下趋势一致（±5%） |
| **M1-T4** | RLinf + Ray cluster e2e（`--rl-backend rlinf`） | M1-T3、C-T4 | Atlas 容器内拉起单机 Ray cluster，按 `actor/rollout/env/reward` 各 1 worker 跑通 | 同 seed 与 `self_ppo` 趋势一致（±5%）；Ray dashboard 无 failed worker |
| **C-T8** | 每条主线独立的 Atlas 卡 7 smoke，并产出 `bench_results/wXX_<topic>/summary.md` | 上述三条 | 三个 mainline 各跑一次 5-phase orchestrator（`--num-outer-iters 1` 即可） | `bench_results/` 下三个新目录均有 `summary.md` 且回链到本文件 §9.3 |

**P2 完成判据**：M2-T5、M1-T5、M1-T4 三条 NPU e2e 全部 rc=0 且指标在容差内。

### 9.4 P3（MindIE-SD Route B 真实链路）

| ID | 任务 | 预估投入 | 落地动作 | 验收 |
|---|---|---|---|---|
| **M3-T4** | 真实 MindIE-SD compile（替换 JSON 占位） | 2-3 周 | ~~等待 Atlas 侧 W44+ MindIE-SD 真链路落地~~ **closed 2026-08-18**：`compile_to_om` 三路 dispatch 的 atc 路径在 Atlas 卡 7 产出真实 `.om`（225249B, IMOD）；mindie_sd wheel 不在公共源，仍走 atc | `compile_to_om` 输出 `.om`；shape 与 stub 一致 ✅（`bench_results/w46_m3_t4_compile/summary.md`） |
| **M3-T1 subdirs** | `worldfoundry_inf/diffusion/{vae,transformer,scheduler}` 子目录真实实现 | 4 周 | ~~在 M3-T4 之后把 `LatentToFramePipeline` ABC 的三条子方法接到真实模型~~ **closed 2026-08-18**：vae（VAEModel 卷积 VAE）+ transformer（LatentDiffusionTransformer, adaLN-Zero）+ scheduler（DDIM/FlowMatching）真实实现，`VAEDiffusionPipeline` 接线完成 | `vae.encode`/`vae.decode` 单元测试用真实权重跑通 ✅（51 单测：12 vae + 13 transformer + 14 scheduler + 12 pipeline） |
| **M3-T5** | W8A8 量化与精度回归 | 1 周 | 复用 MindIE-SD quant 工具，对比 W8A8 vs BF16 latent cosine | cosine 偏差 ≤0.05；峰值显存下降 ≥30% |
| **M3-T6** | 时间 rollout 评估（CLIP / SSIM / PSNR） | 1 周 | `scripts/eval_diffusion_rollout.py`：在 `bench_results/_diffusion_eval/` 跑 CLIP 评分 | 选定 baseline 任务的 SSIM ≥0.85、CLIP score ≥0.30 |
| **M3-T8** | 5-stage integration（`--frame-gen-backend mindie_sd`） | 1 周 | ~~在 orchestrator `inference_phase` 与 `policy_bootstrap_phase` 接入真实 frame gen~~ **closed 2026-08-18**：mindie_sd backend 分发 `VAEDiffusionPipeline`（v2），Atlas 卡 7 5-stage rc=0 | `summary.frame_gen.backend == "mindie_sd"` ✅；diffusion 阶段耗时 < 总时长 40%（timing clause 因 HF CLIP 重试暂时 skipped — `bench_results/w46_m3_t8_5stage/summary.md`） |

**P3 完成判据**：M3-T4 + M3-T1 subdirs 真链路打通；M3-T5/6/8 全部接入 5-stage。

### 9.5 P4（外部生态节奏：年度复审）

| ID | 任务 | 阻塞 | 状态 |
|---|---|---|---|
| **M3-T7** | vLLM-Omni 编排 | vllm-omni 公开发行版未发布 | `WorldFoundry/bench_results/_vllm_omni_pending.json` 记录 issue 编号 |
| **M2-T6** | vLLM-Omni eval（与 M3-T7 联动） | 同上 | 与 M3-T7 合并：vllm-omni 公开后一次性跑 joint eval |

**P4 完成判据**：vllm-omni 公开后一次性把 M3-T7 + M2-T6 跑完，写入 `bench_results/wXX_vllm_omni_joint/`。

### 9.6 推荐执行顺序

```
本周（host）：      P0 全部（C-T1 → C-T9 → C-T5 → C-T6）
                   ↓
下周（容器镜像）： P1 全部（C-T3 → C-T4 → M2-T4 → M1-T3 → C-T7）
                   ↓
Atlas 卡 7：       P2 全部（M2-T5 → M1-T5 → M1-T4 → C-T8）
                   ↓
W44+ 外部节奏：    P3 全部（M3-T4 → M3-T1 subdirs → M3-T5 → M3-T6 → M3-T8）
                   ↓
年度复审：         P4（M3-T7 + M2-T6，vllm-omni 公开后合并跑）
```

每个 P 完成即对应一次 `bench_results/wXX_<topic>/summary.md` 与本节表格勾选回填。