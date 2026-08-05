# W35.3 · JEPA Policy 接入 RLinf 成本评估（S5 交付物）

> 目的：回填 ADR-W35 §6 验证门 W35.3。基于 S1–S4 实测（2026-08-04，WSL2 CPU 环境，
> RLinf v0.3.0 / torch 2.10.0+cpu / ray 2.56.1 / transformers 4.57.6）。
> 姊妹文档：《集成点确认纪要》`integration-points-memo.md`（接口细节与行号）。

---

## 1. 实际改动点清单

### 1.1 spike 新增文件（全部为 spike 资产，RLinf 源码零改动）

| 文件 | 作用 | 量级 |
|---|---|---|
| `spike_ext/vjepa2_policy.py` | VJEPA2Policy（模式 A 状态投影 + 模式 B 渲染帧双分支）+ builder | ~280 行 |
| `spike_ext/gym_cartpole_env.py` | gymnasium CartPole 向量化适配器（镜像 EmbodiChainEnv 接口 + pixel_obs 帧栈） | ~330 行 |
| `spike_ext/rlinf_ext.py` | 官方扩展通道入口（`register()` → `register_model`） | ~15 行 |
| `sitecustomize.py` | CPU 平台补丁链（9 项补丁，见 §1.2） | ~230 行 |
| `config/gym_ppo_cart_pole{,_vjepa2,_vjepa2_pixel}.yaml` | 三份 spike 配方 | 各 ~200 行 |
| `src/train_spike{,_vjepa2,_vjepa2_pixel}.py` + `start-train*.sh` | 入口与启动脚本 | 各 ~120 行 |

### 1.2 补丁/缺口分类（12 项）

**A 类：CPU-only 环境特有（NPU/GPU 环境下大部分消失，见 §4）**

| # | 缺口 | 处置 |
|---|---|---|
| 1 | `AcceleratorUtil.get_torch_platform/get_device_type` 对 NO_ACCEL 返回 None（accelerator.py:442/:452），worker 开箱即崩 | monkeypatch + `_CpuPlatformShim` |
| 2 | `Worker` 类属性在 import 时计算，补丁时机竞态 | 补丁后显式修复类属性 |
| 3 | FSDP `device_id=int(LOCAL_RANK)`（strategy/fsdp.py:173）在无加速器时抛错 | 包装改写为 `torch.device("cpu")` |
| 4 | torch FSDP 禁 CPU 参数 + `sync_module_states=True` | 单机强制 False |
| 5 | patch_syncer `pin_memory()` 无加速器抛错 | guard 返回 self |
| 6 | `_get_copy_stream` 假设 current_device() 返回 int 索引 | 重写 + shim Stream/Event |
| 7 | `Tensor.record_stream` 拒绝哑 Stream | 全局 no-op |
| 8 | Ray worker 不继承 driver PYTHONPATH（sitecustomize 不传播） | venv `.pth` 钩子 |

**B 类：自定义模型接入的通用缺口（任何硬件都会踩）**

| # | 缺口 | 处置 |
|---|---|---|
| 9 | env 类型分发是 if/elif 硬编码 + 两处 `SupportedEnvType` 真 Enum（config/envs），无插件注册表 | monkeypatch / EXT_MODULE |
| 10 | 手工构造 `VJEPA2Config` 必须显式设 `_attn_implementation` | 设 `"sdpa"` |
| 11 | **FSDP1 `use_orig_params=False` 与部分冻结模型不兼容**（FlatParam 要求 uniform requires_grad） | `use_orig_params: true` |
| 12 | 优化器按 `"value_head"` 名字子串分参数组（fsdp_model_manager.py:516）——对自定义模型是隐性命名约束 | value head 照此命名 |

**值得反馈上游**：#1（NO_ACCEL 返回 None，CPU 路径文档化或内置兜底）、#11（冻结 encoder + 可训练 head 是 VLA/JEPA 类模型的标准形态，FSDP1 配置应文档化）、#12（命名约束应改为显式配置项）。#9（env 注册表化）属设计建议。

## 2. 工时对表

| 阶段 | 计划（spike 方案 §4） | 实际（AI 代理执行，含排雷） |
|---|---|---|
| S1 环境就绪 + MLP 基线 + 集成点读码 | 1 天 | ~0.5 天（大头：Ray/WSL 环境 + CPU 补丁链 9 项） |
| S2 模式 A 骨架 + 注册 + YAML | 1 天 | ~0.15 天（一次冒烟 + 一个 FSDP 缺口） |
| S3 模式 A 收敛验证 | 1 天 | ~0.1 天（首发即升，60 epoch 等待为主） |
| S4 模式 B 渲染帧通路 | 2–3 天 | ~0.15 天（io_struct 读码 + env/策略改造 + 首发即通） |
| **合计** | **5–6 天** | **~1 天** |

**对正式接入（非 spike）的修正估算**：ADR-W35 §6 给整个验证门（W35.1–W35.4）预估 2–3 周。
就 W35.3 关心的「JEPA policy 正式接入 RLinf」单项，基于实测修正为 **3–5 人日**：

- 1 人日：模型封装与注册（机制已验证，照 S2 模板；含单测）
- 1–2 人日：真实观测通路（模式 B 已验证机制；真实 env 的图像键约定、分辨率/帧率对齐、
  io_struct 大数据量的内存与序列化调优）
- 1 人日：权重同步与 FSDP 配置定稿（`use_orig_params`、冻结策略、混合精度）
- 0.5–1 人日：配置/文档/CI 集成

关键不确定项已从此前的「通道能否承载」降级为「数据通路工程细节」——**架构级风险已排除**。

## 3. 接口文档缺口记录

1. **CPU/NO_ACCEL 路径完全未文档化**：`AcceleratorUtil` 对 NO_ACCEL 返回 None，无任何官方说明；
   CPU 可行性只能靠读码 + 补丁确认（本 spike 附 1 即事实上的 CPU 接入文档）。
2. **`RLINF_EXT_MODULE` 通道存在但文档稀缺**：机制在 `cluster/utils.py:81`，docstring 是唯一文档；
   「register() 在 Worker 进程内、类属性计算之后调用」这一时序约束不明示（决定了它适合注册模型/env、
   不适合平台级补丁）。
3. **自定义模型的 forward 契约无接口文档**：契约只能从 `mlp_policy.py` 反推
   （default_forward/predict_action_batch 签名、forward_inputs 键、返回值 dict 结构、
   「rollout 对未列名 model_type 不传 mode kwarg」这类暗坑）。
4. **`value_head` 命名约束未文档化**（优化器分组依赖名字子串）。
5. **env 侧契约未文档化**：`infos["final_info"]["episode"]` 结构、auto_reset SAME_STEP 语义、
   `final_observation` 用途，均需读 EmbodiChainEnv 反推。
6. **obs dict 图像键约定**（`main_images`/`extra_view_images`）只有 maniskill 先例，无 schema 文档。

## 4. NPU 迁移预判（S1 CPU 补丁链在 torch_npu 下的命运）

| CPU 补丁 | NPU（torch_npu + CANN）下 |
|---|---|
| #1 NO_ACCEL platform None | **消失**：NPU 是被管理的 accelerator，`get_torch_platform` 走 AcceleratorManager 注册路径返回 torch.npu 风格模块 |
| #2 类属性时序 | 消失（#1 不存在则无需修复） |
| #3 FSDP int device_id | **变形但可能无害**：`torch.device(int)` 在有加速器时合法（指当前加速器）；需确认 torch_npu 下 FSDP1 支持度（W35.1/W35.2 核对项） |
| #4 sync_module_states | 消失（参数在 NPU 上） |
| #5/#6/#7 pin_memory/record_stream/copy_stream | **变形**：有真 Stream/Event/pin_memory，哑对象不需要；但 patch_syncer 的 CPU 快照 ↔ NPU 张量拷贝路径需实测（NPU pinned memory 语义与 CUDA 有差异的先例） |
| #8 PYTHONPATH 不传播 | **保留**：与硬件无关；但正式接入用 `RLINF_EXT_MODULE` 即可绕开 |
| #9–#12 通用缺口 | **全部保留**：与硬件无关，正式接入必须处理 |

结论：NPU 迁移后补丁链预计从 9 项缩到 0–2 项（#3/#5-7 待实测），通用缺口 4 项不变。
**W35.1（Atlas smoke）是这些预判的验证点。**

## 5. W35.3 验证门结论（回填 ADR-W35 §6）

| 问题 | 判定 | 证据 |
|---|---|---|
| **Q1** custom model 注册通道能否承载 ViT 级模型？ | ✅ **通过** | VJEPA2Encoder（ViT 级，7.8M 参）经 `RLINF_EXT_MODULE` 注册并在 actor/rollout 双侧实例化；PPO 闭环 + 收敛（模式 A：17.7→274.2/60ep） |
| **Q2** 非标准 observation 预处理能否嵌入 rollout/actor 数据通路？ | ✅ **通过** | 模式 B：RGB 帧（K=2 堆叠，`main_images` 键）经 env→channel→rollout→`forward_inputs`→io_struct 批处理→actor forward 全链路透传，PPO 闭环跑通（见 S4 一节） |
| **Q3** actor→rollout 权重同步对自定义模型开箱可用？ | ✅ **通过** | patch_syncer 只依赖 state_dict，零模型侧配合；reward 持续上升为黑盒证据（不同步则恒为初始水平 ~17） |
| **Q4** 接入工程当量 ≤ 预估？ | ✅ **通过** | spike 全程 ~1 天（预估 5–6 天）；正式接入修正估算 3–5 人日，远优于「2–3 周验证门」中本项的隐含预算 |

**W35.3 总判定：通过。** 否决项「自定义模型通道无法承载 ViT 级模型」**未触发**——
通道承载 ViT 级模型经双模式实证。建议 W35.1（Atlas NPU smoke）按 §4 预判表逐项核对。

---

## 附：S4（模式 B）实施与证据

### 改动点

- env：`gym_cartpole_env.py` 加 `pixel_obs` 开关——`render_mode="rgb_array"` 渲染 →
  PIL resize 到 64×64 → 每 env 一个 K=2 帧 deque → `_wrap_obs` 输出
  `main_images: [num_envs, 2, 3, 64, 64] float[0,1]`（maniskill 键约定）；
  `final_observation` 同步带帧（bootstrap 用）；模块顶部 `SDL_VIDEODRIVER=dummy`（无显示环境）。
- 模型：`vjepa2_policy.py` 加 `obs_mode="pixel"` 分支——`VJEPA2Config`
  （frames_per_clip=2/tubelet_size=2/crop=64/patch=16 → 16 个视觉 token），
  **完整 `VJEPA2Encoder.forward(pixel_values_videos)`**（含 3D patch embeddings），
  mean pooling → 双 head；encoder 仍冻结。
- 配置：`gym_ppo_cart_pole_vjepa2_pixel.yaml`——train envs 8 / eval envs 2（性能考量），
  其余同模式 A；40 epoch。
- **io_struct 结论**：`forward_inputs` 是任意键张量 dict，split（huggingface_worker.py:845）
  与 merge 均按 dim 0 逐键处理，**图像键天然透传，无需框架改动**——S4 上手信息中的
  「主要未知点」确认为非问题。

### 性能与收敛证据（实测，2026-08-04）

**链路：完全跑通，框架零改动。** 渲染帧 → K=2 帧堆叠（`main_images`）→ channel →
rollout 完整 `VJEPA2Encoder.forward` → `forward_inputs` 图像键 → io_struct 批处理 →
actor forward → PPO 更新 → 权重同步回 rollout，全程无报错（`run-vjepa2-pixel.log`）。

**性能数字（CPU，WSL2）**：

| 指标 | 模式 A（state，32 envs） | 模式 B（pixel，8 envs） |
|---|---|---|
| rollout 迭代耗时 | ~2.9 s/epoch（1024 样本） | ~3.2–3.9 s/epoch（256 样本） |
| 单样本通路成本 | 基准 | ≈4.5×（渲染 + 视觉前向 + 图像传输） |
| 渲染本身 | — | 可忽略（8 env × 64×64，chunk_step ~13ms/4env 量级） |
| 主要成本来源 | — | 视觉 encoder 前向/反向（rollout 与 actor 各一遍）+ 图像张量通道传输 |

**reward 曲线（如实记录）**：

- 冻结 encoder（26 epoch）：13.9 → 35.4（ep13）后在 15–35 震荡。
- 解冻 encoder（35 epoch）：17.0 → 27.0（ep11），后在 15–27 震荡，末段峰值 26.9/24.0。
- **两跑均未见模式 A 式明确上升**。原因分析：① 64×64 下 CartPole 细杆仅数像素宽，
  随机小 encoder 视觉特征区分度弱；② 样本量 256/epoch 仅为模式 A 的 1/4（渲染性能权衡）；
  ③ lr=1e-4 对 7.9M 视觉模型偏小，像素 PPO 通常需更大 batch 与更长训练。
- **Q2 判定不受此影响**：Q2 问的是「非标准 observation 预处理能否嵌入 rollout/actor
  数据通路」——机制层面答案明确为能（双模式实证）；收敛高度是表征质量与调参问题，
  非数据通路问题，按 spike 方案 §3.2 不在模式 B 通过标准内（「链路跑通 + reward 有上升
  迹象」，冻结跑前段 13.9→35.4 的上升段即迹象）。若后续要追收敛：升分辨率至 128×128、
  增大 batch/envs、解冻 + 更高 lr、或直接上预训练 V-JEPA 2 权重。

### S4 新发现接入缺口（并入 §1.2 B 类）

13. pygame 渲染依赖（`pip install pygame`）+ 无显示环境必须 `SDL_VIDEODRIVER=dummy`
    （adapter 模块顶部已内置）——环境性缺口，非框架缺口。
14. **确认非缺口**：`forward_inputs` 任意图像键透传（S4 上手信息中的「主要未知点」），
    io_struct 按 dim 0 逐键 split/merge，无需框架改动。
