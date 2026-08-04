# W35.3 · RLinf 集成点确认纪要（S1 交付物）

> 目的：为 S2（VJEPA2Policy 自定义策略接入）确认 RLinf v0.3 的五个关键集成点。
> 所有行号以 `spikes/w35-3-rlinf/rlinf-src`（GitHub tag `v0.3`，commit `0505431`）为准；
> 与 pip 安装版 `rlinf 0.3.0`（`/home/chad/w35-3/venv/lib/python3.12/site-packages/rlinf`）抽查一致。
> 确认方式：**读码 + CartPole 基线实跑验证**（环境：WSL2 Ubuntu 24.04，torch 2.10.0+cpu，ray 2.56.1）。

---

## 结论速览

| # | 集成点 | 结论 | 对 S2 的意义 |
|---|--------|------|--------------|
| a | 模型注册机制 | builder 函数注册表，`register_model()` 一手包办三处登记 | S2 在入口顶部调一次即可；worker 进程走 `RLINF_EXT_MODULE` 官方扩展通道 |
| b | actor forward 契约 | `BasePolicy` 抽象基类 + 两个方法签名，MLP 有完整先例 | VJEPA2Policy 照抄 MLP 签名即可，注意 `mode` 等 kwarg 必须带默认值 |
| c | rollout 推理路径 | MLP 类模型走 `huggingface` backend，**不经 vLLM/SGLang** | R2 否决风险解除，VJEPA2 前向直接在 rollout worker 内跑 |
| d | 权重同步 | patch_syncer 只依赖 `nn.Module.state_dict()` | 自定义模型**零配合**自动生效；S3 用 reward 上升做黑盒判据 |
| e | env 观测字段 | obs dict 透传；框架支持图像 obs；EmbodiChain CartPole 只有 states | 模式 A（投影）直接可用；模式 B（RGB）需 gymnasium 渲染帧兜底（R5 成立） |

---

## a. 自定义模型注册机制

**结论：注册表是「模型名 → builder 函数」映射（不是类映射），`register_model()` 一个入口完成全部登记。**

证据链：

- `rlinf/models/__init__.py:23` — `_MODEL_REGISTRY: dict[str, ModelBuilder]`。
- `rlinf/models/__init__.py:26` — `register_model(model_type, model_builder, category="embodied", force=False)`。
  内部同时完成两件事：
  1. 调 `SupportedModel.register(...)`（`rlinf/config.py:49`）——注意 `SupportedModel` 是**自维护的 str 注册表，不是真 Enum**；
  2. 把 model_type 加入 `EMBODIED_MODEL` 集合（`rlinf/config.py:110`）。
- `rlinf/models/__init__.py:241` — `get_model(cfg)` 按 `cfg.model_type` 查 `_MODEL_REGISTRY` 分发；
  builder 签名为 `(cfg: DictConfig, torch_dtype) -> nn.Module`，其中 cfg 就是 `actor.model` 配置节点。
  **未注册时返回 None**（`:243-245`）。
- `rlinf/config.py:823` — `validate_cfg` 里有 `assert model_type in EMBODIED_MODEL`。
  ⇒ **注册必须先于 `validate_cfg`**：S2 入口脚本顶部 `register_model("vjepa2_policy", builder)` 即可通过校验。

worker 进程侧（actor/rollout 是独立 Ray actor 进程，不继承 driver 的内存补丁）：

- 官方扩展通道：`rlinf/scheduler/cluster/utils.py:81` `load_user_extension_module(logger)` ——
  读环境变量 **`RLINF_EXT_MODULE`**（`rlinf/scheduler/cluster/cluster.py:68` `ClusterEnvVar.EXT_MODULE`），
  import 该模块并调用其 `register()` 函数；`Worker` 每个进程初始化时调用
  （`rlinf/scheduler/worker/worker.py` `_load_user_extensions`，约 `:377`）。
  该环境变量在 `Cluster.DEFAULT_SYS_ENV_VAR`（cluster.py:125）中，会由 scheduler 自动转发到各 worker。
- **S2 首选此通道**（写一个 `spike_ext/rlinf_ext.py` 暴露 `register()`，设 `RLINF_EXT_MODULE=spike_ext.rlinf_ext`），
  而非依赖 sitecustomize——后者在本次 spike 中证明对 Ray worker 不天然生效（见「CPU 补丁」节）。

**与 spike 方案文档 §3.4 的差异**：§3.4 推测为「模型名 → 类映射」；实际更灵活，是 builder 函数，
且需要同步登记 `SupportedModel` + `EMBODIED_MODEL` 两处（`register_model` 已包办，无需手工三处写）。

## b. actor 训练侧 forward 契约

**结论：自定义模型继承 `BasePolicy`，实现 `default_forward` 与 `predict_action_batch` 两个方法；MLP 策略是完整可照抄的先例。**

证据链：

- `rlinf/models/embodiment/base_policy.py:32` — `BasePolicy`（抽象基类）；
  `:19` `ForwardType` 枚举（DEFAULT/SFT/SAC/…）。抽象方法：`default_forward`、`predict_action_batch`。
- MLP 先例 `rlinf/models/embodiment/mlp_policy.py`：
  - `:181` `default_forward(forward_inputs, compute_logprobs, compute_entropy, compute_values)`
    → 返回 dict，含 `logprobs` / `entropy` / `values`；
    输入取 `forward_inputs["states"]` 与 `forward_inputs["action"]`；
    动作分布为连续高斯 `Normal(mean, std)`。
  - `:275` `predict_action_batch(env_obs, calculate_logprobs, calculate_values, return_obs, mode)`
    → 返回 `(chunk_actions [B, chunks, dim], result)`，
    result 含 `prev_logprobs` / `prev_values` / `forward_inputs`。
  - `:105` `preprocess_env_obs` 把 `env_obs["states"]` 搬上设备。
- actor 训练调用点：`rlinf/workers/actor/fsdp_actor_worker.py:1462` 附近——
  `self.model(forward_inputs=..., compute_logprobs=True, compute_entropy=..., compute_values=(adv_type=="gae"), use_cache=False)`。
- **坑（对 S2 重要）**：rollout 侧对未列名的自定义 model_type 调用 `predict_action_batch` 时**不加 `mode` kwarg**
  （只传 sampling_params，无配置时为空 dict）⇒ 自定义模型的 `predict_action_batch` 所有参数必须带默认值
  （如 `mode="train"`）。

## c. rollout 推理路径

**结论：MLP 类模型走 `rollout.backend: huggingface`，由 `MultiStepRolloutWorker` 在同进程内直接前向，完全不经 vLLM/SGLang。R2（自定义模型或被强制走 vLLM/SGLang）证伪。**

证据链：

- `rlinf/workers/rollout/hf/huggingface_worker.py` — `MultiStepRolloutWorker`：
  - `:138` `get_model` 实例化与 actor 同一个注册模型；
  - `:505` 推理直接调 `hf_model.predict_action_batch(...)`；
  - 配置侧即 spike YAML 里的 `rollout.backend: huggingface`。
- 无 vLLM/SGLang 依赖 ⇒ VJEPA2 encoder 的前向（torch 原生）可直接在 rollout worker 内执行，
  无需对接推理引擎的模型格式/量化/服务化。

## d. 权重同步（actor → rollout）

**结论：WeightSyncer 按配置创建，`patch` 类型做 CPU 快照 + 增量编码，只依赖 `nn.Module.state_dict()`——对自定义模型自动生效，无需模型侧任何配合。R3 缓解确认。**

证据链：

- actor 侧 `rlinf/workers/actor/fsdp_actor_worker.py`：
  `:1088 sync_model_to_rollout()`；`:1084 get_rollout_state_dict`；
  `:1121 init_sender(state_dict, send, recv, param_names_need_sync)`；`:1128 sync`。
- rollout 侧 `rlinf/workers/rollout/hf/huggingface_worker.py`：
  `:552 sync_model_from_actor()`；
  `:581 init_receiver(state_dict=self.hf_model.state_dict(), ...)`；
  `:587 weight_syncer.apply(self.hf_model, recv_func)`。
- 工厂：`rlinf/hybrid_engines/weight_syncer/base.py:76 create`，按 `weight_syncer.type: patch` 创建
  （spike 配置 `config/weight_syncer/patch_syncer.yaml`）。
- ⇒ 只要 VJEPA2Policy 是标准 `nn.Module`（参数都在 state_dict 里），同步链路零改动。
  **S3 验收判据：reward 曲线上升 = 权重同步黑盒生效**（无需白盒断言）。

## e. env 观测字段通路

**结论：env 观测是 dict，env → rollout → `predict_action_batch(env_obs=...)` 原样透传；框架层面支持图像观测（maniskill 先例）；但 EmbodiChain CartPole 的 `_wrap_obs` 只拼低维 states，无 RGB。**

证据链：

- EmbodiChain：`rlinf/envs/embodichain/embodichain_env.py:296` `_wrap_obs`，
  `:313` `return {"states": torch.cat(...)}`（按 state_keys 拼低维状态，**无 RGB**）。
  ⇒ R5 成立：S2 模式 B（视觉输入）若用 CartPole，需 gymnasium `render()` 出帧兜底。
- 框架支持图像观测的先例：`rlinf/envs/maniskill/maniskill_env.py:155/:165/:174`
  返回 `{"main_images", "extra_view_images", "states"}`。
  ⇒ 比 §3.4 推测更乐观：观测 dict 的键集合不由框架校验死，自定义 env 可以自由加键。
- env 类型分发**不是插件注册表**：`rlinf/envs/__init__.py:18` `SupportedEnvType(Enum)` +
  `:39` `get_env_cls` 是 if/elif 硬编码 ⇒ 自定义 env 需要 monkeypatch（本 spike 已实现，
  `sitecustomize.py` patch `rlinf.envs.get_env_cls` 识别 `"gym_cartpole"`）
  或走 `RLINF_EXT_MODULE` 的 `register()`。

---

## 附 1：CPU（NO_ACCEL）可行性补丁 —— R1 实锤与解法

**R1 证实**：RLinf v0.3 worker 层**没有内置 CPU 兜底**。`AcceleratorUtil.get_torch_platform()`
与 `get_device_type()` 对 `NO_ACCEL` **直接返回 None**
（`rlinf/scheduler/hardware/accelerators/accelerator.py:442` / `:452`），
worker 启动即崩于 `huggingface_worker.py:52` `self.torch_platform.current_device()`。
CPU 路线官方未文档化、开箱不可用，但**可补丁修复**（这正是 spike 要记录的接入缺口）。

补丁内容（`spikes/w35-3-rlinf/sitecustomize.py`，monkeypatch，不改 RLinf 源码）：

1. `_CpuPlatformShim`：模拟 torch.cuda 风格 platform 模块，覆盖代码实际用到的 API 面——
   `current_device()`（返回 `torch.device("cpu")`）、`set_device`/`empty_cache`/`synchronize`/`ipc_collect`（no-op）、
   `is_initialized()`→True、`device_count()`→1、`manual_seed_all`、哑 `Event` 类。
   （API 面来自全库 grep：`huggingface_worker.py:52`、`fsdp_actor_worker.py` 多处、
   `utils/distributed.py` 多处、`collective_group.py`（Event/ipc_collect 路径被
   `Worker.has_accelerator`=False 短路，不会走到）。）
2. monkeypatch `AcceleratorUtil.get_torch_platform` / `get_device_type`：
   原方法返回 None 时分别回退到 `_CpuPlatformShim` 与 `"cpu"`。
3. **类属性修复**：`rlinf/scheduler/worker/worker.py:319-321` 的 `Worker` 类属性在模块 import 时计算；
   而 sitecustomize 里 import AcceleratorUtil 会经包 `__init__` 链提前触发 worker.py 导入（补丁尚未挂上），
   类属性已被算成 None ⇒ 补丁后须显式修复 `Worker.torch_platform` / `Worker.torch_device_type`。

补丁投递（Ray worker 不继承 driver 的 `PYTHONPATH`，sitecustomize 不天然传播）：

- 在 venv site-packages 放 `w35_spike_path.pth`
  （`/home/chad/w35-3/venv/lib/python3.12/site-packages/w35_spike_path.pth`），
  内容 `import sys; sys.path.insert(0, "<spike根>")`。
  `.pth` 在 site 阶段先于 sitecustomize 执行，使 spike 根的 `sitecustomize.py` 抢先于
  `/usr/lib/python3.12/sitecustomize.py` 被加载 ⇒ driver 与所有 Ray worker 进程自动获得全部补丁。
- 备选官方通道：`RLINF_EXT_MODULE`（见 a 节）；但注意 `Worker._load_user_extensions` 在
  `__init__` 内、类属性计算之后执行，**单独使用时修不了 torch_platform 类属性问题**，
  适合注册模型/env，不适合本平台补丁。

CPU 兼容性已预排/实排的其他雷（读码 + 实跑确认，全部已修复）：

- FSDP：`hybrid_engines/fsdp/utils.py:64 create_device_mesh` 用 `Worker.torch_device_type`
  （补丁后 = `"cpu"`）+ gloo，无 cuda 硬编码；`fsdp_model_manager.py:133`
  `torch.amp.autocast(device_type="cuda")` 仅在 AMP 开启时触发，spike 配置已关
  （`config/training_backend/fsdp.yaml` `amp_autocast.enabled: False`）。
- Ray 集群端口避开系统 redis（6379 占用）→ 用 6399；WSL 下连自身 eth0 IP 不通 → 用 `127.0.0.1:6399`。

**实跑中追加的补丁（均在 sitecustomize.py，按发现顺序）**：

4. `hybrid_engines/fsdp/strategy/fsdp.py:173` 硬编码 `device_id=int(os.environ["LOCAL_RANK"])`，
   `torch.device(<int>)` 在无加速器时报 "Cannot access accelerator device"
   → 包装 FSDP，无 CUDA 时 int device_id 改写为 `torch.device("cpu")`。
5. torch FSDP 禁止 CPU 参数配 `sync_module_states=True`（`_init_utils.py:1146`）
   → 同一包装内单机强制 `sync_module_states=False`（单 rank 无同步需求）。
6. `weight_syncer/patch_syncer.py:886` `snapshot_value.pin_memory()` 在无加速器时抛错
   → 全局 guard `Tensor.pin_memory`，失败回退返回 self（CPU 语义无损）。
7. `patch_syncer.py:534 _get_copy_stream` 假设 `current_device()` 返回 int 索引
   （`torch.device(device.type, current_device())`）→ 重写该方法 + shim 补
   `Stream`/`stream()`/`Event`；`patch_syncer.py:469` `Tensor.record_stream(哑Stream)` 抛错
   → 全局 no-op（record_stream 本是 CUDA 分配器提示）。
8. `rlinf/envs/action_utils.py:282` 用真 Enum `rlinf.envs.SupportedEnvType` 做分发，
   拦下 `"gym_cartpole"` → shim 把 `gym_cartpole` 映射为 `EMBODICHAIN` 成员
   （`prepare_actions` 走直通分支，与适配器镜像 EmbodiChainEnv 的语义一致）；
   需同时替换 `rlinf.envs.SupportedEnvType` 与 `rlinf.envs.action_utils` 模块内绑定。
9. 适配器自身 bug（非 RLinf 问题）：`infos["final_info"]` 结构须为 `{"episode": {...}}`，
   值为可按 done mask 索引的 per-env 张量（`env_worker.py:438` 契约）——已修。

**基线结果（CPU，60 epoch 完成）**：episode return 从 17.2 → 392.0（末段稳定 200–340，
CartPole-v1 满分 500）。stdout 与 tensorboard 事件文件均在（见 S1 报告）。

## 附 2：与 spike 方案文档（2026-08-04 w35-3 spike plan）的不一致汇总

1. §3.4 推测注册表为「模型名 → 类映射」→ 实为 builder 函数注册（更灵活），三处登记由 `register_model` 包办。
2. R2「rollout 或强制走 vLLM/SGLang」→ **证伪**，huggingface backend 直接前向。
3. R1「CPU 可行性」→ **证实为缺口**：NO_ACCEL 返回 None platform，开箱即崩；需 monkeypatch（附 1）。
4. env 观测通路比 §3.4 推测乐观：框架支持图像 obs（maniskill 先例），键集合可自由扩展；
   但 EmbodiChain CartPole 本体只有 states（模式 B 仍需渲染帧兜底，R5 成立）。
5. embodichain 路线不可行（方案外发现）：`embodichain 0.2.3` 依赖 `dexsim_engine==0.4.3`，
   pypi.org 与清华镜像均无此包 ⇒ S1 基线按预案改走 gymnasium CartPole 兜底
   （自实现 `spike_ext/gym_cartpole_env.py`，完整镜像 `EmbodiChainEnv` 接口）。

---

## 附 3：S2/S3（模式 A）实施记录与结论（2026-08-04）

### VJEPA2Policy 实现要点（spike_ext/vjepa2_policy.py）

- `:36 VJEPA2Policy(nn.Module, BasePolicy)`；`get_model(cfg, torch_dtype)`（`:211`）即注册的 builder。
- encoder：`VJEPA2Config`（hidden=384/layers=4/heads=6，crop=32/patch=16/frames=4/tubelet=2）+
  `transformers` 原生 `VJEPA2Encoder` 随机初始化（`:86`），**7.8M 总参 / 0.11M 可训练**（冻结时）。
- 模式 A 适配（§3.2）：`obs_projection: Linear(4 → 8×384)` reshape 成 8 个伪 token，
  **绕过 `VJEPA2Embeddings`（3D patch conv）直接喂 `encoder.layer` 栈**（`:101 _encode`）——
  可行是因为 `VJEPA2RopeAttention.get_position_ids` 从 `x.size(1)` 内部 arange 推导 RoPE 位置，
  不依赖 embeddings 输出。mean pooling 后接 `actor_mean` + 独立 `actor_logstd` + `value_head`。
- forward 契约逐行镜像 `mlp_policy.py`（default_forward `:117`、predict_action_batch `:181`，
  全参数带默认值——rollout 对未列名 model_type 不传 `mode`）。
- **命名约束**：value head 必须叫 `value_head`（`fsdp_model_manager.py:516/:526` 按名字子串分
  优化器组、挂 `value_lr`）。

### 注册方式（PR #920 官方通道，实测生效）

- `RLINF_EXT_MODULE=spike_ext.rlinf_ext`（`cluster.py:465` 全名 `RLINF_` 前缀）；
  每个 Worker 进程 `load_user_extension_module`（`cluster/utils.py:81`）import 并调 `register()`
  → `register_model("vjepa2_policy", get_model, force=True)`（`spike_ext/rlinf_ext.py`）。
- driver 侧在 `validate_cfg` 前手动 `register()`（`src/train_spike_vjepa2.py:57`）。
- 该变量在 `Cluster.DEFAULT_SYS_ENV_VAR` 中自动转发到所有 worker——S1 的 sitecustomize 困境
  （PYTHONPATH 不传播）不影响它，因为 spike 根已在 venv `.pth` 里。

### S3 收敛证据（CPU，60 epoch，encoder 冻结）

episode return 完整曲线（run-vjepa2.log / results/tensorboard 最新 event 文件）：

- 起点 17.7（≈随机）→ 61.5（ep5）→ 137.1（ep9）→ 188.2（ep20）→ 243.8（ep29）
  → **峰值 274.2（ep46）**；后段在 100–240 宽带震荡，ep60 收 177.5。
- 结论：**reward 明确上升**（10×于随机水平），PPO 闭环与 actor→rollout 权重同步对自定义模型
  生效（黑盒判据：权重不同步则 reward 恒在 ~17）。平台期低于 MLP 基线（300–400 稳定）且方差大——
  冻结随机特征的预期表现；方案允许解冻 encoder 复跑提上限（本轮时间预算内未做）。

### 新发现接入缺口（S2/S3 追加）

10. 手工构造 `VJEPA2Config` 必须显式 `config._attn_implementation = "sdpa"`
    （from_pretrained 才会自动设置；否则 `VJEPA2RopeAttention` KeyError: None）。
11. **FSDP1 `use_orig_params=False` 与「部分冻结模型」不兼容**：FlatParam 要求 uniform
    requires_grad（torch `_flat_param.py:825`）。冻结 encoder + 可训练 head 的自定义模型必须
    `use_orig_params: true`（spike YAML 已改）。这是自定义模型+冻结策略的通用约束，值得反馈上游。
12. 官方 EXT_MODULE 通道本身无缺口，验证通过。

### S4（模式 B：渲染帧 → 真实视觉通路）上手信息

- env 侧：框架支持图像 obs（maniskill_env.py:155/165/174 返回
  `{"main_images", "extra_view_images", "states"}`）；gymnasium CartPole 用
  `render_mode="rgb_array"` 出帧，适配器 `_wrap_obs` 加 `main_images` 键即可（参考
  `spike_ext/gym_cartpole_env.py:119`）。EmbodiChain CartPole 无 RGB（R5 已证）。
- 模型侧：VJEPA2Policy 加像素分支——`VJEPA2Model` 完整 forward 吃 `pixel_values_videos`
  [B, T, C, H, W]（K=2 帧堆叠；config 的 frames_per_clip/tubelet_size 要与 K 对齐，
  如 frames=2/tubelet=1），或直接复用已构造但被绕过的 `encoder.embeddings`。
- **主要未知点（S4 第一天先查）**：actor 训练回路的 `forward_inputs` 当前只含
  states+action（`mlp_policy.py:290` 模式）；图像张量要随 rollout result → io_struct 批处理
  → actor forward 全链路透传，需确认 `rlinf/data/io_struct.py` 对图像键的批处理支持；
  maniskill 配方（`maniskill_ppo_mlp.yaml` 或 VLA 配方）是现成参照。
- 性能预警：CPU 上 32 env × 每步渲染会显著拖慢迭代；建议降 env 数或 K 帧间隔渲染。
