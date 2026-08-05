# W35.3 S1–S5 进度记录（磁盘即记忆，每步更新）

## 当前状态（2026-08-04 21:00）——S1–S5 全部完成 ✅ W35.3 通过

- S4 收尾：链路零改动跑通（Q2 通过）；reward 两跑均 20–35 震荡未见模式 A 式上升，
  已如实记录原因（低分辨率细杆/样本量 1/4/lr 偏小）与追收敛路径（128×128、大 batch、
  预训练权重）。性能：pixel ~3.2–3.9s/epoch（256 样本）vs state ~2.9s（1024 样本），
  单样本成本 ≈4.5×。
- S5 交付：integration-cost-assessment.md（12+2 项缺口分类、工时修正 3–5 人日、
  6 项文档缺口、NPU 迁移预判表、Q1–Q4 全通过判定，可回填 ADR-W35 §6）。
- 训练进程已停止。Ray 集群仍在（6399），如需复跑用 start-train*.sh。

## 历史状态（20:40）——S4 链路已通，解冻复跑中

- 模式 B 首发即通：io_struct 的 forward_inputs 任意键按 dim0 split/merge，图像键天然透传，
  **框架零改动**（上手信息里的"主要未知点"= 非问题）。env 加 pixel_obs（rgb_array→PIL 64×64
  →K=2 帧栈→main_images 键）；策略加 obs_mode=pixel 分支（完整 VJEPA2Encoder forward，
  frames=2/tubelet=2/crop=64/patch=16 → 16 视觉 token）。SDL_VIDEODRIVER=dummy 必须设。
- 冻结 encoder 26 epoch：13.9→35.4 后在 15–35 震荡（随机特征视觉信号弱，符合预期）。
  迭代 ~3.3–3.9s/epoch（8 envs，vs 模式 A ~2.9s）。
- **正在解冻复跑**（vjepa2_freeze_encoder: false，7.9M 全参数训练），日志 run-vjepa2-pixel.log。
- S5 报告 integration-cost-assessment.md 已写完主体，S4 性能/收敛段落待回填。
- 注意：启动 pygame 依赖已装进 venv（tuna 镜像）。

## 历史状态（20:00）——S4 模式 B 开工

计划：1) 查 io_struct 图像键批处理 + forward_inputs 链路（maniskill main_images 先例）
2) env 适配器加 pixel_obs 开关（render_mode=rgb_array，K=2 帧堆叠，main_images 键）
3) VJEPA2Policy 加 obs_mode=pixel 分支（VJEPA2Encoder 完整 forward，frames/tubelet 对齐 K=2）
4) 新 YAML（envs 减到 8、降分辨率）→ 冒烟 → 跑训练，标准：链路通 + reward 有上升迹象
5) S5：integration-cost-assessment.md（改动清单/工时对表/文档缺口/NPU 迁移预判/Q1–Q4 判定）

## 历史状态（19:45）——S1+S2+S3（模式 A）全部完成 ✅

- S3 60 epoch 跑完（encoder 冻结）：17.7 → 137.1(ep9) → 243.8(ep29) → 峰值 274.2(ep46)，
  后段 100–240 震荡，ep60 收 177.5。明确上升 = PPO 闭环 + 权重同步对自定义模型生效。
- 纪要已追加附 3（S2/S3 实施记录、新缺口 #10-12、S4 上手信息）。
- 未做：解冻 encoder 复跑（时间预算）；S4 模式 B（上手信息已备）。
- 关键复用信息：start-train-vjepa2.sh / run-vjepa2.log / config/gym_ppo_cart_pole_vjepa2.yaml。

## 历史状态（19:25）

- S2 首跑遇**新接入缺口**：FSDP1 `use_orig_params=False` 无法 flatten 混合 requires_grad
  （冻结 encoder + 可训练 head）→ 改 `use_orig_params: true` 即过（副作用：参数名保留更完整，
  优化器按 "value_head" 分组不受影响）。
- 改后一发跑通：RLINF_EXT_MODULE 官方注册通道在 worker 进程生效，PPO 闭环+权重同步正常。
- **S3 reward 趋势（encoder 冻结）**：17.7 → 23.6 → 36.0 → 61.5 → 85.8 → 137.1 → 148.5
  → 188.2 → 243.8（epoch 30），其后 90–240 震荡（35 epoch 处）。明确上升趋势，与 MLP 基线
  前段走势同型；权重同步黑盒证据成立（ rollout 持续用新策略，否则 reward 恒 ~17）。
- 待办：跑完 60 epoch 取最终曲线 → 纪要补 S2/S3 章节 → 最终报告（含 S4 上手信息）。

## 历史状态（19:00）——S2 骨架就绪

S2 已建文件：
- `spike_ext/vjepa2_policy.py`：VJEPA2Policy(nn.Module, BasePolicy)。VJEPA2Config 小配置
  （hidden=384/layers=4/heads=6，crop=32/patch=16/frames=4/tubelet=2）随机初始化 VJEPA2Encoder；
  模式 A：Linear(4→8×384) 伪 token → 直接喂 encoder.layer 栈（绕过 3D patch embeddings，
  RoPE 位置由 attention 内部 arange 推导）→ layernorm → mean pool → actor_mean + 独立 logstd
  + value_head（命名必须叫 value_head：fsdp_model_manager.py:516 按名字分优化器组）。
  契约全照 MLPPolicy（default_forward/predict_action_batch 带全默认参数）。
  **坑**：手工构造 VJEPA2Config 必须设 `config._attn_implementation="sdpa"`（否则 KeyError: None）。
  冒烟过：7.8M 总参/0.11M 可训练（encoder 冻结），形状与梯度行为正确。
- `spike_ext/rlinf_ext.py`：官方扩展通道（RLINF_EXT_MODULE，cluster/utils.py:81），
  register() 调 register_model("vjepa2_policy", get_model, force=True)。
- `config/gym_ppo_cart_pole_vjepa2.yaml`：仿 S1，model_type=vjepa2_policy + vjepa2_* 超参。
- `src/train_spike_vjepa2.py`：driver 侧 register() 先于 validate_cfg。
- `start-train-vjepa2.sh`：export RLINF_EXT_MODULE=spike_ext.rlinf_ext；日志 run-vjepa2.log。

下一步：启动训练盯 run-vjepa2.log；新缺口记入 sitecustomize/纪要；S3 看 reward 趋势，
冻结不收剑则解冻 encoder 复跑。

## 历史状态（18:35）——S1 完成 ✅

**基线成功**：60 epoch 跑完，episode return 17.2 → 41.7 → 92.7 → 195.8 → 260.8 → 392.0，
末段稳定 200–340（CartPole-v1 满分 500）。权重同步生效（rollout 持续用新策略）。
证据：`run-baseline.log`（stdout 指标表）+ `results/tensorboard/`（最新 event 文件）。
《集成点确认纪要》已交付：`integration-points-memo.md`（含全部 9 项补丁记录）。
S2 未开工（时间耗尽），上手信息在纪要 a–e 节 + 本文件历史段落。

## 历史状态（18:20）——训练闭环已通 🎉

**MLP+PPO CartPole 基线在 CPU 上完整跑通**：rollout → env → 轨迹回收 → GAE →
actor/critic 更新 → 权重同步全链路无报错，PPO 指标正常输出
（首epoch returns_mean=37.4，grad_norm/loss 齐全）。正在跑 60 epoch 观察 reward 上升。

连环雷与解法（全部记录在 sitecustomize.py，共 6 个补丁 + 1 处适配器修复）：
1. NO_ACCEL platform None → _CpuPlatformShim + AcceleratorUtil monkeypatch + Worker 类属性修复
2. FSDP `device_id=int(LOCAL_RANK)` → int 转 torch.device("cpu")
3. FSDP `sync_module_states=True` 禁 CPU 参数 → 单机强制 False
4. patch_syncer `pin_memory()` → 全局 guard 返回 self
5. patch_syncer `_get_copy_stream` 用 int 设备索引假设 → 重写 + shim 补 Stream/stream()/Event
6. `Tensor.record_stream` 拒绝哑 Stream → 全局 no-op
7. envs.SupportedEnvType 真 Enum 拦 gym_cartpole → shim 映射到 EMBODICHAIN（action_utils 走直通分支）
8. 适配器 bug：`infos["final_info"]` 结构应为 `{"episode": {...}}`（env_worker.py:438 契约）

**待办**：收集 reward 上升证据（tensorboard/stdout）→ 最终报告。S2 若时间够。
启动方式：`setsid bash $D/start-train.sh < /dev/null > /dev/null 2>&1 &`

## 历史状态（17:49）

**CPU 平台补丁链已全通**（探针验证：Worker.torch_platform=_CpuPlatformShim, device_type=cpu）。
三个叠加问题与解法：
1. Ray worker 不继承 driver 的 PYTHONPATH → spike 根 sitecustomize 不加载。
   解法：venv site-packages 放 `w35_spike_path.pth`（.pth 在 site 阶段先于 sitecustomize 执行），
   内容 `import sys; sys.path.insert(0, "<spike根>")` → spike 根插到 sys.path 最前，
   我们的 sitecustomize 抢先于 /usr/lib/python3.12/sitecustomize.py 被 import。
   文件：`/home/chad/w35-3/venv/lib/python3.12/site-packages/w35_spike_path.pth`
2. Ray 复用空闲 worker 进程 → 探针/训练前注意旧进程没补丁（训练重启时 actor 都是新建，无影响）。
3. sitecustomize 里 `import AcceleratorUtil` 会经包 __init__ 链提前触发 worker.py 导入，
   Worker 类属性在补丁前已算成 None → 补丁后必须显式修复 Worker.torch_platform/torch_device_type
   类属性（sitecustomize.py 已含此修复）。

**下一步**：重启训练盯 run-baseline.log（17:49 启动），然后写 integration-points-memo.md。

## 历史状态（17:40）

- Ray 集群活着：`RAY_ADDRESS=127.0.0.1:6399`（6379 被系统 redis 占用）。
- 训练已推进到 workers 作为 Ray actors 启动，但 `MultiStepRolloutWorker.__init__`
  崩于 `huggingface_worker.py:52`：`AttributeError: 'NoneType' object has no attribute 'current_device'`。
- **根因**：`AcceleratorUtil.get_torch_platform()`/`get_device_type()` 对 `NO_ACCEL`
  返回 None（`rlinf/scheduler/hardware/accelerators/accelerator.py:442/:452`）。
  RLinf v0.3 worker 层无内置 CPU 兜底 = R1 集成缺口实锤。
- **补丁已写但未生效**：spike 根目录 `sitecustomize.py` 已加 `_CpuPlatformShim` +
  monkeypatch `AcceleratorUtil` 两个静态方法；但 Ray 探针任务证实
  **worker 进程 sys.path 没有 spike 目录**（PYTHONPATH 不传播），加载的是别人的
  sitecustomize，补丁没装上。

## 下一步计划（按序）

1. 查 worker 里已加载的 sitecustomize 是谁的；把我们的 sitecustomize 拷进
   `/home/chad/w35-3/venv/lib/python3.12/site-packages/`（venv python 的 site-packages
   永远在 sys.path 上，driver + 所有 Ray worker 都会自动加载），并在其中显式
   `sys.path.insert(0, spike根目录)` 保证 spike_ext 可导入。
2. 重启训练（命令见下），盯 `run-baseline.log`，逐个排连环雷
   （FSDP gloo/cpu、empty_cache、weight sync）。30 分钟上限，到点记录现场。
3. **写 `integration-points-memo.md`（最高优先级，素材已齐，见下）**。
4. 时间允许 → S2 VJEPA2Policy 骨架。

## 关键命令

```bash
# 启动训练（spike 根目录）
D=/mnt/c/WorkSpace/WorldFoundry/spikes/w35-3-rlinf; V=/home/chad/w35-3/venv
cd $D && setsid env RAY_ADDRESS=127.0.0.1:6399 PYTHONPATH=$D $V/bin/python src/train_spike.py > run-baseline.log 2>&1 < /dev/null &
# 重启 Ray（如需）
setsid $V/bin/python $V/bin/ray start --head --temp-dir=$HOME/ray_tmp --object-store-memory=500000000 --port=6399 --num-cpus=12 > $HOME/ray-start.log 2>&1 < /dev/null &
```

## 环境事实

- 现役 venv：`/home/chad/w35-3/venv`（原生 ext4）。**必须**用 `$V/bin/python -m pip`
  或 `python <script>` 形式，bin 脚本 shebang 指向旧路径。
- 版本：rlinf 0.3.0 / torch 2.10.0+cpu / ray 2.56.1 / transformers 4.57.6 / gymnasium 0.29.1。
- pip 用 tuna 镜像。embodichain 路线已判死（dexsim_engine 无处可装）→ gymnasium CartPole 兜底。
- Write/Edit 工具必须用 Windows 路径 `C:\WorkSpace\WorldFoundry\...`。
- FSDP 已确认 cpu 兼容：`create_device_mesh` 用 `Worker.torch_device_type`（补丁后="cpu"），
  AMP 已关（fsdp.yaml `amp_autocast.enabled: False`），无 `.cuda()` 硬编码；
  `torch.amp.autocast(device_type="cuda")` 仅在 AMP 开启时触发。
- 后台进程必须 `setsid ... < /dev/null &`；pkill -f 会自杀，用 ps+awk+xargs kill。

## 五集成点读码结论（已确认，行号对 rlinf-src = v0.3 tag 0505431）

a. 注册：`rlinf/models/__init__.py:23` `_MODEL_REGISTRY`；`:26 register_model(model_type, builder, category, force)`；
   `:241 get_model(cfg)` 按 model_type 查表，builder 签名 `(cfg, torch_dtype)->nn.Module`；
   需同步注册 SupportedModel（config.py:49）+ EMBODIED_MODEL（config.py:110），register_model 一手包办；
   config.py:823 assert 要求注册先于 validate_cfg。官方扩展通道：worker 每进程调
   `load_user_extension_module`（scheduler/cluster/utils.py:81），读 `RLINF_EXT_MODULE` 环境变量，
   import 模块并调 `register()`。
b. forward 契约：`rlinf/models/embodiment/base_policy.py:32 BasePolicy`（抽象 default_forward/predict_action_batch）；
   MLP 先例 `mlp_policy.py:181 default_forward(forward_inputs, compute_logprobs, compute_entropy, compute_values)`
   → dict{logprobs,entropy,values}；`:275 predict_action_batch(env_obs, calculate_logprobs, calculate_values, return_obs, mode)`
   → (chunk_actions, result{prev_logprobs,prev_values,forward_inputs})；调用点 fsdp_actor_worker.py:1462 附近。
   rollout 对自定义 model_type 不传 mode kwarg → 参数必须带默认值。
c. rollout 路径：MLP 走 `workers/rollout/hf/huggingface_worker.py` MultiStepRolloutWorker
   （`rollout.backend: huggingface`），`:505` 直接 `predict_action_batch`，不经 vLLM/SGLang → R2 证伪。
d. 权重同步：actor `fsdp_actor_worker.py:1088 sync_model_to_rollout`（:1121 init_sender/:1128 sync）；
   rollout `huggingface_worker.py:552 sync_model_from_actor`（:581 init_receiver/:587 apply）；
   patch_syncer 只依赖 state_dict → 自定义模型自动生效（R3 缓解）。
e. env 观测：obs dict 透传；EmbodiChain `_wrap_obs`（embodichain_env.py:296/:313）只拼 states 无 RGB（R5 成立，
   模式 B 需 gymnasium 渲染帧）；框架支持图像（maniskill_env.py:155/165/174 返回 main_images/extra_view_images/states）。
   env 类型分发是 if/elif 硬编码（envs/__init__.py:18/:39），自定义 env 需 patch/EXT_MODULE。

## 与 spike 方案文档不一致（写入纪要）

- §3.4 推测"模型名→类映射"：实为 builder 函数注册，且三处注册 register_model 包办。
- R2（rollout 强制 vLLM/SGLang）：证伪。
- R1（CPU 可行性）：官方 NO_ACCEL 返回 None platform，开箱即崩，CPU 可行但需补丁（证实 R1）。
- env 观测：框架层支持图像 obs，比 §3.4 推测乐观；EmbodiChain CartPole 本身只有 states。
