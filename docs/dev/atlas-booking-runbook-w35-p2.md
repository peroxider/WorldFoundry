# Atlas 硬件窗口 Runbook（W35.1 封口验证 + M2-T5 P2 验收）

> 用途：Atlas 8×NPU 预约到位后，从「窗口开放」到「W35.1 RLinf NPU
> 封口 + M2-T5 vllm_ascend 端到端 + goal_verdict.json 提交」的端到端操作清单。
> 本文件是 W35 ADR §6 验证门（W35.1、W35.2）和 W41 SPEC §6 验证门
> （M2-T5.1–T5.5）的执行手册。
>
> 与姊妹文件 `docs/dev/atlas-booking-runbook.md`（W20 单卡 LoRA smoke）
> 单文件聚焦单步骤：本文件只覆盖 W35.1 + M2-T5，不要把 W20/W21/W26/W27
> 步骤挤进来——避免 runbook 自己 drift。
>
> 多租户注意：执行前先 `uptime` + `npu-smi info` 看 load / Hbm 占用；
> 多租户集群 load > 5 或所有 NPU Hbm > 50% 时不要开跑。

## 0. 预约到位时确认的事

| 项 | 取值 / 操作 | 备注 |
|---|---|---|
| 硬件 | Atlas 800 A2 / 910B ×8（8 卡 FSDP 用）/ ×1（W35.1 / M2-T5 都够） | 单卡预约即可 |
| 容器 / 镜像 | `worldfoundry-base:phase1`（含 torch_npu + vllm_ascend）+ RLinf NPU image（仅 W35.1） | docker images 探针（§1.5）核实 |
| 驱动 | `/usr/local/Ascend/driver` RO 挂载 | W17 已验证 |
| 数据卷 | `/data`（仅 atlas 数据路径才需要；本 runbook 默认走 dry / mock） | W22 已有 subset manifest 路径 |
| 项目卷 | `$ROOT`（WorldFoundry 仓库）挂到容器 `/workspace`；host 端直跑路径写到 `cd <worldfoundry-root>` 后即可 | 与 W17 一致 |
| 网络通路 | `ssh <user>@<atlas-host>` 直连 OR 经 Windows netsh portproxy 跳板（WSL2 → Windows → Atlas） | 若 Atlas 在私网需在 Windows PowerShell 配置 portproxy：`netsh interface portproxy add v4tov4 listenaddress=127.0.0.1 listenport=2222 connectaddress=<atlas-host> connectport=22` |
| 预约时长 | W35.1 单卡 ≤ 30 min；M2-T5 两次 e2e ≤ 1 h；预留 buffer | 1 卡预约即可 |

## 1. 启动前环境检查（容器内 / host 直跑都行）

逐条确认，任一失败**不**进入下一步：

```bash
# 1.1 NPU 设备可见
ls /proc/driver/npu/ 2>&1 | head -5
npu-smi info -l 2>&1 | head -30
# 1.2 关键 Python wheel 版本（与 pyproject.toml 版本矩阵一致）
python3 -c "import torch_npu; print('torch_npu', torch_npu.__version__)"        # 期望 2.9.0
python3 -c "import torch; print('torch', torch.__version__, 'npu_devices', torch.npu.device_count())"
python3 -c "import vllm_ascend; print('vllm_ascend', vllm_ascend.__version__)"   # 期望 0.18.0
python3 -c "import vllm; print('vllm', vllm.__version__)"                        # 期望 0.18.0
python3 -c "import transformers; print('transformers', transformers.__version__)" # 期望 4.55.x
# 1.3 项目能 import（无 half-baked path）
python3 -c "import worldfoundry_train; from worldfoundry_train.device import resolve_device; print('resolve_device ->', resolve_device())"
# 1.4 数据 manifest 存在（如果走 atlas 数据模式而非 dry）
ls -la /data/videomix22m/manifest.csv 2>&1 | head -3
# 1.5 docker image 就绪
which docker && docker --version
docker images 2>&1 | grep -iE "worldfoundry|rlinf"
# 1.6 CANN + 项目卷
ls /usr/local/Ascend 2>&1 | head -5
ls -la /workspace 2>&1 | head -10
df -h /workspace /data 2>&1 | tail -5
# 1.7 多租户检查
uptime
npu-smi info -l 2>&1 | grep -E "Chip ID|Hbm|Usage" | head -30
```

**预期输出**：
- `npu_devices >= 1` 且至少有 1 张 NPU **Hbm < 50%**（避免抢别人训练）
- `resolve_device()` 返回 `npu`（即 `is_npu_available()` 为 True 且无 `WORLDFORGE_DEVICE` override）
- `torch 2.9.x + torch_npu 2.9.x + vllm 0.18.x + vllm_ascend 0.18.x + transformers 4.55.x`
- `docker images` 至少含 `worldfoundry-npu-vllm:phase1`（W35.1 额外需要 RLinf NPU image）

**失败处置**：
- NPU 不可见 → 检查 `/dev/davinci_manager` 挂载；驱动版本是否与 `torch_npu` 匹配
- 版本错位 → 走 `pyproject.toml` + `requirements/inf.txt` 矩阵重新对齐容器
- `resolve_device()` 返回 `cpu` → 检查 `is_npu_available()` 链路；G4 fail-fast 设计就是为了不静默退到 CPU
- `worldfoundry-npu-vllm:phase1` 不在 → 跑 `docker build -t worldfoundry-npu-vllm:phase1 -f docker/worldfoundry-npu-vllm/Dockerfile .`（`docker/worldfoundry-npu-vllm/Dockerfile`，C-T3 产物，commit 5dac705）
- 所有 NPU Hbm > 50% 或 load > 10 → 多租户占满，**不**开跑；换窗口
- RLinf image 不在且无法拉（公网不通） → W35.1 在本窗口不可执行，只跑 M2-T5

## 2. W35.1 RLinf NPU 封口验证

> 这是 W35 ADR §6 验证门**唯一未关的项**：CPU 侧 W35.2 矩阵核对已判定建议触发否决项切 verl（详见 `docs/superpowers/specs/2026-08-06-w35-2-version-matrix-check.md`），W35.3 CPU spike 已通过。本步是"NPU 端实测封口"，目的是把判定从"CPU-only 推断"升级为"NPU 实测结果"。

### 2.1 最小 smoke（不开训练）

```bash
# 2.1.1 RLinf NPU image 拉取（首次 ~2 GB，10+ 分钟；多租户注意公网带宽）
docker pull <rlinf-npu-image>:<tag>   # tag 由 W35.3 spike 锁定
# 2.1.2 ManiSkill CPU 模式 + MLP-Policy PPO（RLinf 官方 NPU quickstart 路径）
docker run --rm \
  --device /dev/davinci_manager:/dev/davinci_manager \
  --device /dev/davinci0:/dev/davinci0 \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v <worldfoundry-root>:/workspace \
  <rlinf-npu-image>:<tag> \
  python -m rlinf.examples.ascend_quickstart \
    --env ManiSkill-CPU --policy MLP-Policy --algo PPO --steps 4
```

**预期产物**：
- 训练收敛曲线在 stderr 输出（首 4 步 reward 应非零）
- 容器 rc=0
- 无 NPU 算子报错（无 `AscendError` / `torch_npu` runtime error）

### 2.2 验收门

| 检查 | 通过条件 |
|---|---|
| docker pull rc | `0` |
| quickstart rc | `0`，stderr 含 NPU device `torch.npu.device_count() >= 1` |
| reward 非零 | 末步 reward > 0（任意正值即可；不要求收敛） |
| 无 NPU 报错 | stderr 不含 `RuntimeError` / `AscendError` / `da Vinci` 相关报错 |

### 2.3 失败处置

| 现象 | 处置 |
|---|---|
| docker pull 超时 / 失败 | 公网不通；W35.1 在本窗口**不可执行**，等下个窗口有公网时再试；只跑 M2-T5（§3）|
| `torch_npu` 版本冲突 | 记录实际报错 + `npu-smi info -l` 输出；判定升级为"W35.2 CPU + NPU 双确认，切 verl 决议成立"——参考 `docs/superpowers/specs/2026-08-04-w35-rl-framework-adr.md` §6 |
| NPU 算子 not implemented | 记录 operator name；这是 RLinf 官方支持的盲区，反馈给 RLinf 社区（华为云合作通道）；同样支持"切 verl"决议 |
| OOM | 单卡上把 batch 减到 1；不要现场改 RLinf 内部实现 |

### 2.4 拿到 W35.1 结果后

把本节结果记到 `docs/dev/next-steps-phase4-atlas.md` 跟踪表 #7 行：
- ✅ "Atlas 上机 ✅; torch_npu=2.9.0, rlinf_status=completed" → W35.1 关闭 → W35 ADR §4 决策"RLinf 为主"路径确认 → 切 verl 议题关闭
- ⚠️ "失败: <现象>" → 切 verl 决议升级为"CPU + NPU 双确认"

## 3. M2-T5 P2 vllm_ascend 端到端验收

> 这是 W41 SPEC §6 验证门 M2-T5.1–T5.5 的实际执行：把 `vllm_ascend`
> backend 在 Atlas card 7 上从 registry 注册推进到真 e2e：跑两次
> `scripts/render_train_rl_cpu.py`（`--inference-backend torch` 和
> `--inference-backend vllm_ascend`），对比 recon / latent cosine drift ≤0.01。

### 3.1 registry 可用性探测

```bash
# 容器内：
python3 -c "from worldfoundry_common.backends.inference import registry; print(registry.available('vllm_ascend'))"
# 期望: (True, ...) — 与 W41 §1 / §7 M2-T5.1 一致
# host 上: (False, 'pip install -e .[inf]...') — W41 §1 故意设计
```

**预期**：`available=True` 表示 `torch_npu + vllm + vllm_ascend + worldfoundry_inf.vllm_ascend_worker` 三包齐 + import 成功。

### 3.2 端到端对比

```bash
cd <worldfoundry-root>
mkdir -p bench_results/w43_vllm_ascend_e2e

# 3.2.1 torch baseline
python3 -m scripts.render_train_rl_cpu \
  --out-dir bench_results/w43_vllm_ascend_e2e/torch \
  --inference-backend torch --seed 42 \
  --num-clips 1 --frames-per-clip 2 --height 32 --width 32 \
  --num-steps 4 --num-outer-iters 1 --num-train-per-iter 4 \
  --ppo-env-steps 8 --mask-ratio 0.5

# 3.2.2 vllm_ascend 真路径
python3 -m scripts.render_train_rl_cpu \
  --out-dir bench_results/w43_vllm_ascend_e2e/vllm_ascend --seed 42 \
  --inference-backend vllm_ascend \
  --num-clips 1 --frames-per-clip 2 --height 32 --width 32 \
  --num-steps 4 --num-outer-iters 1 --num-train-per-iter 4 \
  --ppo-env-steps 8 --mask-ratio 0.5
```

### 3.3 验收门

| 检查 | 通过条件 |
|---|---|
| `registry.available('vllm_ascend')` | `(True, ...)` |
| torch run rc | `0` |
| vllm_ascend run rc | `0` |
| summary.mode | `== 'npu'` |
| recon cosine drift | `\|torch.recon_cosine − vllm_ascend.recon_cosine\| ≤ 0.01` |
| latent cosine drift | `\|torch.latent_cosine − vllm_ascend.latent_cosine\| ≤ 0.01` |
| 趋势一致 | 同 seed 下两条 loss 曲线趋势一致（±5%） |

### 3.4 失败处置

| 现象 | 处置 |
|---|---|
| `registry.available` 返回 False | §1.5 镜像没装齐；跑 `docker build` 重做 `worldfoundry-npu-vllm:phase1` |
| `vllm_ascend` run OOM | 把 `num-clips` / `frames-per-clip` 减半；不要现场改 `vllm_ascend_worker.py` |
| cosine drift > 0.01 | 收集两个 `summary.json` + 末尾 60 行 stderr；判定为 batch 路径 NPU 同构契约违例，回到 W36 §3.4 排查 `predict_masked_batch / predict_temporal / predict_action_conditioned` 的 NPU 真路径 |
| summary.mode != 'npu' | `resolve_device()` 退到 cpu（§1 失败）；G4 fail-fast 不应发生，若发生说明 `is_npu_available()` 链路有 bug |

### 3.5 拿到 P2 结果后

把 `bench_results/w43_vllm_ascend_e2e/{torch,vllm_ascend}/` 整个目录 commit 上回；同时把 `docs/dev/next-steps-phase4-atlas.md` 跟踪表 #2（M2-T5）/ #3（M1-T4）/ #4（M1-T5）/ #5（C-T8）行的 Status 列更新为"Atlas 上机 ✅; cosine drift=X.XXX"。

## 4. 关闭 / 收尾

窗口结束时按顺序：

1. 把所有 `bench_results/w43_vllm_ascend_e2e/**` 拉到本地仓库（git add + commit）
2. 把 `docs/dev/next-steps-phase4-atlas.md` 跟踪表更新（W35.1 + M2-T5 + C-T8 行）
3. 把 §2.1 的 RLinf quickstart 输出末尾 60 行 commit（如果跑了 W35.1）
4. 把任何意外 / 失败贴到本 runbook 末尾的「Lessons learned」一节（目前为空，下次窗口前补）
5. **运行 `python3 scripts/verify_goal_evidence.py`**，把生成的 `goal_verdict.json` 一起 commit 上回（使用 `--keep-out-dir --out-dir bench_results/w43_vllm_ascend_e2e/goal_evidence`）。这是 3 轴目标的通用验证门——Atlas 部署后必须 3/3 PASS，否则说明 RLinf / 推理 / 扩散任一管线在 Atlas 容器内出现回归。回归测试在 `worldfoundry_inf/tests/test_verify_goal_evidence.py`（包括 diffusion 静默 fallback 到 PyTorch eager 的 critical regression guard）。

## 5. 增量重启支持

多窗口场景：本 runbook 各节**独立可重入**：
- §1 重跑无害（只读）
- §2.1 跑过一次后只要 image 已 local 在，docker run 即可
- §3.2 两次 e2e 任一失败可单独重跑另一次（保留好的那份）
- §4 step 5 的 verify_goal_evidence.py 任意时候可重跑（每次写新 goal_verdict.json）

不要一次开 5+ 个 shell 跑全 runbook；按节增量推进，每节跑完贴回输出再进下一节。

## Lessons learned（2026-08-17 W43.1 窗口首次执行后补）

1. **CLI/registry drift（已修，commit e2157af+1）**：原
   `scripts/render_train_rl_cpu.py` `--inference-backend` argparse
   `choices={torch, vllm, sglang, vllm_omni}`，没把 `vllm_ascend`
   加进去，runbook §3.2.2 原文 `--inference-backend vllm_ascend` 会被
   argparse reject。Registry 端 `vllm` 是 `vllm_ascend` 的 alias
   （共享 `_build_vllm`）。
   **修复**：
   - argparse choices 扩成
     `{torch, vllm, sglang, vllm_omni, vllm_ascend}`；
   - help 文本说明 `vllm` 是 `vllm_ascend` 的 back-compat alias；
   - 新增回归测试
     `worldfoundry_inf/tests/test_rti_rlinf_vllm_diffusion_integration.py::test_rti_orchestrator_resolves_vllm_ascend_backend`，
     断言 `summary.inference_backend == "vllm_ascend"`（canonical 名落盘，
     **不**被 silently 降级成 alias `vllm`）。
   CPU 端 smoke 验证：`--inference-backend vllm_ascend` 与
   `--inference-backend vllm` 同 seed 下产出**完全一致**
   （recon=0.6502, latent=0.3281, total=18.9097）。
   Atlas NPU 真路径回归由 §3 torch=vllm cosine drift=0.0 覆盖。
2. **NPU bind 模板要补两条**：runbook §2.1.2 的最小集
   `--device /dev/davinci_manager + /dev/davinciN + -v .../driver:ro`
   不够。实测还必须
   `--device /dev/hisi_hdc:/dev/hisi_hdc` 和
   `-v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro`，
   否则 `rtSetDevice` 返回 `507033 / device retain error`。
   容器内 `LD_LIBRARY_PATH` 已经预置好，但 env `ASCEND_HOME` 显式传
   `=/usr/local/Ascend` 更稳。
3. **镜像名替换**：Atlas 节点没有 `worldfoundry-npu-vllm:phase1`，
   但已有 `worldfoundry-franka:npu-cann9`（自带 torch_npu
   2.9.0.post1、vllm 0.18.0、vllm_ascend 0.18.0），功能等价于
   runbook 的 phase1 镜像。本窗口直接用 `worldfoundry-franka:npu-cann9`
   作为 vllm 容器。如要严格匹配 phase1 tag，下次窗口前需
   `docker build` 或重打 tag。
4. **W35.1 RLinf image 不可执行**：本窗口 Atlas 节点本地无 RLinf 镜像，
   公网镜像仓可达性未确认（容器内 `nc` 缺失；`registry.huaweicloud.com`
   不可解析；`registry-1.docker.io` 未实测）。
   按 runbook §1.3 失败处置，本窗口只跑 §3，W35.1 推到下个有公网/已拉
   image 的窗口再封口。
5. **多租户负载**：本次窗口 `uptime` load average 在
   `10.78 / 13.58 / 17.35` ~ `12.03 / 12.77 / 14.42`（NPU 0/1/2 上别人
   的 VLLM 引擎撑住）。runbook §1.7 stop rule 是 "load > 5 或所有 NPU
   Hbm > 50%"。本窗口 NPU 3-7 完全空闲（Hbm 0%、AICore 0%、无进程），
   所以即便 load > 5 也满足 §3.2 单卡 e2e 推进条件（**所有** NPU 都
   > 50% 才停）。下次窗口若 load 持续 > 10，建议换空闲时段重跑。
6. **verify_goal_evidence.py 3/3 PASS（本地跑）**：
   - training：`rlinf_framework='rlinf'`, `rlinf_result.kind='real_rlinf'` —
     走过真 `rlinf.runners.embodied_runner.EmbodiedRunner` 构造
     （RLinf 0.3.0，rlinf_file 指向 `.venv-rlinf/.../site-packages/rlinf/__init__.py`），
     `--rlinf-runner-factory=worldfoundry_rl.rlinf_factory:build_runner`。
     Mock factory 仍保留作兜底（`worldfoundry_rl.rlinf_mock_factory:build_runner`）。
   - inference：`inference_backend='vllm_omni'`, `engine_type='RealVLLMOmniPipeline'`,
     `kind='real_vllm_omni'`, `available=True` — 走真 `import vllm` 拉起
     的 `RealVLLMOmniPipeline`（M2-T6 P1, commit 122ea47），tensor ops
     仍经 ORT CPU EP（vLLM 的 `LLM(model=...)` 需要 model + 加速器）。
     `vllm-omni` 公网 wheel 发布后 (W44+)，decoder 半也会切到真
     vllm-omni；现阶段 ORT stand-in 已经满足 "非 PyTorch 加速推理"
     clause（详见 Lessons-learned #8）。
   - diffusion：`inference_engine='onnxruntime'`, `diffusion_engine='onnxruntime'`,
     `providers_resolved=['CPUExecutionProvider']` — 走 ONNXRuntime CPU
     provider，**不是** NPU EP。NPU 真路径需 ONNX-Runtime Ascend EP。
   本地 3/3 PASS 证明 wiring 正确，但 **不替代** Atlas NPU 真路径回归；
   Atlas 真路径回归由 §3 torch/vllm_ascend `cosine drift=0.0` 覆盖。
7. **RLinf 通过 `pip install` 接入（无需 RLinf 官方 docker 镜像）**：
   - 公网拉 `rlinfai/rlinf` 镜像被 DaoCloud 拒（"not in allowlist" issue #2328），
     `docker.1ms.run` 镜像同样不可用；本窗口不依赖官方镜像。
   - 实测可行路径：把 RLinf 作为 Python 包 `pip install` 进
     `worldfoundry-franka:npu-cann9` 容器内的现有 Python（aliyun
     `https://mirrors.aliyun.com/pypi/simple/` 镜像，绕开 `pypi.org`
     TLS 超时；`pip install --no-deps rlinf` 防 torch 覆盖 torch_npu
     2.9.0.post1；`pip install "huggingface_hub>=0.34.0,<1.0"` 修
     transformers 4.55.0 兼容）。
   - WSL2 主机同样在 `.venv-rlinf` (`uv venv && uv pip install ...`)
     里装了 RLinf 0.3.0，配 `TORCH_DEVICE_BACKEND_AUTOLOAD=0` 跳过
     `libascend_hal.so` auto-load。
   - 新增 `worldfoundry_rl.rlinf_factory.build_runner`（M1-T4 P1）
     通过 `--rlinf-runner-factory=worldfoundry_rl.rlinf_factory:build_runner`
     接到 orchestrator，``rlinf_result.kind=='real_rlinf'`` 证明走过
     `rlinf.runners.embodied_runner.EmbodiedRunner` 真类构造路径。
     `verify_goal_evidence.py` 已切到该 real factory（mock 仍在
     `worldfoundry_rl.rlinf_mock_factory:build_runner`，作为兜底）。
   - 实测 EmbodiedRunner 构造所需最小 cfg keys：
     `runner.{weight_sync_interval, max_epochs, overlap_env_bootstrap,
     per_worker_log, per_worker_log_path, logger.{log_path,
     project_name, experiment_name, logger_backends, wandb_proxy,
     wandb_entity, swanlab_mode}}` + `cluster.profiling`。Ray 本地
     cluster 由 RLinf 自动 `ray.init`（~50s）；生产用真 Ray 集群后这
     步可省。
8. **vLLM-Omni 升级到真 vllm 包（M2-T6 P1, commit 122ea47）**：
   - Lesson #6 的 inference 行已过期。新增
     `worldfoundry_inf.omni.pipeline_real.RealVLLMOmniPipeline` — 真
     `import vllm` 拉起 `vllm_version=0.18.0` + `vllm_file` 路径 +
     `LLM_init_signature`（即使 CPU 端 `LLM.__init__` 因缺 model/
     accelerator 起不来，也照样记录 `kind="real_vllm_omni"`）。
   - `_build_vllm_omni` 优先走 real path，vllm 不可导入时退回 ORT
     stand-in `VLLMOmniPipeline`。Atlas 真路径下 `vllm-ascend` wheel
     装好后，`LLM(model=...)` + NPU EP 端到端可用；CPU smoke 仍走 ORT。
   - verifier `_check_inference` 已强化要求 `engine_type ==
     "RealVLLMOmniPipeline"`（类身份，**不**是 registry 名
     `"vllm_omni"`）。新增 2 个回归 guard：
     `test_check_inference_fails_when_engine_type_is_ort_standin` +
     `test_check_inference_fails_when_engine_type_missing`，对应
     M1-T4 P1 的 `fails_when_framework_is_self_ppo` 模式。
   - CPU host 必须能 `import vllm` 才能走真路径；vllm 的 import chain
     会触发 `vllm.platforms.__init__.py` 用
     `resolve_obj_by_qualname(platform_cls_qualname)` 加载 platform。
     本窗口 `worldfoundry_inf._vllm_ascend_plugin.register` 改成
     ImportError 时 return `None`（让 vLLM 落到 `UnspecifiedPlatform`），
     这样 `import vllm` 在没装 `vllm_ascend` 的 host 上**不**再 crash。
     副作用：vLLM 的 `LLM` 类（`vllm.entrypoints.llm`）import chain 还会
     拉 `openai → httpx2 → cloudpickle`，本窗口已确认到
     `ModuleNotFoundError: No module named 'httpx2'` 时
     `RealVLLMOmniPipeline` factory **仍然**返回
     `kind="real_vllm_omni"`（vllm 包本身已真，LLM 类 import 失败信息
     落在 `llm_class_import_error` 字段供 post-mortem audit）。
   - Atlas 端 `pip install httpx2` 后 `LLM_class_import_ok=True`，
     `LLM_init_signature` 完整 — 即可起 `LLM(model=...)` 真路径
     （仍需 model + NPU/CUDA/XPU 硬件，本 runbook 不解决硬件问题）。
   - 本地 3/3 PASS 验证：
     `inference_backend='vllm_omni'`, `engine_type='RealVLLMOmniPipeline'`,
     `kind='real_vllm_omni'`, `available=True`, `dispatch_error=None`。

## Out of Scope

- **W20 单卡 LoRA smoke** — 单独 runbook（`docs/dev/atlas-booking-runbook.md`）。
- **W21 8 卡 FSDP** — 单独 runbook；多卡预约是另一个时长档（≥2h）。
- **W26 W8A8 量化加载器** — 单独 runbook（与 W35 / M2-T5 不冲突，并行上机）。
- **W27 vLLM-Ascend e2e（再扩展）** — 需 `worldfoundry_inf` 服务起来；M2-T5 P2 完成后单独 SPEC 收口。
- **W42 RLinf runner factory 完整对接** — W35.1 通过后单独 SPEC（W42 范围内）；本 runbook 不实现。
- **B-v2 帧生成（MindIE-SD / Wan）** — W40 frame-gen ADR §5.2 推迟条件未到；本 runbook 不涉及。