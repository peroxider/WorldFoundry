# SPEC-W42：RLinf Runner-Factory CLI 接入与 Ray Cluster 端到端路径

> 状态：提议（Proposed）
> 日期：2026-08-15
> 关联：`docs/roadmap/2026-08-14-opensource-extensions-loop-todo.md` §9.2 P1
> （M1-T3）+ §9.3 P2（M1-T4 / M1-T5），`docs/superpowers/specs/2026-08-04-w35-rl-framework-adr.md`
> §4（框架选型；verl > RLinf），`docs/superpowers/specs/2026-08-04-w35-3-rlinf-custom-policy-spike-plan.md`
> （RLinf spike 评估；W35-3 产物），`docs/superpowers/specs/2026-08-06-w36-npu-worker-parity-contract.md` §4（NPU worker seam）。
> 决策范围：M1 主线（RL 框架可插拔）的 Phase-1 CLI seam 收口——
> `--rl-backend {self_ppo, verl, rlinf}` + `--rlinf-runner-factory <callable>` flag 接入
> `scripts/render_train_rl_cpu.py`，host smoke rc=0，Atlas card 7 上 Ray cluster e2e 趋势与 self_ppo 一致。
> 影响：M1-T1 / M1-T2 的 `RLinfRunnerConfig` + `RunnerFactory` Protocol + `WorldModelActor/Critic`
> 薄包装已在 W35-3 spike 落地（`worldfoundry_common/adapters/rlinf.py`）；本 SPEC 补完 CLI seam
> （M1-T3）+ Atlas e2e（M1-T4）+ verl fallback（M1-T5）。

---

## 0. TL;DR

把既有 `RLinfRunner` / `RLinfRunnerConfig` / `RunnerFactory` Protocol（在
`worldfoundry_common/adapters/rlinf.py` 已经落地的 W35-3 spike 成果）由
in-process class 升级为 orchestrator 命令行可达：新增 `--rl-backend rlinf` +
`--rlinf-runner-factory <dotted.path>` 两个 flag，
`build_runner` 处按 backend 分支调用 `RLinfRunner(run)`.run(**kwargs)（mock factory 不依赖
Ray 集群即可创建 runner）；`self_ppo`（shipped）/ `verl`（plan M1-T5）/ `rlinf`（plan
M1-T3/M1-T4）三态收口。Atlas card 7 上用 `bench_results/w43_rlinf_e2e/` 跑 Ray
cluster 同 seed 趋势与 self_ppo 一致（±5%）。

---

## 1. 背景与问题

WorldFoundry 的 M1 主线是 **RL 框架可插拔**（roadmap §M1）：用一条 seam 让 orchestrator
可以在自研 `PPO`（已 ship）/`verl`（已 ship，社区版）/ `RLinf`（W35 spike，
未 ship）三种 RL framework 之间切换。当前 `scripts/render_train_rl_cpu.py`
只走 v2.x 内置 PPO 路径，没有 `--rl-backend` flag；同时 `worldfoundry_common/adapters/rlinf.py`
已经定义好 `RLinfRunner` / `RLinfEnvWorker` / `RLinfRunnerConfig` /
`RunnerFactory` Protocol——M1-T1 / M1-T2 的 seam 已经在位，差的是 CLI 入口
接进去。roadmap §9.2 P1 M1-T3 = 接 CLI。

### 关键事实

- **`RLinfRunner` + `RLinfRunnerConfig` + `RunnerFactory` Protocol 已经落地**
  （W35-3 spike 成果，`worldfoundry_common/adapters/rlinf.py:1-209`）：
    - `RLinfRunner`（line 37-82）：适配 RLinf `EmbodiedRunner`，持有
      `runner_factory: Callable[[dict], Any]`；缺 factory 时跑 `_ensure_runner` 抛
      `NotImplementedError`。
    - `RLinfRunnerConfig`（line 155-194）：`@dataclass`，字段含
      `cluster_role` / `worker_groups` / `world_model_builder` / `extras`；
      支持 `to_dict()` 兼容旧 dict 调用点。
    - `RunnerFactory`（line 196-208）：`@runtime_checkable Protocol`，
      `__call__(config: RLinfRunnerConfig | dict) -> Any`，
      `isinstance(factory, RunnerFactory)` 可判。
- **`verl` 走的是 `requirements/rl.txt` 的 `verl==0.5.0` ship path**（C-T1 锁定），
  已经被项目 `pyproject.toml [project.optional-dependencies].rl` 列为正式可选
  extra。但 orchestrator 当前不调它（M1-T5 推迟到 P2）。
- **M1-T2 的 `WorldModelActor` / `WorldModelCritic` 薄包装**复用
  `worldfoundry_train.model.hf_vjepa_shim.HfJEPAPredictorShim.forward(action=...)`；
  实际代码位置需要确认：roadmap §8.1 提到有"4 个新单测" 覆盖 mock factory 不依赖
  Ray 集群即可创建 runner，意味着 RLinfRunner 已经被 host-side 用 mock factory
  测过——本 SPEC 不再花工程量重新实现。
- **W35 RL 框架 ADR §4 决策**：verl > RLinf（推迟到 W35-3 spike）。spike 产物
  是"`RLinfRunner` 可行，但需要 Ray cluster + 自定义 worker groups，工程量大"。
  本 SPEC 收口不重新评估框架选型——只补 CLI seam（与 §4 决策一致：先 verl ship，
  RLinf 推迟）。
- **W35.2 ADR §2.1 与 §5.2**：RLinf v0.3.0 拉 torch 2.10.0，与 torch_npu 2.9.0
  有 minor 冲突——project 锁版是 torch 2.9.0，所以 RLinf 真实端到端须在
  `worldfoundry-npu-vllm:phase1`（含 `vllm-ascend==0.18.0` 兼容栈）容器内
  跑——本 SPEC 在 §6 显式约束。
- **`--rl-backend` flag 当前不在 orchestrator**：`scripts/render_train_rl_cpu.py`
  的 `parse_args` 没有这条；roadmap §9.2 P1 M1-T3 落地要求在 `parse_args`
  增加 2 个 flag + 在 `build_runner` 处按 backend 分支调用 `RunnerFactory`。
- **`--frame-gen-backend` 与本 SPEC 不耦合**：v1 default `none`；M3-T8 才走
  `mindie_sd`，由独立 SPEC（W44+）收口。

### 与 roadmap §M1 风险的对照

- **RLinf 自定义 actor / critic worker 复杂性**：roadmap §5 风险表列了
  "RLinf runner 包装复杂度高"——本 SPEC 通过把"factory 注入"作为唯一耦合点
  来规避：orchestrator 不知道 RLinf 内部细节；只持有一个 `runner_factory` callable
  ref，落地由工厂负责。
- **Ray cluster 启动开销**：Atlas 单机跑 unit smoke 不需要 Ray 集群——`mock factory`
  单测路径已经在位（M1-T1 阶段已交付）。
- **verl 与 RLinf 不同时存在**：默认两者互斥（不会 dual-import）；
  `requirements/rl.txt` 同时列 `verl==0.5.0 + ray==2.56.1`，跟 RLinf 共享 Ray 这一底座，
  不冲突。

---

## 2. 硬性约束

| # | 约束 | 来源 |
|---|---|---|
| C1 | `--rl-backend rlinf` 在 host 加 mock factory 后跑通 smoke rc=0 | roadmap §9.2 P1 M1-T3 验收 |
| C2 | `RLinfRunner` / `RLinfRunnerConfig` / `RunnerFactory` 已经 ship（M1-T1/T2 W35-3 spike 落地） | `worldfoundry_common/adapters/rlinf.py:1-209` |
| C3 | orchestrator 不直接 import RLinf / Ray 内部 | seam 设计；注入 `runner_factory` 作为唯一耦合点 |
| C4 | `self_ppo` 路径不能被破坏（既有 v2.x 默认） | 向后兼容；orchestrator 不带 `--rl-backend` 时行为不变 |
| C5 | `verl` 路径预留 hook 但不实施（M1-T5 P2） | 本 SPEC 是 seam 收口，verl deferred |
| C6 | `requirements/rl.txt` 锁 `verl==0.5.0 + ray==2.56.1` | C-T1 版本矩阵；不重写 |
| C7 | `rlinf` 真实路径需要 Ray cluster + worker groups，host smoke 仅 mock | W35 ADR §4 + W35.2 §5.2 minor 冲突 |
| C8 | `--rl-backend rlinf` 默认走 mock factory（host smoke），Atlas 才接真 cluster | roadmap §9.2 P1 + §9.3 P2 分层 |

---

## 3. 决策驱动因素

| 驱动因素 | 权重 | 说明 |
|---|---|---|
| D1 seam-level 兼容性（不破坏 v2.x 默认 PPO 行为） | ★★★ | 既有 454+ 单测 + 5-phase orchestrator smoke 不能 regress |
| D2 mock factory 单测覆盖（不依赖 Ray 集群） | ★★★ | M1-T1/T2 已经用 mock factory 闭合过；本 SPEC 复用既有路径 |
| D3 CLI flag 命名一致性（与 C-T5 README §"Optional runtimes" 对齐） | ★★ | `--rl-backend {self_ppo, verl, rlinf}` 不破 README |
| D4 W35 ADR §4 决策遵守（verl > RLinf） | ★★★ | verl 是 ship path；RLinf 是 deferred；本 SPEC 不重评估 |
| D5 W35.2 版本矩阵兼容（torch 2.9 vs RLinf v0.3 拉 torch 2.10） | ★★ | host mock 可，Atlas 真路径需 `worldfoundry-npu-vllm:phase1` 容器内跑 |
| D6 Atlas card 7 真 e2e 可达（M1-T4 P2 验收） | ★★ | Ray cluster 与 GPU worker groups 在容器内拉起；本 SPEC 不锁 |

---

## 4. 方案对比

| 维度 | A. `argparse` 加 `--rl-backend` + dispatch 字典 | B. `argparse` 加 `--rl-backend` + 子模块 provider registry | C. orchestrator 暴露 `--runner-factory path.to.fn` 通用入口 |
|------|-----------------------------------------|----------------------------------------------|-----------------------------------------------------|
| D1 向后兼容 | 🟢 当 flag 缺省 → `self_ppo` | 🟢 同 | 🟢 当 factory 缺省 → `self_ppo` |
| D2 mock factory 单测 | 🟢 直接传 callable ref | 🟢 同 | 🟢 同 |
| D3 CLI flag 命名 | 🟢 对齐 README | 🟢 同 | 🔴 通用 `--runner-factory` 不区分 verl/rlinf，README drift |
| D4 W35 ADR 决策 | 🟢 显式三态 | 🟢 同 | 🟡 隐式 |
| D5 W35.2 版本矩阵 | 🟢 同等 | 🟢 同 | 🟢 同 |
| D6 Atlas e2e | 🟢 RLinf 三态必选 | 🟢 同 | 🟡 config 不直观 |
| 改动量 | 🟢 ~50 LOC + 1 mock factory test | 🔴 引入新 provider 抽象 | 🔴 引入 generic config parsing |
| **本 SPEC 推荐** | **推荐** 🟢 | 不推荐 | 不推荐 |

---

## 5. 决策

**A 方案**：`scripts/render_train_rl_cpu.py` 的 `parse_args` 增加
`--rl-backend {self_ppo, verl, rlinf}` (default `self_ppo`) + `--rlinf-runner-factory <dotted.path>`
两个 flag；`build_runner` 改成 dispatch 字典：

```python
def build_runner(args, env_pool):
    if args.rl_backend == "self_ppo":
        return PPO(policy, cfg)  # 既有的 v2.x 路径，不动
    if args.rl_backend == "rlinf":
        factory = import_dotted(args.rlinf_runner_factory)  # 支持 ``pkg.mod:attr``
        runner = RLinfRunner(runner_factory=factory)
        return runner  # 走 RLinfRunner.run(cfg)
    if args.rl_backend == "verl":
        raise NotImplementedError(
            "verl backend deferred to M1-T5 (P2). Today, falls back to self_ppo."
        )  # 保留 hook；M1-T5 实施（§6 Phase 2）
```

`RLinfRunner.run()` 已经返回 `{"framework": "rlinf", "status": "completed"}`；
mock factory 单测路径已经在 M1-T1 阶段交付（roadmap §8.1 "4 个新单测；mock factory
不依赖 Ray 集群即可创建 runner"）。

### 5.1 决策理由（按权重分层）

1. **D1 向后兼容决定**（★★★）：orchestrator 不带 `--rl-backend` 时走既有的
   v2.x PPO 路径，行为不变；既有 454+ 单测（含 `test_render_train_rl_cpu.py` 28 个）
   不需要改一行。452/454 通过率应在 flag additive 后保持 100%。
2. **D2 mock factory 单测复用**（★★★）：`worldfoundry_common/adapters/rlinf.py`
   已经提供 `RLinfRunner(runner_factory=lambda cfg: MockRunner())` 模式；
   orchestrator 仅持 callable ref，让单测在不拉 Ray 集群的情况下走完整路径。
3. **D3 CLI 命名一致性**（★★）：C-T5 README §"Optional runtimes" 已经定义
   `--rl-backend {self_ppo, verl, rlinf}`；本 SPEC 落到 parse_args 时严格使用该
   choices 列表，README 与 orchestrator choices 不 drift。
4. **D4 W35 ADR §4 决策遵守**（★★★）：verl > RLinf；本 SPEC
   不重评估，但保留 `--rl-backend verl` 选项 + 显式 `NotImplementedError` 指向 M1-T5，
   与 ADR 决策一致。
5. **D5 W35.2 版本矩阵兼容**（★★）：host 上 RLinf v0.3 拉 torch 2.10 与
   torch_npu 2.9 minor 冲突——host smoke 走 mock factory（不真 import RLinf）规避；
   Atlas 真路径在 `worldfoundry-npu-vllm:phase1` 容器内，由 M1-T4 P2 验收。
6. **D6 Atlas e2e 可达**（★★）：本 SPEC 不锁 M1-T4 实现细节；只确保
   `--rl-backend rlinf --rlinf-runner-factory <真实 RLinf 工厂>` 在 Atlas 容器内
   可调用，由 M1-T4 spec（不强在本 SPEC 内）确定真 factory 内容。

### 5.2 与 W35 / W35.2 / W35-3 的因果绑定

- W35 ADR §4：verl > RLinf（framework 选型；先 verl ship，RLinf deferred）
- W35.2 ADR §5.2：RLinf v0.3 vs torch 2.9 minor 冲突；本 SPEC host smoke 走 mock factory 规避
- W35-3 spike：RLinfRunner + WorldModelActor/Critic 薄包装落地；本 SPEC 复用其成果

### 5.3 verl 的去留

`verl` choices 留 hook 不实施。M1-T5 P2 落地时把 `NotImplementedError` 替换成
`VerlRunner(...)` 实现；orchestrator choices 不动。本 SPEC 是 seam 收口 + hook。

### 5.4 import_dotted 工具

orchestrator 加一个 5-LOC `import_dotted(path)`：支持 `pkg.mod:attr` 或
`pkg.mod.attr` 两种 dotted 写法，找不到时抛 `ValueError` 列出可能漏 export 的
符号。1 单测覆盖。

---

## 6. 实现路径

```
W42 (本 SPEC) — host 可达 + Atlas
  ↓
Phase 1 — CLI seam + mock factory 单测（host 闭环）
  ├─ scripts/render_train_rl_cpu.py:
  │   - parse_args 增加 --rl-backend (choices=("self_ppo", "verl", "rlinf"), default="self_ppo")
  │   - parse_args 增加 --rlinf-runner-factory (dotted path or None)
  │   - 新增 build_runner(args, env_pool, policy, cfg) 三态 dispatch
  │   - 新增 import_dotted(path) 5-LOC 工具
  ├─ worldfoundry_train/tests/test_render_train_rl_cpu.py:
  │   - test_rl_backend_self_ppo_default_unchanged（既有行为不变）
  │   - test_rl_backend_rlinf_with_mock_factory_runs
  │   - test_rl_backend_rlinf_without_factory_raises_actionable_error
  │   - test_rl_backend_verl_raises_notimplemented_m1t5
  │   - test_import_dotted_handles_colon_and_dot
  └─ 验收: .venv/bin/python -m pytest worldfoundry_train/tests -q
           expect: 已有 454 测试 + 5 new = 459 通过

Phase 2 — verl 真路径（M1-T5 P2，不在本 SPEC 范围）
  └─ 参照 W35 ADR §4 决策实施；本 SPEC 仅预留 hook

Phase 3 — RLinf + Ray cluster e2e（M1-T4 P2）
  ├─ rsync W42 提交到 /home/w30066257/WorldFoundry/
  ├─ docker build -t worldfoundry-npu-vllm:phase1 -f docker/worldfoundry-npu-vllm/Dockerfile .
  ├─ 在容器内拉起单机 Ray cluster：
  │     ray start --head --num-cpus=4
  │     scripts/render_train_rl_cpu.py \
  │       --rl-backend rlinf \
  │       --rlinf-runner-factory worldfoundry_rl.adapters.factories:build_min_rlinf_runner \
  │       --seed 42 ...
  └─ bench_results/w43_rlinf_e2e/：
        outer_loop.mean_wm_total_loss 同 seed 与 self_ppo 趋势一致（±5%）
        Ray dashboard 无 failed worker
```

---

## 7. 验证门（M1-T1 / M1-T2 / M1-T3 / M1-T4 / M1-T5 各自清单）

| 步骤 | 内容 | 通过标准 |
|---|---|---|
| M1-T1.1 | `RLinfRunner` / `RLinfRunnerConfig` / `RunnerFactory` Protocol 已 ship | `worldfoundry_common/adapters/rlinf.py:1-209` 存在 |
| M1-T1.2 | mock factory 不依赖 Ray 即可创建 runner | roadmap §8.1 "4 个新单测" 通过 |
| M1-T2.1 | `WorldModelActor` / `WorldModelCritic` 薄包装 ship | 复用 `HfJEPAPredictorShim.forward(action=...)` |
| M1-T3.1 | `--rl-backend {self_ppo, verl, rlinf}` choices 与 README 对齐 | `scripts/render_train_rl_cpu.py --help` 含 3 值 |
| M1-T3.2 | `--rlinf-runner-factory` 支持 `pkg.mod:attr` 与 `pkg.mod.attr` | 单测覆盖两种 dotted 写法 |
| M1-T3.3 | mock factory 路径 host 上 rc=0 | `scripts/render_train_rl_cpu.py --rl-backend rlinf --rlinf-runner-factory ...test.mock_factory ...` rc=0 |
| M1-T3.4 | 缺 factory 时 actionable error，不静默 fallback | `NotImplementedError` 含 M1-T5 引用 |
| M1-T3.5 | `--rl-backend verl` 不实施但可控 | `NotImplementedError` 含 M1-T5 引用 |
| M1-T4.1 | Atlas card 7 容器内拉起 Ray cluster | `ray status` 显示 head + 1 worker |
| M1-T4.2 | 真 RLinf factory 在集群内创建 EmbodiedRunner | `cluster_role="head"` + 4 worker groups attach |
| M1-T4.3 | `bench_results/w43_rlinf_e2e/` 同 seed 与 self_ppo 趋势一致（±5%） | outer_loop.mean_wm_total_loss |
| M1-T4.4 | Ray dashboard 无 failed worker | 5-phase orchestrator 跑通 |
| M1-T5.1 | verl 真路径在 M1-T5 P2 spec 实施 | 本 SPEC 仅预留 hook |

---

## 8. 影响与连锁效应

**正面**:
- orchestrator RL 框架三态（self_ppo / verl / rlinf）的 seam 收口，与 roadmap §M1 +
  C-T5 README 对齐——reader 可以 `--rl-backend rlinf --rlinf-runner-factory <...>` 直接跑
- W35-3 spike 成果（`RLinfRunner` / `RLinfRunnerConfig` / `RunnerFactory`）从 class-only
  升级到 CLI 可达，工程复用最大化
- W35 ADR §4 的"先 verl ship、RLinf deferred"决策不被破坏——`--rl-backend rlinf`
  默认走 mock，host 单测 0 依赖

**负面 / 接受成本**:
- `RLinfRunner` 真实路径需要 Ray cluster + worker groups——本 SPEC 在 Phase 1 仅
  实现 mock factory；真路径是 M1-T4 P2 的活
- W35.2 §5.2 版本冲突（RLinf v0.3 vs torch 2.9）意味着 Atlas 真路径须在
  `worldfoundry-npu-vllm:phase1` 容器内——host 上是 mock，不暴露冲突

**对路线图的影响**:
- W41 + W42 完成 = P1 全部就绪；P2（M1-T4 / M1-T5 / M2-T5 / C-T8）上 Atlas 即可端到端
- `--rl-backend verl` 是 hook 占位；M1-T5 P2 实施后 NotImplementedError 被替换

---

## 9. 开放问题

1. **`RLinfRunner.run()` 与 orchestrator `train_phase` / `policy_bootstrap_phase`
   的接口是否一致？** `RLinfRunner.run()` 返回 `dict[str, Any]`（line 77-82），
   而 orchestrator 需要更结构化的 schema（policy_loss / value_loss / entropy）。
   本 SPEC 默认 RLinfRunner 委托内部 runner.run() 后由 W35-3 actor/critic 收集指标；
   完整 schema 校准在 M1-T4 实施时定（不在 host-side Phase 1 闭环内）。
2. **mock factory 是否要 mock env / mock rollout？** M1-T1 阶段已说"4 个新单测；mock
   factory 不依赖 Ray 集群即可创建 runner"——意味着 mock factory 内部已经处理
   env / rollout 的占位，本 SPEC 不重复实现。
3. **verl 与 RLinf 是否能同进程 dual-import？** 不需要；orchestrator 一次跑一个 backend。
4. **Atlas 真 RLinf 版本号**：M1-T4 spec 内锁定（v0.3.0 vs v0.5.x）；本 SPEC 仅约束
   container 内跑，不锁版本。

---

## 10. 决策日志

| 日期 | 决策 | 状态 |
|---|---|---|
| 2026-08-15 | Phase 1 = `--rl-backend {self_ppo, verl, rlinf}` + `--rlinf-runner-factory` dispatch（A 方案） | 🟡 提议（本 SPEC） |
| 2026-08-15 | 沿用 M1-T1 / M1-T2 既有 `RLinfRunner` / `RLinfRunnerConfig` / `RunnerFactory`，不重写 | 🟡 提议 |
| 2026-08-15 | `RLinfRunner.run()` schema 校准留 M1-T4 P2 实施；host Phase 1 仅闭环 mock factory 路径 | 🟡 提议 |
| 2026-08-15 | verl 真路径 M1-T5 P2 spec 实施；本 SPEC 仅预留 hook（NotImplementedError） | 🟡 提议 |
| 2026-08-15 | W35 ADR §4 决策（verl > RLinf）不被破坏；本 SPEC 是 seam-level CLI 收口 | 🟡 提议 |

---

## 附录：信息来源

- Roadmap §9.2 P1 M1-T3 + §9.3 P2 M1-T4 / M1-T5：`docs/roadmap/2026-08-14-opensource-extensions-loop-todo.md`
- W35 RL 框架 ADR：`docs/superpowers/specs/2026-08-04-w35-rl-framework-adr.md` §4
- W35.2 版本矩阵 ADR：`docs/superpowers/specs/2026-08-06-w35-2-version-matrix-check.md` §5.2
- W35-3 RLinf spike plan：`docs/superpowers/specs/2026-08-04-w35-3-rlinf-custom-policy-spike-plan.md`
- M1-T1 / M1-T2 既有实现：`worldfoundry_common/adapters/rlinf.py:1-209`（`RLinfRunner`,
  `RLinfRunnerConfig`, `RunnerFactory` Protocol, `RLinfEnvWorker`）
- C-T5 README flag quickref：`README.md` §"Optional runtimes"（`--rl-backend` choices 列表）
- C-T6 cpu-rti-quickstart flag parity：`docs/dev/cpu-rti-quickstart.md` §"CLI flags" 表
  `--rl-backend` 行
- C-T1 版本矩阵：`requirements/rl.txt`（`verl==0.5.0`、`ray==2.56.1`）+ `pyproject.toml [project.optional-dependencies].rl`
- W36 NPU 同构契约：`docs/superpowers/specs/2026-08-06-w36-npu-worker-parity-contract.md` §4
- Actor / Critic 薄包装：`worldfoundry_train.model.hf_vjepa_shim.HfJEPAPredictorShim.forward(action=...)`
