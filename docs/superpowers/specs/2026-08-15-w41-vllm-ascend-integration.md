# SPEC-W41：vLLM-Ascend 推理后端注册与 Atlas 端到端路径

> 状态：提议（Proposed）
> 日期：2026-08-15
> 关联：`docs/roadmap/2026-08-14-opensource-extensions-loop-todo.md` §9.1 P0 / §9.2 P1
> （M2-T4 / M2-T5 / M2-T6），`docs/superpowers/specs/2026-08-06-w36-npu-worker-parity-contract.md`
> §3.4（NPU 同构契约 fast API），`docs/superpowers/specs/2026-08-06-w40-frame-gen-adr.md` §5.2（B-v2
> 推迟条件），`docker/worldfoundry-npu-vllm/Dockerfile`（C-T3 产物）。
> 决策范围：M2 主线（推理引擎可替换）的 Phase-1 收口 + Atlas card 7 端到端
> 验证 + vllm-omni 公开后的衔接位。
> 影响：`scripts/render_train_rl_cpu.py --inference-backend {torch,vllm_ascend}`
> 在 Atlas 容器内可走真实 `VJEPA2NPUWorker` 而非 torch fallback；W43+
> M2-T5 e2e smoke 与 vllm-omni joint eval (M3-T7+M2-T6) 的依赖前置。

---

## 0. TL;DR

将 `VJEPA2NPUWorker`（位于 `worldfoundry_inf/vllm_ascend_worker.py`）在 `worldfoundry_common/backends/inference.py`
的 `BackendRegistry` 中升级为正式注册的 `vllm_ascend`（与既有的 `vllm` 同义；命名收敛以对齐 C-T5 README
flag quickref）；`available()` 探测逻辑强制要求 `torch_npu + vllm + vllm_ascend` 三个 import 全部
成功；在 Atlas card 7 容器内跑 `bench_results/w43_vllm_ascend_e2e/`，对比 `torch` backend 的
recon / latent cosine 漂移 ≤0.01。vllm-omni 公开 wheel 后（M3-T7）走同一条路径。

---

## 1. 背景与问题

WorldFoundry 的 M2 主线是 **推理引擎可插拔**（roadmap §M2）：用一条注册表 seam 替换 `VJEPA2Worker` /
`VJEPA2NPUWorker` / SGLang / MindIE-SD / vllm-omni 等多种 runtime，让 orchestrator 在不改业务
逻辑的前提下选择。当前 `worldfoundry_common/backends/inference.py` 已经把 5 个 backend 都注册了
（`torch` / `vllm` / `sglang` / `mindie_sd` / `vllm_omni`），但 roadpm §9.2 P1 M2-T4 把
"`vllm_ascend` 后端注册"单独列为 P1 任务——意味着收敛命名 + 探测强化 + 容器内真接口 ≠ 当前
host-side 注册语义。

### 关键事实

- **现有注册命名是 `vllm`**：`worldfoundry_common/backends/inference.py:151-159` 注册名
  是 `"vllm"`，`build` 通过 lazy-import `worldfoundry_inf.vllm_ascend_worker.VJEPA2NPUWorker`
  构造。orchestrator `--inference-backend` 在 `scripts/render_train_rl_cpu.py:1537` 接受
  `choices=("torch", "vllm", "sglang")`，所以现在 host 调 `--inference-backend vllm` 已经
  会触发 `VJEPA2NPUWorker` 的 lazy import（仅在 import 失败时降级）。
- **`available()` 探测只看自家 worker import，不验证 vllm + vllm-ascend wheel**：
  `worldfoundry_common/backends/inference.py:39-45` 的 `_vllm_available()` 只
  `import worldfoundry_inf.vllm_ascend_worker`，不验证运行时 `torch_npu + vllm + vllm_ascend`
  是否齐备。M2-T4 验收要求 `registry.available("vllm_ascend")` 在 Atlas 容器内返回 `(True, ...)`，
  需要把探测强化成三包齐 import。
- **`vllm_ascend` 与 `vllm` 同名收敛**：roadmap §9.2 P1 M2-T4 的验收文案是
  `registry.available("vllm_ascend")`，但既有的 `vllm` 已经是同一含义。两条路：
  (a) 改名 `vllm → vllm_ascend`（破坏既有 CLI / README / orchestrator flag choices）；
  (b) 追加 `vllm_ascend` 别名（与 `vllm` 同 factory / 同 available / 同 hint），保留
  `vllm` 向后兼容。本 SPEC 选 (b)，理由见 §5。
- **C-T3 产物**：`docker/worldfoundry-npu-vllm/Dockerfile` 已经把
  `requirements/inf.txt`（`vllm==0.18.0 + vllm-ascend==0.18.0`）打进 child image
  （commit 5dac705），从而 Atlas 端 `docker build -t worldfoundry-npu-vllm:phase1`
  在容器内 `python -c "import vllm"` 已经能成功（与 C-T3 验收一致）。
- **vllm-omni 公开 wheel 仍未发布**：`_vllm_omni_available` 永远返回 `False`
  （line 116-118）；`_build_vllm_omni` 抛 `NotImplementedError`。W44+ 才能填真实现。
- **W36 NPU 同构契约约束**：`VJEPA2NPUWorker` 必须与 `VJEPA2Worker`（host-cpu fallback）
  在 `encode / predict / plan` 三个 surface 上 100% 对镜（`docs/superpowers/specs/2026-08-06-w36-npu-worker-parity-contract.md` §3.4）。
  M2-T1..T3 已经把 `predict_masked_batch / predict_temporal / predict_action_conditioned`
  的 CPU fallback 落到 `worldfoundry_inf/vllm_ascend_worker.py`（26.9 KB），seam 已闭合。

### 与 roadmap §M2 风险的对照

- **vLLM-Ascend `predict_*_batch` 完整真路径不闭环**：roadmap §6 风险表已记
  "当前 `VJEPA2NPUWorker` 端到端不闭环，仅在单条路径可用"。本 SPEC 收口的不是
  "完整 batch 路径"（那是 M2-T1..T3 的 W41 spike 范围，已经 CPU-fallback 落地），
  而是"registered backend + 强化探测 + Atlas e2e"。两者分工。
- **MindIE-SD 链路占位不阻塞本 SPEC**：W40 frame-gen ADR §5.2 把 MindIE-SD 推迟到
  B-v2；W41 中只对 `mindie_sd` / `vllm_omni` 名称做记录，不改实现。

---

## 2. 硬性约束

| # | 约束 | 来源 |
|---|---|---|
| C1 | `vllm_ascend` backend 必须可在 Atlas card 7 上调用 `VJEPA2NPUWorker.encode/predict/plan` | W36 §3.4 同构契约；`VJEPA2NPUWorker` seam 收口 |
| C2 | `registry.available("vllm_ascend")` 在 Atlas 容器内返回 `(True, ...)`，host 返回 `(False, "pip install -e '.[inf]'")` | roadmap §9.2 P1 M2-T4 验收 |
| C3 | `VJEPA2NPUWorker` 与 `VJEPA2Worker` 行为偏差 ≤0.01 cosine（recon / latent） | M2-T5 验收；收敛 M2 §NPU 同构契约 |
| C4 | `--inference-backend vllm_ascend` 走真 NPU worker，host 走 torch fallback | README §"Optional runtimes"；C-T5 落地的 flag 名 |
| C5 | `vllm` 旧名称保持向后兼容（既有 docs / orchestrator choices 不破） | 不能破坏既有 wheel 用户；本 SPEC 是 additive |
| C6 | `requirements/inf.txt` 锁定 `vllm==0.18.0 + vllm-ascend==0.18.0` | C-T1 版本矩阵 |
| C7 | Docker image `worldfoundry-npu-vllm:phase1`（C-T3 产物）能跑通 e2e | C-T3 验收硬约束 |

---

## 3. 决策驱动因素

| 驱动因素 | 权重 | 说明 |
|---|---|---|
| D1 命名一致性（flag / backend / README） | ★★ | `--inference-backend` 的 choices 与 registry 名对齐；reader 看名字即可懂含义 |
| D2 探测可靠性（区别"in-tree 模块" vs "运行时 wheel 齐备"） | ★★★ | host 不能误报 True；Atlas 必须报 True |
| D3 向后兼容（既有 `--inference-backend vllm` / 注册名 `vllm`） | ★★★ | 不破坏既有 orchestrator / test / doc |
| D4 W36 NPU 同构契约遵守 | ★★★ | `VJEPA2NPUWorker` 与 `VJEPA2Worker` 形同；不能引入新 surface |
| D5 Atlas card 7 真负载可达 | ★★ | M2-T5 e2e 是 P2 验收，本 SPEC 把路径打通，不是上负载 |
| D6 vllm-omni 衔接（M2-T6 / M3-T7） | ★ | 长期；本 SPEC 只预留入口，不实施 |

---

## 4. 方案对比

| 维度 | A. 改名 `vllm → vllm_ascend` | B. 追加 `vllm_ascend` 别名（保留 `vllm`） | C. 只更新探测，命名不动 |
|------|------------------------------|--------------------------------------------|----------------------------|
| D1 命名一致性 | 🟢 一次性拉齐 | 🟢 拉齐且不破坏 | 🔴 README 仍说 `vllm`，roadmap 说 `vllm_ascend`，drift |
| D2 探测可靠性 | 🟢 同时改 | 🟢 同时改 | 🟢 同时改 |
| D3 向后兼容 | 🔴 破既有 orchestrator choices + C-T5 README + cpu_tests | 🟢 additive；两个名字都通 | 🟢 additive |
| D4 NPU 同构契约 | 🟢 不动 surface | 🟢 不动 surface | 🟢 不动 surface |
| D5 Atlas 真负载 | 🟢 可达 | 🟢 可达 | 🟢 可达 |
| D6 vllm-omni 衔接 | 🟢 接口已对齐 | 🟢 接口已对齐 | 🟡 README / roadmap 仍 drift |
| 改动量 | 🔴 orchestrator choices + README + 2 specs 都得改 | 🟢 50 LOC 增量 | 🟢 30 LOC，但 README drift |
| **本 SPEC 推荐** | 不推荐 | **推荐** 🟢 | 不推荐（drift） |

---

## 5. 决策

**B 方案（追加 `vllm_ascend` 别名，保留 `vllm`）**：在 `worldfoundry_common/backends/inference.py`
中 `registry.register("vllm_ascend", Backend(..., build=_build_vllm, available=_vllm_ascend_available))`，
把 `_vllm_available` 重构为同时探测 `torch_npu + vllm + vllm_ascend + worldfoundry_inf.vllm_ascend_worker`
的 `_vllm_ascend_available`，既有的 `_vllm_available` / `_build_vllm` / `register("vllm", ...)`
全部保留成同义薄壳。`hint` 字段统一指向 `pip install -e '.[inf]'`。

### 5.1 决策理由（按权重分层）

1. **D3 向后兼容决定**（★★★）：`scripts/render_train_rl_cpu.py:1537` 的
   `choices=("torch", "vllm", "sglang")`、C-T5 README §"Optional runtimes" 表的
   `--inference-backend` 行、M1-T5 / M2-T5 等后续验收都用 `vllm`。一旦改名，上述 doc +
   orchestrator choices + unit test 全部要改；M2-T4 的本意是"注册成 Atlas 通路"，不是
   "rename `vllm`"。追加别名是 additive，不破任何既有行为。
2. **D2 探测可靠性决定**（★★★）：现有 `_vllm_available` 只 `import worldfoundry_inf.vllm_ascend_worker`
   ——host 上的 in-tree stub 通过，但 host 没装 vllm-ascend wheel；Atlas 装了。强化探测把
   `torch_npu + vllm + vllm_ascend` 三包齐作为 gate，与 `sglang_available()` 同形。
3. **D1 命名一致性收口**（★★）：roadmap / 本 SPEC / README 都用 `vllm_ascend`；
   既有 `vllm` 保留为同义薄壳并在 hint 里 cross-reference /
   "alias of `vllm_ascend` (canonical name; `vllm` retained for backward compatibility)"。
4. **D4 NPU 同构契约遵守**（★★★）：别名只是 registry 注册，不引入新 surface；
   `encode / predict / plan` 三个方法保持同一 `VJEPA2NPUWorker` 实例。
5. **D5 Atlas 真负载可达**（★★）：C-T3 image `worldfoundry-npu-vllm:phase1`
   已经把 wheel 全装好（commit 5dac705）；别名 + 强化探测立刻让
   `registry.available("vllm_ascend") == True` 在该 image 内生效。
6. **D6 vllm-omni 衔接**（★）：本 SPEC 不实施（M3-T7 / M2-T6 的活）；但
   `vllm_omni` 的占位 stub + hint（`worldfoundry_common/backends/inference.py:180-189`）
   已经在位，W44+ public wheel 一发布即生效。

### 5.2 与 B-v2 / W44 的因果绑定

- B1: vllm + vllm-ascend 真编译 / 注册链路在 Atlas card 7 上跑通（关闭 M2-T4 + M2-T5）
- B2: B-v2 数据算力预算批准（不阻塞 M2-T4/T5）
- B3: vllm-omni 公开 wheel 发布（M2-T6 + M3-T7）

### 5.3 vllm 旧名的去留

`vllm` 注册不删，对外仅以 `hint` 标注 "alias of `vllm_ascend`"。理由：
- host-only unit test 用 `backend="vllm"` 跑回归（W2.18+ 既有契约）
- orchestrator `--inference-backend vllm` 在既有 `--inference-backend choices=("torch", "vllm", "sglang")`
  范围内继续工作（M2-T5 验收不动）
- README / cpu-rti-quickstart 不需要二次变更

---

## 6. 实现路径

```
W41 (本 SPEC) — host 可达
  ↓
Phase 1 — registry 强化（host unit test 闭环）
  ├─ worldfoundry_common/backends/inference.py:
  │   - 新增 _vllm_ascend_available() 三包齐探测
  │   - 既有 _vllm_available 复用 _vllm_ascend_available
  │   - 新增 register("vllm_ascend", Backend(...))
  │   - 既 register("vllm", ...) 的 hint 改写为 "alias of vllm_ascend"
  ├─ worldfoundry_common/tests/test_backends.py:
  │   - test_vllm_ascend_available_in_atlas / _on_host
  │   - test_vllm_aliases_to_vllm_ascend（同 build / 同 available）
  │   - test_registry_lists_vllm_ascend（与 `vllm` 同列）
  └─ 验收: .venv/bin/python -m pytest worldfoundry_common/tests -q
           expect: n_passed >= 现有 + 3 new

Phase 2 — Atlas card 7 端到端（M2-T5，P2 验收）
  ├─ rsync W41 提交到 /home/w30066257/WorldFoundry/
  ├─ docker build -t worldfoundry-npu-vllm:phase1 -f docker/worldfoundry-npu-vllm/Dockerfile .
  ├─ docker run --rm ... worldfoundry-npu-vllm:phase1 \
  │     python -c "from worldfoundry_common.backends import inference; print(inference.registry.available('vllm_ascend'))"
  │   expect: (True, ...)
  ├─ bench_results/w43_vllm_ascend_e2e/ 跑两次：
  │     scripts/render_train_rl_cpu.py --inference-backend torch   --seed 42 ...
  │     scripts/render_train_rl_cpu.py --inference-backend vllm_ascend --seed 42 ...
  │   cosine drift on recon / latent ≤ 0.01
  └─ 验收: summary.mode == "npu", bench_results/w43_vllm_ascend_e2e/summary.md
```

---

## 7. 验证门（M2-T4 / M2-T5 / M2-T6 各自清单）

| 步骤 | 内容 | 通过标准 |
|---|---|---|
| M2-T4.1 | 既有 `vllm` 注册保留 | `registry.list()` 仍包含 `vllm` |
| M2-T4.2 | 追加 `vllm_ascend` 别名 | `registry.list()` 含 `vllm_ascend` |
| M2-T4.3 | `vllm` 与 `vllm_ascend` 同 factory / 同 available / 同 hint | `is` 相等、`available()` 调用返回值相等 |
| M2-T4.4 | `_vllm_ascend_available` host 返回 False | `.venv/bin/python -m pytest worldfoundry_common/tests/test_backends.py` 通过 |
| M2-T4.5 | 强化探测：缺 `torch_npu` 或 `vllm_ascend` wheel 即 False | 单元 test 覆盖 |
| M2-T5.1 | Atlas 卡 7 容器 `registry.available("vllm_ascend")` 返回 `(True, ...)` | 在 `worldfoundry-npu-vllm:phase1` 内验证 |
| M2-T5.2 | `bench_results/w43_vllm_ascend_e2e/` 5-phase orchestrator rc=0 | 与 torch backend 同 seed 趋势一致（±5%）|
| M2-T5.3 | recon cosine drift ≤ 0.01 | `summary.inference.recon_cosine` 与 torch backend 同 seed 差 ≤ 0.01 |
| M2-T5.4 | latent cosine drift ≤ 0.01 | 同 |
| M2-T5.5 | `summary.mode == "npu"` | orchestrator output |
| M2-T6.1 | vllm-omni public wheel 状态追踪 | `WorldFoundry/bench_results/_vllm_omni_pending.json` 记录 issue |
| M2-T6.2 | 公开后一次性把 `vllm_omni` build + available 落实（不在本 SPEC 范围） | M3-T7 联合跑 |

---

## 8. 影响与连锁效应

**正面**:
- registry / CLI flag / README 命名收敛到一个 canonical name `vllm_ascend`，roadmap / spec
  / orchestrator 三处不再 drift
- `vllm_ascend` Atlas 实路径打通的 seam-level 校验，比 host 上 stub 真一档
- C-T3 docker image（`worldfoundry-npu-vllm:phase1`）的可信度由此跨进 e2e 阶段
- 未来 vllm-omni 公开 wheel 一发布，按 `vllm_omni_available = True` 即生效（hint / 探测已经在位）

**负面 / 接受成本**:
- host 上 `vllm_ascend_available == False` 不能用——这是故意的；用户用 `vllm` 走 torch fallback
- `vllm` / `vllm_ascend` 双名共存期间需在 hint 里 cross-reference，避免 reader 误以为有两个不同 backend

**对路线图的影响**:
- W41 + W42 完成 = P1 全部就绪；P2（M2-T5 / M1-T4 / M1-T5 / C-T8）上 Atlas 即可端到端跑
- M3-T7 + M2-T6 在 vllm-omni 公开后由独立 SPEC（W44+）收口，与本 SPEC 不耦合

---

## 9. 开放问题

1. **是否需要在 `worldfoundry_inf.vllm_ascend_worker.VJEPA2NPUWorker` 加 NPU-device pin 探测？
   目前 `_vllm_ascend_available` 只看 import；Atlas 容器内 `torch.npu.device_count()` ≥ 1 才算
   "Atlas-ready"，但本 SPEC 不锁，由 M2-T5 e2e 在容器内实际跑出来替代。
2. **`vllm` 旧名的去留期**：若是 6 个月内无外部消费者，可发一发 deprecation warning；
   现在保留双注册。
3. **vllm-omni 公开 wheel 落地后是否复用本 SPEC 的别名结构？** 待 M3-T7 spec 决定，本 SPEC
   不预设。
4. **HW probe 是否走 `WORLDFOUNDRY_ATLAS` env var 还是 `/proc/driver/npu` 文件探测？**
   C-T9 conftest 已经定 `/proc/driver/npu` 的 Python-side；M2-T4 沿用同约定，
   不引入新探测点。

---

## 10. 决策日志

| 日期 | 决策 | 状态 |
|---|---|---|
| 2026-08-15 | Phase 1 = 追加 `vllm_ascend` 别名（保留 `vllm`） | 🟡 提议（本 SPEC） |
| 2026-08-15 | 强化 `available()` 探测三包齐（`torch_npu + vllm + vllm_ascend`） | 🟡 提议 |
| 2026-08-15 | M2-T6 / M3-T7 不在本 SPEC（W44+ 公开 wheel 后独立 SPEC） | 🟡 提议 |

---

## 附录：信息来源

- Roadmap §9.2 P1 M2-T4 / M2-T5 / §9.4 M2-T6：`docs/roadmap/2026-08-14-opensource-extensions-loop-todo.md`
- C-T3 产物（C-T3 acceptance gate 跑的 image）：`docker/worldfoundry-npu-vllm/Dockerfile`（commit 5dac705）
- 既有 registry：`worldfoundry_common/backends/inference.py:142-189`
- 既有 NPU worker seam：`worldfoundry_inf/vllm_ascend_worker.py`（26.9 KB，commit history 中涵盖 M2-T1/T2/T3）
- W36 同构契约：`docs/superpowers/specs/2026-08-06-w36-npu-worker-parity-contract.md` §3.4
- C-T5 README flag quickref：`README.md` §"Optional runtimes"
- C-T1 版本矩阵：`requirements/inf.txt`（`vllm==0.18.0`、`vllm-ascend==0.18.0`）+ `pyproject.toml [project.optional-dependencies] inf`
- W35 RL 框架 ADR：`docs/superpowers/specs/2026-08-04-w35-rl-framework-adr.md`
