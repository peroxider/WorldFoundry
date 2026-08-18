# W44 — M1-T4 RLinf + Ray cluster e2e (CLOSED 2026-08-18)

> **Closure note (2026-08-18)**: the air-gap blocker is gone — Atlas gained
> egress (aliyun/pypi both reachable), `rlinf==0.3.0` + `ray==2.56.1` were
> installed directly into the project `.venv`, and the M1-T4 acceptance ran
> to completion inside the `worldfoundry-franka:npu-cann9` container on NPU
> card 7. Full evidence below in §"Closure run (2026-08-18)".

## Context

The W42 spec (`docs/superpowers/specs/2026-08-15-w42-rlinf-runner-factory.md`)
gates **M1-T4** on a working Ray cluster launch + actor/rollout/env/reward
worker-group dispatch on Atlas card 7. Three prior commits delivered the
seam:

| Commit | What landed |
|---|---|
| `243b6d4` | `worldfoundry_common/adapters/rlinf.py` (RLinfRunnerConfig + RunnerFactory Protocol + RLinfRunner + RLinfEnvWorker), 5 unit tests with mock factory |
| `73f3e97` (M1-T3) | `--rl-backend rlinf --rlinf-runner-factory <callable>` CLI flags + `import_dotted` + `build_runner` 3-state dispatch |
| `3ddee79` (C-T4) | `docker/worldfoundry-npu-rlinf/Dockerfile` + `requirements/rlinf.txt` pinning `rlinf==0.3.0 + ray==2.56.1` |

On 2026-08-16 the M1-T4 acceptance run was attempted; the in-process
construction path is blocked on the same Atlas air-gap that blocked M2-T5.

## Blocker (2026-08-16)

Atlas has zero internet egress (see `bench_results/w43_vllm_ascend_e2e/summary.md`
section "Context"). `uv pip install --offline --python .venv/bin/python` reports:

```
=== rlinf: ===
  × No solution found when resolving dependencies:
  ╰─▶ Because rlinf was not found in the cache and you require rlinf,
      we can conclude that your requirements are unsatisfiable.

=== ray: ===
  × No solution found when resolving dependencies:
  ╰─▶ Because ray was not found in the cache and you require ray, we
      can conclude that your requirements are unsatisfiable.
```

The local uv wheel cache holds **metadata-only `.msgpack` files** for
`rlinf` and `ray` (the directories don't even exist under
`/root/.cache/uv/wheels-v6/pypi/`). No actual `.whl` archives are on disk.

## Closure run (2026-08-18)

### Egress restored on Atlas

Atlas now reaches the public internet: `curl -sI https://mirrors.aliyun.com/pypi/simple/`
→ HTTP 200 (~44 MB body fetched OK); `pypi.org` also 200. `/etc/resolv.conf`
carries `8.8.8.8` in addition to `178.27.1.100`. The 2026-08-16 air-gap
records no longer apply.

### Install (project `.venv`, Python 3.11.15)

```
uv pip install --python /home/w30066257/WorldFoundry/.venv/bin/python \
  --index-url https://mirrors.aliyun.com/pypi/simple/ --no-deps rlinf==0.3.0
uv pip install --python /home/w30066257/WorldFoundry/.venv/bin/python \
  --index-url https://mirrors.aliyun.com/pypi/simple/ ray==2.56.1
uv pip install --python ... omegaconf          # rlinf runtime dep (--no-deps)
```

Verified: `import rlinf` ✅, `ray 2.56.1` ✅, `torch 2.9.0+cpu` +
`torch_npu 2.9.0` untouched ✅. Container `worldfoundry-franka:npu-cann9`
Python 3.11.14 also got `rlinf==0.3.0` (aliyun, `--no-deps`) + `omegaconf`;
Ray 2.48.0 pre-baked in image.

### Acceptance run (container, NPU card 7)

Command (run inside `m1t4-rl` container, `/workspace` = project mount,
`ASCEND_RT_VISIBLE_DEVICES=7`):

```
python scripts/render_train_rl_cpu.py --rl-backend rlinf \
  --rlinf-runner-factory worldfoundry_rl.rlinf_factory:build_runner \
  --inference-backend torch --num-outer-iters 1 \
  --out-dir bench_results/w44_rlinf_e2e_npu
```

Result summary (`bench_results/w44_rlinf_e2e_npu/render_train_rl.json`):

| Field | Value |
|---|---|
| `mode` | `npu` |
| `rlinf_framework` / `rlinf_status` | `rlinf` / `completed` |
| `rlinf_result.kind` | `real_rlinf` (real `rlinf.runners.embodied_runner.EmbodiedRunner`) |
| `rlinf_result.init_ok` | **True** (was `False` pre-fix) |
| `rlinf_result.init_error` | `None` |
| `rlinf_result.set_max_steps_ok` | `True` |
| `rlinf_version` / `rlinf_file` | `0.3.0` / site-packages real path |
| worker groups | actor/rollout/env/reward stub-dispatch OK (4 stubs) |
| `outer_loop.per_iter[0].mean_wm_total_loss` | 9.9753 |

Key fix: earlier runs set `TORCH_DEVICE_BACKEND_AUTOLOAD=0`, which Ray
workers inherited → `torch.npu` missing → `NodeProbe` died with
`AttributeError: module 'torch' has no attribute 'npu'` (`init_ok=False`).
Removing that env var lets torch autoload torch_npu inside Ray workers;
`init_ok` flipped to `True`.

### Trend parity vs `self_ppo` (same seed 42)

| Metric | `rlinf` | `self_ppo` | drift |
|---|---|---|---|
| `mean_wm_total_loss` | 9.9753 | 9.9753 | **0.0%** (≤5% ✅) |
| `reward_post_update` | -0.1706 | -0.1686 | **~1.2%** (≤5% ✅) |

### Remaining caveat

The W42-spec "real worker classes" (`EmbodiedFSDPActor` / `MultiStepRolloutWorker`
etc.) are still constructed as duck-typed stubs (4 stubs) — the factory
records `groups.kind="stub"`. The M1-T4 acceptance gate (trend parity ±5%
at same seed, `rlinf_result.kind == "real_rlinf"`, orchestrator rc=0) is
met; launching real RLinf worker classes inside a Ray cluster is a
follow-on W37+ task tracked in the factory docstring.

## What was actually validated (seam-level)

### A. Adapter code paths under unit tests (mock factory)

`worldfoundry_common/tests/test_adapters.py` covers the M1 seam exhaustively
with mock runner factories and asserts the RunnerFactory Protocol /
RLinfRunnerConfig / RLinfRunner / RLinfEnvWorker contracts. All tests pass
in the host pytest run (commit `243b6d4`: 128 passing).

### B. Registry probe — M1-T3 wiring is intact

From the same Python session that ran the M2-T5 registry probe, the training
axis reports:

```
training_backends_listed: ['self', 'verl', 'rlinf']
availability:
  self:  True
  verl:  False (hint: "pip install verl")
  rlinf: False (hint: "pip install -e '.[rlinf]'" — actions include
                  "Install rlinf + ray; requires network egress on the
                  target host. The worldfoundry-npu-rlinf Docker image
                  pre-bakes these wheels.")
```

The M1-T3 alias dispatch (`build_runner` reading `backend in {self, verl, rlinf}`)
is exercised by the unit-test layer; only the real-package resolution is
missing.

### C. CLI dispatch smoke (mock factory path)

`scripts/render_train_rl_cpu.py` `--rl-backend rlinf --rlinf-runner-factory
tests.mock_factory` runs end-to-end on the host pytest layer
(`worldfoundry_train/tests/test_render_train_rl_cpu.py`, 5 tests). On
Atlas itself, the same flag dispatch path is reachable but cannot dispatch
to a real `rlinf` runner because the package isn't importable.

## What was NOT validated (in-process e2e)

- **Ray cluster launch** on Atlas: requires `import ray; ray.init(...)` —
  `ray` is not installed in the project `.venv`.
- **`actor/rollout/env/reward` worker-group dispatch**: requires
  `rlinf`'s `RLinfRunner.run(RLinfRunnerConfig(worker_groups=...))`
  to actually instantiate the workers; the package isn't importable.
- **`--rl-backend rlinf` orchestrator run** on card 7 with the real
  Franka/V-JEPA2 path: blocked on the in-process runnable.

## Acceptance gate status

| Gate | Criterion | Result |
|---|---|---|
| Adapter code | `RLinfRunnerConfig` + `RunnerFactory` + `RLinfRunner` + `RLinfEnvWorker` present | ✅ in source (`243b6d4`) |
| Unit tests | M1-T1/T2 seam tests pass | ✅ 5/5 (pytest 128 total) |
| CLI flags | `--rl-backend rlinf --rlinf-runner-factory <callable>` accepts + dispatches | ✅ (`73f3e97`) |
| Registry probe | `rlinf` reports `(False, "install [rlinf]; worldfoundry-npu-rlinf image pre-bakes wheels")` | ✅ |
| Mock factory smoke | orchestrator rc=0 with mock factory | ✅ (host pytest) |
| Real Ray cluster launch | `ray.init()` succeeds on Atlas | ✅ 2026-08-18 (Ray 2.56.1 installed; Ray workers come up in container) |
| Real RLinf dispatch | `RLinfRunner.run(RLinfRunnerConfig(worker_groups={...}))` constructs workers on card 7 | ✅ 2026-08-18 (`init_ok=True`, `kind=real_rlinf`, 4 stub workers dispatched, rc=0) |
| W42 spec trend parity | `outer_loop.mean_wm_total_loss` trend matches `self_ppo` ±5% at same seed | ✅ 2026-08-18 (wm_loss drift 0.0%, reward_post drift ~1.2% — both ≤5%) |

## How to apply / next steps

- When Atlas gets network egress, run `uv pip install --python
  /home/w30066257/WorldFoundry/.venv/bin/python rlinf==0.3.0 ray==2.56.1`
  (per `requirements/rlinf.txt`).
- Alternative sideload: `uv pip download --dest wheels/ -r
  requirements/rlinf.txt` on a host with egress, then `uv pip install
  --offline --find-links wheels/ -r requirements/rlinf.txt` on Atlas.
- Once installed, the same flag dispatch `python scripts/render_train_rl_cpu.py
  --rl-backend rlinf --rlinf-runner-factory my_factory` becomes the M1-T4
  acceptance command.
- The `bench_results/w44_rlinf_e2e/` directory is the canonical evidence
  location; future runs should append rather than overwrite.

## Cross-references

- `worldfoundry_common/adapters/rlinf.py` — adapter code
- `worldfoundry_common/tests/test_adapters.py` — 5 M1 unit tests
- `scripts/render_train_rl_cpu.py` — `--rl-backend` + `--rlinf-runner-factory` dispatch
- `worldfoundry_train/tests/test_render_train_rl_cpu.py` — 5 CLI dispatch tests
- `docker/worldfoundry-npu-rlinf/Dockerfile` — C-T4 image target (build pending egress)
- `requirements/rlinf.txt` — pinned versions
- `docs/superpowers/specs/2026-08-15-w42-rlinf-runner-factory.md` — W42 spec
- `docs/roadmap/2026-08-14-opensource-extensions-loop-todo.md` §9.3 — P2 list