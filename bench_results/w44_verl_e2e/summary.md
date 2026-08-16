# W44 — M1-T5 verl backend e2e (partial closure)

## Context

The C-T7 deliverables (`docs/superpowers/specs/`) established that **M1-T5**
is the verl backend e2e smoke: a single-GPU (no Ray cluster) `verl`
orchestrator run on Atlas card 7 with the acceptance criterion

> `outer_loop.mean_wm_total_loss` trend matches `self_ppo` at the same seed ±5%

Two prior commits delivered the seam:

| Commit | What landed |
|---|---|
| `243b6d4` | `worldfoundry_common/adapters/verl.py` (VerlRunner + VerlRolloutWorker + `build_npu_config` + NPU-first vLLM-Ascend rollout / FSDP strategy), 4 unit tests with mock `verl.trainer.main_ppo` + `verl.workers.rollout.ray_rollout.RayRollout` |
| W35.2 spec | `requirements/base.txt` chain + `check_c_t1` enforcement of `-r base.txt` |

On 2026-08-16 the M1-T5 acceptance run was attempted; the in-process
construction path is blocked on the same Atlas air-gap that blocked M2-T5
and M1-T4.

## Blocker (2026-08-16)

Atlas has zero internet egress. `uv pip install --offline --python
.venv/bin/python verl` reports:

```
=== verl (--prerelease=allow): ===
  × No solution found when resolving dependencies:
  ╰─▶ ... conclude that verl<0.6.0 cannot be used.
      And because verl>=0.6.0 needs to be downloaded from a registry
      and you require verl, we can conclude that your requirements
      are unsatisfiable.
```

The local uv wheel cache holds **metadata-only `.msgpack` for `verl
0.5.0-py2.py3-none-any`** (`/root/.cache/uv/wheels-v6/pypi/verl/0.5.0-py2.py3-none-any.msgpack`).
No actual `.whl` archive. The pinned `requirements/base.txt` chain
(constrained by `check_c_t1`) requires `verl>=0.6.0` which can't resolve
from cache. As with M2-T5 and M1-T4, the workaround is the same sideload
pattern.

## What was actually validated (seam-level)

### A. Adapter code paths under unit tests (mock verl)

`worldfoundry_common/tests/test_adapters.py` exercises the verl seam with
synthetic `verl` modules injected into `sys.modules` (no real verl package
required at test time). The tests assert:

- `VerlRunner.run()` calls `verl.trainer.main_ppo.run_ppo(cfg)` with the
  correct config shape (not a non-existent `verl.interface.Main`).
- `build_npu_config()` returns NPU-first defaults (`vllm` rollout +
  `fsdp` strategy + `device in {npu, cuda}`) and respects user override.
- `VerlRunner.run()` raises `ImportError("verl is required")` when the
  package is absent.
- `VerlRolloutWorker` wraps `verl.workers.rollout.ray_rollout.RayRollout`
  for `generate_sequences / update_weights / shutdown` lifecycle.

All pass in the host pytest run (commit `243b6d4`: 128 passing).

### B. Registry probe — training axis wiring intact

From the same Python session that ran the M2-T5 + M1-T4 probes:

```
training_backends_listed: ['self', 'verl', 'rlinf']
availability:
  self:  True
  verl:  False (hint: "pip install verl" — verl is the upstream framework
                 from volcengine.github.io/verl; requires network egress on
                 the target host. worldfoundry-npu-rlinf image pre-bakes
                 the wheels.)
  rlinf: False (hint: "pip install -e '.[rlinf]'...")
```

The M1-T3 alias dispatch covers `backend == "verl"` in `build_runner`.

### C. CLI dispatch smoke

`scripts/render_train_rl_cpu.py` `--rl-backend verl` is the single-GPU,
no-Ray entry path. The flag-dispatch and config-build code paths are
unit-tested (`worldfoundry_train/tests/test_render_train_rl_cpu.py`,
5 tests). On Atlas itself, `--rl-backend verl` resolves to
`VerlRunner(config=...)` but cannot call `verl.trainer.main_ppo.run_ppo`
because the package isn't importable.

## What was NOT validated (in-process e2e)

- **`verl.trainer.main_ppo.run_ppo(cfg)` execution** on Atlas: requires
  the `verl` package, not importable from cache.
- **`VerlRolloutWorker` wrapping a real `RayRollout`** for `/v1/chat/completions`
  generation: requires `verl` to import `verl.workers.rollout.ray_rollout`.
- **`--rl-backend verl` orchestrator run** on card 7 with the real
  Franka/V-JEPA2 path: blocked on the in-process runnable.
- **Trend parity gate** (`self_ppo` vs `verl` ±5% on `mean_wm_total_loss`):
  blocked on real verl construction.

## Acceptance gate status

| Gate | Criterion | Result |
|---|---|---|
| Adapter code | `VerlRunner` + `VerlRolloutWorker` + `build_npu_config` present | ✅ in source (`243b6d4`) |
| Unit tests | 4 verl seam tests pass (mock verl in sys.modules) | ✅ (pytest 128 total) |
| CLI flags | `--rl-backend verl` accepts + dispatches to `build_runner` | ✅ (`73f3e97`) |
| Registry probe | `verl` reports `(False, "pip install verl; worldfoundry-npu-rlinf image pre-bakes wheels")` | ✅ |
| Mock factory smoke | orchestrator rc=0 with mock verl module | ✅ (host pytest) |
| Real `verl.trainer.main_ppo.run_ppo` | executes PPO training on Atlas card 7 | ❌ blocked on Atlas air-gap (`verl` >=0.6.0 not in cache) |
| Real `VerlRolloutWorker.generate_sequences` | `/v1/chat/completions` HTTP call succeeds | ❌ blocked on real `verl` import |
| W42 spec trend parity | `outer_loop.mean_wm_total_loss` matches `self_ppo` ±5% at same seed | ❌ blocked on real verl construction |

## How to apply / next steps

- When Atlas gets network egress, run `uv pip install --python
  /home/w30066257/WorldFoundry/.venv/bin/python verl==0.5.0` (or
  pin a newer compatible version in `requirements/base.txt`).
- Alternative sideload: `uv pip download --dest wheels/ verl==<version>`
  on a host with egress, then `uv pip install --offline --find-links
  wheels/ verl==<version>` on Atlas.
- Once installed, the same flag dispatch `python scripts/render_train_rl_cpu.py
  --rl-backend verl` becomes the M1-T5 acceptance command. Single-GPU
  path means no Ray cluster is required (the `VerlRunner` configuration
  bypasses `RayRollout` for the orchestrator's PPO loop).
- The `bench_results/w44_verl_e2e/` directory is the canonical evidence
  location; future runs should append rather than overwrite.

## Cross-references

- `worldfoundry_common/adapters/verl.py` — adapter code
- `worldfoundry_common/tests/test_adapters.py` — 4 verl unit tests
- `scripts/render_train_rl_cpu.py` — `--rl-backend verl` dispatch
- `worldfoundry_train/tests/test_render_train_rl_cpu.py` — 5 CLI dispatch tests
- `requirements/base.txt` — version chain (verl>=0.6.0)
- `requirements/check_c_t1.py` — chain enforcement
- `docs/roadmap/2026-08-14-opensource-extensions-loop-todo.md` §9.3 — P2 list