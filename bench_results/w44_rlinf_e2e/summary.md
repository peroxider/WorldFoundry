# W44 — M1-T4 RLinf + Ray cluster e2e (partial closure)

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
| Real Ray cluster launch | `ray.init()` succeeds on Atlas | ❌ blocked on Atlas air-gap (`ray` not in cache) |
| Real RLinf dispatch | `RLinfRunner.run(RLinfRunnerConfig(worker_groups={...}))` constructs workers on card 7 | ❌ blocked on Atlas air-gap (`rlinf` not in cache) |
| W42 spec trend parity | `outer_loop.mean_wm_total_loss` trend matches `self_ppo` ±5% at same seed | ❌ blocked on real-rlinf construction |

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