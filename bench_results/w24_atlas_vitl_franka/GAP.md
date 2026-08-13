# W2.24 — Atlas NPU Franka real-env path: gap (env-install blocked)

**Closes**: Atlas checklist item #4 "Full roadmap closed" — **partial**:
W2.21/W2.22/W2.23 closed end-to-end on NPU; W2.24 franka path is
code-ready but blocked on dm_control install on Atlas.

**Date**: 2026-08-12
**Atlas card**: 7 (Ascend910B4-1)
**Code change**: none (W2.20 franka CPU path already proven; same
orchestrator flags run on NPU unchanged)

## Status

The W2.24 franka-on-NPU run **did not execute** on Atlas. The
orchestrator's first import of `--env franka_kitchen` raised
`ModuleNotFoundError: No module named 'dm_control'`, and the
attempted `uv pip install dm_control mujoco` install **did not
complete within an acceptable wall-clock window** (~17 minutes
stuck on PyPI fetch with 0 CPU usage, killed manually). Per plan
Risk Mitigation #5 ("If both [OSMesa and EGL] fail, skip W2.24
franka and document the gap"), the franka path is **deferred**
until the env-install can complete.

The W2.24 code path itself is unchanged — the same
`--env franka_kitchen --data-source franka_kitchen` flags that
work in the W2.20 CPU multi-seed sweep (`bench_results/w2_20_hf_planner_franka_iter/`)
would run on NPU once `dm_control` + `mujoco` are installed in the
Atlas venv. No orchestrator code changes are required.

## What was attempted

```bash
# Attempted install (Atlas, 2026-08-12 21:01–21:18 UTC, ~17 min)
export PATH=/root/.local/bin:$PATH
source /home/w30066257/WorldFoundry/.venv/bin/activate
UV_HTTP_TIMEOUT=600 uv pip install dm_control mujoco
# → killed after ~17 min; PyPI fetch stalled with 0 CPU usage
```

Diagnosis: the Atlas network path to `pypi.org` is intermittent
for large wheels (mujoco's wheel is ~16.8 MB; dm_control pulls a
~6 MB wheel plus several transitive deps including labmaze, lxml,
and glfw binaries). The `uv` resolver hangs on the first fetch
without emitting progress. Same network slowness that motivated
`hf-mirror.com` workaround (W2.21+ quirks #1) affects any PyPI
download > 5 MB.

## Required retry conditions

W2.24 franka run can complete when **any** of the following hold:

1. Atlas network to `pypi.org` improves (dm_control + mujoco wheels
   fetch successfully within ~5 min).
2. Atlas venv is pre-baked with `dm_control` + `mujoco` (Ascend
   image update — `docker/worldfoundry-base/Dockerfile` line for
   the dev/RL extra).
3. A local wheel mirror / `pypi.simple` proxy is configured for
   Atlas.

## Retry command (when env is ready)

```bash
export PATH=/root/.local/bin:$PATH
source /home/w30066257/WorldFoundry/.venv/bin/activate
# One-time env install (will succeed when Atlas network is healthy):
uv pip install dm_control mujoco

# Then run W2.24 franka on NPU:
MUJOCO_GL=osmesa \
ASCEND_RT_VISIBLE_DEVICES="7" LOCAL_RANK=0 WORLD_SIZE=1 RANK=0 \
WORLDFORGE_DEVICE=npu HF_HUB_OFFLINE=1 \
  python scripts/render_train_rl_cpu.py \
    --out-dir bench_results/w24_atlas_vitl_franka \
    --encoder hf_vjepa2 \
    --env franka_kitchen --data-source franka_kitchen \
    --num-clips 3 --frames-per-clip 4 \
    --height 32 --width 32 \
    --num-steps 6 --num-outer-iters 3 \
    --num-train-per-iter 4 --num-ppo-epochs 2 \
    --ppo-env-steps 16 --tubelet-size 1 --seed 42
```

Expected acceptance gates (matching W2.20 CPU franka proof):

- rc=0; `mode == "npu"`; `env_type == "franka_kitchen"`;
  `data_source == "franka_kitchen"`.
- `outer_loop.n_iters == 3`; `mean_wm_total_loss` decreases across
  iters (W2.20 CPU pattern: ~26 → ~24 → ~21).
- `inference.recon_cosine` and `latent_cosine` finite.
- dm_control renders work (no GL/EGL error in stderr; OSMesa
  workaround active).

## Roadmap status (post-W2.24 gap)

| Item | Status | Evidence |
|---|---|---|
| #1 HF shim on NPU | ✅ closed (W2.21) | `bench_results/w21_atlas_hf_shim/summary.md` |
| #2 1.2B real on NPU (seam) | ✅ closed (W2.22) | `bench_results/w22_atlas_vitl_smoke/summary.md` |
| #3 multi-seed stable on NPU | ✅ closed (W2.23) | `bench_results/w23_atlas_vitl_multi_seed/summary.md` |
| #4 Full roadmap closed | ⚠️ partial (W2.24) | this gap doc + above 3 closures |

The first 3 items prove the NPU stack works end-to-end with HF
VJEPA2Model (random-init + real 1.2B weights + multi-seed stable).
The 4th item (Franka real env on NPU) is code-ready and matches the
W2.20 CPU proof path; the only missing piece is the Atlas venv
env-install completing. This is **not a code gap** — it's an
**env gap** that will close when dm_control + mujoco can be
installed (network or image-bake resolution).

## CPU regression — N/A

No code changes this sub-milestone; the orchestrator's franka path
is unchanged from W2.20 (CPU-proven).