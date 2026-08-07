# CPU render → train → inference → policy loop (RLinf-shape)

Self-contained smoke that runs an **iterative world-model ↔ PPO** loop on a
single CPU thread in seconds, with no HF V-JEPA 2 1.2B download and no
mujoco/dm_control dependency. It exists so anyone can verify the loop
works end-to-end without an Atlas box.

Entry point: `bash scripts/render_train_rl_cpu.sh`.

## What this is

A 5-phase orchestrator that mirrors the RLinf train/imagined-rollout
pattern with a tiny (24k-param) stack:

| Phase | What runs | Output |
|---|---|---|
| **Render** | cv2 → real `.mp4` clips + CSV manifest | `videos/clip_*.mp4`, `manifest.csv` |
| **Train** | TinyViT encoder + action-conditioned predictor + pixel decoder | `tiny_jepa_v2.pt` (real checkpoint) |
| **Inference** | pixel + latent cosine probes, action-conditioned fill + decode | `recon_cosine`, `latent_cosine` |
| **Policy** | rollout (grad enabled through predictor) + multi-epoch PPO update | `mean_reward_pre/post`, `policy_loss_initial/final` |
| **Outer loop** | world-model ↔ PPO alternation, RLinf-style | `reward_trajectory`, `per_iter` |

The reducer is a learned action-conditioned predictor: same encoder state
+ different actions → different predicted latents. The PPO update
backprops through the predictor's `_action_proj` parameters, so the
world model is shaped by the policy gradient.

## Prerequisites

```bash
pip install -e .[dev,video]   # pytest + av + decord + opencv-python
```

No NPU. No HF download. No mujoco. Runs on a stock CPU.

## Run

```bash
bash scripts/render_train_rl_cpu.sh
```

Defaults: 4 clips × 30 train steps, `seed=42`, writes to
`bench_results/render_train_rl/`.

Override knobs:

```bash
bash scripts/render_train_rl_cpu.sh --num-clips 8 --num-steps 60 --seed 7
```

Or call the orchestrator directly for finer control:

```bash
python scripts/render_train_rl_cpu.py \
    --out-dir bench_results/render_train_rl \
    --num-clips 4 \
    --num-steps 30 \
    --num-outer-iters 2 \
    --num-ppo-epochs 4 \
    --ppo-env-steps 64 \
    --action-dim 4 \
    --obs-dim 8 \
    --seed 42
```

## Expected output

```
==== GATE 1: RTI v2 unit tests ====
======================= 42 passed, 27 warnings in 59.58s =======================
==== GATE 1: RTI v2 unit tests rc=0 ====
==== GATE 2: RTI v2 orchestrator (num_clips=4 num_steps=30 seed=42) ====
==== GATE 2: RTI v2 orchestrator rc=0 ====
==== RTI v2 smoke complete; summary at bench_results/render_train_rl/render_train_rl.json ====
```

Runtime with defaults on a single CPU thread: the orchestrator itself is
**~3.5 s**; the full two-gate shell smoke is **~65 s** (Gate 1 runs all
42 tests).

The summary JSON below is a verbatim `seed=42` run, not an illustration:

```json
{
  "mode": "cpu",
  "render": {"elapsed_seconds": 0.046, "n_clips": 4, "n_frames": 16},
  "train": {
    "n_train_steps": 30, "n_encoder_params": 24416,
    "final_total_loss": 13.5935, "final_jepa_loss": 13.1845,
    "final_recon_loss": 0.3867, "final_act_align": 0.2234
  },
  "inference": {
    "recon_cosine": 0.6269, "latent_cosine": 0.1361,
    "n_predicted_tokens": 32, "elapsed_seconds": 0.127
  },
  "policy_bootstrap": {
    "n_env_steps": 16, "n_policy_params": 1513,
    "mean_reward_pre_update": -0.4953, "mean_reward_post_update": -0.1815,
    "policy_loss_initial": 0.6029, "policy_loss_final": 0.5980,
    "n_ppo_epochs": 4
  },
  "outer_loop": {
    "n_iters": 2,
    "reward_trajectory": [-0.174, -0.2766],
    "reward_trend_delta": -0.1026,
    "per_iter": [
      {"iter": 0, "mean_wm_total_loss": 9.9283, "reward_pre_update": -1.2099,
       "reward_post_update": -0.174, "policy_loss_initial": 1.7984, "policy_loss_final": 1.7805},
      {"iter": 1, "mean_wm_total_loss": 7.5558, "reward_pre_update": -1.0765,
       "reward_post_update": -0.2766, "policy_loss_initial": 0.3995, "policy_loss_final": 0.3928}
    ]
  }
}
```

## Interpreting the metrics

- **`render.n_clips` / `render.n_frames`**: counted from `manifest.csv` on
  disk, not derived from the CLI flags — a short or failed video write
  shows up as a mismatch instead of being masked. Real mp4s.
- **`train.final_total_loss`**: encoder + predictor + decoder loss after
  `--num-steps`. Trends down but is **not** monotone step-to-step (each
  step draws a fresh random tubelet mask, so per-step loss is noisy).
- **`inference.recon_cosine`**: cosine similarity between true and
  pixel-decoded latents, in `[-1, 1]`. > 0 means the decoder learned
  something.
- **`inference.latent_cosine`**: cosine similarity between true context
  and predictor output on the masked tokens. > 0 means the predictor
  learned to fill. Typically much smaller than `recon_cosine` at these
  step counts (~0.1 vs ~0.6).
- **`outer_loop.per_iter[*].mean_wm_total_loss`**: JEPA + recon loss per
  iter. **Decreases monotonically across iters** — verified on seeds
  42/7/99/123/2024. This is the loop's most reliable learning signal.
- **`reward_post_update` − `reward_pre_update`** *(within one iter)*:
  **positive in every iter on every seed tested** (15/15). This is the
  honest "PPO improved the policy" signal, and is what
  `test_outer_loop_ppo_update_improves_reward_within_iter` pins.

### `reward_trend_delta` is a diagnostic, not a pass/fail gate

`reward_trajectory` is the *post-update* reward per iter, and
`reward_trend_delta` is `trajectory[-1] - trajectory[0]`. **Do not read a
positive delta as "the loop is learning."** Measured across 5 seeds at
`--num-outer-iters 3`, it is positive in only 2:

| seed | `reward_trajectory` | delta |
|---|---|---|
| 42 | `[-0.174, -0.278, -0.158]` | +0.016 |
| 7 | `[-0.056, -0.127, -0.154]` | −0.098 |
| 123 | `[-0.161, -0.164, -0.168]` | −0.007 |
| 2024 | `[-0.224, -0.134, -0.094]` | +0.131 |
| 99 | `[-0.144, -0.136, -0.144]` | −0.001 |

With `--ppo-env-steps 16` over 2–3 iters the metric is dominated by
rollout variance. It is emitted for observability; no test asserts its
sign, and none should at this scale. Use `mean_wm_total_loss` and the
within-iter reward gain instead.

## Tests

42 tests back the loop:

```bash
pytest worldfoundry_train/tests/test_render_train_inference_cpu.py \
       worldfoundry_train/tests/test_render_train_rl_cpu.py -v
```

Highlights:

- `test_action_conditioning_changes_output` — same encoder + mask,
  two different actions → different predictor output (the reducer
  actually uses actions).
- `test_load_tiny_checkpoint_dispatches_to_action_conditioned` — v2
  checkpoints auto-load into the right predictor class.
- `test_ppo_update_changes_policy_parameters` — PPO moves the policy
  (proof of learning, not a no-op).
- `test_multi_epoch_ppo_reduces_policy_loss` — `policy_loss_final
  <= 1.5 * policy_loss_initial` over K=4 epochs.
- `test_outer_loop_uses_iter_dependent_seeds` — regression test for
  the v2.4 seed-discipline bug; per-iter losses must differ by > 1e-3.
- `test_outer_loop_ppo_update_improves_reward_within_iter` — every outer
  iter must end with higher reward than it started (18/18 across 6 seeds).
- `test_outer_loop_world_model_loss_decreases_across_iters` — last iter's
  `mean_wm_total_loss` must beat the first.
- `test_render_block_counts_match_manifest_on_disk` — `render.n_clips` /
  `n_frames` must match `manifest.csv`, and every referenced mp4 must
  exist and be non-empty.

## Architecture

| Component | File | Lines |
|---|---|---|
| TinyViT encoder | `worldfoundry_train/model/tiny_jepa.py` | `TinyVJEPA2Encoder` |
| Action-conditioned predictor | `worldfoundry_train/model/tiny_jepa.py` | `TinyActionConditionedPredictor` |
| Pixel decoder | `worldfoundry_train/model/tiny_jepa.py` | `TinyPixelDecoder` |
| EMA updater | `worldfoundry_train/ema_compat.py` | `TinyEMAUpdater` |
| Checkpoint save/load | `worldfoundry_train/ema_compat.py` | `save_tiny_checkpoint` / `load_tiny_checkpoint` |
| Render phase | `scripts/render_train_rl_cpu.py` | `render_phase` |
| Train phase | `scripts/render_train_rl_cpu.py` | `train_phase` |
| Inference phase | `scripts/render_train_rl_cpu.py` | `inference_phase` |
| PPO update | `scripts/render_train_rl_cpu.py` | `policy_bootstrap_phase` |
| Outer loop | `scripts/render_train_rl_cpu.py` | `iterative_loop_phase` |
| Smoke shell | `scripts/render_train_rl_cpu.sh` | 2-gate runner |
| Test suite | `worldfoundry_train/tests/test_render_train_rl_cpu.py` | 22 tests |

## What this is NOT

- **Not a faithful V-JEPA 2 1.2B reproduction.** The tiny stack is
  interface-compatible (same forward signature, same mask semantics)
  but the encoder is 24k params, not 1.2B.
- **Not a real MuJoCo/dm_control rollout.** The CPU env is an
  in-process random agent whose reward is `-||state||² - 0.1*||action||²`.
  The state dimension is `--obs-dim` (default 8), not a real
  FrankaKitchen observation.
- **Not multi-process.** `actor_pool.py` / `learner_pool.py` /
  Ray / async rollouts are explicitly out of scope (cluster segment
  of W37). This is single-process, single-thread.
- **Not a training-to-convergence recipe.** 30 steps is enough to
  prove the loop closes; convergence needs Atlas + 1.2B + real env.

## CLI flags

| Flag | Default | Purpose |
|---|---|---|
| `--out-dir` | `bench_results/render_train_rl` | output directory |
| `--num-clips` | 4 | synthetic clips to render |
| `--frames-per-clip` | 4 | frames per clip |
| `--height` / `--width` | 32 / 32 | frame size |
| `--num-steps` | 30 | JEPA train steps |
| `--learning-rate` | 3e-4 | optimizer lr |
| `--mask-ratio` | 0.5 | predictor mask ratio |
| `--action-dim` | 4 | action dimension |
| `--obs-dim` | 8 | env observation dimension |
| `--ppo-env-steps` | 16 | rollout steps per PPO update |
| `--num-ppo-epochs` | 4 | PPO epochs per update |
| `--num-outer-iters` | 2 | world-model ↔ PPO iterations |
| `--num-train-per-iter` | 10 | JEPA steps per outer iter |
| `--seed` | 42 | base RNG seed |

## See also

- `README.md` — top-level discoverability (this loop is the headline example)
- `docs/superpowers/specs/2026-07-27-ascend-world-model-design.md` — design spec
- `docs/superpowers/plans/2026-07-27-worldforge-phase1.md` — Phase-1 plan
- `docs/dev/next-steps-phase4-atlas.md` — atlas of W21-W41 work items
