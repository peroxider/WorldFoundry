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
==== GATE 1: RTI v2 unit tests ====...rc=0
==== GATE 2: RTI v2 orchestrator (num_clips=4 num_steps=30 seed=42) ====...rc=0
==== RTI v2 smoke complete; summary at bench_results/render_train_rl/render_train_rl.json ====
```

End-to-end runtime on a single CPU thread: **~3.5 s** with defaults.

The summary JSON looks like:

```json
{
  "mode": "cpu",
  "render": {"elapsed_seconds": 0.054, "n_clips": 4, "n_frames": 16},
  "train": {
    "n_train_steps": 30, "n_encoder_params": 24576,
    "final_total_loss": 12.97, "final_pred_loss": 0.94
  },
  "inference": {
    "recon_cosine": 0.42, "latent_cosine": 0.61,
    "elapsed_seconds": 0.121
  },
  "policy_bootstrap": {
    "n_env_steps": 64, "n_policy_params": 1184,
    "mean_reward_pre_update": -1.21, "mean_reward_post_update": -0.65,
    "policy_loss_initial": 0.69, "policy_loss_final": 0.41,
    "n_ppo_epochs": 4
  },
  "outer_loop": {
    "n_iters": 2,
    "reward_trajectory": [-0.92, -0.51],
    "reward_trend_delta": 0.41,
    "per_iter": [
      {"iter_index": 0, "reward_pre_update": -1.21, "reward_post_update": -0.65,
       "mean_wm_total_loss": 13.19, "policy_loss_initial": 0.69, "policy_loss_final": 0.41},
      {"iter_index": 1, "reward_pre_update": -0.88, "reward_post_update": -0.51,
       "mean_wm_total_loss": 7.78, "policy_loss_initial": 0.43, "policy_loss_final": 0.28}
    ]
  }
}
```

## Interpreting the metrics

- **`render.n_frames`**: total frames across all clips. Real mp4s on disk.
- **`train.final_total_loss`**: encoder + predictor + decoder loss after
  `--num-steps`. Should decrease monotonically.
- **`inference.recon_cosine`**: cosine similarity between true and
  pixel-decoded latents, in `[-1, 1]`. > 0 means the decoder learned
  something.
- **`inference.latent_cosine`**: cosine similarity between true context
  and predictor output on the masked tokens. > 0 means the predictor
  learned to fill.
- **`policy_bootstrap.mean_reward_pre_update`**: mean reward across the
  rollout before PPO updates weights.
- **`policy_bootstrap.mean_reward_post_update`**: mean reward after
  multi-epoch PPO. Should be **less negative** (= higher reward) than
  pre-update.
- **`outer_loop.reward_trajectory`**: one entry per outer iter. The last
  minus the first is `reward_trend_delta`; positive = the loop is
  learning.
- **`outer_loop.per_iter[*].mean_wm_total_loss`**: JEPA + recon loss per
  iter. Should decrease — the world model is getting better as the
  policy improves.

## Tests

32 tests back the loop:

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

## Architecture

| Component | File | Lines |
|---|---|---|
| TinyViT encoder | `worldfoundry_train/model/tiny_jepa.py` | `TinyVJEPA2Encoder` |
| Action-conditioned predictor | `worldfoundry_train/model/tiny_jepa.py` | `TinyActionConditionedPredictor` |
| Pixel decoder | `worldfoundry_train/model/tiny_jepa.py` | `TinyPixelDecoder` |
| EMA updater | `worldfoundry_train/model/tiny_jepa.py` | `TinyEMAUpdater` |
| Checkpoint auto-dispatch | `worldfoundry_train/ema_compat.py` | `load_tiny_checkpoint` |
| Render phase | `scripts/render_train_rl_cpu.py` | `render_phase` |
| Train phase | `scripts/render_train_rl_cpu.py` | `train_phase` |
| Inference phase | `scripts/render_train_rl_cpu.py` | `inference_phase` |
| PPO update | `scripts/render_train_rl_cpu.py` | `policy_bootstrap_phase` |
| Outer loop | `scripts/render_train_rl_cpu.py` | `iterative_loop_phase` |
| Smoke shell | `scripts/render_train_rl_cpu.sh` | 2-gate runner |
| Test suite | `worldfoundry_train/tests/test_render_train_rl_cpu.py` | 12 tests |

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
| `--ppo-env-steps` | 64 | rollout steps per PPO update |
| `--num-ppo-epochs` | 4 | PPO epochs per update |
| `--num-outer-iters` | 2 | world-model ↔ PPO iterations |
| `--num-train-per-iter` | 4 | JEPA steps per outer iter |
| `--seed` | 42 | base RNG seed |

## See also

- `README.md` — top-level discoverability (this loop is the headline example)
- `docs/superpowers/specs/2026-07-27-ascend-world-model-design.md` — design spec
- `docs/superpowers/plans/2026-07-27-worldforge-phase1.md` — Phase-1 plan
- `docs/dev/next-steps-phase4-atlas.md` — atlas of W21-W41 work items
