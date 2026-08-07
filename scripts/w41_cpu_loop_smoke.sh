#!/usr/bin/env bash
# W41 Phase-5 v1 CPU end-to-end loop smoke.
#
# Usage:
#     bash scripts/w41_cpu_loop_smoke.sh                  # default 192 pool-steps
#     bash scripts/w41_cpu_loop_smoke.sh --steps 192      # explicit total pool-steps
#     bash scripts/w41_cpu_loop_smoke.sh --seed 42        # reproducible env init
#
# Validates G7-extension Phase-5 v1: ManipulatorEnv (W41 deliverable 1) +
# VisionPolicy (W41 deliverable 2) + the existing SyncEnvPool + PPO trainer
# close an end-to-end on-policy RL loop on CPU. Sits between unit tests
# (which prove individual surfaces) and the Atlas booking (which would
# require W20 .to("npu") + Atlas hardware) — i.e. the highest-fidelity
# reproducibility check available without an NPU.
#
# Gate (the unit tests must pass before we run anything heavy):
#     worldfoundry_rl/tests/test_env_manipulator.py
#     worldfoundry_rl/tests/test_vision_policy.py
#
# Then a 192-step (6 update × 32 rollout_steps) inline loop:
#     SyncEnvPool([ManipulatorEnv]) -> VisionPolicy.from_state_dim -> PPO.train
#
# Writes:
#     bench_results/w41/cpu_loop_smoke.json
#     bench_results/w41/cpu_loop_smoke.log
#
# Exit codes:
#     0  all gates pass + JSON written
#     2  usage / config error
#     3  unit-test gate failed
#     4  e2e loop failed
set -uo pipefail  # not -e: continue past pytest/unit-test failures to write report

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p bench_results/w41

PY="${PY:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
    echo "FATAL: $PY not on PATH" >&2
    exit 2
fi

TOTAL_STEPS=192   # 6 PPO updates @ rollout_steps=32
SEED=42
while [[ $# -gt 0 ]]; do
    case "$1" in
        --steps) TOTAL_STEPS="$2"; shift 2 ;;
        --seed)  SEED="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,17p' "$0"
            exit 0
            ;;
        *)
            echo "WARN: unknown arg $1 (ignored)" >&2
            shift
            ;;
    esac
done

if ! [[ "$TOTAL_STEPS" =~ ^[0-9]+$ ]] || [[ "$TOTAL_STEPS" -lt 32 ]]; then
    echo "FATAL: --steps must be int >= 32, got $TOTAL_STEPS" >&2
    exit 2
fi

OUT_DIR="bench_results/w41"
LOG="$OUT_DIR/cpu_loop_smoke.log"
JSON="$OUT_DIR/cpu_loop_smoke.json"

run_step() {
    local title="$1"; shift
    local tag="$1"; shift
    local log="$OUT_DIR/${tag}.log"
    echo "==== $title ===="
    if "$@" 2>"$log"; then
        local rc=0
    else
        local rc=$?
    fi
    echo "==== $title rc=$rc ===="
    if [[ "$rc" -ne 0 ]]; then
        echo "---- last 60 lines of $log ----"
        tail -n 60 "$log" || true
        echo "---- end ----"
    fi
    return $rc
}

# ------------------------------------------------------------------
# Gate 1: Phase-5 v1 unit tests (env_manipulator + vision_policy)
# ------------------------------------------------------------------
run_step "GATE 1: env_manipulator + vision_policy unit tests" "w41-unit" \
    "$PY" -m pytest \
    worldfoundry_rl/tests/test_env_manipulator.py \
    worldfoundry_rl/tests/test_vision_policy.py \
    -v --tb=short || {
        echo "FATAL: Phase-5 v1 unit-test gate failed (rc=$?)" >&2
        exit 3
    }

# ------------------------------------------------------------------
# Gate 2: end-to-end CPU loop
#
#   ManipulatorEnv x num_envs=4  ──┐
#                                  ├─► SyncEnvPool → rollout (32 steps)
#   VisionPolicy.from_state_dim ──┘              │
#                                                ▼
#                                          PPO.update (clip + value + entropy)
#                                                │
#                                                ▼
#                                    bench_results/w41/cpu_loop_smoke.json
# ------------------------------------------------------------------
# Note: the failure handler MUST be on a line after `PY` (the heredoc
# terminator) — bash otherwise consumes it as heredoc text and reports
# "unexpected end of file". `|| python_rc=$?` on the same line as `<<'PY'`
# is OK because it's parsed before the heredoc body starts.
python_rc=0
echo "==== GATE 2: CPU end-to-end loop (total_steps=$TOTAL_STEPS seed=$SEED) ===="
"$PY" - "$TOTAL_STEPS" "$SEED" "$OUT_DIR" <<'PY' 2>"$OUT_DIR/w41-loop.log" || python_rc=$?
import json
import sys
import time
from pathlib import Path

total_steps = int(sys.argv[1])
seed = int(sys.argv[2])
out_dir = Path(sys.argv[3])

import numpy as np
import torch

from worldfoundry_rl.env_manipulator import ManipulatorEnv
from worldfoundry_rl.env_pool import SyncEnvPool
from worldfoundry_rl.ppo import PPO, PPOConfig
from worldfoundry_rl.vision_policy import VisionPolicy

t0 = time.perf_counter()

NUM_ENVS = 4
ROLLOUT_STEPS = 32
EPOCHS_PER_UPDATE = 4
TOTAL_POOL_STEPS = total_steps  # 6 updates @ rollout=32 default
UPDATES = -(-TOTAL_POOL_STEPS // ROLLOUT_STEPS)

pool = SyncEnvPool(
    [lambda: ManipulatorEnv(task_kwargs={"random": seed}) for _ in range(NUM_ENVS)]
)
try:
    seed_obs = pool.reset(seed=seed)
    sample = seed_obs[0]
    obs_dim = sum(int(np.asarray(v).ravel().size) for v in sample.values())
    action_dim = int(pool.single_action_space.shape[0])

    policy = VisionPolicy.from_state_dim(
        obs_dim=obs_dim, action_dim=action_dim, hidden_dim=64,
    )
    n_params = sum(p.numel() for p in policy.parameters())

    cfg = PPOConfig(
        rollout_steps=ROLLOUT_STEPS,
        epochs_per_update=EPOCHS_PER_UPDATE,
        minibatch_size=min(64, ROLLOUT_STEPS * NUM_ENVS),
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        learning_rate=3e-4,
        normalize_advantages=True,
    )

    ppo = PPO(policy, cfg)
    history = ppo.train(pool, total_steps=TOTAL_POOL_STEPS)
finally:
    pool.close()

t_train = time.perf_counter() - t0

if len(history) != UPDATES:
    raise RuntimeError(
        f"PPO.train ran {len(history)} updates, expected {UPDATES}"
    )


def _mean(xs):
    return float(np.mean(xs)) if xs else 0.0


first_two_pl = _mean([h["policy_loss"] for h in history[:2]])
last_two_pl = _mean([h["policy_loss"] for h in history[-2:]])
first_two_vl = _mean([h["value_loss"] for h in history[:2]])
last_two_vl = _mean([h["value_loss"] for h in history[-2:]])
first_two_ent = _mean([h["entropy"] for h in history[:2]])
last_two_ent = _mean([h["entropy"] for h in history[-2:]])
first_two_mer = _mean([h["mean_episode_reward"] for h in history[:2]])
last_two_mer = _mean([h["mean_episode_reward"] for h in history[-2:]])

if len(history) == 1:
    first_two_pl = last_two_pl = history[0]["policy_loss"]
    first_two_vl = last_two_vl = history[0]["value_loss"]
    first_two_ent = last_two_ent = history[0]["entropy"]
    first_two_mer = last_two_mer = history[0]["mean_episode_reward"]

reward_improved = last_two_mer > first_two_mer
loss_improved = last_two_pl < first_two_pl

out = {
    "label": "W41 Phase-5 v1 CPU end-to-end loop smoke",
    "mode": "cpu",
    "spec": "docs/dev/next-steps-phase4-atlas.md W41 row #1",
    "task": {
        "name": "manipulator:bring_ball",
        "description": (
            "dm_control 5-DoF planar arm brings a ball to a target position. "
            "27-dim flattened obs (arm_pos/vel/touch + hand/object/target "
            "XYZ), 5-DoF continuous action."
        ),
        "env_class": "worldfoundry_rl.env_manipulator.ManipulatorEnv",
        "env_kwargs": {"random": seed},
        "expected_random_episode_return": "approximately 0.0 (sparse reward)",
    },
    "config": {
        "num_envs": NUM_ENVS,
        "rollout_steps": ROLLOUT_STEPS,
        "minibatch_size": cfg.minibatch_size,
        "epochs_per_update": EPOCHS_PER_UPDATE,
        "learning_rate": cfg.learning_rate,
        "entropy_coef": cfg.entropy_coef,
        "gamma": cfg.gamma,
        "gae_lambda": cfg.gae_lambda,
        "clip_ratio": cfg.clip_ratio,
        "value_loss_coef": cfg.value_loss_coef,
        "max_grad_norm": cfg.max_grad_norm,
        "normalize_advantages": cfg.normalize_advantages,
        "total_pool_steps": TOTAL_POOL_STEPS,
        "policy_params": int(n_params),
    },
    "summary": {
        "updates": len(history),
        "first_two_policy_loss_mean": round(first_two_pl, 4),
        "last_two_policy_loss_mean":  round(last_two_pl, 4),
        "first_two_value_loss_mean":  round(first_two_vl, 4),
        "last_two_value_loss_mean":   round(last_two_vl, 4),
        "first_two_entropy_mean":     round(first_two_ent, 4),
        "last_two_entropy_mean":      round(last_two_ent, 4),
        "first_two_mean_episode_reward": round(first_two_mer, 4),
        "last_two_mean_episode_reward":  round(last_two_mer, 4),
        "reward_improved": bool(reward_improved),
        "loss_decreased":  bool(loss_improved),
        "wall_seconds": round(t_train, 3),
    },
    "per_update": [
        {
            "u": i,
            "policy_loss": round(h["policy_loss"], 4),
            "value_loss":  round(h["value_loss"], 4),
            "entropy":     round(h["entropy"], 4),
            "approx_kl":   round(h["approx_kl"], 4),
            "mean_episode_reward": round(h["mean_episode_reward"], 4),
        }
        for i, h in enumerate(history)
    ],
    "notes": [
        "W41 Phase-5 v1 closes the loop on CPU: Env -> SyncEnvPool -> VisionPolicy -> PPO.",
        "VisionPolicy is the Phase-5 v1 pluggable-encoder surface; .to('npu') is the future W41-on-Atlas story via W20.",
        (
            f"reward_improved={bool(reward_improved)} loss_decreased={bool(loss_improved)} "
            "are reported but NOT enforced: manipulator:bring_ball's sparse reward produces noisy "
            "per-update episode returns. The gate is 'loop runs without crashing and writes a JSON'."
        ),
        "Atlas NPU path is W41-on-Atlas — gated on W20 (.to('npu')) and Atlas single-card booking (W21+).",
    ],
}

out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "cpu_loop_smoke.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"[W41] CPU loop JSON: {out_path}")
print(
    f"[W41] updates={len(history)} "
    f"policy_loss {first_two_pl:+.3f}->{last_two_pl:+.3f} "
    f"value_loss {first_two_vl:.3f}->{last_two_vl:.3f} "
    f"mean_episode_reward {first_two_mer:+.3f}->{last_two_mer:+.3f} "
    f"wall={t_train:.2f}s"
)
PY
echo "==== GATE 2: CPU end-to-end loop rc=$python_rc ===="
if [[ "$python_rc" -ne 0 ]]; then
    echo "---- last 60 lines of w41-loop.log ----"
    tail -n 60 "$OUT_DIR/w41-loop.log" || true
    echo "---- end ----"
    echo "FATAL: CPU e2e loop crashed (rc=$python_rc); see $OUT_DIR/w41-loop.log" >&2
    exit 4
fi

# Mirror the JSON to a .log (the regression suite greps for `[W41]` tags).
{
    echo "==== W41 CPU loop smoke complete ===="
    echo "JSON:    $JSON"
    echo "Pytest:  $OUT_DIR/w41-unit.log"
    echo "Loop:    $OUT_DIR/w41-loop.log"
    cat "$JSON" 2>/dev/null || true
} >"$LOG"

echo "==== W41 smoke complete; throughput at $JSON ===="
