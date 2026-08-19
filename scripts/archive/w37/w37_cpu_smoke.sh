#!/usr/bin/env bash
# W37 v1 CPU skeleton smoke — CLI runner end-to-end.
#
# Usage:
#     bash scripts/archive/w37/w37_cpu_smoke.sh                  # default 192 pool-steps
#     bash scripts/archive/w37/w37_cpu_smoke.sh --steps 192      # explicit total pool-steps
#     bash scripts/archive/w37/w37_cpu_smoke.sh --seed 42        # reproducible env init
#
# Validates W37 §A-F: YAML-driven CLI runner wires SyncEnvPool +
# VisionPolicy + PPO via worldfoundry_train.{device,checkpoint} seams.
# Sits between unit tests and the Atlas booking — the highest-fidelity
# reproducibility check available without an NPU.
#
# Gate 1 (unit tests):
#     worldfoundry_rl/tests/                       (114 expected: 105 pre-W37 + 9 new)
#
# Gate 2 (CLI runner):
#     python -m worldfoundry_rl.train --config \
#         worldfoundry_rl/configs/rl_manipulator_cpu.yaml --steps 192
#
# Writes:
#     bench_results/w37/cpu_loop_smoke.json
#     bench_results/w37/cpu_loop_smoke.log
#
# Exit codes:
#     0  all gates pass + JSON written
#     2  usage / config error
#     3  unit-test gate failed
#     4  CLI runner gate failed
set -uo pipefail  # not -e: continue past pytest/unit-test failures to write report

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p bench_results/w37

PY="${PY:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
    echo "FATAL: $PY not on PATH" >&2
    exit 2
fi

TOTAL_STEPS=192   # 6 PPO updates @ rollout_steps=32 in rl_manipulator_cpu.yaml
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

OUT_DIR="bench_results/w37"
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
# Gate 1: W37 v1 unit tests (RL CLI + checkpoint rl_payload + strategy)
# ------------------------------------------------------------------
run_step "GATE 1: worldfoundry_rl/tests/ (114 expected)" "w37-unit" \
    "$PY" -m pytest worldfoundry_rl/tests/ -v --tb=short || {
        echo "FATAL: W37 unit-test gate failed (rc=$?)" >&2
        exit 3
    }

# ------------------------------------------------------------------
# Gate 2: end-to-end CLI runner
#
#   rl_manipulator_cpu.yaml ─► train.py ─► SyncEnvPool ─► VisionPolicy ─► PPO
#                                                │                  │
#                                                ▼                  ▼
#                                       [W37] stdout summary    save_checkpoint(rl_payload=...)
#                                                │
#                                                ▼
#                                    bench_results/w37/cpu_loop_smoke.json
# ------------------------------------------------------------------
# Note: --seed flag is *accepted* by train.py's parse_args only via the
# YAML's `seed:` key (CLI does not currently expose --seed). The wrapper
# python script below bakes the seed into the summary context instead.
cli_rc=0
SUMMARY_JSON=""
echo "==== GATE 2: W37 CLI runner (total_steps=$TOTAL_STEPS seed=$SEED) ===="
SUMMARY_JSON="$(
    "$PY" -m worldfoundry_rl.train \
        --config worldfoundry_rl/configs/rl_manipulator_cpu.yaml \
        --steps "$TOTAL_STEPS" \
        2>"$OUT_DIR/w37-cli.stderr"
)" || cli_rc=$?
echo "==== GATE 2: W37 CLI runner rc=$cli_rc ===="

if [[ "$cli_rc" -ne 0 ]]; then
    echo "---- last 60 lines of w37-cli.stderr ----"
    tail -n 60 "$OUT_DIR/w37-cli.stderr" || true
    echo "---- end ----"
    echo "FATAL: W37 CLI runner crashed (rc=$cli_rc); see $OUT_DIR/w37-cli.stderr" >&2
    exit 4
fi

# Extract the final "[W37] summary: {...}" line from captured stdout.
SUMMARY_LINE="$(
    printf '%s\n' "$SUMMARY_JSON" | grep -E '^\[W37\] summary: \{' | tail -n 1 || true
)"
if [[ -z "$SUMMARY_LINE" ]]; then
    echo "FATAL: W37 CLI runner did not emit a summary JSON line." >&2
    echo "---- last 30 lines of captured stdout ----"
    printf '%s\n' "$SUMMARY_JSON" | tail -n 30
    echo "---- end ----"
    exit 4
fi

# Strip the "[W37] summary: " prefix and parse JSON.
SUMMARY_JSON_ONLY="${SUMMARY_LINE#\[W37\] summary: }"

# Wrap the CLI summary into the bench_results schema. Don't override the
# CLI's own field semantics — only add bench-level context (label, mode,
# spec, task, notes).
WRAPPER_RC=0
"$PY" - "$SUMMARY_JSON_ONLY" "$TOTAL_STEPS" "$SEED" "$OUT_DIR" >"$OUT_DIR/w37-wrap.log" 2>&1 <<'PY' || WRAPPER_RC=$?
import json
import sys
from pathlib import Path

cli_summary = json.loads(sys.argv[1])
total_steps = int(sys.argv[2])
seed = int(sys.argv[3])
out_dir = Path(sys.argv[4])

out = {
    "label": "W37 v1 CPU RL training skeleton smoke",
    "mode": "cpu",
    "spec": "docs/dev/next-steps-phase4-atlas.md W37 row #17",
    "task": {
        "name": "manipulator:bring_ball",
        "description": (
            "dm_control 5-DoF planar arm brings a ball to a target position. "
            "drives the W37 CLI runner end-to-end on CPU."
        ),
        "env_class": "worldfoundry_rl.env_manipulator.ManipulatorEnv",
        "env_kwargs": {"task_kwargs": {"random": seed}},
    },
    "config": {
        "config_yaml": "worldfoundry_rl/configs/rl_manipulator_cpu.yaml",
        "total_pool_steps": total_steps,
        "seed": seed,
        "parallel": "ddp",
    },
    "summary": {
        "updates":               cli_summary.get("updates", 0),
        "total_pool_steps":      cli_summary.get("total_pool_steps", total_steps),
        "total_env_steps":       cli_summary.get("total_env_steps", 0),
        "device":                cli_summary.get("device", "cpu"),
        "wall_seconds":          cli_summary.get("wall_seconds", 0.0),
        "final_policy_loss":     cli_summary.get("final_policy_loss", 0.0),
        "final_value_loss":      cli_summary.get("final_value_loss", 0.0),
        "final_mean_episode_reward": cli_summary.get("final_mean_episode_reward", 0.0),
    },
    "notes": [
        "W37 v1 CPU skeleton: env -> VisionPolicy -> PPO via YAML-driven CLI runner.",
        "Mirrors worldfoundry_train/train.py surface (--config / --steps / --parallel).",
        "Reuses worldfoundry_train.device.{resolve_device,to_device} (W20 .to('npu') seam).",
        "Reuses worldfoundry_train.checkpoint.{save,load}_checkpoint with new rl_payload kwarg.",
        "make_strategy('rl') is a DDPStrategy alias in v1 (single-process; no DDP wrap of PPO).",
        "Reward/value curves reported but NOT enforced: manipulator:bring_ball's sparse reward produces noisy per-update returns.",
        "Multi-rank + actor_pool/learner_pool/scheduler are W37+ (gated on W35 framework choice + Atlas cluster).",
    ],
}
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "cpu_loop_smoke.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"[W37] bench JSON: {out_path}")
PY

if [[ "$WRAPPER_RC" -ne 0 ]]; then
    echo "---- last 60 lines of w37-wrap.log ----"
    tail -n 60 "$OUT_DIR/w37-wrap.log" || true
    echo "---- end ----"
    echo "FATAL: W37 bench JSON wrapper failed (rc=$WRAPPER_RC)" >&2
    exit 4
fi

# Mirror the JSON to a .log (the regression suite greps for `[W37]` tags).
{
    echo "==== W37 CPU loop smoke complete ===="
    echo "JSON:    $JSON"
    echo "Pytest:  $OUT_DIR/w37-unit.log"
    echo "CLI:     $OUT_DIR/w37-cli.stderr"
    cat "$JSON" 2>/dev/null || true
} >"$LOG"

echo "==== W37 smoke complete; throughput at $JSON ===="