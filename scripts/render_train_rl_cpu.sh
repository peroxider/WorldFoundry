#!/usr/bin/env bash
# RTI v2 CPU render->train->inference->policy smoke (RLinf-shape).
#
# Usage:
#     bash scripts/render_train_rl_cpu.sh                  # defaults
#     bash scripts/render_train_rl_cpu.sh --num-clips 2
#     bash scripts/render_train_rl_cpu.sh --num-steps 10
#
# Writes:
#     bench_results/render_train_rl/{videos,manifest.csv,tiny_jepa_v2.pt,render_train_rl.json}
#
# Exit codes:
#     0  all gates pass + summary JSON written
#     2  usage / config error
#     3  unit-test gate failed
#     4  orchestrator gate failed
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p bench_results/render_train_rl

PY="${PY:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
    echo "FATAL: $PY not on PATH" >&2
    exit 2
fi

NUM_CLIPS=4
NUM_STEPS=30
SEED=42
OUT_DIR="bench_results/render_train_rl"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-clips) NUM_CLIPS="$2"; shift 2 ;;
        --num-steps) NUM_STEPS="$2"; shift 2 ;;
        --seed)      SEED="$2"; shift 2 ;;
        --out-dir)   OUT_DIR="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,10p' "$0"
            exit 0
            ;;
        *)
            echo "WARN: unknown arg $1 (ignored)" >&2
            shift
            ;;
    esac
done

mkdir -p "$OUT_DIR"

run_step() {
    local title="$1"; shift
    local tag="$1"; shift
    local log="$OUT_DIR/${tag}.log"
    echo "==== $title ===="
    if "$@" 2>"$log"; then local rc=0; else local rc=$?; fi
    echo "==== $title rc=$rc ===="
    if [[ "$rc" -ne 0 ]]; then
        echo "---- last 60 lines of $log ----"
        tail -n 60 "$log" || true
        echo "---- end ----"
    fi
    return $rc
}

# Gate 1: unit tests (tiny model + ema_compat + RTI v2 predictor).
run_step "GATE 1: RTI v2 unit tests" "rti2-unit" \
    "$PY" -m pytest worldfoundry_train/tests/test_render_train_inference_cpu.py -q --tb=short || {
        echo "FATAL: RTI v2 unit-test gate failed (rc=$?)" >&2
        exit 3
    }

# Gate 2: orchestrator.
cli_rc=0
echo "==== GATE 2: RTI v2 orchestrator (num_clips=$NUM_CLIPS num_steps=$NUM_STEPS seed=$SEED) ===="
"$PY" scripts/render_train_rl_cpu.py \
    --out-dir "$OUT_DIR" \
    --num-clips "$NUM_CLIPS" \
    --num-steps "$NUM_STEPS" \
    --seed "$SEED" \
    2>"$OUT_DIR/rti2-orch.stderr" || cli_rc=$?
echo "==== GATE 2: RTI v2 orchestrator rc=$cli_rc ===="

if [[ "$cli_rc" -ne 0 ]]; then
    echo "---- last 60 lines of rti2-orch.stderr ----"
    tail -n 60 "$OUT_DIR/rti2-orch.stderr" || true
    echo "---- end ----"
    echo "FATAL: RTI v2 orchestrator crashed (rc=$cli_rc); see $OUT_DIR/rti2-orch.stderr" >&2
    exit 4
fi

{
    echo "==== RTI v2 CPU render->train->inference->policy smoke complete ===="
    echo "JSON:    $OUT_DIR/render_train_rl.json"
    cat "$OUT_DIR/render_train_rl.json" 2>/dev/null || true
} >"$OUT_DIR/cpu_loop_smoke.log"

echo "==== RTI v2 smoke complete; summary at $OUT_DIR/render_train_rl.json ===="