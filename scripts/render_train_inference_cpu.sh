#!/usr/bin/env bash
# RTI v1 CPU render->train->inference smoke.
#
# Usage:
#     bash scripts/render_train_inference_cpu.sh                 # default 8 clips, 50 steps
#     bash scripts/render_train_inference_cpu.sh --num-clips 4
#     bash scripts/render_train_inference_cpu.sh --num-steps 30
#
# Validates the full loop on CPU:
#     pytest worldfoundry_train/tests/test_render_train_inference_cpu.py
#     python scripts/render_train_inference_cpu.py --num-clips 8 --num-steps 50
#
# Writes:
#     bench_results/render_train_inference/videos/*.mp4    Render phase
#     bench_results/render_train_inference/manifest.csv    Render phase
#     bench_results/render_train_inference/tiny_jepa.pt    Train phase ckpt
#     bench_results/render_train_inference/render_vs_predicted.png  Inference phase
#     bench_results/render_train_inference/render_train_inference.json  summary
#
# Exit codes:
#     0  all gates pass + summary JSON written
#     2  usage / config error
#     3  unit-test gate failed
#     4  orchestrator gate failed
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p bench_results/render_train_inference

PY="${PY:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
    echo "FATAL: $PY not on PATH" >&2
    exit 2
fi

NUM_CLIPS=8
NUM_STEPS=50
SEED=42
OUT_DIR="bench_results/render_train_inference"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-clips) NUM_CLIPS="$2"; shift 2 ;;
        --num-steps) NUM_STEPS="$2"; shift 2 ;;
        --seed)      SEED="$2"; shift 2 ;;
        --out-dir)   OUT_DIR="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,16p' "$0"
            exit 0
            ;;
        *)
            echo "WARN: unknown arg $1 (ignored)" >&2
            shift
            ;;
    esac
done

mkdir -p "$OUT_DIR"

LOG="$OUT_DIR/cpu_loop_smoke.log"
JSON="$OUT_DIR/render_train_inference.json"

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
# Gate 1: RTI unit tests
# ------------------------------------------------------------------
run_step "GATE 1: RTI unit tests" "rti-unit" \
    "$PY" -m pytest worldfoundry_train/tests/test_render_train_inference_cpu.py -v --tb=short || {
        echo "FATAL: RTI unit-test gate failed (rc=$?)" >&2
        exit 3
    }

# ------------------------------------------------------------------
# Gate 2: render -> train -> inference orchestrator
# ------------------------------------------------------------------
cli_rc=0
echo "==== GATE 2: RTI orchestrator (num_clips=$NUM_CLIPS num_steps=$NUM_STEPS seed=$SEED) ===="
"$PY" scripts/render_train_inference_cpu.py \
    --out-dir "$OUT_DIR" \
    --num-clips "$NUM_CLIPS" \
    --num-steps "$NUM_STEPS" \
    --seed "$SEED" \
    2>"$OUT_DIR/rti-orch.stderr" || cli_rc=$?
echo "==== GATE 2: RTI orchestrator rc=$cli_rc ===="

if [[ "$cli_rc" -ne 0 ]]; then
    echo "---- last 60 lines of rti-orch.stderr ----"
    tail -n 60 "$OUT_DIR/rti-orch.stderr" || true
    echo "---- end ----"
    echo "FATAL: RTI orchestrator crashed (rc=$cli_rc); see $OUT_DIR/rti-orch.stderr" >&2
    exit 4
fi

# Mirror the JSON to a .log (regression suite greps for [RTI] tags).
{
    echo "==== RTI CPU render->train->inference smoke complete ===="
    echo "JSON:    $JSON"
    echo "Pytest:  $OUT_DIR/rti-unit.log"
    echo "CLI:     $OUT_DIR/rti-orch.stderr"
    cat "$JSON" 2>/dev/null || true
} >"$LOG"

echo "==== RTI smoke complete; summary at $JSON ===="