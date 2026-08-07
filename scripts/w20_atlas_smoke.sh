#!/usr/bin/env bash
# W20 Atlas single-card LoRA smoke.
#
# Usage:
#     bash scripts/w20_atlas_smoke.sh                  # default: 5 + 100 steps
#     bash scripts/w20_atlas_smoke.sh --steps 5         # 5-step smoke only
#     bash scripts/w20_atlas_smoke.sh --steps 100       # 100-step run only
#     WORLDFORGE_NO_DOCKER=1 bash scripts/w20_atlas_smoke.sh
#                                                     # skip Docker wrapper
#                                                     # (assume NPU already mounted)
#
# Runs in worldfoundry-base:phase1 Docker container on Atlas 1x NPU host.
# Validates G1 (训练链路 .to("npu") 迁移): single-card LoRA micro-training
# on Atlas lands .to("npu") via the W20 CPU-prep migration point in
# worldfoundry_train/device.py, loss decreases monotonically, checkpoint
# saves/loads cleanly.
#
# The 5-step smoke is the gate: if it fails, do not proceed to the 100-step
# run (save Atlas GPU-hours). On 5-step PASS, the 100-step run produces
# bench_results/atlas_baseline/single_card.json (clips/s, step latency).
#
# Env vars:
#     WORLDFORGE_NO_DOCKER=1      skip Docker (assume NPUs already mounted)
#     WORLDFORGE_CONFIG           YAML config (default configs/vjepa2_vitl_1.2b.yaml)
#     WORLDFORGE_BASELINE_OUT     baseline output dir
#                                 (default bench_results/atlas_baseline)
set -uo pipefail  # not -e: continue past failures to write report

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p bench_results/atlas_baseline

CONFIG="${WORLDFORGE_CONFIG:-worldfoundry_train/configs/vjepa2_vitl_1.2b.yaml}"
BASELINE_OUT="${WORLDFORGE_BASELINE_OUT:-bench_results/atlas_baseline}"

# Args
SMOKE_STEPS=5
FULL_STEPS=100
RUN_BOTH=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --steps)
            SMOKE_STEPS="$2"
            FULL_STEPS="$2"
            RUN_BOTH=0
            shift 2
            ;;
        --smoke-steps)  SMOKE_STEPS="$2"; shift 2 ;;
        --full-steps)   FULL_STEPS="$2"; shift 2 ;;
        --config)       CONFIG="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,21p' "$0"
            exit 0
            ;;
        *)
            echo "WARN: unknown arg $1 (ignored)" >&2
            shift
            ;;
    esac
done

if [[ ! -f "$CONFIG" ]]; then
    echo "FATAL: config not found: $CONFIG" >&2
    exit 2
fi

# Per-step stderr capture (mirrors w17_atlas_w8a8_eval.sh pattern)
run_step() {
    local title="$1"; shift
    local steps="$1"; shift
    local tag="$1"; shift
    local log="bench_results/${tag}.log"
    echo "==== $title (steps=$steps) ===="
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

# The actual training invocation. One CLI call covers the full LoRA path:
#   python -m worldfoundry_train.train loads YAML, resolves device via
#   worldfoundry_train.device.resolve_device() (= "npu" on Atlas),
#   wraps the LoRA-adapted VJEPA2Model via strategy.wrap(), then calls
#   worldfoundry_train.device.to_device() (the W20 CPU-prep migration
#   point — see worldfoundry_train/device.py:78 + trainer.py wrap step).
#   We force --parallel ddp so the single-card path goes through
#   worldfoundry_train/parallel/ddp.py:setup_ddp (rank 0 only) rather
#   than FSDP's all-gather, which on a 1-NPU host is a no-op anyway.
run_train() {
    local steps="$1"; shift
    local tag="$1"; shift
    "$PY" -m worldfoundry_train.train \
        --config "$CONFIG" \
        --steps "$steps" \
        --mode lora \
        --parallel ddp \
        --save-every-n 0
}

run_w20() {
    # Step 0: preflight — NPU + torch_npu + vllm-ascend sanity (no model load).
    run_step "PREFLIGHT: NPU + torch_npu + transformers" "0" "w20-preflight" \
        "$PY" -c "
from worldfoundry_common import npu_env
import transformers
print(f'[preflight] npu_available={npu_env.is_npu_available()}')
print(f'[preflight] npu_device_count={npu_env.npu_device_count()}')
print(f'[preflight] transformers={transformers.__version__}')
assert npu_env.is_npu_available(), 'NPU not available — abort'
" || return $?

    # Step 1: 5-step smoke (the gate). Loss should drop within 5 steps on a
    # fresh LoRA init + a real encoder; if not, the .to("npu") wiring or
    # dtype cache is broken. Capture rc but do NOT abort yet — the script
    # continues to print the full failure context so the next operator
    # session can diagnose without re-running.
    run_step "SMOKE: 5-step LoRA on Atlas NPU" "$SMOKE_STEPS" "w20-smoke-5" \
        run_train "$SMOKE_STEPS" "w20-smoke-5" || return $?

    if [[ "$RUN_BOTH" -eq 0 ]]; then
        echo "==== W20: --steps $SMOKE_STEPS only; skipping 100-step run ===="
        return 0
    fi

    # Step 2: 100-step baseline run. Produces clips/s + step latency
    # throughput record for the Phase-3 regression check.
    run_step "BASELINE: 100-step LoRA on Atlas NPU" "$FULL_STEPS" "w20-baseline-100" \
        run_train "$FULL_STEPS" "w20-baseline-100" || return $?

    # Step 3: harvest throughput numbers into the canonical baseline file.
    # The CLI emits step-time + wall-time to stderr under WORLDFORGE_TRAINING_TIMING=1;
    # we keep the raw text under w20-baseline-100.log and surface a minimal
    # JSON summary that the W29 regression check consumes.
    run_step "BASELINE: write throughput JSON" "$FULL_STEPS" "w20-baseline-json" \
        "$PY" - <<PY || return $?
import json
import re
from pathlib import Path

log = Path("bench_results/w20-baseline-100.log").read_text(errors="ignore")
step_re = re.compile(r"step=(\d+).*?loss=([0-9.eE+-]+).*?dt=([0-9.eE+-]+)s")
points = [(int(m[1]), float(m[2]), float(m[3])) for m in step_re.finditer(log)]
out = {
    "label": "W20 Atlas single-card LoRA 100-step baseline",
    "config": "$CONFIG",
    "steps_completed": len(points),
    "avg_step_s": round(sum(p[2] for p in points) / max(len(points), 1), 6),
    "first_loss": points[0][1] if points else None,
    "last_loss":  points[-1][1] if points else None,
    "loss_decreased": (
        points[-1][1] < points[0][1] if len(points) >= 2 else None
    ),
    "device": "npu",
}
out_path = Path("$BASELINE_OUT/single_card.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(out, indent=2))
print(f"[W20] baseline JSON: {out_path}")
print(f"[W20] steps={out['steps_completed']} "
      f"avg_dt={out['avg_step_s']:.4f}s "
      f"loss {out['first_loss']} -> {out['last_loss']} "
      f"decreased={out['loss_decreased']}")
PY
}

if [[ "${WORLDFORGE_NO_DOCKER:-0}" == "1" ]]; then
    PY=python
    run_w20
else
    # Resolve python (python3 fallback for WSL / some Linux distros).
    if command -v python >/dev/null 2>&1; then
        PY=python
    elif command -v python3 >/dev/null 2>&1; then
        PY=python3
    else
        echo "FATAL: neither 'python' nor 'python3' is on PATH" >&2
        exit 2
    fi
    DEVICE_ARGS=(--device /dev/davinci0:/dev/davinci0 --device /dev/davinci_manager)
    docker run --rm \
        "${DEVICE_ARGS[@]}" \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
        -v "$ROOT:/workspace" -v /data:/data \
        -e WORLDFORGE_CONFIG="$CONFIG" \
        -e WORLDFORGE_BASELINE_OUT="$BASELINE_OUT" \
        worldfoundry-base:phase1 bash -c "cd /workspace && $(declare -f run_step run_train run_w20) && run_w20"
fi

echo "==== W20 smoke complete; throughput at $BASELINE_OUT/single_card.json ===="