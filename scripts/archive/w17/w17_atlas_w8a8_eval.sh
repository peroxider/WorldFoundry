#!/usr/bin/env bash
# W17 Atlas W8A8 validation end-to-end driver.
#
# Usage:
#     bash scripts/w17_atlas_w8a8_eval.sh
#
# Runs in worldfoundry-base:phase1 Docker container on Atlas 8x NPU host.
# Produces docs/superpowers/specs/W11_W12_QUANT_REPORT.md populated with
# measured metrics + GO/NO-GO verdict.
#
# Env vars:
#     WORLDFORGE_QUANT_MODEL      model dir (default models/vjepa2-vitl-1.2b)
#     WORLDFORGE_QUANT_OUTPUT     output dir (default models/vjepa2-vitl-1.2b-w8a8)
#     WORLDFORGE_NO_DOCKER=1      skip Docker wrapper (assume NPUs already mounted)
set -uo pipefail  # not -e: continue past failures to write report

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p bench_results docs/superpowers/specs

MODEL_DIR="${WORLDFORGE_QUANT_MODEL:-models/vjepa2-vitl-1.2b}"
OUTPUT_DIR="${WORLDFORGE_QUANT_OUTPUT:-models/vjepa2-vitl-1.2b-w8a8}"
EVAL_REPORT="bench_results/w17_w8a8_eval.json"

# Per-step stderr capture (mirrors w9_eval.sh pattern)
run_step() {
    local title="$1"; shift
    local tag="$1"; shift
    local log="bench_results/${tag}.log"
    echo "==== $title ===="
    if "$@" 2>"$log"; then
        local rc=0
    else
        local rc=$?
    fi
    echo "==== $title rc=$rc ===="
    if [[ "$rc" -ne 0 ]]; then
        echo "---- last 50 lines of $log ----"
        tail -n 50 "$log" || true
        echo "---- end ----"
    fi
    return $rc
}

run_w17() {
    # Step 0: preflight (must pass — bail loudly otherwise)
    run_step "PREFLIGHT: env check" "w17-preflight" \
        python scripts/archive/w17/w17_preflight.py || return $?

    # Step 1: W8A8 quantize
    run_step "QUANTIZE: bf16 -> W8A8" "w17-quantize" \
        python -m worldfoundry_inf.compile.quantize \
            "$MODEL_DIR" "$OUTPUT_DIR" \
            --quant-type w8a8 \
            --calibration-dataset videomix22m-512 || return $?

    # Step 2: cosine eval -> JSON
    run_step "EVAL: cosine >= 0.98" "w17-eval" \
        python -c "
from pathlib import Path
import json
from worldfoundry_inf.eval.quant_eval import cosine_similarity
report = cosine_similarity(
    baseline_path=Path('$MODEL_DIR'),
    quantized_path=Path('$OUTPUT_DIR'),
    threshold=0.98,
    n_samples=100,
)
Path('$EVAL_REPORT').write_text(json.dumps({
    'aggregate_cosine': report.aggregate_cosine,
    'per_sample_cosine': report.per_sample_cosine,
    'passed': report.passed,
    'threshold': report.threshold,
    'n_samples': report.n_samples,
}, indent=2))
print(f\"cosine={report.aggregate_cosine:.4f} {'PASS' if report.passed else 'FAIL'}\")
" || return $?
}

if [[ "${WORLDFORGE_NO_DOCKER:-0}" == "1" ]]; then
    run_w17
else
    DEVICE_ARGS=()
    for i in 0 1 2 3 4 5 6 7; do
        DEVICE_ARGS+=("--device" "/dev/davinci${i}:/dev/davinci${i}")
    done
    docker run --rm \
        "${DEVICE_ARGS[@]}" \
        --device /dev/davinci_manager \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
        -v "$ROOT:/workspace" -v /data:/data \
        -e WORLDFORGE_QUANT_MODEL="$MODEL_DIR" \
        -e WORLDFORGE_QUANT_OUTPUT="$OUTPUT_DIR" \
        worldfoundry-base:phase1 bash -c "cd /workspace && $(declare -f run_step run_w17) && run_w17"
fi

# Step 3: report populate (separate Python script — see T2)
echo "==== REPORT: populate W11_W12_QUANT_REPORT.md ===="
python scripts/w17_populate_report.py \
    --quant-meta "$OUTPUT_DIR/quant_meta.json" \
    --eval-report "$EVAL_REPORT" \
    --output docs/superpowers/specs/W11_W12_QUANT_REPORT.md

echo "==== W17 eval complete; see docs/superpowers/specs/W11_W12_QUANT_REPORT.md ===="
