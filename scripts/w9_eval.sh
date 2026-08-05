#!/usr/bin/env bash
# W9 MindSpeed-MM evaluation end-to-end driver (Phase-2 W9).
#
# Usage:
#     bash scripts/w9_eval.sh
#
# Runs in worldfoundry-base:phase1 Docker container on 8x NPU host.
# Produces docs/superpowers/specs/W9_EVAL_REPORT.md with GO/NO-GO verdict.
set -uo pipefail  # NOTE: not -e: we want to continue past bench failures to write report

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p bench_results docs/superpowers/specs

# Capture stderr to a per-step log so a bench failure's exception (e.g.,
# NotImplementedError from MMStrategy.wrap) is preserved alongside its
# rc. The report references bench_results/<tag>.log so the W9 evidence
# includes failure context, not just a "file missing" symptom.
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

run_benches() {
    # 1. FSDP baseline (always succeeds)
    run_step "BENCH: FSDP baseline (train profile)" "fsdp-baseline" \
        python -m worldfoundry_train.bench \
            --profile train --steps 200 --warmup 50 --tag fsdp-baseline || true

    # 2. MM eval (may fail — that's the point)
    run_step "BENCH: MM eval (train-mmspeed profile)" "mindspeed-eval" \
        python -m worldfoundry_train.bench \
            --profile train-mmspeed --steps 200 --warmup 50 --tag mindspeed-eval || true
}

if [[ "${WORLDFORGE_NO_DOCKER:-0}" == "1" ]]; then
    run_benches
else
    # Expose all 8 NPU devices per plan Task 6.
    DEVICE_ARGS=()
    for i in 0 1 2 3 4 5 6 7; do
        DEVICE_ARGS+=("--device" "/dev/davinci${i}:/dev/davinci${i}")
    done
    docker run --rm \
        "${DEVICE_ARGS[@]}" \
        --device /dev/davinci_manager \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
        -v "$ROOT:/workspace" -v /data:/data \
        -e WORLDFORGE_PROFILE=train-mmspeed \
        worldfoundry-base:phase1 bash -c "cd /workspace && $(declare -f run_step run_benches) && run_benches"
fi

# 3. Compare + write report
echo "==== COMPARE: writing W9_EVAL_REPORT.md ===="
python -c "
from pathlib import Path
from worldfoundry_train.bench.compare import compare
verdict = compare(
    Path('bench_results/fsdp-baseline.json'),
    Path('bench_results/mindspeed-eval.json'),
    Path('docs/superpowers/specs/W9_EVAL_REPORT.md'),
)
print(f'verdict={verdict}')
"

echo "==== W9 eval complete; see docs/superpowers/specs/W9_EVAL_REPORT.md ===="
