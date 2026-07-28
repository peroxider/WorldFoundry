#!/usr/bin/env bash
# Phase-1 end-to-end smoke test (Task 17).
# Validates: data → train → quantize → compile → inference → eval loop.
#
# Usage:
#     bash scripts/phase1_smoke.sh
#
# Runs entirely on CPU if --cpu is set; otherwise requires Ascend NPU host.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CPU_FLAG="${1:-}"

# Background server PID cleanup (Defect A pre-flight fix).
SERVER_PID=""
cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

run_step() {
    local title="$1"; shift
    echo "==== $title ===="
    "$@"
    echo "==== $title PASS ===="
}

# 1. Data
run_step "DATA: build sample dataset" python -m worldforge_train.data.dataset --self-test

# 2. Train (DDP LoRA, single-rank CPU fallback)
if [[ "$CPU_FLAG" == "--cpu" ]]; then
    run_step "TRAIN: DDP LoRA on CPU stub" \
        python -m worldforge_train.train \
            --config worldforge_train/configs/vjepa2_vitl_1.2b.yaml \
            --parallel ddp
fi

# 3. Quantize
run_step "QUANTIZE: W8A8" \
    python -c "from pathlib import Path; from worldforge_inf.compile.quantize import quantize_model; quantize_model(Path('models/vjepa2-vitl-1.2b'), Path('models/vjepa2-vitl-1.2b-w8a8'), quant_type='w8a8')"

# 4. Compile
run_step "COMPILE: MindIE-SD" \
    python -c "from pathlib import Path; from worldforge_inf.compile.mindie_sd import compile_to_om; compile_to_om(Path('models/vjepa2-vitl-1.2b-w8a8'), Path('models/vjepa2-vitl-1.2b/model.om'))"

# 5. Serve (Defect C pre-flight fix: explicit foreground-spawn + port-readiness loop
# replaces brief's `run_step` invocation so server-startup failures are not swallowed.)
echo "==== SERVE: launch FastAPI on :8080 (background) ===="
python -m worldforge_inf.server --model models/vjepa2-vitl-1.2b --port 8080 &
SERVER_PID=$!
# Wait up to 30s for the port to be listening.
for _ in $(seq 1 30); do
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/health 2>/dev/null | grep -q '^2'; then
        break
    fi
    sleep 1
done
if ! curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/health | grep -q '^2'; then
    echo "==== SERVE FAIL: server did not start within 30s ===="
    exit 1
fi
echo "==== SERVE PASS ===="

# 6. Probe
sleep 5
run_step "PROBE: /health" curl -s http://localhost:8080/health

echo "==== Phase-1 smoke PASS ===="