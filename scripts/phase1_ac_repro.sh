#!/usr/bin/env bash
# Phase-1 AC finetuning reproduction runner (Task 21).
# Per spec §5.3: this runs the AC finetuning phase only (latent action
# prediction), NOT the full 80% Meta reproduction.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODEL_PATH="${WORLDFORGE_MODEL:-models/vjepa2-vitl-1.2b}"
MANIFEST="${WORLDFORGE_DROID_MANIFEST:-/data/droid/manifest.csv}"

echo "==== Phase-1 AC finetuning ===="
echo "model:    $MODEL_PATH"
echo "manifest: $MANIFEST"

python -m worldfoundry_rl.train_ac \
    --model "$MODEL_PATH" \
    --manifest "$MANIFEST" \
    --epochs "${WORLDFORGE_AC_EPOCHS:-100}" \
    --kl-coef "${WORLDFORGE_AC_KL_COEF:-0.1}" \
    --max-steps "${WORLDFORGE_AC_MAX_STEPS:-50}"

echo "==== Phase-1 AC finetuning PASS ===="