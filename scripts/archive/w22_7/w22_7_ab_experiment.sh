#!/usr/bin/env bash
# W22.7 A/B/C experiment driver (design §9.2 + review §2.1): proxy-loss
# baseline (Arm A) vs true JEPA objective (Arm B) vs zero-mask sanity
# check (Arm C).
#
# Usage:
#     bash scripts/w22_7_ab_experiment.sh              # dry self-check only (CPU)
#     W22_7_MODE=atlas bash scripts/w22_7_ab_experiment.sh   # full Atlas run
#
# Sequence: dry-run self-check -> [atlas] subset manifest -> pilot {A,B,C}
# -> atlas {A,B,C} -> aggregate report (docs/superpowers/specs/
# 2026-08-04-w22-7-ab-experiment-report.md).
#
# Env vars:
#     W22_7_MODE          dry (default) | atlas
#     W22_7_SKIP_PILOT=1  skip the pilot stage (formal run starts directly)
#     W22_7_SKIP_ARM_C=1  skip arm C (zero-mask sanity check) entirely
#     W22_7_PILOT_STEPS   pilot steps per arm (default 200 atlas / 40 dry)
#     W22_7_SOURCE_MANIFEST   full manifest for subset sampling
#                             (atlas; default /data/videomix22m/manifest.csv)
#     W22_7_SUBSET_N      subset size (atlas; default 10000)
#     W22_7_STEPS         micro-batch steps per arm (atlas; default = cfg epochs)
#
# TODO(W20): 8-card hccl launch. When NPU device migration lands, wrap each
# atlas arm as:
#     torchrun --nproc_per_node=8 scripts/archive/w22_7/w22_7_run_arm.py --arm {a,b,c} --mode atlas
# with setup_fsdp() hccl init + tensors on NPU, FSDP FULL_SHARD, bf16.
# The single-process atlas entry below already exercises the full
# config -> training -> checkpoint -> artifact path.
set -uo pipefail  # not -e: continue past failures to write report

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# scripts/ is sys.path[0] for `python scripts/x.py`; make the package importable.
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

MODE="${W22_7_MODE:-dry}"
RESULTS="bench_results/w22_7"
mkdir -p "$RESULTS"

run_step() {  # mirrors scripts/w17_atlas_w8a8_eval.sh pattern
    local title="$1"; shift
    local tag="$1"; shift
    local log="${RESULTS}/${tag}.log"
    echo "==== $title ===="
    if "$@" 2>"$log"; then local rc=0; else local rc=$?; fi
    echo "==== $title rc=$rc ===="
    if [[ "$rc" -ne 0 ]]; then
        echo "---- last 50 lines of $log ----"
        tail -n 50 "$log" || true
        echo "---- end ----"
    fi
    return $rc
}

# Step 0: dry self-check — always runs. Proves metrics computable, timing
# written, artifacts land, before any hardware time is spent.
run_step "DRY-RUN arm A (proxy baseline, CPU)" "dry-arm-a" \
    python scripts/archive/w22_7/w22_7_run_arm.py --arm a --mode dry || exit $?
run_step "DRY-RUN arm B (true JEPA, CPU)" "dry-arm-b" \
    python scripts/archive/w22_7/w22_7_run_arm.py --arm b --mode dry || exit $?
if [[ "${W22_7_SKIP_ARM_C:-0}" != "1" ]]; then
    run_step "DRY-RUN arm C (zero-mask sanity check, CPU)" "dry-arm-c" \
        python scripts/archive/w22_7/w22_7_run_arm.py --arm c --mode dry || exit $?
fi

# Step 0.4: fixed subset manifest (atlas only) — pilot and formal arms all
# consume it, so it must exist before the pilot stage.
if [[ "$MODE" == "atlas" ]]; then
    run_step "SUBSET: fixed manifest" "subset" \
        python scripts/archive/w22_7/w22_7_make_subset.py \
            --source "${W22_7_SOURCE_MANIFEST:-/data/videomix22m/manifest.csv}" \
            --n "${W22_7_SUBSET_N:-10000}" \
            --out "$RESULTS/subset_manifest.csv" || exit $?
fi

# Step 0.5: pilot (short run + early-stop checks) — dry backend validates the
# harness on CPU; atlas backend gates the formal run on real hardware.
if [[ "${W22_7_SKIP_PILOT:-0}" != "1" ]]; then
    PILOT_BACKEND="dry"
    if [[ "$MODE" == "atlas" ]]; then PILOT_BACKEND="atlas"; fi
    pilot_failed=0
    run_step "PILOT arm A ($PILOT_BACKEND)" "pilot-arm-a" \
        python scripts/archive/w22_7/w22_7_run_arm.py --arm a --mode pilot \
            --backend "$PILOT_BACKEND" || pilot_failed=$?
    run_step "PILOT arm B ($PILOT_BACKEND)" "pilot-arm-b" \
        python scripts/archive/w22_7/w22_7_run_arm.py --arm b --mode pilot \
            --backend "$PILOT_BACKEND" || pilot_failed=$?
    if [[ "${W22_7_SKIP_ARM_C:-0}" != "1" ]]; then
        run_step "PILOT arm C ($PILOT_BACKEND)" "pilot-arm-c" \
            python scripts/archive/w22_7/w22_7_run_arm.py --arm c --mode pilot \
                --backend "$PILOT_BACKEND" || pilot_failed=$?
    fi
    if [[ "$pilot_failed" -ne 0 ]]; then
        echo "==== PILOT NO-GO (arm rc=$pilot_failed): 中止正式跑，请人工检查"
        echo "     bench_results/w22_7/{a,b,c}/pilot/verdict.json ===="
        # Still emit the report so the NO-GO verdict is archived.
        run_step "REPORT: aggregate (pilot NO-GO)" "report" \
            python scripts/archive/w22_7/w22_7_build_report.py --results "$RESULTS"
        exit 3
    fi
fi

if [[ "$MODE" == "atlas" ]]; then
    STEPS_ARGS=()
    if [[ -n "${W22_7_STEPS:-}" ]]; then
        STEPS_ARGS+=(--steps "${W22_7_STEPS}")
    fi
    # TODO(W20): replace with torchrun --nproc_per_node=8 once NPU device
    # migration lands; see header comment.
    run_step "ATLAS arm A: proxy baseline" "atlas-arm-a" \
        python scripts/archive/w22_7/w22_7_run_arm.py --arm a --mode atlas \
            "${STEPS_ARGS[@]}" || true
    run_step "ATLAS arm B: true JEPA" "atlas-arm-b" \
        python scripts/archive/w22_7/w22_7_run_arm.py --arm b --mode atlas \
            "${STEPS_ARGS[@]}" || true
    if [[ "${W22_7_SKIP_ARM_C:-0}" != "1" ]]; then
        run_step "ATLAS arm C: zero-mask sanity check" "atlas-arm-c" \
            python scripts/archive/w22_7/w22_7_run_arm.py --arm c --mode atlas \
                "${STEPS_ARGS[@]}" || true
    fi
fi

# Final: aggregate report (always — TBD placeholders for missing numbers).
run_step "REPORT: aggregate" "report" \
    python scripts/archive/w22_7/w22_7_build_report.py --results "$RESULTS"

echo "==== W22.7 done; report: docs/superpowers/specs/2026-08-04-w22-7-ab-experiment-report.md ===="
