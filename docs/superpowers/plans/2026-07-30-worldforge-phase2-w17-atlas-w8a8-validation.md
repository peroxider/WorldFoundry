# W17 Atlas W8A8 Validation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) to execute this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

## Atlas Operator Instructions

This plan has two phases:
- **CPU-side (T1-T2)**: Dev host executes. Pure Python + shell + tests. No NPU required.
- **Atlas-side (T3-T7)**: Atlas 800 A2 / 910B host executes. Requires `torch_npu`, `msmodelslim`, V-JEPA 2 weights, CANN driver.

**Fast path (single NPU)** — for cos-sin gate only: 1 NPU is sufficient. Quantize + eval both run on a single device; msModelSlim shards calibration internally.

**Slow path (8×NPU)** — production target: 8 NPU devices, maps to `scripts/archive/w9/w9_eval.sh` Docker wrapper. The driver auto-maps `/dev/davinci0..7` if `$WORLDFORGE_NO_DOCKER` is unset.

**Env-var cheat sheet**:
- `WORLDFORGE_QUANT_MODEL` — model dir (default `models/vjepa2-vitl-1.2b`)
- `WORLDFORGE_QUANT_OUTPUT` — output dir (default `models/vjepa2-vitl-1.2b-w8a8`)
- `WORLDFORGE_NO_DOCKER=1` — skip Docker wrapper (assume NPUs already mounted)
- `WORLDFORGE_RUN_QUANT_E2E=1` — enable the NPU-gated integration test
- `WORLDFORGE_QUANT_TEST_MODEL` — override the surrogate model used by the NPU-gated test

## Context

W11-W12 shipped the CPU-runnable quantization stack end-to-end on `master`:

- `worldforge_inf/compile/quantize.py` — facade dispatching by `quant_type` (`w8a8` supported, `w4a8` registered as deferred).
- `worldforge_inf/compile/quantize_backends/msmodelslim.py` — `W8A8Backend` wrapping `msmodelslim.MSModelSlim(...)` with import-guard so the module stays importable on x86 dev/CI hosts.
- `worldforge_inf/data/calib_videomix22m.py` — fixed 512-sample VideoMix22M loader (seed=42).
- `worldforge_inf/eval/quant_eval.py` — `cosine_similarity(baseline, quantized, threshold=0.98, n_samples=100, seed=123)` gate + CLI exit 0/1.
- `worldforge_inf/tests/test_quant_integration_npu.py` — NPU-gated E2E test (default skip via `WORLDFORGE_RUN_QUANT_E2E`, uses `facebook/dino-vits8` surrogate).
- `docs/superpowers/specs/W11_W12_QUANT_REPORT.md` — report template with all metrics TBD.

**W17 closes the gap between "the code is shippable" and "we have evidence it works on production hardware".** Without W17, we have no proof that msModelSlim produces an OM whose downstream features stay within cosine ≥ 0.98 on real Atlas silicon against the real V-JEPA 2 vitl-1.2b (1.2B params, bf16 baseline). The 8×NPU run is the spec §9 row #6 acceptance gate that has been deferred since the W11-W12 merge.

W17's job:
1. **CPU-doable work (T1-T2)** — write the plan doc, ship the shell driver + pre-flight env check that automates the Atlas-side run, enrich the report template with explicit metric-collection fields.
2. **Atlas-side runs (T3-T7)** — actually execute the 5-step validation on 8×NPU and populate measured numbers into the report.

## Decisions

1. **Branch:** `phase2/w17-atlas-w8a8-validation` from master (NOT direct commits on master, because T3-T7 may need iteration if cosine < 0.98). Mirror the W11-W12 / W13 branch strategy. If the dev host cannot create the branch (classifier restriction), work on master and the operator can `git checkout -b phase2/w17-atlas-w8a8-validation` to begin the operator-side branch.
2. **Shell driver:** Analogous to `scripts/archive/w9/w9_eval.sh` — captures per-step stderr to `bench_results/<tag>.log`, exposes all 8 NPU devices, writes the final report. Lives at `scripts/archive/w17/w17_atlas_w8a8_eval.sh`.
3. **Pre-flight check:** New `scripts/archive/w17/w17_preflight.py` exits 0 only if `torch_npu`, `msmodelslim`, V-JEPA 2 weights dir, and CANN driver all present. Returns actionable hints otherwise. Driver runs it before any expensive step.
4. **Model weights:** V-JEPA 2 vitl-1.2b bf16 weights fetched via `huggingface-cli download facebook/vjepa2-vitl-1.2b` to `models/vjepa2-vitl-1.2b/` (the W11-W12 plan §3 default). Pre-flight checks dir exists + contains `config.json` + `*.safetensors`.
5. **Hardware target:** Atlas 800 A2 / 910B × 8 (the production target noted in `W11_W12_QUANT_REPORT.md`). Single-host is acceptable for the cosine gate (8× only needed if we want to also do training on the quantized model, which is out of scope for W17).
6. **Decision gate:** Cosine ≥ 0.98 → **GO**; cosine < 0.98 → **NO-GO** (defer W8A8 to a follow-up sub-project, possibly tune calibration size or fall back to a different backend). Compression ≥ 50% as soft secondary metric.
7. **Test coverage:** CPU tests already cover the wiring (`test_quantize.py`, `test_quantize_backends_*`, `test_quant_eval.py`, `test_quant_integration_cpu.py`, `test_quant_facade_dispatch.py`). W17 adds no new CPU tests against the quantization stack — the work is hardware-side. The NPU-gated `test_quant_integration_npu.py` test will be re-run on Atlas with `WORLDFORGE_QUANT_TEST_MODEL=models/vjepa2-vitl-1.2b` to produce the canonical run. The new CPU tests in W17 are for the pre-flight + populate scripts only (15 tests total).

## Goal

Atlas NPU produces a quantized `models/vjepa2-vitl-1.2b-w8a8/` OM with `quant_meta.json` sidecar + cosine ≥ 0.98 vs the bf16 baseline on the 100-sample VideoMix22M holdout. `W11_W12_QUANT_REPORT.md` populated with measured metrics + GO/NO-GO verdict. Two CPU-doable commits (T1 + T2), then operator runs T3-T7 on Atlas and merges the final report.

## Architecture

```
scripts/archive/w17/w17_preflight.py (new)
    |
    +-- checks: torch_npu present, msmodelslim importable, weights dir valid,
    |           CANN driver accessible, NPU device count >= 1
    +-- exits 0 on success, 1 with actionable hints otherwise

scripts/archive/w17/w17_atlas_w8a8_eval.sh (new)
    |
    +-- preflight (must pass)
    +-- step 1: python -m worldforge_inf.compile.quantize quantize_model(...) →
    |           models/vjepa2-vitl-1.2b-w8a8/  (writes quant_meta.json + *.om)
    +-- step 2: python -m worldforge_inf.eval.quant_eval cosine_similarity(...) →
    |           bench_results/w17_w8a8_eval.json  (aggregate + per-sample cosines)
    +-- step 3: scripts/archive/w17/w17_populate_report.py → populates
    |           docs/superpowers/specs/W11_W12_QUANT_REPORT.md

scripts/archive/w17/w17_populate_report.py (new)
    |
    +-- reads <output_dir>/quant_meta.json + bench_results/w17_w8a8_eval.json
    +-- substitutes bracketed [W17-MEASURED: ...] markers in the template
    +-- writes the final report with GO/NO-GO verdict

docs/superpowers/plans/2026-07-30-worldforge-phase2-w17-atlas-w8a8-validation.md (new)
    |
    +-- mirror of this plan

docs/superpowers/specs/W11_W12_QUANT_REPORT.md (modified, template)
    |
    +-- add explicit [W17-MEASURED: ...] field markers
    |   so the populate script can substitute measured values cleanly
```

The existing `worldforge_inf/compile/quantize.py` + `quantize_backends/msmodelslim.py` + `eval/quant_eval.py` are NOT modified. W17 is verification + reporting, not new features.

## Global Constraints

- **G1**: Atlas 800 A2 / 910B × NPU (single is fine for the cosine gate); CANN 8.5.1 + torch_npu 2.9.0 (already pinned in `pyproject.toml`). NPU driver accessible at `/usr/local/Ascend/driver`.
- **G2**: `msmodelslim==26.1.0` already pinned in `[quant]` extra (W11-W12 T1). No new deps.
- **G3 (W17)**:
  - **Allowed to create**: `scripts/archive/w17/w17_preflight.py`, `scripts/archive/w17/w17_atlas_w8a8_eval.sh`, `scripts/archive/w17/w17_populate_report.py`, `worldforge_inf/tests/test_w17_preflight.py`, `worldforge_inf/tests/test_w17_populate_report.py`, `docs/superpowers/plans/2026-07-30-worldforge-phase2-w17-atlas-w8a8-validation.md`.
  - **Allowed to modify**: `docs/superpowers/specs/W11_W12_QUANT_REPORT.md` (add field markers only), `.superpowers/sdd/progress.md` (W17 launch note).
  - **NOT TOUCHED**: `worldforge_inf/compile/*`, `worldforge_inf/data/*`, `worldforge_inf/eval/*`, `pyproject.toml`, the existing `test_quant_*` suite, anything in `worldforge_train/*`, `worldforge_common/*`.
- **G4**: No silent failures. Pre-flight exits non-zero with actionable hint on any missing dep; shell driver captures per-step stderr to log; final report records actual measured values (not "TBD").
- **G5**: Calibration seed=42, eval seed=123 — already fixed in `calib_videomix22m.py` and `quant_eval.py`.
- **G6**: Single source of truth for dispatch is `quantize_backends/__init__.py:get_backend(quant_type)`. W17 does not introduce new dispatch paths.
- **G7**: N/A (no I/O concurrency).
- **G8**: Each task ends with commit on the W17 branch.
- **G9**: CPU tests for the pre-flight + populate scripts use monkeypatch / tmp_path (no real NPU).

## Critical Files (read before implementation)

- `worldforge_inf/compile/quantize.py:28-69` — facade; what `quantize_model(...)` does on success and failure.
- `worldforge_inf/compile/quantize_backends/msmodelslim.py:81-114` — what the W8A8 backend actually runs (msModelSlim call + sidecar write).
- `worldforge_inf/eval/quant_eval.py:33-95` — `cosine_similarity` signature + threshold logic + report shape.
- `worldforge_inf/data/calib_videomix22m.py:21-49` — loader signature + yield shape.
- `worldforge_inf/tests/test_quant_integration_npu.py:24-45` — NPU-gated integration test pattern (template for pre-flight).
- `scripts/archive/w9/w9_eval.sh:1-82` — shell driver pattern: `run_step` helper, Docker NPU device mapping, `|| true` for soft coupling, compare-then-report.
- `docs/superpowers/specs/W11_W12_QUANT_REPORT.md` — the template to be enriched + populated.

---

## Task 1 — W17 plan + shell driver + pre-flight check (CPU-doable)

**Files:**
- Create: `docs/superpowers/plans/2026-07-30-worldforge-phase2-w17-atlas-w8a8-validation.md` (this plan, mirrored)
- Create: `scripts/archive/w17/w17_preflight.py` (~70 lines)
- Create: `scripts/archive/w17/w17_atlas_w8a8_eval.sh` (~100 lines, mirrors `w9_eval.sh`)
- Create: `worldforge_inf/tests/test_w17_preflight.py` (~10 CPU tests, monkeypatch-based)
- Modify: `docs/superpowers/specs/W11_W12_QUANT_REPORT.md` (add explicit `[W17-MEASURED: ...]` field markers)
- Modify: `.superpowers/sdd/progress.md` (note W17 launch)

**Why first:** Plan + driver are the only CPU-doable pieces. Without them, the Atlas operator has no script to run. Pre-flight fails loudly on missing deps so they don't get halfway through calibration before realizing msModelSlim is not installed.

### Step 1.1: Write the plan doc (mirror)

This file. Mirrors the system plan with the operator-instructions section at the top.

### Step 1.2: Write `scripts/archive/w17/w17_preflight.py`

```python
"""Pre-flight env check for W17 Atlas W8A8 validation.

Exits 0 only if:
  - torch_npu is importable AND at least one NPU device is reachable
  - msmodelslim is importable
  - V-JEPA 2 vitl-1.2b weights dir exists at $WORLDFORGE_QUANT_MODEL
    (default models/vjepa2-vitl-1.2b) and contains config.json + *.safetensors
  - CANN driver path /usr/local/Ascend/driver is readable

Each failure prints a one-line actionable hint to stderr. Never silent.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def _have_torch_npu() -> tuple[bool, str]:
    if importlib.util.find_spec("torch_npu") is None:
        return False, "torch_npu not installed (pip install torch_npu==2.9.0)"
    try:
        import torch_npu  # noqa: F401
        if not torch_npu.npu.is_available():
            return False, "torch_npu installed but no NPU devices visible"
        return True, ""
    except Exception as exc:
        return False, f"torch_npu import failed: {exc}"


def _have_msmodelslim() -> tuple[bool, str]:
    if importlib.util.find_spec("msmodelslim") is None:
        return False, "msmodelslim not installed (pip install worldforge[quant])"
    return True, ""


def _have_weights() -> tuple[bool, str]:
    path = Path(os.environ.get("WORLDFORGE_QUANT_MODEL",
                                "models/vjepa2-vitl-1.2b"))
    if not path.is_dir():
        return False, f"weights dir missing: {path} (huggingface-cli download facebook/vjepa2-vitl-1.2b --local-dir {path})"
    if not (path / "config.json").is_file():
        return False, f"weights dir {path} missing config.json"
    if not list(path.glob("*.safetensors")) and not list(path.glob("*.bin")):
        return False, f"weights dir {path} has no .safetensors or .bin files"
    return True, ""


def _have_cann_driver() -> tuple[bool, str]:
    path = Path("/usr/local/Ascend/driver")
    if not path.is_dir():
        return False, f"CANN driver path missing: {path} (apt-installed ascend driver)"
    if not os.access(path, os.R_OK):
        return False, f"CANN driver path {path} not readable"
    return True, ""


def preflight() -> tuple[bool, list[str]]:
    """Return (ok, list_of_failure_hints)."""
    failures: list[str] = []
    for name, check in [
        ("torch_npu", _have_torch_npu),
        ("msmodelslim", _have_msmodelslim),
        ("weights", _have_weights),
        ("cann_driver", _have_cann_driver),
    ]:
        ok, hint = check()
        if not ok:
            failures.append(f"[{name}] {hint}")
    return (len(failures) == 0, failures)


def main() -> int:
    ok, failures = preflight()
    if ok:
        print("[W17] preflight OK")
        return 0
    print("[W17] preflight FAILED:", file=sys.stderr)
    for f in failures:
        print(f"  {f}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

### Step 1.3: Write `scripts/archive/w17/w17_atlas_w8a8_eval.sh`

```bash
#!/usr/bin/env bash
# W17 Atlas W8A8 validation end-to-end driver.
#
# Usage:
#     bash scripts/archive/w17/w17_atlas_w8a8_eval.sh
#
# Runs in worldforge-base:phase1 Docker container on Atlas 8x NPU host.
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
        python -m worldforge_inf.compile.quantize \
            "$MODEL_DIR" "$OUTPUT_DIR" \
            --quant-type w8a8 \
            --calibration-dataset videomix22m-512 || return $?

    # Step 2: cosine eval -> JSON
    run_step "EVAL: cosine >= 0.98" "w17-eval" \
        python -c "
from pathlib import Path
import json
from worldforge_inf.eval.quant_eval import cosine_similarity
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
        worldforge-base:phase1 bash -c "cd /workspace && $(declare -f run_step run_w17) && run_w17"
fi

# Step 3: report populate (separate Python script — see T2)
echo "==== REPORT: populate W11_W12_QUANT_REPORT.md ===="
python scripts/archive/w17/w17_populate_report.py \
    --quant-meta "$OUTPUT_DIR/quant_meta.json" \
    --eval-report "$EVAL_REPORT" \
    --output docs/superpowers/specs/W11_W12_QUANT_REPORT.md

echo "==== W17 eval complete; see docs/superpowers/specs/W11_W12_QUANT_REPORT.md ===="
```

Note: the shell driver calls `scripts/archive/w17/w17_populate_report.py` which is created in Task 2.

### Step 1.4: Write `worldforge_inf/tests/test_w17_preflight.py`

```python
"""CPU smoke for scripts/w17_preflight (mock-free, no real NPU).

Each test patches one of the four checks (torch_npu, msmodelslim, weights,
cann_driver) to simulate success/failure; preflight() aggregates correctly.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


@pytest.fixture
def w17_preflight(monkeypatch):
    """Import scripts/w17_preflight as a module."""
    scripts_dir = Path(__file__).resolve().parents[3] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    if "w17_preflight" in sys.modules:
        del sys.modules["w17_preflight"]
    return importlib.import_module("w17_preflight")


def test_preflight_passes_when_all_checks_ok(w17_preflight, monkeypatch):
    monkeypatch.setattr(w17_preflight, "_have_torch_npu", lambda: (True, ""))
    monkeypatch.setattr(w17_preflight, "_have_msmodelslim", lambda: (True, ""))
    monkeypatch.setattr(w17_preflight, "_have_weights", lambda: (True, ""))
    monkeypatch.setattr(w17_preflight, "_have_cann_driver", lambda: (True, ""))
    ok, failures = w17_preflight.preflight()
    assert ok is True
    assert failures == []


def test_preflight_fails_on_missing_torch_npu(w17_preflight, monkeypatch):
    monkeypatch.setattr(w17_preflight, "_have_torch_npu",
                        lambda: (False, "pip install torch_npu==2.9.0"))
    monkeypatch.setattr(w17_preflight, "_have_msmodelslim", lambda: (True, ""))
    monkeypatch.setattr(w17_preflight, "_have_weights", lambda: (True, ""))
    monkeypatch.setattr(w17_preflight, "_have_cann_driver", lambda: (True, ""))
    ok, failures = w17_preflight.preflight()
    assert ok is False
    assert any("torch_npu" in f for f in failures)


def test_preflight_fails_on_missing_msmodelslim(w17_preflight, monkeypatch):
    monkeypatch.setattr(w17_preflight, "_have_torch_npu", lambda: (True, ""))
    monkeypatch.setattr(w17_preflight, "_have_msmodelslim",
                        lambda: (False, "pip install worldforge[quant]"))
    monkeypatch.setattr(w17_preflight, "_have_weights", lambda: (True, ""))
    monkeypatch.setattr(w17_preflight, "_have_cann_driver", lambda: (True, ""))
    ok, failures = w17_preflight.preflight()
    assert ok is False
    assert any("msmodelslim" in f for f in failures)


def test_preflight_fails_on_missing_weights_dir(w17_preflight, monkeypatch):
    monkeypatch.setattr(w17_preflight, "_have_torch_npu", lambda: (True, ""))
    monkeypatch.setattr(w17_preflight, "_have_msmodelslim", lambda: (True, ""))
    monkeypatch.setattr(w17_preflight, "_have_weights",
                        lambda: (False, "weights dir missing"))
    monkeypatch.setattr(w17_preflight, "_have_cann_driver", lambda: (True, ""))
    ok, failures = w17_preflight.preflight()
    assert ok is False
    assert any("weights" in f for f in failures)


def test_preflight_fails_on_missing_cann_driver(w17_preflight, monkeypatch):
    monkeypatch.setattr(w17_preflight, "_have_torch_npu", lambda: (True, ""))
    monkeypatch.setattr(w17_preflight, "_have_msmodelslim", lambda: (True, ""))
    monkeypatch.setattr(w17_preflight, "_have_weights", lambda: (True, ""))
    monkeypatch.setattr(w17_preflight, "_have_cann_driver",
                        lambda: (False, "CANN driver not mounted"))
    ok, failures = w17_preflight.preflight()
    assert ok is False
    assert any("cann_driver" in f for f in failures)


def test_preflight_aggregates_multiple_failures(w17_preflight, monkeypatch):
    monkeypatch.setattr(w17_preflight, "_have_torch_npu",
                        lambda: (False, "no torch_npu"))
    monkeypatch.setattr(w17_preflight, "_have_msmodelslim",
                        lambda: (False, "no msmodelslim"))
    monkeypatch.setattr(w17_preflight, "_have_weights", lambda: (True, ""))
    monkeypatch.setattr(w17_preflight, "_have_cann_driver", lambda: (True, ""))
    ok, failures = w17_preflight.preflight()
    assert ok is False
    assert len(failures) == 2


def test_have_weights_accepts_valid_dir(w17_preflight, tmp_path):
    weights = tmp_path / "vjepa2"
    weights.mkdir()
    (weights / "config.json").write_text("{}")
    (weights / "model.safetensors").write_bytes(b"\x00")
    import os
    os.environ["WORLDFORGE_QUANT_MODEL"] = str(weights)
    try:
        ok, hint = w17_preflight._have_weights()
        assert ok is True
        assert hint == ""
    finally:
        del os.environ["WORLDFORGE_QUANT_MODEL"]


def test_have_weights_rejects_empty_dir(w17_preflight, tmp_path):
    weights = tmp_path / "empty"
    weights.mkdir()
    ok, hint = w17_preflight._have_weights()
    assert ok is False
    assert "config.json" in hint or "no .safetensors" in hint


def test_have_weights_uses_default_when_env_unset(w17_preflight, monkeypatch):
    monkeypatch.delenv("WORLDFORGE_QUANT_MODEL", raising=False)
    ok, hint = w17_preflight._have_weights()
    assert ok is False
    assert "models/vjepa2-vitl-1.2b" in hint


def test_main_exits_0_on_success(w17_preflight, monkeypatch, capsys):
    monkeypatch.setattr(w17_preflight, "preflight", lambda: (True, []))
    rc = w17_preflight.main()
    assert rc == 0
    captured = capsys.readouterr()
    assert "preflight OK" in captured.out


def test_main_exits_1_on_failure(w17_preflight, monkeypatch, capsys):
    monkeypatch.setattr(w17_preflight, "preflight",
                        lambda: (False, ["[torch_npu] missing"]))
    rc = w17_preflight.main()
    assert rc == 1
    captured = capsys.readouterr()
    assert "preflight FAILED" in captured.err
    assert "torch_npu" in captured.err
```

### Step 1.5: Enrich `W11_W12_QUANT_REPORT.md` template

Edit `/mnt/c/WorkSpace/WorldForge/docs/superpowers/specs/W11_W12_QUANT_REPORT.md` to replace TBD placeholders with explicit bracketed field markers like `[W17-MEASURED: OM size vs bf16 baseline]`. This makes the substitute step unambiguous when `scripts/archive/w17/w17_populate_report.py` runs. Specifically:
- Replace `| OM size vs bf16 baseline | ~50% compression | TBD |` row's TBD with `[W17-MEASURED: OM size]`.
- Replace `| Downstream feature cosine >= 0.98 | yes | TBD |` row's TBD with `[W17-MEASURED: aggregate cosine]`.
- Replace `| Quantize wall time | <30 min on 8xNPU | TBD |` row's TBD with `[W17-MEASURED: quantize wall time]`.
- Replace `- **GO** if cosine >= 0.98 and OM compression ~50%` → `- **[W17-MEASURED: GO/NO-GO verdict]** (cosine ≥ 0.98 → GO; < 0.98 → NO-GO)`.

### Step 1.6: Run pre-flight tests + commit

```bash
git checkout -b phase2/w17-atlas-w8a8-validation
python3 -m pytest worldforge_inf/tests/test_w17_preflight.py -v
# expected: 10 tests pass
chmod +x scripts/archive/w17/w17_atlas_w8a8_eval.sh
git add docs/superpowers/plans/2026-07-30-worldforge-phase2-w17-atlas-w8a8-validation.md \
        scripts/archive/w17/w17_preflight.py \
        scripts/archive/w17/w17_atlas_w8a8_eval.sh \
        worldforge_inf/tests/test_w17_preflight.py \
        docs/superpowers/specs/W11_W12_QUANT_REPORT.md \
        .superpowers/sdd/progress.md
git commit -m "feat(w17): plan + shell driver + pre-flight check (W17 T1)"
```

---

## Task 2 — `scripts/archive/w17/w17_populate_report.py` (CPU-doable)

**Files:**
- Create: `scripts/archive/w17/w17_populate_report.py` (~50 lines)
- Create: `worldforge_inf/tests/test_w17_populate_report.py` (~5 CPU tests)

**Why:** Step 3 of the shell driver calls this. It substitutes measured values from `quant_meta.json` + `bench_results/w17_w8a8_eval.json` into the enriched template's bracketed field markers, producing the final GO/NO-GO report. Doing this with a Python script (not sed) keeps the substitution logic unit-testable.

### Step 2.1: Write `scripts/archive/w17/w17_populate_report.py`

```python
"""Populate docs/superpowers/specs/W11_W12_QUANT_REPORT.md from measured values.

Reads:
  - <output_dir>/quant_meta.json (msmodelslim_version, calibration_samples, etc.)
  - bench_results/w17_w8a8_eval.json (aggregate_cosine, per_sample_cosine, passed)

Substitutes bracketed field markers in the report template and writes the
final report. Determines verdict from cosine >= 0.98.

Usage:
    python scripts/archive/w17/w17_populate_report.py \\
        --quant-meta models/vjepa2-vitl-1.2b-w8a8/quant_meta.json \\
        --eval-report bench_results/w17_w8a8_eval.json \\
        --output docs/superpowers/specs/W11_W12_QUANT_REPORT.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPORT_TEMPLATE = Path("docs/superpowers/specs/W11_W12_QUANT_REPORT.md")
FIELD_OM_SIZE = "[W17-MEASURED: OM size]"
FIELD_QUANT_WALL = "[W17-MEASURED: quantize wall time]"
FIELD_COSINE = "[W17-MEASURED: aggregate cosine]"
FIELD_VERDICT = "[W17-MEASURED: GO/NO-GO verdict]"


def _load_json(path: Path) -> dict:
    if not path.is_file():
        print(f"ERROR: {path} missing", file=sys.stderr)
        sys.exit(2)
    return json.loads(path.read_text())


def _compute_om_size(meta_dir: Path) -> str:
    bf16_bytes = sum(p.stat().st_size for p in meta_dir.glob("*.safetensors"))
    bf16_bytes += sum(p.stat().st_size for p in meta_dir.glob("*.bin"))
    om_files = list(meta_dir.parent.glob("*.om")) if meta_dir.parent.exists() else []
    if not om_files:
        return "TBD (no .om file)"
    om_bytes = sum(p.stat().st_size for p in om_files)
    ratio = om_bytes / max(bf16_bytes, 1)
    return f"{om_bytes / 1e6:.1f}MB OM ({ratio:.1%} of {bf16_bytes / 1e6:.1f}MB bf16)"


def populate(quant_meta_path: Path, eval_report_path: Path,
             template_path: Path, output_path: Path) -> None:
    meta = _load_json(quant_meta_path)
    eval_report = _load_json(eval_report_path)
    template = template_path.read_text()

    om_size = _compute_om_size(quant_meta_path.parent)
    cosine = eval_report["aggregate_cosine"]
    verdict = "GO" if eval_report["passed"] else "NO-GO"

    populated = template
    populated = populated.replace(FIELD_OM_SIZE, om_size)
    populated = populated.replace(FIELD_QUANT_WALL,
                                   meta.get("quantize_wall_seconds", "TBD"))
    populated = populated.replace(FIELD_COSINE, f"{cosine:.4f}")
    populated = populated.replace(FIELD_VERDICT,
                                   f"**{verdict}** (cosine {cosine:.4f} "
                                   f"vs threshold {eval_report['threshold']:.2f})")

    output_path.write_text(populated)
    print(f"Wrote {output_path}: verdict={verdict}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--quant-meta", type=Path, required=True)
    p.add_argument("--eval-report", type=Path, required=True)
    p.add_argument("--template", type=Path, default=REPORT_TEMPLATE)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    populate(args.quant_meta, args.eval_report, args.template, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### Step 2.2: Write tests

```python
"""CPU tests for scripts/archive/w17/w17_populate_report.py (no NPU required)."""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def populate_module():
    scripts_dir = Path(__file__).resolve().parents[3] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    if "w17_populate_report" in sys.modules:
        del sys.modules["w17_populate_report"]
    return importlib.import_module("w17_populate_report")


def _write_meta(tmp_path, *, msmodelslim_version="26.1.0",
                calibration_samples=512):
    out = tmp_path / "quant"
    out.mkdir()
    (out / "config.json").write_text("{}")
    (out / "model.safetensors").write_bytes(b"\x00" * 1000)
    (out / "quant_meta.json").write_text(json.dumps({
        "msmodelslim_version": msmodelslim_version,
        "calibration_samples": calibration_samples,
    }))
    return out / "quant_meta.json"


def _write_eval(tmp_path, *, aggregate_cosine=0.985, passed=True, threshold=0.98):
    p = tmp_path / "eval.json"
    p.write_text(json.dumps({
        "aggregate_cosine": aggregate_cosine,
        "per_sample_cosine": [aggregate_cosine],
        "passed": passed,
        "threshold": threshold,
        "n_samples": 1,
    }))
    return p


def _write_template(tmp_path, populate_module):
    t = tmp_path / "tpl.md"
    t.write_text(
        "OM size: " + populate_module.FIELD_OM_SIZE + "\n"
        "Wall: " + populate_module.FIELD_QUANT_WALL + "\n"
        "Cosine: " + populate_module.FIELD_COSINE + "\n"
        "Verdict: " + populate_module.FIELD_VERDICT + "\n"
    )
    return t


def test_populate_substitutes_all_fields(populate_module, tmp_path):
    meta = _write_meta(tmp_path)
    eval_p = _write_eval(tmp_path, aggregate_cosine=0.985)
    tpl = _write_template(tmp_path, populate_module)
    out = tmp_path / "out.md"
    populate_module.populate(meta, eval_p, tpl, out)
    text = out.read_text()
    assert "[W17-MEASURED" not in text
    assert "0.9850" in text
    assert "GO" in text


def test_populate_writes_no_go_when_below_threshold(populate_module, tmp_path):
    meta = _write_meta(tmp_path)
    eval_p = _write_eval(tmp_path, aggregate_cosine=0.96, passed=False)
    tpl = _write_template(tmp_path, populate_module)
    out = tmp_path / "out.md"
    populate_module.populate(meta, eval_p, tpl, out)
    text = out.read_text()
    assert "NO-GO" in text
    assert "0.9600" in text


def test_populate_exits_2_on_missing_meta(populate_module, tmp_path):
    eval_p = _write_eval(tmp_path)
    tpl = _write_template(tmp_path, populate_module)
    out = tmp_path / "out.md"
    with pytest.raises(SystemExit) as exc_info:
        populate_module.populate(tmp_path / "nope.json", eval_p, tpl, out)
    assert exc_info.value.code == 2


def test_populate_exits_2_on_missing_eval(populate_module, tmp_path):
    meta = _write_meta(tmp_path)
    tpl = _write_template(tmp_path, populate_module)
    out = tmp_path / "out.md"
    with pytest.raises(SystemExit) as exc_info:
        populate_module.populate(meta, tmp_path / "nope.json", tpl, out)
    assert exc_info.value.code == 2


def test_compute_om_size_handles_missing_om(populate_module, tmp_path):
    meta = _write_meta(tmp_path)  # no .om files
    size = populate_module._compute_om_size(meta.parent)
    assert "TBD" in size or "no .om" in size.lower()
```

### Step 2.3: Run tests + commit

```bash
python3 -m pytest worldforge_inf/tests/test_w17_preflight.py \
                   worldforge_inf/tests/test_w17_populate_report.py -v
# expected: 10 + 5 = 15 tests pass
git add scripts/archive/w17/w17_populate_report.py worldforge_inf/tests/test_w17_populate_report.py
git commit -m "feat(w17): populate_report.py + tests (W17 T2)"
```

---

## Task 3 — Atlas hardware bring-up (operator run, blocked on Atlas access)

**Hardware required.** Cannot be executed on WSL2 dev host.

**Operator steps:**

- [ ] SSH into Atlas 800 A2 / 910B × NPU host.
- [ ] Verify CANN 8.5.1: `cat /usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/include/version.h | grep ASCEND`
- [ ] Verify torch_npu: `python -c "import torch_npu; print(torch_npu.npu.is_available(), torch_npu.npu.device_count())"` — expect `(True, 8)` (or `(True, 1)` for single-NPU fast path).
- [ ] Verify msmodelslim: `python -c "import msmodelslim; print(msmodelslim.__version__)"` — expect `26.1.0` (matches the `[quant]` pin).
- [ ] Clone the W17 branch to the Atlas host (or pull + cherry-pick the T1 + T2 commits).
- [ ] Install the `[quant]` extra: `pip install -e ".[quant]"` — this pulls msmodelslim==26.1.0 from the pinned index.
- [ ] Download V-JEPA 2 vitl-1.2b weights:
      `huggingface-cli download facebook/vjepa2-vitl-1.2b --local-dir models/vjepa2-vitl-1.2b`
      (~2.4GB safetensors).
- [ ] Run pre-flight: `python scripts/archive/w17/w17_preflight.py` — must print `[W17] preflight OK`.

---

## Task 4 — Run W8A8 quantization on real V-JEPA 2 (operator run)

**Operator steps:**

- [ ] `WORLDFORGE_NO_DOCKER=1 bash scripts/archive/w17/w17_atlas_w8a8_eval.sh` (assumes NPUs already mounted; if not, omit the env var and let the Docker wrapper mount them).
- [ ] Watch `bench_results/w17-quantize.log` — should show msModelSlim calibration + quantize over ~5-30 min.
- [ ] Confirm `models/vjepa2-vitl-1.2b-w8a8/` contains:
      - `config.json` (from bf16)
      - `*.safetensors` (from bf16, frozen base)
      - `quant_meta.json` (W11-W12 sidecar with `msmodelslim_version`, `calibration_samples`, `quant_type=w8a8`)
      - `*.om` (Ascend offline model, msModelSlim output)
- [ ] Record `quantize_wall_seconds` from log into the report (or add to `quant_meta.json` sidecar if the operator wants richer metadata).

---

## Task 5 — Run cosine eval (operator run)

**Operator steps:**

- [ ] Re-run the shell driver OR run the eval step standalone:
      `python -c "from pathlib import Path; from worldforge_inf.eval.quant_eval import cosine_similarity; r = cosine_similarity(Path('models/vjepa2-vitl-1.2b'), Path('models/vjepa2-vitl-1.2b-w8a8'), threshold=0.98); print(r.aggregate_cosine, r.passed)"`
- [ ] Confirm `bench_results/w17_w8a8_eval.json` is written with `aggregate_cosine`, `per_sample_cosine`, `passed`.
- [ ] Note: the eval requires loading the quantized model. On NPU, `transformers.AutoModel.from_pretrained(...)` returns a torch NPU model. The cosine eval already runs on whatever device the model is loaded onto; no explicit `.npu()` needed.

---

## Task 6 — Run NPU-gated integration test (operator run, optional but recommended)

**Operator steps:**

- [ ] `WORLDFORGE_RUN_QUANT_E2E=1 WORLDFORGE_QUANT_TEST_MODEL=models/vjepa2-vitl-1.2b python -m pytest worldforge_inf/tests/test_quant_integration_npu.py -v`
- [ ] Confirm `test_w8a8_quantization_end_to_end` passes with `cosine >= 0.95` (the test uses a more permissive threshold; production target is 0.98 from the report).

---

## Task 7 — Final report populate + commit (operator run, last mile)

**Operator steps:**

- [ ] `bash scripts/archive/w17/w17_atlas_w8a8_eval.sh` (re-run from the top; the driver now reaches Step 3 and invokes `scripts/archive/w17/w17_populate_report.py`).
- [ ] `git diff docs/superpowers/specs/W11_W12_QUANT_REPORT.md` — should show measured values replacing all `[W17-MEASURED: ...]` markers.
- [ ] If the verdict is GO: commit and open a merge PR.
- [ ] If the verdict is NO-GO: edit the report's "## Verdict" section with the failure analysis (which per-sample cosine dropped? what's the quantization error concentration?). The plan opens a follow-up sub-project (W17b: tune calibration size or fall back to a different backend) — defer that decision.

---

## Self-Review

### Spec coverage

| Requirement | Task |
|---|---|
| Atlas hardware validation of W11-W12 W8A8 | T1-T7 (T3-T7 operator-side) |
| Plan doc with operator instructions | T1.1 |
| Shell driver with pre-flight + quantize + eval + report | T1.3, T2.1 |
| Pre-flight env check (NPU + msmodelslim + weights + driver) | T1.2 |
| Cosine ≥ 0.98 gate | T5 (existing `quant_eval.py`) |
| OM size measurement | T2.1 `_compute_om_size` |
| Populated report (no TBD left) | T1.5 (template enrichment) + T7 (populate) |
| Mock-free CPU tests for new modules | T1.4 (10), T2.2 (5) |
| G3 boundary clean (no out-of-scope file mods) | T1.6 commit diff |
| Multi-rank FSDP deferred (not in W17 scope) | Out of scope (W18 concern) |
| Multi-rank training eval on quantized model | Out of scope (W18 concern) |

### Caveats

- **Atlas access not available on dev host.** T1-T2 are the only CPU-doable pieces. T3-T7 require real Atlas hardware; the operator (the user, when they SSH into the Atlas host) runs them. The plan + shell driver + pre-flight + populate script are designed so the operator has zero ambiguity about what to run and what to expect.
- **NPU Docker mount assumed.** The shell driver's Docker wrapper mirrors `w9_eval.sh`'s pattern. Operators who already have NPUs mounted (e.g., direct PCIe install, or non-Ascend NPU boxes) set `WORLDFORGE_NO_DOCKER=1`. The plan documents both paths.
- **`quant_meta.json` doesn't currently record `quantize_wall_seconds`.** W8A8Backend can be extended to add this field; that's a one-line modification in `msmodelslim.py:113`. NOT included in this plan per G3 — but noted as a follow-up if the operator wants richer metrics.
- **Single-NPU fast path vs 8×NPU.** The cosine gate only needs 1 NPU for inference (eval is single-device). Quantization also runs on a single NPU in msModelSlim (it shards calibration data internally; not multi-rank). So 8×NPU is a deployment choice, not a correctness requirement. Operators on a 1-NPU box can run the full validation.
- **`test_quant_integration_npu.py` uses cosine ≥ 0.95.** The W17 spec target is 0.98. The test threshold is intentionally more permissive (CI smoke); the production gate is in `W11_W12_QUANT_REPORT.md`. Do not change the test threshold.
- **Per-sample cosine is preserved in the JSON report.** This lets the post-mortem on a NO-GO verdict see which samples dropped. Not actionable by code; just nice to have for debugging.
- **`scripts/archive/w17/w17_populate_report.py` requires `[W17-MEASURED: ...]` markers in the template.** T1.5 explicitly adds these; without them the substitution is a no-op (output == template) and the report still has TBD. Markers are the contract.
- **The populate script's `_compute_om_size` searches for `.om` files in the parent dir of `quant_meta.json`.** This matches the W8A8Backend layout where `output_path/quant_meta.json` and `output_path/*.om` are siblings. If msModelSlim's output layout changes (e.g., puts `.om` in a subdir), the heuristic breaks. Acceptable for v1; document in the plan's caveats section.
- **`docs/superpowers/specs/W11_W12_QUANT_REPORT.md` filename is preserved.** It's the spec doc, not a per-run report. Populating it in place keeps a single source of truth and matches the W11-W12 T8 commit pattern. If the operator wants per-run snapshots, they can `git tag w17-run-<date>` before re-populating.
- **First W17 commit is T1+T2 together (~15 new CPU tests).** T3-T7 are operator-side and have no commits on the dev host until the operator runs them.

### Validation matrix

| Failure mode | Behavior | Caught by |
|---|---|---|
| `torch_npu` not installed | preflight exits 1 with `pip install torch_npu==2.9.0` | `test_preflight_fails_on_missing_torch_npu` |
| `msmodelslim` not installed | preflight exits 1 with `pip install worldforge[quant]` | `test_preflight_fails_on_missing_msmodelslim` |
| V-JEPA 2 weights missing | preflight exits 1 with `huggingface-cli download ...` | `test_preflight_fails_on_missing_weights_dir` |
| CANN driver not mounted | preflight exits 1 | `test_preflight_fails_on_missing_cann_driver` |
| Multiple missing deps | preflight lists all of them | `test_preflight_aggregates_multiple_failures` |
| Weights dir lacks config.json | `_have_weights` returns False | `test_have_weights_rejects_empty_dir` |
| Eval JSON missing | `populate_report` exits 2 | `test_populate_exits_2_on_missing_eval` |
| Cosine below 0.98 (on Atlas) | report verdict = NO-GO | `test_populate_writes_no_go_when_below_threshold` |
| `.om` file not produced by msModelSlim | report OM size = "TBD (no .om file)" | `test_compute_om_size_handles_missing_om` |
| Operator runs driver without `WORLDFORGE_NO_DOCKER` | Docker wrapper mounts 8 NPU devices | Driver bash logic |
| Operator runs driver with `WORLDFORGE_NO_DOCKER=1` | Direct python invocation, expects mounted NPUs | Driver bash logic |
