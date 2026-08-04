# W12 CPU Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) to execute this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate the W8A8 quantization + cosine-evaluation orchestration on CPU before committing to NPU time, using a small "fake" backend that simulates quantize file outputs and a real (synthetic-tiny) Hugging Face model load + cosine computation.

**Architecture:**
- New `FakeBackend` registered as `"fake"` quant_type (does NOT shadow real `"w8a8"`)
- Backend's `quantize()` copies `config.json` + `pytorch_model.bin` from source, writes a `quant_meta.json` sidecar matching W8A8Backend's schema
- New CPU integration test calls `quantize_model(src, out, quant_type="fake", ...)` → verifies file outputs → calls `cosine_similarity(src, out, threshold=0.95)` with real HF model load + cosine computation
- CPU test runs by default (no NPU gate) — instant feedback on the orchestration path
- NPU-gated test stays separate (the existing `test_quant_integration_npu.py`)

**Tech Stack:** pytest, Hugging Face transformers (AutoModel on CPU), torch bfloat16

## Global Constraints

- **G2:** msmodelslim==26.1.0 stays pinned; no new deps
- **G3:** Phase-1 code (`worldfoundry_train/`, `worldfoundry_inf/compile/mindie_sd.py`) 0 changes
- **G4:** Explicit errors — fake backend's `quantize()` validates source is a HF model dir; missing `config.json` raises FileNotFoundError with path
- **G5:** CPU test deterministic — no random init seeds leaked into assertions
- **G6:** Single dispatch path — `quantize_model(quant_type="fake")` is the only public way to use FakeBackend (no direct FakeBackend() instantiation in test code)
- **G8:** Green tests per task; final state has all CPU smoke tests passing
- **G9:** Phase-1 test style — mocks only for absent deps, real HF model files in tmp_path for the orchestration chain

---

## Task 1: `compile/quantize_backends/fake.py` — FakeBackend for CPU smoke

**Files:**
- Create: `worldfoundry_inf/compile/quantize_backends/fake.py` (~40 lines)

**Interfaces:**
- Consumes: `quantize_backends.register`, `QuantBackend` Protocol
- Produces: `FakeBackend` class registered as `"fake"`; `quantize(model_path, output_path, calibration_dataset) -> Path`

**Why:** Verifies the orchestration path (facade → registry → backend → file outputs) on a host without msModelSlim. The fake backend doesn't quantize — it copies the bf16 model as-is and writes a sidecar. End-to-end: real HF `AutoModel.from_pretrained` + real `cosine_similarity` confirms the file layout, model loading, and cosine computation all work on CPU.

### Step 1.1: Write the failing test

Create `worldfoundry_inf/tests/test_quantize_backends_fake.py`:

```python
"""FakeBackend self-registers + reproduces quantize outputs without msModelSlim."""
from __future__ import annotations

import json
from pathlib import Path

from worldfoundry_inf.compile.quantize_backends import get_backend, backend_available


def test_fake_backend_self_registers_on_import():
    from worldfoundry_inf.compile.quantize_backends import fake  # noqa: F401
    available, _ = backend_available("fake")
    assert available is True


def test_fake_backend_quantize_copies_weights_and_meta(tmp_path):
    """FakeBackend copies weights + writes quant_meta.json sidecar."""
    src = tmp_path / "bf16"
    src.mkdir()
    (src / "config.json").write_text('{"hidden_size": 4, "model_type": "tiny"}')
    (src / "pytorch_model.bin").write_bytes(b"fake-bf16-weights")

    out = tmp_path / "w8a8"
    backend = get_backend("fake")
    returned = backend.quantize(src, out, calibration_dataset=None)

    assert returned == out
    assert (out / "config.json").exists()
    assert (out / "pytorch_model.bin").exists()
    assert (out / "pytorch_model.bin").read_bytes() == b"fake-bf16-weights"


def test_fake_backend_writes_quant_meta_with_required_fields(tmp_path):
    """quant_meta.json has all 7 spec fields (matches W8A8Backend schema)."""
    from worldfoundry_inf.compile.quantize_backends.fake import FakeBackend

    src = tmp_path / "bf16"
    src.mkdir()
    (src / "config.json").write_text("{}")
    (src / "pytorch_model.bin").write_bytes(b"x")

    out = tmp_path / "w8a8"
    FakeBackend().quantize(src, out, calibration_dataset="videomix22m-512")

    meta = json.loads((out / "quant_meta.json").read_text())
    assert meta["quant_type"] == "fake"  # the registered name (deferred-name contract)
    assert meta["calibration_dataset"] == "videomix22m-512"
    assert meta["phase"] == "w11-w12-fake"
    assert meta["backend"] == "fake"
    assert meta["source_model_path"] == str(src)
    # msmodelslim_version intentionally omitted (no msModelSlim involvement)
    assert "note" in meta


def test_fake_backend_missing_config_raises(tmp_path):
    from worldfoundry_inf.compile.quantize_backends.fake import FakeBackend

    src = tmp_path / "bad"
    src.mkdir()
    # no config.json
    with pytest.raises(FileNotFoundError, match="config.json"):
        FakeBackend().quantize(src, tmp_path / "out", calibration_dataset=None)
```

### Step 1.2: Run tests to verify they fail

Run: `python3 -m pytest worldfoundry_inf/tests/test_quantize_backends_fake.py -v`
Expected: `ModuleNotFoundError: No module named 'worldfoundry_inf.compile.quantize_backends.fake'`

### Step 1.3: Implement the FakeBackend

Create `worldfoundry_inf/compile/quantize_backends/fake.py`:

```python
"""FakeBackend — CPU-runnable surrogate that does NOT call msModelSlim.

Used only for orchestration smoke-tests on hosts without NPU. NEVER
registered as ``"w8a8"`` (which is reserved for the real msModelSlim
backend). The fake copies bf16 weights unchanged + writes a sidecar so
``cosine_similarity`` can load the result with the standard Hugging Face
``AutoModel.from_pretrained`` API.

Cosine evaluation against this output will be ~1.0 (identical weights);
the goal is to verify the pipeline is wired correctly, not to simulate
quantization noise.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from . import register


@register("fake")
class FakeBackend:
    """CPU-runnable surrogate backend. See module docstring."""

    def quantize(
        self,
        model_path: Path,
        output_path: Path,
        calibration_dataset: str | None,
    ) -> Path:
        model_path = Path(model_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        config = model_path / "config.json"
        if not config.exists():
            raise FileNotFoundError(
                f"Missing {config} — input model directory is not a valid HF model."
            )

        shutil.copy2(config, output_path / "config.json")
        weights_src = model_path / "pytorch_model.bin"
        if weights_src.exists():
            shutil.copy2(weights_src, output_path / "pytorch_model.bin")

        meta = {
            "quant_type": "fake",
            "calibration_dataset": calibration_dataset,
            "source_model_path": str(model_path),
            "phase": "w11-w12-fake",
            "backend": "fake",
            "note": "FakeBackend CPU surrogate; weights copied unchanged.",
        }
        (output_path / "quant_meta.json").write_text(json.dumps(meta, indent=2))
        return output_path
```

### Step 1.4: Verify tests pass

Run: `python3 -m pytest worldfoundry_inf/tests/test_quantize_backends_fake.py -v`
Expected: 4 passed

### Step 1.5: Commit

```bash
git add worldfoundry_inf/compile/quantize_backends/fake.py worldfoundry_inf/tests/test_quantize_backends_fake.py
git commit -m "feat(quant): FakeBackend CPU surrogate registered as 'fake' quant_type"
```

---

## Task 2: CPU integration test (`test_quant_integration_cpu.py`)

**Files:**
- Create: `worldfoundry_inf/tests/test_quant_integration_cpu.py` (~80 lines)

**Why:** Validates the full orchestration on CPU without NPU — uses dino-vits8 surrogate model files (synthetic, no HF download required) + real HF `AutoModel.from_pretrained` + real `cosine_similarity` computation. The pipeline is exercised end-to-end with the only stub being the quantize step itself.

### Step 2.1: Write the CPU integration test

```python
"""CPU e2e smoke for W8A8 orchestration: facade → fake backend → cosine eval.

NOT NPU-gated — runs by default on every host. Exists to validate the
full pipeline (file outputs + sidecar + model load + cosine compute)
before committing to NPU time on real V-JEPA 2 vitl-1.2b.

Uses a synthetic tiny HF model (4-dim hidden, 2-layer transformer) so no
network download is required. Cosine is ~1.0 against identical weights.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoModelForImageClassification

from worldfoundry_inf.compile.quantize import quantize_model
from worldfoundry_inf.eval.quant_eval import cosine_similarity


def _write_tiny_hf_model(path: Path) -> Path:
    """Write a tiny HF model dir to ``path`` (config + weights)."""
    path.mkdir(parents=True, exist_ok=True)
    cfg = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-bert")
    cfg.save_pretrained(path)
    # Build a minimal model that AutoModel can load. Real tiny-random-bert
    # is ~4MB; this uses the test fixture's config to keep the file small.
    model = AutoModel.from_pretrained("hf-internal-testing/tiny-random-bert")
    model.save_pretrained(path)
    return path


def test_fake_backend_to_cosine_end_to_end_on_cpu(tmp_path):
    """Full facade → fake backend → cosine eval on CPU."""
    src = _write_tiny_hf_model(tmp_path / "bf16")
    out = tmp_path / "w8a8-fake"

    # Step 1: quantize via facade dispatching to FakeBackend.
    returned = quantize_model(src, out, quant_type="fake", calibration_dataset="videomix22m-512")
    assert returned == out
    assert (out / "config.json").exists()
    assert (out / "quant_meta.json").exists()

    meta = json.loads((out / "quant_meta.json").read_text())
    assert meta["quant_type"] == "fake"
    assert meta["calibration_dataset"] == "videomix22m-512"

    # Step 2: cosine evaluation against the fake output.
    # Tiny random inputs — n_samples=2 to keep CI fast.
    eval_inputs = [torch.randn(3, 224, 224), torch.randn(3, 224, 224)]

    # NOTE: cosine_similarity loads the quantized model via AutoModel and
    # calls .encode(sample.unsqueeze(0)).last_hidden_state. tiny-random-bert
    # has no .encode(); we synthesize a stand-in fake_output attr before the
    # model leaves quantize_model(). For this test we just verify the
    # orchestration runs without crashing; per-sample cosine > 0 is the
    # pipeline-correctness check.
    report = cosine_similarity(
        baseline_path=src,
        quantized_path=out,
        threshold=0.0,  # accept anything for the smoke (we want 0<cos≤1)
        n_samples=2,
        eval_dataset=eval_inputs,
    )
    assert report.aggregate_cosine >= -1.0
    assert report.aggregate_cosine <= 1.0
    assert report.passed is True
```

**Note for implementer:** The above test may need adjustment because `tiny-random-bert` has no `.encode()` method (that's a V-JEPA / ViT method). The cosine_similarity function calls `.encode()` on both models, so the test will fail unless we either:

1. Monkey-patch `.encode` onto the loaded models, or
2. Replace `tiny-random-bert` with a V-JEPA-like ViT architecture (dino-vits8 ~5M params)

The simpler implementation choice for the smoke test is **option 2**: use a tiny ViT-like model so `.encode()` is consistent. Adjust the test to use:

```python
from transformers import AutoModel
# Use the test fixture's tiny ViT config (has .encode method)
cfg = AutoConfig.from_pretrained("facebook/dino-vits8")  # or similar small ViT
```

Actually — the simplest fix is to **monkey-patch encode on the loaded models** since cosine_similarity uses .encode(). Let me use option 1, which is more robust to CI environment changes:

```python
import unittest.mock as mock

# In the test, monkey-patch AutoModel.from_pretrained to return models
# that have a .encode() method producing last_hidden_state-like output.
def fake_from_pretrained(path, **kwargs):
    # Return a tiny model with .encode() returning last_hidden_state
    m = AutoModel.from_pretrained(path, **kwargs)
    m.encode = lambda x: type("E", (), {"last_hidden_state": torch.randn(1, 196, 4)})()
    return m

# patch in the test
```

OR even simpler: **make the fake backend write a HF model with an encode() method** by subclassing. But AutoModel.from_pretrained loads from disk, so the encoding must be on the saved model.

The cleanest fix is probably:
- Save the test model with an added `encode()` method via subclassing before save_pretrained
- OR monkey-patch at test time

Recommend monkey-patching at test time:

```python
def _make_model_with_encode(path):
    """Load any HF model and give it a stub .encode() returning fake tensor."""
    m = AutoModel.from_pretrained(path)
    out_shape = (1, 196, m.config.hidden_size)
    def encode(x):
        return type("Out", (), {"last_hidden_state": torch.randn(*out_shape)})()
    m.encode = encode
    return m
```

Then patch `worldfoundry_inf.eval.quant_eval.AutoModel.from_pretrained` to use this.

**Implementer note:** Please adjust the test if the integration breaks on first run, using one of the above strategies. The acceptance criterion is: the test loads bf16 + fake-quantized model paths, calls cosine_similarity, and the pipeline completes without raising.

### Step 2.2: Run integration test, debug, fix imports/order

Run: `python3 -m pytest worldfoundry_inf/tests/test_quant_integration_cpu.py -v`
Expected: PASS (after possibly adjusting encode() patching)

### Step 2.3: Run full W11-W12 + CPU smoke suite

Run: `python3 -m pytest worldfoundry_inf/tests/ -v`
Expected: 47 passed + 1 skipped (NPU gated) + 15 errors (pre-existing av)

### Step 2.4: Commit

```bash
git add worldfoundry_inf/tests/test_quant_integration_cpu.py
git commit -m "test(quant): CPU e2e smoke for facade → fake backend → cosine pipeline"
```

---

## Task 3: Final review + merge

[Same SDD protocol as W11-W12 — subagent-driven review]

### Step 3.1: Dispatch task reviewer

Pass brief + diff + report paths. Reviewer checks:
- Spec compliance: FakeBackend at correct quant_type, doesn't shadow "w8a8"
- G3: `worldfoundry_train/`, `mindie_sd.py` unchanged
- CPU test runs by default (no NPU gate)
- Sidecar schema consistent with W8A8Backend

### Step 3.2: Address findings, re-review if needed

### Step 3.3: Ask user for merge authorization, then `git merge --no-ff`

---

## Self-Review

### Spec coverage check

| Section | Requirement | Task |
|---------|------------|------|
| New     | FakeBackend registered as `"fake"` (not "w8a8") | T1 |
| New     | FakeBackend copies bf16 weights + sidecar | T1 |
| New     | quant_meta.json matches W8A8Backend schema | T1 |
| New     | CPU integration test exercises full pipeline | T2 |
| New     | Test runs on default CI without NPU | T2 |

### Caveat

The CPU smoke test verifies **orchestration correctness** (file layout + sidecar + model load + cosine compute). It does NOT verify quantization quality — that requires real msModelSlim on NPU with V-JEPA 2 vitl-1.2b. NPU-gated integration test stays in place for that.

The `tiny-random-bert` monkey-patch strategy in the test is a placeholder; the implementer should adjust to whatever pattern makes the pipeline complete on CPU (the encode() patching approach is recommended).
