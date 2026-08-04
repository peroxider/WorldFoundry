# Phase-2 W11-W12 — V-JEPA 2 W8A8 Quantization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Phase-1 quantize stub with real W8A8 static quantization via msModelSlim, plus end-to-end cosine ≥ 0.98 evaluation harness.

**Architecture:** Sub-package split — `compile/quantize_backends/msmodelslim.py` (backend), `data/calib_videomix22m.py` (calibration), `eval/quant_eval.py` (cosine ≥ 0.98 gate). `compile/quantize.py` becomes a thin facade that dispatches to the registered backend by `quant_type`. Phase-1 `quantize_model()` API surface preserved verbatim.

**Tech Stack:** Python 3.11, msModelSlim (Huawei official Ascend quant tool), transformers 4.55.0, torch 2.9.0 + torch_npu 2.9.0, datasets 3.6.0, pytest 8.x.

**Spec:** `docs/superpowers/specs/2026-07-29-worldforge-phase2-w11-w12-quantization-design.md`

## Global Constraints

- **G1**: CANN 8.5.1 + torch_npu 2.9.0 locked (already pinned in env) — only matters for integration test (Task 7)
- **G2**: New deps pin exact version — `msmodelslim==X.Y.Z` (version discovered in Task 1, exact pin)
- **G3**: Phase-1 code 0 changes EXCEPT internal dispatch body of `quantize_model` — signature `quantize_model(model_path, output_path, quant_type="w8a8", calibration_dataset=None) -> Path` preserved verbatim; existing 2 tests in `test_quantize.py` stay green without modification
- **G4**: No silent defaults; explicit errors — `ImportError` with `pip install worldforge[quant]` hint when msModelSlim missing; `ValueError` on unknown `quant_type`/`quant_type="w4a8"` (deferred); shape mismatch raises
- **G5**: Reproducibility — calibration seed=42 (fixed in `load_calibration_dataset`); eval seed=123 (fixed in CLI default)
- **G6**: Single source of truth — `quantize_backends/__init__.py:get_backend(quant_type)` is the only dispatch path
- **G7**: Async-friendly — N/A (no I/O concurrency in this sub-project)
- **G8**: Each task ends with green test + own commit
- **G9**: Tests follow Phase-1 style — pytest + `tmp_path` fixture + minimal mocking (no real NPU / no real V-JEPA 2 weights in unit tests)

---

## Task 1: pyproject.toml `[quant]` extra + version discovery

**Files:**
- Modify: `pyproject.toml` (lines 29-33, the `[project.optional-dependencies]` block)
- Create: `.superpowers/sdd/w11-w12-msmodelslim-version.txt` (version pin record, gitignored scratch)

**Why this task first:** msModelSlim version is a runtime decision (latest stable). We pin once and reuse the same string in Task 3 (backend import) + Task 7 (integration test). Discover the version BEFORE writing any backend code so the import strings match.

**Interfaces:**
- Produces: `msmodelslim_version` string (e.g., `"0.1.0"` or whatever is current) recorded in scratch file for downstream tasks

- [ ] **Step 1: Discover current msModelSlim version**

```bash
pip index versions msmodelslim 2>&1 | head -5
# OR (if pip index not available):
pip install msmodelslim== 2>&1 | grep -oE 'from versions: \([^)]*\)' | head -1
```

If neither yields a usable version (msModelSlim may not be on public PyPI — it ships via Huawei昇腾 community channels), document the fallback: use `msmodelslim>=0.1.0,<0.2` with a `note` in the scratch file, AND require manual install instructions in `pyproject.toml` comment.

Expected: a version string like `0.1.0` or a documented fallback policy.

- [ ] **Step 2: Write version to scratch file**

Write to `/mnt/c/WorkSpace/WorldForge/.superpowers/sdd/w11-w12-msmodelslim-version.txt`:

```
msmodelslim_version=<discovered_version>
install_source=<pypi OR huawei-ascend-community>
fallback_policy=<if version discovery failed>
```

- [ ] **Step 3: Add `[quant]` extra to pyproject.toml**

In `pyproject.toml` lines 29-33 (`[project.optional-dependencies]` block), append:

```toml
quant = ["msmodelslim==<VERSION_FROM_SCRATCH_FILE>"]
```

Example if version is `0.1.0`:
```toml
quant = ["msmodelslim==0.1.0"]
```

If fallback applies (no public PyPI), use:
```toml
# msModelSlim is distributed via Huawei昇腾 community channels, not public PyPI.
# Install manually: pip install msmodelslim (after adding Huawei昇腾 pip index).
quant = ["msmodelslim>=0.1.0"]
```

- [ ] **Step 4: Verify pyproject.toml parses**

Run: `python3 -c "import tomllib; tomllib.load(open('pyproject.toml','rb'))"`
Expected: no exception, exit 0.

- [ ] **Step 5: Verify Phase-1 tests still pass (G3 baseline)**

Run: `cd /mnt/c/WorkSpace/WorldForge && python3 -m pytest worldforge_inf/tests/test_quantize.py -v`
Expected: 2 passed (test_quantize_model_creates_output_file, test_quantize_model_rejects_unknown_type). These still work because the stub body is unchanged in this task.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml
git commit -m "feat(quant): add [quant] extra with pinned msmodelslim"
```

---

## Task 2: `compile/quantize_backends/` package + registry

**Files:**
- Create: `worldforge_inf/compile/quantize_backends/__init__.py`
- Create: `worldforge_inf/tests/test_quantize_backends_registry.py`

**Why before backend:** the registry must exist before `W8A8Backend` can register itself, and before `quantize.py` facade can dispatch to it. Test-first per TDD discipline.

**Interfaces:**
- Produces:
  - `class QuantBackend(Protocol)` with `quantize(model_path: Path, output_path: Path, calibration_dataset: str | None) -> Path`
  - `def register(quant_type: str) -> Callable[[type[QuantBackend]], type[QuantBackend]]`
  - `def get_backend(quant_type: str) -> QuantBackend` (raises `ValueError` if not registered)
  - `def backend_available(quant_type: str) -> tuple[bool, str | None]` (returns `(False, hint)` if msModelSlim import failed)

- [ ] **Step 1: Write the failing test**

Create `worldforge_inf/tests/test_quantize_backends_registry.py`:

```python
"""Registry + factory tests for quantize_backends subpackage."""
from __future__ import annotations

import pytest

from worldforge_inf.compile.quantize_backends import (
    QuantBackend,
    backend_available,
    get_backend,
    register,
)


def test_register_and_get_backend_roundtrip():
    """A class decorated with @register('foo') is retrievable via get_backend('foo')."""
    @register("foo")
    class FakeBackend:
        def quantize(self, model_path, output_path, calibration_dataset):
            return output_path
    backend = get_backend("foo")
    assert isinstance(backend, FakeBackend)


def test_get_backend_unknown_raises_value_error():
    """get_backend('nope') raises ValueError with available types listed."""
    with pytest.raises(ValueError, match="No backend registered for quant_type='nope'"):
        get_backend("nope")


def test_backend_available_true_when_registered():
    """backend_available returns (True, None) for a registered type."""
    @register("bar")
    class BarBackend:
        def quantize(self, model_path, output_path, calibration_dataset):
            return output_path
    available, hint = backend_available("bar")
    assert available is True
    assert hint is None


def test_protocol_quantize_signature():
    """QuantBackend Protocol declares the expected quantize signature."""
    # Static check via runtime introspection
    import inspect
    sig = inspect.signature(QuantBackend.quantize)
    params = list(sig.parameters.keys())
    assert params == ["self", "model_path", "output_path", "calibration_dataset"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/c/WorkSpace/WorldForge && python3 -m pytest worldforge_inf/tests/test_quantize_backends_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'worldforge_inf.compile.quantize_backends'`.

- [ ] **Step 3: Write minimal implementation**

Create `worldforge_inf/compile/quantize_backends/__init__.py`:

```python
"""Backend registry for quantization implementations.

Subclasses register themselves via the @register decorator; get_backend()
is the only dispatch path used by worldforge_inf.compile.quantize.

W11 only ships the W8A8 backend (msModelSlim). Future W4A8 backend can
land here without disturbing the facade.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Protocol


class QuantBackend(Protocol):
    """Contract every quantization backend must satisfy."""

    def quantize(
        self,
        model_path: Path,
        output_path: Path,
        calibration_dataset: str | None,
    ) -> Path: ...


_REGISTRY: dict[str, type[QuantBackend]] = {}


def register(quant_type: str) -> Callable[[type[QuantBackend]], type[QuantBackend]]:
    """Class decorator that registers a backend under ``quant_type``."""

    def deco(cls: type[QuantBackend]) -> type[QuantBackend]:
        _REGISTRY[quant_type] = cls
        return cls

    return deco


def get_backend(quant_type: str) -> QuantBackend:
    """Return a fresh instance of the backend registered under ``quant_type``.

    Raises:
        ValueError: no backend is registered for ``quant_type``.
    """
    if quant_type not in _REGISTRY:
        raise ValueError(
            f"No backend registered for quant_type={quant_type!r}. "
            f"Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[quant_type]()


_BACKEND_IMPORT_ERROR: Exception | None = None


def backend_available(quant_type: str) -> tuple[bool, str | None]:
    """Return ``(available, hint)`` for ``quant_type``.

    ``hint`` is non-None when the backend is unavailable due to a missing
    dependency (e.g. msModelSlim not installed). The facade uses this to
    emit an ImportError with an actionable install instruction (G4).
    """
    if _BACKEND_IMPORT_ERROR is not None:
        return False, (
            "W8A8 quantization requires msmodelslim. "
            "Install with: pip install worldforge[quant]\n"
            f"Underlying error: {_BACKEND_IMPORT_ERROR}"
        )
    if quant_type not in _REGISTRY:
        return False, (
            f"No backend registered for quant_type={quant_type!r}. "
            f"Available: {sorted(_REGISTRY)}"
        )
    return True, None


# Trigger backend module imports so they self-register. msModelSlim backend
# (Task 3) imports msmodelslim at module top; if missing, capture the error
# so backend_available() can surface it later.
try:
    from . import msmodelslim  # noqa: F401  (W8A8Backend self-registers via @register)
except ImportError as exc:
    _BACKEND_IMPORT_ERROR = exc
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /mnt/c/WorkSpace/WorldForge && python3 -m pytest worldforge_inf/tests/test_quantize_backends_registry.py -v`
Expected: 4 passed (test_register_and_get_backend_roundtrip, test_get_backend_unknown_raises_value_error, test_backend_available_true_when_registered, test_protocol_quantize_signature).

- [ ] **Step 5: Commit**

```bash
git add worldforge_inf/compile/quantize_backends/__init__.py worldforge_inf/tests/test_quantize_backends_registry.py
git commit -m "feat(quant): backend registry with Protocol contract + factory"
```

---

## Task 3: `compile/quantize_backends/msmodelslim.py` — W8A8Backend (mocked in unit tests)

**Files:**
- Create: `worldforge_inf/compile/quantize_backends/msmodelslim.py`
- Create: `worldforge_inf/tests/test_quantize_backends_msmodelslim.py`

**Why:** This is the core deliverable. The test mocks `msmodelslim` so the unit tests run without NPU; integration test (Task 7) hits the real path on NPU.

**Interfaces:**
- Produces: `class W8A8Backend` registered as `"w8a8"`, with method `quantize(model_path, output_path, calibration_dataset) -> Path` that writes `quant_meta.json` to `output_path`.

- [ ] **Step 1: Write the failing tests**

Create `worldforge_inf/tests/test_quantize_backends_msmodelslim.py`:

```python
"""W8A8Backend tests — mock msmodelslim so tests run on any host."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_w8a8_backend_self_registers_on_import():
    """Importing msmodelslim.py registers 'w8a8' in the registry."""
    from worldforge_inf.compile.quantize_backends import _REGISTRY
    assert "w8a8" in _REGISTRY


def test_w8a8_backend_quantize_writes_meta(tmp_path):
    """W8A8Backend.quantize() writes quant_meta.json with the spec's fields."""
    src = tmp_path / "src_model"
    src.mkdir()
    (src / "config.json").write_text('{"model_type": "vjepa2"}')
    out = tmp_path / "quantized"
    out.mkdir()

    # Mock the calibration loader to avoid HF datasets network in unit tests.
    fake_samples = [MagicMock() for _ in range(4)]

    # Mock msmodelslim + transformers.AutoModel inside the backend module.
    fake_msmodelslim = MagicMock()
    fake_msmodelslim.__version__ = "0.1.0"
    fake_slim = MagicMock()
    fake_quantized = MagicMock()
    fake_slim.quantize.return_value = fake_quantized
    fake_msmodelslim.MSModelSlim.return_value = fake_slim

    fake_auto_model = MagicMock()
    fake_model_instance = MagicMock()
    fake_auto_model.from_pretrained.return_value = fake_model_instance

    with patch.dict(sys.modules, {
        "msmodelslim": fake_msmodelslim,
    }):
        # Reimport backend module so its top-level msmodelslim import picks
        # up the mock.
        import importlib
        from worldforge_inf.compile import quantize_backends
        from worldforge_inf.compile.quantize_backends import msmodelslim as ms_mod
        importlib.reload(ms_mod)
        importlib.reload(quantize_backends)

        # Patch calibration loader + AutoModel after reload
        with patch.object(ms_mod, "load_calibration_dataset", return_value=fake_samples):
            with patch.object(ms_mod, "AutoModel", fake_auto_model):
                backend_class = quantize_backends._REGISTRY["w8a8"]
                backend = backend_class()
                result = backend.quantize(src, out, calibration_dataset="videomix22m-512")

    assert result == out
    meta_path = out / "quant_meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["quant_type"] == "w8a8"
    assert meta["calibration_dataset"] == "videomix22m-512"
    assert meta["backend"] == "msmodelslim"
    assert meta["phase"] == "phase2-w11"
    # Phase-1 contract fields preserved
    assert "source_model_path" in meta


def test_w8a8_backend_missing_config_raises(tmp_path):
    """Backend raises FileNotFoundError when source dir has no config.json."""
    src = tmp_path / "src_model"
    src.mkdir()
    # no config.json
    out = tmp_path / "quantized"

    fake_msmodelslim = MagicMock()
    fake_msmodelslim.__version__ = "0.1.0"

    with patch.dict(sys.modules, {"msmodelslim": fake_msmodelslim}):
        import importlib
        from worldforge_inf.compile import quantize_backends
        from worldforge_inf.compile.quantize_backends import msmodelslim as ms_mod
        importlib.reload(ms_mod)
        importlib.reload(quantize_backends)
        backend_class = quantize_backends._REGISTRY["w8a8"]
        backend = backend_class()
        with pytest.raises(FileNotFoundError, match="missing config.json"):
            backend.quantize(src, out, calibration_dataset=None)


def test_w4a8_raises_with_deferral_message():
    """backend_available('w4a8') returns (False, hint) noting W4A8 deferred."""
    from worldforge_inf.compile.quantize_backends import backend_available
    # Clear registry for this test (w4a8 not registered, by design)
    available, hint = backend_available("w4a8")
    assert available is False
    assert hint is not None
    assert "w4a8" in hint
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /mnt/c/WorkSpace/WorldForge && python3 -m pytest worldforge_inf/tests/test_quantize_backends_msmodelslim.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'worldforge_inf.compile.quantize_backends.msmodelslim'`.

- [ ] **Step 3: Write minimal implementation**

Create `worldforge_inf/compile/quantize_backends/msmodelslim.py`:

```python
"""W8A8 static quantization via Huawei msModelSlim.

This module imports `msmodelslim` at top-level. If msModelSlim is not
installed, the ImportError is captured by the parent package's
``_BACKEND_IMPORT_ERROR`` and surfaced by ``backend_available()`` as
an actionable install hint (G4).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModel

import msmodelslim

from worldforge_inf.compile.quantize_backends import register
from worldforge_inf.data.calib_videomix22m import load_calibration_dataset


@register("w8a8")
class W8A8Backend:
    """W8A8 symmetric per-channel-weight / per-tensor-activation quant."""

    def quantize(
        self,
        model_path: Path,
        output_path: Path,
        calibration_dataset: str | None,
    ) -> Path:
        model_path = Path(model_path)
        output_path = Path(output_path)

        if not (model_path / "config.json").exists():
            raise FileNotFoundError(
                f"Model dir {model_path} missing config.json"
            )

        # Resolve calibration samples
        if calibration_dataset is None or calibration_dataset == "videomix22m-512":
            samples = list(load_calibration_dataset(n=512, seed=42))
        else:
            raise ValueError(
                f"Unknown calibration_dataset={calibration_dataset!r}. "
                f"Supported: None | 'videomix22m-512'"
            )

        # Load V-JEPA 2 ViT in bf16
        model = AutoModel.from_pretrained(
            str(model_path), torch_dtype=torch.bfloat16
        )

        # msModelSlim quantization
        slim = msmodelslim.MSModelSlim(model)
        quantized_model = slim.quantize(
            quant_type="W8A8",
            quant_mode="static",
            calibration_data=samples,
            calibration_size=len(samples),
        )

        # Persist quantized model + meta sidecar
        output_path.mkdir(parents=True, exist_ok=True)
        quantized_model.save_pretrained(str(output_path))
        meta = {
            "quant_type": "w8a8",
            "calibration_dataset": calibration_dataset or "videomix22m-512",
            "source_model_path": str(model_path),
            "phase": "phase2-w11",
            "backend": "msmodelslim",
            "msmodelslim_version": msmodelslim.__version__,
            "note": "W8A8 static quantization via msModelSlim.",
        }
        (output_path / "quant_meta.json").write_text(json.dumps(meta, indent=2))
        return output_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /mnt/c/WorkSpace/WorldForge && python3 -m pytest worldforge_inf/tests/test_quantize_backends_msmodelslim.py -v`
Expected: 4 passed.

Note: If `msmodelslim` is NOT installed on the dev host (likely), the test `test_w8a8_backend_quantize_writes_meta` may still need adjustment because the module-level `import msmodelslim` in `msmodelslim.py` will fail BEFORE the test can mock it. If that happens, restructure the test to:
1. Install `msmodelslim` via `pip install worldforge[quant]` (which adds it to the env), OR
2. Skip the failing test in CI and only run it on NPU hosts (mark with `@pytest.mark.skipif(not _msmodelslim_installed, ...)`)

Document whichever path was taken in the test docstring.

- [ ] **Step 5: Verify Phase-1 tests still green (G3)**

Run: `cd /mnt/c/WorkSpace/WorldForge && python3 -m pytest worldforge_inf/tests/test_quantize.py -v`
Expected: 2 passed. The `quantize.py` facade hasn't been changed yet — these still pass against the stub body.

- [ ] **Step 6: Commit**

```bash
git add worldforge_inf/compile/quantize_backends/msmodelslim.py worldforge_inf/tests/test_quantize_backends_msmodelslim.py
git commit -m "feat(quant): W8A8Backend via msModelSlim with quant_meta.json sidecar"
```

---

## Task 4: `data/calib_videomix22m.py` — calibration data loader

**Files:**
- Create: `worldforge_inf/data/__init__.py` (if not exists; check first)
- Create: `worldforge_inf/data/calib_videomix22m.py`
- Create: `worldforge_inf/tests/test_calib_videomix22m.py`

**Interfaces:**
- Produces: `def load_calibration_dataset(n: int = 512, seed: int = 42, image_processor=None) -> Iterable[torch.Tensor]` yielding `n` tensors of shape `(3, 224, 224)`.

- [ ] **Step 1: Verify `worldforge_inf/data/` does NOT exist yet**

Run: `ls /mnt/c/WorkSpace/WorldForge/worldforge_inf/data/ 2>&1`
Expected: `No such file or directory` (this sub-project creates the dir fresh).

If it does exist (someone added files), STOP and ask the user — this is unexpected per the spec.

- [ ] **Step 2: Write the failing test**

Create `worldforge_inf/tests/test_calib_videomix22m.py`:

```python
"""calib_videomix22m loader tests — mock HF datasets to avoid network in CI."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch


def _make_fake_dataset(n: int):
    """Return a MagicMock standing in for a HF Dataset with `n` samples."""
    samples = [{"video": [MagicMock() for _ in range(16)]} for _ in range(n)]
    fake = MagicMock()
    fake.shuffle.return_value.select.return_value = samples
    return fake, samples


def test_load_returns_n_tensors():
    """load_calibration_dataset(n=8) yields exactly 8 tensors."""
    from worldforge_inf.data.calib_videomix22m import load_calibration_dataset

    fake_ds, _ = _make_fake_dataset(8)

    # Mock processor to return deterministic (3, 224, 224) tensors
    def fake_processor(images, return_tensors):
        return {"pixel_values": torch.zeros(1, 3, 224, 224)}

    with patch(
        "worldforge_inf.data.calib_videomix22m.load_dataset", return_value=fake_ds
    ):
        with patch(
            "worldforge_inf.data.calib_videomix22m.AutoImageProcessor.from_pretrained"
        ) as fake_proc_factory:
            fake_proc_factory.return_value = MagicMock(
                __call__=fake_processor
            )
            tensors = list(load_calibration_dataset(n=8, seed=42))

    assert len(tensors) == 8
    assert all(t.shape == (3, 224, 224) for t in tensors)


def test_load_propagates_dataset_error():
    """HF datasets errors propagate (no silent fallback — G4)."""
    from worldforge_inf.data.calib_videomix22m import load_calibration_dataset

    with patch(
        "worldforge_inf.data.calib_videomix22m.load_dataset",
        side_effect=RuntimeError("network unreachable"),
    ):
        with pytest.raises(RuntimeError, match="network unreachable"):
            list(load_calibration_dataset(n=4, seed=42))
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd /mnt/c/WorkSpace/WorldForge && python3 -m pytest worldforge_inf/tests/test_calib_videomix22m.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'worldforge_inf.data.calib_videomix22m'`.

- [ ] **Step 4: Write minimal implementation**

Create `worldforge_inf/data/__init__.py`:

```python
"""Data loading utilities for WorldForge inference path."""
```

Create `worldforge_inf/data/calib_videomix22m.py`:

```python
"""Load a fixed 512-sample calibration subset from VideoMix22M.

Used by W8A8Backend to drive msModelSlim's static calibration. Yields
tensors preprocessed by the V-JEPA 2 image processor.

G5 reproducibility: seed=42 is the only seed value (callers should not
override without explicit reason).
"""
from __future__ import annotations

from typing import Iterable

import torch
from datasets import load_dataset
from transformers import AutoImageProcessor

# V-JEPA 2 vitl-1.2b default processor (matches parent spec §4.4)
_DEFAULT_PROCESSOR_MODEL = "facebook/vjepa2-vitl-1.2b"


def load_calibration_dataset(
    n: int = 512,
    seed: int = 42,
    image_processor=None,
) -> Iterable[torch.Tensor]:
    """Yield ``n`` preprocessed (3, 224, 224) tensors from VideoMix22M.

    Args:
        n: Number of calibration samples to yield.
        seed: Random seed for the shuffle (G5 reproducibility).
        image_processor: Optional pre-loaded processor; defaults to
            V-JEPA 2 vitl-1.2b's processor.

    Yields:
        torch.Tensor of shape (3, 224, 224), dtype float32.

    Raises:
        Whatever `datasets.load_dataset` raises on network/auth failures
        (no silent fallback — G4).
    """
    processor = image_processor or AutoImageProcessor.from_pretrained(
        _DEFAULT_PROCESSOR_MODEL
    )
    ds = load_dataset("videomix22m", split="train", streaming=True)
    ds = ds.shuffle(seed=seed).select(range(n))
    for sample in ds:
        frames = sample["video"]  # list of PIL.Image, T frames
        tensor = processor(images=frames, return_tensors="pt")["pixel_values"]
        yield tensor[0]  # (3, 224, 224)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /mnt/c/WorkSpace/WorldForge && python3 -m pytest worldforge_inf/tests/test_calib_videomix22m.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add worldforge_inf/data/__init__.py worldforge_inf/data/calib_videomix22m.py worldforge_inf/tests/test_calib_videomix22m.py
git commit -m "feat(quant): VideoMix22M 512-sample calibration loader"
```

---

## Task 5: `eval/quant_eval.py` — cosine similarity library + happy path

**Files:**
- Create: `worldforge_inf/eval/__init__.py`
- Create: `worldforge_inf/eval/quant_eval.py`
- Create: `worldforge_inf/tests/test_quant_eval.py`

**Interfaces:**
- Produces:
  - `@dataclass class EvalReport(aggregate_cosine, per_sample_cosine, passed, threshold, n_samples)`
  - `def cosine_similarity(baseline_path, quantized_path, eval_dataset=None, threshold=0.98, n_samples=100, seed=123) -> EvalReport`

- [ ] **Step 1: Write the failing tests (happy path + below-threshold)**

Create `worldforge_inf/tests/test_quant_eval.py`:

```python
"""quant_eval cosine similarity tests — synthetic bf16 vs int8 features."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch


def _make_model(features: torch.Tensor):
    """Mock a model whose .encode(x) returns given features."""
    m = MagicMock()
    m.encode.return_value.last_hidden_state = features
    return m


def test_cosine_above_threshold_passes():
    """cosine=0.99 (above 0.98) → report.passed=True."""
    from worldforge_inf.eval.quant_eval import cosine_similarity

    bf16_feat = torch.randn(1, 4, 8)
    # Make int8 features ~bf16 features (high cosine)
    int8_feat = bf16_feat + torch.randn_like(bf16_feat) * 0.01

    bf16_model = _make_model(bf16_feat)
    int8_model = _make_model(int8_feat)

    with patch(
        "worldforge_inf.eval.quant_eval.AutoModel.from_pretrained",
        side_effect=[bf16_model, int8_model],
    ):
        report = cosine_similarity(
            baseline_path=Path("/fake/bf16"),
            quantized_path=Path("/fake/int8"),
            threshold=0.98,
            n_samples=1,
        )

    assert report.passed is True
    assert report.n_samples == 1
    assert report.aggregate_cosine > 0.98
    assert report.threshold == 0.98


def test_cosine_below_threshold_fails():
    """cosine~0.5 (well below 0.98) → report.passed=False."""
    from worldforge_inf.eval.quant_eval import cosine_similarity

    bf16_feat = torch.randn(1, 4, 8)
    # Make int8 features orthogonal (low cosine)
    int8_feat = -bf16_feat + torch.randn_like(bf16_feat) * 0.1

    bf16_model = _make_model(bf16_feat)
    int8_model = _make_model(int8_feat)

    with patch(
        "worldforge_inf.eval.quant_eval.AutoModel.from_pretrained",
        side_effect=[bf16_model, int8_model],
    ):
        report = cosine_similarity(
            baseline_path=Path("/fake/bf16"),
            quantized_path=Path("/fake/int8"),
            threshold=0.98,
            n_samples=1,
        )

    assert report.passed is False
    assert report.aggregate_cosine < 0.98


def test_aggregate_cosine_is_mean_of_per_sample():
    """aggregate_cosine == mean(per_sample_cosine)."""
    from worldforge_inf.eval.quant_eval import cosine_similarity

    bf16_feats = [torch.randn(1, 4, 8) for _ in range(3)]
    int8_feats = [f + torch.randn_like(f) * 0.01 for f in bf16_feats]

    bf16_model = _make_model(bf16_feats[0])
    int8_model = _make_model(int8_feats[0])

    # Use 3 samples by mocking encode() to return different features each call
    bf16_model.encode.side_effect = [
        MagicMock(last_hidden_state=f) for f in bf16_feats
    ]
    int8_model.encode.side_effect = [
        MagicMock(last_hidden_state=f) for f in int8_feats
    ]

    with patch(
        "worldforge_inf.eval.quant_eval.AutoModel.from_pretrained",
        side_effect=[bf16_model, int8_model],
    ):
        report = cosine_similarity(
            baseline_path=Path("/fake/bf16"),
            quantized_path=Path("/fake/int8"),
            threshold=0.98,
            n_samples=3,
        )

    expected_mean = sum(report.per_sample_cosine) / len(report.per_sample_cosine)
    assert abs(report.aggregate_cosine - expected_mean) < 1e-5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /mnt/c/WorkSpace/WorldForge && python3 -m pytest worldforge_inf/tests/test_quant_eval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'worldforge_inf.eval'`.

- [ ] **Step 3: Write minimal implementation (library only; CLI in Task 6)**

Create `worldforge_inf/eval/__init__.py`:

```python
"""Evaluation harnesses for WorldForge inference path."""
```

Create `worldforge_inf/eval/quant_eval.py`:

```python
"""End-to-end cosine similarity eval: bf16 baseline vs quantized model.

Used as the W11 acceptance gate per parent spec §9 #6:
"下游特征 cosine similarity 与 bf16 baseline ≥ 0.98".

CLI usage (see __main__ block at bottom):
    python -m worldforge_inf.eval.quant_eval \\
        --baseline models/vjepa2-vitl-1.2b-bf16 \\
        --quantized models/vjepa2-vitl-1.2b-w8a8 \\
        --threshold 0.98
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from transformers import AutoModel


@dataclass
class EvalReport:
    """Result of a cosine similarity eval run."""
    aggregate_cosine: float
    per_sample_cosine: list[float] = field(default_factory=list)
    passed: bool = False
    threshold: float = 0.98
    n_samples: int = 0


def cosine_similarity(
    baseline_path: Path,
    quantized_path: Path,
    eval_dataset: Iterable[torch.Tensor] | None = None,
    threshold: float = 0.98,
    n_samples: int = 100,
    seed: int = 123,
) -> EvalReport:
    """Compare bf16 baseline vs quantized model on shared eval samples.

    Args:
        baseline_path: Path to bf16 baseline model directory.
        quantized_path: Path to quantized model directory (must have config.json).
        eval_dataset: Optional iterable of (3, 224, 224) tensors. If None,
            uses the videomix22m 100-sample holdout (seed=123).
        threshold: Pass threshold for aggregate cosine (default 0.98).
        n_samples: Number of eval samples to run (default 100).
        seed: Random seed for the holdout (G5 reproducibility).

    Returns:
        EvalReport with aggregate_cosine, per_sample_cosine, passed.

    Raises:
        ValueError: empty eval dataset, or model output shape mismatch.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    baseline = AutoModel.from_pretrained(str(baseline_path), torch_dtype=torch.bfloat16)
    quantized = AutoModel.from_pretrained(str(quantized_path))

    if eval_dataset is None:
        # Default holdout: same videomix22m source, seed=123, n_samples
        from worldforge_inf.data.calib_videomix22m import load_calibration_dataset
        eval_dataset = list(load_calibration_dataset(n=n_samples, seed=seed))

    samples = list(eval_dataset)
    if not samples:
        raise ValueError(f"eval_dataset is empty (n_samples={n_samples}).")

    per_sample_cosine: list[float] = []
    for sample in samples:
        with torch.no_grad():
            bf16_out = baseline.encode(sample.unsqueeze(0)).last_hidden_state
            int8_out = quantized.encode(sample.unsqueeze(0)).last_hidden_state

        if bf16_out.shape != int8_out.shape:
            raise ValueError(
                f"Shape mismatch: bf16={tuple(bf16_out.shape)} "
                f"vs int8={tuple(int8_out.shape)}. Cannot compute cosine."
            )

        cos = F.cosine_similarity(bf16_out, int8_out, dim=-1).mean().item()
        per_sample_cosine.append(cos)

    aggregate = sum(per_sample_cosine) / len(per_sample_cosine)
    return EvalReport(
        aggregate_cosine=aggregate,
        per_sample_cosine=per_sample_cosine,
        passed=aggregate >= threshold,
        threshold=threshold,
        n_samples=len(per_sample_cosine),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /mnt/c/WorkSpace/WorldForge && python3 -m pytest worldforge_inf/tests/test_quant_eval.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add worldforge_inf/eval/__init__.py worldforge_inf/eval/quant_eval.py worldforge_inf/tests/test_quant_eval.py
git commit -m "feat(quant): cosine similarity eval library + EvalReport"
```

---

## Task 6: `eval/quant_eval.py` — error handling + CLI entry point

**Files:**
- Modify: `worldforge_inf/eval/quant_eval.py` (append `if __name__ == "__main__"` block; the `n_samples <= 0` check is already in Task 5's implementation)
- Modify: `worldforge_inf/tests/test_quant_eval.py` (add shape-mismatch and empty-dataset tests)

**Why separate from Task 5:** Task 5 focused on the happy path + dataclass. Task 6 adds the explicit error paths (G4 compliance) and the CLI entry point.

**Interfaces:**
- Produces: CLI entry `python -m worldforge_inf.eval.quant_eval --baseline ... --quantized ... [--threshold 0.98] [--n-samples 100]` exiting 0 on pass, 1 on fail.

- [ ] **Step 1: Add failing tests for shape mismatch and empty dataset**

Append to `worldforge_inf/tests/test_quant_eval.py`:

```python
def test_shape_mismatch_raises_value_error():
    """Different bf16 vs int8 output shapes → ValueError."""
    from worldforge_inf.eval.quant_eval import cosine_similarity

    bf16_feat = torch.randn(1, 4, 8)
    int8_feat = torch.randn(1, 4, 16)  # different last dim

    bf16_model = _make_model(bf16_feat)
    int8_model = _make_model(int8_feat)

    with patch(
        "worldforge_inf.eval.quant_eval.AutoModel.from_pretrained",
        side_effect=[bf16_model, int8_model],
    ):
        with pytest.raises(ValueError, match="Shape mismatch"):
            cosine_similarity(
                baseline_path=Path("/fake/bf16"),
                quantized_path=Path("/fake/int8"),
                n_samples=1,
            )


def test_empty_eval_dataset_raises():
    """n_samples=0 → ValueError before model load."""
    from worldforge_inf.eval.quant_eval import cosine_similarity

    with pytest.raises(ValueError, match="n_samples must be positive"):
        cosine_similarity(
            baseline_path=Path("/fake/bf16"),
            quantized_path=Path("/fake/int8"),
            n_samples=0,
        )
```

- [ ] **Step 2: Run new tests to verify they pass (already implemented in Task 5)**

Run: `cd /mnt/c/WorkSpace/WorldForge && python3 -m pytest worldforge_inf/tests/test_quant_eval.py -v`
Expected: 5 passed (3 from Task 5 + 2 new).

- [ ] **Step 3: Append CLI block to quant_eval.py**

Append to `worldforge_inf/eval/quant_eval.py`:

```python


def _cli() -> int:
    """CLI entry point. Returns process exit code (0 = pass, 1 = fail)."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Cosine similarity eval: bf16 baseline vs quantized model"
    )
    parser.add_argument("--baseline", required=True, type=Path,
                        help="Path to bf16 baseline model directory")
    parser.add_argument("--quantized", required=True, type=Path,
                        help="Path to quantized model directory")
    parser.add_argument("--threshold", type=float, default=0.98,
                        help="Cosine threshold (default 0.98)")
    parser.add_argument("--n-samples", type=int, default=100,
                        help="Number of eval samples (default 100)")
    args = parser.parse_args()

    try:
        report = cosine_similarity(
            baseline_path=args.baseline,
            quantized_path=args.quantized,
            threshold=args.threshold,
            n_samples=args.n_samples,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2  # setup/config error

    print(
        f"Cosine: {report.aggregate_cosine:.4f} "
        f"(threshold {report.threshold:.4f}, "
        f"n={report.n_samples}) → "
        f"{'PASS' if report.passed else 'FAIL'}"
    )
    return 0 if report.passed else 1


if __name__ == "__main__":
    import sys
    sys.exit(_cli())
```

- [ ] **Step 4: Verify CLI help works**

Run: `cd /mnt/c/WorkSpace/WorldForge && python3 -m worldforge_inf.eval.quant_eval --help`
Expected: prints argparse help text; exit 0.

- [ ] **Step 5: Verify CLI rejects missing args**

Run: `cd /mnt/c/WorkSpace/WorldForge && python3 -m worldforge_inf.eval.quant_eval 2>&1; echo "exit=$?"`
Expected: argparse error + non-zero exit code.

- [ ] **Step 6: Commit**

```bash
git add worldforge_inf/eval/quant_eval.py worldforge_inf/tests/test_quant_eval.py
git commit -m "feat(quant): quant_eval CLI with shape/empty guards"
```

---

## Task 7: `compile/quantize.py` facade — wire up dispatch (G3 boundary)

**Files:**
- Modify: `worldforge_inf/compile/quantize.py` (replace stub body with backend dispatch)
- Create: `worldforge_inf/tests/test_quantize_facade_dispatch.py` (verify facade calls backend)

**Why last:** The facade modification is the G3 boundary. Until now, `quantize.py` has been untouched. This task changes its internal body to dispatch to the registered backend, while preserving the exact public signature and the Phase-1 contract tests.

**Interfaces:**
- Consumes: `from worldforge_inf.compile.quantize_backends import backend_available, get_backend`
- Produces: `quantize_model(model_path, output_path, quant_type="w8a8", calibration_dataset=None) -> Path` with internal dispatch.

- [ ] **Step 1: Write the failing dispatch test**

Create `worldforge_inf/tests/test_quantize_facade_dispatch.py`:

```python
"""Verify quantize_model facade dispatches to the registered backend (G3 + G6)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from worldforge_inf.compile.quantize import quantize_model


def test_facade_dispatches_w8a8_to_backend(tmp_path):
    """quantize_model('w8a8') calls W8A8Backend.quantize with the same args."""
    src = tmp_path / "bf16"
    src.mkdir()
    out = tmp_path / "w8a8"

    fake_backend = MagicMock()
    fake_backend.quantize.return_value = out

    with patch(
        "worldforge_inf.compile.quantize.get_backend", return_value=fake_backend
    ) as mock_get:
        with patch(
            "worldforge_inf.compile.quantize.backend_available",
            return_value=(True, None),
        ):
            result = quantize_model(src, out, quant_type="w8a8",
                                    calibration_dataset="videomix22m-512")

    assert result == out
    mock_get.assert_called_once_with("w8a8")
    fake_backend.quantize.assert_called_once_with(
        src, out, "videomix22m-512"
    )


def test_facade_unknown_quant_type_raises():
    """quantize_model(quant_type='w3a3') raises ValueError (Phase-1 preserved)."""
    with pytest.raises(ValueError, match="Unsupported quant_type"):
        quantize_model(
            Path("/fake"), Path("/fake"),
            quant_type="w3a3",
        )


def test_facade_backend_unavailable_raises_import_error():
    """When backend_available returns (False, hint), facade raises ImportError."""
    src = Path("/fake/bf16")
    out = Path("/fake/w8a8")
    with patch(
        "worldforge_inf.compile.quantize.backend_available",
        return_value=(False, "pip install worldforge[quant]"),
    ):
        with pytest.raises(ImportError, match="pip install worldforge\\[quant\\]"):
            quantize_model(src, out, quant_type="w8a8")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/c/WorkSpace/WorldForge && python3 -m pytest worldforge_inf/tests/test_quantize_facade_dispatch.py -v`
Expected: FAIL (the first test fails because the current `quantize.py` stub does NOT call `get_backend` — it just copies files).

- [ ] **Step 3: Modify `quantize.py` to dispatch (G3 boundary)**

Modify `worldforge_inf/compile/quantize.py`. Replace the existing body that copies files + writes a stub `quant_meta.json` with:

```python
"""Quantization facade — dispatches by quant_type to registered backend.

Phase-1 API surface preserved verbatim (G3). The actual quantization work
lives in worldforge_inf.compile.quantize_backends.*.
"""
from __future__ import annotations

from pathlib import Path

from worldforge_inf.compile.quantize_backends import (
    SUPPORTED_QUANT_TYPES,
    backend_available,
    get_backend,
)

__all__ = ["quantize_model", "SUPPORTED_QUANT_TYPES"]


def quantize_model(
    model_path: Path,
    output_path: Path,
    quant_type: str = "w8a8",
    calibration_dataset: str | None = None,
) -> Path:
    """Quantize a model directory and write the result to ``output_path``.

    Returns the output directory path. Raises ``ValueError`` on unsupported
    ``quant_type``, ``ImportError`` if the backend's dependencies are not
    installed (with an actionable install hint), or any exception propagated
    from the backend (e.g. ``FileNotFoundError`` for missing model files).
    """
    if quant_type not in SUPPORTED_QUANT_TYPES:
        raise ValueError(
            f"Unsupported quant_type={quant_type!r}. "
            f"Supported: {SUPPORTED_QUANT_TYPES}"
        )

    available, hint = backend_available(quant_type)
    if not available:
        raise ImportError(hint)

    backend = get_backend(quant_type)
    return backend.quantize(Path(model_path), Path(output_path), calibration_dataset)
```

- [ ] **Step 4: Add SUPPORTED_QUANT_TYPES export to `quantize_backends/__init__.py`**

Modify `worldforge_inf/compile/quantize_backends/__init__.py` — append at end:

```python


# Public list of quant types the facade accepts. Backends register themselves
# at import time; the facade only dispatches to types listed here.
SUPPORTED_QUANT_TYPES: tuple[str, ...] = ("w8a8", "w4a8")  # w4a8 deferred to future sub-project
```

- [ ] **Step 5: Run new dispatch tests**

Run: `cd /mnt/c/WorkSpace/WorldForge && python3 -m pytest worldforge_inf/tests/test_quantize_facade_dispatch.py -v`
Expected: 3 passed.

- [ ] **Step 6: CRITICAL — Phase-1 tests must still pass (G3 verification)**

Run: `cd /mnt/c/WorkSpace/WorldForge && python3 -m pytest worldforge_inf/tests/test_quantize.py -v`
Expected: 2 passed (Phase-1 tests). If they fail, the facade change is too aggressive — back out and use a gentler dispatch.

NOTE: `test_quantize_model_creates_output_file` may now fail because the real W8A8 backend (if msModelSlim IS installed) would error in the stub test's fake model dir. Two options:
- (a) Mark `test_quantize_model_creates_output_file` with `@pytest.mark.skipif(msmodelslim_available, reason="Real backend now active")` and add a new "stub mode" test for the legacy behavior
- (b) Have the Phase-1 test use a `quant_type` that has no registered backend (e.g. `"legacy-stub"`) — but this requires keeping the stub mode alive

Discuss with the user which path to take before committing this step.

- [ ] **Step 7: Run full test suite to confirm no regressions**

Run: `cd /mnt/c/WorkSpace/WorldForge && python3 -m pytest worldforge_inf/tests worldforge_train/tests -q`
Expected: all previously-passing tests still pass; new tests added in Tasks 2-6 all pass; any av-related errors (pre-existing Phase-1 env limitation) unchanged.

- [ ] **Step 8: Commit**

```bash
git add worldforge_inf/compile/quantize.py worldforge_inf/compile/quantize_backends/__init__.py worldforge_inf/tests/test_quantize_facade_dispatch.py
git commit -m "feat(quant): facade dispatches to registered backend by quant_type"
```

---

## Task 8: Integration test (NPU gated) + final report

**Files:**
- Create: `worldforge_inf/tests/test_quant_integration_npu.py`
- Create: `docs/superpowers/specs/W11_W12_QUANT_REPORT.md` (template; real NPU run fills it in later)

**Why last:** integration test requires real NPU + msModelSlim installed + V-JEPA 2 weights. Default skip on non-NPU hosts. This task also writes the report template + verifies the full test suite end-to-end.

- [ ] **Step 1: Add integration test (default skip)**

Create `worldforge_inf/tests/test_quant_integration_npu.py`:

```python
"""End-to-end W8A8 quantization + cosine eval (NPU-gated, default skip).

Run with: WORLDFORGE_RUN_QUANT_E2E=1 pytest worldforge_inf/tests/test_quant_integration_npu.py

Requires:
- torch_npu installed + NPU available
- msmodelslim installed (worldforge[quant])
- V-JEPA 2 vitl-1.2b weights downloaded to models/vjepa2-vitl-1.2b/

Uses a smaller surrogate ViT (facebook/dino-vits8, ~5M params) for CI;
production W11 acceptance runs against real V-JEPA 2 vitl-1.2b on 8×NPU.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

torch_npu_available = pytest.importorskip("torch_npu").npu.is_available()


@pytest.mark.skipif(
    not torch_npu_available or not os.environ.get("WORLDFORGE_RUN_QUANT_E2E"),
    reason="Requires NPU + WORLDFORGE_RUN_QUANT_E2E=1",
)
def test_w8a8_quantization_end_to_end(tmp_path):
    """End-to-end: load bf16 → W8A8 quant → cosine ≥ 0.95."""
    from worldforge_inf.compile.quantize import quantize_model
    from worldforge_inf.eval.quant_eval import cosine_similarity

    src = Path(os.environ.get("WORLDFORGE_QUANT_TEST_MODEL",
                              "models/dino-vits8"))  # tiny ViT for CI
    out = tmp_path / "w8a8"

    quantize_model(src, out, quant_type="w8a8",
                   calibration_dataset="videomix22m-512")

    assert (out / "quant_meta.json").exists()
    meta = json.loads((out / "quant_meta.json").read_text())
    assert meta["quant_type"] == "w8a8"

    report = cosine_similarity(src, out, threshold=0.95, n_samples=4)
    assert report.passed, f"cosine={report.aggregate_cosine} below 0.95"
```

Add `import json` at top if not present.

- [ ] **Step 2: Verify integration test is correctly skipped on non-NPU**

Run: `cd /mnt/c/WorkSpace/WorldForge && python3 -m pytest worldforge_inf/tests/test_quant_integration_npu.py -v`
Expected: 1 skipped (reason: "Requires NPU + WORLDFORGE_RUN_QUANT_E2E=1").

- [ ] **Step 3: Write the W11_W12_QUANT_REPORT.md template**

Create `docs/superpowers/specs/W11_W12_QUANT_REPORT.md`:

```markdown
# W11-W12 Quantization Report (template)

**Status**: Template — populated after real 8×NPU hardware run.

## Environment

- Hardware: Atlas 800 A2 / 910B × 8 (production target)
- Software: CANN 8.5.1, torch_npu 2.9.0, msmodelslim <version>
- Model: facebook/vjepa2-vitl-1.2b (bf16 baseline)
- Calibration: VideoMix22M 512-sample subset, seed=42
- Eval holdout: VideoMix22M 100-sample subset, seed=123

## Results

| Metric | Target | Measured |
|--------|--------|----------|
| OM size vs bf16 baseline | ~50% compression | TBD |
| Downstream feature cosine ≥ 0.98 | yes | TBD |
| Calibration wall time | <10 min on 8×NPU | TBD |
| Quantize wall time | <30 min on 8×NPU | TBD |

## Verdict

- **GO** if cosine ≥ 0.98 and OM compression ~50%
- **NO-GO** otherwise (deferred to future sub-project)

## Artifacts

- Quantized model dir: `models/vjepa2-vitl-1.2b-w8a8/`
- Eval report JSON: `bench_results/w11_w12_quant_eval.json`
- Logs: `bench_results/w11_w12_quant.log`
```

- [ ] **Step 4: Run full test suite (final regression check)**

Run: `cd /mnt/c/WorkSpace/WorldForge && python3 -m pytest worldforge_inf/tests worldforge_train/tests -q`
Expected: same pass/error counts as the pre-task baseline; new tests added (Task 2-6, 7) all pass; pre-existing av errors unchanged.

- [ ] **Step 5: Commit**

```bash
git add worldforge_inf/tests/test_quant_integration_npu.py docs/superpowers/specs/W11_W12_QUANT_REPORT.md
git commit -m "feat(quant): NPU-gated integration test + report template"
```

---

## Self-Review

### Spec coverage check

| Spec §  | Requirement | Task |
|---------|------------|------|
| §1.2 | W8A8 only | Task 3 (W8A8Backend only) |
| §1.2 | msModelSlim backend | Tasks 1, 3 (pin + use) |
| §1.2 | videomix22m 512-sample calib | Task 4 (loader) |
| §1.2 | cosine ≥ 0.98 eval | Task 5 (library) + Task 6 (CLI) |
| §1.2 | `[quant]` extra | Task 1 (pyproject.toml) |
| §1.2 | Sub-package split | Tasks 2, 3, 4, 5 (3 new subpackages) |
| §2.1 | File structure | Tasks 2-8 (each file in the right place) |
| §3.1 | `quantize.py` facade preserved | Task 7 (G3 verification) |
| §3.2 | Registry + factory | Task 2 |
| §3.3 | W8A8Backend implementation | Task 3 |
| §3.4 | calib_videomix22m loader | Task 4 |
| §3.5 | quant_eval library + CLI | Tasks 5, 6 |
| §3.6 | `[quant]` pyproject extra | Task 1 |
| §4.1 | End-to-end quantize flow | Task 3 (mocked) + Task 8 (real NPU) |
| §4.2 | Eval flow | Task 5 (mocked) + Task 8 (real NPU) |
| §5 | Error handling (all 9 modes) | Tasks 3, 5, 6, 7 |
| §6.1 | 11 unit tests | Tasks 2, 3, 4, 5, 6, 7 = 4+4+2+3+2+3 = 18 test functions across 6 files (more than spec minimum; covers all error paths) |
| §6.2 | 1 integration test (NPU gated) | Task 8 |
| §7 | Acceptance criteria 1-6 | Tasks 1, 3, 5, 7, 8 |
| §8.1 | All 11 deliverables | Tasks 1-8 (verified) |
| §8.2 | msmodelslim pinned exactly | Task 1 |
| §9 | Self-audit | This section |

All spec requirements mapped to tasks. No gaps.

### Placeholder scan

- ✅ No "TBD" in task bodies (version discovery happens in Task 1 step 1; docstring placeholder in Task 7 step 6 is intentional user-decision point, not a missing spec)
- ✅ No "TODO" / "implement later"
- ✅ Every code step shows the actual code

### Type consistency check

- `quantize_model(model_path, output_path, quant_type="w8a8", calibration_dataset=None) -> Path` — used consistently in Tasks 3, 5, 6, 7
- `EvalReport(aggregate_cosine, per_sample_cosine, passed, threshold, n_samples)` — Task 5 defines, Tasks 6, 8 use
- `class W8A8Backend` registered as `"w8a8"` — Task 3 defines, Tasks 2, 7 reference
- `load_calibration_dataset(n=512, seed=42) -> Iterable[torch.Tensor]` — Task 4 defines, Task 3 calls
- `cosine_similarity(baseline_path, quantized_path, eval_dataset=None, threshold=0.98, n_samples=100, seed=123) -> EvalReport` — Task 5 defines, Tasks 6, 8 call
- `quant_meta.json` fields: `{quant_type, calibration_dataset, source_model_path, phase, backend, msmodelslim_version, note}` — Task 3 defines; Phase-1 fields preserved

No type mismatches across tasks.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-29-worldforge-phase2-w11-w12-quantization.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

Per the user's "按照推荐方案继续执行直到完成" goal and the W9-W10 precedent (which used subagent-driven-development), the recommendation is option 1. Proceeding unless you indicate otherwise.