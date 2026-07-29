"""W8A8Backend tests — mock msmodelslim so tests run on any host.

msModelSlim is an Ascend-only wheel and is NOT installed on the dev host, so
``msmodelslim.py`` guards its import with try/except and exposes
``_MSMODELSLIM_AVAILABLE``. These tests inject a MagicMock into
``sys.modules`` and ``importlib.reload()`` the backend module so the guarded
top-level import picks up the mock. The real NPU path is exercised by the
Task 7 integration test.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _reload_with_mock_msmodelslim(fake_msmodelslim):
    """Reload the backend module with ``msmodelslim`` mocked in sys.modules."""
    from worldforge_inf.compile.quantize_backends import msmodelslim as ms_mod

    with patch.dict(sys.modules, {"msmodelslim": fake_msmodelslim}):
        importlib.reload(ms_mod)
    return ms_mod


def _make_fake_msmodelslim():
    fake = MagicMock()
    fake.__version__ = "26.1.0"  # matches the pin in pyproject.toml [quant]
    fake_slim = MagicMock()
    fake_slim.quantize.return_value = MagicMock()
    fake.MSModelSlim.return_value = fake_slim
    return fake


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

    fake_msmodelslim = _make_fake_msmodelslim()
    fake_auto_model = MagicMock()
    fake_auto_model.from_pretrained.return_value = MagicMock()

    ms_mod = _reload_with_mock_msmodelslim(fake_msmodelslim)
    assert ms_mod._MSMODELSLIM_AVAILABLE is True

    with patch.object(ms_mod, "_load_calibration_dataset", return_value=fake_samples):
        with patch.object(ms_mod, "AutoModel", fake_auto_model):
            backend = ms_mod.W8A8Backend()
            result = backend.quantize(src, out, calibration_dataset="videomix22m-512")

    assert result == out
    meta_path = out / "quant_meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["quant_type"] == "w8a8"
    assert meta["calibration_dataset"] == "videomix22m-512"
    assert meta["backend"] == "msmodelslim"
    assert meta["phase"] == "phase2-w11"
    assert meta["msmodelslim_version"] == "26.1.0"
    # Phase-1 contract fields preserved
    assert "source_model_path" in meta
    assert meta["source_model_path"] == str(src)


def test_w8a8_backend_missing_config_raises(tmp_path):
    """Backend raises FileNotFoundError when source dir has no config.json."""
    src = tmp_path / "src_model"
    src.mkdir()
    # no config.json
    out = tmp_path / "quantized"

    ms_mod = _reload_with_mock_msmodelslim(_make_fake_msmodelslim())
    backend = ms_mod.W8A8Backend()
    with pytest.raises(FileNotFoundError, match="missing config.json"):
        backend.quantize(src, out, calibration_dataset=None)


def test_w8a8_backend_raises_actionable_error_when_msmodelslim_absent(tmp_path):
    """Without msModelSlim installed, quantize() raises an install hint (G4)."""
    src = tmp_path / "src_model"
    src.mkdir()
    (src / "config.json").write_text('{"model_type": "vjepa2"}')
    out = tmp_path / "quantized"

    from worldforge_inf.compile.quantize_backends import msmodelslim as ms_mod

    with patch.object(ms_mod, "_MSMODELSLIM_AVAILABLE", False):
        backend = ms_mod.W8A8Backend()
        with pytest.raises(ImportError, match=r"pip install worldforge\[quant\]"):
            backend.quantize(src, out, calibration_dataset=None)


def test_w4a8_raises_with_deferral_message():
    """backend_available('w4a8') returns (False, hint) noting W4A8 deferred."""
    from worldforge_inf.compile.quantize_backends import backend_available

    # w4a8 is intentionally not registered in W11 (deferred to a later phase).
    available, hint = backend_available("w4a8")
    assert available is False
    assert hint is not None
    assert "w4a8" in hint
