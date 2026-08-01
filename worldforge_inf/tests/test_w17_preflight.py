"""CPU smoke for scripts/w17_preflight (mock-free, no real NPU).

Each test patches one of the four checks (torch_npu, msmodelslim, weights,
cann_driver) to simulate success/failure; preflight() aggregates correctly.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest


@pytest.fixture
def w17_preflight(monkeypatch):
    """Import scripts/w17_preflight as a module."""
    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
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
    os.environ["WORLDFORGE_QUANT_MODEL"] = str(weights)
    try:
        ok, hint = w17_preflight._have_weights()
        assert ok is False
        assert "config.json" in hint or "no .safetensors" in hint
    finally:
        del os.environ["WORLDFORGE_QUANT_MODEL"]


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
