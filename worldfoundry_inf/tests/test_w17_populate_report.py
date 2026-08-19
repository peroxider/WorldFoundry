"""CPU tests for scripts/w17_populate_report.py (no NPU required)."""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def populate_module():
    scripts_dir = Path(__file__).resolve().parents[2] / "scripts" / "archive" / "w17"
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
