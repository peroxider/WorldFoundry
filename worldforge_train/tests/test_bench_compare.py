import json
from pathlib import Path

import pytest


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload))
    return path


def _good_payload(clips_per_s: float = 10.0, loss: float = 0.5) -> dict:
    return {
        "profile": "train",
        "tag": "x",
        "world_size": 1,
        "warmup_steps": 2,
        "measurement_steps": 8,
        "clips_per_s": clips_per_s,
        "loss_curve": [loss] * 8,
    }


def test_all_pass_writes_go(tmp_path):
    from worldforge_train.bench.compare import compare
    base = _write_json(tmp_path / "base.json", _good_payload(10.0))
    ev = _write_json(tmp_path / "eval.json", _good_payload(8.0))  # ratio 0.8 >= 0.7
    out = tmp_path / "report.md"
    verdict = compare(base, ev, out)
    assert verdict == "GO"
    assert "**结论**：GO" in out.read_text()


def test_throughput_fail_writes_nogo(tmp_path):
    from worldforge_train.bench.compare import compare
    base = _write_json(tmp_path / "base.json", _good_payload(10.0))
    ev = _write_json(tmp_path / "eval.json", _good_payload(5.0))  # ratio 0.5 < 0.7
    out = tmp_path / "report.md"
    verdict = compare(base, ev, out)
    assert verdict == "NO-GO"
    body = out.read_text()
    assert "**结论**：NO-GO" in body
    assert "ratio=0.50" in body


def test_loss_curve_atol_boundary(tmp_path):
    """Loss curve max abs diff == 0.005 passes (atol=1e-2); == 0.02 fails."""
    from worldforge_train.bench.compare import compare
    base = _write_json(tmp_path / "base.json", _good_payload(loss=0.5))
    near = _write_json(tmp_path / "near.json", {**_good_payload(loss=0.5), "loss_curve": [0.505] * 8})
    out_near = tmp_path / "near.md"
    assert compare(base, near, out_near) == "GO"

    far = _write_json(tmp_path / "far.json", {**_good_payload(loss=0.5), "loss_curve": [0.52] * 8})
    out_far = tmp_path / "far.md"
    assert compare(base, far, out_far) == "NO-GO"