import json
from pathlib import Path



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
    from worldfoundry_train.bench.compare import compare
    base = _write_json(tmp_path / "base.json", _good_payload(10.0))
    ev = _write_json(tmp_path / "eval.json", _good_payload(8.0))  # ratio 0.8 >= 0.7
    out = tmp_path / "report.md"
    verdict = compare(base, ev, out)
    assert verdict == "GO"
    assert "**结论**：GO" in out.read_text()


def test_throughput_fail_writes_nogo(tmp_path):
    from worldfoundry_train.bench.compare import compare
    base = _write_json(tmp_path / "base.json", _good_payload(10.0))
    ev = _write_json(tmp_path / "eval.json", _good_payload(5.0))  # ratio 0.5 < 0.7
    out = tmp_path / "report.md"
    verdict = compare(base, ev, out)
    assert verdict == "NO-GO"
    body = out.read_text()
    assert "**结论**：NO-GO" in body
    assert "ratio=0.50" in body


def test_ratio_just_below_threshold_is_nogo(tmp_path):
    """Raw ratio 0.6999 must NOT round to 0.7000 and pass the gate.

    Rounding before the binary gate turns a NO-GO into a GO (silent default).
    Compute gate on full-precision raw ratio; round only for display.
    """
    from worldfoundry_train.bench.compare import compare
    # base=10.0, eval=6.999 -> raw ratio=0.6999 -> should be NO-GO
    base = _write_json(tmp_path / "base.json", _good_payload(10.0))
    ev = _write_json(tmp_path / "eval.json", _good_payload(6.999))
    out = tmp_path / "report.md"
    verdict = compare(base, ev, out)
    assert verdict == "NO-GO"
    body = out.read_text()
    assert "**结论**：NO-GO" in body


def test_loss_curve_atol_boundary(tmp_path):
    """Loss curve max abs diff == 0.005 passes (atol=1e-2); == 0.02 fails."""
    from worldfoundry_train.bench.compare import compare
    base = _write_json(tmp_path / "base.json", _good_payload(loss=0.5))
    near = _write_json(tmp_path / "near.json", {**_good_payload(loss=0.5), "loss_curve": [0.505] * 8})
    out_near = tmp_path / "near.md"
    assert compare(base, near, out_near) == "GO"

    far = _write_json(tmp_path / "far.json", {**_good_payload(loss=0.5), "loss_curve": [0.52] * 8})
    out_far = tmp_path / "far.md"
    assert compare(base, far, out_far) == "NO-GO"


def test_null_loss_curve_marks_not_evaluable(tmp_path):
    """Malformed input: eval loss_curve = null must NOT silently pass.

    The comparator must mark the condition as not evaluable, list the
    malformed field, and produce a NO-GO report — no traceback.
    """
    from worldfoundry_train.bench.compare import compare
    base = _write_json(tmp_path / "base.json", _good_payload())
    bad = _write_json(tmp_path / "eval.json", {**_good_payload(), "loss_curve": None})
    out = tmp_path / "report.md"
    verdict = compare(base, bad, out)
    assert verdict == "NO-GO"
    body = out.read_text()
    assert "**结论**：NO-GO" in body
    assert "loss_curve" in body  # field flagged in not_evaluable


def test_non_numeric_clips_is_nogo(tmp_path):
    """Malformed input: eval clips_per_s = 'ten' (string) must produce NO-GO,
    not a TypeError traceback.
    """
    from worldfoundry_train.bench.compare import compare
    base = _write_json(tmp_path / "base.json", _good_payload())
    bad = _write_json(tmp_path / "eval.json", {**_good_payload(), "clips_per_s": "ten"})
    out = tmp_path / "report.md"
    verdict = compare(base, bad, out)
    assert verdict == "NO-GO"
    body = out.read_text()
    assert "**结论**：NO-GO" in body


def test_malformed_json_produces_nogo(tmp_path):
    """Malformed JSON file must produce NO-GO report, not a traceback."""
    from worldfoundry_train.bench.compare import compare
    base = _write_json(tmp_path / "base.json", _good_payload())
    bad = tmp_path / "eval.json"
    bad.write_text("{this is not valid json")
    out = tmp_path / "report.md"
    verdict = compare(base, bad, out)
    assert verdict == "NO-GO"
    body = out.read_text()
    assert "**结论**：NO-GO" in body
    assert "无法解析" in body or "Eval JSON" in body or "eval" in body


def test_missing_eval_preserves_stderr_from_log(tmp_path):
    """When eval JSON is missing, the report must surface stderr from
    bench_results/<tag>.log so the W9 evidence includes the failure
    context (e.g., NotImplementedError), not just "file missing".
    """
    from worldfoundry_train.bench.compare import compare
    base = _write_json(tmp_path / "base.json", _good_payload())
    eval_json = tmp_path / "eval.json"  # deliberately NOT created
    log = tmp_path / "eval.log"
    log.write_text(
        "Traceback (most recent call last):\n"
        "  File .../mindspeed.py\", line 14, in wrap\n"
        "    raise NotImplementedError(\"MMStrategy not implemented\")\n"
        "NotImplementedError: MMStrategy not implemented\n"
    )
    out = tmp_path / "report.md"
    verdict = compare(base, eval_json, out)
    assert verdict == "NO-GO"
    body = out.read_text()
    assert "**NO-GO**" in body
    # stderr excerpt must be present in the report
    assert "NotImplementedError" in body
    assert "MMStrategy not implemented" in body
