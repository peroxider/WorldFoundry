"""Unit tests for :mod:`worldfoundry_inf.compile.mindie_sd`.

Pure-Python / CPU tests; no Atlas hardware or MindIE-SD needed. The
tests cover the M3-T4 partial-closure dual-track:

* ``is_mindie_sd_available`` / ``is_atc_available`` probes.
* :func:`compile_path` resolution (auto + explicit prefer).
* :func:`compile_to_om` stub path preserves the phase-1 JSON schema
  (``phase == "phase1-stub"``) and adds the new
  ``compile_path`` + ``stub_reason`` keys.
* :func:`compile_to_om` ATC subprocess path runs ``atc`` with the right
  flags (``--model``, ``--framework 5``, ``--output``, ``--soc-version
  Ascend910B4``), writes a real ``.om``, and surfaces ATC non-zero rc
  as :class:`OMCompileError` with the stderr excerpt.
* :func:`compile_to_om` mindie_sd path calls
  ``mindie.sd.tools.compile`` and writes the returned artifact.
* :func:`provider_summary` always returns the diagnostic dict.
"""
from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from worldfoundry_common.exceptions import OMCompileError
from worldfoundry_inf.compile import mindie_sd
from worldfoundry_inf.compile.mindie_sd import compile_to_om


# --------------------------------------------------------------------------- #
# Existing phase-1 stubs (regression)
# --------------------------------------------------------------------------- #


def test_compile_to_om_writes_om_file(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text("{}")
    out = tmp_path / "model.om"
    result = compile_to_om(
        src, out, target_hardware="Atlas 800 A2 / 910B", require_quantized=False,
    )
    assert result.exists()
    assert result.suffix == ".om"


def test_compile_to_om_raises_on_missing_quant_meta(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text("{}")
    out = tmp_path / "model.om"
    with pytest.raises(OMCompileError):
        compile_to_om(src, out, require_quantized=True)


# --------------------------------------------------------------------------- #
# M3-T4 partial closure: probe + dispatch
# --------------------------------------------------------------------------- #


def test_compile_path_returns_stub_when_neither_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mindie_sd, "is_mindie_sd_available", lambda: False)
    monkeypatch.setattr(mindie_sd, "is_atc_available", lambda: False)
    assert mindie_sd.compile_path() == "stub"


def test_compile_path_returns_mindie_sd_when_wheel_importable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mindie_sd, "is_mindie_sd_available", lambda: True)
    monkeypatch.setattr(mindie_sd, "is_atc_available", lambda: False)
    assert mindie_sd.compile_path() == "mindie_sd"


def test_compile_path_returns_atc_when_cli_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mindie_sd, "is_mindie_sd_available", lambda: False)
    monkeypatch.setattr(mindie_sd, "is_atc_available", lambda: True)
    monkeypatch.setattr(mindie_sd.shutil, "which", lambda _: "/fake/path/atc")
    assert mindie_sd.compile_path() == "atc"


def test_compile_path_prefer_mindie_sd_raises_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mindie_sd, "is_mindie_sd_available", lambda: False)
    with pytest.raises(OMCompileError, match="mindie_sd"):
        mindie_sd.compile_path(prefer="mindie_sd")


def test_compile_path_prefer_atc_raises_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mindie_sd, "is_atc_available", lambda: False)
    with pytest.raises(OMCompileError, match="atc CLI"):
        mindie_sd.compile_path(prefer="atc")


def test_compile_path_prefer_stub_always_succeeds() -> None:
    """``prefer='stub'`` is unconditional — it doesn't probe anything."""
    assert mindie_sd.compile_path(prefer="stub") == "stub"


# --------------------------------------------------------------------------- #
# Stub path payload schema (backward compat + additive fields)
# --------------------------------------------------------------------------- #


def test_compile_to_om_stub_preserves_phase1_schema(tmp_path: Path) -> None:
    """Stub keeps ``phase == 'phase1-stub'`` for backward compat."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text("{}")
    (src / "quant_meta.json").write_text("{}")
    out = tmp_path / "model.om"
    result = compile_to_om(src, out)
    payload = json.loads(result.read_text())
    assert payload["phase"] == "phase1-stub"
    assert payload["compile_path"] == "stub"
    assert "stub_reason" in payload
    assert payload["target_hardware"] == "Atlas 800 A2 / 910B"
    assert "note" in payload


def test_compile_to_om_stub_checksum_is_deterministic(tmp_path: Path) -> None:
    """Same source content → same checksum across runs (G5 reproducibility)."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text('{"x": 1}')
    (src / "quant_meta.json").write_text('{"q": "w8a8"}')
    out1 = tmp_path / "model1.om"
    out2 = tmp_path / "model2.om"
    compile_to_om(src, out1)
    compile_to_om(src, out2)
    p1 = json.loads(out1.read_text())
    p2 = json.loads(out2.read_text())
    assert p1["checksum"] == p2["checksum"]
    assert len(p1["checksum"]) == 64  # sha256 hex


# --------------------------------------------------------------------------- #
# ATC subprocess path
# --------------------------------------------------------------------------- #


def test_compile_to_om_via_atc_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ATC path: subprocess.run with correct flags + writes real .om."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text("{}")
    (src / "quant_meta.json").write_text("{}")
    (src / "model.onnx").write_bytes(b"\x08\x07" + b"fake-onnx")
    out = tmp_path / "model.om"

    monkeypatch.setattr(mindie_sd, "is_mindie_sd_available", lambda: False)
    monkeypatch.setattr(mindie_sd, "is_atc_available", lambda: True)
    monkeypatch.setattr(mindie_sd.shutil, "which", lambda _: "/fake/path/atc")

    captured: dict = {}

    def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        # atc writes the artifact at cmd[--output + 1]
        out_idx = cmd.index("--output") + 1
        Path(cmd[out_idx]).write_bytes(b"FAKE_OM_BINARY")
        return subprocess.CompletedProcess(cmd, 0, stdout="OK", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = compile_to_om(src, out, prefer="atc")
    assert result == out
    assert captured["cmd"][0].endswith("atc")
    assert captured["cmd"][1] == "--model"
    assert captured["cmd"][3] == "--framework"
    assert captured["cmd"][5] == "--output"
    assert captured["cmd"][7] == "--soc-version"
    assert captured["cmd"][8] == "Ascend910B4"
    assert out.read_bytes() == b"FAKE_OM_BINARY"


def test_compile_to_om_atc_missing_onnx_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ATC path requires ``model.onnx`` in the source directory."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text("{}")
    (src / "quant_meta.json").write_text("{}")
    out = tmp_path / "model.om"
    monkeypatch.setattr(mindie_sd, "is_mindie_sd_available", lambda: False)
    monkeypatch.setattr(mindie_sd, "is_atc_available", lambda: True)
    monkeypatch.setattr(mindie_sd.shutil, "which", lambda _: "/fake/path/atc")
    with pytest.raises(OMCompileError, match="model.onnx"):
        compile_to_om(src, out, prefer="atc")


def test_compile_to_om_atc_nonzero_rc_raises_with_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ATC non-zero rc → OMCompileError with stderr excerpt."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text("{}")
    (src / "quant_meta.json").write_text("{}")
    (src / "model.onnx").write_bytes(b"fake")
    out = tmp_path / "model.om"
    monkeypatch.setattr(mindie_sd, "is_mindie_sd_available", lambda: False)
    monkeypatch.setattr(mindie_sd, "is_atc_available", lambda: True)
    monkeypatch.setattr(mindie_sd.shutil, "which", lambda _: "/fake/path/atc")

    def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        return subprocess.CompletedProcess(
            cmd, 1, stdout="", stderr="ERROR: unsupported layer X",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(OMCompileError, match="atc compile failed"):
        compile_to_om(src, out, prefer="atc")
    with pytest.raises(OMCompileError, match="unsupported layer X"):
        compile_to_om(src, out, prefer="atc")


def test_compile_to_om_atc_timeout_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ATC subprocess timeout → OMCompileError."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text("{}")
    (src / "quant_meta.json").write_text("{}")
    (src / "model.onnx").write_bytes(b"fake")
    out = tmp_path / "model.om"
    monkeypatch.setattr(mindie_sd, "is_mindie_sd_available", lambda: False)
    monkeypatch.setattr(mindie_sd, "is_atc_available", lambda: True)
    monkeypatch.setattr(mindie_sd.shutil, "which", lambda _: "/fake/path/atc")

    def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        raise subprocess.TimeoutExpired(cmd, timeout=1800)

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(OMCompileError, match="timed out"):
        compile_to_om(src, out, prefer="atc")


def test_compile_to_om_atc_910c_soc_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Atlas 900 A3 / 910C target → ``Ascend910C`` soc-version flag."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text("{}")
    (src / "quant_meta.json").write_text("{}")
    (src / "model.onnx").write_bytes(b"fake")
    out = tmp_path / "model.om"
    monkeypatch.setattr(mindie_sd, "is_mindie_sd_available", lambda: False)
    monkeypatch.setattr(mindie_sd, "is_atc_available", lambda: True)
    monkeypatch.setattr(mindie_sd.shutil, "which", lambda _: "/fake/path/atc")

    captured: dict = {}

    def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        captured["cmd"] = cmd
        Path(cmd[cmd.index("--output") + 1]).write_bytes(b"OM")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    compile_to_om(
        src, out, target_hardware="Atlas 900 A3 / 910C", prefer="atc",
    )
    assert captured["cmd"][8] == "Ascend910C"


# --------------------------------------------------------------------------- #
# mindie.sd.tools.compile path
# --------------------------------------------------------------------------- #


def test_compile_to_om_via_mindie_sd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """mindie.sd.tools.compile path: called with right args + writes artifact."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text("{}")
    (src / "quant_meta.json").write_text("{}")
    out = tmp_path / "model.om"

    monkeypatch.setattr(mindie_sd, "is_mindie_sd_available", lambda: True)

    captured: dict = {}

    def fake_compile(model_path, output_path, target_hardware):  # type: ignore[no-untyped-def]
        captured["model_path"] = model_path
        captured["output_path"] = output_path
        captured["target_hardware"] = target_hardware
        Path(output_path).write_bytes(b"FAKE_OM_FROM_MINDIE_SD")
        return output_path

    fake_mod = types.ModuleType("mindie.sd.tools")
    fake_mod.compile = fake_compile  # type: ignore[attr-defined]
    fake_pkg_sd = types.ModuleType("mindie.sd")
    fake_pkg_sd.__path__ = []  # type: ignore[attr-defined]
    fake_pkg = types.ModuleType("mindie")
    fake_pkg.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mindie", fake_pkg)
    monkeypatch.setitem(sys.modules, "mindie.sd", fake_pkg_sd)
    monkeypatch.setitem(sys.modules, "mindie.sd.tools", fake_mod)

    result = compile_to_om(src, out, prefer="mindie_sd")
    assert result == out
    assert captured["target_hardware"] == "Atlas 800 A2 / 910B"
    assert out.read_bytes() == b"FAKE_OM_FROM_MINDIE_SD"


def test_compile_to_om_mindie_sd_internal_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """mindie.sd.tools.compile raised → OMCompileError with exception repr."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text("{}")
    (src / "quant_meta.json").write_text("{}")
    out = tmp_path / "model.om"
    monkeypatch.setattr(mindie_sd, "is_mindie_sd_available", lambda: True)

    def fake_compile(**_kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("graph rewrite failed at node X")

    fake_mod = types.ModuleType("mindie.sd.tools")
    fake_mod.compile = fake_compile  # type: ignore[attr-defined]
    fake_pkg_sd = types.ModuleType("mindie.sd")
    fake_pkg_sd.__path__ = []  # type: ignore[attr-defined]
    fake_pkg = types.ModuleType("mindie")
    fake_pkg.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mindie", fake_pkg)
    monkeypatch.setitem(sys.modules, "mindie.sd", fake_pkg_sd)
    monkeypatch.setitem(sys.modules, "mindie.sd.tools", fake_mod)

    with pytest.raises(OMCompileError, match="mindie.sd.tools.compile failed"):
        compile_to_om(src, out, prefer="mindie_sd")


# --------------------------------------------------------------------------- #
# Validation + diagnostics
# --------------------------------------------------------------------------- #


def test_compile_to_om_unsupported_hardware(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text("{}")
    out = tmp_path / "model.om"
    with pytest.raises(OMCompileError, match="not in"):
        compile_to_om(src, out, target_hardware="Bogus 9000")


def test_provider_summary_diagnostic() -> None:
    """provider_summary always returns a diagnostic dict with the right shape."""
    s = mindie_sd.provider_summary()
    assert s["engine"] == "compile_to_om"
    assert s["compile_path"] in mindie_sd.COMPILE_PATHS
    assert isinstance(s["mindie_sd_available"], bool)
    assert isinstance(s["atc_available"], bool)
    assert "Atlas 800 A2 / 910B" in s["supported_hardware"]
    assert "Ascend910B4" in s["soc_version_map"].values()
