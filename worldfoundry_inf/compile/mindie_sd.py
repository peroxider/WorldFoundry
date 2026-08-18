"""MindIE-SD / ATC compile wrapper for V-JEPA 2 diffusion model.

Per spec §4.4: the diffusion model compile backend that vLLM-Omni Q2
2026 is integrating. For V-JEPA 2 we use it to produce offline ``.om``
(Ascend offline model) artifacts from a quantized PyTorch / ONNX
checkpoint.

Phase-2 (this commit, M3-T4 partial closure): real compile path with
three-way resolution:

1. ``mindie.sd.tools.compile`` — preferred, uses CANN EP-aware toolchain.
2. ``atc`` CLI subprocess — runs on any host with the Ascend toolchain
   (x86-64 or ARM64), does not require an NPU device to be present.
3. JSON placeholder stub — backward-compat fallback for hosts that have
   neither. Always writes the JSON payload to ``output_path`` so
   downstream tooling that consumes ``compile_to_om`` output (benchmark
   scripts, dashboards) can always find a file at the expected path.

Public API
----------
* :func:`compile_to_om` — main entry; dispatches to one of the three
  paths above based on availability + the ``prefer`` hint.
* :func:`compile_path` — pure probe (no I/O): which path would
  ``compile_to_om`` use RIGHT NOW given the current environment.
* :func:`is_mindie_sd_available` — ``mindie_sd`` wheel importable.
* :func:`is_atc_available` — ``atc`` CLI on PATH.
* :func:`provider_summary` — diagnostic dict for CI logs / bench reports.

Backward compat
---------------
The phase-1 stub payload schema is preserved verbatim — existing
consumers reading ``phase == "phase1-stub"`` keep working. The new
keys ``compile_path`` and ``stub_reason`` are additive.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Literal

from worldfoundry_common.exceptions import OMCompileError

SUPPORTED_HARDWARE = (
    "Atlas 800 A2 / 910B",
    "Atlas 900 A3 / 910C",
)

# Path names returned by ``compile_path`` / written into the JSON stub
# payload. Kept as a tuple so callers can iterate the priority order.
COMPILE_PATHS = ("mindie_sd", "atc", "stub")

CompilePath = Literal["mindie_sd", "atc", "stub"]

# Mapping from the spec's target_hardware strings to ATC ``--soc-version``
# values. ATC is picky about the exact chip identifier; using the wrong
# one fails fast at compile time with "soc_version not supported".
_SOC_VERSION_FOR: dict[str, str] = {
    "Atlas 800 A2 / 910B": "Ascend910B4",
    "Atlas 900 A3 / 910C": "Ascend910C",
}


def _soc_version_for(target_hardware: str) -> str:
    """Translate :data:`SUPPORTED_HARDWARE` strings to ATC soc-version ids.

    Raises:
        OMCompileError: if ``target_hardware`` is not in
            :data:`SUPPORTED_HARDWARE` (caller should have validated).
    """
    try:
        return _SOC_VERSION_FOR[target_hardware]
    except KeyError as exc:  # pragma: no cover -- caller should validate
        raise OMCompileError(
            f"target_hardware={target_hardware!r} not in {SUPPORTED_HARDWARE}"
        ) from exc


def is_mindie_sd_available() -> bool:
    """Whether the ``mindie_sd`` Python package is importable.

    Detection uses :func:`importlib.util.find_spec` so the test never
    imports the wheel (which can fail inside Atlas containers that
    lack CANN even when the wheel metadata is present).
    """
    import importlib.util

    return importlib.util.find_spec("mindie_sd") is not None


def is_atc_available() -> bool:
    """Whether the ``atc`` Ascend Tensor Compiler CLI is on PATH."""
    return shutil.which("atc") is not None


def compile_path(
    prefer: CompilePath | str = "auto",
) -> CompilePath:
    """Probe which compile path :func:`compile_to_om` would use RIGHT NOW.

    Args:
        prefer: ``"mindie_sd"`` / ``"atc"`` / ``"stub"`` to force a
            specific path (raises if unavailable). ``"auto"`` (default)
            returns the highest-priority path that is currently reachable.

    Returns:
        One of :data:`COMPILE_PATHS`.

    Raises:
        OMCompileError: if ``prefer`` requests a path that is not
            available in the current environment.
    """
    if prefer == "auto":
        if is_mindie_sd_available():
            return "mindie_sd"
        if is_atc_available():
            return "atc"
        return "stub"
    if prefer == "mindie_sd":
        if not is_mindie_sd_available():
            raise OMCompileError(
                "prefer='mindie_sd' but mindie_sd is not importable; "
                "pip install mindie-sd or set prefer='atc'/'auto'."
            )
        return "mindie_sd"
    if prefer == "atc":
        if not is_atc_available():
            raise OMCompileError(
                "prefer='atc' but atc CLI not found on PATH; "
                "source the Ascend toolkit or set prefer='stub'/'auto'."
            )
        return "atc"
    if prefer == "stub":
        return "stub"
    raise OMCompileError(
        f"prefer={prefer!r} not in {COMPILE_PATHS + ('auto',)}"
    )


# ---------------------------------------------------------------------------
# Real compile paths
# ---------------------------------------------------------------------------


def _compile_via_mindie_sd(
    model_path: Path,
    output_path: Path,
    target_hardware: str,
) -> Path:
    """Compile via ``mindie.sd.tools.compile``.

    Raises:
        OMCompileError: the wheel is missing (ImportError) or the compile
            call raised (any other exception).
    """
    try:
        from mindie.sd.tools import compile as msd_compile  # type: ignore
    except ImportError as exc:
        raise OMCompileError(
            f"mindie_sd wheel importable at probe time but failed at "
            f"import: {exc!r}; check CANN runtime / torch_npu linkage."
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = msd_compile(
            model_path=str(model_path),
            output_path=str(output_path),
            target_hardware=target_hardware,
        )
    except Exception as exc:  # mindie_sd raises a wide exception hierarchy
        raise OMCompileError(
            f"mindie.sd.tools.compile failed: {exc!r}"
        ) from exc
    return Path(result) if result is not None else output_path


def _onnx_input_shape(onnx_file: Path) -> str | None:
    """Materialize ATC ``--input_shape`` from an ONNX graph's inputs.

    ``torch.onnx.export(..., dynamic_axes=...)`` writes symbolic dims
    (``dim_value == 0``) into the graph; ATC rejects those with
    ``Invalid_Argument: Value NULL for parameter --input_shape``. This
    helper reads each graph input's shape, replaces symbolic/empty dims
    with ``1``, and returns the ATC ``--input_shape`` argument string
    (e.g. ``"latent:1,64,64"``). Returns ``None`` when ``onnx`` is not
    importable or the graph has no inputs (let ATC report its own error).
    """
    try:
        import onnx  # noqa: F401  -- optional runtime dep
    except ImportError:  # pragma: no cover -- probe is best-effort
        return None
    model = onnx.load(str(onnx_file))
    spec_parts: list[str] = []
    for graph_input in model.graph.input:
        dims: list[str] = []
        for dim in graph_input.type.tensor_type.shape.dim:
            if dim.dim_value > 0:
                dims.append(str(int(dim.dim_value)))
            else:  # symbolic / dynamic dim -> ATC needs a concrete value
                dims.append("1")
        if dims:
            spec_parts.append(f"{graph_input.name}:{','.join(dims)}")
    return ";".join(spec_parts) if spec_parts else None


def _compile_via_atc(
    model_path: Path,
    output_path: Path,
    target_hardware: str,
) -> Path:
    """Compile via the ``atc`` CLI subprocess.

    ATC requires a single ONNX / caffe / pb model file as input. The
    convention here is ``<model_path>/model.onnx`` — the onnx export
    step is upstream of this function (see
    ``worldfoundry_inf.quantize_backends`` for the export pipeline).

    Raises:
        OMCompileError: ``model.onnx`` missing, ``atc`` not on PATH,
            or non-zero return code from ``atc``.
    """
    atc = shutil.which("atc")
    if atc is None:  # pragma: no cover -- guarded by compile_path() above
        raise OMCompileError(
            "atc CLI not found on PATH; source the Ascend toolkit."
        )

    onnx_file = model_path / "model.onnx"
    if not onnx_file.exists():
        raise OMCompileError(
            f"ATC compile requires {onnx_file}; export the quantized "
            f"model to ONNX first (worldfoundry_inf.quantize_backends)."
        )

    soc_version = _soc_version_for(target_hardware)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        atc,
        "--model", str(onnx_file),
        "--framework", "5",  # 5 = ONNX in ATC framework enum
        "--output", str(output_path),
        "--soc-version", soc_version,
    ]
    # ATC rejects dynamic axes with "Value NULL for parameter
    # --input_shape is invalid". torch.onnx.export with dynamic_axes
    # produces exactly that, so materialize concrete input shapes from
    # the graph (dynamic dims -> 1) and pass --input_shape explicitly.
    input_shape = _onnx_input_shape(onnx_file)
    if input_shape:
        cmd += ["--input_shape", input_shape]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=1800,
        )
    except subprocess.TimeoutExpired as exc:
        raise OMCompileError(
            f"atc compile timed out after 1800s; check model size / "
            f"hardware contention."
        ) from exc

    if result.returncode != 0:
        # Keep the last 2KB of stderr for the post-mortem; truncate
        # longer outputs so the OMCompileError stays log-friendly.
        stderr_tail = (result.stderr or "")[-2000:]
        stdout_tail = (result.stdout or "")[-500:]
        raise OMCompileError(
            f"atc compile failed (rc={result.returncode}).\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stderr (tail): {stderr_tail}\n"
            f"stdout (tail): {stdout_tail}"
        )

    # ATC appends ".om" to the --output value (e.g. "model.om" ->
    # "model.om.om"), so check both the requested path and the
    # suffixed one; rename to the requested path to keep the public
    # contract (`compile_to_om` returns exactly `output_path`).
    if not output_path.exists():
        atc_appended = Path(str(output_path) + ".om")
        if atc_appended.exists():
            atc_appended.rename(output_path)
    if not output_path.exists():
        raise OMCompileError(
            f"atc returned 0 but did not write {output_path}; "
            f"check atc output dir permissions."
        )
    return output_path


def _compile_via_stub(
    model_path: Path,
    output_path: Path,
    target_hardware: str,
    reason: str,
) -> Path:
    """JSON placeholder stub — phase-1 backward-compat path.

    Always writes a deterministic JSON file at ``output_path``. The
    payload keeps the original ``phase == "phase1-stub"`` field so
    consumers reading that key keep working, and adds two new fields:
    ``compile_path == "stub"`` and ``stub_reason``.
    """
    h = hashlib.sha256()
    for fname in ("config.json", "quant_meta.json"):
        p = model_path / fname
        if p.exists():
            h.update(p.read_bytes())
    payload = {
        "phase": "phase1-stub",
        "compile_path": "stub",
        "stub_reason": reason,
        "source": str(model_path),
        "target_hardware": target_hardware,
        "checksum": h.hexdigest(),
        "note": (
            "Real MindIE-SD compile lands when mindie_sd wheel is "
            "importable OR atc CLI is on PATH. See "
            "worldfoundry_inf.compile.mindie_sd.compile_path()."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    return output_path


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def compile_to_om(
    model_path: Path,
    output_path: Path,
    target_hardware: str = "Atlas 800 A2 / 910B",
    require_quantized: bool = True,
    prefer: CompilePath | str = "auto",
) -> Path:
    """Compile a quantized model directory into a ``.om`` offline model file.

    Resolution order (when ``prefer == 'auto'``):

    1. ``mindie.sd.tools.compile`` if the wheel is importable.
    2. ``atc`` CLI subprocess if ``atc`` is on PATH.
    3. JSON placeholder stub (always succeeds, writes metadata only).

    Callers can pin a path explicitly via ``prefer``.

    Args:
        model_path: directory containing ``config.json`` and (when
            ``require_quantized``) ``quant_meta.json``, plus a
            ``model.onnx`` for ATC.
        output_path: target ``.om`` file path.
        target_hardware: one of :data:`SUPPORTED_HARDWARE`.
        require_quantized: enforce presence of ``quant_meta.json``.
        prefer: ``"auto"`` (default) or one of :data:`COMPILE_PATHS`.

    Returns:
        The output ``.om`` (or stub ``.json``) path.

    Raises:
        OMCompileError: bad ``target_hardware``, missing
            ``quant_meta.json``, missing ``model.onnx``, ATC non-zero
            rc, ``mindie_sd`` raised, or explicit ``prefer`` requested
            an unavailable path.
    """
    if target_hardware not in SUPPORTED_HARDWARE:
        raise OMCompileError(
            f"target_hardware={target_hardware!r} not in {SUPPORTED_HARDWARE}"
        )

    src = Path(model_path)
    out = Path(output_path)
    if require_quantized and not (src / "quant_meta.json").exists():
        raise OMCompileError(
            f"Source directory {src} is missing quant_meta.json. "
            f"Run quantize_model(quant_type='w8a8') first. "
            f"Per spec §4.4 (quantization before compile)."
        )

    path_used = compile_path(prefer=prefer)

    if path_used == "mindie_sd":
        return _compile_via_mindie_sd(src, out, target_hardware)
    if path_used == "atc":
        return _compile_via_atc(src, out, target_hardware)
    # Stub path: reason depends on what triggered the fallback.
    reason = (
        "neither mindie_sd wheel importable nor atc CLI on PATH"
        if prefer == "auto"
        else f"prefer={prefer!r} fell back to stub"
    )
    return _compile_via_stub(src, out, target_hardware, reason=reason)


def provider_summary() -> dict[str, Any]:
    """Diagnostic summary of the compile environment.

    Useful for ``scripts/verify_goal_evidence.py``-style CI gates and
    for bench summaries that need to record which path the compile
    took.
    """
    path = compile_path(prefer="auto")
    return {
        "engine": "compile_to_om",
        "compile_path": path,
        "mindie_sd_available": is_mindie_sd_available(),
        "atc_available": is_atc_available(),
        "atc_binary": shutil.which("atc"),
        "supported_hardware": list(SUPPORTED_HARDWARE),
        "soc_version_map": dict(_SOC_VERSION_FOR),
    }


__all__ = [
    "compile_to_om",
    "compile_path",
    "is_mindie_sd_available",
    "is_atc_available",
    "provider_summary",
    "COMPILE_PATHS",
    "SUPPORTED_HARDWARE",
]
