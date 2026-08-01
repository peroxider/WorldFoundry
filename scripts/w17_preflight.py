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
        return False, (f"weights dir missing: {path} "
                       f"(huggingface-cli download facebook/vjepa2-vitl-1.2b "
                       f"--local-dir {path})")
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
