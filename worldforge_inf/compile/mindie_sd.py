"""MindIE-SD compile wrapper.

Per spec §4.4: this is the diffusion model compile backend that vLLM-Omni Q2 2026
roadmap is integrating. For V-JEPA 2 we use it to produce offline OM (Ascend
offline model) artifacts from a quantized PyTorch checkpoint.

Phase-1 stub: writes a placeholder .om file with metadata. Real MindIE-SD
integration lands in Phase-2 W3.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from worldforge_common.exceptions import OMCompileError

SUPPORTED_HARDWARE = (
    "Atlas 800 A2 / 910B",
    "Atlas 900 A3 / 910C",
)


def compile_to_om(
    model_path: Path,
    output_path: Path,
    target_hardware: str = "Atlas 800 A2 / 910B",
    require_quantized: bool = True,
) -> Path:
    """Compile a quantized model directory into a `.om` offline model file.

    Returns the output .om path. Raises OMCompileError if quant_meta.json is
    missing and require_quantized=True.
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

    # Compute a deterministic hash from config + quant_meta for the placeholder
    h = hashlib.sha256()
    for fname in ("config.json", "quant_meta.json"):
        p = src / fname
        if p.exists():
            h.update(p.read_bytes())

    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": "phase1-stub",
        "source": str(src),
        "target_hardware": target_hardware,
        "checksum": h.hexdigest(),
        "note": "Phase-2 W3: replace with real MindIE-SD compile.",
    }
    out.write_text(json.dumps(payload, indent=2))
    return out
