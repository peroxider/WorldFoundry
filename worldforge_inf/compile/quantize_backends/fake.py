"""FakeBackend — CPU-runnable surrogate that does NOT call msModelSlim.

Used only for orchestration smoke-tests on hosts without NPU. NEVER
registered as ``"w8a8"`` (which is reserved for the real msModelSlim
backend). The fake copies bf16 weights unchanged + writes a sidecar so
``cosine_similarity`` can load the result with the standard Hugging Face
``AutoModel.from_pretrained`` API.

Cosine evaluation against this output will be ~1.0 (identical weights);
the goal is to verify the pipeline is wired correctly, not to simulate
quantization noise.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from . import register


@register("fake")
class FakeBackend:
    """CPU-runnable surrogate backend. See module docstring."""

    def quantize(
        self,
        model_path: Path,
        output_path: Path,
        calibration_dataset: str | None,
    ) -> Path:
        model_path = Path(model_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        config = model_path / "config.json"
        if not config.exists():
            raise FileNotFoundError(
                f"Missing {config} — input model directory is not a valid HF model."
            )

        shutil.copy2(config, output_path / "config.json")
        # Copy weights in whichever format the source uses. Modern HF
        # saves as ``model.safetensors``; legacy save uses
        # ``pytorch_model.bin``. AutoModel.from_pretrained can load
        # either as long as the file lives next to ``config.json``.
        weights_bin = model_path / "pytorch_model.bin"
        weights_safetensors = model_path / "model.safetensors"
        if weights_bin.exists():
            shutil.copy2(weights_bin, output_path / "pytorch_model.bin")
        elif weights_safetensors.exists():
            shutil.copy2(weights_safetensors, output_path / "model.safetensors")

        meta = {
            "quant_type": "fake",
            "calibration_dataset": calibration_dataset,
            "source_model_path": str(model_path),
            "phase": "w11-w12-fake",
            "backend": "fake",
            "note": "FakeBackend CPU surrogate; weights copied unchanged.",
        }
        (output_path / "quant_meta.json").write_text(json.dumps(meta, indent=2))
        return output_path
