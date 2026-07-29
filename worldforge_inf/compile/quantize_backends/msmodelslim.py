"""W8A8 static quantization via Huawei msModelSlim.

``msmodelslim`` is an Ascend-only wheel (pinned as ``msmodelslim==26.1.0`` in
the ``[quant]`` extra) and is not importable on x86 dev/CI hosts. Importing it
unguarded would make this module unimportable, which would in turn prevent
``W8A8Backend`` from self-registering — so ``backend_available("w8a8")`` could
never distinguish "unknown quant type" from "missing dependency".

Instead the import is guarded and recorded in ``_MSMODELSLIM_AVAILABLE``. The
module always imports cleanly and always registers the backend; the actionable
install hint (G4) is raised from ``quantize()`` at the point of use.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoModel

from worldforge_inf.compile.quantize_backends import register

try:  # pragma: no cover - environment dependent
    import msmodelslim

    _MSMODELSLIM_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    msmodelslim = None  # type: ignore[assignment]
    _MSMODELSLIM_AVAILABLE = False

_INSTALL_HINT = (
    "W8A8 quantization requires msModelSlim, which is not installed. "
    "Run: pip install worldforge[quant]"
)

SUPPORTED_CALIBRATION_DATASETS = ("videomix22m-512",)
DEFAULT_CALIBRATION_DATASET = "videomix22m-512"
_CALIBRATION_SAMPLES = 512
_CALIBRATION_SEED = 42


def _load_calibration_dataset(
    n: int = _CALIBRATION_SAMPLES, seed: int = _CALIBRATION_SEED
) -> Iterable[Any]:
    """Load calibration samples, importing the loader lazily.

    The import is deferred so this module stays importable before
    ``worldforge_inf.data.calib_videomix22m`` lands, and so unit tests can
    patch this function instead of reaching for HF datasets over the network.
    """
    from worldforge_inf.data.calib_videomix22m import load_calibration_dataset

    return load_calibration_dataset(n=n, seed=seed)


@register("w8a8")
class W8A8Backend:
    """W8A8 symmetric per-channel-weight / per-tensor-activation quant."""

    def quantize(
        self,
        model_path: Path,
        output_path: Path,
        calibration_dataset: str | None,
    ) -> Path:
        model_path = Path(model_path)
        output_path = Path(output_path)

        if not (model_path / "config.json").exists():
            raise FileNotFoundError(f"Model dir {model_path} missing config.json")

        if calibration_dataset is not None and (
            calibration_dataset not in SUPPORTED_CALIBRATION_DATASETS
        ):
            raise ValueError(
                f"Unknown calibration_dataset={calibration_dataset!r}. "
                f"Supported: None | {' | '.join(map(repr, SUPPORTED_CALIBRATION_DATASETS))}"
            )

        if not _MSMODELSLIM_AVAILABLE:
            raise ImportError(_INSTALL_HINT)

        samples = list(
            _load_calibration_dataset(n=_CALIBRATION_SAMPLES, seed=_CALIBRATION_SEED)
        )

        # Load V-JEPA 2 ViT in bf16
        model = AutoModel.from_pretrained(str(model_path), torch_dtype=torch.bfloat16)

        # msModelSlim quantization
        slim = msmodelslim.MSModelSlim(model)
        quantized_model = slim.quantize(
            quant_type="W8A8",
            quant_mode="static",
            calibration_data=samples,
            calibration_size=len(samples),
        )

        # Persist quantized model + meta sidecar
        output_path.mkdir(parents=True, exist_ok=True)
        quantized_model.save_pretrained(str(output_path))
        meta = {
            "quant_type": "w8a8",
            "calibration_dataset": calibration_dataset or DEFAULT_CALIBRATION_DATASET,
            "source_model_path": str(model_path),
            "phase": "phase2-w11",
            "backend": "msmodelslim",
            "msmodelslim_version": msmodelslim.__version__,
            "calibration_samples": len(samples),
            "note": "W8A8 static quantization via msModelSlim.",
        }
        (output_path / "quant_meta.json").write_text(json.dumps(meta, indent=2))
        return output_path
