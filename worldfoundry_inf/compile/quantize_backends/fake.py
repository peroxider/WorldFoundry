"""FakeBackend — CPU-runnable surrogate that does NOT call msModelSlim.

Used only for orchestration smoke-tests on hosts without NPU. NEVER
registered as ``"w8a8"`` (which is reserved for the real msModelSlim
backend). The fake copies bf16 weights unchanged + writes a sidecar so
``cosine_similarity`` can load the result with the standard Hugging Face
``AutoModel.from_pretrained`` API.

Cosine evaluation against this output will be ~1.0 (identical weights);
the goal is to verify the pipeline is wired correctly, not to simulate
quantization noise.

M3-T5 sibling: :class:`FakeSDXLBackend` registered as ``"fake_sdxl"`` —
also CPU-runnable, also does NOT call msModelSlim, but instead of
copying weights unchanged it round-trips every 2-D float weight through
the real W8A8 ``pack_w8a8_weight`` / ``unpack_w8a8_weight`` from
:mod:`worldfoundry_inf.compile.quantize_primitives`. The MSE < 1e-3 contract of the W26
packers guarantees cosine ≥ 0.99 vs the BF16 source, which satisfies
the M3-T5 acceptance gate ("W8A8 vs BF16 latent cosine ≤ 0.05") on any
host with torch installed. Diffusers-shaped models (SDXL/VAE) typically
use ``diffusion_pytorch_model.safetensors`` as their weight filename;
the SDXL backend handles that and skips 4-D conv weights with a
recorded note (they need a separate conv-aware quant recipe).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch

from worldfoundry_inf.compile.quantize_primitives import (
    ACCEPTANCE_MSE_THRESHOLD,
    pack_w8a8_weight,
    unpack_w8a8_weight,
)

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


@register("fake_sdxl")
class FakeSDXLBackend:
    """CPU-runnable surrogate backend for SDXL/VAE W8A8 quant smoke (M3-T5).

    Loads every tensor from the source safetensors file (SDXL/VAE use
    ``diffusion_pytorch_model.safetensors``), round-trips every 2-D
    float weight through the real :func:`pack_w8a8_weight` /
    :func:`unpack_w8a8_weight` from :mod:`worldfoundry_inf.compile.quantize_primitives`,
    writes the round-tripped weights back to the same filename under
    ``output_path``, and emits a ``quant_meta.json`` sidecar that
    ``worldfoundry_inf.eval.quant_eval.cosine_similarity_sdxl`` reads
    to compute the cosine vs the BF16 source.

    Why round-trip instead of copy?
        The M3-T5 acceptance gate is "W8A8 vs BF16 latent cosine ≤ 0.05
        deviation". A copy-only artifact trivially passes with cosine =
        1.0 — but it doesn't exercise the W8A8 packer at all. The
        round-trip uses the real W26 packers (no msModelSlim needed) so
        the artifact is genuinely W8A8-shaped: cosine ≥ 0.99 vs source
        by the packer's MSE < 1e-3 contract.

    Why 2-D only?
        :func:`pack_w8a8_weight` requires ``(out_features, in_features)``.
        Conv2D 4-D weights need a separate conv-aware quant recipe that
        is out of scope for the host-side smoke path. They are skipped
        with a ``n_weights_skipped`` counter so the sidecar is honest
        about what was processed.

    Attributes recorded in ``quant_meta.json``:
        ``quant_type="fake_sdxl"``, ``phase="m3-t5-fake"``,
        ``backend="fake_sdxl"``, ``round_trip_method="pack_w8a8_weight"``,
        ``mse_threshold=1e-3``, ``n_weights_total``, ``n_weights_rounded``,
        ``n_weights_skipped``, ``weight_filename``, ``note``.
    """

    DEFAULT_WEIGHT_FILENAME = "diffusion_pytorch_model.safetensors"

    def quantize(
        self,
        model_path: Path,
        output_path: Path,
        calibration_dataset: str | None,
    ) -> Path:
        try:
            from safetensors import safe_open
            from safetensors.torch import save_file
        except ImportError as exc:
            raise ImportError(
                "FakeSDXLBackend requires the `safetensors` package. "
                "Run: pip install safetensors"
            ) from exc

        model_path = Path(model_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        config = model_path / "config.json"
        if not config.exists():
            raise FileNotFoundError(
                f"Missing {config} — input model directory is not a valid HF model."
            )
        shutil.copy2(config, output_path / "config.json")

        source_weights = model_path / self.DEFAULT_WEIGHT_FILENAME
        if not source_weights.exists():
            raise FileNotFoundError(
                f"Missing {source_weights} — FakeSDXLBackend requires "
                f"{self.DEFAULT_WEIGHT_FILENAME} (the SDXL/VAE weight filename). "
                f"For V-JEPA-shaped models use FakeBackend (registered as 'fake')."
            )

        n_total = 0
        n_rounded = 0
        n_skipped = 0
        rounded: dict[str, torch.Tensor] = {}
        with safe_open(str(source_weights), framework="pt") as f:
            for key in f.keys():
                n_total += 1
                tensor = f.get_tensor(key)
                if (
                    tensor.ndim == 2
                    and tensor.is_floating_point()
                    and tensor.dtype != torch.float64
                ):
                    packed = pack_w8a8_weight(tensor)
                    rounded[key] = unpack_w8a8_weight(packed)
                    n_rounded += 1
                else:
                    rounded[key] = tensor
                    n_skipped += 1

        save_file(rounded, str(output_path / self.DEFAULT_WEIGHT_FILENAME))

        meta = {
            "quant_type": "fake_sdxl",
            "calibration_dataset": calibration_dataset,
            "source_model_path": str(model_path),
            "phase": "m3-t5-fake",
            "backend": "fake_sdxl",
            "round_trip_method": "pack_w8a8_weight",
            "mse_threshold": ACCEPTANCE_MSE_THRESHOLD,
            "weight_filename": self.DEFAULT_WEIGHT_FILENAME,
            "n_weights_total": n_total,
            "n_weights_rounded": n_rounded,
            "n_weights_skipped": n_skipped,
            "note": (
                "FakeSDXLBackend CPU surrogate (M3-T5); every 2-D float "
                "weight round-tripped through pack_w8a8_weight/"
                "unpack_w8a8_weight from worldfoundry_inf.compile.quantize_primitives. "
                "Conv2D 4-D + non-float tensors are skipped unchanged."
            ),
        }
        (output_path / "quant_meta.json").write_text(json.dumps(meta, indent=2))
        return output_path
