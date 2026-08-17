"""End-to-end cosine similarity eval: bf16 baseline vs quantized model.

Used as the W11 acceptance gate per parent spec §9 #6:
"下游特征 cosine similarity 与 bf16 baseline ≥ 0.98".

CLI usage (see __main__ block at bottom):
    python -m worldfoundry_inf.eval.quant_eval \\
        --baseline models/vjepa2-vitl-1.2b-bf16 \\
        --quantized models/vjepa2-vitl-1.2b-w8a8 \\
        --threshold 0.98
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from transformers import AutoModel


@dataclass
class EvalReport:
    """Result of a cosine similarity eval run."""
    aggregate_cosine: float
    per_sample_cosine: list[float] = field(default_factory=list)
    passed: bool = False
    threshold: float = 0.98
    n_samples: int = 0


@dataclass
class SDXLEvalReport:
    """Result of an SDXL/VAE W8A8 cosine-eval run (M3-T5).

    Mirrors :class:`EvalReport`'s shape but trades ``per_sample_cosine``
    for ``per_tensor_cosine`` (SDXL/VAE weights are tensors, not HF
    inference samples) and adds ``mse_per_tensor`` so post-mortem tools
    can identify which tensors drift the most. ``passed`` follows the
    M3-T5 acceptance gate: ``aggregate_cosine >= threshold`` (default
    0.95 = ≤ 0.05 deviation).
    """
    aggregate_cosine: float
    per_tensor_cosine: dict[str, float] = field(default_factory=dict)
    mse_per_tensor: dict[str, float] = field(default_factory=dict)
    passed: bool = False
    threshold: float = 0.95
    n_layers: int = 0


def cosine_similarity(
    baseline_path: Path,
    quantized_path: Path,
    eval_dataset: Iterable[torch.Tensor] | None = None,
    threshold: float = 0.98,
    n_samples: int = 100,
    seed: int = 123,
) -> EvalReport:
    """Compare bf16 baseline vs quantized model on shared eval samples.

    Args:
        baseline_path: Path to bf16 baseline model directory.
        quantized_path: Path to quantized model directory (must have config.json).
        eval_dataset: Optional iterable of (3, 224, 224) tensors. If None,
            uses the videomix22m 100-sample holdout (seed=123).
        threshold: Pass threshold for aggregate cosine (default 0.98).
        n_samples: Number of eval samples to run (default 100).
        seed: Random seed for the holdout (G5 reproducibility).

    Returns:
        EvalReport with aggregate_cosine, per_sample_cosine, passed.

    Raises:
        ValueError: empty eval dataset, or model output shape mismatch.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    baseline = AutoModel.from_pretrained(str(baseline_path), torch_dtype=torch.bfloat16)
    quantized = AutoModel.from_pretrained(str(quantized_path))

    if eval_dataset is None:
        # Default holdout: same videomix22m source, seed=123, n_samples
        from worldfoundry_inf.data.calib_videomix22m import load_calibration_dataset
        eval_dataset = list(load_calibration_dataset(n=n_samples, seed=seed))

    samples = list(eval_dataset)
    if not samples:
        raise ValueError(f"eval_dataset is empty (n_samples={n_samples}).")

    per_sample_cosine: list[float] = []
    for sample in samples:
        with torch.no_grad():
            bf16_out = baseline.encode(sample.unsqueeze(0)).last_hidden_state
            int8_out = quantized.encode(sample.unsqueeze(0)).last_hidden_state

        if bf16_out.shape != int8_out.shape:
            raise ValueError(
                f"Shape mismatch: bf16={tuple(bf16_out.shape)} "
                f"vs int8={tuple(int8_out.shape)}. Cannot compute cosine."
            )

        cos = F.cosine_similarity(bf16_out, int8_out, dim=-1).mean().item()
        per_sample_cosine.append(cos)

    aggregate = sum(per_sample_cosine) / len(per_sample_cosine)
    return EvalReport(
        aggregate_cosine=aggregate,
        per_sample_cosine=per_sample_cosine,
        passed=aggregate >= threshold,
        threshold=threshold,
        n_samples=len(per_sample_cosine),
    )


def _cli() -> int:
    """CLI entry point. Returns process exit code (0 = pass, 1 = fail)."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Cosine similarity eval: bf16 baseline vs quantized model"
    )
    parser.add_argument("--baseline", required=True, type=Path,
                        help="Path to bf16 baseline model directory")
    parser.add_argument("--quantized", required=True, type=Path,
                        help="Path to quantized model directory")
    parser.add_argument("--threshold", type=float, default=0.98,
                        help="Cosine threshold (default 0.98)")
    parser.add_argument("--n-samples", type=int, default=100,
                        help="Number of eval samples (default 100)")
    args = parser.parse_args()

    try:
        report = cosine_similarity(
            baseline_path=args.baseline,
            quantized_path=args.quantized,
            threshold=args.threshold,
            n_samples=args.n_samples,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2  # setup/config error

    print(
        f"Cosine: {report.aggregate_cosine:.4f} "
        f"(threshold {report.threshold:.4f}, "
        f"n={report.n_samples}) → "
        f"{'PASS' if report.passed else 'FAIL'}"
    )
    return 0 if report.passed else 1


def cosine_similarity_sdxl(
    baseline_path: Path,
    quantized_path: Path,
    *,
    threshold: float = 0.95,
    weight_filename: str = "diffusion_pytorch_model.safetensors",
) -> SDXLEvalReport:
    """Cosine similarity eval for SDXL/VAE W8A8 quant artifacts (M3-T5).

    Loads ``weight_filename`` from each model directory via
    ``safetensors.safe_open``, computes per-tensor cosine + MSE between
    baseline (BF16 source) and quantized (round-tripped), and reports
    the aggregate cosine + per-tensor breakdown. M3-T5 acceptance gate
    is ``aggregate_cosine >= 0.95`` (i.e. ≤ 0.05 deviation), which the
    W26 packer's MSE < 1e-3 contract guarantees when
    :class:`worldfoundry_inf.compile.quantize_backends.fake.FakeSDXLBackend`
    produces the artifact.

    Args:
        baseline_path: Path to BF16 source model directory.
        quantized_path: Path to W8A8-round-tripped model directory
            produced by :class:`FakeSDXLBackend`.
        threshold: Pass threshold (default 0.95).
        weight_filename: Filename of the safetensors weight file (default
            ``diffusion_pytorch_model.safetensors``).

    Returns:
        :class:`SDXLEvalReport` with aggregate cosine, per-tensor
        cosine, per-tensor MSE, passed flag, threshold, and tensor count.

    Raises:
        FileNotFoundError: Either model directory is missing the
            ``weight_filename`` or ``config.json``.
    """
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ImportError(
            "cosine_similarity_sdxl requires the `safetensors` package. "
            "Run: pip install safetensors"
        ) from exc

    baseline_path = Path(baseline_path)
    quantized_path = Path(quantized_path)
    base_weights = baseline_path / weight_filename
    quant_weights = quantized_path / weight_filename
    if not base_weights.exists():
        raise FileNotFoundError(f"Missing baseline weights: {base_weights}")
    if not quant_weights.exists():
        raise FileNotFoundError(f"Missing quantized weights: {quant_weights}")

    per_tensor_cosine: dict[str, float] = {}
    mse_per_tensor: dict[str, float] = {}
    with safe_open(str(base_weights), framework="pt") as fb, \
         safe_open(str(quant_weights), framework="pt") as fq:
        base_keys = set(fb.keys())
        quant_keys = set(fq.keys())
        if base_keys != quant_keys:
            raise ValueError(
                f"Key mismatch: baseline={sorted(base_keys - quant_keys)} "
                f"quantized={sorted(quant_keys - base_keys)}"
            )
        for key in sorted(base_keys):
            bt = fb.get_tensor(key).to(torch.float32)
            qt = fq.get_tensor(key).to(torch.float32)
            denom = bt.norm() * qt.norm()
            cos = float(
                (bt * qt).sum() / denom
            ) if denom.item() > 1e-12 else 1.0
            per_tensor_cosine[key] = cos
            mse_per_tensor[key] = float(((bt - qt) ** 2).mean())

    if per_tensor_cosine:
        aggregate = sum(per_tensor_cosine.values()) / len(per_tensor_cosine)
    else:
        aggregate = 1.0
    return SDXLEvalReport(
        aggregate_cosine=aggregate,
        per_tensor_cosine=per_tensor_cosine,
        mse_per_tensor=mse_per_tensor,
        passed=aggregate >= threshold,
        threshold=threshold,
        n_layers=len(per_tensor_cosine),
    )


def _cli_sdxl() -> int:
    """CLI entry point for SDXL/VAE cosine eval (M3-T5)."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="SDXL/VAE cosine similarity eval: BF16 baseline vs W8A8-round-tripped"
    )
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--quantized", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--weight-filename", type=str,
                        default="diffusion_pytorch_model.safetensors")
    args = parser.parse_args()

    try:
        report = cosine_similarity_sdxl(
            baseline_path=args.baseline,
            quantized_path=args.quantized,
            threshold=args.threshold,
            weight_filename=args.weight_filename,
        )
    except (ValueError, FileNotFoundError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        f"SDXL cosine: {report.aggregate_cosine:.4f} "
        f"(threshold {report.threshold:.4f}, "
        f"n_tensors={report.n_layers}) → "
        f"{'PASS' if report.passed else 'FAIL'}"
    )
    return 0 if report.passed else 1


if __name__ == "__main__":
    import sys
    if "--sdxl" in sys.argv:
        sys.argv.remove("--sdxl")
        sys.exit(_cli_sdxl())
    sys.exit(_cli())