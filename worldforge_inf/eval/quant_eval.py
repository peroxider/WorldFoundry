"""End-to-end cosine similarity eval: bf16 baseline vs quantized model.

Used as the W11 acceptance gate per parent spec §9 #6:
"下游特征 cosine similarity 与 bf16 baseline ≥ 0.98".

CLI usage (see __main__ block at bottom):
    python -m worldforge_inf.eval.quant_eval \\
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
        from worldforge_inf.data.calib_videomix22m import load_calibration_dataset
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


if __name__ == "__main__":
    import sys
    sys.exit(_cli())