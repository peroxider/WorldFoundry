"""quant_eval cosine similarity tests — synthetic bf16 vs int8 features."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch


def _make_model(features: torch.Tensor):
    """Mock a model whose .encode(x) returns given features."""
    m = MagicMock()
    m.encode.return_value.last_hidden_state = features
    return m


def test_cosine_above_threshold_passes():
    """cosine=0.99 (above 0.98) → report.passed=True."""
    from worldforge_inf.eval.quant_eval import cosine_similarity

    bf16_feat = torch.randn(1, 4, 8)
    # Make int8 features ~bf16 features (high cosine)
    int8_feat = bf16_feat + torch.randn_like(bf16_feat) * 0.01

    bf16_model = _make_model(bf16_feat)
    int8_model = _make_model(int8_feat)

    with patch(
        "worldforge_inf.eval.quant_eval.AutoModel.from_pretrained",
        side_effect=[bf16_model, int8_model],
    ):
        report = cosine_similarity(
            baseline_path=Path("/fake/bf16"),
            quantized_path=Path("/fake/int8"),
            eval_dataset=[torch.zeros(3, 224, 224)],
            threshold=0.98,
            n_samples=1,
        )

    assert report.passed is True
    assert report.n_samples == 1
    assert report.aggregate_cosine > 0.98
    assert report.threshold == 0.98


def test_cosine_below_threshold_fails():
    """cosine~0.5 (well below 0.98) → report.passed=False."""
    from worldforge_inf.eval.quant_eval import cosine_similarity

    bf16_feat = torch.randn(1, 4, 8)
    # Make int8 features orthogonal (low cosine)
    int8_feat = -bf16_feat + torch.randn_like(bf16_feat) * 0.1

    bf16_model = _make_model(bf16_feat)
    int8_model = _make_model(int8_feat)

    with patch(
        "worldforge_inf.eval.quant_eval.AutoModel.from_pretrained",
        side_effect=[bf16_model, int8_model],
    ):
        report = cosine_similarity(
            baseline_path=Path("/fake/bf16"),
            quantized_path=Path("/fake/int8"),
            eval_dataset=[torch.zeros(3, 224, 224)],
            threshold=0.98,
            n_samples=1,
        )

    assert report.passed is False
    assert report.aggregate_cosine < 0.98


def test_aggregate_cosine_is_mean_of_per_sample():
    """aggregate_cosine == mean(per_sample_cosine)."""
    from worldforge_inf.eval.quant_eval import cosine_similarity

    bf16_feats = [torch.randn(1, 4, 8) for _ in range(3)]
    int8_feats = [f + torch.randn_like(f) * 0.01 for f in bf16_feats]

    bf16_model = _make_model(bf16_feats[0])
    int8_model = _make_model(int8_feats[0])

    # Use 3 samples by mocking encode() to return different features each call
    bf16_model.encode.side_effect = [
        MagicMock(last_hidden_state=f) for f in bf16_feats
    ]
    int8_model.encode.side_effect = [
        MagicMock(last_hidden_state=f) for f in int8_feats
    ]

    with patch(
        "worldforge_inf.eval.quant_eval.AutoModel.from_pretrained",
        side_effect=[bf16_model, int8_model],
    ):
        report = cosine_similarity(
            baseline_path=Path("/fake/bf16"),
            quantized_path=Path("/fake/int8"),
            eval_dataset=[
                torch.zeros(3, 224, 224),
                torch.zeros(3, 224, 224),
                torch.zeros(3, 224, 224),
            ],
            threshold=0.98,
            n_samples=3,
        )

    expected_mean = sum(report.per_sample_cosine) / len(report.per_sample_cosine)
    assert abs(report.aggregate_cosine - expected_mean) < 1e-5


def test_shape_mismatch_raises_value_error():
    """Different bf16 vs int8 output shapes → ValueError."""
    from worldforge_inf.eval.quant_eval import cosine_similarity

    bf16_feat = torch.randn(1, 4, 8)
    int8_feat = torch.randn(1, 4, 16)  # different last dim

    bf16_model = _make_model(bf16_feat)
    int8_model = _make_model(int8_feat)

    with patch(
        "worldforge_inf.eval.quant_eval.AutoModel.from_pretrained",
        side_effect=[bf16_model, int8_model],
    ):
        with pytest.raises(ValueError, match="Shape mismatch"):
            cosine_similarity(
                baseline_path=Path("/fake/bf16"),
                quantized_path=Path("/fake/int8"),
                eval_dataset=[torch.zeros(3, 224, 224)],
                n_samples=1,
            )


def test_empty_eval_dataset_raises():
    """n_samples=0 → ValueError before model load."""
    from worldforge_inf.eval.quant_eval import cosine_similarity

    with pytest.raises(ValueError, match="n_samples must be positive"):
        cosine_similarity(
            baseline_path=Path("/fake/bf16"),
            quantized_path=Path("/fake/int8"),
            n_samples=0,
        )


def test_empty_iter_eval_dataset_raises():
    """eval_dataset=[] with positive n_samples → ValueError after model load."""
    from worldforge_inf.eval.quant_eval import cosine_similarity

    bf16_model = _make_model(torch.randn(1, 4, 8))
    int8_model = _make_model(torch.randn(1, 4, 8))

    with patch(
        "worldforge_inf.eval.quant_eval.AutoModel.from_pretrained",
        side_effect=[bf16_model, int8_model],
    ):
        with pytest.raises(ValueError, match="eval_dataset is empty"):
            cosine_similarity(
                baseline_path=Path("/fake/bf16"),
                quantized_path=Path("/fake/int8"),
                eval_dataset=[],
                n_samples=1,
            )
