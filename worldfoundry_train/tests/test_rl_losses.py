"""W32: unit tests for ``worldfoundry_train.rl_losses.action_prediction_loss``.

Covers: zero loss when pred == target, loss > 0 when pred != target,
gradient flows back to pred, all three reductions, beta param,
shape / unknown-reduction error paths.
"""
from __future__ import annotations

import pytest
import torch

from worldfoundry_train.rl_losses import action_prediction_loss


def test_action_prediction_loss_zero_when_pred_equals_target():
    pred = torch.randn(4, 7)
    out = action_prediction_loss(pred, pred.clone())
    assert out.item() == 0.0


def test_action_prediction_loss_positive_when_pred_differs_from_target():
    pred = torch.zeros(4, 7)
    target = torch.ones(4, 7)
    out = action_prediction_loss(pred, target)
    assert out.item() > 0.0


def test_action_prediction_loss_grad_flows_to_pred():
    pred = torch.randn(3, 4, requires_grad=True)
    target = torch.randn(3, 4)
    out = action_prediction_loss(pred, target)
    out.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_action_prediction_loss_sum_reduction_returns_scalar_sum():
    pred = torch.zeros(2, 3)
    target = torch.ones(2, 3)
    out = action_prediction_loss(pred, target, reduction="sum")
    # smooth_l1 at |1| with beta=1 -> 0.5 * 1^2 = 0.5 per element; sum = 3.0
    assert torch.allclose(out, torch.tensor(3.0), atol=1e-5)


def test_action_prediction_loss_none_reduction_returns_per_element():
    pred = torch.zeros(2, 3)
    target = torch.ones(2, 3)
    out = action_prediction_loss(pred, target, reduction="none")
    assert out.shape == (2, 3)


def test_action_prediction_loss_bf16_dtype():
    pred = torch.zeros(2, 3, dtype=torch.bfloat16)
    target = torch.ones(2, 3, dtype=torch.bfloat16)
    out = action_prediction_loss(pred, target)
    assert out.dtype == torch.bfloat16
    assert out.item() > 0.0


def test_action_prediction_loss_rejects_wrong_pred_ndim():
    pred = torch.randn(2, 3, 4)  # 3-D
    target = torch.randn(2, 3, 4)
    with pytest.raises(ValueError, match=r"\(B, action_dim\)"):
        action_prediction_loss(pred, target)


def test_action_prediction_loss_rejects_shape_mismatch():
    pred = torch.randn(2, 4)
    target = torch.randn(2, 5)
    with pytest.raises(ValueError, match="shape mismatch"):
        action_prediction_loss(pred, target)


def test_action_prediction_loss_rejects_unknown_reduction():
    pred = torch.randn(2, 4)
    target = torch.randn(2, 4)
    with pytest.raises(ValueError, match="unknown reduction"):
        action_prediction_loss(pred, target, reduction="avg")


def test_action_prediction_loss_rejects_non_positive_beta():
    pred = torch.randn(2, 4)
    target = torch.randn(2, 4)
    with pytest.raises(ValueError, match="beta must be > 0"):
        action_prediction_loss(pred, target, beta=0.0)
