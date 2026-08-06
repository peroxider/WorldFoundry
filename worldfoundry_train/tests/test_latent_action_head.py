"""W32: unit tests for ``worldfoundry_train.model.latent_action_head.LatentActionHead``.

Covers: construction (valid + invalid args), forward shape contract,
last-layer zero-init, gradient flow through every trainable parameter,
bf16 dtype preservation, and shape-mismatch error paths.
"""
from __future__ import annotations

import pytest
import torch

from worldfoundry_train.model.latent_action_head import LatentActionHead


# ---------------------------------------------------------------------------
# Construction + validation
# ---------------------------------------------------------------------------


def test_latent_action_head_default_construction():
    head = LatentActionHead(encoder_dim=1024, action_dim=7)
    assert head.encoder_dim == 1024
    assert head.action_dim == 7
    assert head.hidden_dim >= 64  # default = max(64, 2*action_dim)


def test_latent_action_head_custom_hidden_dim():
    head = LatentActionHead(encoder_dim=64, action_dim=4, hidden_dim=128)
    assert head.hidden_dim == 128


@pytest.mark.parametrize("bad_arg", [{"encoder_dim": 0}, {"action_dim": 0}])
def test_latent_action_head_rejects_non_positive_dims(bad_arg):
    kwargs = {"encoder_dim": 64, "action_dim": 4}
    kwargs.update(bad_arg)
    with pytest.raises(ValueError, match=r"must be > 0"):
        LatentActionHead(**kwargs)


# ---------------------------------------------------------------------------
# Forward shape + dtype
# ---------------------------------------------------------------------------


def test_latent_action_head_forward_shape():
    head = LatentActionHead(encoder_dim=64, action_dim=7)
    enc = torch.randn(3, 64)
    act = torch.randn(3, 7)
    out = head(enc, act)
    assert out.shape == (3, 7)


def test_latent_action_head_bf16_dtype_preserved():
    """bf16 inputs flow through the head without upcast to fp32."""
    head = LatentActionHead(encoder_dim=64, action_dim=4).to(torch.bfloat16)
    enc = torch.randn(2, 64, dtype=torch.bfloat16)
    act = torch.randn(2, 4, dtype=torch.bfloat16)
    out = head(enc, act)
    assert out.dtype == torch.bfloat16
    assert out.shape == (2, 4)


# ---------------------------------------------------------------------------
# Zero-init at construction (W32 day-0 contract)
# ---------------------------------------------------------------------------


def test_latent_action_head_is_zero_function_at_init():
    """At init the head outputs 0 for any (encoder_features, current_action),
    so swapping the W7 passthrough for the W32 head is a no-op on day-0."""
    torch.manual_seed(0)
    head = LatentActionHead(encoder_dim=128, action_dim=7)
    enc = torch.randn(8, 128)
    act = torch.randn(8, 7)
    out = head(enc, act)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_latent_action_head_last_linear_is_zero_init():
    head = LatentActionHead(encoder_dim=64, action_dim=4)
    last = head.net[-1]
    assert torch.allclose(last.weight, torch.zeros_like(last.weight))
    assert torch.allclose(last.bias, torch.zeros_like(last.bias))


def test_latent_action_head_first_linear_is_not_zero_init():
    """The non-last linears use trunc_normal(0.02), not zero init — otherwise
    the head would be permanently stuck at zero after a step."""
    head = LatentActionHead(encoder_dim=64, action_dim=4)
    first = head.net[0]
    assert not torch.allclose(first.weight, torch.zeros_like(first.weight))


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_latent_action_head_gradient_flows_to_all_params():
    head = LatentActionHead(encoder_dim=64, action_dim=4)
    enc = torch.randn(2, 64, requires_grad=False)
    act = torch.randn(2, 4, requires_grad=False)
    out = head(enc, act)
    loss = out.sum()
    loss.backward()
    for name, p in head.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"


# ---------------------------------------------------------------------------
# Shape / batch error paths
# ---------------------------------------------------------------------------


def test_latent_action_head_rejects_wrong_encoder_dim():
    head = LatentActionHead(encoder_dim=64, action_dim=4)
    enc = torch.randn(2, 32)  # wrong last dim
    act = torch.randn(2, 4)
    with pytest.raises(ValueError, match="encoder_features last dim"):
        head(enc, act)


def test_latent_action_head_rejects_wrong_action_dim():
    head = LatentActionHead(encoder_dim=64, action_dim=4)
    enc = torch.randn(2, 64)
    act = torch.randn(2, 7)  # wrong last dim
    with pytest.raises(ValueError, match="current_action last dim"):
        head(enc, act)


def test_latent_action_head_rejects_batch_mismatch():
    head = LatentActionHead(encoder_dim=64, action_dim=4)
    enc = torch.randn(2, 64)
    act = torch.randn(3, 4)  # different B
    with pytest.raises(ValueError, match="batch mismatch"):
        head(enc, act)


def test_latent_action_head_rejects_3d_encoder_features():
    head = LatentActionHead(encoder_dim=64, action_dim=4)
    enc = torch.randn(2, 5, 64)  # extra dim
    act = torch.randn(2, 4)
    with pytest.raises(ValueError, match="encoder_features must be"):
        head(enc, act)
