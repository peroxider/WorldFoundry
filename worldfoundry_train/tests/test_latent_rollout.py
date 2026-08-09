"""Tests for :mod:`worldfoundry_train.model.latent_rollout` (W2.17).

Three tests cover the planner's headline contracts:

- ``test_latent_rollout_planner_predicts_horizon_steps`` — shapes match
  the plan (latents ``(horizon, N, D)``, pixels
  ``(horizon, T_first, 3, H, W)``), pixels are clamped to ``[0, 1]``,
  and the planner is reusable across calls.
- ``test_action_conditioning_changes_rollout`` — running the planner
  with two distinct action sequences produces latents whose L2 distance
  is non-trivial (proves the action conditioning actually flows).
- ``test_rollout_grad_flows_through_predictor`` — backpropping through
  the imagined latents updates the predictor's action projection
  weights (proves the gradient path is alive).
- ``test_mask_all_tokens_helper_shape_and_dtype`` — covers the helper
  free function.

All four tests are torch-only; no dm_control, no HF.
"""
from __future__ import annotations

import pytest
import torch

from worldfoundry_train.model.latent_rollout import (
    LatentRolloutPlanner,
    mask_all_tokens,
)
from worldfoundry_train.model.tiny_jepa import (
    TinyActionConditionedPredictor,
    TinyEncoderConfig,
    TinyPixelDecoder,
    TinyVJEPA2Encoder,
)


def _build_stack(image_size: int = 16, patch_size: int = 4, action_dim: int = 4):
    cfg = TinyEncoderConfig(
        image_size=image_size, patch_size=patch_size,
        embed_dim=24, depth=2, num_heads=2,
    )
    encoder = TinyVJEPA2Encoder(cfg)
    target = TinyVJEPA2Encoder(cfg)
    target.load_state_dict(encoder.state_dict())
    pred = TinyActionConditionedPredictor(cfg, action_dim=action_dim)
    dec = TinyPixelDecoder(cfg)
    return cfg, encoder, target, pred, dec


def test_latent_rollout_planner_predicts_horizon_steps() -> None:
    """Shape contract + sigmoid-clamped pixels + re-usability."""
    cfg, encoder, target, pred, dec = _build_stack()
    planner = LatentRolloutPlanner(
        encoder=encoder, target_encoder=target, predictor=pred, decoder=dec,
        image_size=cfg.image_size, patch_size=cfg.patch_size,
        action_dim=pred.action_dim, horizon=3,
        grad_through_predictor=False,
    )
    assert planner.horizon == 3
    assert planner.num_tokens == (cfg.image_size // cfg.patch_size) ** 2
    assert planner.embed_dim == cfg.embed_dim
    assert not planner.context_initialized

    T_first = planner.num_context_frames
    pixel_first_frame = torch.rand(1, T_first, 3, cfg.image_size, cfg.image_size)
    actions_seq = torch.randn(planner.horizon, 1, pred.action_dim)
    latents, pixels = planner.predict_sequence(pixel_first_frame, actions_seq)
    assert latents.shape == (planner.horizon, planner.num_tokens, cfg.embed_dim)
    assert pixels.shape == (planner.horizon, T_first, 3, cfg.image_size, cfg.image_size)
    assert float(pixels.min()) >= 0.0
    assert float(pixels.max()) <= 1.0

    # Reusability: a second call should not crash.
    latents2, _ = planner.predict_sequence(pixel_first_frame, actions_seq)
    assert latents2.shape == latents.shape
    assert torch.isfinite(latents2).all()


def test_action_conditioning_changes_rollout() -> None:
    """Different action sequences produce latents with non-trivial L2."""
    torch.manual_seed(7)
    cfg, encoder, target, pred, dec = _build_stack(action_dim=4)
    planner = LatentRolloutPlanner(
        encoder=encoder, target_encoder=target, predictor=pred, decoder=dec,
        image_size=cfg.image_size, patch_size=cfg.patch_size,
        action_dim=pred.action_dim, horizon=2,
        grad_through_predictor=False,
    )
    pixel_first_frame = torch.rand(
        1, planner.num_context_frames, 3, cfg.image_size, cfg.image_size,
    )
    actions_a = torch.zeros(planner.horizon, 1, pred.action_dim)
    actions_b = torch.ones(planner.horizon, 1, pred.action_dim) * 1.0
    latents_a, _ = planner.predict_sequence(pixel_first_frame, actions_a)
    latents_b, _ = planner.predict_sequence(pixel_first_frame, actions_b)
    assert latents_a.shape == latents_b.shape
    diff = float((latents_a - latents_b).pow(2).mean().sqrt())
    # Lower bound is intentionally loose — the predictor is randomly
    # initialized so the absolute separation depends on weights.
    assert diff > 1e-3, f"action conditioning did not change rollout (L2={diff})"


def test_rollout_grad_flows_through_predictor() -> None:
    """Backward through imagined latents updates predictor parameters."""
    torch.manual_seed(11)
    cfg, encoder, target, pred, dec = _build_stack(action_dim=4)
    planner = LatentRolloutPlanner(
        encoder=encoder, target_encoder=target, predictor=pred, decoder=dec,
        image_size=cfg.image_size, patch_size=cfg.patch_size,
        action_dim=pred.action_dim, horizon=1,
        grad_through_predictor=True,
    )
    assert any(p.requires_grad for p in pred.parameters())
    pixel_first_frame = torch.rand(
        1, planner.num_context_frames, 3, cfg.image_size, cfg.image_size,
    )
    planner.reset_context(pixel_first_frame)
    action = torch.randn(1, pred.action_dim)
    imagined = planner.predict_step(action)
    assert imagined.requires_grad
    imagined.sum().backward()
    assert pred._action_proj.weight.grad is not None
    grad_norm = float(pred._action_proj.weight.grad.norm().cpu())
    assert grad_norm > 0.0


def test_mask_all_tokens_helper_shape_and_dtype() -> None:
    """``mask_all_tokens`` returns a ``(batch, N)`` bool tensor."""
    mask = mask_all_tokens(8, batch=2)
    assert mask.shape == (2, 8)
    assert mask.dtype == torch.bool
    assert mask.all().item()
    with pytest.raises(ValueError):
        mask_all_tokens(0)
    with pytest.raises(ValueError):
        mask_all_tokens(4, batch=0)