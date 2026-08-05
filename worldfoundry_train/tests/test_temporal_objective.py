"""Unit tests for TemporalObjective (W26).

Sister to ``test_jepa_objective.py``: covers construction validation,
forward shape + content, EMA wiring (target encoder frozen, after_step
mutates target), gradient flow to all trainable params, invalid video
shape rejection, signature/hyperparams content, and state_dict round-trip.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from worldfoundry_train.temporal_objective import TemporalObjective
from worldfoundry_train.model.temporal_predictor import TemporalPredictor

ENC_DIM = 32
PRED_DIM = 16
PAST_GRID = (2, 2, 2)
FUTURE_GRID = (2, 2, 2)
N_PAST = 8
N_FUTURE = 8
PAST_FRAMES = 2
FUTURE_FRAMES = 2
TOTAL_FRAMES = PAST_FRAMES + FUTURE_FRAMES


class _RecordingEncoder(nn.Module):
    """Mock encoder: records call kwargs, returns fixed features with grad."""

    def __init__(self, feats: torch.Tensor) -> None:
        super().__init__()
        self._feats = feats
        self.calls: list[dict] = []
        # Encoder-side trainable params so EMAUpdater has something to track.
        self.online_p_a = nn.Parameter(torch.randn(2, 2))
        self.online_p_b = nn.Parameter(torch.randn(2, 2))

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(last_hidden_state=self._feats)


def _temporal_predictor() -> TemporalPredictor:
    return TemporalPredictor(
        num_past_positions=N_PAST,
        num_future_positions=N_FUTURE,
        past_grid=PAST_GRID,
        future_grid=FUTURE_GRID,
        encoder_dim=ENC_DIM,
        predictor_dim=PRED_DIM,
        depth=2,
        num_heads=2,
    )


def _build_objective(loss_type: str = "smooth_l1", beta: float = 1.0) -> TemporalObjective:
    torch.manual_seed(0)
    online_feats = torch.randn(1, N_PAST, ENC_DIM)
    target_feats = torch.randn(1, N_FUTURE, ENC_DIM)
    online = _RecordingEncoder(online_feats)
    target = _RecordingEncoder(target_feats)
    pred = _temporal_predictor()
    obj = TemporalObjective(
        online_encoder=online,
        target_encoder=target,
        temporal_predictor=pred,
        past_frames=PAST_FRAMES,
        future_frames=FUTURE_FRAMES,
        past_grid=PAST_GRID,
        future_grid=FUTURE_GRID,
        momentum_start=0.5,
        momentum_end=0.5,
        schedule_steps=10,
        loss_type=loss_type,
        beta=beta,
    )
    return obj


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_invalid_loss_type_raises():
    obj = _build_objective()
    with pytest.raises(ValueError, match="loss_type"):
        TemporalObjective(
            online_encoder=obj.online_encoder,
            target_encoder=obj.target_encoder,
            temporal_predictor=obj.predictor,
            past_frames=PAST_FRAMES,
            future_frames=FUTURE_FRAMES,
            past_grid=PAST_GRID,
            future_grid=FUTURE_GRID,
            loss_type="mse",
        )


def test_target_encoder_params_frozen_after_init():
    obj = _build_objective()
    for p in obj.target_encoder.parameters():
        assert not p.requires_grad


# ---------------------------------------------------------------------------
# Forward shape + content
# ---------------------------------------------------------------------------


def test_forward_splits_video_into_past_and_future():
    """forward(video) runs online_encoder on past slice, target on future slice."""
    obj = _build_objective()
    video = torch.zeros(1, TOTAL_FRAMES, 3, 16, 16)
    loss = obj(video)
    assert len(obj.online_encoder.calls) == 1
    assert len(obj.target_encoder.calls) == 1
    past_video = obj.online_encoder.calls[0]["pixel_values_videos"]
    future_video = obj.target_encoder.calls[0]["pixel_values_videos"]
    assert past_video.shape == (1, PAST_FRAMES, 3, 16, 16)
    assert future_video.shape == (1, FUTURE_FRAMES, 3, 16, 16)
    assert torch.isfinite(loss)
    assert loss.dim() == 0  # scalar


def test_forward_loss_is_finite_nonnegative():
    obj = _build_objective(loss_type="smooth_l1")
    video = torch.zeros(2, TOTAL_FRAMES, 3, 16, 16)
    loss = obj(video)
    assert torch.isfinite(loss)
    assert float(loss) >= 0.0


def test_forward_rejects_wrong_video_ndim():
    obj = _build_objective()
    with pytest.raises(ValueError, match=r"\(B, T, C, H, W\)"):
        obj(torch.zeros(4, 3, 16, 16))  # 4D, missing T


def test_forward_rejects_wrong_frame_count():
    obj = _build_objective()
    with pytest.raises(ValueError, match="frames"):
        obj(torch.zeros(1, TOTAL_FRAMES + 1, 3, 16, 16))


# ---------------------------------------------------------------------------
# EMA behavior
# ---------------------------------------------------------------------------


def test_after_step_advances_ema_and_mutates_target():
    """after_step advances the EMA counter and updates target encoder params."""
    obj = _build_objective()
    initial_target_a = obj.target_encoder.online_p_a.detach().clone()
    assert obj.ema.step == 0

    obj.after_step()

    assert obj.ema.step == 1
    # Constant momentum=0.5 -> target must move.
    assert not torch.equal(obj.target_encoder.online_p_a, initial_target_a)


def test_target_encoder_stays_frozen_after_ema_step():
    """After after_step runs, target's requires_grad is still False."""
    obj = _build_objective()
    obj.after_step()
    for p in obj.target_encoder.parameters():
        assert not p.requires_grad


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_backward_flows_to_predictor_and_online_features():
    """Loss backward should reach predictor params and the input features leaf.

    The stub encoder returns its fixture tensor unchanged, so to make the leaf
    graph-attached we mark its features with ``requires_grad=True`` (a real
    encoder produces these via trainable layers, but the shim does not). The
    W22 zero-init on the last block's ``linear2`` means deep blocks in the
    trunk can receive effectively-zero gradient during warm-start — we just
    check the autograd graph attaches (``.grad is not None``), not magnitude.
    """
    obj = _build_objective()
    obj.online_encoder._feats.requires_grad_(True)
    video = torch.zeros(1, TOTAL_FRAMES, 3, 16, 16)
    loss = obj(video)
    loss.backward()
    for name, p in obj.predictor.named_parameters():
        assert p.grad is not None, f"no grad for predictor.{name}"
    assert obj.online_encoder._feats.grad is not None


def test_target_encoder_does_not_receive_grads():
    """Target encoder stays a no-grad consumer (grads flow from EMA, not autograd)."""
    obj = _build_objective()
    video = torch.zeros(1, TOTAL_FRAMES, 3, 16, 16)
    loss = obj(video)
    loss.backward()
    assert obj.target_encoder.online_p_a.grad is None
    assert obj.target_encoder.online_p_b.grad is None


# ---------------------------------------------------------------------------
# Signature / hyperparams / state_dict
# ---------------------------------------------------------------------------


def test_signature_includes_temporal_fields():
    obj = _build_objective()
    sig = obj.signature()
    for key in (
        "pred_dim", "pred_depth", "pred_heads", "encoder_dim",
        "ema_schedule", "loss_type",
        "past_frames", "future_frames", "past_grid", "future_grid",
    ):
        assert key in sig, f"missing signature field {key}"
    assert sig["past_frames"] == PAST_FRAMES
    assert sig["future_frames"] == FUTURE_FRAMES
    assert sig["past_grid"] == list(PAST_GRID)
    assert sig["future_grid"] == list(FUTURE_GRID)


def test_hyperparams_has_momentum_end_and_loss_beta():
    obj = _build_objective(beta=0.5)
    hp = obj.hyperparams()
    assert hp["momentum_end"] == 0.5
    assert hp["loss_beta"] == 0.5


def test_state_dict_round_trip_preserves_predictor_and_ema():
    obj1 = _build_objective()
    payload = obj1.state_dict()

    obj2 = _build_objective()
    obj2.after_step()
    assert obj2.ema.step == 1

    obj2.load_state_dict(payload)
    assert obj2.ema.step == 0  # restored
    for (n1, p1), (n2, p2) in zip(
        obj1.predictor.named_parameters(),
        obj2.predictor.named_parameters(),
    ):
        assert n1 == n2
        assert torch.equal(p1, p2), f"predictor.{n1} not restored"


def test_state_dict_contains_expected_keys():
    obj = _build_objective()
    payload = obj.state_dict()
    for key in (
        "predictor_state", "ema_state", "ema_step",
        "temporal_signature", "temporal_hyperparams",
    ):
        assert key in payload, f"state_dict missing {key!r}"
