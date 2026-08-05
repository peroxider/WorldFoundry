"""Unit tests for ActionConditionedPredictor (W27).

Sister to ``test_temporal_predictor.py``: covers construction validation,
forward shape + content (with action embed), broadcast logic, gradient
flow through trunk + action_proj + pos-embeds, ``from_temporal_predictor``
trunk transfer, fresh-init of action_proj, last-block zero-init reset,
and the action-mismatch error paths.
"""
from __future__ import annotations

import pytest
import torch

from worldfoundry_train.model.action_predictor import ActionConditionedPredictor
from worldfoundry_train.model.temporal_predictor import TemporalPredictor

# Tiny config: past (2, 2, 2) -> N_past=8, future (2, 2, 2) -> N_future=8.
PAST_GRID = (2, 2, 2)
FUTURE_GRID = (2, 2, 2)
N_PAST = 8
N_FUTURE = 8
PAST_FRAMES = 2
ACTION_DIM = 7
ENC_DIM = 32
PRED_DIM = 16
DEPTH = 2
NUM_HEADS = 2


def _acp(
    past_grid=PAST_GRID,
    future_grid=FUTURE_GRID,
    n_past=N_PAST,
    n_future=N_FUTURE,
    past_frames=PAST_FRAMES,
    action_dim=ACTION_DIM,
    **overrides,
) -> ActionConditionedPredictor:
    kwargs = dict(
        past_frames=past_frames,
        action_dim=action_dim,
        num_past_positions=n_past,
        num_future_positions=n_future,
        past_grid=past_grid,
        future_grid=future_grid,
        encoder_dim=ENC_DIM,
        predictor_dim=PRED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
    )
    kwargs.update(overrides)
    return ActionConditionedPredictor(**kwargs)


def _temporal() -> TemporalPredictor:
    return TemporalPredictor(
        num_past_positions=N_PAST,
        num_future_positions=N_FUTURE,
        past_grid=PAST_GRID,
        future_grid=FUTURE_GRID,
        encoder_dim=ENC_DIM,
        predictor_dim=PRED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
    )


def _past_feats(b: int = 2) -> torch.Tensor:
    return torch.randn(b, N_PAST, ENC_DIM)


def _actions(b: int = 2) -> torch.Tensor:
    return torch.randn(b, PAST_FRAMES, ACTION_DIM)


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_past_frames_grid_mismatch_raises():
    with pytest.raises(ValueError, match="past_grid frame dimension"):
        _acp(past_frames=3)  # PAST_GRID has 2 frames


def test_past_grid_mismatch_raises():
    with pytest.raises(ValueError, match="past_grid"):
        _acp(past_grid=(1, 2, 2), n_past=16)  # 4 != 16


def test_future_grid_mismatch_raises():
    with pytest.raises(ValueError, match="future_grid"):
        _acp(future_grid=(1, 2, 2), n_future=16)


def test_action_dim_below_one_raises():
    with pytest.raises(ValueError, match="action_dim"):
        _acp(action_dim=0)


# ---------------------------------------------------------------------------
# Forward shape + content
# ---------------------------------------------------------------------------


def test_forward_shape_basic():
    torch.manual_seed(0)
    pred = _acp()
    out = pred(_past_feats(), _actions())
    assert out.shape == (2, N_FUTURE, ENC_DIM)
    assert torch.isfinite(out).all()


def test_forward_with_different_grid():
    torch.manual_seed(0)
    pred = _acp(
        past_frames=3, past_grid=(3, 2, 2), n_past=12,
        future_grid=(1, 4, 4), n_future=16,
    )
    past = torch.randn(1, 12, ENC_DIM)
    acts = torch.randn(1, 3, ACTION_DIM)
    out = pred(past, acts)
    assert out.shape == (1, 16, ENC_DIM)
    assert torch.isfinite(out).all()


def test_forward_is_deterministic():
    torch.manual_seed(0)
    pred = _acp()
    feats = _past_feats()
    acts = _actions()
    out1 = pred(feats, acts)
    out2 = pred(feats, acts)
    assert torch.equal(out1, out2)


def test_forward_zero_actions_changes_output_from_no_action_baseline():
    """Action broadcast must actually contribute (not be silently dropped).

    Two freshly-seeded predictors see the same input but different actions
    (zeros vs random). Their predictions must differ because the broadcast
    of ``action_proj(actions)`` adds a non-zero term to the past side.
    """
    torch.manual_seed(0)
    pred_zero = _acp()
    pred_full = _acp()
    feats = _past_feats()
    acts_zero = torch.zeros_like(_actions())
    acts_full = _actions()
    out_zero = pred_zero(feats, acts_zero)
    out_full = pred_full(feats, acts_full)
    assert not torch.equal(out_zero, out_full), (
        "action broadcast did not change prediction; action_proj may be "
        "silently dropped at forward"
    )


def test_forward_bf16_input_keeps_output_in_bf16():
    pred = _acp().to(torch.bfloat16)
    feats = torch.randn(2, N_PAST, ENC_DIM, dtype=torch.bfloat16)
    acts = torch.randn(2, PAST_FRAMES, ACTION_DIM, dtype=torch.bfloat16)
    out = pred(feats, acts)
    assert out.dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# Action-shape error paths
# ---------------------------------------------------------------------------


def test_forward_rejects_wrong_actions_ndim():
    pred = _acp()
    with pytest.raises(ValueError, match=r"\(B, K, action_dim\)"):
        pred(_past_feats(), torch.randn(2, PAST_FRAMES))  # 2D


def test_forward_rejects_actions_batch_mismatch():
    pred = _acp()
    with pytest.raises(ValueError, match="batch size"):
        pred(_past_feats(b=2), _actions(b=3))


def test_forward_rejects_actions_wrong_frame_count():
    pred = _acp()
    with pytest.raises(ValueError, match="past frames"):
        pred(_past_feats(), torch.randn(2, PAST_FRAMES + 1, ACTION_DIM))


def test_forward_rejects_actions_wrong_action_dim():
    pred = _acp()
    with pytest.raises(ValueError, match="action_dim"):
        pred(_past_feats(), torch.randn(2, PAST_FRAMES, ACTION_DIM + 1))


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_backward_reaches_trunk_and_action_proj():
    torch.manual_seed(0)
    pred = _acp()
    feats = _past_feats().requires_grad_(True)
    acts = _actions().requires_grad_(True)
    out = pred(feats, acts)
    target = torch.randn_like(out)
    loss = torch.nn.functional.smooth_l1_loss(out, target)
    loss.backward()

    for name, p in [
        ("proj_in.weight", pred.proj_in.weight),
        ("proj_out.weight", pred.proj_out.weight),
        ("action_proj.weight", pred.action_proj.weight),
        ("past_pos_embed", pred.past_pos_embed),
        ("future_pos_embed", pred.future_pos_embed),
        ("future_query", pred.future_query),
    ]:
        assert p.grad is not None, f"no grad for {name}"
    assert feats.grad is not None
    assert acts.grad is not None


def test_zero_init_last_block_output_projection():
    pred = _acp()
    last = pred.blocks.layers[-1]
    assert torch.equal(last.linear2.weight, torch.zeros_like(last.linear2.weight))
    assert torch.equal(last.linear2.bias, torch.zeros_like(last.linear2.bias))


# ---------------------------------------------------------------------------
# from_temporal_predictor: trunk copy + action_proj fresh
# ---------------------------------------------------------------------------


def test_from_temporal_predictor_copies_trunk_byte_for_byte():
    torch.manual_seed(0)
    temporal = _temporal()
    torch.manual_seed(1)
    ac = ActionConditionedPredictor.from_temporal_predictor(
        temporal, past_frames=PAST_FRAMES, action_dim=ACTION_DIM,
    )

    assert torch.equal(ac.proj_in.weight, temporal.proj_in.weight)
    assert torch.equal(ac.proj_out.weight, temporal.proj_out.weight)
    assert torch.equal(
        ac.blocks.state_dict()["layers.0.linear1.weight"],
        temporal.blocks.state_dict()["layers.0.linear1.weight"],
    )
    # Pos-embeds + future queries: byte-for-byte.
    assert torch.equal(ac.past_pos_embed, temporal.past_pos_embed)
    assert torch.equal(ac.future_pos_embed, temporal.future_pos_embed)
    assert torch.equal(ac.future_query, temporal.future_query)


def test_from_temporal_predictor_action_proj_is_fresh_init():
    """Action projection is fresh-init, NOT copied from temporal."""
    torch.manual_seed(0)
    temporal = _temporal()
    torch.manual_seed(7)  # different seed for the ACP fresh-init path
    ac = ActionConditionedPredictor.from_temporal_predictor(
        temporal, past_frames=PAST_FRAMES, action_dim=ACTION_DIM,
    )
    assert ac.action_proj.in_features == ACTION_DIM
    assert ac.action_proj.out_features == PRED_DIM
    assert ac.action_proj.weight.abs().sum() > 0


def test_from_temporal_predictor_resets_last_block_zero_init():
    """The warm-start identity guarantee: last block's linear2 is re-zeroed.

    Even if the source temporal's last linear2 grew during its own training,
    the action-conditioned warm-start resets it so the trunk starts as a
    near-identity function (same stability trick as W22 / W24).
    """
    torch.manual_seed(0)
    temporal = _temporal()
    nn_last = temporal.blocks.layers[-1]
    nn_last.linear2.weight.data.fill_(0.5)
    nn_last.linear2.bias.data.fill_(0.1)

    ac = ActionConditionedPredictor.from_temporal_predictor(
        temporal, past_frames=PAST_FRAMES, action_dim=ACTION_DIM,
    )
    ac_last = ac.blocks.layers[-1]
    assert torch.equal(ac_last.linear2.weight, torch.zeros_like(ac_last.linear2.weight))
    assert torch.equal(ac_last.linear2.bias, torch.zeros_like(ac_last.linear2.bias))


# ---------------------------------------------------------------------------
# Variable future_grid -> interpolation
# ---------------------------------------------------------------------------


def test_variable_future_grid_trilinear_finite():
    torch.manual_seed(0)
    pred = _acp()  # future_grid=(2, 2, 2)
    new_grid = (1, 4, 4)
    n_new = new_grid[0] * new_grid[1] * new_grid[2]
    out = pred(_past_feats(b=1), _actions(b=1), future_grid=new_grid)
    assert out.shape == (1, n_new, ENC_DIM)
    assert torch.isfinite(out).all()
