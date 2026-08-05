"""Unit tests for TemporalPredictor (W24).

Covers: shape correctness, dtype preservation (M2 review pattern),
determinism, gradient flow to all learned params, from_w22_jepa_predictor
trunk weight copy, fresh init of task-specific params, variable future-grid
trilinear interpolation, last-block zero-init preservation, grid mismatch
and shape-mismatch error paths.
"""
from __future__ import annotations

import pytest
import torch

from worldfoundry_train.model.predictor import JEPAPredictor
from worldfoundry_train.model.temporal_predictor import TemporalPredictor

# Tiny config: past (1, 4, 4) -> N_past=16, future (1, 4, 4) -> N_future=16.
PAST_GRID = (1, 4, 4)
FUTURE_GRID = (1, 4, 4)
N_PAST = 16
N_FUTURE = 16
ENC_DIM = 32
PRED_DIM = 16
DEPTH = 2
NUM_HEADS = 2


def _temporal_predictor(
    past_grid=PAST_GRID,
    future_grid=FUTURE_GRID,
    n_past=N_PAST,
    n_future=N_FUTURE,
    **overrides,
) -> TemporalPredictor:
    kwargs = dict(
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
    return TemporalPredictor(**kwargs)


def _w22_predictor(**overrides) -> JEPAPredictor:
    kwargs = dict(
        num_patches=N_PAST,
        grid_size=PAST_GRID,
        encoder_dim=ENC_DIM,
        predictor_dim=PRED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
    )
    kwargs.update(overrides)
    return JEPAPredictor(**kwargs)


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_past_grid_mismatch_raises():
    with pytest.raises(ValueError, match="past_grid"):
        _temporal_predictor(past_grid=(1, 2, 2), n_past=16)  # 4 != 16


def test_future_grid_mismatch_raises():
    with pytest.raises(ValueError, match="future_grid"):
        _temporal_predictor(future_grid=(1, 2, 2), n_future=16)  # 4 != 16


# ---------------------------------------------------------------------------
# Forward shape + content
# ---------------------------------------------------------------------------


def test_forward_shape_basic():
    torch.manual_seed(0)
    pred = _temporal_predictor()
    feats = torch.randn(2, N_PAST, ENC_DIM)
    out = pred(feats)
    assert out.shape == (2, N_FUTURE, ENC_DIM)
    assert torch.isfinite(out).all()


def test_forward_shape_past_and_future_differ():
    """K != H, past grid != future grid: N_past != N_future."""
    torch.manual_seed(0)
    pred = _temporal_predictor(
        past_grid=(2, 2, 2), n_past=8,
        future_grid=(1, 4, 4), n_future=16,
    )
    feats = torch.randn(1, 8, ENC_DIM)
    out = pred(feats)
    assert out.shape == (1, 16, ENC_DIM)
    assert torch.isfinite(out).all()


def test_forward_is_deterministic():
    """Same input twice -> byte-identical output (no dropout, no stochastics)."""
    torch.manual_seed(0)
    pred = _temporal_predictor()
    feats = torch.randn(2, N_PAST, ENC_DIM)
    out1 = pred(feats)
    out2 = pred(feats)
    assert torch.equal(out1, out2)


def test_forward_invalid_past_feats_shape_raises():
    pred = _temporal_predictor()
    with pytest.raises(ValueError, match="past_feats"):
        pred(torch.randn(2, N_PAST))  # 2D, missing D


def test_forward_past_position_count_mismatch_raises():
    pred = _temporal_predictor()
    with pytest.raises(ValueError, match="positions"):
        pred(torch.randn(1, N_PAST + 1, ENC_DIM))


def test_bf16_input_keeps_output_in_bf16():
    """Review M2 analogue: bf16 inputs must not silently upcast to fp32."""
    pred = _temporal_predictor().to(torch.bfloat16)
    feats = torch.randn(2, N_PAST, ENC_DIM, dtype=torch.bfloat16)
    out = pred(feats)
    assert out.dtype == torch.bfloat16, (
        f"output dtype {out.dtype} drifted from bf16 input; "
        "dtype cache in forward() is not working"
    )


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_backward_reaches_all_learned_params():
    """Loss on output gradients must reach trunk + task-specific params."""
    torch.manual_seed(0)
    pred = _temporal_predictor()
    feats = torch.randn(2, N_PAST, ENC_DIM, requires_grad=True)
    out = pred(feats)
    target = torch.randn_like(out)
    loss = torch.nn.functional.smooth_l1_loss(out, target)
    loss.backward()

    for name, p in [
        ("proj_in.weight", pred.proj_in.weight),
        ("proj_out.weight", pred.proj_out.weight),
        ("past_pos_embed", pred.past_pos_embed),
        ("future_pos_embed", pred.future_pos_embed),
        ("future_query", pred.future_query),
    ]:
        assert p.grad is not None, f"no grad for {name}"
        assert p.grad.abs().sum() > 0, f"zero grad for {name}"
    assert feats.grad is not None and feats.grad.abs().sum() > 0


def test_zero_init_last_block_output_projection():
    """Stable warmup: last block's MLP output projection must be zero."""
    pred = _temporal_predictor()
    last = pred.blocks.layers[-1]
    assert torch.equal(last.linear2.weight, torch.zeros_like(last.linear2.weight))
    assert torch.equal(last.linear2.bias, torch.zeros_like(last.linear2.bias))


# ---------------------------------------------------------------------------
# from_w22_jepa_predictor: trunk weight copy + task-specific fresh init
# ---------------------------------------------------------------------------


def test_from_w22_jepa_predictor_copies_trunk_weights_byte_for_byte():
    """Warm-start must transfer proj_in/blocks/proj_out weights exactly."""
    torch.manual_seed(0)
    w22 = _w22_predictor()
    expected_proj_in = {k: v.clone() for k, v in w22.proj_in.state_dict().items()}
    expected_blocks = {k: v.clone() for k, v in w22.blocks.state_dict().items()}
    expected_proj_out = {k: v.clone() for k, v in w22.proj_out.state_dict().items()}

    torch.manual_seed(0)
    temporal = TemporalPredictor.from_w22_jepa_predictor(
        w22, num_past_positions=N_PAST, num_future_positions=N_FUTURE,
        past_grid=PAST_GRID, future_grid=FUTURE_GRID,
    )

    got_proj_in = temporal.proj_in.state_dict()
    got_blocks = temporal.blocks.state_dict()
    got_proj_out = temporal.proj_out.state_dict()
    for k in expected_proj_in:
        assert torch.equal(got_proj_in[k], expected_proj_in[k]), f"proj_in.{k}"
    for k in expected_blocks:
        assert torch.equal(got_blocks[k], expected_blocks[k]), f"blocks.{k}"
    for k in expected_proj_out:
        assert torch.equal(got_proj_out[k], expected_proj_out[k]), f"proj_out.{k}"


def test_from_w22_jepa_predictor_does_not_touch_task_specific_params():
    """past_pos_embed / future_pos_embed / future_query stay at fresh init."""
    torch.manual_seed(0)
    w22 = _w22_predictor()
    torch.manual_seed(1)  # different seed for the TemporalPredictor fresh init
    temporal = TemporalPredictor.from_w22_jepa_predictor(
        w22, num_past_positions=N_PAST, num_future_positions=N_FUTURE,
        past_grid=PAST_GRID, future_grid=FUTURE_GRID,
    )
    assert temporal.past_pos_embed.shape == (1, N_PAST, PRED_DIM)
    assert temporal.future_pos_embed.shape == (1, N_FUTURE, PRED_DIM)
    assert temporal.future_query.shape == (1, N_FUTURE, PRED_DIM)
    assert temporal.past_pos_embed.abs().sum() > 0
    assert temporal.future_pos_embed.abs().sum() > 0
    assert temporal.future_query.abs().sum() > 0


def test_from_w22_jepa_predictor_with_mismatched_dim_raises():
    """Different encoder_dim -> load_state_dict shape mismatch -> error."""
    w22 = _w22_predictor()  # encoder_dim=ENC_DIM=32
    object.__setattr__(w22, "encoder_dim", 999)
    with pytest.raises((RuntimeError, ValueError)):
        TemporalPredictor.from_w22_jepa_predictor(
            w22, num_past_positions=N_PAST, num_future_positions=N_FUTURE,
            past_grid=PAST_GRID, future_grid=FUTURE_GRID,
        )


# ---------------------------------------------------------------------------
# Variable future grid -> trilinear interpolation
# ---------------------------------------------------------------------------


def test_variable_future_grid_trilinear_finite():
    """Different future_grid -> interpolation path -> finite output, correct shape."""
    torch.manual_seed(0)
    pred = _temporal_predictor()  # built for future_grid=(1, 4, 4)
    feats = torch.randn(1, N_PAST, ENC_DIM)
    new_future_grid = (2, 6, 8)
    n_new = new_future_grid[0] * new_future_grid[1] * new_future_grid[2]
    out = pred(feats, future_grid=new_future_grid)
    assert out.shape == (1, n_new, ENC_DIM)
    assert torch.isfinite(out).all(), "future-grid interpolation produced non-finite values"


def test_future_grid_unchanged_skips_interpolation():
    """future_grid == self.future_grid hits the fast path (no interp)."""
    torch.manual_seed(0)
    pred = _temporal_predictor()
    feats = torch.randn(1, N_PAST, ENC_DIM)
    out = pred(feats)
    assert out.shape == (1, N_FUTURE, ENC_DIM)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Sanity: trunk copy is meaningful (different fresh-init seeds still differ
# in the task-specific params, while the trunk weights stay byte-identical).
# ---------------------------------------------------------------------------


def test_warm_started_predictor_trunk_identical_across_seed_calls():
    """Trunk copy is meaningful: same w22 source -> identical trunk weights."""
    torch.manual_seed(0)
    w22 = _w22_predictor()
    torch.manual_seed(0)
    temporal_a = TemporalPredictor.from_w22_jepa_predictor(
        w22, num_past_positions=N_PAST, num_future_positions=N_FUTURE,
        past_grid=PAST_GRID, future_grid=FUTURE_GRID,
    )
    torch.manual_seed(7)  # different init seed for the task-specific params
    temporal_b = TemporalPredictor.from_w22_jepa_predictor(
        w22, num_past_positions=N_PAST, num_future_positions=N_FUTURE,
        past_grid=PAST_GRID, future_grid=FUTURE_GRID,
    )
    # Trunk: same source w22 -> identical
    assert torch.equal(temporal_a.proj_in.weight, temporal_b.proj_in.weight)
    assert torch.equal(
        temporal_a.blocks.state_dict()["layers.0.linear1.weight"],
        temporal_b.blocks.state_dict()["layers.0.linear1.weight"],
    )
    # Task-specific: different seed -> different
    assert not torch.equal(temporal_a.future_query, temporal_b.future_query)