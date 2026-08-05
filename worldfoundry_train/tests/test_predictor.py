"""Unit tests for JEPAPredictor (W22.1, design §9.1 test_predictor).

Covers: forward shape, all-False / all-True mask boundaries, gradient flow to
mask_token/pos_embed/projections, and the variable-grid trilinear
interpolation path.
"""
from __future__ import annotations

import pytest
import torch

from worldfoundry_train.model.predictor import JEPAPredictor

# Tiny config: grid (1, 4, 4) -> N=16, encoder 32, predictor 16.
GRID = (1, 4, 4)
N = 16
ENC_DIM = 32
PRED_DIM = 16


def _predictor(**overrides) -> JEPAPredictor:
    kwargs = dict(
        num_patches=N,
        grid_size=GRID,
        encoder_dim=ENC_DIM,
        predictor_dim=PRED_DIM,
        depth=2,
        num_heads=2,
    )
    kwargs.update(overrides)
    return JEPAPredictor(**kwargs)


def test_forward_shape_masked_positions_only():
    torch.manual_seed(0)
    pred = _predictor()
    feats = torch.randn(2, N, ENC_DIM)
    mask = torch.rand(2, N) < 0.5
    out = pred(feats, mask)
    m = int(mask.sum())
    assert out.shape == (m, ENC_DIM)


def test_all_false_mask_returns_empty_tensor_no_error():
    pred = _predictor()
    feats = torch.randn(2, N, ENC_DIM)
    mask = torch.zeros(2, N, dtype=torch.bool)
    out = pred(feats, mask)
    assert out.shape == (0, ENC_DIM)


def test_all_true_mask_boundary():
    pred = _predictor()
    feats = torch.randn(2, N, ENC_DIM)
    mask = torch.ones(2, N, dtype=torch.bool)
    out = pred(feats, mask)
    assert out.shape == (2 * N, ENC_DIM)
    assert torch.isfinite(out).all()


def test_backward_reaches_mask_token_pos_embed_and_projections():
    torch.manual_seed(0)
    pred = _predictor()
    feats = torch.randn(2, N, ENC_DIM, requires_grad=True)
    mask = torch.rand(2, N) < 0.5
    while int(mask.sum()) == 0:  # ensure non-empty regression target
        mask = torch.rand(2, N) < 0.5
    out = pred(feats, mask)
    target = torch.randn_like(out)
    loss = torch.nn.functional.smooth_l1_loss(out, target)
    loss.backward()

    for name, p in [
        ("mask_token", pred.mask_token),
        ("pos_embed", pred.pos_embed),
        ("proj_in.weight", pred.proj_in.weight),
        ("proj_out.weight", pred.proj_out.weight),
    ]:
        assert p.grad is not None, f"no grad for {name}"
        assert p.grad.abs().sum() > 0, f"zero grad for {name}"
    assert feats.grad is not None and feats.grad.abs().sum() > 0


def test_variable_grid_trilinear_interpolation_path():
    torch.manual_seed(0)
    pred = _predictor()  # built for grid (1, 4, 4)
    # Batch with a different grid: (2, 4, 4) -> N=32 tokens.
    feats = torch.randn(1, 32, ENC_DIM)
    mask = torch.rand(1, 32) < 0.5
    while int(mask.sum()) == 0:
        mask = torch.rand(1, 32) < 0.5
    out = pred(feats, mask, grid_size=(2, 4, 4))
    assert out.shape == (int(mask.sum()), ENC_DIM)
    assert torch.isfinite(out).all()


def test_grid_size_must_match_num_patches():
    with pytest.raises(ValueError, match="num_patches"):
        _predictor(grid_size=(2, 4, 4))  # implies 32 != N=16


def test_mask_shape_mismatch_raises():
    pred = _predictor()
    feats = torch.randn(2, N, ENC_DIM)
    with pytest.raises(ValueError, match="bool_masked_pos"):
        pred(feats, torch.zeros(2, N + 1, dtype=torch.bool))


def test_zero_init_last_block_output_projection():
    pred = _predictor()
    last = pred.blocks.layers[-1]
    assert torch.equal(last.linear2.weight, torch.zeros_like(last.linear2.weight))
    assert torch.equal(last.linear2.bias, torch.zeros_like(last.linear2.bias))


# ---------------------------------------------------------------------------
# Review addendum 2026-08-05: M1 (trilinear axis coverage) + M2 (dtype stays
# in the input dtype, no silent fp32 upcast from torch.where / add).
# ---------------------------------------------------------------------------


def test_bf16_input_keeps_predictor_output_in_bf16():
    """Review M2: the predictor must not silently upcast to fp32 on bf16 input.

    Without the dtype cache, ``torch.where(mask_token, x)`` with bf16 x and
    fp32 mask_token promoted the result to fp32, doubling predictor activation
    memory and adding O(N) extra dtype-promotion ops per forward.

    Mirrors production: predictor weights are cast to the encoder's dtype
    via ``.to(bfloat16)`` so that ``proj_in`` accepts bf16 inputs natively.
    """
    pred = _predictor().to(torch.bfloat16)
    feats = torch.randn(2, N, ENC_DIM, dtype=torch.bfloat16)
    mask = torch.rand(2, N) < 0.5
    while int(mask.sum()) == 0:
        mask = torch.rand(2, N) < 0.5
    out = pred(feats, mask)
    assert out.dtype == torch.bfloat16, (
        f"predictor output dtype {out.dtype} drifted from input bf16; "
        "the dtype cache in forward() is not working"
    )


def test_t_axis_trilinear_interpolation_finite():
    """Review M1: T-axis variation (1 -> 3 frames) must produce finite outputs.

    V-JEPA 2 typically uses T=8 or T=16, but trilinear on a small T axis can
    degenerate; we at minimum require finite values and correct shape.
    """
    torch.manual_seed(0)
    pred = _predictor()  # built for grid (1, 4, 4)
    new_grid = (3, 4, 4)
    new_n = new_grid[0] * new_grid[1] * new_grid[2]
    feats = torch.randn(1, new_n, ENC_DIM)
    mask = torch.rand(1, new_n) < 0.5
    while int(mask.sum()) == 0:
        mask = torch.rand(1, new_n) < 0.5
    out = pred(feats, mask, grid_size=new_grid)
    assert out.shape == (int(mask.sum()), ENC_DIM)
    assert torch.isfinite(out).all(), "T-axis interpolation produced non-finite values"


def test_h_w_axis_trilinear_interpolation_finite():
    """Review M1: H/W-axis variation (4,4 -> 6,8) must produce finite outputs.

    Spatial axes are typically larger than T, but the trilinear path is the
    same code — verify no degenerate case in the common H/W-resize path.
    """
    torch.manual_seed(0)
    pred = _predictor()  # built for grid (1, 4, 4)
    new_grid = (1, 6, 8)
    new_n = new_grid[0] * new_grid[1] * new_grid[2]
    feats = torch.randn(1, new_n, ENC_DIM)
    mask = torch.rand(1, new_n) < 0.5
    while int(mask.sum()) == 0:
        mask = torch.rand(1, new_n) < 0.5
    out = pred(feats, mask, grid_size=new_grid)
    assert out.shape == (int(mask.sum()), ENC_DIM)
    assert torch.isfinite(out).all(), "H/W-axis interpolation produced non-finite values"


def test_grid_size_unchanged_skips_interpolation():
    """Construct grid == forward grid: pos_embed returned directly (no interp).

    The fast-path branch in ``_pos_embed_for`` returns ``self.pos_embed``
    unchanged when grid_size matches; verify it runs without error.
    """
    pred = _predictor()  # grid (1, 4, 4)
    feats = torch.randn(1, N, ENC_DIM)
    mask = torch.rand(1, N) < 0.5
    while int(mask.sum()) == 0:
        mask = torch.rand(1, N) < 0.5
    # No explicit grid_size -> defaults to self.grid_size; must hit fast path.
    out = pred(feats, mask)
    assert out.shape == (int(mask.sum()), ENC_DIM)
    assert torch.isfinite(out).all()
