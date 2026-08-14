import numpy as np
import pytest

from worldfoundry_common.exceptions import MaskingError
from worldfoundry_train.data.tubelet_mask import (
    default_mask_tubelet_size,
    multiblock_mask,
    tubelet_mask,
)


def test_multiblock_mask_shape_and_dtype():
    shape = (8, 14, 14)
    mask = multiblock_mask(
        shape=shape, num_blocks=4, spatial_only=False, target_ratio=0.5,
        rng=np.random.default_rng(0),
    )
    assert mask.shape == shape
    assert mask.dtype == bool
    # blocks accumulate until target_ratio is reached
    assert mask.mean() >= 0.5


def test_multiblock_mask_spatial_only_zero_t():
    shape = (8, 14, 14)
    mask = multiblock_mask(
        shape=shape, num_blocks=4, spatial_only=True, target_ratio=0.5,
        rng=np.random.default_rng(0),
    )
    # spatial_only: same mask broadcast across T
    assert np.all(mask[0] == mask[1])


def test_tubelet_mask_is_subset_of_full():
    shape = (8, 14, 14)
    full = np.ones(shape, dtype=bool)
    masked = tubelet_mask(shape=shape, tubelet_size=(2, 14, 14), rng=np.random.default_rng(0))
    assert masked.shape == shape
    assert masked.sum() < full.sum()
    assert masked.dtype == bool


def test_multiblock_mask_raises_when_target_unreachable():
    # tiny max_attempts forces the unreachable-target failure mode; the message
    # must name both the achieved fraction and the target so a regression that
    # swaps the inequality or skips the failure branch is caught.
    with pytest.raises(MaskingError, match=r"of target 0\.99"):
        multiblock_mask(
            shape=(8, 14, 14), num_blocks=1, target_ratio=0.99,
            max_attempts=2, rng=np.random.default_rng(0),
        )


def test_tubelet_mask_raises_when_shape_not_divisible():
    # T=10 not divisible by tubelet_size_t=3 -> G8: explicit failure, no silent truncation
    with pytest.raises(MaskingError, match=r"not divisible"):
        tubelet_mask(
            shape=(10, 14, 14), tubelet_size=(3, 14, 14),
            rng=np.random.default_rng(0),
        )


# --- W2.25+: default_mask_tubelet_size auto-scaler ---

def test_default_mask_tubelet_size_smoke():
    """Smoke (4, 4, 4) -> (1, 1, 1) per-token masking."""
    assert default_mask_tubelet_size((4, 4, 4)) == (1, 1, 1)


def test_default_mask_tubelet_size_vitl_native():
    """ViT-L native (32, 16, 16) -> (8, 4, 4) yielding 64 blocks."""
    assert default_mask_tubelet_size((32, 16, 16)) == (8, 4, 4)


def test_default_mask_tubelet_size_paper_default():
    """Paper default (4, 14, 14) -> (1, 2, 2).

    The naive ``Hg//4 = 3`` does not divide ``Hg=14``; the helper walks
    down to the largest divisor (2) so the result is always
    shape-compatible.
    """
    assert default_mask_tubelet_size((4, 14, 14)) == (1, 2, 2)


def test_tubelet_mask_with_default_block_sizes_is_divisible():
    """The auto-scaled block size must always divide the input grid."""
    for grid in [(4, 4, 4), (32, 16, 16), (4, 14, 14)]:
        block = default_mask_tubelet_size(grid)
        assert all(s % b == 0 for s, b in zip(grid, block)), (
            f"grid {grid} not divisible by auto-block {block}"
        )
        # And actually produce a mask with no MaskingError
        mask = tubelet_mask(shape=grid, tubelet_size=block, rng=np.random.default_rng(0))
        assert mask.shape == grid
        assert mask.dtype == bool


# --- W2.25+: default_mask_tubelet_size auto-scaler ---

def test_default_mask_tubelet_size_smoke():
    """Smoke (4, 4, 4) -> (1, 1, 1) per-token masking."""
    assert default_mask_tubelet_size((4, 4, 4)) == (1, 1, 1)


def test_default_mask_tubelet_size_vitl_native():
    """ViT-L native (32, 16, 16) -> (8, 4, 4) yielding 64 blocks."""
    assert default_mask_tubelet_size((32, 16, 16)) == (8, 4, 4)


def test_default_mask_tubelet_size_paper_default():
    """Paper default (4, 14, 14) -> (1, 2, 2).

    The naive ``Hg//4 = 3`` does not divide ``Hg=14``; the helper walks
    down to the largest divisor (2) so the result is always
    shape-compatible.
    """
    assert default_mask_tubelet_size((4, 14, 14)) == (1, 2, 2)


def test_tubelet_mask_with_default_block_sizes_is_divisible():
    """The auto-scaled block size must always divide the input grid."""
    for grid in [(4, 4, 4), (32, 16, 16), (4, 14, 14)]:
        block = default_mask_tubelet_size(grid)
        assert all(s % b == 0 for s, b in zip(grid, block)), (
            f"grid {grid} not divisible by auto-block {block}"
        )
        # And actually produce a mask with no MaskingError
        mask = tubelet_mask(shape=grid, tubelet_size=block, rng=np.random.default_rng(0))
        assert mask.shape == grid
        assert mask.dtype == bool