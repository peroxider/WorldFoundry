import numpy as np
import pytest

from worldforge_common.exceptions import MaskingError
from worldforge_train.data.tubelet_mask import multiblock_mask, tubelet_mask


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