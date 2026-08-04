"""Unit tests for worldfoundry_train.data.collator (mock-free, fast)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from worldfoundry_train.data.collator import collate_vjepa2_batch


def _np_sample(T=4, H=28, W=28, Hg=2, Wg=2, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "frames": rng.random((T, 3, H, W), dtype=np.float32),
        "mask": rng.integers(0, 2, size=(T, Hg, Wg), dtype=bool),
    }


def _torch_sample(T=4, H=28, W=28, Hg=2, Wg=2, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "frames": torch.from_numpy(rng.random((T, 3, H, W), dtype=np.float32)),
        "mask": torch.from_numpy(rng.integers(0, 2, size=(T, Hg, Wg), dtype=bool)),
    }


def test_collate_stacks_three_samples_to_expected_shapes():
    torch.manual_seed(42)
    samples = [_np_sample(seed=i) for i in range(3)]
    out = collate_vjepa2_batch(samples)
    assert out["pixel_values_videos"].shape == (3, 4, 3, 28, 28)
    assert out["pixel_values_videos"].dtype == torch.float32
    assert out["bool_masked_pos"].shape == (3, 4 * 2 * 2)
    assert out["bool_masked_pos"].dtype == torch.bool


def test_collate_accepts_torch_tensor_samples_directly():
    torch.manual_seed(42)
    samples = [_torch_sample(seed=i) for i in range(2)]
    out = collate_vjepa2_batch(samples)
    assert out["pixel_values_videos"].shape == (2, 4, 3, 28, 28)
    assert out["bool_masked_pos"].shape == (2, 4 * 2 * 2)


def test_collate_preserves_values():
    torch.manual_seed(42)
    samples = [_np_sample(seed=i) for i in range(2)]
    out = collate_vjepa2_batch(samples)
    expected = torch.from_numpy(samples[0]["frames"]).float()
    assert torch.equal(out["pixel_values_videos"][0], expected)


def test_collate_preserves_mask_positions():
    torch.manual_seed(42)
    samples = [_np_sample(seed=i) for i in range(2)]
    out = collate_vjepa2_batch(samples)
    expected_mask = torch.from_numpy(samples[1]["mask"]).bool().reshape(-1)
    assert torch.equal(out["bool_masked_pos"][1], expected_mask)


def test_collate_empty_list_raises_value_error():
    with pytest.raises(ValueError, match="empty list"):
        collate_vjepa2_batch([])


def test_collate_mismatched_frame_shape_raises_value_error():
    s0 = _np_sample(H=28, W=28, seed=0)
    s1 = _np_sample(H=28, W=56, seed=1)
    with pytest.raises(ValueError, match="frames shape"):
        collate_vjepa2_batch([s0, s1])


def test_collate_mismatched_T_raises_value_error():
    s0 = _np_sample(T=4, seed=0)
    s1 = _np_sample(T=8, seed=1)
    with pytest.raises(ValueError, match="frames shape"):
        collate_vjepa2_batch([s0, s1])


def test_collate_non_tensor_frames_raises_type_error():
    # A bare Python object (not list/ndarray/Tensor) is rejected by
    # torch.as_tensor itself, surfacing as TypeError or RuntimeError.
    bad = {"frames": object(), "mask": np.zeros((1, 1, 1), dtype=bool)}
    with pytest.raises((TypeError, RuntimeError)):
        collate_vjepa2_batch([bad])


def test_collate_missing_frame_key_raises_key_error():
    s = _np_sample()
    del s["frames"]
    with pytest.raises(KeyError):
        collate_vjepa2_batch([s])


def test_collate_missing_mask_key_raises_key_error():
    s = _np_sample()
    del s["mask"]
    with pytest.raises(KeyError):
        collate_vjepa2_batch([s])


def test_collate_masks_flattened_to_one_dim():
    torch.manual_seed(42)
    samples = [_np_sample(T=2, Hg=3, Wg=4, seed=i) for i in range(2)]
    out = collate_vjepa2_batch(samples)
    assert out["bool_masked_pos"].shape == (2, 2 * 3 * 4)


def test_collate_dtype_conversion_uint8_to_float32():
    torch.manual_seed(42)
    s = {
        "frames": np.random.randint(0, 256, size=(2, 3, 16, 16), dtype=np.uint8),
        "mask": np.ones((2, 1, 1), dtype=bool),
    }
    out = collate_vjepa2_batch([s, s])
    assert out["pixel_values_videos"].dtype == torch.float32
    assert out["pixel_values_videos"].shape == (2, 2, 3, 16, 16)