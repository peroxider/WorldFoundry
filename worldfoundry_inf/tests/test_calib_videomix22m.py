"""calib_videomix22m loader tests — mock HF datasets to avoid network in CI."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch


def _make_fake_dataset(n: int):
    """Return a MagicMock standing in for a HF Dataset with `n` samples."""
    samples = [{"video": [MagicMock() for _ in range(16)]} for _ in range(n)]
    fake = MagicMock()
    fake.shuffle.return_value.select.return_value = samples
    return fake, samples


def test_load_returns_n_tensors():
    """load_calibration_dataset(n=8) yields exactly 8 tensors."""
    from worldfoundry_inf.data.calib_videomix22m import load_calibration_dataset

    fake_ds, _ = _make_fake_dataset(8)

    # Mock processor to return deterministic (3, 224, 224) tensors
    def fake_processor(images, return_tensors):
        return {"pixel_values": torch.zeros(1, 3, 224, 224)}

    with patch(
        "worldfoundry_inf.data.calib_videomix22m.load_dataset", return_value=fake_ds
    ):
        with patch(
            "worldfoundry_inf.data.calib_videomix22m.AutoImageProcessor.from_pretrained"
        ) as fake_proc_factory:
            fake_proc_factory.return_value = MagicMock(side_effect=fake_processor)
            tensors = list(load_calibration_dataset(n=8, seed=42))

    assert len(tensors) == 8
    assert all(t.shape == (3, 224, 224) for t in tensors)


def test_load_propagates_dataset_error():
    """HF datasets errors propagate (no silent fallback — G4)."""
    from worldfoundry_inf.data.calib_videomix22m import load_calibration_dataset

    with patch(
        "worldfoundry_inf.data.calib_videomix22m.load_dataset",
        side_effect=RuntimeError("network unreachable"),
    ):
        with pytest.raises(RuntimeError, match="network unreachable"):
            list(load_calibration_dataset(n=4, seed=42))
