import numpy as np
import pytest

from worldforge_train.data.dataset import Vjepa2Dataset


@pytest.fixture
def manifest(tmp_path, sample_mp4):
    """Write a tiny manifest CSV: video_path, num_frames, height, width."""
    csv = tmp_path / "manifest.csv"
    csv.write_text(f"video_path,num_frames,height,width\n{sample_mp4},20,64,64\n")
    return csv


def test_dataset_len_matches_manifest(manifest):
    ds = Vjepa2Dataset(
        manifest_path=manifest,
        mask_fn=lambda shape, rng: np.ones(shape, dtype=bool),
    )
    assert len(ds) == 1


def test_dataset_getitem_returns_correct_shapes(manifest):
    ds = Vjepa2Dataset(
        manifest_path=manifest,
        mask_fn=lambda shape, rng: np.ones(shape, dtype=bool),
    )
    item = ds[0]
    assert "frames" in item and "mask" in item
    # frames: (T, C, H, W) float32 in [0,1]
    assert item["frames"].shape == (20, 3, 64, 64)
    assert item["frames"].dtype == np.float32
    assert 0.0 <= item["frames"].min() and item["frames"].max() <= 1.0
    # mask: (T, H_grid, W_grid) bool with H_grid, W_grid = H/patch_size, W/patch_size
    assert item["mask"].dtype == bool
    assert item["mask"].shape == (20, 4, 4)


def test_dataset_getitem_mask_grid_scales_with_patch_size(manifest):
    # patch_size=8 on a 64x64 frame -> (20, 8, 8) grid
    ds = Vjepa2Dataset(
        manifest_path=manifest, patch_size=8,
        mask_fn=lambda shape, rng: np.zeros(shape, dtype=bool),
    )
    item = ds[0]
    assert item["mask"].shape == (20, 8, 8)
