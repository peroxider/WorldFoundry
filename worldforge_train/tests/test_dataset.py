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
    T, C, H, W = item["frames"].shape
    assert (T, C, H, W) == (20, 3, 64, 64)
    assert item["frames"].dtype == np.float32
    assert 0.0 <= item["frames"].min() and item["frames"].max() <= 1.0
    # mask: (T, H_grid, W_grid) bool; cross-check each axis against frames
    assert item["mask"].dtype == bool
    T_m, Hg, Wg = item["mask"].shape
    assert T == T_m == 20
    assert C == 3
    assert H == W == 64
    assert (Hg, Wg) == (H // 14, W // 14)
    assert (Hg, Wg) == (4, 4)


def test_dataset_getitem_mask_grid_scales_with_patch_size(manifest):
    # patch_size=8 on a 64x64 frame -> (20, 8, 8) grid
    ds = Vjepa2Dataset(
        manifest_path=manifest, patch_size=8,
        mask_fn=lambda shape, rng: np.zeros(shape, dtype=bool),
    )
    item = ds[0]
    # Cross-check that the grid axes derive from the frames shape, not just
    # the total count, so a swap of (H_grid, W_grid) would not pass.
    assert item["mask"].shape == (20, 8, 8)
    assert item["frames"].shape[2] == item["frames"].shape[3] == 64
    assert 8 == 64 // 8


def test_dataset_raises_when_patch_size_exceeds_height_or_width(tmp_path, sample_mp4):
    csv = tmp_path / "manifest.csv"
    csv.write_text(f"video_path,num_frames,height,width\n{sample_mp4},20,64,64\n")
    with pytest.raises(ValueError):
        Vjepa2Dataset(
            manifest_path=csv, patch_size=70,
            mask_fn=lambda shape, rng: np.zeros(shape, dtype=bool),
        )


def test_dataset_raises_for_wrong_manifest_schema(tmp_path, sample_mp4):
    # Drop num_frames: schema becomes (video_path, height, width).
    csv = tmp_path / "manifest.csv"
    csv.write_text(f"video_path,height,width\n{sample_mp4},64,64\n")
    with pytest.raises(AssertionError):
        Vjepa2Dataset(
            manifest_path=csv,
            mask_fn=lambda shape, rng: np.zeros(shape, dtype=bool),
        )


def test_dataset_raises_for_missing_manifest(tmp_path):
    with pytest.raises(FileNotFoundError):
        Vjepa2Dataset(
            manifest_path=tmp_path / "nope.csv",
            mask_fn=lambda shape, rng: np.zeros(shape, dtype=bool),
        )


def test_dataset_raises_for_missing_video(tmp_path):
    csv = tmp_path / "manifest.csv"
    csv.write_text(
        "video_path,num_frames,height,width\n"
        f"{tmp_path / 'missing.mp4'},20,64,64\n"
    )
    ds = Vjepa2Dataset(
        manifest_path=csv,
        mask_fn=lambda shape, rng: np.zeros(shape, dtype=bool),
    )
    with pytest.raises(FileNotFoundError):
        ds[0]


def test_dataset_raises_for_zero_patch_size(manifest):
    with pytest.raises(ValueError):
        Vjepa2Dataset(
            manifest_path=manifest, patch_size=0,
            mask_fn=lambda shape, rng: np.zeros(shape, dtype=bool),
        )
