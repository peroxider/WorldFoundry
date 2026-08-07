"""Build a Vjepa2Dataset-compatible CSV manifest from synthetic .mp4 files.

The orchestrator (RTI-3) calls :func:`build_synthetic_manifest` once
during the Render phase to generate ``num_clips`` real ``.mp4`` files
on disk and a matching CSV manifest that
:class:`worldfoundry_train.data.dataset.Vjepa2Dataset` can consume
without any schema change (its strict assertion
``reader.fieldnames == ["video_path", "num_frames", "height", "width"]``
drives this layout).

Paths in the CSV are written as absolute so the dataset can be loaded
from any working directory (matches the convention used by
``bench_results/w22_7/subset_manifest.csv``).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from worldfoundry_train.data.synthetic_video import write_synthetic_clip


@dataclass(frozen=True)
class SyntheticManifestSpec:
    """Inputs to :func:`build_synthetic_manifest`."""

    out_dir: Path
    num_clips: int
    frames_per_clip: int
    height: int
    width: int
    seed: int


def build_synthetic_manifest(spec: SyntheticManifestSpec) -> Path:
    """Write ``num_clips`` ``.mp4`` files + a CSV manifest. Returns manifest path.

    Manifest columns (strict order — Vjepa2Dataset asserts on this):

    - ``video_path``: absolute path to the ``.mp4`` file.
    - ``num_frames``: ``frames_per_clip`` (same for every row in v1).
    - ``height``: ``height`` (same for every row in v1).
    - ``width``: ``width`` (same for every row in v1).
    """
    if spec.num_clips < 1:
        raise ValueError(f"num_clips must be >= 1; got {spec.num_clips}")
    if spec.frames_per_clip < 2:
        raise ValueError(f"frames_per_clip must be >= 2; got {spec.frames_per_clip}")
    if spec.height < 8 or spec.width < 8:
        raise ValueError(f"height/width must be >= 8; got {spec.height}x{spec.width}")

    spec.out_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = spec.out_dir / "videos"
    videos_dir.mkdir(exist_ok=True)
    manifest_path = spec.out_dir / "manifest.csv"

    with manifest_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "num_frames", "height", "width"])
        for i in range(spec.num_clips):
            video_path = write_synthetic_clip(
                videos_dir,
                i,
                num_frames=spec.frames_per_clip,
                height=spec.height,
                width=spec.width,
                seed=spec.seed,
            )
            writer.writerow(
                [str(video_path.resolve()), spec.frames_per_clip, spec.height, spec.width],
            )
    return manifest_path


__all__ = ["SyntheticManifestSpec", "build_synthetic_manifest"]