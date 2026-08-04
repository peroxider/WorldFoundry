"""Generate the FIXED data subset manifest for the W22.7 A/B experiment.

Design §9.2 requires both arms to see the exact same sample sequence. This
script samples N rows from the full manifest with a fixed seed and writes
the result to disk ONCE; both arm configs point ``data.manifest`` at the
generated file. The dataset consumes manifests sequentially (no shuffle) and
``mask_fn`` is seeded per sample index (``data/dataset.py:76``), so sample +
mask sequences are bit-identical across arms.

Usage:
    python scripts/w22_7_make_subset.py \
        --source /data/videomix22m/manifest.csv \
        --n 10000 \
        --out bench_results/w22_7/subset_manifest.csv
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

FIELDS = ["video_path", "num_frames", "height", "width"]


def make_subset(source: Path, n: int, out: Path, seed: int = 20260803) -> Path:
    if not source.is_file():
        raise FileNotFoundError(f"source manifest not found: {source}")
    with source.open() as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != FIELDS:
            raise ValueError(
                f"manifest schema must be {FIELDS}; got {reader.fieldnames}"
            )
        rows = list(reader)
    if n > len(rows):
        raise ValueError(f"--n {n} > source rows {len(rows)}")
    rng = random.Random(seed)
    subset = rng.sample(rows, n)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(subset)
    print(f"[W22.7] subset manifest: {out} ({n} rows, seed={seed})")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--out", type=Path,
                    default=Path("bench_results/w22_7/subset_manifest.csv"))
    ap.add_argument("--seed", type=int, default=20260803)
    args = ap.parse_args()
    make_subset(args.source, args.n, args.out, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
