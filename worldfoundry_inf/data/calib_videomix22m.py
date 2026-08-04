"""Load a fixed 512-sample calibration subset from VideoMix22M.

Used by W8A8Backend to drive msModelSlim's static calibration. Yields
tensors preprocessed by the V-JEPA 2 image processor.

G5 reproducibility: seed=42 is the only seed value (callers should not
override without explicit reason).
"""
from __future__ import annotations

from typing import Iterable

import torch
from datasets import load_dataset
from transformers import AutoImageProcessor

# V-JEPA 2 vitl-1.2b default processor (matches parent spec §4.4)
_DEFAULT_PROCESSOR_MODEL = "facebook/vjepa2-vitl-1.2b"


def load_calibration_dataset(
    n: int = 512,
    seed: int = 42,
    image_processor=None,
) -> Iterable[torch.Tensor]:
    """Yield ``n`` preprocessed (3, 224, 224) tensors from VideoMix22M.

    Args:
        n: Number of calibration samples to yield.
        seed: Random seed for the shuffle (G5 reproducibility).
        image_processor: Optional pre-loaded processor; defaults to
            V-JEPA 2 vitl-1.2b's processor.

    Yields:
        torch.Tensor of shape (3, 224, 224), dtype float32.

    Raises:
        Whatever `datasets.load_dataset` raises on network/auth failures
        (no silent fallback — G4).
    """
    ds = load_dataset("videomix22m", split="train", streaming=True)
    ds = ds.shuffle(seed=seed).select(range(n))
    processor = image_processor or AutoImageProcessor.from_pretrained(
        _DEFAULT_PROCESSOR_MODEL
    )
    for sample in ds:
        frames = sample["video"]  # list of PIL.Image, T frames
        tensor = processor(images=frames, return_tensors="pt")["pixel_values"]
        yield tensor[0]  # (3, 224, 224)
