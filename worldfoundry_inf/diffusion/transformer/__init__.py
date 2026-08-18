"""DiT-class transformer backbone for the latent→frame pipeline (M3-T1)."""

from worldfoundry_inf.diffusion.transformer.model import (
    DiTConfig,
    LatentDiffusionTransformer,
)

__all__ = ["DiTConfig", "LatentDiffusionTransformer"]
