"""VAE building blocks for the latent→frame diffusion pipeline (M3-T1)."""

from worldfoundry_inf.diffusion.vae.model import (
    VAEConfig,
    VAEDecoder,
    VAEEncoder,
    VAEModel,
)

__all__ = ["VAEConfig", "VAEEncoder", "VAEDecoder", "VAEModel"]
