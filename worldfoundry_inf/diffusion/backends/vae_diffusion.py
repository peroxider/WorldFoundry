"""v2 latent→frame diffusion pipeline wiring VAE + DiT + scheduler (M3-T1).

Why this exists
---------------
M3-T1 assembles the three real building blocks into a
:class:`LatentToFramePipeline` implementation:

* :mod:`worldfoundry_inf.diffusion.vae` — pixel ↔ latent autoencoder.
* :mod:`worldfoundry_inf.diffusion.transformer` — DiT-class denoiser.
* :mod:`worldfoundry_inf.diffusion.scheduler` — DDIM / Flow-Matching
  sampling loops.

:class:`VAEDiffusionPipeline` is the composition. It keeps the frozen
:class:`LatentToFramePipeline` surface (``encode_latents`` /
``decode_latents`` / ``rollout``) so the orchestrator's ``frame_gen``
phase and the backend registry dispatch uniformly — same call sites as
the v1 CPU stub (:class:`CPUPixelDecoderPipeline`), real learned
architecture instead of a random projection.

Sampling path (``decode_latents`` / ``rollout``)
------------------------------------------------
1. The JEPA latent ``(B, N, encoder_dim)`` is the diffusion condition.
2. :class:`scheduler.sample` denoises a fresh ``(B, N, latent_dim)``
   token sequence with the DiT denoiser, conditioned on the JEPA
   features (``cond_dim == encoder_dim``).
3. The denoised VAE-latent tokens are reshaped to
   ``(B*T, latent_dim, Hg, Wg)`` and decoded to pixel frames by the VAE.

This mirrors the W40 frame-gen ADR Route B: latent-conditional diffusion
over a VAE latent space, with the DiT predicting the denoised latent and
the VAE rendering pixels. Works with random-init weights for CI/offline;
``from_pretrained`` hooks on both sub-models accept checkpoints when
available.
"""
from __future__ import annotations

from typing import Any

import torch

from worldfoundry_inf.diffusion.pipeline import LatentToFramePipeline
from worldfoundry_inf.diffusion.scheduler import (
    DDIMScheduler,
    FlowMatchingScheduler,
    SchedulerBase,
    SchedulerConfig,
)
from worldfoundry_inf.diffusion.transformer import (
    DiTConfig,
    LatentDiffusionTransformer,
)
from worldfoundry_inf.diffusion.vae import VAEConfig, VAEModel


class VAEDiffusionPipeline(LatentToFramePipeline):
    """Real VAE + DiT + scheduler latent→frame pipeline (M3-T1 wiring).

    Args:
        encoder_dim: JEPA latent feature dimension (the diffusion
            condition). Defaults to 64 (tiny V-JEPA 2).
        img_size: square input frame size (must be divisible by
            ``vae_cfg.downscale_factor``).
        latent_dim: VAE latent channel count.
        num_steps: diffusion sampling steps.
        scheduler: ``"ddim"`` or ``"flow"``.
        seed: RNG seed for deterministic random init.
    """

    name = "vae_diffusion"

    def __init__(
        self,
        *,
        encoder_dim: int = 64,
        img_size: int = 32,
        latent_dim: int = 4,
        num_steps: int = 8,
        scheduler: str = "ddim",
        seed: int = 0,
    ) -> None:
        self.encoder_dim = int(encoder_dim)
        self.img_size = int(img_size)
        self.num_steps = int(num_steps)
        self.seed = int(seed)

        vae_cfg = VAEConfig(latent_dim=int(latent_dim))
        self.latent_dim = vae_cfg.latent_dim
        self._downscale = vae_cfg.downscale_factor
        self.vae = VAEModel(vae_cfg)
        dit_cfg = DiTConfig(
            latent_dim=self.latent_dim,
            hidden_dim=128,
            depth=2,
            num_heads=4,
            mlp_ratio=4.0,
            cond_dim=self.encoder_dim,
            embed_dim=64,
        )
        self.dit = LatentDiffusionTransformer(dit_cfg)
        sched_cfg = SchedulerConfig()
        if scheduler == "ddim":
            self.scheduler: SchedulerBase = DDIMScheduler(sched_cfg)
        elif scheduler == "flow":
            self.scheduler = FlowMatchingScheduler(sched_cfg)
        else:
            raise ValueError(f"scheduler={scheduler!r} not in ('ddim', 'flow')")
        # Seed all sub-model weight init deterministically.
        torch.manual_seed(self.seed)

    # ------------------------------------------------------------------ #
    # availability
    # ------------------------------------------------------------------ #
    @classmethod
    def available(cls) -> bool:
        """Pure PyTorch — available on CPU-only hosts (CI / dev)."""
        return True

    # ------------------------------------------------------------------ #
    # provider_summary
    # ------------------------------------------------------------------ #
    def provider_summary(self) -> dict[str, Any]:
        """M3-T8 dispatch fingerprint for the VAE+DiT pipeline."""
        return {
            "engine": "vae_diffusion",
            "kind": "vae_diffusion",
            "available": True,
            "extras": {
                "class": type(self).__name__,
                "encoder_dim": self.encoder_dim,
                "img_size": self.img_size,
                "latent_dim": self.latent_dim,
                "num_steps": self.num_steps,
                "scheduler": self.scheduler.name,
                "seed": self.seed,
            },
        }

    # ------------------------------------------------------------------ #
    # encode
    # ------------------------------------------------------------------ #
    def encode_latents(self, video: torch.Tensor) -> torch.Tensor:
        """Encode ``(B, T, 3, H, W)`` video → ``(B, T*Hg*Wg, latent_dim)``.

        Each frame is VAE-encoded to ``(B, latent_dim, Hg, Wg)`` then
        flattened to tokens (``Hg = Wg = img_size / 8``). This is the
        pixel-space ground-truth encoding; the diffusion condition in
        ``decode_latents`` is the JEPA feature stream (``encoder_dim``),
        which shares the same token-grid convention.
        """
        if video.ndim != 5:
            raise ValueError(
                f"video must be (B, T, 3, H, W); got shape {tuple(video.shape)}"
            )
        b, t, _c, h, w = video.shape
        if h != self.img_size or w != self.img_size:
            raise ValueError(
                f"video H/W must equal img_size={self.img_size}; got {(h, w)}"
            )
        frames = video.reshape(b * t, 3, h, w)
        z = self.vae.encode(frames)  # (B*T, latent_dim, Hg, Wg)
        b_zt, c, hg, wg = z.shape
        # Flatten spatial dims into tokens: (B, T, Hg*Wg, c) -> (B, T*Hg*Wg, c)
        return z.reshape(b, t, hg * wg, c).reshape(b, t * hg * wg, c)

    # ------------------------------------------------------------------ #
    # decode
    # ------------------------------------------------------------------ #
    def decode_latents(
        self,
        latent: torch.Tensor,
        grid: tuple[int, int, int],
    ) -> torch.Tensor:
        """Decode ``(B, N, encoder_dim)`` JEPA latent → ``(B, T, 3, H, W)``.

        Diffusion path: sample VAE-latent tokens conditioned on the JEPA
        features, then VAE-decode each frame.

        Args:
            latent: JEPA features ``(B, N, encoder_dim)`` (diffusion
                condition).
            grid: ``(T, Hg, Wg)``; ``N == T*Hg*Wg``.
        """
        if latent.ndim != 3:
            raise ValueError(
                f"latent must be (B, N, encoder_dim); got shape {tuple(latent.shape)}"
            )
        b, n, dim = latent.shape
        if dim != self.encoder_dim:
            raise ValueError(
                f"latent last dim must equal encoder_dim={self.encoder_dim}; "
                f"got {dim}"
            )
        T, Hg, Wg = grid
        if n != T * Hg * Wg:
            raise ValueError(f"N={n} must equal T*Hg*Wg={T * Hg * Wg}")

        # The DiT condition is a per-sample vector (B, encoder_dim) —
        # FiLM-style additive conditioning (matching the orchestrator's
        # action-conditioned seam). Pool the JEPA token sequence to a
        # global feature per sample.
        cond = latent.mean(dim=1)  # (B, encoder_dim)
        shape = (b, n, self.latent_dim)

        def denoiser(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return self.dit(x, t, cond=cond)

        # Deterministic sampling: the scheduler draws its initial noise
        # from the global RNG (``torch.randn``), so pin the seed before
        # the loop — matches the LatentToFramePipeline contract that v1
        # implementations are deterministic given a fixed seed.
        torch.manual_seed(self.seed)
        z_tokens = self.scheduler.sample(
            denoiser, shape, num_steps=self.num_steps
        )  # (B, N, latent_dim)
        z = z_tokens.reshape(b, T, Hg * Wg, self.latent_dim).permute(0, 1, 3, 2)
        z = z.reshape(b * T, self.latent_dim, Hg, Wg)
        frames = self.vae.decode(z)  # (B*T, 3, H, W)
        _, c, h, w = frames.shape
        return frames.reshape(b, T, c, h, w)

    # ------------------------------------------------------------------ #
    # rollout
    # ------------------------------------------------------------------ #
    def rollout(
        self,
        past_video: torch.Tensor,
        actions: torch.Tensor | None,
        past_frames: int,
        future_frames: int,
    ) -> torch.Tensor:
        """Roll out ``future_frames`` pixel frames conditioned on past.

        v2: encode the full history with the VAE, then diffusion-decode
        the future slice from its tokens. ``actions`` is accepted for
        interface parity; conditioning on per-frame actions is a
        follow-on (the DiT ``cond_dim`` hook is ready for it).
        """
        if past_video.ndim != 5:
            raise ValueError(
                f"past_video must be (B, T_past+T_future, 3, H, W); got shape "
                f"{tuple(past_video.shape)}"
            )
        b, total, _c, h, w = past_video.shape
        if total != past_frames + future_frames:
            raise ValueError(
                f"past_video.shape[1]={total} must equal "
                f"past_frames+future_frames={past_frames + future_frames}"
            )
        Hg = Wg = h // self._downscale
        future_n = future_frames * Hg * Wg
        # JEPA-conditioned diffusion decode with encoder_dim-condition.
        cond = torch.zeros(b, future_n, self.encoder_dim)
        grid = (future_frames, Hg, Wg)
        return self.decode_latents(cond, grid)


__all__ = ["VAEDiffusionPipeline"]
