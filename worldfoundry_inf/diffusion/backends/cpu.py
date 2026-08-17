"""v1 CPU pixel-decoder pipeline (Route A from the W40 frame-gen ADR).

Per TODO M3-T1 / spec
``docs/superpowers/specs/2026-08-06-w40-frame-gen-adr.md`` §5.2: this is
the **lightweight pixel-decoder stub** used for CI / local dev and as
the CPU fallback for the ``mindie_sd`` backend's first-cut diffusion
path. It is **not** a learned decoder — it applies a fixed random
projection from the V-JEPA 2 encoder's latent space to pixel space so
that shape, dtype, and end-to-end plumbing can be exercised without
Atlas hardware.

Key invariants:

* Determinism — given a fixed ``seed``, the projection matrix is
  reproducible across runs (so unit tests can pin exact frame outputs).
* Shape/dtype correctness — every method preserves the documented
  ``(B, T, 3, H, W)`` / ``(B, N, encoder_dim)`` shapes and matches the
  input dtype for the ``encode``/``decode`` round-trip.
* ``available()`` always returns ``True`` so the backend can be built
  on CPU-only hosts (CI / dev boxes). Real NPU backends (M3-T4) gate
  on actual hardware.

Replace this stub with :class:`NPUDiffusionPipeline` once MindIE-SD
compiles a real DiT/VAE for the Atlas NPU (W44+).
"""
from __future__ import annotations

import math

import torch

from worldfoundry_inf.diffusion.pipeline import LatentToFramePipeline


class CPUPixelDecoderPipeline(LatentToFramePipeline):
    """v1 CPU pixel-decoder stub (random linear projection).

    Implements the :class:`LatentToFramePipeline` surface without a
    learned decoder. ``encode_latents`` mirrors the V-JEPA 2 patch grid
    with a deterministic fixed projection (so re-running the same seed
    gives the same latents); ``decode_latents`` does a fixed random
    linear projection back to pixel space; ``rollout`` decodes each
    frame independently and returns the future slice.

    Determinism: the projection matrix is seeded by ``self.seed`` at
    ``__init__`` time, so two pipelines with the same seed produce
    identical outputs.
    """

    name = "cpu_pixel_decoder"

    def __init__(
        self,
        encoder_dim: int = 64,
        img_size: int = 196,
        patch_size: int = 14,
        seed: int = 0,
    ) -> None:
        # Pull defaults from the typical V-JEPA 2 tiny config so unit
        # tests don't need to override every dimension. Real callers
        # pass values from their checkpoint metadata.
        self.encoder_dim = encoder_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.seed = int(seed)
        self._generator = torch.Generator().manual_seed(self.seed)
        # Cache grids so encode/decode agree on (T, Hg, Wg).
        Hg = img_size // patch_size
        Wg = img_size // patch_size
        self._grid_hw = (Hg, Wg)
        # Projection (encoder_dim) -> (patch_size*patch_size*3) so that
        # ``decode_latents`` always produces ``(patch_size**2 * 3)``-D
        # patch embeddings regardless of the actual encoder_dim.
        out_dim = patch_size * patch_size * 3
        proj = torch.randn(encoder_dim, out_dim, generator=self._generator)
        # Keep values small so the random-projection decoder output is
        # bounded — the unit tests only assert shape, not magnitude.
        proj *= 1.0 / math.sqrt(encoder_dim)
        self._proj = proj  # (encoder_dim, patch_size**2 * 3)

    # ------------------------------------------------------------------ #
    # availability
    # ------------------------------------------------------------------ #
    @classmethod
    def available(cls) -> bool:
        """Always available — pure PyTorch, no external deps."""
        return True

    # ------------------------------------------------------------------ #
    # provider_summary
    # ------------------------------------------------------------------ #
    def provider_summary(self) -> dict:
        """M3-T8 dispatch fingerprint for the CPU pixel-decoder stub.

        Surfaces the class identity plus the construction kwargs that
        influence shape / determinism so the orchestrator's
        ``frame_gen_dispatched`` block is enough to reconstruct the
        pipeline config from the summary alone (no separate config
        file needed). Matches the W40 frame-gen ADR sec 5.2 contract:
        ``engine`` is the canonical name; ``kind`` is the provider
        family; ``extras`` carries shape / seed metadata.
        """
        return {
            "engine": "cpu_pixel_decoder",
            "kind": "cpu_pixel_decoder",
            "available": True,
            "extras": {
                "class": type(self).__name__,
                "encoder_dim": self.encoder_dim,
                "img_size": self.img_size,
                "patch_size": self.patch_size,
                "seed": self.seed,
            },
        }

    # ------------------------------------------------------------------ #
    # encode
    # ------------------------------------------------------------------ #
    def encode_latents(self, video: torch.Tensor) -> torch.Tensor:
        """Encode ``(B, T, 3, H, W)`` video to ``(B, N, encoder_dim)``.

        v1 stub: derives ``N = T * Hg * Wg`` from the configured patch
        grid, splits the video into patches, and runs the cached random
        projection (transposed) to produce a deterministic latent
        tensor. We do **not** reshape to ``(B, N, encoder_dim)`` and
        apply a learned encoder — that is the job of
        :class:`VJEPA2Worker.encode`, which the API endpoint dispatches
        to. This stub exists only to keep the diffusion pipeline surface
        consistent on CPU-only hosts.
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
        Hg, Wg = self._grid_hw
        p = self.patch_size
        # Reshape video into patches: (B, T, Hg, Wg, p, p, 3) -> (B, T*Hg*Wg, p*p*3)
        patches = video.reshape(b, t, Hg, p, Wg, p, 3)
        patches = patches.permute(0, 1, 3, 5, 2, 4, 6).reshape(b, t * Hg * Wg, p * p * 3)
        # Linear projection (decoder) transposed: (B, N, p*p*3) @ (p*p*3, encoder_dim)
        proj_t = self._proj.transpose(0, 1)
        latents = patches @ proj_t
        return latents.to(video.dtype)

    # ------------------------------------------------------------------ #
    # decode
    # ------------------------------------------------------------------ #
    def decode_latents(
        self,
        latent: torch.Tensor,
        grid: tuple[int, int, int],
    ) -> torch.Tensor:
        """Decode ``(B, N, encoder_dim)`` to ``(B, T, 3, H, W)``.

        ``grid = (T, Hg, Wg)`` must match ``N = T * Hg * Wg``. The stub
        applies the cached random projection to recover patch pixels,
        then re-assembles them into frames.
        """
        if latent.ndim != 3:
            raise ValueError(
                f"latent must be (B, N, encoder_dim); got shape {tuple(latent.shape)}"
            )
        b, n, dim = latent.shape
        if dim != self.encoder_dim:
            raise ValueError(
                f"latent last dim must equal encoder_dim={self.encoder_dim}; got {dim}"
            )
        T, Hg, Wg = grid
        if n != T * Hg * Wg:
            raise ValueError(
                f"N={n} must equal T*Hg*Wg={T * Hg * Wg}"
            )
        if (Hg, Wg) != self._grid_hw:
            raise ValueError(
                f"grid (Hg,Wg)={(Hg, Wg)} must match pipeline grid "
                f"{self._grid_hw}; rebuild with matching img_size/patch_size."
            )
        p = self.patch_size
        # (B, N, encoder_dim) @ (encoder_dim, p*p*3) -> (B, N, p*p*3)
        pixels = latent @ self._proj
        # Reshape back into video: (B, T, Hg, Wg, p, p, 3) -> (B, T, 3, H, W)
        pixels = pixels.reshape(b, T, Hg, Wg, p, p, 3)
        pixels = pixels.permute(0, 1, 6, 2, 4, 3, 5).reshape(b, T, 3, Hg * p, Wg * p)
        return pixels.to(latent.dtype)

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

        v1 stub ignores ``actions`` and tiles the last decoded past
        frame ``future_frames`` times — enough for shape/dtype round
        trips. Real diffusion (M3-T4) replaces this with a sampler that
        conditions on JEPA latents + actions.
        """
        if past_video.ndim != 5:
            raise ValueError(
                f"past_video must be (B, T_past+T_future, 3, H, W); got shape "
                f"{tuple(past_video.shape)}"
            )
        b, total, _c, h, w = past_video.shape
        if total != past_frames + future_frames:
            raise ValueError(
                f"past_video.shape[1]={total} must equal past_frames+future_frames="
                f"{past_frames + future_frames}"
            )
        Hg, Wg = self._grid_hw
        # Encode past, decode, then repeat the last frame future_frames times.
        past_latents = self.encode_latents(past_video[:, :past_frames])
        decoded = self.decode_latents(past_latents, (past_frames, Hg, Wg))
        # Tile the last past frame forward_frames times: (B, future_frames, C, H, W).
        last = decoded[:, -1:].expand(b, future_frames, _c, h, w)
        return last.to(past_video.dtype)


__all__ = ["CPUPixelDecoderPipeline"]