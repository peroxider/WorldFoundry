"""HF VJEPA2Model + LoRA shim for the CPU render-train-inference orchestrator (W2.18).

Mirrors the surface of :class:`worldfoundry_train.model.tiny_jepa.TinyVJEPA2Encoder`
on top of ``transformers.VJEPA2Model`` so the orchestrator's
``_build_stack`` chokepoint can swap from a 24k-param tiny stack to a
real HF backbone without any downstream edits. The action-conditioning
predictor stays Tiny-only; the HF smoke uses :class:`JEPAPredictor` and
documents that the ``action`` kwarg is accepted but ignored, so the
orchestrator's uniform ``pred(h, mask, action)`` call site works for
both encoder kinds.

The factory :meth:`HfVJEPA2EncoderShim.from_random_init` builds a tiny,
randomly-initialised HF config (default ``hidden_size=64``,
``num_hidden_layers=2``, etc.) and wraps it in LoRA via
:func:`worldfoundry_train.model.lora_wrapper.build_lora_model`. No
HF Hub call — the smoke proves the seam without downloading 1.2B
weights. Real 1.2B end-to-end is Atlas-side and out of scope here.

CPU feasibility (live-probed before this module shipped):

    HfVJEPA2EncoderShim.from_random_init()  ->  ~80k params (LoRA subset ~1k)
    forward((1, 4, 3, 32, 32))  ->  last_hidden_state (1, 16, 64)
                                    ~0.12 s on a single CPU thread
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

from torch import Tensor, nn

from worldfoundry_train.model.lora_wrapper import build_lora_model
from worldfoundry_train.model.predictor import JEPAPredictor
from worldfoundry_train.model.tiny_jepa import (
    _TinyOutput,
    TinyPixelDecoder,
)


@dataclass(frozen=True)
class HfEncoderShimConfig:
    """Geometry surface the orchestrator reads from ``encoder.config``.

    Mirrors the four fields :class:`TinyEncoderConfig` exposes to
    downstream callers, mapped onto HF V-JEPA 2's vocabulary. The
    :meth:`from_vjepa2` factory builds this from a real
    ``transformers.VJEPA2Config``; the shim never lets HF field names
    leak past the boundary.

    LoRA fields (``lora_r``, ``lora_alpha``, ``lora_dropout``) live on
    the config so the random-init factory can be reconstructed
    byte-shape-compatible from the saved checkpoint signature.
    """

    image_size: int
    patch_size: int
    in_channels: int
    embed_dim: int
    num_heads: int
    depth: int
    tubelet_size: int
    frames_per_clip: int
    lora_r: int = 4
    lora_alpha: int = 8
    lora_dropout: float = 0.0

    @property
    def grid_h(self) -> int:
        return self.image_size // self.patch_size

    @property
    def grid_w(self) -> int:
        return self.image_size // self.patch_size

    def signature(self) -> dict:
        """Architecture-identifying fields for checkpoint round-trip.

        Mirrors the schema :class:`TinyEncoderConfig.signature` writes so
        :func:`save_tiny_checkpoint` can store ``encoder_config``
        uniformly across encoder kinds. The seven HF-specific fields
        (``hidden_size``, ``num_attention_heads``, ``tubelet_size``,
        ``frames_per_clip``, ``lora_r``, ``lora_alpha``,
        ``lora_dropout``) round-trip the random-init factory's full
        argument set so :func:`load_tiny_checkpoint` can rebuild the
        LoRA-wrapped ``VJEPA2Model`` byte-shape-compatible with the
        one that wrote the checkpoint. Tiny-path consumers ignore the
        extra keys.
        """
        return {
            "image_size": int(self.image_size),
            "patch_size": int(self.patch_size),
            "in_channels": int(self.in_channels),
            "embed_dim": int(self.embed_dim),
            "depth": int(self.depth),
            "num_heads": int(self.num_heads),
            "mlp_ratio": 0.0,
            "encoder_kind": "hf_vjepa2",
            "hidden_size": int(self.embed_dim),
            "num_attention_heads": int(self.num_heads),
            "tubelet_size": int(self.tubelet_size),
            "frames_per_clip": int(self.frames_per_clip),
            "lora_r": int(self.lora_r),
            "lora_alpha": int(self.lora_alpha),
            "lora_dropout": float(self.lora_dropout),
        }

    @classmethod
    def from_vjepa2(cls, hf_cfg) -> "HfEncoderShimConfig":
        """Build from a ``transformers.VJEPA2Config``.

        ``VJEPA2Config`` exposes its spatial side length as both
        ``image_size`` (property, derived from ``crop_size``) and
        ``crop_size`` (the constructor kwarg). We prefer the
        attribute to be agnostic to which the caller set.
        """
        image_size = int(
            getattr(hf_cfg, "image_size", None)
            or getattr(hf_cfg, "crop_size", 0)
        )
        return cls(
            image_size=image_size,
            patch_size=int(hf_cfg.patch_size),
            in_channels=int(hf_cfg.in_chans),
            embed_dim=int(hf_cfg.hidden_size),
            num_heads=int(hf_cfg.num_attention_heads),
            depth=int(hf_cfg.num_hidden_layers),
            tubelet_size=int(hf_cfg.tubelet_size),
            frames_per_clip=int(hf_cfg.frames_per_clip),
            # LoRA fields have no analogue on the raw HF config; the
            # shim's LoRA wrap is applied later by ``from_random_init``.
            # Defaults match the random-init factory's defaults.
            lora_r=4,
            lora_alpha=8,
            lora_dropout=0.0,
        )


class HfVJEPA2EncoderShim(nn.Module):
    """``VJEPA2Model`` adapter exposing :class:`TinyVJEPA2Encoder`'s surface.

    Args:
        base_model: a PEFT-LoRA-wrapped ``VJEPA2Model`` (the
            ``peft.peft_model.PeftModel`` returned by
            :func:`build_lora_model`).
        shim_config: geometry describing the wrapped base model.
    """

    def __init__(
        self,
        base_model: nn.Module,
        shim_config: HfEncoderShimConfig,
    ) -> None:
        super().__init__()
        if base_model is None:
            raise ValueError("base_model is required (a LoRA-wrapped VJEPA2Model)")
        if shim_config is None:
            raise ValueError("shim_config is required")
        if shim_config.image_size < 8 or shim_config.patch_size < 1:
            raise ValueError(
                f"image_size={shim_config.image_size} and patch_size="
                f"{shim_config.patch_size} must be >= 8 and >= 1 respectively"
            )
        if shim_config.image_size % shim_config.patch_size != 0:
            raise ValueError(
                f"image_size={shim_config.image_size} must be divisible "
                f"by patch_size={shim_config.patch_size}"
            )
        if shim_config.embed_dim < 1:
            raise ValueError(f"embed_dim must be >= 1; got {shim_config.embed_dim}")
        if shim_config.num_heads < 1:
            raise ValueError(f"num_heads must be >= 1; got {shim_config.num_heads}")
        if shim_config.tubelet_size < 1:
            raise ValueError(
                f"tubelet_size must be >= 1; got {shim_config.tubelet_size}"
            )
        if shim_config.frames_per_clip < 1:
            raise ValueError(
                f"frames_per_clip must be >= 1; got {shim_config.frames_per_clip}"
            )

        self.base = base_model
        self.config = shim_config

    def forward(
        self,
        pixel_values_videos: Tensor,
        *,
        skip_predictor: bool = True,
        **_: object,
    ) -> _TinyOutput:
        """Encode a video batch.

        Args:
            pixel_values_videos: ``(B, T, C, H, W)`` float in ``[0, 1]``
                or any normalised range the wrapped HF model handles.
            skip_predictor: forwarded to ``VJEPA2Model.forward``. The HF
                model's built-in (frozen) predictor submodule is dead
                compute for the W22 objective (see
                ``jepa_objective.py:25-31``), so we always skip it.

        Returns:
            :class:`_TinyOutput` carrying ``last_hidden_state`` of
            shape ``(B, T*Hg*Wg, embed_dim)`` -- byte-shape compatible
            with :class:`TinyVJEPA2Encoder`'s output.
        """
        if pixel_values_videos.ndim != 5:
            raise ValueError(
                f"pixel_values_videos must be (B, T, C, H, W); got shape "
                f"{tuple(pixel_values_videos.shape)}"
            )
        out = self.base(
            pixel_values_videos=pixel_values_videos,
            skip_predictor=skip_predictor,
        )
        return _TinyOutput(last_hidden_state=out.last_hidden_state)

    def freeze_encoder(self) -> int:
        """Set ``requires_grad=False`` on every base parameter.

        Used by the orchestrator when the encoder should stay frozen
        (matches ``VisionPolicy.from_image_obs(freeze_encoder=True)``).
        Returns the number of params frozen (for logging).
        """
        n = 0
        for p in self.base.parameters():
            if p.requires_grad:
                p.requires_grad_(False)
                n += 1
        return n

    @classmethod
    def from_random_init(
        cls,
        *,
        image_size: int = 32,
        patch_size: int = 8,
        in_channels: int = 3,
        hidden_size: int = 64,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 2,
        tubelet_size: int = 2,
        frames_per_clip: int = 4,
        lora_r: int = 4,
        lora_alpha: int = 8,
        lora_dropout: float = 0.0,
    ) -> "HfVJEPA2EncoderShim":
        """Build a random-init HF VJEPA2Model + LoRA wrapped in the shim.

        Defaults are tuned for a CPU-tractable forward pass
        (~0.12 s on a single CPU thread per (1, T, 3, 32, 32) input;
        ~80k total params, ~1k trainable via LoRA).

        No HF Hub call. Real 1.2B end-to-end is Atlas-side and out of
        scope for this factory.
        """
        # Lazy import so tests that don't exercise the factory don't
        # require ``transformers`` to be importable.
        from transformers import VJEPA2Config, VJEPA2Model

        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size={image_size} must be divisible by "
                f"patch_size={patch_size}"
            )
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by "
                f"num_attention_heads={num_attention_heads}"
            )
        if tubelet_size < 1:
            raise ValueError(f"tubelet_size must be >= 1; got {tubelet_size}")
        if frames_per_clip < 1:
            raise ValueError(
                f"frames_per_clip must be >= 1; got {frames_per_clip}"
            )
        if lora_r < 1:
            raise ValueError(f"lora_r must be >= 1; got {lora_r}")
        if lora_alpha < 1:
            raise ValueError(f"lora_alpha must be >= 1; got {lora_alpha}")

        cfg = VJEPA2Config(
            hidden_size=int(hidden_size),
            num_hidden_layers=int(num_hidden_layers),
            num_attention_heads=int(num_attention_heads),
            crop_size=int(image_size),
            patch_size=int(patch_size),
            tubelet_size=int(tubelet_size),
            frames_per_clip=int(frames_per_clip),
            in_chans=int(in_channels),
            drop_path_rate=0.0,
        )
        base = VJEPA2Model(cfg)
        lora_result = build_lora_model(
            base,
            r=int(lora_r),
            alpha=int(lora_alpha),
            dropout=float(lora_dropout),
            bias="none",
        )
        # Build the shim config directly (instead of via ``from_vjepa2``)
        # so the LoRA fields round-trip into ``signature()``.
        shim_config = HfEncoderShimConfig(
            image_size=int(image_size),
            patch_size=int(patch_size),
            in_channels=int(in_channels),
            embed_dim=int(hidden_size),
            num_heads=int(num_attention_heads),
            depth=int(num_hidden_layers),
            tubelet_size=int(tubelet_size),
            frames_per_clip=int(frames_per_clip),
            lora_r=int(lora_r),
            lora_alpha=int(lora_alpha),
            lora_dropout=float(lora_dropout),
        )
        shim = cls(
            base_model=lora_result.peft_model,
            shim_config=shim_config,
        )
        # Stash lora stats for callers that want to surface them in the
        # orchestrator's summary (e.g. ``hf_lora_trainable_params``).
        shim._lora_trainable_params = int(lora_result.trainable_params)
        shim._lora_total_params = int(lora_result.total_params)
        return shim

    def lora_stats(self) -> dict[str, int]:
        """Return ``{"trainable": int, "total": int}`` for the LoRA wrap.

        Zero-cost accessor for the orchestrator's summary block. Falls
        back to ``0`` if the shim wasn't built via :meth:`from_random_init`
        (e.g. constructed directly from a pre-built LoRA model).
        """
        return {
            "trainable": int(getattr(self, "_lora_trainable_params", 0)),
            "total": int(getattr(self, "_lora_total_params", 0)),
        }

    def deep_copy_base(self) -> "HfVJEPA2EncoderShim":
        """Return a new shim sharing ``shim_config`` but with a deepcopy
        of ``self.base``. Used by the orchestrator to build the EMA
        target encoder without sharing parameters with the online one.
        """
        return type(self)(
            base_model=copy.deepcopy(self.base),
            shim_config=self.config,
        )


class HfJEPAPredictorShim(nn.Module):
    """Action-agnostic :class:`JEPAPredictor` adapter for the HF smoke.

    Wraps the W22 :class:`JEPAPredictor` (full V-JEPA 2 architecture)
    with the orchestrator's uniform ``(context_tokens, mask, action)``
    call signature. The ``action`` kwarg is accepted and ignored -- the
    HF smoke drops action conditioning because the story is already
    proven by :class:`TinyActionConditionedPredictor` (W2.17). A future
    W2.19+ can compose FiLM on top of HF if needed.

    Args:
        num_patches: ``N = T * Hg * Wg`` of the training grid.
        grid_size: ``(T, Hg, Wg)`` matching ``num_patches``.
        encoder_dim: HF ``hidden_size`` (e.g. 64 for the smoke).
        predictor_dim: predictor bottleneck width (64 for the smoke).
        depth: predictor transformer layers (2 for the smoke).
        num_heads: predictor attention heads (2 for the smoke).
    """

    def __init__(
        self,
        *,
        num_patches: int,
        grid_size: tuple[int, int, int],
        encoder_dim: int,
        predictor_dim: int = 64,
        depth: int = 2,
        num_heads: int = 2,
    ) -> None:
        super().__init__()
        if num_patches < 1:
            raise ValueError(f"num_patches must be >= 1; got {num_patches}")
        if encoder_dim < 1:
            raise ValueError(f"encoder_dim must be >= 1; got {encoder_dim}")
        if predictor_dim % num_heads != 0:
            raise ValueError(
                f"predictor_dim={predictor_dim} must be divisible by "
                f"num_heads={num_heads}"
            )

        self.predictor = JEPAPredictor(
            num_patches=int(num_patches),
            grid_size=tuple(int(s) for s in grid_size),
            encoder_dim=int(encoder_dim),
            predictor_dim=int(predictor_dim),
            depth=int(depth),
            num_heads=int(num_heads),
        )

    def forward(
        self,
        context_tokens: Tensor,
        bool_masked_pos: Tensor,
        action: Tensor | None = None,
    ) -> Tensor:
        """Predict target features at masked positions.

        Args:
            context_tokens: ``(B, N, encoder_dim)`` -- HF encoder output.
            bool_masked_pos: ``(B, N)`` bool mask -- positions to predict.
            action: accepted for orchestrator uniformity; **ignored** in
                the HF smoke. See class docstring.

        Returns:
            ``(M_total, encoder_dim)`` -- predictions at masked
            positions, concatenated across the batch dim (matches HF +
            :class:`JEPAPredictor` semantics).
        """
        del action  # intentionally ignored -- see class docstring
        return self.predictor(context_tokens, bool_masked_pos)


class HfPixelDecoderShim(nn.Module):
    """Pixel decoder adapter wrapping :class:`TinyPixelDecoder`.

    Re-uses the existing decoder -- its geometry depends only on
    ``embed_dim``, ``patch_size``, and ``in_channels``, all of which
    live on :class:`HfEncoderShimConfig`. The shim exists so the
    orchestrator's ``_build_stack`` can stay uniform across encoder
    kinds.
    """

    def __init__(self, shim_config: HfEncoderShimConfig) -> None:
        super().__init__()
        if shim_config is None:
            raise ValueError("shim_config is required")
        # We construct a TinyEncoderConfig with only the fields
        # TinyPixelDecoder reads. TinyEncoderConfig is a dataclass with
        # frozen defaults; the decoder uses embed_dim + patch_size +
        # in_channels + the derived grid_h/grid_w. Defaults for fields
        # the decoder ignores (depth, num_heads, mlp_ratio) stay at
        # their TinyEncoderConfig defaults.
        from worldfoundry_train.model.tiny_jepa import TinyEncoderConfig

        encoder_config = TinyEncoderConfig(
            image_size=int(shim_config.image_size),
            patch_size=int(shim_config.patch_size),
            in_channels=int(shim_config.in_channels),
            embed_dim=int(shim_config.embed_dim),
            depth=int(shim_config.depth),
            num_heads=int(shim_config.num_heads),
        )
        self.decoder = TinyPixelDecoder(encoder_config)
        self.config = shim_config
        # Duck-typed alias for ``TinyPixelDecoder`` consumers (e.g.
        # ``LatentRolloutPlanner`` checks ``decoder.encoder_config.
        # image_size``). The HF path's geometry is identical to the
        # tiny one because we forward ``embed_dim`` / ``patch_size`` /
        # ``in_channels`` from the shim config; aliasing keeps both
        # backends on the same planner code path.
        self.encoder_config = encoder_config

    def forward(self, latents: Tensor, num_frames: int) -> Tensor:
        """Decode ``(B, T*Hg*Wg, embed_dim)`` to ``(B, T, C, H, W)``."""
        return self.decoder(latents, num_frames=num_frames)


__all__ = [
    "HfEncoderShimConfig",
    "HfVJEPA2EncoderShim",
    "HfJEPAPredictorShim",
    "HfPixelDecoderShim",
]