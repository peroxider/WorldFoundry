"""VisionPolicy — pluggable-encoder actor-critic for embodied RL (W41 Phase-5 v1).

A PPO-compatible actor-critic that takes **any encoder** (state MLP, CNN,
HF V-JEPA 2 backbone) and exposes the same surface as
:class:`worldfoundry_rl.ppo.PPOActor` (``forward`` / ``act`` /
``evaluate_actions`` / ``distribution``) so it slots into the existing
``PPO`` trainer unchanged.

The encoder seam is the **NPU port story** (Phase-5 v1 §3 of the plan):

* **CPU**: any ``nn.Module`` works; pass a tiny state-MLP via
  :meth:`from_state_dim` for state-only obs, or any vision backbone
  (HF VJEPA2Model, a CNN) for pixel obs.
* **NPU**: call ``policy.to("npu")`` after construction — the migration
  point in :mod:`worldfoundry_train.device` is the same one the JEPA
  trainer uses (W20). No new NPU code path.

The diagonal-Gaussian state-independent ``log_std`` parameter is
identical to :class:`PPOActor` so policy-gradient math is unchanged.

Per W41 Phase-5 v1 plan §Deliverable 2.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class VisionPolicy(nn.Module):
    """Pluggable-encoder actor-critic for embodied RL.

    Args:
        encoder: ``nn.Module`` mapping the observation batch ``(B, ...)``
            to a feature batch ``(B, feature_dim)``. **Mandatory**: pass
            a state MLP, a CNN, or a vision-backbone wrapper. For state
            obs the convenience factory :meth:`from_state_dim` builds
            the encoder for you.
        feature_dim: output width of ``encoder``. The policy/value heads
            consume this directly.
        action_dim: continuous action dimensionality.
        log_std_init: initial value of the state-independent log-σ.
        log_std_min: lower clamp on log-σ (matches :class:`PPOActor`).
        log_std_max: upper clamp on log-σ.
    """

    def __init__(
        self,
        encoder: nn.Module,
        feature_dim: int,
        action_dim: int,
        *,
        log_std_init: float = 0.0,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()
        if encoder is None:
            raise ValueError(
                "encoder is required; use VisionPolicy.from_state_dim(obs_dim, "
                "action_dim) to build the default state-MLP encoder."
            )
        if feature_dim < 1:
            raise ValueError(f"feature_dim must be >= 1, got {feature_dim}")
        if action_dim < 1:
            raise ValueError(f"action_dim must be >= 1, got {action_dim}")
        if log_std_min >= log_std_max:
            raise ValueError(
                f"log_std_min ({log_std_min}) must be < log_std_max ({log_std_max})"
            )

        self.encoder = encoder
        self.feature_dim = int(feature_dim)
        self.action_dim = int(action_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        self.policy_head = nn.Linear(self.feature_dim, self.action_dim)
        self.value_head = nn.Linear(self.feature_dim, 1)
        self.log_std = nn.Parameter(
            torch.full((self.action_dim,), float(log_std_init))
        )

        # Standard PPO init: small-gain policy head keeps the initial
        # policy close to deterministic; value head gets unit gain.
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    # ------------------------------------------------------------------
    # factories
    # ------------------------------------------------------------------

    @classmethod
    def from_state_dim(
        cls,
        obs_dim: int,
        action_dim: int,
        *,
        hidden_dim: int = 64,
        **kwargs,
    ) -> "VisionPolicy":
        """Build a VisionPolicy with a built-in 2-layer tanh state MLP.

        The encoder is ``Linear(obs_dim, hidden_dim) -> Tanh ->
        Linear(hidden_dim, hidden_dim) -> Tanh``, so ``feature_dim ==
        hidden_dim``. This is the W41 Phase-5 v1 default — state-only
        CPU smoke does not need a vision backbone.
        """
        if obs_dim < 1:
            raise ValueError(f"obs_dim must be >= 1, got {obs_dim}")
        encoder = nn.Sequential(
            nn.Linear(int(obs_dim), int(hidden_dim)),
            nn.Tanh(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.Tanh(),
        )
        return cls(
            encoder=encoder,
            feature_dim=int(hidden_dim),
            action_dim=int(action_dim),
            **kwargs,
        )

    @classmethod
    def from_image_obs(
        cls,
        image_size: int,
        action_dim: int,
        *,
        patch_size: int = 8,
        embed_dim: int = 96,
        depth: int = 2,
        num_heads: int = 3,
        in_channels: int = 3,
        freeze_encoder: bool = True,
        **kwargs,
    ) -> "VisionPolicy":
        """Build a VisionPolicy backed by :class:`TinyVJEPA2Encoder`.

        Used by the W2.17 multi-step latent rollout path: the actor
        receives ``(B, T, 3, H, W)`` pixel batches, the encoder maps
        them to ``(B, T*Hg*Wg, embed_dim)``, and we pool to
        ``(B, embed_dim)`` by mean-pooling the token sequence so the
        policy head sees a single feature vector per batch element.

        Args:
            image_size: spatial side length (H == W).
            patch_size: encoder patch size; must divide ``image_size``.
            embed_dim: encoder width; equals ``feature_dim``.
            depth/num_heads: encoder depth / attention heads.
            in_channels: number of input channels (default 3 RGB).
            freeze_encoder: when ``True`` (default), the encoder
                parameters have ``requires_grad=False`` so the policy
                update does not perturb the world model. Set to
                ``False`` for end-to-end fine-tuning runs.
            **kwargs: forwarded to :class:`VisionPolicy` (log_std
                bounds, etc.).
        """
        # Local import keeps the state-only factory import-light.
        from worldfoundry_train.model.tiny_jepa import (
            TinyEncoderConfig,
            TinyVJEPA2Encoder,
        )

        if image_size < 8 or patch_size < 1:
            raise ValueError(
                f"image_size={image_size} and patch_size={patch_size} "
                f"must be >= 8 and >= 1 respectively"
            )
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size={image_size} must be divisible by "
                f"patch_size={patch_size}"
            )

        config = TinyEncoderConfig(
            image_size=int(image_size),
            patch_size=int(patch_size),
            in_channels=int(in_channels),
            embed_dim=int(embed_dim),
            depth=int(depth),
            num_heads=int(num_heads),
        )
        encoder = TinyVJEPA2Encoder(config)
        if freeze_encoder:
            for p in encoder.parameters():
                p.requires_grad_(False)

        # Wrap encoder so its ``(B, T*Hg*Wg, embed_dim)`` output collapses
        # to ``(B, embed_dim)`` via mean pooling over the token sequence.
        class _Pool(nn.Module):
            def __init__(self, backbone: nn.Module) -> None:
                super().__init__()
                self.backbone = backbone

            def forward(self, obs: Tensor) -> Tensor:
                # ``obs`` may be (B, T, 3, H, W) or (B, 3, H, W); we
                # always treat it as (B, T, 3, H, W) by adding a leading
                # time dim if needed.
                if obs.dim() == 4:
                    obs = obs.unsqueeze(1)
                out = self.backbone(obs).last_hidden_state
                # (B, T*Hg*Wg, embed_dim) -> (B, embed_dim)
                return out.mean(dim=1)

        pooled = _Pool(encoder)
        return cls(
            encoder=pooled,
            feature_dim=int(embed_dim),
            action_dim=int(action_dim),
            **kwargs,
        )

    @classmethod
    def from_hf_image_obs(
        cls,
        image_size: int,
        action_dim: int,
        *,
        patch_size: int = 8,
        embed_dim: int = 64,
        depth: int = 2,
        num_heads: int = 2,
        in_channels: int = 3,
        tubelet_size: int = 1,
        frames_per_clip: int = 4,
        lora_r: int = 4,
        lora_alpha: int = 8,
        lora_dropout: float = 0.0,
        freeze_encoder: bool = True,
        # W2.22+: real-weights routing. When ``pretrained_repo_id`` is
        # set the factory calls ``HfVJEPA2EncoderShim.from_pretrained``
        # (loads HF VJEPA2Model from Hub / local path) instead of
        # ``from_random_init``. ``None`` (default) preserves the
        # W2.18/19 random-init behaviour.
        pretrained_repo_id: str | None = None,
        pretrained_revision: str | None = None,
        **kwargs,
    ) -> "VisionPolicy":
        """Build a VisionPolicy backed by HF VJEPA2Model + LoRA via shim.

        Mirror of :meth:`from_image_obs` for the W2.18+ HF encoder
        path: the actor ingests ``(B, T, 3, H, W)`` pixel batches,
        the LoRA-wrapped :class:`HfVJEPA2EncoderShim` maps them to
        ``(B, T*Hg*Wg, embed_dim)``, and we mean-pool to
        ``(B, embed_dim)`` for the policy head. Defaults are tuned
        for the CPU smoke (matches the orchestrator's
        ``_build_stack`` HF branch).

        Args:
            image_size: spatial side length (H == W).
            patch_size: encoder patch size; must divide ``image_size``.
            embed_dim: HF ``hidden_size``; equals ``feature_dim``.
            depth/num_heads: HF ``num_hidden_layers`` /
                ``num_attention_heads``.
            in_channels: number of input channels (default 3 RGB).
            tubelet_size: HF temporal patch size. ``1`` matches the
                tiny encoder's token count (``T*Hg*Wg``).
            frames_per_clip: HF ``frames_per_clip``; the encoder
                expects ``(B, T, 3, H, W)`` with ``T`` matching this.
            lora_r/lora_alpha/lora_dropout: LoRA wrap parameters
                forwarded to
                :meth:`HfVJEPA2EncoderShim.from_random_init`.
            freeze_encoder: when ``True`` (default), the encoder
                parameters have ``requires_grad=False`` so the policy
                update does not perturb the world model.
            **kwargs: forwarded to :class:`VisionPolicy` (log_std
                bounds, etc.).
        """
        from worldfoundry_train.model.hf_vjepa_shim import HfVJEPA2EncoderShim

        if image_size < 8 or patch_size < 1:
            raise ValueError(
                f"image_size={image_size} and patch_size={patch_size} "
                f"must be >= 8 and >= 1 respectively"
            )
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size={image_size} must be divisible by "
                f"patch_size={patch_size}"
            )

        encoder = HfVJEPA2EncoderShim.from_random_init(
            image_size=int(image_size),
            patch_size=int(patch_size),
            in_channels=int(in_channels),
            hidden_size=int(embed_dim),
            num_hidden_layers=int(depth),
            num_attention_heads=int(num_heads),
            tubelet_size=int(tubelet_size),
            frames_per_clip=int(frames_per_clip),
            lora_r=int(lora_r),
            lora_alpha=int(lora_alpha),
            lora_dropout=float(lora_dropout),
        ) if not pretrained_repo_id else HfVJEPA2EncoderShim.from_pretrained(
            pretrained_repo_id=str(pretrained_repo_id),
            revision=pretrained_revision,
            image_size=int(image_size),
            patch_size=int(patch_size),
            in_channels=int(in_channels),
            hidden_size=int(embed_dim),
            num_hidden_layers=int(depth),
            num_attention_heads=int(num_heads),
            tubelet_size=int(tubelet_size),
            frames_per_clip=int(frames_per_clip),
            lora_r=int(lora_r),
            lora_alpha=int(lora_alpha),
            lora_dropout=float(lora_dropout),
        )
        if freeze_encoder:
            encoder.freeze_encoder()

        # Wrap encoder so its ``(B, T*Hg*Wg, embed_dim)`` output collapses
        # to ``(B, embed_dim)`` via mean pooling over the token sequence.
        # Identical pooling pattern to ``from_image_obs``.
        class _Pool(nn.Module):
            def __init__(self, backbone: nn.Module) -> None:
                super().__init__()
                self.backbone = backbone

            def forward(self, obs: Tensor) -> Tensor:
                # ``obs`` may be (B, T, 3, H, W) or (B, 3, H, W); we
                # always treat it as (B, T, 3, H, W) by adding a leading
                # time dim if needed.
                if obs.dim() == 4:
                    obs = obs.unsqueeze(1)
                out = self.backbone(obs).last_hidden_state
                # (B, T*Hg*Wg, embed_dim) -> (B, embed_dim)
                return out.mean(dim=1)

        pooled = _Pool(encoder)
        return cls(
            encoder=pooled,
            feature_dim=int(embed_dim),
            action_dim=int(action_dim),
            **kwargs,
        )

    # ------------------------------------------------------------------
    # forward / distribution / sample / evaluate
    # ------------------------------------------------------------------

    def _flatten_single(self, obs) -> Tensor:
        """Flatten a single env observation (dict / tensor / array) to a
        1-D tensor. Dict keys are sorted for stability across steps.
        Mirrors :func:`worldfoundry_rl.ppo.flatten_obs`.
        """
        if isinstance(obs, Tensor):
            return obs.detach().to(torch.float32).reshape(-1)
        if isinstance(obs, dict):
            parts = []
            for k in sorted(obs.keys()):
                v = obs[k]
                if isinstance(v, Tensor):
                    parts.append(v.detach().to(torch.float32).reshape(-1))
                else:
                    parts.append(torch.as_tensor(v, dtype=torch.float32).reshape(-1))
            if not parts:
                return torch.zeros(0, dtype=torch.float32)
            return torch.cat(parts)
        return torch.as_tensor(obs, dtype=torch.float32).reshape(-1)

    def _features(self, obs) -> Tensor:
        """Run the encoder. Accepts a single obs (dict / tensor) or a
        batched ``(B, ...)`` tensor; in the single-obs case the leading
        batch dim is added.
        """
        if isinstance(obs, dict) or (
            not isinstance(obs, Tensor) and obs is not None
        ):
            flat = self._flatten_single(obs).unsqueeze(0)
        elif isinstance(obs, Tensor) and obs.ndim == 1:
            flat = obs.unsqueeze(0)
        else:
            flat = obs
        return self.encoder(flat)

    def forward(self, obs) -> tuple[Tensor, Tensor]:
        """Returns ``(action_mean, value)`` — shapes ``(B, A)`` and ``(B,)``."""
        features = self._features(obs)
        mean = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return mean, value

    def distribution(self, obs) -> tuple[torch.distributions.Normal, Tensor]:
        """Returns ``(Normal(mean, std), value)``."""
        mean, value = self.forward(obs)
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max)
        std = log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std), value

    @torch.no_grad()
    def act(self, obs) -> tuple[Tensor, Tensor, Tensor]:
        """Sample an action. Returns ``(action, log_prob, value)``."""
        dist, value = self.distribution(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value

    def evaluate_actions(
        self,
        obs,
        actions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Re-score stored actions. Returns ``(log_prob, entropy, value)``."""
        dist, value = self.distribution(obs)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value


__all__ = ["VisionPolicy"]