"""Multi-step action-conditioned latent rollout (W2.17).

Given a frozen context encoder + action-conditioned predictor + pixel
decoder, :class:`LatentRolloutPlanner` chains ``horizon`` "predict
latents -> decode to pixels -> re-encode" steps so the actor can look
``horizon`` steps into the future without ever touching a real env.

Public surface
--------------

- :func:`mask_all_tokens` — returns a ``(B, N)`` all-True mask; the
  predictor interprets this as "predict at every token", which is the
  pattern that makes ``TinyActionConditionedPredictor.forward`` return
  a dense ``(B*N, embed_dim)`` tensor that we can reshape back to
  ``(B, N, embed_dim)`` for the decoder.
- :class:`LatentRolloutPlanner` — ``nn.Module`` wrapping the encoder +
  predictor + decoder + EMA target encoder, with ``reset_context``,
  ``predict_step``, and ``predict_sequence`` methods.

Why the planner needs a separate ``encoder`` / ``target_encoder``:
``encoder`` is the live (gradient-tracked) encoder used to refresh the
context latent between imagined steps; ``target_encoder`` is the EMA
copy that provides the "ground truth" latents the predictor is
trained against (when callers want to compare imagined vs target,
which is a future feature — kept here so the seam exists for free).

Install/runtime: only ``torch``. No dm_control, no HF, no decord, no
opencv. The module is import-light.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

# The planner is duck-typed on the four module shapes:
# - encoder / target_encoder: ``forward(pixel_values_videos).last_hidden_state``
#   returning ``(B, T*Hg*Wg, embed_dim)``; ``config.{image_size,
#   patch_size, embed_dim}`` accessible as attributes.
# - predictor: ``forward(h, mask, action)`` returning either
#   ``(M_total, embed_dim)`` (HF path via :class:`JEPAPredictor`) or
#   ``(B, N, embed_dim)`` (tiny action-conditioned path).
# - decoder: ``forward(latents, num_frames)`` returning ``(B, T, 3, H, W)``
#   with ``encoder_config.image_size`` accessible.
#
# Both the v2.x tiny stack and the W2.18 HF VJEPA2Model + LoRA
# shim satisfy the surface, so the planner accepts any ``nn.Module``
# matching the contract.
from worldfoundry_train.model.tiny_jepa import (  # noqa: F401
    TinyActionConditionedPredictor,
    TinyPixelDecoder,
    TinyVJEPA2Encoder,
)


def mask_all_tokens(num_tokens: int, *, batch: int = 1, device: torch.device | str = "cpu") -> torch.Tensor:
    """Return a ``(batch, num_tokens)`` all-True bool mask.

    Used by :meth:`LatentRolloutPlanner.predict_step` to make the
    predictor emit a dense ``(batch*num_tokens, embed_dim)`` tensor
    that we can reshape to ``(batch, num_tokens, embed_dim)`` for the
    decoder. Keeping this as a free helper lets tests assert the
    contract without instantiating a planner.
    """
    if num_tokens < 1:
        raise ValueError(f"num_tokens must be >= 1; got {num_tokens}")
    if batch < 1:
        raise ValueError(f"batch must be >= 1; got {batch}")
    return torch.ones((batch, int(num_tokens)), dtype=torch.bool, device=device)


class LatentRolloutPlanner(nn.Module):
    """Multi-step latent imagination in front of the actor (W2.17).

    The orchestrator wires one of these per outer-loop iteration when
    ``cfg.policy_encoder == "tiny_vjepa2"`` and ``cfg.rollout_horizon > 1``.
    The actor's policy head is trained on the planner's imagined
    latents instead of the v2.x "zero pixels -> encoder -> pred ->
    flatten" placeholder, so the policy gradient has a real
    world-model-driven signal.

    Lifecycle:
        planner = LatentRolloutPlanner(...)
        planner.reset_context(pixel_first_frame)        # (1, T_first, 3, H, W)
        for t in range(horizon):
            imagined = planner.predict_step(action_t)   # (1, N, embed_dim)
        # Or, as one call:
        latents, pixels = planner.predict_sequence(pixel_first_frame, actions_seq)

    Args:
        encoder: live encoder used to refresh the context latent after
            each imagined step. Typically shared with the train phase
            (so the policy gradient flows back into the encoder).
        target_encoder: EMA copy of ``encoder`` kept here only so the
            planner's API surface mirrors the train/inference seam. Not
            used by :meth:`predict_step` directly.
        predictor: action-conditioned predictor (see
            :class:`worldfoundry_train.model.tiny_jepa.TinyActionConditionedPredictor`).
            Its forward signature is
            ``(context_tokens (B,N,D), bool_masked_pos (B,N), action_embedding (B,A))``
            returning ``(M_total, D)``; we reshape the dense case back to
            ``(B, N, D)``.
        decoder: per-patch pixel decoder (see
            :class:`worldfoundry_train.model.tiny_jepa.TinyPixelDecoder`).
            Used to render the imagined latent back to pixels before
            re-encoding for the next step.
        image_size: spatial side length expected by ``encoder`` /
            ``decoder``. Must equal ``encoder.config.image_size``.
        patch_size: encoder patch size. Must equal
            ``encoder.config.patch_size``.
        action_dim: dimensionality of the continuous action vector;
            must equal ``predictor.action_dim``.
        horizon: how many imagined steps to chain per
            :meth:`predict_sequence` call. Default ``1`` matches the
            v2.x single-step placeholder; ``>=2`` enables the multi-step
            latent rollout pattern.
        grad_through_predictor: when ``True`` (default), the imagined
            latents stay in the autograd graph so a downstream policy
            gradient can backprop through the planner. When ``False``
            (eval / log-only), the imagined latents are detached so
            their construction does not allocate optimizer state.
        num_context_frames: number of frames ``encoder`` expects per
            context input (``T_first``). Default ``1`` — the actor sees
            one frame at a time, so context is single-frame. The
            encoder accepts up to 16 frames so callers can raise this
            without touching the planner.
    """

    def __init__(
        self,
        *,
        encoder: nn.Module,
        target_encoder: nn.Module,
        predictor: nn.Module,
        decoder: nn.Module,
        image_size: int,
        patch_size: int,
        action_dim: int,
        horizon: int = 1,
        grad_through_predictor: bool = True,
        num_context_frames: int = 1,
    ) -> None:
        super().__init__()
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1; got {horizon}")
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
        if action_dim < 1:
            raise ValueError(f"action_dim must be >= 1; got {action_dim}")
        if num_context_frames < 1:
            raise ValueError(
                f"num_context_frames must be >= 1; got {num_context_frames}"
            )
        # Sanity-check the module wiring so a mismatch in
        # patch_size/image_size/action_dim blows up at construction
        # time, not at the first imagined step.
        if encoder.config.image_size != image_size:
            raise ValueError(
                f"encoder.config.image_size={encoder.config.image_size} "
                f"must equal image_size={image_size}"
            )
        if encoder.config.patch_size != patch_size:
            raise ValueError(
                f"encoder.config.patch_size={encoder.config.patch_size} "
                f"must equal patch_size={patch_size}"
            )
        # Action-conditioning is optional. The W2.18 HF smoke uses
        # ``HfJEPAPredictorShim`` which accepts-and-ignores ``action``
        # (see ``worldfoundry_train/model/hf_vjepa_shim.py``); it does
        # not expose ``action_dim`` and the planner's ``predict_step``
        # still calls ``predictor(h, mask, action)`` (the shim ignores
        # the kwarg). When the predictor doesn't expose ``action_dim``
        # we skip the wiring check — the planner still passes an action
        # tensor through, but the shim treats it as a no-op tag.
        predictor_action_dim = getattr(predictor, "action_dim", None)
        if predictor_action_dim is not None and predictor_action_dim != action_dim:
            raise ValueError(
                f"predictor.action_dim={predictor_action_dim} must equal "
                f"action_dim={action_dim}"
            )
        if decoder.encoder_config.image_size != image_size:
            raise ValueError(
                f"decoder.encoder_config.image_size="
                f"{decoder.encoder_config.image_size} must equal "
                f"image_size={image_size}"
            )

        self.encoder = encoder
        self.target_encoder = target_encoder
        self.predictor = predictor
        self.decoder = decoder
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.grad_through_predictor = bool(grad_through_predictor)
        self.num_context_frames = int(num_context_frames)

        self._grid_h = self.image_size // self.patch_size
        self._grid_w = self.image_size // self.patch_size
        self._tokens_per_frame = self._grid_h * self._grid_w
        # The encoder flattens (T, Hg, Wg) -> T*Hg*Wg along the token
        # axis, so the predictor's mask has to cover all of them.
        self._num_tokens = self.num_context_frames * self._tokens_per_frame
        self._embed_dim = self.encoder.config.embed_dim

        # Context latent cache. None until ``reset_context`` is called.
        # Registered as a buffer so ``.to(device)`` moves it along with
        # the planner, but marked ``persistent=False`` so it does not
        # end up in the state_dict.
        self.register_buffer(
            "_h_context",
            torch.zeros(0),
            persistent=False,
        )
        self._context_initialized = False

    @property
    def num_tokens(self) -> int:
        return self._num_tokens

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def context_initialized(self) -> bool:
        return self._context_initialized

    def reset_context(self, pixel_first_frame: torch.Tensor) -> None:
        """Encode ``pixel_first_frame`` and stash it as the rollout seed.

        Args:
            pixel_first_frame: ``(1, T_first, 3, H, W)`` float in
                ``[0, 1]`` (any normalized range the encoder accepts).
                ``T_first`` must equal :attr:`num_context_frames`.
        """
        self._validate_pixels(pixel_first_frame, expected_T=self.num_context_frames)
        # Drop the gradient so the context latent stays a constant seed
        # across the entire rollout — backprop only flows through the
        # predictor's imagined latents.
        with torch.no_grad():
            out = self.encoder(pixel_first_frame)
        # Detach again in case ``encoder`` has grad enabled elsewhere.
        self._h_context = out.last_hidden_state.detach()
        self._context_initialized = True

    def predict_step(self, action: torch.Tensor) -> torch.Tensor:
        """Imagine the next-frame latent for a single ``action``.

        Args:
            action: ``(1, action_dim)`` float tensor (the actor's
                chosen action for this imagined step).

        Returns:
            ``(1, num_tokens, embed_dim)`` latent — the predictor's
            output for every token position, reshaped from the
            ``(M_total, embed_dim)`` flattened form.

        Raises:
            RuntimeError: if :meth:`reset_context` was not called.
            ValueError: if ``action`` shape is wrong.
        """
        if not self._context_initialized:
            raise RuntimeError(
                "predict_step called before reset_context; call "
                "reset_context(pixel_first_frame) first"
            )
        self._validate_action(action)
        B = int(action.shape[0])
        if B != 1:
            raise ValueError(
                f"predict_step expects batch=1; got batch={B}. "
                f"Use a batched reset_context + predict_sequence for "
                f"batched rollouts."
            )
        mask = mask_all_tokens(
            self._num_tokens, batch=1, device=self._h_context.device,
        )
        if self.grad_through_predictor:
            predicted = self.predictor(self._h_context, mask, action)
        else:
            with torch.no_grad():
                predicted = self.predictor(self._h_context, mask, action)
        # predicted: (N, embed_dim) -> (1, N, embed_dim)
        predicted = predicted.reshape(1, self._num_tokens, self._embed_dim)
        return predicted

    def refresh_context_from_latents(
        self, imagined_latents: torch.Tensor, num_frames: int = 1,
    ) -> None:
        """Decode ``imagined_latents`` to pixels and re-encode into ``_h_context``.

        Called internally by :meth:`predict_sequence` between steps;
        exposed for callers that want to hand-roll a custom rollout
        pattern (e.g. gradient-aware chains where the encoder stays in
        the graph).
        """
        self._validate_latents(imagined_latents)
        if self.grad_through_predictor:
            pixels = self.decoder(imagined_latents, num_frames=num_frames)
            out = self.encoder(pixels)
        else:
            with torch.no_grad():
                pixels = self.decoder(imagined_latents, num_frames=num_frames)
                out = self.encoder(pixels)
        self._h_context = out.last_hidden_state.detach()
        self._context_initialized = True

    def predict_sequence(
        self,
        pixel_first_frame: torch.Tensor,
        actions_seq: Sequence[torch.Tensor] | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the full ``horizon``-step imagined rollout.

        Args:
            pixel_first_frame: ``(1, T_first, 3, H, W)`` seed frame.
            actions_seq: sequence of ``horizon`` action tensors, each
                ``(1, action_dim)``. A ``torch.Tensor`` of shape
                ``(horizon, 1, action_dim)`` is also accepted and is
                split internally.

        Returns:
            ``(predicted_latents, predicted_pixels)`` where
            ``predicted_latents`` is ``(horizon, num_tokens, embed_dim)``
            and ``predicted_pixels`` is ``(horizon, T_first, 3, H, W)``.
            Both are clamped to ``[0, 1]`` for the pixels (the decoder
            applies a sigmoid internally).
        """
        if isinstance(actions_seq, torch.Tensor):
            if actions_seq.dim() != 3 or actions_seq.shape[0] != self.horizon:
                raise ValueError(
                    f"actions_seq tensor must have shape "
                    f"(horizon={self.horizon}, 1, action_dim="
                    f"{self.action_dim}); got {tuple(actions_seq.shape)}"
                )
            actions = [actions_seq[t] for t in range(self.horizon)]
        else:
            actions = list(actions_seq)
        if len(actions) != self.horizon:
            raise ValueError(
                f"actions_seq length {len(actions)} must equal "
                f"horizon={self.horizon}"
            )

        self.reset_context(pixel_first_frame)
        latents_out: list[torch.Tensor] = []
        pixels_out: list[torch.Tensor] = []
        for t in range(self.horizon):
            imagined_latents = self.predict_step(actions[t])
            # imagined_latents: (1, N, embed_dim) -> squeeze batch dim so
            # the returned sequence is (horizon, N, embed_dim).
            latents_out.append(imagined_latents.squeeze(0).detach().cpu())
            # Decode for visualization + next-step seed.
            pixels = self.decoder(imagined_latents, num_frames=self.num_context_frames)
            # pixels: (1, T, C, H, W) -> (T, C, H, W) so the returned
            # sequence stacks cleanly to (horizon, T, C, H, W).
            pixels_out.append(pixels.squeeze(0).detach().cpu())
            # Refresh context for the next step (unless this is the last).
            if t < self.horizon - 1:
                self.refresh_context_from_latents(
                    imagined_latents, num_frames=self.num_context_frames,
                )
        predicted_latents = torch.stack(latents_out, dim=0)
        predicted_pixels = torch.stack(pixels_out, dim=0)
        return predicted_latents, predicted_pixels

    # --- validation helpers ----------------------------------------------

    def _validate_pixels(
        self, pixel_first_frame: torch.Tensor, *, expected_T: int,
    ) -> None:
        if pixel_first_frame.dim() != 5:
            raise ValueError(
                f"pixel_first_frame must be (B, T, C, H, W); got shape "
                f"{tuple(pixel_first_frame.shape)}"
            )
        B, T, C, H, W = pixel_first_frame.shape
        if B != 1:
            raise ValueError(
                f"pixel_first_frame batch must be 1; got {B}. "
                f"Batched rollouts go through predict_sequence with a "
                f"batched reset_context in a future revision."
            )
        if T != expected_T:
            raise ValueError(
                f"pixel_first_frame T={T} must equal "
                f"num_context_frames={expected_T}"
            )
        if H != self.image_size or W != self.image_size:
            raise ValueError(
                f"pixel_first_frame H/W must equal image_size="
                f"{self.image_size}; got H={H}, W={W}"
            )
        if C != self.encoder.config.in_channels:
            raise ValueError(
                f"pixel_first_frame C={C} must equal "
                f"encoder.in_channels={self.encoder.config.in_channels}"
            )

    def _validate_action(self, action: torch.Tensor) -> None:
        if action.dim() != 2:
            raise ValueError(
                f"action must be (B, action_dim={self.action_dim}); "
                f"got shape {tuple(action.shape)}"
            )
        if action.shape[1] != self.action_dim:
            raise ValueError(
                f"action.shape[1]={action.shape[1]} must equal "
                f"action_dim={self.action_dim}"
            )

    def _validate_latents(self, latents: torch.Tensor) -> None:
        expected = (1, self._num_tokens, self._embed_dim)
        if tuple(latents.shape) != expected:
            raise ValueError(
                f"latents must have shape {expected}; got "
                f"{tuple(latents.shape)}"
            )


__all__ = ["LatentRolloutPlanner", "mask_all_tokens"]