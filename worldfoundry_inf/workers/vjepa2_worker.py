"""VJEPA2Worker — minimal vLLM-Ascend plugin skeleton for V-JEPA 2 inference.

Per spec §4.1 + §4.2:
- V-JEPA 2 is non-autoregressive ViT; we DO NOT use vLLM's PagedAttention scheduler.
- We DO reuse vLLM-Ascend's Hardware Pluggable mechanism: register a worker class
  that exposes encode/predict/plan methods, sharing the operator library
  (FlashAttention, RMSNorm, quant).
- Phase-1: minimal skeleton; Phase-2 W3 wires the full Hardware Pluggable registration.

W24 (Phase-3): the worker optionally loads a W22 JEPA training checkpoint
(``predictor_ckpt``) to expose :meth:`predict_temporal` — past K frames ->
future H frame latents. When ``predictor_ckpt`` is None, the worker is
byte-compatible with the W22 surface (``encode`` / ``predict`` / ``plan``
only); when set, ``predict_temporal`` is enabled and the bundle's
online/target encoders + temporal predictor are used for the temporal path.

W27 v2 (Phase-3): the worker optionally loads a W27 V-JEPA 2-AC action-conditioned
checkpoint (``action_predictor_ckpt``) to expose :meth:`predict_action_conditioned`
— past K frames + K actions -> future H frame latents. When set, the action
bundle's online/target encoders + ActionConditionedPredictor are used for the
action path. The two ckpts are independent: a worker can run W22-only,
W27-only, or both.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from worldfoundry_train.model.action_jepa_inference import (
    load_action_conditioned_inference_model,
)
from worldfoundry_train.model.jepa_inference import (
    JEPInferenceBundle,
    load_jepa_inference_model,
)

# V-JEPA 2 ViT-L: image_size=224, patch_size=16 -> 14x14 patch grid per frame.
DEFAULT_PATCH_SIZE = 16
DEFAULT_IMAGE_SIZE = 224


def _load_hf_model(path: Path, dtype: torch.dtype = torch.bfloat16):
    """Lazy import of HF transformers; isolated to allow test mocking."""
    from transformers import VJEPA2Model
    return VJEPA2Model.from_pretrained(str(path), torch_dtype=dtype).eval()


class VJEPA2Worker:
    """Wraps a V-JEPA 2 model for batch inference.

    Args:
        model_path: path to the HF V-JEPA 2 model directory.
        adapter_name: optional LoRA adapter name (Phase-2 W3).
        dtype: model dtype for HF model load.
        predictor_ckpt: optional path to a W22 JEPA training checkpoint
            (``load_jepa_inference_model``). When set, :meth:`predict_temporal`
            is enabled and ``past_frames`` / ``future_frames`` configure the
            :class:`TemporalPredictor` grid.
        action_predictor_ckpt: optional path to a W27 V-JEPA 2-AC action-
            conditioned checkpoint (``load_action_conditioned_inference_model``).
            When set, :meth:`predict_action_conditioned` is enabled and the
            action bundle's :class:`ActionConditionedPredictor` is used for the
            past-K + K-actions -> future-H path. The ``action_dim`` must match
            the training-time action projection dimensionality.
        action_dim: per-frame action vector width (e.g. 7 for joint positions
            + gripper; 8 for joint + gripper_state). Recorded on the worker
            for the contract that the action tensor sent to
            :meth:`predict_action_conditioned` must have ``last_dim=action_dim``.
        past_frames: K (past-frame count) for the temporal/action predictor's
            input grid; defaults to 1.
        future_frames: H (future-frame count) for the temporal/action
            predictor's output grid; defaults to 1.
        patch_size: ViT patch size (16 for ViT-L).
        image_size: encoder input spatial size (224 for ViT-L).
    """

    def __init__(
        self,
        model_path: Path,
        adapter_name: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
        predictor_ckpt: Path | str | None = None,
        action_predictor_ckpt: Path | str | None = None,
        action_dim: int = 7,
        past_frames: int = 1,
        future_frames: int = 1,
        patch_size: int = DEFAULT_PATCH_SIZE,
        image_size: int = DEFAULT_IMAGE_SIZE,
    ) -> None:
        self.model_path = Path(model_path)
        self.adapter_name = adapter_name
        self.dtype = dtype
        self.model = _load_hf_model(self.model_path, dtype=self.dtype)

        # W24: optional JEPA bundle (online/target encoders + predictors).
        # When predictor_ckpt is None, ``self.bundle`` stays None and
        # ``predict_temporal`` raises RuntimeError (the API endpoint maps
        # that to 503 "temporal mode not enabled for this worker").
        self.bundle: Optional[JEPInferenceBundle] = None
        # W27 v2: separate action-conditioned bundle. Independent of
        # ``self.bundle`` so W22-only and W27-only configs both work.
        self.action_bundle: Optional[JEPInferenceBundle] = None
        self.action_dim = int(action_dim)
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.patch_size = patch_size
        self.image_size = image_size
        self._past_grid: tuple[int, int, int] | None = None
        self._future_grid: tuple[int, int, int] | None = None
        if predictor_ckpt is not None:
            self.bundle = self._build_bundle(predictor_ckpt)
        if action_predictor_ckpt is not None:
            self.action_bundle = self._build_action_bundle(action_predictor_ckpt)

    def _build_bundle(self, ckpt: Path | str) -> JEPInferenceBundle:
        gh = self.image_size // self.patch_size
        gw = self.image_size // self.patch_size
        past_grid = (self.past_frames, gh, gw)
        future_grid = (self.future_frames, gh, gw)
        self._past_grid = past_grid
        self._future_grid = future_grid
        return load_jepa_inference_model(
            ckpt, self.model,
            past_grid=past_grid, future_grid=future_grid,
            map_location="cpu",
        )

    def _build_action_bundle(self, ckpt: Path | str) -> JEPInferenceBundle:
        """Build the W27 v2 action-conditioned bundle (sister to _build_bundle).

        Reuses the same ``_past_grid`` / ``_future_grid`` so the API stays
        consistent with the temporal path. The action predictor's
        ``action_dim`` is recorded on the bundle's ``action_signature`` for
        downstream introspection.
        """
        if self._past_grid is None or self._future_grid is None:
            # If only action_predictor_ckpt was set (no W22), build the
            # grid here so the action bundle has consistent shape.
            gh = self.image_size // self.patch_size
            gw = self.image_size // self.patch_size
            self._past_grid = (self.past_frames, gh, gw)
            self._future_grid = (self.future_frames, gh, gw)
        # The W27 v2 loader overwrites self.model in-place with the ckpt's
        # online encoder weights (same as W24). When both predictor_ckpt and
        # action_predictor_ckpt are set, the latter wins because it is
        # loaded last — the encoder is JEPA-trained on the same video
        # stream so the two ckpts' encoder weights are functionally the same.
        bundle = load_action_conditioned_inference_model(
            ckpt, self.model,
            past_grid=self._past_grid, future_grid=self._future_grid,
            map_location="cpu",
        )
        # Sanity: the ckpt's action_dim must match the worker's contract.
        ck_action_dim = int(bundle.action_signature.get("action_dim", self.action_dim))
        if ck_action_dim != self.action_dim:
            raise ValueError(
                f"action_predictor_ckpt trained with action_dim={ck_action_dim} "
                f"but worker was constructed with action_dim={self.action_dim}. "
                f"Pass --action-dim={ck_action_dim} to the server."
            )
        return bundle

    @torch.no_grad()
    def encode(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """Encode a batch of videos into dense features.

        Args:
            video_tensor: (B, T, C, H, W) float tensor (normalized).
        Returns:
            (B, N_tokens, hidden_dim) features.
        """
        out = self.model(pixel_values_videos=video_tensor)
        return out.last_hidden_state

    @torch.no_grad()
    def predict(self, masked_video: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Predict masked tokens (V-JEPA 2 native objective)."""
        out = self.model(pixel_values_videos=masked_video, bool_masked_pos=mask)
        return out.last_hidden_state

    @torch.no_grad()
    def predict_masked_batch(
        self,
        videos: list[torch.Tensor],
        mask_ratios: list[float],
    ) -> list[torch.Tensor]:
        """Batched masked-mode prediction (W25 dynamic batcher).

        Stacks ``videos`` (each ``(T, C, H, W)``) into ``(B, T, C, H, W)`` and
        builds a per-sample bool mask, then runs a single forward. Returns
        per-sample outputs in the same order. All videos must share
        ``T``/``C``/``H``/``W``; mixed-shape batches raise ValueError.
        """
        if len(videos) != len(mask_ratios):
            raise ValueError(
                f"videos ({len(videos)}) and mask_ratios ({len(mask_ratios)}) "
                f"length mismatch"
            )
        if not videos:
            return []
        ref = videos[0].shape
        for i, v in enumerate(videos):
            if tuple(v.shape) != tuple(ref):
                raise ValueError(
                    f"video[{i}] has shape {tuple(v.shape)} != reference "
                    f"{tuple(ref)}; batched masked predict requires uniform "
                    f"shape"
                )
        batched = torch.stack(videos, dim=0)
        b, t = batched.shape[:2]
        mask = torch.zeros(b, t, dtype=torch.bool)
        for i, r in enumerate(mask_ratios):
            mask[i, : int(r * t)] = True
        out = self.model(pixel_values_videos=batched, bool_masked_pos=mask)
        return [out.last_hidden_state[i] for i in range(b)]

    @torch.no_grad()
    def predict_temporal_batch(
        self,
        videos: list[torch.Tensor],
        past_frames_list: list[int],
        future_frames_list: list[int],
    ) -> list[tuple[torch.Tensor, float]]:
        """Batched temporal-mode prediction (W25 dynamic batcher).

        Each video must have at least ``past + future`` frames; the batcher
        requires uniform ``past_frames`` and ``future_frames`` across the
        batch (matching this worker's configured grid). Returns
        ``(predicted_features, confidence)`` per request in input order.
        """
        if not (len(videos) == len(past_frames_list) == len(future_frames_list)):
            raise ValueError(
                "videos, past_frames_list, future_frames_list must be the "
                "same length"
            )
        if not videos:
            return []
        if self.bundle is None:
            raise RuntimeError(
                "predict_temporal_batch() requires a worker constructed with "
                "predictor_ckpt; current worker is masked-mode only"
            )
        k = past_frames_list[0]
        h = future_frames_list[0]
        if any(pk != k for pk in past_frames_list) or any(fh != h for fh in future_frames_list):
            raise ValueError(
                "temporal-mode batch requires uniform past_frames and "
                "future_frames across the batch; got "
                f"past_frames={set(past_frames_list)}, "
                f"future_frames={set(future_frames_list)}"
            )
        if k != self.past_frames or h != self.future_frames:
            raise RuntimeError(
                f"batch past/future frames ({k}/{h}) don't match worker "
                f"grid ({self.past_frames}/{self.future_frames})"
            )

        ref = videos[0].shape
        for i, v in enumerate(videos):
            if tuple(v.shape) != tuple(ref):
                raise ValueError(
                    f"video[{i}] has shape {tuple(v.shape)} != reference "
                    f"{tuple(ref)}; batched temporal predict requires "
                    f"uniform shape"
                )
        batched = torch.stack(videos, dim=0)  # (B, T, C, H, W)
        past_video = batched[:, :k]
        future_video = batched[:, k:k + h]

        past_feats = self.bundle.online(
            pixel_values_videos=past_video
        ).last_hidden_state
        predicted = self.bundle.temporal_predictor(
            past_feats, future_grid=self._future_grid
        )
        target = self.bundle.target(
            pixel_values_videos=future_video
        ).last_hidden_state

        sim = torch.nn.functional.cosine_similarity(predicted, target, dim=-1)
        confidence_per_sample = sim.mean(dim=-1)  # (B,)
        return [
            (predicted[i], float(confidence_per_sample[i]))
            for i in range(batched.shape[0])
        ]

    @torch.no_grad()
    def predict_temporal(
        self,
        video_tensor: torch.Tensor,
        past_frames: int,
        future_frames: int,
    ) -> tuple[torch.Tensor, float]:
        """Predict the next ``future_frames`` frame latents from the first
        ``past_frames`` frame latents (W24 temporal mode).

        Args:
            video_tensor: (B, T, C, H, W) float tensor; T must be >=
                past_frames + future_frames. The first ``past_frames``
                frames feed the online encoder; the next ``future_frames``
                frames are encoded by the EMA target encoder to score the
                prediction's confidence.
            past_frames: K (must match this worker's configured
                ``past_frames``).
            future_frames: H (must match this worker's configured
                ``future_frames``).

        Returns:
            (future_features, confidence): ``future_features`` is
            ``(B, N_future, hidden_dim)``; ``confidence`` is the mean cosine
            similarity between predicted and target-encoder features on
            [-1, 1] (higher = more confident).

        Raises:
            RuntimeError: this worker was constructed without
                ``predictor_ckpt`` (temporal mode not enabled), or
                ``past_frames`` / ``future_frames`` don't match the
                worker's configured grid, or the video is too short.
        """
        if self.bundle is None:
            raise RuntimeError(
                "predict_temporal() requires a worker constructed with "
                "predictor_ckpt; current worker is masked-mode only"
            )
        if past_frames != self.past_frames or future_frames != self.future_frames:
            raise RuntimeError(
                f"worker is configured for past_frames={self.past_frames}, "
                f"future_frames={self.future_frames}; request has "
                f"past_frames={past_frames}, future_frames={future_frames}. "
                f"Adjust the worker's predictor_ckpt past/future grid"
            )
        if video_tensor.shape[1] < past_frames + future_frames:
            raise RuntimeError(
                f"video has {video_tensor.shape[1]} frames; need at least "
                f"past_frames+future_frames={past_frames + future_frames}"
            )

        past_video = video_tensor[:, :past_frames]
        future_video = video_tensor[:, past_frames:past_frames + future_frames]

        past_feats = self.bundle.online(
            pixel_values_videos=past_video
        ).last_hidden_state
        predicted = self.bundle.temporal_predictor(
            past_feats, future_grid=self._future_grid
        )
        target = self.bundle.target(
            pixel_values_videos=future_video
        ).last_hidden_state

        sim = torch.nn.functional.cosine_similarity(predicted, target, dim=-1)
        confidence = float(sim.mean())
        return predicted, confidence

    @torch.no_grad()
    def predict_action_conditioned_batch(
        self,
        videos: list[torch.Tensor],
        actions_per_sample: list[torch.Tensor],
        past_frames_list: list[int],
        future_frames_list: list[int],
    ) -> list[tuple[torch.Tensor, float]]:
        """Batched action-conditioned prediction (W33 + W25 dynamic batcher).

        Sister to :meth:`predict_temporal_batch`. Each request must share
        uniform ``past_frames`` / ``future_frames`` (matching this worker's
        configured grid); ``actions_per_sample[i]`` must be shape
        ``(past_frames, action_dim)``. Returns ``(predicted_features,
        confidence)`` per request in input order.
        """
        if not (
            len(videos)
            == len(actions_per_sample)
            == len(past_frames_list)
            == len(future_frames_list)
        ):
            raise ValueError(
                "videos, actions_per_sample, past_frames_list, "
                "future_frames_list must all be the same length"
            )
        if not videos:
            return []
        if self.action_bundle is None:
            raise RuntimeError(
                "predict_action_conditioned_batch() requires a worker "
                "constructed with action_predictor_ckpt; current worker is "
                "temporal-mode only (or no predictor at all)"
            )
        k = past_frames_list[0]
        h = future_frames_list[0]
        if any(pk != k for pk in past_frames_list) or any(
            fh != h for fh in future_frames_list
        ):
            raise ValueError(
                "action-mode batch requires uniform past_frames and "
                "future_frames across the batch; got "
                f"past_frames={set(past_frames_list)}, "
                f"future_frames={set(future_frames_list)}"
            )
        if k != self.past_frames or h != self.future_frames:
            raise RuntimeError(
                f"batch past/future frames ({k}/{h}) don't match worker "
                f"grid ({self.past_frames}/{self.future_frames})"
            )

        ref = videos[0].shape
        for i, v in enumerate(videos):
            if tuple(v.shape) != tuple(ref):
                raise ValueError(
                    f"video[{i}] has shape {tuple(v.shape)} != reference "
                    f"{tuple(ref)}; batched action predict requires uniform "
                    f"video shape"
                )
        ref_action = actions_per_sample[0].shape
        for i, a in enumerate(actions_per_sample):
            if tuple(a.shape) != tuple(ref_action):
                raise ValueError(
                    f"actions[{i}] has shape {tuple(a.shape)} != reference "
                    f"{tuple(ref_action)}; batched action predict requires "
                    f"uniform action shape"
                )

        batched = torch.stack(videos, dim=0)  # (B, T, C, H, W)
        batched_actions = torch.stack(actions_per_sample, dim=0)  # (B, K, D)
        past_video = batched[:, :k]
        future_video = batched[:, k:k + h]

        past_feats = self.action_bundle.online(
            pixel_values_videos=past_video
        ).last_hidden_state
        predicted = self.action_bundle.action_predictor(
            past_feats, batched_actions, future_grid=self._future_grid,
        )
        target = self.action_bundle.target(
            pixel_values_videos=future_video
        ).last_hidden_state

        sim = torch.nn.functional.cosine_similarity(predicted, target, dim=-1)
        confidence_per_sample = sim.mean(dim=-1)  # (B,)
        return [
            (predicted[i], float(confidence_per_sample[i]))
            for i in range(batched.shape[0])
        ]

    @torch.no_grad()
    def plan(self, video_tensor: torch.Tensor, actions: list[int]) -> tuple[torch.Tensor, list[int]]:
        """V-JEPA 2-AC: given (video, actions), return next-state features.

        Phase-1: passthrough that returns the encoded features and the input
        actions. Phase-2 W4 implements the action-conditioned predictor from
        facebookresearch/vjepa2.
        """
        feats = self.encode(video_tensor)
        return feats, list(actions)

    @torch.no_grad()
    def predict_action_conditioned(
        self,
        video_tensor: torch.Tensor,
        actions: torch.Tensor,
        past_frames: int,
        future_frames: int,
    ) -> tuple[torch.Tensor, float]:
        """V-JEPA 2-AC: past K frames + K actions -> future H frame latents (W27 v2).

        Sister to :meth:`predict_temporal` but additionally conditioned on a
        per-frame action sequence. The action bundle's online encoder
        encodes the past K frames; the ActionConditionedPredictor maps
        (past_feats + action broadcast) to future H frame latents; the EMA
        target encoder scores the prediction's confidence by cosine
        similarity against target-encoded future frames.

        Args:
            video_tensor: ``(B, T, C, H, W)`` float tensor; ``T`` must be
                >= ``past_frames + future_frames``.
            actions: ``(B, past_frames, action_dim)`` per-frame action
                sequence. ``action_dim`` must match the worker / ckpt
                contract.
            past_frames: K (must match this worker's configured
                ``past_frames``).
            future_frames: H (must match this worker's configured
                ``future_frames``).

        Returns:
            (future_features, confidence): ``future_features`` is
            ``(B, N_future, encoder_dim)``; ``confidence`` is the mean
            cosine similarity between predicted and target-encoder
            features on [-1, 1].

        Raises:
            RuntimeError: this worker was constructed without
                ``action_predictor_ckpt`` (action mode not enabled), or
                ``past_frames`` / ``future_frames`` don't match the
                worker's configured grid, or the video is too short, or
                ``actions`` has the wrong shape.
        """
        if self.action_bundle is None:
            raise RuntimeError(
                "predict_action_conditioned() requires a worker constructed "
                "with action_predictor_ckpt; current worker is "
                "temporal-mode only (or no predictor at all)"
            )
        if past_frames != self.past_frames or future_frames != self.future_frames:
            raise RuntimeError(
                f"worker is configured for past_frames={self.past_frames}, "
                f"future_frames={self.future_frames}; request has "
                f"past_frames={past_frames}, future_frames={future_frames}. "
                f"Adjust the worker's action_predictor_ckpt past/future grid"
            )
        if video_tensor.shape[1] < past_frames + future_frames:
            raise RuntimeError(
                f"video has {video_tensor.shape[1]} frames; need at least "
                f"past_frames+future_frames={past_frames + future_frames}"
            )
        if actions.ndim != 3 or actions.shape[1] != past_frames:
            raise RuntimeError(
                f"actions must be (B, {past_frames}, action_dim), got "
                f"{tuple(actions.shape)}"
            )
        if actions.shape[2] != self.action_dim:
            raise RuntimeError(
                f"actions last dim {actions.shape[2]} != worker action_dim "
                f"{self.action_dim}"
            )

        past_video = video_tensor[:, :past_frames]
        future_video = video_tensor[:, past_frames:past_frames + future_frames]

        past_feats = self.action_bundle.online(
            pixel_values_videos=past_video
        ).last_hidden_state
        predicted = self.action_bundle.action_predictor(
            past_feats, actions, future_grid=self._future_grid
        )
        target = self.action_bundle.target(
            pixel_values_videos=future_video
        ).last_hidden_state

        sim = torch.nn.functional.cosine_similarity(predicted, target, dim=-1)
        confidence = float(sim.mean())
        return predicted, confidence
