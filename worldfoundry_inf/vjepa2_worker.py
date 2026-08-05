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
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

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
        past_frames: K (past-frame count) for the temporal predictor's input
            grid; defaults to 1.
        future_frames: H (future-frame count) for the temporal predictor's
            output grid; defaults to 1.
        patch_size: ViT patch size (16 for ViT-L).
        image_size: encoder input spatial size (224 for ViT-L).
    """

    def __init__(
        self,
        model_path: Path,
        adapter_name: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
        predictor_ckpt: Path | str | None = None,
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
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.patch_size = patch_size
        self.image_size = image_size
        self._past_grid: tuple[int, int, int] | None = None
        self._future_grid: tuple[int, int, int] | None = None
        if predictor_ckpt is not None:
            self.bundle = self._build_bundle(predictor_ckpt)

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
    def plan(self, video_tensor: torch.Tensor, actions: list[int]) -> tuple[torch.Tensor, list[int]]:
        """V-JEPA 2-AC: given (video, actions), return next-state features.

        Phase-1: passthrough that returns the encoded features and the input
        actions. Phase-2 W4 implements the action-conditioned predictor from
        facebookresearch/vjepa2.
        """
        feats = self.encode(video_tensor)
        return feats, list(actions)
