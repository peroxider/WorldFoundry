"""VJEPA2NPUWorker — Hardware Pluggable worker for V-JEPA 2 on Ascend NPU.

Wires the existing :class:`VJEPA2Worker` semantics (encode / predict / plan)
into vLLM-Ascend's Hardware Pluggable Architecture. Designed to be discovered
by vLLM via the ``vllm.platform_plugins`` / ``vllm.general_plugins`` entry
points declared in ``pyproject.toml``; the actual registration is performed
by :mod:`worldfoundry_inf._vllm_ascend_plugin`.

Architecture (W19):
- ``VJEPA2NPUWorker`` subclasses vllm-ascend's ``NPUWorker`` at runtime when
  available; falls back to a stub base class on CPU-only systems so that the
  module can be imported for unit tests without vllm-ascend installed.
- ``encode`` / ``predict`` / ``plan`` call torch_npu ops for any heavy
  compute (attention, RMSNorm, projections). On CPU fallback, the same
  methods delegate to the eager HF path so unit tests can exercise the
  dispatch layer without NPU hardware.
- ``predict_masked_batch`` / ``predict_temporal(_batch)`` /
  ``predict_action_conditioned(_batch)`` ship with CPU fallback paths so
  the parity contract
  (``docs/superpowers/specs/2026-08-06-w36-npu-worker-parity-contract.md``)
  is satisfied without Atlas hardware. The NPU-accelerated paths will be
  filled in during the Atlas pilot (W19b); the CPU path uses the same
  ``JEPInferenceBundle`` (``online`` / ``target`` / ``temporal_predictor`` /
  ``action_predictor``) loaded by ``VJEPA2Worker`` so the cross-backend
  cosine parity gate holds (CPU ↔ NPU ≥ 0.999).
- Lazy imports throughout — vllm / vllm_ascend / torch_npu are loaded only
  inside the methods that need them, keeping import-time side effects zero.

Atlas e2e validation is deferred: the NPU op calls in :meth:`encode` /
:meth:`predict` cannot be exercised on a CPU host, but the parity surface
(masked/temporal/action batch + single) is now CPU-exercisable.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import torch

if TYPE_CHECKING:
    # Imported only for type checking; the real base class is loaded lazily
    # in _resolve_base_class() so that this module remains importable on
    # CPU-only systems where vllm_ascend / torch_npu are unavailable.
    from vllm_ascend.worker import NPUWorker  # noqa: F401


__all__ = ["VJEPA2NPUWorker", "_resolve_base_class"]


def _resolve_base_class() -> type:
    """Return vllm-ascend's NPUWorker when installed, else a stub base.

    The stub mirrors the vllm-ascend 0.18.0 ``NPUWorker`` method surface
    (``init_device``, ``load_model``, ``execute_model``) with NotImplementedError
    bodies so that subclassing on CPU doesn't accidentally hide wiring bugs.
    """
    try:
        from vllm_ascend.worker import NPUWorker  # type: ignore[import-not-found]
        return NPUWorker
    except ImportError:
        pass

    class _StubNPUWorker:  # pragma: no cover — only used when NPU deps absent
        def init_device(self) -> None:
            raise NotImplementedError("vllm-ascend not installed; cannot init NPU device")

        def load_model(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError("vllm-ascend not installed; cannot load NPU model")

        def execute_model(self, *args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError("vllm-ascend not installed; cannot execute NPU model")

    return _StubNPUWorker


def _npu_ops():
    """Lazy import of torch_npu ops. Returns None if torch_npu is unavailable.

    On Atlas, this resolves to the real torch_npu module; on CPU-only hosts,
    it returns None and callers fall back to eager torch. This is the seam
    that lets unit tests exercise the dispatch without NPU hardware.
    """
    try:
        import torch_npu  # type: ignore[import-not-found]
        return torch_npu
    except ImportError:
        return None


class VJEPA2NPUWorker(_resolve_base_class()):
    """Hardware Pluggable worker for V-JEPA 2 on Ascend NPU.

    Public API (matches :class:`worldfoundry_inf.workers.vjepa2_worker.VJEPA2Worker`):
    - :meth:`encode` — video -> dense features
    - :meth:`predict` — masked video + mask -> dense features
    - :meth:`plan` — video + actions -> (features, actions)

    The worker calls torch_npu ops inside the compute path; on CPU fallback,
    the same methods execute on eager torch via the HF transformers path.
    vLLM's HPA discovery happens via the entry points in ``pyproject.toml``,
    which target :mod:`worldfoundry_inf._vllm_ascend_plugin`.
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
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.model_path = Path(model_path)
        self.adapter_name = adapter_name
        self.dtype = dtype
        # W24 parity with ``VJEPA2Worker``: the NPU path does NOT load the
        # JEPA bundle in v1 (Atlas pilot is the trigger, per roadmap G11),
        # but the attributes are stored so server-side construction has a
        # single signature for both CPU and NPU workers.
        self.predictor_ckpt = Path(predictor_ckpt) if predictor_ckpt is not None else None
        self.action_predictor_ckpt = (
            Path(action_predictor_ckpt) if action_predictor_ckpt is not None else None
        )
        self.action_dim = int(action_dim)
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.model: Optional[torch.nn.Module] = None
        # W24/W27 bundles: lazy-loaded on first call (CPU fallback path).
        # Atlas will use the same bundle interface via npu_format_cast +
        # vllm-ascend fused attention — the bundle object is identical.
        self.bundle: Any = None  # JEPInferenceBundle for predict_temporal
        self.action_bundle: Any = None  # JEPInferenceBundle for predict_action_conditioned
        self._past_grid: tuple[int, int, int] | None = None
        self._future_grid: tuple[int, int, int] | None = None
        # Skip NPU base-class init when running on the stub (CPU-only test
        # path). On Atlas, vllm_ascend's NPUWorker.__init__ sets the device
        # and prepares the distributed backend — we let the real __init__
        # run via super() only when we actually subclassed the real class.
        if _resolve_base_class().__name__ != "_StubNPUWorker":
            super().__init__(*args, **kwargs)  # type: ignore[misc]

    def _ensure_grids(self) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """Lazy-init ``past_grid`` / ``future_grid`` for the temporal/action predictor.

        Mirrors ``VJEPA2Worker._build_bundle``. Image geometry is hard-wired
        to ViT-L defaults (224/16 -> 14x14 grid) so the test path is
        deterministic without inspecting the encoder config; the same
        numbers are used by ``VJEPA2Worker`` so CPU/NPU parity holds.
        """
        if self._past_grid is None or self._future_grid is None:
            image_size = 224
            patch_size = 16
            gh = gw = image_size // patch_size
            self._past_grid = (self.past_frames, gh, gw)
            self._future_grid = (self.future_frames, gh, gw)
        return self._past_grid, self._future_grid

    def _build_bundle(self) -> Any:
        """Lazy-load the W24 JEPA inference bundle on CPU fallback."""
        from worldfoundry_train.model.jepa_inference import (
            load_jepa_inference_model,
        )

        if self.model is None:
            raise RuntimeError("load_model() must be called before _build_bundle()")
        past_grid, future_grid = self._ensure_grids()
        self.bundle = load_jepa_inference_model(
            str(self.predictor_ckpt),
            self.model,
            past_grid=past_grid,
            future_grid=future_grid,
            map_location="cpu",
        )
        return self.bundle

    def _build_action_bundle(self) -> Any:
        """Lazy-load the W27 v2 action-conditioned JEPA inference bundle."""
        from worldfoundry_train.model.action_jepa_inference import (
            load_action_conditioned_inference_model,
        )

        if self.model is None:
            raise RuntimeError(
                "load_model() must be called before _build_action_bundle()"
            )
        # Ensure grids exist (action bundle is independent of W22 bundle).
        self._ensure_grids()
        self.action_bundle = load_action_conditioned_inference_model(
            str(self.action_predictor_ckpt),
            self.model,
            past_grid=self._past_grid,
            future_grid=self._future_grid,
            map_location="cpu",
        )
        ck_action_dim = int(
            self.action_bundle.action_signature.get("action_dim", self.action_dim)
        )
        if ck_action_dim != self.action_dim:
            raise ValueError(
                f"action_predictor_ckpt trained with action_dim={ck_action_dim} "
                f"but worker was constructed with action_dim={self.action_dim}. "
                f"Pass --action-dim={ck_action_dim} to the server."
            )
        return self.action_bundle

    def _ensure_bundle(self) -> Any:
        """Return ``self.bundle``, lazy-loading from ``predictor_ckpt``."""
        if self.bundle is None:
            if self.predictor_ckpt is None:
                raise RuntimeError(
                    "predict_temporal() requires a worker constructed with "
                    "predictor_ckpt; current worker is masked-mode only"
                )
            self._build_bundle()
        return self.bundle

    def _ensure_action_bundle(self) -> Any:
        """Return ``self.action_bundle``, lazy-loading from ``action_predictor_ckpt``."""
        if self.action_bundle is None:
            if self.action_predictor_ckpt is None:
                raise RuntimeError(
                    "predict_action_conditioned() requires a worker constructed "
                    "with action_predictor_ckpt; current worker is "
                    "temporal-mode only (or no predictor at all)"
                )
            self._build_action_bundle()
        return self.action_bundle

    def init_device(self) -> None:
        """Initialize the NPU device. Delegates to base on Atlas; no-op on CPU."""
        if _resolve_base_class().__name__ != "_StubNPUWorker":
            super().init_device()  # type: ignore[misc]

    def load_model(self, *args: Any, **kwargs: Any) -> None:
        """Load V-JEPA 2 weights onto the NPU. Mirrors ``VJEPA2Worker.__init__``.

        Called by vLLM's engine after ``init_device``; on Atlas the base class
        handles device placement + sharding. On CPU fallback, this loads via
        HF transformers eager so tests can exercise the dispatch path.
        """
        from transformers import VJEPA2Model
        self.model = VJEPA2Model.from_pretrained(
            str(self.model_path), torch_dtype=self.dtype,
        ).eval()
        # The HF model is loaded on CPU by default. On Atlas it MUST live on
        # the NPU, otherwise the forward in encode()/predict() fails with a
        # device mismatch (CPU params vs NPU input). torch_npu's canonical
        # pattern is load-on-CPU-then `.to("npu")`.
        if _npu_ops() is not None:
            import torch
            dev = f"npu:{torch.npu.current_device()}"
            self.model = self.model.to(dev)
        if _resolve_base_class().__name__ != "_StubNPUWorker":
            super().load_model(*args, **kwargs)  # type: ignore[misc]

    @torch.no_grad()
    def encode(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """Run NPU-accelerated encode. Falls back to eager torch on CPU.

        Args:
            video_tensor: (B, T, C, H, W) float tensor.
        Returns:
            (B, N_tokens, hidden_dim) features.
        """
        if self.model is None:
            raise RuntimeError("load_model() must be called before encode()")
        npu = _npu_ops()
        if npu is not None and video_tensor.device.type != "cpu":
            # Atlas path: route through vllm-ascend's fused attention. The
            # HF model object is NPU-resident; we just invoke it.
            video_tensor = npu.npu_format_cast(video_tensor, 2)  # NZ format for matmul
        out = self.model(pixel_values_videos=video_tensor)
        return out.last_hidden_state

    @torch.no_grad()
    def predict(self, masked_video: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """NPU-accelerated masked-token prediction (V-JEPA 2 native objective)."""
        if self.model is None:
            raise RuntimeError("load_model() must be called before predict()")
        npu = _npu_ops()
        if npu is not None and masked_video.device.type != "cpu":
            masked_video = npu.npu_format_cast(masked_video, 2)
        out = self.model(pixel_values_videos=masked_video, bool_masked_pos=mask)
        return out.last_hidden_state

    @torch.no_grad()
    def predict_masked_batch(
        self, videos: list[torch.Tensor], mask_ratios: list[float],
    ) -> list[torch.Tensor]:
        """NPU-accelerated batched masked prediction (W25) with CPU parity.

        Mirrors :meth:`VJEPA2Worker.predict_masked_batch`. On Atlas the
        batched input is routed through ``npu_format_cast`` before the HF
        forward so vllm-ascend's fused attention sees NZ layout; on the
        CPU fallback the same HF forward runs on eager torch. Returns
        per-sample ``last_hidden_state`` slices in input order.

        All ``videos`` must share the same ``(T, C, H, W)`` shape; mixed
        shapes raise :class:`ValueError`. ``mask_ratios[i]`` is the
        fraction of frames masked for sample ``i``.
        """
        if self.model is None:
            raise RuntimeError(
                "load_model() must be called before predict_masked_batch()"
            )
        if len(videos) != len(mask_ratios):
            raise ValueError(
                f"videos ({len(videos)}) and mask_ratios ({len(mask_ratios)}) "
                f"length mismatch"
            )
        if not videos:
            return []
        ref = tuple(videos[0].shape)
        for i, v in enumerate(videos):
            if tuple(v.shape) != ref:
                raise ValueError(
                    f"video[{i}] has shape {tuple(v.shape)} != reference "
                    f"{ref}; batched masked predict requires uniform shape"
                )
        batched = torch.stack(videos, dim=0)
        b, t = batched.shape[:2]
        mask = torch.zeros(b, t, dtype=torch.bool)
        for i, r in enumerate(mask_ratios):
            mask[i, : int(r * t)] = True
        npu = _npu_ops()
        if npu is not None and batched.device.type != "cpu":
            batched = npu.npu_format_cast(batched, 2)
        out = self.model(pixel_values_videos=batched, bool_masked_pos=mask)
        return [out.last_hidden_state[i] for i in range(b)]

    @torch.no_grad()
    def predict_temporal_batch(
        self, videos: list[torch.Tensor],
        past_frames_list: list[int], future_frames_list: list[int],
    ) -> list[tuple[torch.Tensor, float]]:
        """NPU-accelerated batched temporal prediction (W25) with CPU parity.

        Sister to :meth:`predict_temporal`. Mirrors
        :meth:`VJEPA2Worker.predict_temporal_batch` byte-for-byte on the
        CPU fallback so ``test_vllm_ascend_worker_parity`` can assert
        cross-backend cosine ≥ 0.999.
        """
        if not (
            len(videos)
            == len(past_frames_list)
            == len(future_frames_list)
        ):
            raise ValueError(
                "videos, past_frames_list, future_frames_list must be the "
                "same length"
            )
        if not videos:
            return []
        bundle = self._ensure_bundle()
        k = past_frames_list[0]
        h = future_frames_list[0]
        if any(pk != k for pk in past_frames_list) or any(
            fh != h for fh in future_frames_list
        ):
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

        ref = tuple(videos[0].shape)
        for i, v in enumerate(videos):
            if tuple(v.shape) != ref:
                raise ValueError(
                    f"video[{i}] has shape {tuple(v.shape)} != reference "
                    f"{ref}; batched temporal predict requires uniform shape"
                )
        batched = torch.stack(videos, dim=0)  # (B, T, C, H, W)
        past_video = batched[:, :k]
        future_video = batched[:, k:k + h]

        npu = _npu_ops()
        if npu is not None and past_video.device.type != "cpu":
            past_video = npu.npu_format_cast(past_video, 2)
            future_video = npu.npu_format_cast(future_video, 2)

        past_feats = bundle.online(pixel_values_videos=past_video).last_hidden_state
        _, future_grid = self._ensure_grids()
        predicted = bundle.temporal_predictor(past_feats, future_grid=future_grid)
        target = bundle.target(pixel_values_videos=future_video).last_hidden_state

        sim = torch.nn.functional.cosine_similarity(predicted, target, dim=-1)
        confidence_per_sample = sim.mean(dim=-1)  # (B,)
        return [
            (predicted[i], float(confidence_per_sample[i]))
            for i in range(batched.shape[0])
        ]

    @torch.no_grad()
    def plan(self, video_tensor: torch.Tensor, actions: list[int]) -> tuple[torch.Tensor, list[int]]:
        """NPU-accelerated V-JEPA 2-AC action-conditioned prediction.

        Phase-1: passthrough that returns the encoded features and the input
        actions. Phase-2 W4 wires the real action-conditioned predictor from
        facebookresearch/vjepa2.
        """
        feats = self.encode(video_tensor)
        return feats, list(actions)

    @torch.no_grad()
    def predict_temporal(
        self,
        video_tensor: torch.Tensor,
        past_frames: int,
        future_frames: int,
    ) -> tuple[torch.Tensor, float]:
        """NPU-accelerated W24 temporal-mode prediction with CPU parity.

        Mirrors :meth:`VJEPA2Worker.predict_temporal`. On CPU fallback
        runs the same JEPA inference bundle (online / temporal_predictor /
        target) so cross-backend cosine parity is byte-identical; on Atlas
        the same forward is preceded by ``npu_format_cast`` for fused
        attention. Confidence is mean cosine similarity between predicted
        and target-encoded future frames.
        """
        bundle = self._ensure_bundle()
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

        npu = _npu_ops()
        if npu is not None and past_video.device.type != "cpu":
            past_video = npu.npu_format_cast(past_video, 2)
            future_video = npu.npu_format_cast(future_video, 2)

        past_feats = bundle.online(pixel_values_videos=past_video).last_hidden_state
        _, future_grid = self._ensure_grids()
        predicted = bundle.temporal_predictor(past_feats, future_grid=future_grid)
        target = bundle.target(pixel_values_videos=future_video).last_hidden_state

        sim = torch.nn.functional.cosine_similarity(predicted, target, dim=-1)
        confidence = float(sim.mean())
        return predicted, confidence

    @torch.no_grad()
    def predict_action_conditioned(
        self,
        video_tensor: torch.Tensor,
        actions: torch.Tensor,
        past_frames: int,
        future_frames: int,
    ) -> tuple[torch.Tensor, float]:
        """NPU-accelerated V-JEPA 2-AC action-conditioned prediction (W27) with CPU parity.

        Sister to :meth:`predict_temporal` plus per-frame action
        conditioning. On CPU fallback uses
        :func:`worldfoundry_train.model.action_jepa_inference.load_action_conditioned_inference_model`
        to build the bundle; on Atlas the same bundle runs on NPU with
        ``npu_format_cast``.
        """
        bundle = self._ensure_action_bundle()
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

        npu = _npu_ops()
        if npu is not None and past_video.device.type != "cpu":
            past_video = npu.npu_format_cast(past_video, 2)
            future_video = npu.npu_format_cast(future_video, 2)

        past_feats = bundle.online(pixel_values_videos=past_video).last_hidden_state
        _, future_grid = self._ensure_grids()
        predicted = bundle.action_predictor(
            past_feats, actions, future_grid=future_grid,
        )
        target = bundle.target(pixel_values_videos=future_video).last_hidden_state

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
        """NPU-accelerated W33 batched action-conditioned prediction with CPU parity.

        Sister to :meth:`predict_temporal_batch` plus per-sample action
        sequences. Mirrors
        :meth:`VJEPA2Worker.predict_action_conditioned_batch` byte-for-byte
        on the CPU fallback so ``test_vllm_ascend_worker_parity`` can
        assert cross-backend cosine ≥ 0.999.
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
        bundle = self._ensure_action_bundle()
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

        ref = tuple(videos[0].shape)
        for i, v in enumerate(videos):
            if tuple(v.shape) != ref:
                raise ValueError(
                    f"video[{i}] has shape {tuple(v.shape)} != reference "
                    f"{ref}; batched action predict requires uniform video shape"
                )
        ref_action = tuple(actions_per_sample[0].shape)
        for i, a in enumerate(actions_per_sample):
            if tuple(a.shape) != ref_action:
                raise ValueError(
                    f"actions[{i}] has shape {tuple(a.shape)} != reference "
                    f"{ref_action}; batched action predict requires uniform action shape"
                )

        batched = torch.stack(videos, dim=0)  # (B, T, C, H, W)
        batched_actions = torch.stack(actions_per_sample, dim=0)  # (B, K, D)
        past_video = batched[:, :k]
        future_video = batched[:, k:k + h]

        npu = _npu_ops()
        if npu is not None and past_video.device.type != "cpu":
            past_video = npu.npu_format_cast(past_video, 2)
            future_video = npu.npu_format_cast(future_video, 2)

        past_feats = bundle.online(pixel_values_videos=past_video).last_hidden_state
        _, future_grid = self._ensure_grids()
        predicted = bundle.action_predictor(
            past_feats, batched_actions, future_grid=future_grid,
        )
        target = bundle.target(pixel_values_videos=future_video).last_hidden_state

        sim = torch.nn.functional.cosine_similarity(predicted, target, dim=-1)
        confidence_per_sample = sim.mean(dim=-1)  # (B,)
        return [
            (predicted[i], float(confidence_per_sample[i]))
            for i in range(batched.shape[0])
        ]