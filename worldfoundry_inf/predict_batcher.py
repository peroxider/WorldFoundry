"""Per-request predict batching (W25).

Wraps a ``VJEPA2Worker`` (or any object exposing ``predict_masked_batch``
and optionally ``predict_temporal_batch``) so multiple concurrent HTTP
``/v1/world/predict`` requests share a single GPU forward pass. Requests
are accumulated up to ``max_batch`` or ``max_latency_ms``, whichever
comes first, then dispatched in one call.

Masked-mode requests are always batchable (uniform per-batch shape).
Temporal-mode requests must share ``past_frames`` / ``future_frames``
(matching the worker's configured grid); the batcher routes mismatched
temporal requests through ``worker.predict_temporal(...)`` one at a time
rather than raising.

Action-mode requests (W33) follow the same uniformity contract as
temporal mode (``past_frames`` / ``future_frames`` / ``actions.shape``
must match across the batch). Uniform action batches are dispatched via
``worker.predict_action_conditioned_batch(...)`` when available;
otherwise they fall through to ``worker.predict_action_conditioned(...)``
one at a time.
"""
from __future__ import annotations

import threading
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class PredictRequest:
    """One in-flight predict request (decoded tensor)."""

    video: torch.Tensor  # (T, C, H, W)
    mode: str  # "masked" | "temporal" | "action"
    mask_ratio: float = 0.0  # masked mode
    past_frames: int = 0  # temporal / action mode
    future_frames: int = 0  # temporal / action mode
    actions: torch.Tensor | None = None  # (past_frames, action_dim) for action mode


class PredictBatcher:
    """Thread-safe per-request predict batcher.

    Args:
        worker: object exposing ``predict_masked_batch`` /
            ``predict_temporal_batch`` / ``predict_action_conditioned_batch``
            (a ``VJEPA2Worker``).
        max_batch: flush when this many requests have accumulated.
        max_latency_ms: flush when this much wall time has passed since
            the first request in the current batch arrived.
    """

    def __init__(
        self,
        worker: Any,
        max_batch: int = 8,
        max_latency_ms: float = 50.0,
    ) -> None:
        self.worker = worker
        self.max_batch = max_batch
        self.max_latency_ms = max_latency_ms
        self._lock = threading.Lock()
        self._queue: list[tuple[PredictRequest, Future]] = []

    def submit(self, request: PredictRequest) -> Future:
        """Enqueue a request; flush now if full, else schedule a deadline flush."""
        fut: Future = Future()
        with self._lock:
            self._queue.append((request, fut))
            should_flush = len(self._queue) >= self.max_batch
        if should_flush:
            self._flush()
        else:
            threading.Timer(self.max_latency_ms / 1000.0, self._flush).start()
        return fut

    def _flush(self) -> None:
        with self._lock:
            if not self._queue:
                return
            batch = self._queue[:]
            self._queue.clear()

        # Group by mode + uniform temporal/action params (when applicable).
        masked_idx: list[int] = []
        temporal_uniform_idx: list[int] = []
        temporal_singleton_idx: list[int] = []
        action_uniform_idx: list[int] = []
        action_singleton_idx: list[int] = []
        for i, (req, _) in enumerate(batch):
            if req.mode == "masked":
                masked_idx.append(i)
            elif req.mode == "temporal":
                if temporal_uniform_idx and (
                    batch[temporal_uniform_idx[0]][0].past_frames != req.past_frames
                    or batch[temporal_uniform_idx[0]][0].future_frames != req.future_frames
                ):
                    temporal_singleton_idx.append(i)
                else:
                    temporal_uniform_idx.append(i)
            elif req.mode == "action":
                if action_uniform_idx and (
                    batch[action_uniform_idx[0]][0].past_frames != req.past_frames
                    or batch[action_uniform_idx[0]][0].future_frames != req.future_frames
                    or batch[action_uniform_idx[0]][0].actions.shape != req.actions.shape
                ):
                    action_singleton_idx.append(i)
                else:
                    action_uniform_idx.append(i)
            else:
                # Unknown mode -> route to singleton fallback so it gets
                # a clear error rather than silently dropped.
                temporal_singleton_idx.append(i)

        results: list[Any] = [None] * len(batch)

        # 1) Masked batched dispatch.
        if masked_idx:
            videos = [batch[i][0].video for i in masked_idx]
            ratios = [batch[i][0].mask_ratio for i in masked_idx]
            try:
                outputs = self.worker.predict_masked_batch(videos, ratios)
                for k, i in enumerate(masked_idx):
                    results[i] = outputs[k]
            except Exception as exc:
                for i in masked_idx:
                    if not batch[i][1].done():
                        batch[i][1].set_exception(exc)

        # 2) Temporal uniform batched dispatch.
        if temporal_uniform_idx:
            first = batch[temporal_uniform_idx[0]][0]
            uniform_k = first.past_frames
            uniform_h = first.future_frames
            uniform_videos = [batch[i][0].video for i in temporal_uniform_idx]
            ks = [uniform_k] * len(temporal_uniform_idx)
            hs = [uniform_h] * len(temporal_uniform_idx)
            try:
                outputs = self.worker.predict_temporal_batch(uniform_videos, ks, hs)
                for k, i in enumerate(temporal_uniform_idx):
                    results[i] = outputs[k]
            except Exception as exc:
                for i in temporal_uniform_idx:
                    if not batch[i][1].done():
                        batch[i][1].set_exception(exc)

        # 3) Temporal singleton fallback (mismatched K/H or unknown mode).
        for i in temporal_singleton_idx:
            req, fut = batch[i]
            if fut.done():
                continue
            try:
                if req.mode == "temporal" and hasattr(self.worker, "predict_temporal"):
                    out = self.worker.predict_temporal(
                        req.video, req.past_frames, req.future_frames,
                    )
                else:
                    fut.set_exception(
                        RuntimeError(
                            f"unsupported mode {req.mode!r} or worker lacks "
                            f"predict_temporal()"
                        )
                    )
                    continue
                results[i] = out
            except Exception as exc:
                fut.set_exception(exc)

        # 4) Action uniform batched dispatch (W33).
        if action_uniform_idx and hasattr(self.worker, "predict_action_conditioned_batch"):
            uniform_videos = [batch[i][0].video for i in action_uniform_idx]
            uniform_actions = [batch[i][0].actions for i in action_uniform_idx]
            ks = [batch[action_uniform_idx[0]][0].past_frames] * len(action_uniform_idx)
            hs = [batch[action_uniform_idx[0]][0].future_frames] * len(action_uniform_idx)
            try:
                outputs = self.worker.predict_action_conditioned_batch(
                    uniform_videos, uniform_actions, ks, hs,
                )
                for k, i in enumerate(action_uniform_idx):
                    results[i] = outputs[k]
            except Exception as exc:
                for i in action_uniform_idx:
                    if not batch[i][1].done():
                        batch[i][1].set_exception(exc)

        # 5) Action singleton fallback (mismatched K/H/action shape, or worker
        # lacks predict_action_conditioned_batch).
        for i in action_singleton_idx:
            req, fut = batch[i]
            if fut.done():
                continue
            try:
                if req.mode == "action" and hasattr(self.worker, "predict_action_conditioned"):
                    out = self.worker.predict_action_conditioned(
                        req.video, req.actions, req.past_frames, req.future_frames,
                    )
                else:
                    fut.set_exception(
                        RuntimeError(
                            f"unsupported mode {req.mode!r} or worker lacks "
                            f"predict_action_conditioned()"
                        )
                    )
                    continue
                results[i] = out
            except Exception as exc:
                fut.set_exception(exc)

        # 6) Hand results to futures.
        for i, (_, fut) in enumerate(batch):
            if fut.done() or results[i] is None:
                continue
            fut.set_result(results[i])
