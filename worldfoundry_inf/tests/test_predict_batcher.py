"""Tests for the W25 PredictBatcher.

Covers: submit + flush on max_batch, flush on latency deadline, mixed
masked/temporal dispatch, exception propagation to futures, uniform vs
non-uniform temporal (singleton fallback), and routing through the
``VJEPA2Worker.predict_masked_batch`` / ``predict_temporal_batch`` methods.
"""
from __future__ import annotations

import threading
import time

import pytest
import torch

from worldfoundry_inf.predict_batcher import PredictBatcher, PredictRequest


class _StubWorker:
    """Stub worker that records batched calls and returns deterministic output."""

    def __init__(self, raise_on_call: Exception | None = None) -> None:
        self.masked_calls: list[tuple[list[torch.Tensor], list[float]]] = []
        self.temporal_calls: list[tuple[list[torch.Tensor], list[int], list[int]]] = []
        self.temporal_singleton_calls: list[tuple[torch.Tensor, int, int]] = []
        self.action_calls: list[tuple[
            list[torch.Tensor], list[torch.Tensor], list[int], list[int],
        ]] = []
        self.action_singleton_calls: list[
            tuple[torch.Tensor, torch.Tensor, int, int]
        ] = []
        self.raise_on_call = raise_on_call

    def predict_masked_batch(
        self, videos: list[torch.Tensor], mask_ratios: list[float],
    ) -> list[torch.Tensor]:
        if self.raise_on_call is not None:
            raise self.raise_on_call
        self.masked_calls.append((videos, mask_ratios))
        return [torch.full((1, 196, 1024), float(i + 1)) for i in range(len(videos))]

    def predict_temporal_batch(
        self, videos: list[torch.Tensor],
        past_frames_list: list[int], future_frames_list: list[int],
    ) -> list[tuple[torch.Tensor, float]]:
        self.temporal_calls.append((videos, past_frames_list, future_frames_list))
        n = len(videos)
        return [
            (torch.zeros(1, 196, 1024), 0.5 + 0.1 * i)
            for i in range(n)
        ]

    def predict_temporal(
        self, video: torch.Tensor, past_frames: int, future_frames: int,
    ) -> tuple[torch.Tensor, float]:
        self.temporal_singleton_calls.append((video, past_frames, future_frames))
        return torch.zeros(1, 196, 1024), 0.9

    def predict_action_conditioned_batch(
        self,
        videos: list[torch.Tensor],
        actions_per_sample: list[torch.Tensor],
        past_frames_list: list[int],
        future_frames_list: list[int],
    ) -> list[tuple[torch.Tensor, float]]:
        self.action_calls.append(
            (videos, actions_per_sample, past_frames_list, future_frames_list),
        )
        n = len(videos)
        return [
            (torch.zeros(1, 196, 1024), 0.7 + 0.1 * i)
            for i in range(n)
        ]

    def predict_action_conditioned(
        self,
        video: torch.Tensor,
        actions: torch.Tensor,
        past_frames: int,
        future_frames: int,
    ) -> tuple[torch.Tensor, float]:
        self.action_singleton_calls.append((video, actions, past_frames, future_frames))
        return torch.zeros(1, 196, 1024), 0.95


def _video(t: int = 4) -> torch.Tensor:
    return torch.zeros(t, 3, 64, 64)


# ---------------------------------------------------------------------------
# Basic shape + immediate-flush on max_batch
# ---------------------------------------------------------------------------


def test_submit_flushes_immediately_when_max_batch_reached():
    worker = _StubWorker()
    batcher = PredictBatcher(worker, max_batch=3, max_latency_ms=10000.0)

    f0 = batcher.submit(PredictRequest(video=_video(), mode="masked", mask_ratio=0.5))
    f1 = batcher.submit(PredictRequest(video=_video(), mode="masked", mask_ratio=0.3))
    f2 = batcher.submit(PredictRequest(video=_video(), mode="masked", mask_ratio=0.1))

    assert f0.result(timeout=2.0).shape == (1, 196, 1024)
    assert f1.result(timeout=2.0).shape == (1, 196, 1024)
    assert f2.result(timeout=2.0).shape == (1, 196, 1024)
    # Exactly one batched call with 3 videos.
    assert len(worker.masked_calls) == 1
    assert len(worker.masked_calls[0][0]) == 3
    assert worker.masked_calls[0][1] == [0.5, 0.3, 0.1]


# ---------------------------------------------------------------------------
# Latency-deadline flush
# ---------------------------------------------------------------------------


def test_submit_flushes_on_latency_deadline_when_below_max_batch():
    worker = _StubWorker()
    batcher = PredictBatcher(worker, max_batch=100, max_latency_ms=15.0)

    fut = batcher.submit(PredictRequest(video=_video(), mode="masked", mask_ratio=0.5))
    t0 = time.perf_counter()
    fut.result(timeout=2.0)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    # Latency deadline is 15ms; allow generous slack for slow CI.
    assert elapsed_ms < 1000.0, f"latency flush took {elapsed_ms:.1f}ms (expected <1s)"
    assert len(worker.masked_calls) == 1
    assert len(worker.masked_calls[0][0]) == 1


# ---------------------------------------------------------------------------
# Mixed-mode batching
# ---------------------------------------------------------------------------


def test_mixed_masked_and_temporal_routes_through_separate_batch_calls():
    worker = _StubWorker()
    batcher = PredictBatcher(worker, max_batch=100, max_latency_ms=20.0)

    fm = batcher.submit(PredictRequest(video=_video(), mode="masked", mask_ratio=0.7))
    ft = batcher.submit(
        PredictRequest(video=_video(), mode="temporal", past_frames=2, future_frames=2),
    )

    fm.result(timeout=2.0)
    ft.result(timeout=2.0)

    assert len(worker.masked_calls) == 1
    assert len(worker.temporal_calls) == 1
    assert worker.masked_calls[0][1] == [0.7]
    assert worker.temporal_calls[0][1] == [2]
    assert worker.temporal_calls[0][2] == [2]


def test_uniform_temporal_groups_into_single_batched_call():
    worker = _StubWorker()
    batcher = PredictBatcher(worker, max_batch=100, max_latency_ms=20.0)

    f0 = batcher.submit(PredictRequest(video=_video(), mode="temporal", past_frames=1, future_frames=1))
    f1 = batcher.submit(PredictRequest(video=_video(), mode="temporal", past_frames=1, future_frames=1))
    f2 = batcher.submit(PredictRequest(video=_video(), mode="temporal", past_frames=1, future_frames=1))

    f0.result(timeout=2.0)
    f1.result(timeout=2.0)
    f2.result(timeout=2.0)

    # All 3 share (K=1, H=1) -> single batched call.
    assert len(worker.temporal_calls) == 1
    assert len(worker.temporal_calls[0][0]) == 3
    assert len(worker.temporal_singleton_calls) == 0


def test_non_uniform_temporal_falls_back_to_singleton_calls():
    """First batch: (K=1, H=1). Then (K=2, H=2) -> singleton fallback."""
    worker = _StubWorker()
    batcher = PredictBatcher(worker, max_batch=100, max_latency_ms=20.0)

    f_uni = batcher.submit(PredictRequest(video=_video(), mode="temporal", past_frames=1, future_frames=1))
    f_diff = batcher.submit(PredictRequest(video=_video(), mode="temporal", past_frames=2, future_frames=2))

    f_uni.result(timeout=2.0)
    f_diff.result(timeout=2.0)

    # First request was uniform (alone), second mismatched -> singleton.
    assert len(worker.temporal_calls) == 1
    assert len(worker.temporal_calls[0][0]) == 1
    assert worker.temporal_calls[0][1] == [1]
    assert len(worker.temporal_singleton_calls) == 1
    assert worker.temporal_singleton_calls[0][1] == 2
    assert worker.temporal_singleton_calls[0][2] == 2


# ---------------------------------------------------------------------------
# Exception propagation
# ---------------------------------------------------------------------------


def test_worker_exception_propagates_to_futures():
    worker = _StubWorker(raise_on_call=RuntimeError("boom"))
    batcher = PredictBatcher(worker, max_batch=2, max_latency_ms=10000.0)

    f0 = batcher.submit(PredictRequest(video=_video(), mode="masked", mask_ratio=0.5))
    f1 = batcher.submit(PredictRequest(video=_video(), mode="masked", mask_ratio=0.5))

    for fut in (f0, f1):
        with pytest.raises(RuntimeError, match="boom"):
            fut.result(timeout=2.0)


# ---------------------------------------------------------------------------
# Unknown mode
# ---------------------------------------------------------------------------


def test_unknown_mode_future_receives_error():
    worker = _StubWorker()
    batcher = PredictBatcher(worker, max_batch=100, max_latency_ms=15.0)

    fut = batcher.submit(PredictRequest(video=_video(), mode="bogus"))
    with pytest.raises(RuntimeError, match="unsupported mode"):
        fut.result(timeout=2.0)


# ---------------------------------------------------------------------------
# Thread safety smoke
# ---------------------------------------------------------------------------


def test_concurrent_submits_all_get_responses():
    """100 submits across 4 threads -> all futures resolve with the right shape."""
    worker = _StubWorker()
    batcher = PredictBatcher(worker, max_batch=4, max_latency_ms=20.0)

    futures = []
    lock = threading.Lock()

    def submit_n(n: int) -> None:
        for _ in range(n):
            with lock:
                futures.append(
                    batcher.submit(PredictRequest(video=_video(), mode="masked", mask_ratio=0.5)),
                )

    threads = [threading.Thread(target=submit_n, args=(25,)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for fut in futures:
        out = fut.result(timeout=5.0)
        assert out.shape == (1, 196, 1024)
    assert len(worker.masked_calls) >= 1


# ---------------------------------------------------------------------------
# W33: action-mode (predict_action_conditioned_batch / predict_action_conditioned)
# ---------------------------------------------------------------------------


def _actions(past_frames: int = 2, action_dim: int = 7) -> torch.Tensor:
    return torch.zeros(past_frames, action_dim)


def test_uniform_action_mode_groups_into_single_batched_call():
    """W33: 3 action requests with identical K/H/action shape -> 1 batched call."""
    worker = _StubWorker()
    batcher = PredictBatcher(worker, max_batch=100, max_latency_ms=20.0)

    f0 = batcher.submit(
        PredictRequest(
            video=_video(), mode="action",
            past_frames=2, future_frames=2, actions=_actions(2, 7),
        ),
    )
    f1 = batcher.submit(
        PredictRequest(
            video=_video(), mode="action",
            past_frames=2, future_frames=2, actions=_actions(2, 7),
        ),
    )
    f2 = batcher.submit(
        PredictRequest(
            video=_video(), mode="action",
            past_frames=2, future_frames=2, actions=_actions(2, 7),
        ),
    )

    f0.result(timeout=2.0)
    f1.result(timeout=2.0)
    f2.result(timeout=2.0)

    assert len(worker.action_calls) == 1
    assert len(worker.action_calls[0][0]) == 3  # 3 videos batched
    assert len(worker.action_calls[0][1]) == 3  # 3 action tensors batched
    assert worker.action_calls[0][2] == [2, 2, 2]  # past_frames
    assert worker.action_calls[0][3] == [2, 2, 2]  # future_frames
    assert len(worker.action_singleton_calls) == 0


def test_action_mode_shape_mismatch_falls_back_to_singleton():
    """W33: second request has a different action shape -> singleton fallback."""
    worker = _StubWorker()
    batcher = PredictBatcher(worker, max_batch=100, max_latency_ms=20.0)

    f_uni = batcher.submit(
        PredictRequest(
            video=_video(), mode="action",
            past_frames=2, future_frames=2, actions=_actions(2, 7),
        ),
    )
    f_diff = batcher.submit(
        PredictRequest(
            video=_video(), mode="action",
            past_frames=2, future_frames=2, actions=_actions(2, 9),  # different action_dim
        ),
    )

    f_uni.result(timeout=2.0)
    f_diff.result(timeout=2.0)

    # First request batched alone; second singleton.
    assert len(worker.action_calls) == 1
    assert len(worker.action_calls[0][0]) == 1
    assert len(worker.action_singleton_calls) == 1
    assert worker.action_singleton_calls[0][3] == 2  # future_frames preserved


def test_action_mode_past_frames_mismatch_falls_back_to_singleton():
    """W33: second request has different past_frames -> singleton fallback."""
    worker = _StubWorker()
    batcher = PredictBatcher(worker, max_batch=100, max_latency_ms=20.0)

    f_uni = batcher.submit(
        PredictRequest(
            video=_video(), mode="action",
            past_frames=2, future_frames=2, actions=_actions(2, 7),
        ),
    )
    f_diff = batcher.submit(
        PredictRequest(
            video=_video(), mode="action",
            past_frames=3, future_frames=2, actions=_actions(3, 7),
        ),
    )

    f_uni.result(timeout=2.0)
    f_diff.result(timeout=2.0)

    assert len(worker.action_calls) == 1
    assert len(worker.action_calls[0][0]) == 1
    assert len(worker.action_singleton_calls) == 1


def test_action_mode_returns_action_bundle_results():
    """W33: per-sample (features, confidence) from the action batcher."""
    worker = _StubWorker()
    batcher = PredictBatcher(worker, max_batch=10, max_latency_ms=15.0)

    fut = batcher.submit(
        PredictRequest(
            video=_video(), mode="action",
            past_frames=2, future_frames=2, actions=_actions(2, 7),
        ),
    )
    features, confidence = fut.result(timeout=2.0)
    assert features.shape == (1, 196, 1024)
    assert isinstance(confidence, float)
    assert 0.0 <= confidence <= 1.0
