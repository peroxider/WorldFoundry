"""Video batcher for inference requests.

Aggregates single-video requests up to `max_batch` or `max_latency_ms`,
whichever comes first. Used as the bridge between the HTTP layer and the
VJEPA2WorkerPool.
"""
from __future__ import annotations

import threading
from concurrent.futures import Future
from typing import Callable, Optional

import numpy as np


class VideoBatcher:
    """Thread-safe batcher; one instance per process."""

    def __init__(
        self,
        max_batch: int = 8,
        max_latency_ms: float = 50.0,
        process_fn: Optional[Callable[[list[np.ndarray]], object]] = None,
    ) -> None:
        self.max_batch = max_batch
        self.max_latency_ms = max_latency_ms
        self.process_fn = process_fn or (lambda v: v)
        self._lock = threading.Lock()
        self._queue: list[tuple[np.ndarray, str, Future]] = []

    def submit(self, video: np.ndarray, adapter_name: str) -> Future:
        fut: Future = Future()
        with self._lock:
            self._queue.append((video, adapter_name, fut))
            should_flush = len(self._queue) >= self.max_batch
        if should_flush:
            self._flush()
        else:
            # Schedule a latency-deadline flush
            threading.Timer(self.max_latency_ms / 1000.0, self._flush).start()
        return fut

    def _flush(self) -> None:
        with self._lock:
            if not self._queue:
                return
            batch = self._queue[:]
            self._queue.clear()
        videos = [v for v, _, _ in batch]
        try:
            result = self.process_fn(videos)
        except Exception as exc:
            for _, _, fut in batch:
                if not fut.done():
                    fut.set_exception(exc)
            return
        if not isinstance(result, list):
            result = [result] * len(batch)
        for (_, _, fut), r in zip(batch, result):
            if not fut.done():
                fut.set_result(r)