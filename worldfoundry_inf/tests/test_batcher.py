import time

import numpy as np

from worldfoundry_inf.batcher import VideoBatcher


def test_batcher_submits_and_returns_result():
    batcher = VideoBatcher(max_batch=4, max_latency_ms=20, process_fn=lambda videos: sum(v.shape[0] for v in videos))
    fut = batcher.submit(np.zeros((4, 224, 224, 3), dtype=np.uint8), adapter_name="default")
    result = fut.result(timeout=2.0)
    assert result == 4


def test_batcher_flushes_on_latency_deadline():
    batcher = VideoBatcher(max_batch=100, max_latency_ms=10, process_fn=lambda videos: len(videos))
    fut = batcher.submit(np.zeros((2, 224, 224, 3), dtype=np.uint8), adapter_name="default")
    result = fut.result(timeout=2.0)
    assert result == 1