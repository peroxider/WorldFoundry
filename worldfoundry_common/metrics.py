"""Prometheus metrics shared by train/inf/rl subsystems."""
from __future__ import annotations

import numpy as np
from prometheus_client import Counter, Histogram

WORLDFORGE_RPS = Counter(
    "worldforge_rps",
    "WorldFoundry inference requests per second",
    ["endpoint"],
)
WORLDFORGE_LATENCY_MS = Histogram(
    "worldforge_latency_ms",
    "WorldFoundry inference latency in ms",
    ["endpoint"],
    buckets=(10, 25, 50, 100, 200, 400, 800, 1600, 3200),
)
WORLDFORGE_OM_LOAD_ERRORS = Counter(
    "worldforge_om_load_errors",
    "WorldFoundry OM load failures",
    ["model_path"],
)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray | float:
    """Cosine similarity between vectors. 1D inputs return float; 2D return per-row."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.ndim == 1 and b.ndim == 1:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)
    if a.ndim == 2 and b.ndim == 2:
        denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        denom = np.where(denom == 0, 1.0, denom)
        return (np.sum(a * b, axis=1) / denom).astype(np.float32)
    raise ValueError(f"Unsupported shapes: {a.shape}, {b.shape}")