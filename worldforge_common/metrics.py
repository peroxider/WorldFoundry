"""Prometheus metrics shared by train/inf/rl subsystems."""
from __future__ import annotations

from prometheus_client import Counter, Histogram

WORLDFORGE_RPS = Counter(
    "worldforge_rps",
    "WorldForge inference requests per second",
    ["endpoint"],
)
WORLDFORGE_LATENCY_MS = Histogram(
    "worldforge_latency_ms",
    "WorldForge inference latency in ms",
    ["endpoint"],
    buckets=(10, 25, 50, 100, 200, 400, 800, 1600, 3200),
)
WORLDFORGE_OM_LOAD_ERRORS = Counter(
    "worldforge_om_load_errors",
    "WorldForge OM load failures",
    ["model_path"],
)