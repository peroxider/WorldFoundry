"""NPU environment setup and version guard.

Per Global Constraint G2 (spec §6.2):
  torch_npu 2.9.0 + vllm-ascend 0.18.0 + CANN 8.5.1 + transformers 4.55.0
"""
from __future__ import annotations

import importlib.metadata
import logging
import os

from worldfoundry_common.exceptions import VersionMismatchError

logger = logging.getLogger(__name__)

LOCKED_VERSIONS: dict[str, str] = {
    "torch": "2.9.0",
    "torch_npu": "2.9.0",
    "transformers": "4.55.0",
    "peft": "0.18.0",
    "accelerate": "1.10.0",
}

MISSING_VERSION: str = "0.0.0+missing"


def _get_version(pkg: str) -> str:
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return MISSING_VERSION


def check_versions(strict: bool = False) -> list[tuple[str, str, str]]:
    """Verify installed package versions against the locked matrix.

    Returns list of (pkg, expected, actual). If strict=True and any mismatch
    is found, raises VersionMismatchError. When the package itself is absent
    (PackageNotFoundError) the error message distinguishes "not installed"
    from a real version mismatch.
    """
    mismatches: list[tuple[str, str, str]] = []
    for pkg, expected in LOCKED_VERSIONS.items():
        actual = _get_version(pkg)
        if actual != expected:
            mismatches.append((pkg, expected, actual))
    if strict and mismatches:
        absent = [m for m in mismatches if m[2] == MISSING_VERSION]
        mismatched = [m for m in mismatches if m[2] != MISSING_VERSION]
        parts: list[str] = []
        if absent:
            not_installed = ", ".join(pkg for pkg, _, _ in absent)
            parts.append(f"package(s) not installed: {not_installed}")
        if mismatched:
            details = ", ".join(f"{pkg} (expected {expected}, got {actual})" for pkg, expected, actual in mismatched)
            parts.append(f"version mismatch(es): {details}")
        raise VersionMismatchError(
            "Locked version mismatch: " + "; ".join(parts)
            + ". See spec §6.2 and pyproject.toml for the matrix."
        )
    return mismatches


def is_npu_available() -> bool:
    """Return True iff torch.npu is importable and at least one device is reachable.

    A genuine ``ImportError`` from ``torch`` itself propagates so that a broken
    install is surfaced (G8 — no silent substitution). Only the absence of
    ``torch_npu`` is treated as "no NPU".
    """
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return False
    import torch
    return torch.npu.is_available() if hasattr(torch, "npu") else False


def npu_device_count() -> int:
    """Number of NPU devices available (0 on CPU-only machines)."""
    if not is_npu_available():
        return 0
    import torch
    return torch.npu.device_count()


def init_distributed(backend: str = "hccl") -> None:
    """Initialize torch.distributed.

    On NPU the caller-requested ``backend`` is honored (default ``hccl``).
    On CPU machines ``gloo`` is substituted and a warning is logged so the
    substitution is visible in logs (G8 — no silent defaults).
    """
    import torch
    import torch.distributed as dist
    if is_npu_available():
        dist.init_process_group(backend=backend)
    else:
        logger.warning(
            "NPU not available; substituting requested backend %r with %r for torch.distributed.init.",
            backend,
            "gloo",
        )
        dist.init_process_group(backend="gloo")


# --------------------------------------------------------------------------- #
# NPU runtime env (baseline: verl + vllm-ascend on torch_npu 2.9.0)
# --------------------------------------------------------------------------- #
DEFAULT_NPU_ENV: dict[str, str] = {
    # Visible NPU devices (CUDA_VISIBLE_DEVICES' NPU equivalent).
    "ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
    # HCCL port ranges — required for multi-process / multi-card NPU collectives
    # (verl + Ray on a single card runs several processes that need distinct ports).
    "HCCL_HOST_SOCKET_PORT_RANGE": "60000-60050",
    "HCCL_NPU_SOCKET_PORT_RANGE": "61000-61050",
    # Ray does not understand ASCEND_RT_VISIBLE_DEVICES by default; this flag
    # stops Ray from trying to manage the NPU device visibility (verl/SGLang docs).
    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
    # torch_npu memory-allocation tuning (SGLang/verl NPU docs).
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
}


def npu_env_vars(visible_devices: str | None = None) -> dict[str, str]:
    """Return the NPU runtime env vars the baseline requires.

    Values are computed from ``os.environ`` when already set (caller wins),
    else the documented defaults from the verl / vllm-ascend / sglang-ascend
    community guides. ``visible_devices`` overrides
    ``ASCEND_RT_VISIBLE_DEVICES`` for a specific device set.
    """
    env: dict[str, str] = dict(DEFAULT_NPU_ENV)
    if visible_devices is not None:
        env["ASCEND_RT_VISIBLE_DEVICES"] = visible_devices
    # Preserve any already-exported values (explicit user config wins).
    for key in env:
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def setup_npu_env(visible_devices: str | None = None) -> None:
    """Apply the baseline NPU runtime env vars to the current process.

    Idempotent; existing ``os.environ`` values are preserved. Call this before
    spawning Ray / initializing torch.distributed on an NPU host.
    """
    for key, value in npu_env_vars(visible_devices=visible_devices).items():
        os.environ.setdefault(key, value)