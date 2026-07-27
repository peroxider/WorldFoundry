"""NPU environment setup and version guard.

Per Global Constraint G2 (spec §6.2):
  torch_npu 2.9.0 + vllm-ascend 0.18.0 + CANN 8.5.1 + transformers 4.55.0
"""
from __future__ import annotations

import importlib.metadata
from typing import Tuple

from worldforge_common.exceptions import VersionMismatchError

LOCKED_VERSIONS: dict[str, str] = {
    "torch": "2.9.0",
    "torch_npu": "2.9.0",
    "transformers": "4.55.0",
    "peft": "0.18.0",
    "accelerate": "1.10.0",
}


def _get_version(pkg: str) -> str:
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0+missing"


def check_versions(strict: bool = False) -> list[tuple[str, str, str]]:
    """Verify installed package versions against the locked matrix.

    Returns list of (pkg, expected, actual). If strict=True and any mismatch
    is found, raises VersionMismatchError.
    """
    mismatches: list[tuple[str, str, str]] = []
    for pkg, expected in LOCKED_VERSIONS.items():
        actual = _get_version(pkg)
        if actual != expected:
            mismatches.append((pkg, expected, actual))
    if strict and mismatches:
        raise VersionMismatchError(
            f"Locked version mismatch: {mismatches}. "
            f"See spec §6.2 and pyproject.toml for the matrix."
        )
    return mismatches


def is_npu_available() -> bool:
    """Return True iff torch.npu is importable and at least one device is reachable."""
    try:
        import torch_npu  # noqa: F401
        import torch
        return torch.npu.is_available() if hasattr(torch, "npu") else False
    except ImportError:
        return False


def npu_device_count() -> int:
    """Number of NPU devices available (0 on CPU-only machines)."""
    if not is_npu_available():
        return 0
    import torch
    return torch.npu.device_count()


def init_distributed(backend: str = "hccl") -> None:
    """Initialize torch.distributed with HCCL on NPU, Gloo on CPU fallback."""
    import torch
    import torch.distributed as dist
    if is_npu_available():
        dist.init_process_group(backend="hccl")
    else:
        dist.init_process_group(backend="gloo")
