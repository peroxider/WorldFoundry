"""C-T9: pytest configuration.

Auto-skips tests tagged with the ``host_skip`` marker when running on a
headless host (no Atlas NPU container). On Atlas the tests run normally.

Atlas detection:
  * ``/proc/driver/npu`` exists (the Huawei Ascend kernel module is loaded), OR
  * ``WORLDFOUNDRY_ATLAS=1`` is set explicitly (escape hatch for CI variants).
"""
from __future__ import annotations

import os

import pytest


def _on_atlas() -> bool:
    if os.environ.get("WORLDFOUNDRY_ATLAS") == "1":
        return True
    return os.path.exists("/proc/driver/npu")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-skip ``host_skip``-marked tests when not running in Atlas container."""
    if _on_atlas():
        return  # Atlas: run everything as normal.
    skip_marker = pytest.mark.skip(
        reason="C-T9 host_skip: requires Atlas NPU container (dm_control EGL / "
        "GLFW context); set WORLDFOUNDRY_ATLAS=1 to override."
    )
    for item in items:
        if "host_skip" in item.keywords:
            item.add_marker(skip_marker)