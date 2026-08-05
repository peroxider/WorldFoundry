"""CPU tests for worldfoundry_inf._vllm_ascend_plugin (no vllm, no NPU).

Validates the registration entry-point functions:
- register() returns the vllm-ascend platform class path string
- register_model() is a no-op + log when vllm is not installed (graceful)
- register_model() calls ModelRegistry.register_model when vllm IS installed
  (we monkeypatch the lazy import to inject a fake ModelRegistry)
"""
from __future__ import annotations

import logging

import pytest

from worldfoundry_inf import _vllm_ascend_plugin as plugin


def test_register_returns_npu_platform_path():
    """register() returns vllm_ascend.platform.NPUPlatform — the documented contract."""
    assert plugin.register() == "vllm_ascend.platform.NPUPlatform"


def test_register_model_no_op_when_vllm_missing(caplog):
    """When vllm is not installed, register_model() logs + returns, no exception."""
    with caplog.at_level(logging.INFO, logger="worldfoundry_inf._vllm_ascend_plugin"):
        plugin.register_model()  # must not raise
    assert "vllm not installed" in caplog.text


def test_register_model_calls_model_registry_register(monkeypatch):
    """When vllm IS importable, register_model() dispatches to ModelRegistry.register_model."""
    seen = {}

    class _FakeRegistry:
        @staticmethod
        def register_model(name, worker_cls):
            seen["name"] = name
            seen["worker_cls"] = worker_cls

    # Monkeypatch the lazy import inside _vllm_ascend_plugin to return our fake.
    monkeypatch.setattr(plugin, "_vllm_model_registry", lambda: _FakeRegistry)
    plugin.register_model()
    assert seen["name"] == "VJEPA2Model"
    assert seen["worker_cls"].__name__ == "VJEPA2NPUWorker"


def test_register_model_swallows_registry_exceptions(monkeypatch, caplog):
    """If ModelRegistry.register_model raises, register_model() logs a warning rather than crashing."""
    class _BoomRegistry:
        @staticmethod
        def register_model(name, worker_cls):
            raise RuntimeError("simulated registry failure")

    monkeypatch.setattr(plugin, "_vllm_model_registry", lambda: _BoomRegistry)
    with caplog.at_level(logging.WARNING, logger="worldfoundry_inf._vllm_ascend_plugin"):
        plugin.register_model()  # must not raise
    assert "registration failed" in caplog.text
    assert "simulated registry failure" in caplog.text


def test_plugin_module_public_api():
    """Module exposes register and register_model at module level."""
    assert callable(plugin.register)
    assert callable(plugin.register_model)