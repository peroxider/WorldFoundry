import importlib

import pytest


def test_fsdp_module_imports():
    mod = importlib.import_module("worldforge_train.parallel.fsdp")
    assert hasattr(mod, "setup_fsdp")
    assert hasattr(mod, "cleanup_fsdp")
