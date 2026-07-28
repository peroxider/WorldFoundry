from pathlib import Path

import pytest

from worldforge_common.model_registry import MODEL_REGISTRY, resolve_model_path


def test_vjepa2_registered():
    assert "vjepa2-vitl-1.2b" in MODEL_REGISTRY
    entry = MODEL_REGISTRY["vjepa2-vitl-1.2b"]
    assert entry.hf_repo == "facebook/vjepa2-vitl-1.2b"
    assert isinstance(entry.default_adapter_dir, Path)


def test_resolve_unknown_raises():
    with pytest.raises(KeyError):
        resolve_model_path("nonexistent")
