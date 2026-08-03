"""Checkpoint helpers for FSDP-wrapped models (CPU-safe with mocks).

True FSDP requires a GPU/NPU accelerator, so these tests stub the FSDP type
check and verify that checkpoint.py uses the FSDP state-dict APIs.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn
from torch.optim import AdamW

from worldforge_train.checkpoint import (
    LoraSignature, _is_fsdp_module, load_checkpoint, save_checkpoint,
)


def _make_model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 1))


def test_is_fsdp_module_false_for_plain_module():
    assert _is_fsdp_module(nn.Linear(4, 4)) is False


def test_trainable_state_dict_uses_full_state_dict_for_fsdp(tmp_path, monkeypatch):
    """When the model appears FSDP-wrapped, _trainable_state_dict should gather
    the full state dict via FSDP.state_dict_type(..., FULL_STATE_DICT)."""
    pytest.importorskip("torch.distributed.fsdp")
    from torch.distributed import fsdp as torch_fsdp
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.api import FullStateDictConfig, StateDictType
    from worldforge_train.checkpoint import _trainable_state_dict

    model = _make_model()
    # Make only the last layer trainable to test filtering.
    for p in model[0].parameters():
        p.requires_grad = False

    monkeypatch.setattr(
        "worldforge_train.checkpoint._is_fsdp_module", lambda m: True
    )

    captured: dict = {}

    class FakeFSDP:
        @staticmethod
        def state_dict_type(module, state_dict_type, config=None):
            class Ctx:
                def __enter__(self):
                    captured["state_dict_type"] = state_dict_type
                    captured["config"] = config
                    return self
                def __exit__(self, *args):
                    return False
            return Ctx()

    monkeypatch.setattr(torch_fsdp, "FullyShardedDataParallel", FakeFSDP)
    monkeypatch.setattr(torch_fsdp.api, "FullStateDictConfig", MagicMock)
    monkeypatch.setattr(torch_fsdp.api, "StateDictType", StateDictType)

    state = _trainable_state_dict(model)

    assert captured["state_dict_type"].name == "FULL_STATE_DICT"
    assert "2.weight" in state
    assert "0.weight" not in state


def test_save_checkpoint_calls_fsdp_optim_state_dict(tmp_path, monkeypatch):
    """FSDP save path extracts optimizer state via FSDP.optim_state_dict."""
    pytest.importorskip("torch.distributed.fsdp")
    from torch.distributed import fsdp as torch_fsdp
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.api import FullStateDictConfig, StateDictType

    model = _make_model()
    opt = AdamW(model.parameters(), lr=0.01)
    monkeypatch.setattr("worldforge_train.checkpoint._is_fsdp_module", lambda m: True)

    captured_optim = {}

    class FakeFSDP:
        @staticmethod
        def state_dict_type(module, state_dict_type, config=None):
            class Ctx:
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    return False
            return Ctx()

        @staticmethod
        def optim_state_dict(model, optimizer):
            captured_optim["called"] = True
            return optimizer.state_dict()

    monkeypatch.setattr(torch_fsdp, "FullyShardedDataParallel", FakeFSDP)
    monkeypatch.setattr(torch_fsdp.api, "FullStateDictConfig", MagicMock)
    monkeypatch.setattr(torch_fsdp.api, "StateDictType", StateDictType)

    path = tmp_path / "ckpt.pt"
    sig = LoraSignature(r=8, alpha=16, dropout=0.0,
                        target_modules=("q_proj",), trainable_params=10)
    save_checkpoint(path, model, opt, step=1, lora_signature=sig)

    assert captured_optim.get("called")
    assert path.exists()


def test_load_checkpoint_calls_fsdp_optim_state_dict_to_load(tmp_path, monkeypatch):
    """FSDP load path restores optimizer state via FSDP.optim_state_dict_to_load."""
    pytest.importorskip("torch.distributed.fsdp")
    from torch.distributed import fsdp as torch_fsdp
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.api import FullStateDictConfig, StateDictType

    model = _make_model()
    opt = AdamW(model.parameters(), lr=0.01)
    monkeypatch.setattr("worldforge_train.checkpoint._is_fsdp_module", lambda m: True)

    captured = {}

    class FakeFSDP:
        @staticmethod
        def state_dict_type(module, state_dict_type, config=None):
            class Ctx:
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    return False
            return Ctx()

        @staticmethod
        def optim_state_dict(model, optimizer):
            return optimizer.state_dict()

        @staticmethod
        def optim_state_dict_to_load(model, optimizer, optim_state):
            captured["called"] = True
            return optim_state

    monkeypatch.setattr(torch_fsdp, "FullyShardedDataParallel", FakeFSDP)
    monkeypatch.setattr(torch_fsdp.api, "FullStateDictConfig", MagicMock)
    monkeypatch.setattr(torch_fsdp.api, "StateDictType", StateDictType)

    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, opt, step=2)

    captured.clear()
    load_checkpoint(path, model, opt)

    assert captured.get("called")
