"""Unit tests for worldfoundry_train.checkpoint — save/load round-trips, atomicity,
signature validation. CPU, no mocks. tmp_path-style pytest fixtures."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn
from torch.optim import AdamW

from worldfoundry_train.checkpoint import (
    LoraSignature, _trainable_state_dict,
    load_checkpoint, save_checkpoint, validate_lora_signature,
)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(42)
    np.random.seed(42)


def _make_model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 1))


def _make_optimizer(model: nn.Module) -> AdamW:
    return AdamW(model.parameters(), lr=0.01)


def _sample_sig(trainable_params: int = 100) -> LoraSignature:
    return LoraSignature(r=8, alpha=16, dropout=0.1,
                         target_modules=("q_proj",),
                         trainable_params=trainable_params)


def test_trainable_state_dict_excludes_frozen_params():
    class MixedModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.frozen = nn.Linear(4, 4)
            self.frozen.requires_grad_(False)
            self.trainable = nn.Linear(4, 1)
        def forward(self, x):  # pragma: no cover
            return self.trainable(self.frozen(x))

    m = MixedModel()
    state = m.state_dict()
    assert "frozen.weight" in state
    assert "trainable.weight" in state

    filt = _trainable_state_dict(m)
    assert "trainable.weight" in filt
    assert "frozen.weight" not in filt
    assert "frozen.bias" not in filt


def test_save_load_round_trip_preserves_weights(tmp_path):
    m1 = _make_model()
    opt1 = _make_optimizer(m1)
    m1(torch.randn(2, 4)).sum().backward()
    opt1.step()

    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, m1, opt1, step=5, lora_signature=_sample_sig())

    torch.manual_seed(0)
    m2 = _make_model()
    opt2 = _make_optimizer(m2)

    result = load_checkpoint(path, m2, opt2)
    assert result["step"] == 5
    assert result["lora_signature"]["r"] == 8
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(p1, p2), "weights diverged across save/load"


def test_save_load_round_trip_preserves_optimizer_state(tmp_path):
    m1 = _make_model()
    opt1 = _make_optimizer(m1)
    for _ in range(3):
        m1(torch.randn(2, 4)).sum().backward()
        opt1.step()

    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, m1, opt1, step=3)

    torch.manual_seed(0)
    m2 = _make_model()
    opt2 = _make_optimizer(m2)
    load_checkpoint(path, m2, opt2)

    # AdamW state per param: {step: int, exp_avg: Tensor, exp_avg_sq: Tensor}.
    # `step` is an int and may differ in identity; only Tensor fields must match.
    sd1 = opt1.state_dict()
    sd2 = opt2.state_dict()
    assert set(sd1["state"].keys()) == set(sd2["state"].keys())
    for param_idx in sd1["state"]:
        for k in ("exp_avg", "exp_avg_sq"):
            t1 = sd1["state"][param_idx][k]
            t2 = sd2["state"][param_idx][k]
            assert torch.allclose(t1, t2), f"optimizer state[{param_idx}].{k} diverged"


def test_save_load_round_trip_preserves_numpy_rng_state(tmp_path):
    np.random.seed(42)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, _make_model(), _make_optimizer(_make_model()), step=0)
    expected = np.random.rand(5)

    np.random.seed(99)
    _ = np.random.rand(5)

    load_checkpoint(path, _make_model(), _make_optimizer(_make_model()))
    actual = np.random.rand(5)
    np.testing.assert_array_equal(actual, expected)


def test_load_missing_checkpoint_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "nonexistent.pt",
                        _make_model(), _make_optimizer(_make_model()))


def test_load_checkpoint_missing_keys_raises_key_error(tmp_path):
    path = tmp_path / "broken.pt"
    torch.save({"step": 0, "model_state": {}}, path)
    with pytest.raises(KeyError):
        load_checkpoint(path, _make_model(), _make_optimizer(_make_model()))


def test_load_corrupted_checkpoint_raises_io_error(tmp_path):
    path = tmp_path / "corrupt.pt"
    path.write_bytes(b"not a torch save format")
    with pytest.raises(IOError):
        load_checkpoint(path, _make_model(), _make_optimizer(_make_model()))


def test_lora_signature_validation_passes_on_match():
    sig = _sample_sig()
    validate_lora_signature(sig.to_dict(), sig)


def test_lora_signature_validation_skips_when_ckpt_sig_is_none():
    validate_lora_signature(None, _sample_sig())


def test_lora_signature_validation_raises_on_r_mismatch():
    ckpt = _sample_sig()
    current = LoraSignature(r=64, alpha=16, dropout=0.1,
                            target_modules=("q_proj",), trainable_params=100)
    with pytest.raises(ValueError, match="lora_signature.r"):
        validate_lora_signature(ckpt.to_dict(), current)


def test_lora_signature_validation_raises_on_target_modules_mismatch():
    ckpt = _sample_sig()
    current = LoraSignature(r=8, alpha=16, dropout=0.1,
                            target_modules=("k_proj",), trainable_params=100)
    with pytest.raises(ValueError, match="lora_signature.target_modules"):
        validate_lora_signature(ckpt.to_dict(), current)


def test_trainable_params_drift_warns_not_raises():
    ckpt = LoraSignature(r=8, alpha=16, dropout=0.1,
                         target_modules=("q_proj",), trainable_params=100)
    current = LoraSignature(r=8, alpha=16, dropout=0.1,
                            target_modules=("q_proj",), trainable_params=200)
    with pytest.warns(UserWarning, match="trainable_params"):
        validate_lora_signature(ckpt.to_dict(), current)


def test_atomic_save_no_partial_file_on_simulated_crash(tmp_path, monkeypatch):
    path = tmp_path / "step_42.pt"

    def crashing_save(payload, p):
        raise RuntimeError("simulated disk full")

    monkeypatch.setattr(torch, "save", crashing_save)

    with pytest.raises(RuntimeError, match="simulated"):
        save_checkpoint(path, _make_model(), _make_optimizer(_make_model()), step=42)

    assert not path.exists(), "atomic save should not leave a partial .pt file"
    assert not (tmp_path / "step_42.pt.tmp").exists()
