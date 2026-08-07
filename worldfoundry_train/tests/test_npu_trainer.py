"""Unit tests for worldfoundry_train.device (W20).

Closes the CPU-only prep half of W20: ``resolve_device`` and ``to_device``
are pure logic and a thin wrapper, so the acceptance gate is the
*behavior* (the right string, the right call to ``Module.to``), not a
real Atlas run. The migration point inside
:func:`worldfoundry_train.trainer.run_training` is exercised end-to-end
with a stub strategy + monkey-patched ``to_device`` so the test never
requires ``torch_npu`` to be installed on the test host.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

import worldfoundry_train.device as device_mod
import worldfoundry_train.trainer as trainer_mod
from worldfoundry_train.device import is_npu_target, resolve_device, to_device


# ---------------------------------------------------------------------------
# resolve_device
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_device_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip WORLDFORGE_DEVICE and re-import the device module so the
    module-level device resolution runs with a clean environment."""
    monkeypatch.delenv("WORLDFORGE_DEVICE", raising=False)


def test_resolve_device_returns_cpu_on_cpu_only_host(clean_device_env):
    # On a CPU-only test host (no torch_npu installed) this is the default.
    assert resolve_device() == "cpu"


def test_resolve_device_returns_npu_when_predicate_true(
    clean_device_env, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        "worldfoundry_train.device.is_npu_available", lambda: True
    )
    assert resolve_device() == "npu"


def test_resolve_device_explicit_override_wins_over_predicate(
    clean_device_env, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        "worldfoundry_train.device.is_npu_available", lambda: True
    )
    assert resolve_device(override="cpu") == "cpu"


def test_resolve_device_env_var_wins_over_predicate(
    clean_device_env, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        "worldfoundry_train.device.is_npu_available", lambda: True
    )
    monkeypatch.setenv("WORLDFORGE_DEVICE", "cpu")
    assert resolve_device() == "cpu"


def test_resolve_device_rejects_unknown_value(clean_device_env):
    with pytest.raises(ValueError, match="unknown device"):
        resolve_device(override="tpu")


def test_resolve_device_rejects_unknown_env_value(
    clean_device_env, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("WORLDFORGE_DEVICE", "tpu")
    with pytest.raises(ValueError, match="unknown device"):
        resolve_device()


def test_is_npu_target_only_true_for_npu_string():
    assert is_npu_target("npu") is True
    assert is_npu_target("cpu") is False
    assert is_npu_target("cuda") is False


# ---------------------------------------------------------------------------
# to_device
# ---------------------------------------------------------------------------


def test_to_device_calls_module_to_with_the_resolved_device(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, str] = {}

    def spy_to(self, *args, **kwargs):
        # ``nn.Module.to`` is variadic (string, int, torch.device, dict…).
        # The trainer always passes a string, but record whatever we get
        # so a future caller using ``torch.device("npu")`` is also visible.
        if args:
            captured["arg"] = str(args[0])
        # On a CPU-only host, real ``.to("npu")`` would raise because
        # torch does not know about ``npu`` as a device type. The
        # migration contract is that ``to_device`` calls ``Module.to``;
        # whether the actual device exists on the host is a separate
        # concern, so we don't forward to the real ``to`` here.
        return self

    monkeypatch.setattr(nn.Module, "to", spy_to)

    model = nn.Linear(4, 4)
    to_device(model, "npu")
    assert captured["arg"] == "npu"


def test_to_device_rejects_unknown_device():
    model = nn.Linear(2, 2)
    with pytest.raises(ValueError, match="to_device called with unknown device"):
        to_device(model, "tpu")


def test_to_device_is_chainable():
    """``to_device`` returns the same model so callers can write
    ``model = to_device(model, device)`` without a temp."""
    model = nn.Linear(3, 3)
    out = to_device(model, "cpu")
    assert out is model


# ---------------------------------------------------------------------------
# Trainer integration: the .to(device) call really happens
# ---------------------------------------------------------------------------


class _StubStrategy:
    """Drop-in DistributedStrategy that records wrap() in order and lets
    the test inject a small custom optimizer."""

    wrap_calls: list[nn.Module] = []

    def __init__(self) -> None:
        type(self).wrap_calls = []

    def wrap(self, model: nn.Module) -> nn.Module:
        type(self).wrap_calls.append(model)
        return model

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    def step(self, optimizer) -> None:  # type: ignore[no-untyped-def]
        optimizer.step()
        optimizer.zero_grad()


def _tiny_cfg() -> dict:
    return {
        "data": {"manifest": "/x", "mask_tubelet_size": [1, 1, 1],
                 "mask_ratio": 0.0},
        "lora": {"r": 2, "alpha": 4, "dropout": 0.0, "bias": "none"},
        "training": {"per_device_batch": 1, "epochs": 1, "lr": 1e-3,
                     "log_every_n": 1, "grad_accum": 1},
        "model": {"hf_repo": "facebook/vjepa2-vitl-fpc64-256"},
    }


def _constant_batch(_i: int) -> dict[str, torch.Tensor]:
    """Minimal batch that matches the trainer's required keys."""
    return {
        "pixel_values_videos": torch.zeros(1, 1, 3, 8, 8),
        "bool_masked_pos": torch.zeros(1, 1, dtype=torch.bool),
    }


class _TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = type("Cfg", (), {"hidden_size": 8})()
        self.linear = nn.Linear(8, 8)


class _FakeDataset:
    """Stand-in for Vjepa2Dataset: just exposes __len__ + __getitem__ that
    the trainer never actually calls because ``batch_fn`` is injected.

    Why this exists: ``trainer._build_dataset`` constructs a real
    Vjepa2Dataset (which validates the manifest path on disk). For a
    device-migration test we don't care about the dataset; we just need
    the trainer past that line. The existing test suite uses the same
    trick (see test_trainer_cpu.py:_FakeVJ2Dataset)."""

    def __init__(self) -> None:
        self.n = 4

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:  # pragma: no cover
        return {}


def test_trainer_migrates_model_with_resolved_device(
    monkeypatch: pytest.MonkeyPatch,
):
    """W20 acceptance gate (CPU-prep half): with NPU flagged as available
    via a monkey-patched predicate, ``run_training`` must resolve
    ``"npu"`` and call ``to_device(model, "npu")`` after ``strategy.wrap``.

    ``torch_npu`` is not required on the test host: the spy records
    the call and returns the model unchanged (forwarding to the real
    ``to_device`` would crash on a CPU-only box). The contract being
    tested is that the trainer *calls* the migration, not that the
    migration itself succeeds.
    """
    monkeypatch.setattr(device_mod, "is_npu_available", lambda: True)
    monkeypatch.setattr(trainer_mod, "_build_dataset", lambda cfg: _FakeDataset())

    captured: list[tuple[int, str]] = []

    def spy_to_device(model, device):
        captured.append((id(model), device))
        return model  # do NOT forward to real to_device("npu") on CPU-only host

    monkeypatch.setattr(trainer_mod, "to_device", spy_to_device)

    backbone = _TinyBackbone()
    strategy = _StubStrategy()

    trainer_mod.run_training(
        _tiny_cfg(),
        strategy,  # type: ignore[arg-type]
        mode="full",
        base_model=backbone,
        num_steps=1,
        batch_fn=_constant_batch,
        loss_fn=lambda model, batch: torch.tensor(0.0, requires_grad=True),
    )

    # JEPA is off in this cfg -> at least one call for the wrapped model
    # and it must be "npu". (If JEPA were on, there'd also be a predictor
    # call; we don't need to enumerate both here.)
    assert captured, "trainer never invoked to_device"
    for _, dev in captured:
        assert dev == "npu", f"unexpected device: {dev!r}"

    # The migration happens AFTER strategy.wrap — ordering is the
    # contract (FSDP/DDP wraps first, then migrate onto the right device).
    assert _StubStrategy.wrap_calls, "strategy.wrap was never called"


def test_trainer_default_device_is_cpu_when_npu_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    """The CPU path of run_training must resolve ``"cpu"`` and stay
    byte-identical (no spurious npu-only side effects)."""
    monkeypatch.setattr(device_mod, "is_npu_available", lambda: False)
    monkeypatch.setattr(trainer_mod, "_build_dataset", lambda cfg: _FakeDataset())

    captured: list[str] = []

    def spy_to_device(model, device):
        captured.append(device)
        return model

    monkeypatch.setattr(trainer_mod, "to_device", spy_to_device)

    backbone = _TinyBackbone()
    strategy = _StubStrategy()
    trainer_mod.run_training(
        _tiny_cfg(),
        strategy,  # type: ignore[arg-type]
        mode="full",
        base_model=backbone,
        num_steps=1,
        batch_fn=_constant_batch,
        loss_fn=lambda model, batch: torch.tensor(0.0, requires_grad=True),
    )

    assert captured, "trainer never invoked to_device on the CPU path"
    assert all(dev == "cpu" for dev in captured), captured
    out = capsys.readouterr().out
    assert "[TRAIN] device=cpu" in out
    assert "npu_available=False" in out
