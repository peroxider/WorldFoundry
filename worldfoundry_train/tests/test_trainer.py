"""Unit tests for worldfoundry_train.trainer (mock-free, fast)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from worldfoundry_train.parallel.strategy import DistributedStrategy
from worldfoundry_train.trainer import run_training


class _PassthroughStrategy(DistributedStrategy):
    """Inline no-op strategy for trainer tests."""

    def __init__(self):
        self.backward_calls: list[float] = []
        self.step_calls: list[int] = []
        self.wrapped_models: list = []

    def wrap(self, model):
        self.wrapped_models.append(model)
        return model

    def backward(self, loss):
        self.backward_calls.append(float(loss.item()))
        loss.backward()

    def step(self, optimizer):
        self.step_calls.append(len(optimizer.param_groups))
        optimizer.step()


class _TinyLinearModel(nn.Module):
    """Mimics VJEPA2Model's forward(pixel_values_videos, bool_masked_pos) API."""

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(3 * 16 * 16, 4)
        self.linear = nn.Linear(3 * 16 * 16, 4)

    def forward(self, *, pixel_values_videos, bool_masked_pos):
        return SimpleNamespace(
            last_hidden_state=self.q_proj(pixel_values_videos.flatten(2))
        )


def _fake_batch():
    return {
        "pixel_values_videos": torch.zeros(1, 1, 3, 16, 16),
        "bool_masked_pos": torch.zeros(1, 1, 4, dtype=torch.bool),
    }


class _FakeDataset:
    """Fake dataset: returns synthetic numpy samples without disk/av decoding."""

    def __init__(self, n=10, T=1, H=16, W=16, seed=0):
        self.n = n
        self.T = T
        self.H = H
        self.W = W
        np.random.seed(seed)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng(i + 1)
        return {
            "frames": rng.random((self.T, 3, self.H, self.W), dtype=np.float32),
            "mask": rng.integers(0, 2, size=(self.T, 1, 1), dtype=bool),
        }


def _cfg(**overrides):
    base = {
        "data": {"manifest": "/nonexistent", "mask_tubelet_size": [1, 14, 14],
                 "mask_ratio": 0.5},
        "lora": {"r": 4, "alpha": 8, "dropout": 0.0, "bias": "none"},
        "training": {"per_device_batch": 2, "epochs": 1, "lr": 1e-4,
                     "log_every_n": 2},
    }
    base.update(overrides)
    return base


def _loss_fn(model, batch):
    return model(
        pixel_values_videos=batch["pixel_values_videos"],
        bool_masked_pos=batch["bool_masked_pos"],
    ).last_hidden_state.sum()


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(42)
    np.random.seed(42)


def test_run_training_returns_per_step_losses():
    cfg = _cfg()
    model = _TinyLinearModel()
    strategy = _PassthroughStrategy()
    counter = {"n": 0}

    def batch_fn(_i):
        counter["n"] += 1
        return _fake_batch()

    import worldfoundry_train.trainer as trainer_mod
    original_ds = trainer_mod._build_dataset
    trainer_mod._build_dataset = lambda cfg: _FakeDataset(n=4)
    try:
        losses = run_training(
            cfg, strategy, base_model=model, num_steps=3,
            batch_fn=batch_fn, loss_fn=_loss_fn,
            logger=lambda s, loss: None,
        )
        assert len(losses) == 3
        for v in losses:
            assert isinstance(v, float)
            assert v == v
    finally:
        trainer_mod._build_dataset = original_ds


def test_run_training_num_steps_derived_from_yaml_when_unset():
    cfg = _cfg()
    model = _TinyLinearModel()
    strategy = _PassthroughStrategy()
    counter = {"n": 0}

    def batch_fn(_i):
        counter["n"] += 1
        return _fake_batch()

    import worldfoundry_train.trainer as trainer_mod
    original = trainer_mod._build_dataset
    trainer_mod._build_dataset = lambda cfg: _FakeDataset(n=4)
    try:
        run_training(
            cfg, strategy, base_model=model,
            batch_fn=batch_fn, loss_fn=_loss_fn,
            logger=lambda s, loss: None,
        )
        assert counter["n"] == 2
    finally:
        trainer_mod._build_dataset = original


def test_run_training_cli_steps_override_beats_yaml_derivation():
    cfg = _cfg()
    model = _TinyLinearModel()
    strategy = _PassthroughStrategy()
    counter = {"n": 0}

    def batch_fn(_i):
        counter["n"] += 1
        return _fake_batch()

    import worldfoundry_train.trainer as trainer_mod
    original = trainer_mod._build_dataset
    trainer_mod._build_dataset = lambda cfg: _FakeDataset(n=8)
    try:
        run_training(
            cfg, strategy, base_model=model, num_steps=1,
            batch_fn=batch_fn, loss_fn=_loss_fn,
            logger=lambda s, loss: None,
        )
        assert counter["n"] == 1
    finally:
        trainer_mod._build_dataset = original


def test_run_training_empty_dataset_raises_value_error():
    cfg = _cfg()
    model = _TinyLinearModel()
    strategy = _PassthroughStrategy()
    import worldfoundry_train.trainer as trainer_mod
    original = trainer_mod._build_dataset
    trainer_mod._build_dataset = lambda cfg: _FakeDataset(n=0)
    try:
        with pytest.raises(ValueError, match="Vjepa2Dataset is empty"):
            run_training(
                cfg, strategy, base_model=model, num_steps=1,
                batch_fn=lambda i: _fake_batch(),
                logger=lambda s, loss: None,
            )
    finally:
        trainer_mod._build_dataset = original


def test_run_training_zero_num_steps_raises_value_error():
    cfg = _cfg()
    cfg["training"]["epochs"] = 0
    cfg["training"]["per_device_batch"] = 99
    model = _TinyLinearModel()
    strategy = _PassthroughStrategy()
    import worldfoundry_train.trainer as trainer_mod
    original = trainer_mod._build_dataset
    trainer_mod._build_dataset = lambda cfg: _FakeDataset(n=4)
    try:
        with pytest.raises(ValueError, match="num_steps="):
            run_training(
                cfg, strategy, base_model=model,
                batch_fn=lambda i: _fake_batch(),
                logger=lambda s, loss: None,
            )
    finally:
        trainer_mod._build_dataset = original


def test_run_training_negative_num_steps_kwarg_raises():
    cfg = _cfg()
    model = _TinyLinearModel()
    strategy = _PassthroughStrategy()
    import worldfoundry_train.trainer as trainer_mod
    original = trainer_mod._build_dataset
    trainer_mod._build_dataset = lambda cfg: _FakeDataset(n=4)
    try:
        with pytest.raises(ValueError, match="num_steps="):
            run_training(
                cfg, strategy, base_model=model, num_steps=-1,
                batch_fn=lambda i: _fake_batch(),
                logger=lambda s, loss: None,
            )
    finally:
        trainer_mod._build_dataset = original


def test_run_training_optimizer_filters_to_trainable_params_only():
    cfg = _cfg()
    model = _TinyLinearModel()
    for n, p in model.named_parameters():
        if "q_proj" not in n:
            p.requires_grad_(False)
    strategy = _PassthroughStrategy()
    captured = {}
    import worldfoundry_train.trainer as trainer_mod
    original_opt = trainer_mod._build_optimizer

    def capturing(model, lr):
        opt = original_opt(model, lr)
        captured["params"] = [tuple(p.shape) for p in opt.param_groups[0]["params"]]
        return opt

    original_ds = trainer_mod._build_dataset
    trainer_mod._build_optimizer = capturing
    trainer_mod._build_dataset = lambda cfg: _FakeDataset(n=4)
    try:
        run_training(
            cfg, strategy, base_model=model, num_steps=1,
            batch_fn=lambda i: _fake_batch(), loss_fn=_loss_fn,
            logger=lambda s, loss: None,
        )
        # Only q_proj.weight + q_proj.bias (2 tensors)
        assert len(captured["params"]) == 2
    finally:
        trainer_mod._build_optimizer = original_opt
        trainer_mod._build_dataset = original_ds


def test_run_training_default_logger_emits_at_log_every_n():
    cfg = _cfg()
    cfg["training"]["log_every_n"] = 2
    model = _TinyLinearModel()
    strategy = _PassthroughStrategy()
    log_calls = []
    import worldfoundry_train.trainer as trainer_mod
    original_ds = trainer_mod._build_dataset
    trainer_mod._build_dataset = lambda cfg: _FakeDataset(n=8)
    try:
        run_training(
            cfg, strategy, base_model=model, num_steps=4,
            batch_fn=lambda i: _fake_batch(), loss_fn=_loss_fn,
            logger=lambda s, loss: log_calls.append((s, loss)),
        )
        assert [s for s, _ in log_calls] == [2, 4]
    finally:
        trainer_mod._build_dataset = original_ds


def test_run_training_injected_logger_overrides_default():
    cfg = _cfg()
    cfg["training"]["log_every_n"] = 1
    model = _TinyLinearModel()
    strategy = _PassthroughStrategy()
    log_calls = []
    import worldfoundry_train.trainer as trainer_mod
    original_ds = trainer_mod._build_dataset
    trainer_mod._build_dataset = lambda cfg: _FakeDataset(n=4)
    try:
        run_training(
            cfg, strategy, base_model=model, num_steps=2,
            batch_fn=lambda i: _fake_batch(), loss_fn=_loss_fn,
            logger=lambda s, loss: log_calls.append((s, loss)),
        )
        assert len(log_calls) == 2
    finally:
        trainer_mod._build_dataset = original_ds


def test_run_training_injected_loss_fn_overrides_default():
    cfg = _cfg()
    model = _TinyLinearModel()
    strategy = _PassthroughStrategy()
    sentinel = {"called": False}

    def sentinel_loss(model, batch):
        sentinel["called"] = True
        return torch.tensor(0.0, requires_grad=True)

    import worldfoundry_train.trainer as trainer_mod
    original_ds = trainer_mod._build_dataset
    trainer_mod._build_dataset = lambda cfg: _FakeDataset(n=4)
    try:
        run_training(
            cfg, strategy, base_model=model, num_steps=1,
            batch_fn=lambda i: _fake_batch(),
            loss_fn=sentinel_loss,
            logger=lambda s, loss: None,
        )
        assert sentinel["called"]
    finally:
        trainer_mod._build_dataset = original_ds


def test_run_training_strategy_wrap_called_with_lora_wrapped_model():
    cfg = _cfg()
    model = _TinyLinearModel()
    strategy = _PassthroughStrategy()
    import worldfoundry_train.trainer as trainer_mod
    original_ds = trainer_mod._build_dataset
    trainer_mod._build_dataset = lambda cfg: _FakeDataset(n=4)
    try:
        run_training(
            cfg, strategy, base_model=model, num_steps=1,
            batch_fn=lambda i: _fake_batch(), loss_fn=_loss_fn,
            logger=lambda s, loss: None,
        )
        assert len(strategy.wrapped_models) == 1
        wrapped = strategy.wrapped_models[0]
        # After LoRA wrap, the trainer hands strategy.wrap a peft.PeftModel
        # (which is NOT a bare _TinyLinearModel).
        assert not isinstance(wrapped, _TinyLinearModel)
    finally:
        trainer_mod._build_dataset = original_ds


def test_run_training_strategy_backward_called_once_per_step():
    cfg = _cfg()
    model = _TinyLinearModel()
    strategy = _PassthroughStrategy()
    import worldfoundry_train.trainer as trainer_mod
    original_ds = trainer_mod._build_dataset
    trainer_mod._build_dataset = lambda cfg: _FakeDataset(n=4)
    try:
        run_training(
            cfg, strategy, base_model=model, num_steps=3,
            batch_fn=lambda i: _fake_batch(), loss_fn=_loss_fn,
            logger=lambda s, loss: None,
        )
        assert len(strategy.backward_calls) == 3
    finally:
        trainer_mod._build_dataset = original_ds


# --- W16 Checkpoint tests ---

from pathlib import Path  # noqa: E402  (used by W16 tests below)


@pytest.fixture
def _patch_checkpoint_safe_dist(monkeypatch):
    """Force torch.distributed.is_initialized() -> False so multi-rank guard does not fire."""
    import torch.distributed as _dist
    monkeypatch.setattr(_dist, "is_initialized", lambda: False)


def test_run_training_saves_final_checkpoint(tmp_path, _patch_checkpoint_safe_dist):
    cfg = _cfg()
    model = _TinyLinearModel()
    import worldfoundry_train.trainer as trainer_mod
    original_ds = trainer_mod._build_dataset
    trainer_mod._build_dataset = lambda cfg: _FakeDataset(n=4)
    try:
        run_training(
            cfg, _PassthroughStrategy(), base_model=model, num_steps=3,
            batch_fn=lambda i: _fake_batch(), loss_fn=_loss_fn,
            logger=lambda s, loss: None,
            checkpoint_dir=tmp_path,
        )
        saved = tmp_path / "step_3.pt"
        assert saved.is_file()
    finally:
        trainer_mod._build_dataset = original_ds


def test_run_training_save_every_n_produces_multiple_files(tmp_path, _patch_checkpoint_safe_dist):
    cfg = _cfg()
    model = _TinyLinearModel()
    import worldfoundry_train.trainer as trainer_mod
    original_ds = trainer_mod._build_dataset
    trainer_mod._build_dataset = lambda cfg: _FakeDataset(n=8)
    try:
        run_training(
            cfg, _PassthroughStrategy(), base_model=model, num_steps=6,
            batch_fn=lambda i: _fake_batch(), loss_fn=_loss_fn,
            logger=lambda s, loss: None,
            checkpoint_dir=tmp_path,
            save_every_n=2,
        )
        saved_files = sorted(p.name for p in tmp_path.glob("step_*.pt"))
        assert saved_files == ["step_2.pt", "step_4.pt", "step_6.pt"]
    finally:
        trainer_mod._build_dataset = original_ds


def test_run_training_save_every_n_without_dir_raises(_patch_checkpoint_safe_dist):
    cfg = _cfg()
    model = _TinyLinearModel()
    import worldfoundry_train.trainer as trainer_mod
    original_ds = trainer_mod._build_dataset
    trainer_mod._build_dataset = lambda cfg: _FakeDataset(n=4)
    try:
        with pytest.raises(ValueError, match="save_every_n=2 but checkpoint_dir is None"):
            run_training(
                cfg, _PassthroughStrategy(), base_model=model, num_steps=3,
                batch_fn=lambda i: _fake_batch(), loss_fn=_loss_fn,
                logger=lambda s, loss: None,
                save_every_n=2,
            )
    finally:
        trainer_mod._build_dataset = original_ds


def test_run_training_resume_from_checkpoint_continues_losses(tmp_path, _patch_checkpoint_safe_dist):
    """5 steps + save + fresh cfg + load + 5 more steps should match 10-step run."""
    def _reset_rng():
        torch.manual_seed(42)
        np.random.seed(42)

    cfg = _cfg()
    import worldfoundry_train.trainer as trainer_mod
    original_ds = trainer_mod._build_dataset
    trainer_mod._build_dataset = lambda cfg: _FakeDataset(n=8)
    try:
        _reset_rng()
        losses_10 = run_training(
            cfg, _PassthroughStrategy(), base_model=_TinyLinearModel(), num_steps=10,
            batch_fn=lambda i: _fake_batch(), loss_fn=_loss_fn,
            logger=lambda s, loss: None,
        )

        _reset_rng()
        losses_first = run_training(
            cfg, _PassthroughStrategy(), base_model=_TinyLinearModel(), num_steps=5,
            batch_fn=lambda i: _fake_batch(), loss_fn=_loss_fn,
            logger=lambda s, loss: None,
            checkpoint_dir=tmp_path,
        )
        saved = tmp_path / "step_5.pt"
        assert saved.is_file()

        _reset_rng()
        losses_second = run_training(
            cfg, _PassthroughStrategy(), base_model=_TinyLinearModel(), num_steps=10,
            batch_fn=lambda i: _fake_batch(), loss_fn=_loss_fn,
            logger=lambda s, loss: None,
            resume_from=saved,
        )

        assert len(losses_first) == 5
        assert len(losses_second) == 5
        np.testing.assert_allclose(
            np.array(losses_first + losses_second),
            np.array(losses_10),
            rtol=1e-5,
            err_msg="resume did not reproduce single-run losses",
        )
    finally:
        trainer_mod._build_dataset = original_ds


def test_run_training_resume_with_lora_signature_mismatch_raises(tmp_path, _patch_checkpoint_safe_dist):
    cfg = _cfg()
    model = _TinyLinearModel()
    import worldfoundry_train.trainer as trainer_mod
    original_ds = trainer_mod._build_dataset
    trainer_mod._build_dataset = lambda cfg: _FakeDataset(n=4)
    try:
        run_training(
            cfg, _PassthroughStrategy(), base_model=model, num_steps=3,
            batch_fn=lambda i: _fake_batch(), loss_fn=_loss_fn,
            logger=lambda s, loss: None,
            checkpoint_dir=tmp_path,
        )
        saved = tmp_path / "step_3.pt"

        # Change alpha (not r): r determines LoRA matrix shapes so a different r
        # would surface as a torch RuntimeError during load_checkpoint before
        # validate_lora_signature runs. Alpha mismatch keeps shapes intact, so
        # load succeeds and the signature validator raises the expected ValueError.
        cfg_changed = _cfg()
        cfg_changed["lora"]["alpha"] = 999
        with pytest.raises(ValueError, match="lora_signature.alpha"):
            run_training(
                cfg_changed, _PassthroughStrategy(), base_model=_TinyLinearModel(),
                num_steps=6, batch_fn=lambda i: _fake_batch(), loss_fn=_loss_fn,
                logger=lambda s, loss: None, resume_from=saved,
            )
    finally:
        trainer_mod._build_dataset = original_ds


def test_run_training_resume_missing_file_raises(_patch_checkpoint_safe_dist):
    cfg = _cfg()
    model = _TinyLinearModel()
    import worldfoundry_train.trainer as trainer_mod
    original_ds = trainer_mod._build_dataset
    trainer_mod._build_dataset = lambda cfg: _FakeDataset(n=4)
    try:
        with pytest.raises(FileNotFoundError):
            run_training(
                cfg, _PassthroughStrategy(), base_model=model, num_steps=3,
                batch_fn=lambda i: _fake_batch(), loss_fn=_loss_fn,
                logger=lambda s, loss: None,
                resume_from=Path("/nonexistent/ckpt.pt"),
            )
    finally:
        trainer_mod._build_dataset = original_ds


# --- W25-roadmap: full-parameter fine-tuning mode ---


def test_run_training_unknown_mode_raises():
    cfg = _cfg()
    model = _TinyLinearModel()
    strategy = _PassthroughStrategy()
    import worldfoundry_train.trainer as trainer_mod
    original_ds = trainer_mod._build_dataset
    trainer_mod._build_dataset = lambda cfg: _FakeDataset(n=4)
    try:
        with pytest.raises(ValueError, match="unknown training mode"):
            run_training(
                cfg, strategy, base_model=model, num_steps=1, mode="bogus",
                batch_fn=lambda i: _fake_batch(), loss_fn=_loss_fn,
                logger=lambda s, loss: None,
            )
    finally:
        trainer_mod._build_dataset = original_ds


def test_run_training_full_mode_skips_lora_wrapper():
    """W25-roadmap: in 'full' mode the strategy receives the bare base model
    (not a peft.PeftModel) and every parameter has requires_grad=True."""
    cfg = _cfg()
    model = _TinyLinearModel()
    strategy = _PassthroughStrategy()
    import worldfoundry_train.trainer as trainer_mod
    original_ds = trainer_mod._build_dataset
    trainer_mod._build_dataset = lambda cfg: _FakeDataset(n=4)
    try:
        run_training(
            cfg, strategy, base_model=model, num_steps=1, mode="full",
            batch_fn=lambda i: _fake_batch(), loss_fn=_loss_fn,
            logger=lambda s, loss: None,
        )
        assert len(strategy.wrapped_models) == 1
        wrapped = strategy.wrapped_models[0]
        # Full mode bypasses build_lora_model: the model handed to strategy.wrap
        # is still the original _TinyLinearModel (not a peft.PeftModel).
        assert isinstance(wrapped, _TinyLinearModel)
        for parameter in wrapped.parameters():
            assert parameter.requires_grad
    finally:
        trainer_mod._build_dataset = original_ds


def test_run_training_full_mode_does_not_call_build_lora():
    """W25-roadmap: build_lora_model is never invoked in 'full' mode."""
    cfg = _cfg()
    model = _TinyLinearModel()
    strategy = _PassthroughStrategy()
    import worldfoundry_train.trainer as trainer_mod
    original_ds = trainer_mod._build_dataset
    original_lora = trainer_mod.build_lora_model
    lora_calls = {"n": 0}

    def tracking(*args, **kwargs):
        lora_calls["n"] += 1
        return original_lora(*args, **kwargs)

    trainer_mod.build_lora_model = tracking
    trainer_mod._build_dataset = lambda cfg: _FakeDataset(n=4)
    try:
        run_training(
            cfg, strategy, base_model=model, num_steps=1, mode="full",
            batch_fn=lambda i: _fake_batch(), loss_fn=_loss_fn,
            logger=lambda s, loss: None,
        )
        assert lora_calls["n"] == 0
    finally:
        trainer_mod._build_dataset = original_ds
        trainer_mod.build_lora_model = original_lora


def test_run_training_full_mode_checkpoint_round_trip(tmp_path, _patch_checkpoint_safe_dist):
    """W25-roadmap: a 3-step full-FT run writes a final checkpoint whose
    lora_signature is None and whose state_dict restores without error."""
    cfg = _cfg()
    import worldfoundry_train.trainer as trainer_mod
    original_ds = trainer_mod._build_dataset
    trainer_mod._build_dataset = lambda cfg: _FakeDataset(n=4)
    try:
        run_training(
            cfg, _PassthroughStrategy(), base_model=_TinyLinearModel(), num_steps=3,
            mode="full",
            batch_fn=lambda i: _fake_batch(), loss_fn=_loss_fn,
            logger=lambda s, loss: None,
            checkpoint_dir=tmp_path,
        )
        saved = tmp_path / "step_3.pt"
        assert saved.is_file()

        # The full-mode payload has lora_signature=None (already tolerated
        # by load_checkpoint), and the model_state still contains the base
        # model parameters so the inference loader can rehydrate it.
        import torch
        payload = torch.load(saved, map_location="cpu", weights_only=False)
        assert payload["lora_signature"] is None
        assert any("q_proj" in k or "linear" in k for k in payload["model_state"])
    finally:
        trainer_mod._build_dataset = original_ds
