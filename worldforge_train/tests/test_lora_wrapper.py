import pytest
import torch
import torch.nn as nn

from worldforge_train.model.lora_wrapper import build_lora_model


class _MatMul:
    def __matmul__(self, other):
        return other


class _TensorStub:
    def __init__(self):
        self.shape = (1, 1)


class _AttentionStub(nn.Module):
    def __init__(self):
        super().__init__()
        # Use real nn.Linear so discover_lora_targets picks the names up via
        # named_modules(). Stubs must be nn.Modules per the implementer prompt.
        self.q_proj = nn.Linear(2, 2)
        self.k_proj = nn.Linear(2, 2)
        self.v_proj = nn.Linear(2, 2)
        self.o_proj = nn.Linear(2, 2)


class _MLPStub(nn.Module):
    def __init__(self):
        super().__init__()
        # Use ModuleList so nested attribute names "fc1" and "fc2" register.
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 2)


class _BlockStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _AttentionStub()
        self.mlp = _MLPStub()


class _TinyVJEPA2(nn.Module):
    """Stand-in for transformers.VJEPA2Model with realistic submodule names."""

    def __init__(self):
        super().__init__()
        # predictor.head.weight as a Parameter so the nested module is registered.
        self.predictor = nn.Module()
        self.predictor.head = nn.Module()
        self.predictor.head.weight = nn.Parameter(torch.zeros(2, 2))
        self.blocks = nn.ModuleList([_BlockStub(), _BlockStub()])


def test_build_lora_returns_peft_model():
    # We patch peft.LoraConfig + peft.get_peft_model to avoid the real HF
    # dep in unit tests. Patching peft.* (not lora_wrapper.*) is required
    # because build_lora_model uses `from peft import LoraConfig, get_peft_model`
    # which creates local names that shadow module-level rebindings.
    import peft

    captured = {}

    class FakeLoraConfig:
        def __init__(self, **kw):
            captured["config"] = kw

    class FakePeftModel:
        def __init__(self, base, config):
            captured["base"] = base
            captured["peft_config"] = config

        def parameters(self):
            # Empty iterator so trainable_params == 0; the
            # trainable_params==0 guard runs but the assertion below happens
            # BEFORE we ever reach it (because build_lora_model raises).
            # We return a small param-like shim that yields at least one
            # trainable element so the success path returns normally.
            class _P:
                numel = staticmethod(lambda: 1)
                requires_grad = True
            return iter([_P()])

    original_LoraConfig = peft.LoraConfig
    original_get_peft_model = peft.get_peft_model
    peft.LoraConfig = FakeLoraConfig
    peft.get_peft_model = lambda base, cfg: FakePeftModel(base, cfg)
    try:
        base = _TinyVJEPA2()
        model = build_lora_model(base, r=16, alpha=32)

        assert captured["base"] is base
        # target_modules should have been auto-discovered from the actual names
        assert "blocks.0.attn.q_proj" in captured["config"]["target_modules"]
        assert "blocks.0.mlp.fc1" in captured["config"]["target_modules"]
    finally:
        peft.LoraConfig = original_LoraConfig
        peft.get_peft_model = original_get_peft_model


def test_build_lora_raises_when_no_targets_match():
    import peft

    class Empty(nn.Module):
        def __init__(self):
            super().__init__()
            self.predictor = nn.Module()
            self.predictor.head = nn.Module()
            self.predictor.head.weight = nn.Parameter(torch.zeros(2, 2))
            self.blocks = nn.ModuleList()

    original_LoraConfig = peft.LoraConfig
    original_get_peft_model = peft.get_peft_model
    peft.LoraConfig = lambda **kw: None
    peft.get_peft_model = lambda base, cfg: object()
    try:
        with pytest.raises(ValueError, match="LoRA targets matched 0 modules"):
            build_lora_model(Empty())
    finally:
        peft.LoraConfig = original_LoraConfig
        peft.get_peft_model = original_get_peft_model


def test_build_lora_raises_when_peft_model_has_zero_trainable_params():
    """Targets may be non-empty (e.g. wrong bias, or all matched modules
    skipped) but the resulting PEFT model can still expose 0 trainable
    parameters. Per G4 we must abort in that case too.
    """
    import peft

    class FakeLoraConfig:
        def __init__(self, **kw):
            pass

    class FakePeftModelZeroTrainable:
        def __init__(self, base, config):
            pass

        def parameters(self):
            # Yield only params with requires_grad=False so trainable_params == 0.
            class _P:
                numel = staticmethod(lambda: 4)
                requires_grad = False
            return iter([_P(), _P()])

    original_LoraConfig = peft.LoraConfig
    original_get_peft_model = peft.get_peft_model
    peft.LoraConfig = FakeLoraConfig
    peft.get_peft_model = lambda base, cfg: FakePeftModelZeroTrainable(base, cfg)
    try:
        base = _TinyVJEPA2()  # has real matching targets so discover_lora_targets != []
        with pytest.raises(ValueError, match=r"0 trainable params"):
            build_lora_model(base)
    finally:
        peft.LoraConfig = original_LoraConfig
        peft.get_peft_model = original_get_peft_model
