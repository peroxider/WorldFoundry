import torch
import torch.nn as nn

from worldfoundry_train.model.module_names import discover_lora_targets


class FakeLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2, 2))


def test_discover_returns_only_present_names():
    import torch

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = FakeLinear()
            self.k_proj = FakeLinear()
            self.unrelated = nn.Linear(2, 2)

    model = TinyModel()
    targets = discover_lora_targets(
        model,
        candidates=["q_proj", "k_proj", "v_proj", "o_proj", "mlp.fc1", "mlp.fc2"],
    )
    assert "q_proj" in targets
    assert "k_proj" in targets
    assert "v_proj" not in targets
    assert "mlp.fc1" not in targets


def test_discover_supports_nested_module_names():
    import torch

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(2, 2)
            self.fc2 = nn.Linear(2, 2)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = MLP()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([Block(), Block()])

    model = Model()
    targets = discover_lora_targets(model, ["mlp.fc1"])
    assert "blocks.0.mlp.fc1" in targets
    assert "blocks.1.mlp.fc1" in targets


def test_discover_suffix_only_false_requires_exact_full_path():
    """With suffix_only=False, candidate must equal the module's full path."""

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(2, 2)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = Attention()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([Block(), Block()])

    model = Model()

    # Leaf-only candidate does NOT match the full path blocks.0.attn.q_proj.
    targets = discover_lora_targets(
        model, ["q_proj"], suffix_only=False
    )
    assert targets == []

    # Fully-qualified candidate matches only that exact path.
    targets = discover_lora_targets(
        model, ["blocks.0.attn.q_proj"], suffix_only=False
    )
    assert targets == ["blocks.0.attn.q_proj"]
