import json
import torch


def test_run_bench_writes_json(tmp_path, monkeypatch):
    """runner must write <tag>.json with clips_per_s + loss_curve."""
    from worldforge_train.bench import runner

    # Stub model: a single Linear that outputs scalar loss.
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 1)
        def forward(self, x):
            return self.linear(x).sum()

    # Stub strategy (single-rank returns model unchanged).
    class StubStrategy:
        def wrap(self, model): return model
        def backward(self, loss): loss.backward()
        def step(self, optimizer): pass  # no-op for stub

    monkeypatch.setattr(runner, "select_strategy", lambda profile: StubStrategy())
    monkeypatch.setattr(runner, "_build_tiny_model", lambda: TinyModel())
    monkeypatch.setattr(runner, "_fixed_batch", lambda: {"x": torch.zeros(1, 4)})

    out = runner.run_bench(
        profile="train",
        steps=10,
        warmup=2,
        tag="test",
        output_dir=tmp_path,
    )
    data = json.loads(out.read_text())
    assert data["profile"] == "train"
    assert data["tag"] == "test"
    assert data["warmup_steps"] == 2
    assert data["measurement_steps"] == 8  # 10 - 2
    assert len(data["loss_curve"]) == 8
    assert "clips_per_s" in data
