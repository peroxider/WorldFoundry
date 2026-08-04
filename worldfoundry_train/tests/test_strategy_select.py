import pytest


def test_train_returns_fsdp():
    from worldfoundry_train.parallel.strategy import select_strategy
    from worldfoundry_train.parallel.fsdp import FSDPStrategy
    s = select_strategy("train")
    assert isinstance(s, FSDPStrategy)


def test_train_mmspeed_returns_mm(monkeypatch):
    from worldfoundry_train.parallel import mindspeed
    class FakeMMStrategy:
        def wrap(self, model): return model
        def backward(self, loss): pass
        def step(self, optimizer): pass
    monkeypatch.setattr(mindspeed, "MMStrategy", FakeMMStrategy)
    from worldfoundry_train.parallel.strategy import select_strategy
    s = select_strategy("train-mmspeed")
    assert isinstance(s, FakeMMStrategy)


def test_unknown_raises():
    from worldfoundry_train.parallel.strategy import select_strategy
    with pytest.raises(ValueError, match="foo"):
        select_strategy("foo")


def test_mm_missing_dep_raises(monkeypatch):
    """If mindspeed import fails, error message must include pip install hint (G4)."""
    import builtins
    real_import = builtins.__import__
    def fake_import(name, *args, **kwargs):
        if name == "worldfoundry_train.parallel.mindspeed":
            raise ImportError("simulated missing")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    from worldfoundry_train.parallel.strategy import select_strategy
    with pytest.raises(ImportError, match=r"pip install"):
        select_strategy("train-mmspeed")