import pytest
import torch


def test_wrap_calls_fully_sharded_data_parallel(monkeypatch):
    """On multi-rank, wrap() must invoke FullyShardedDataParallel(model, use_orig_params=True)."""
    pytest.importorskip("torch.distributed.fsdp")
    import torch.distributed as dist
    from worldforge_train.parallel.fsdp import FSDPStrategy

    captured = {}

    class FakeFSDPModule:
        def __init__(self, model, **kwargs):
            captured["model"] = model
            captured["kwargs"] = kwargs

    # wrap() does `from torch.distributed.fsdp import FullyShardedDataParallel`
    # inside the function, so we patch the symbol on that submodule.
    from torch.distributed import fsdp as torch_fsdp
    monkeypatch.setattr(torch_fsdp, "FullyShardedDataParallel", FakeFSDPModule)

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)

    model = torch.nn.Linear(4, 4)
    FSDPStrategy().wrap(model)
    assert "kwargs" in captured
    assert captured["kwargs"].get("use_orig_params") is True


def test_step_all_reduces_and_steps(monkeypatch):
    """step() all-reduces gradients before optimizer.step().

    Also pins equivalence with Phase-1 trainer (G3): FSDPStrategy.step() must
    use the same primitives Phase-1 used (all_reduce -> step -> zero_grad) so
    W9 baseline numbers stay comparable to Phase-1 (spec §3.2 equivalence).
    """
    from worldforge_train.parallel.fsdp import FSDPStrategy
    import torch.distributed as dist
    calls = []
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "all_reduce", lambda grad, **kwargs: calls.append("all_reduce"))
    p = torch.nn.Parameter(torch.ones(1)); p.grad = torch.ones(1)
    class FakeOpt:
        param_groups = [{"params": [p]}]
        def step(self): calls.append("step")
        def zero_grad(self): calls.append("zero_grad")
    FSDPStrategy().step(FakeOpt())
    # Same set of primitives as Phase-1: optimizer.step() + zero_grad()
    assert calls == ["all_reduce", "step", "zero_grad"]
    assert calls.index("all_reduce") < calls.index("step")
    assert "zero_grad" in calls


def test_equivalent_to_phase1_trainer(monkeypatch):
    """FSDPStrategy.step() must produce same loss as direct optimizer.step()
    (Phase-1 T3 trainer behavior) within atol=1e-6 (spec §6.1 row 7).

    Proves W9 baseline numbers (from FSDPStrategy) are comparable to Phase-1.
    """
    import torch
    from worldforge_train.parallel.fsdp import FSDPStrategy

    # Seed before each model construction so both Linear(4,1) inits match.
    # The brief assumes identical inits; without re-seeding, path A's
    # backward+step consumes RNG state and path B gets different weights.
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.zeros(1, 4)

    # Path A: FSDPStrategy.step()
    out_a = model(x).sum()
    out_a.backward()
    FSDPStrategy().step(optimizer)

    # Path B: direct optimizer.step() (Phase-1 trainer behavior)
    torch.manual_seed(0)
    model_b = torch.nn.Linear(4, 1)
    opt_b = torch.optim.SGD(model_b.parameters(), lr=0.01)
    out_b = model_b(x).sum()
    out_b.backward()
    opt_b.step()
    opt_b.zero_grad()

    # Both paths should produce same gradient updates → same param values
    a_params = list(model.parameters())
    b_params = list(model_b.parameters())
    for pa, pb in zip(a_params, b_params):
        assert torch.allclose(pa, pb, atol=1e-6), "FSDPStrategy diverges from Phase-1 trainer"