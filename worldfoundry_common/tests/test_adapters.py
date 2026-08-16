"""Unit tests for the external-framework adapters (verl / rlinf / sglang).

These adapters never import their third-party framework at import/build time;
each test stubs the framework's classes via ``sys.modules`` (or injects a
factory) to verify the wiring without the real dependency installed.
"""
from __future__ import annotations

import asyncio
import sys
import types

import pytest

from worldfoundry_common.adapters import (
    RLinfEnvWorker,
    RLinfRunner,
    RLinfRunnerConfig,
    RunnerFactory,
    SGLangWorker,
    VerlRolloutWorker,
    VerlRunner,
    build_npu_config,
)
from worldfoundry_common.adapters.base import TrainingRunner


# --------------------------------------------------------------------------- #
# Import-time laziness
# --------------------------------------------------------------------------- #
def test_adapter_modules_import_without_frameworks() -> None:
    # Importing the package (and its submodules) must not require verl/sglang.
    # (rlinf is checked separately because a later "requires package" test may
    # legitimately import it if it is installed on this box.)
    from worldfoundry_common import adapters  # noqa: F401

    assert "verl" not in sys.modules
    assert "sglang" not in sys.modules


def test_all_adapters_are_trainingrunner_contract() -> None:
    assert isinstance(VerlRunner(), TrainingRunner)
    assert isinstance(RLinfRunner(), TrainingRunner)


# --------------------------------------------------------------------------- #
# verl
# --------------------------------------------------------------------------- #
def _install_fake_verl(run_ppo=None, ray_rollout_cls=None) -> None:
    """Register a fake ``verl`` package (+ submodules) in sys.modules."""
    verl = types.ModuleType("verl")
    verl.__path__ = []
    sys.modules["verl"] = verl

    if run_ppo is not None:
        trainer = types.ModuleType("verl.trainer")
        trainer.__path__ = []
        main_ppo = types.ModuleType("verl.trainer.main_ppo")
        main_ppo.run_ppo = run_ppo
        sys.modules["verl.trainer"] = trainer
        sys.modules["verl.trainer.main_ppo"] = main_ppo

    if ray_rollout_cls is not None:
        base = types.ModuleType("verl.workers")
        base.__path__ = []
        rollout = types.ModuleType("verl.workers.rollout")
        rollout.__path__ = []
        ray = types.ModuleType("verl.workers.rollout.ray_rollout")
        ray.RayRollout = ray_rollout_cls
        sys.modules["verl.workers"] = base
        sys.modules["verl.workers.rollout"] = rollout
        sys.modules["verl.workers.rollout.ray_rollout"] = ray


def test_verl_runner_run(monkeypatch) -> None:
    called = {}

    def fake_run_ppo(cfg):
        called["cfg"] = cfg

    _install_fake_verl(run_ppo=fake_run_ppo)
    try:
        runner = VerlRunner(config={"data": {}})
        result = runner.run()
        assert result == {"framework": "verl", "status": "completed"}
        # run_ppo (not a non-existent verl.interface.Main) receives the config.
        assert called["cfg"] == {"data": {}}
    finally:
        for name in (
            "verl",
            "verl.trainer",
            "verl.trainer.main_ppo",
        ):
            sys.modules.pop(name, None)


def test_verl_build_npu_config() -> None:
    cfg = build_npu_config(
        model_path="Qwen/Qwen2.5-0.5B",
        train_files="train.parquet",
        val_files="val.parquet",
    )
    # NPU-first baseline: vLLM-Ascend rollout + FSDP training.
    assert cfg["actor_rollout_ref"]["rollout"]["name"] == "vllm"
    assert cfg["actor_rollout_ref"]["actor"]["strategy"] == "fsdp"
    assert cfg["trainer"]["device"] in ("npu", "cuda")
    assert cfg["data"]["train_files"] == "train.parquet"
    # Deep-merge of user override.
    cfg2 = build_npu_config(
        model_path="m", train_files="t", val_files="v",
        override={"data": {"train_batch_size": 32}},
    )
    assert cfg2["data"]["train_batch_size"] == 32


def test_verl_runner_requires_package(monkeypatch) -> None:
    # With verl absent, run() raises an actionable ImportError.
    monkeypatch.delitem(sys.modules, "verl", raising=False)
    with pytest.raises(ImportError, match="verl is required"):
        VerlRunner().run({"data": {}})


def test_verl_rollout_worker(monkeypatch) -> None:
    class FakeRayRollout:
        def __init__(self, config):
            self.config = config

        def generate_sequences(self, prompts, **kw):
            return {"prompts": prompts, **kw}

        def update_weights(self, sd):
            self.weights = sd

        def shutdown(self):
            self.closed = True

    _install_fake_verl(ray_rollout_cls=FakeRayRollout)
    try:
        worker = VerlRolloutWorker(config={"rollout": {"name": "hf"}})
        out = worker.generate_sequences("hi", n=2)
        assert out == {"prompts": "hi", "n": 2}
        worker.update_weights({"w": 1})
        assert worker._rollout.weights == {"w": 1}
        rr = worker._rollout
        worker.shutdown()
        assert rr.closed is True
    finally:
        for name in (
            "verl",
            "verl.interface",
            "verl.workers",
            "verl.workers.rollout",
            "verl.workers.rollout.ray_rollout",
        ):
            sys.modules.pop(name, None)


# --------------------------------------------------------------------------- #
# rlinf
# --------------------------------------------------------------------------- #
def test_rlinf_runner_with_factory() -> None:
    built = {}

    def factory(cfg):
        built["cfg"] = cfg
        return _FakeRunner()

    class _FakeRunner:
        def run(self):
            self.ran = True

    runner = RLinfRunner(config={"env": {}}, runner_factory=factory)
    result = runner.run()
    assert built["cfg"] == {"env": {}}
    assert result == {"framework": "rlinf", "status": "completed"}


def test_rlinf_runner_requires_factory_or_package() -> None:
    runner = RLinfRunner(config={"env": {}})
    with pytest.raises((ImportError, NotImplementedError)):
        runner.run()


def test_rlinf_env_worker_rolls_out_worldfoundry_env() -> None:
    from worldfoundry_common.backends import build_backend

    factory = build_backend("sim_env", "mock", obs_dim=4, action_dim=2)
    worker = RLinfEnvWorker(factory, num_action_chunks=2)
    obs, info = worker.reset()
    assert obs.shape == (4,)
    # chunk_step returns (obs_list, rewards, terms, truncs, infos_list) — one per chunk.
    obs_list, rewards, terms, truncs, infos = worker.chunk_step(
        [[0.0, 0.0], [0.0, 0.0]]
    )
    assert len(obs_list) == 2
    assert len(rewards) == 2
    assert len(terms) == 2
    assert len(truncs) == 2
    assert len(infos) == 2
    worker.close()


# --------------------------------------------------------------------------- #
# M1-T1: RLinfRunnerConfig + RunnerFactory Protocol
# --------------------------------------------------------------------------- #
def test_rlinf_runner_config_defaults_and_to_dict() -> None:
    """Defaults match the TODO M1-T1 spec; to_dict mirrors the dataclass."""
    cfg = RLinfRunnerConfig()
    assert cfg.cluster_role == "head"
    assert cfg.worker_groups == {}
    assert cfg.world_model_builder is None
    assert cfg.extras == {}
    rendered = cfg.to_dict()
    assert set(rendered.keys()) == {
        "cluster_role", "worker_groups", "world_model_builder", "extras",
    }
    assert rendered["cluster_role"] == "head"
    # to_dict copies so downstream mutation doesn't leak back.
    rendered["worker_groups"]["actor"] = "MyActor"
    assert cfg.worker_groups == {}


def test_rlinf_runner_config_accepts_world_model_builder() -> None:
    """world_model_builder is the seam for M1-T2 WorldModelActor/Critic."""

    def builder():
        return ("fake_world_model", {"dim": 64})

    cfg = RLinfRunnerConfig(
        cluster_role="worker",
        worker_groups={"actor": "MyActor", "rollout": "MyRollout"},
        world_model_builder=builder,
        extras={"num_outer_iters": 1},
    )
    assert cfg.cluster_role == "worker"
    assert cfg.worker_groups["actor"] == "MyActor"
    assert cfg.world_model_builder() == ("fake_world_model", {"dim": 64})
    assert cfg.extras["num_outer_iters"] == 1


def test_rlinf_runner_factory_protocol_is_runtime_checkable() -> None:
    """A plain function with the right signature passes isinstance() checks."""

    # Body is never invoked; we only assert the runtime-checkable Protocol
    # matches the factory's __call__ shape. Keeping the body minimal avoids
    # pulling in test-only class scaffolding.
    def factory(config):
        return None

    assert isinstance(factory, RunnerFactory)


def test_rlinf_runner_with_typed_config_factory() -> None:
    """A factory built from RLinfRunnerConfig produces a runner that runs."""
    seen: list[RLinfRunnerConfig] = []

    class _FakeRunner:
        def __init__(self):
            self.ran = False

        def run(self):
            self.ran = True

    def factory(config):
        # Normalize both forms (typed dataclass and legacy dict) to the
        # dataclass before delegating to the real factory body.
        if isinstance(config, dict):
            config = RLinfRunnerConfig(**config)
        seen.append(config)
        return _FakeRunner()

    cfg = RLinfRunnerConfig(
        cluster_role="worker",
        worker_groups={"actor": "MyActor"},
        world_model_builder=lambda: ("wm", None),
    )
    runner = RLinfRunner(config=cfg, runner_factory=factory)
    result = runner.run()
    assert result == {"framework": "rlinf", "status": "completed"}
    assert len(seen) == 1
    assert seen[0].cluster_role == "worker"
    assert seen[0].worker_groups["actor"] == "MyActor"


# --------------------------------------------------------------------------- #
# sglang
# --------------------------------------------------------------------------- #
def test_sglang_worker_builds_engine_lazily(monkeypatch) -> None:
    from dataclasses import dataclass

    @dataclass
    class FakeServerArgs:
        model_path: str
        tp_size: int = 1
        dtype: str = "bfloat16"
        skip_tokenizer_init: bool = False
        device: str = "cuda"
        attention_backend: str = "triton"

    class FakeEngine:
        def __init__(self, server_args=None, **kwargs):
            self.server_args = server_args
            self.tokenizer_manager = _FakeTokenizerManager()

        async def async_generate(self, **kw):
            return {"gen": kw}

        async def async_encode(self, **kw):
            return {"enc": kw}

        def shutdown(self):
            self.closed = True

    class _FakeTokenizerManager:
        def sync_hf_weight(self, sd):
            self.sd = sd

    sglang_pkg = types.ModuleType("sglang")
    srt = types.ModuleType("sglang.srt")
    srt.__path__ = []
    entry = types.ModuleType("sglang.srt.entrypoints")
    entry.__path__ = []
    engine_mod = types.ModuleType("sglang.srt.entrypoints.engine")
    engine_mod.Engine = FakeEngine
    args_mod = types.ModuleType("sglang.srt.server_args")
    args_mod.ServerArgs = FakeServerArgs
    sys.modules["sglang"] = sglang_pkg
    sys.modules["sglang.srt"] = srt
    sys.modules["sglang.srt.entrypoints"] = entry
    sys.modules["sglang.srt.entrypoints.engine"] = engine_mod
    sys.modules["sglang.srt.server_args"] = args_mod
    try:
        worker = SGLangWorker("model", tp_size=2)
        # Engine is built lazily on first use.
        assert worker._engine is None
        result = asyncio.run(worker.generate_sequences(["hi"], sampling_params={}))
        assert result == {"gen": {"prompt": ["hi"], "sampling_params": {}}}
        assert worker._engine is not None
        worker.update_weights({"w": 1})
        assert worker._engine.tokenizer_manager.sd == {"w": 1}
        engine = worker._engine
        worker.shutdown()
        assert engine.closed is True
    finally:
        for name in (
            "sglang",
            "sglang.srt",
            "sglang.srt.entrypoints",
            "sglang.srt.entrypoints.engine",
            "sglang.srt.server_args",
        ):
            sys.modules.pop(name, None)


def test_sglang_worker_requires_package(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "sglang", raising=False)
    with pytest.raises(ImportError, match="sglang is required"):
        SGLangWorker("model").generate_sequences(["hi"])
