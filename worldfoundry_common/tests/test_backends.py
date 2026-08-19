"""Unit tests for the multi-axis backend registry (worldfoundry_common.backends)."""
from __future__ import annotations

import pytest

from worldfoundry_common.backends import (
    AXES,
    backend_available,
    build_backend,
    compose,
    get_backend,
    list_backends,
    load_stack,
    resolve,
    resolve_config,
)
from worldfoundry_common.backends.registry import Backend
from worldfoundry_common.backends.sim_env import resolve_env_spec


def test_all_five_axes_registered() -> None:
    assert AXES == ("device", "inference", "model", "sim_env", "training")


def test_every_axis_has_backends() -> None:
    for axis in AXES:
        assert list_backends(axis), f"axis {axis!r} has no backends"


def test_get_unknown_backend_raises() -> None:
    with pytest.raises(KeyError):
        get_backend("device", "nope")


def test_get_unknown_axis_raises() -> None:
    with pytest.raises(KeyError):
        list_backends("nope")


def test_get_backend_returns_record() -> None:
    backend = get_backend("device", "cpu")
    assert isinstance(backend, Backend)
    assert backend.name == "cpu"
    assert backend.build() == "cpu"


def test_mock_sim_env_is_available_and_builds_factory() -> None:
    ok, hint = backend_available("sim_env", "mock")
    assert ok is True
    assert hint is None
    factory = build_backend("sim_env", "mock", obs_dim=4, action_dim=2)
    env = factory()
    obs, info = env.reset()
    assert obs.shape == (4,)
    obs, reward, term, trunc, _ = env.step([0.0, 0.0])
    assert obs.shape == (4,)
    assert isinstance(reward, float)


def test_self_training_backend_builds_runner() -> None:
    runner = build_backend("training", "self")
    assert callable(runner)


def test_unintegrated_frameworks_build_adapters() -> None:
    # verl / rlinf / sglang now build real lazy adapters (they never import the
    # third-party framework at build time), regardless of whether the package is
    # installed. Availability reflects whether the package is present.
    from worldfoundry_common.adapters.rlinf import RLinfRunner
    from worldfoundry_common.adapters.sglang import SGLangWorker
    from worldfoundry_common.adapters.verl import VerlRunner

    assert isinstance(build_backend("training", "verl"), VerlRunner)
    assert isinstance(build_backend("training", "rlinf"), RLinfRunner)
    assert isinstance(
        build_backend("inference", "sglang", model_path="fake"), SGLangWorker
    )
    # The package-presence checks must never raise, only report (ok, hint).
    for axis, name in (("training", "verl"), ("training", "rlinf"), ("inference", "sglang")):
        ok, hint = backend_available(axis, name)
        assert isinstance(ok, bool)
        assert hint is None or isinstance(hint, str)


def test_resolve_validates_whole_config() -> None:
    records = resolve(
        {"device": "cpu", "sim_env": "mock", "model": "self", "training": "self"}
    )
    assert records["device"].name == "cpu"
    assert records["sim_env"].name == "mock"
    with pytest.raises(KeyError):
        resolve({"device": "cpu", "sim_env": "does_not_exist"})


def test_model_aliases() -> None:
    assert get_backend("model", "tiny").name == "tiny"
    assert get_backend("model", "self").name == "self"
    assert get_backend("model", "hf_vjepa2").name == "hf_vjepa2"
    assert get_backend("model", "transformers").name == "transformers"


def test_native_training_backends() -> None:
    # pytorch aliases self; fsdp / mindspeed build parallel strategies.
    assert callable(build_backend("training", "pytorch"))
    assert "fsdp" in list_backends("training")
    assert "mindspeed" in list_backends("training")
    from worldfoundry_train.parallel.fsdp import FSDPStrategy
    from worldfoundry_train.parallel.mindspeed import MMStrategy

    assert isinstance(build_backend("training", "fsdp"), FSDPStrategy)
    assert isinstance(build_backend("training", "mindspeed"), MMStrategy)


def test_resolve_env_spec_aliases() -> None:
    assert resolve_env_spec("mock") == "worldfoundry_rl.env_mock:MockEnv"
    assert resolve_env_spec("franka_kitchen") == (
        "worldfoundry_rl.env_franka_kitchen:FrankaKitchenEnv"
    )
    # Full specs pass through unchanged.
    full = "worldfoundry_rl.env_manipulator:ManipulatorEnv"
    assert resolve_env_spec(full) == full
    with pytest.raises(KeyError):
        resolve_env_spec("does_not_exist")


def test_compose_builds_full_stack() -> None:
    stack = compose(
        {
            "device": "cpu",
            "model": "tiny",
            "sim_env": "mock",
            "inference": "torch",
            "training": "self",
        }
    )
    assert stack.device() == "cpu"
    # model build returns a tiny JEPA encoder module with .parameters().
    encoder = stack.model()
    assert hasattr(encoder, "parameters")
    # sim_env build returns an env factory that yields a working env.
    env = stack.env(obs_dim=4, action_dim=2)()
    obs, info = env.reset()
    assert obs.shape == (4,)
    # training build returns the RL runner entry.
    assert callable(stack.training())


def test_compose_validates_names() -> None:
    with pytest.raises(KeyError):
        compose({"device": "cpu", "sim_env": "nope"})


def test_resolve_config_fills_defaults() -> None:
    # A one-axis map keeps the other axes at their defaults.
    stack = resolve_config({"device": "cpu"})
    assert stack.device() == "cpu"
    assert stack.backends["sim_env"].name == "mock"
    assert stack.backends["training"].name == "self"
    assert stack.backends["inference"].name == "torch"
    assert stack.backends["model"].name == "self"


def test_resolve_config_none_and_empty_give_defaults() -> None:
    assert resolve_config().backends["device"].name == "auto"
    assert resolve_config({}).backends["sim_env"].name == "mock"


def test_resolve_config_ignores_none_values() -> None:
    stack = resolve_config({"device": "cpu", "sim_env": None})
    assert stack.device() == "cpu"
    assert stack.backends["sim_env"].name == "mock"


def test_resolve_config_validates_names() -> None:
    with pytest.raises(KeyError):
        resolve_config({"device": "tpu"})


def test_load_stack_reads_backends_block(tmp_path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "backends:\n"
        "  device: cpu\n"
        "  sim_env: mock\n"
        "training:\n"
        "  total_pool_steps: 10\n"
    )
    stack, full = load_stack(p)
    assert stack.device() == "cpu"
    assert stack.backends["sim_env"].name == "mock"
    # Full config is returned unchanged so callers keep their YAML.
    assert full["training"]["total_pool_steps"] == 10


def test_load_stack_missing_block_gives_defaults(tmp_path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text("env:\n  type: mock\n")
    stack, full = load_stack(p)
    assert stack.backends["device"].name == "auto"
    assert full["env"]["type"] == "mock"


# --- M2-T4: vllm_ascend alias + strengthened probe --------------------------


def test_vllm_ascend_listed_in_inference_axis() -> None:
    """Both ``vllm`` and ``vllm_ascend`` are registered on the inference axis (M2-T4)."""
    names = list_backends("inference")
    assert "vllm" in names
    assert "vllm_ascend" in names


def test_vllm_aliases_to_vllm_ascend() -> None:
    """``vllm`` and ``vllm_ascend`` share factory + available probe (backward-compat alias)."""
    vllm = get_backend("inference", "vllm")
    vllm_ascend = get_backend("inference", "vllm_ascend")
    assert vllm.build is vllm_ascend.build, "factory must be shared"
    # Both probes are the strengthened triple-import gate (one wraps the
    # other), so they MUST return identical availability on every host.
    assert vllm.available() is vllm_ascend.available() or (
        vllm.available() == vllm_ascend.available()
    )
    # The legacy ``vllm`` hint cross-references the canonical ``vllm_ascend``.
    assert vllm.hint is not None
    assert "vllm_ascend" in vllm.hint


def test_vllm_ascend_unavailable_on_host_without_wheels() -> None:
    """Host without ``torch_npu + vllm + vllm_ascend`` wheels reports ``(False, hint)``.

    Strengthened probe (M2-T4): even when ``worldfoundry_inf.workers.vllm_ascend_worker``
    is in-tree, missing any one of the three wheels flips availability to False.
    """
    ok, hint = backend_available("inference", "vllm_ascend")
    assert ok is False
    assert isinstance(hint, str)
    assert ".[inf]" in hint


def test_vllm_available_matches_vllm_ascend_outcome_on_host() -> None:
    """``vllm`` and ``vllm_ascend`` share availability outcome on every host.

    Both backends share the same probe (one wraps the other), so the
    orchestrator can pick either name and route identically.
    """
    ok_vllm, hint_vllm = backend_available("inference", "vllm")
    ok_ascend, hint_ascend = backend_available("inference", "vllm_ascend")
    assert ok_vllm == ok_ascend
    assert hint_vllm is not None and hint_ascend is not None
    assert "[inf]" in hint_vllm and "[inf]" in hint_ascend
