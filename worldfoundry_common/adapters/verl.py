"""verl framework runner/worker adapters (verl 0.5.0, Ascend NPU baseline).

Wraps verl's programmatic entry point behind WorldFoundry's adapter contracts:

* :class:`VerlRunner` — drives verl 0.5.0's RL loop via the documented
  programmatic entry ``verl.trainer.main_ppo.run_ppo(config)`` (there is no
  ``verl.interface.Main`` in verl 0.5.0). The adapter assembles the OmegaConf
  three-part config (``algorithm`` / ``data`` / ``actor_rollout_ref`` /
  ``trainer`` / ``critic`` / ``reward_model``) and applies the Ascend NPU
  defaults required by the baseline: ``trainer.device=npu``, an
  ``vllm-ascend`` rollout backend (``rollout.name=vllm``), PyTorch FSDP
  training, and the HCCL/Ray NPU env vars.
* :class:`VerlRolloutWorker` — wraps verl's Ray rollout worker behind the
  :class:`~worldfoundry_common.adapters.base.RolloutWorker` surface
  (``generate_sequences`` / ``update_weights`` / ``shutdown``).

verl is declared under the ``rl`` extra (``verl==0.5.0``, see pyproject). Both
adapters import ``verl`` lazily inside their methods so this module imports
cleanly when verl is not installed.
"""
from __future__ import annotations

import os
from typing import Any

from worldfoundry_common.adapters.base import TrainingRunner
from worldfoundry_common.npu_env import is_npu_available, setup_npu_env


def _require_verl(component: str) -> Any:
    """Import the verl package, raising an actionable error if absent."""
    try:
        import verl  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ImportError(
            f"verl is required to build the {component}. "
            "Run: pip install -e '.[rl]'"
        ) from exc
    return verl


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (nested dicts merge)."""
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


# Minimal verl 0.5.0 PPO config keys required for a run (see
# verl/trainer/config/ppo_trainer.yaml + Ascend quick start).
def build_npu_config(
    *,
    model_path: str,
    train_files: str,
    val_files: str,
    train_batch_size: int = 128,
    max_prompt_length: int = 512,
    max_response_length: int = 512,
    ppo_mini_batch_size: int = 64,
    ppo_micro_batch_size: int = 8,
    rollout_logprob_micro_batch_size: int = 8,
    ref_logprob_micro_batch_size: int = 8,
    tensor_model_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.5,
    kl_coef: float = 0.01,
    n_gpus_per_node: int = 1,
    nnodes: int = 1,
    total_epochs: int = 1,
    total_training_steps: int | None = None,
    logger: str = "console",
    override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a minimal, NPU-ready verl 0.5.0 OmegaConf config (as a dict).

    The returned dict follows verl's ``ppo_trainer.yaml`` structure. It is
    NPU-first: ``trainer.device=npu``, rollout backend ``vllm`` (vLLM-Ascend),
    and PyTorch FSDP actor/ref strategy. Pass ``override`` to tune any key;
    nested keys are deep-merged.
    """
    base: dict[str, Any] = {
        "algorithm": {
            "adv_estimator": "gae",
            "kl_ctrl": {"kl_coef": kl_coef, "type": "fixed"},
        },
        "data": {
            "train_files": train_files,
            "val_files": val_files,
            "train_batch_size": train_batch_size,
            "max_prompt_length": max_prompt_length,
            "max_response_length": max_response_length,
            "truncation": "left",
        },
        "actor_rollout_ref": {
            "hybrid_engine": True,
            "model": {"path": model_path},
            "actor": {
                "strategy": "fsdp",
                "optim": {"lr": 1e-6, "type": "adamw"},
                "ppo_mini_batch_size": ppo_mini_batch_size,
                "ppo_micro_batch_size_per_gpu": ppo_micro_batch_size,
                "fsdp_config": {
                    "param_dtype": "bf16",
                    "reduce_dtype": "bf16",
                    "auto_wrap": "size_based",
                },
            },
            "rollout": {
                "name": "vllm",  # vLLM-Ascend NPU path
                "log_prob_micro_batch_size_per_gpu": rollout_logprob_micro_batch_size,
                "tensor_model_parallel_size": tensor_model_parallel_size,
                "gpu_memory_utilization": gpu_memory_utilization,
                "temperature": 1.0,
                "top_k": -1,
                "top_p": 1.0,
                "max_num_seqs": 256,
                "engine_kwargs": {
                    "vllm": {
                        "enforce_eager": True,
                        "device": "npu",
                    }
                },
            },
            "ref": {
                "strategy": "fsdp",
                "log_prob_micro_batch_size_per_gpu": ref_logprob_micro_batch_size,
                "fsdp_config": {
                    "param_dtype": "bf16",
                    "reduce_dtype": "bf16",
                    "auto_wrap": "size_based",
                },
            },
        },
        "critic": {
            "strategy": "fsdp",
            "model": {"path": model_path},
            "optim": {"lr": 1e-6, "type": "adamw"},
            "ppo_micro_batch_size_per_gpu": ppo_micro_batch_size,
            "fsdp_config": {
                "param_dtype": "bf16",
                "reduce_dtype": "bf16",
                "auto_wrap": "size_based",
            },
        },
        "reward_model": {"model": {"path": model_path}},
        "trainer": {
            "logger": logger,
            "device": "npu" if is_npu_available() else "cuda",
            "n_gpus_per_node": n_gpus_per_node,
            "nnodes": nnodes,
            "save_freq": 100,
            "val_before_train": False,
            "val_only": False,
            "total_epochs": total_epochs,
            "default_hdfs_dir": None,
        },
    }
    if total_training_steps is not None:
        base["trainer"]["total_training_steps"] = int(total_training_steps)

    if override:
        base = _deep_merge(base, override)
    return base


class VerlRunner(TrainingRunner):
    """Adapter that runs verl 0.5.0's RL loop through ``run_ppo``.

    verl 0.5.0 has no ``verl.interface.Main``; its programmatic entry is
    ``verl.trainer.main_ppo.run_ppo(config)`` (Hydra's ``main`` is just a thin
    CLI wrapper over it). On an NPU host the adapter exports the HCCL/Ray env
    vars and defaults ``trainer.device`` to ``npu`` with an ``vllm-ascend``
    rollout backend.

    Args:
        config: optional default config dict (OmegaConf-able, see
            :func:`build_npu_config`). May also be passed to :meth:`run`.
    """

    def __init__(self, config: dict | None = None) -> None:
        self._config: dict = config or {}

    def run(self, config: dict | None = None, **kwargs: Any) -> dict[str, Any]:
        """Run verl 0.5.0's PPO loop for the given config.

        ``config`` overrides the constructor default when provided. On NPU the
        process-level env vars are set first, then ``run_ppo`` is invoked
        in-process. Returns a summary dict (verl runs its own full loop).
        """
        cfg = config if config is not None else self._config
        _require_verl("verl runner")

        if is_npu_available():
            setup_npu_env()
            os.environ.setdefault("WORLD_SIZE", "1")

        from verl.trainer.main_ppo import run_ppo  # type: ignore[import-not-found]

        run_ppo(cfg)
        return {"framework": "verl", "status": "completed"}


class VerlRolloutWorker:
    """Adapter that drives verl's Ray rollout backend.

    Implements the :class:`~worldfoundry_common.adapters.base.RolloutWorker`
    surface (``generate_sequences`` / ``update_weights`` / ``shutdown``). In
    verl 0.5.0 the rollout worker owns ``generate_sequences``; weight sync and
    resource release are proxied to the underlying Ray rollout object.
    """

    def __init__(self, config: dict | None = None) -> None:
        self._config: dict = config or {}
        self._rollout: Any = None

    def _ensure_rollout(self) -> Any:
        if self._rollout is None:
            _require_verl("verl rollout worker")
            from verl.workers.rollout.ray_rollout import RayRollout  # type: ignore[import-not-found]

            self._rollout = RayRollout(self._config)
        return self._rollout

    def generate_sequences(self, prompts: Any, **kwargs: Any) -> Any:
        """Generate sequences for a ``verl.protocol.DataProto`` prompt batch."""
        return self._ensure_rollout().generate_sequences(prompts, **kwargs)

    def update_weights(self, state_dict: dict[str, Any]) -> None:
        """Sync the latest actor weights into the rollout workers."""
        return self._ensure_rollout().update_weights(state_dict)

    def shutdown(self) -> None:
        """Release the rollout worker resources."""
        rollout = self._rollout
        self._rollout = None
        if rollout is not None and hasattr(rollout, "shutdown"):
            rollout.shutdown()


__all__ = ["VerlRunner", "VerlRolloutWorker", "build_npu_config"]
