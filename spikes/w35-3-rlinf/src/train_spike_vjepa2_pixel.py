# W35.3 S2 spike entry: gymnasium CartPole + VJEPA2Policy (mode A) + PPO on CPU.
# Same as train_spike.py plus VJEPA2Policy registration before validate_cfg
# (config.py:823 asserts model_type in EMBODIED_MODEL). Worker processes get the
# registration through RLINF_EXT_MODULE=spike_ext.rlinf_ext (official channel).

import json
import os
import sys

# Ensure spike root is importable (spike_ext package + sitecustomize patch).
_SPIKE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SPIKE_ROOT not in sys.path:
    sys.path.insert(0, _SPIKE_ROOT)

# In-process patch (sitecustomize covers spawned/ray subprocesses).
import rlinf.envs as _rlinf_envs

_orig_get_env_cls = _rlinf_envs.get_env_cls


def _spike_get_env_cls(env_type, env_cfg=None):
    if env_type == "gym_cartpole":
        from spike_ext.gym_cartpole_env import GymnasiumCartPoleEnv

        return GymnasiumCartPoleEnv
    return _orig_get_env_cls(env_type, env_cfg)


_rlinf_envs.get_env_cls = _spike_get_env_cls

# Tolerate the custom env type in rlinf.config validation.
import rlinf.config as _rlinf_config

_orig_supported_env_type = _rlinf_config.SupportedEnvType


class _SupportedEnvTypeShim:
    def __call__(self, value):
        try:
            return _orig_supported_env_type(value)
        except ValueError:
            if value == "gym_cartpole":
                return "gym_cartpole"
            raise

    def __getattr__(self, name):
        return getattr(_orig_supported_env_type, name)


_rlinf_config.SupportedEnvType = _SupportedEnvTypeShim()

# Register VJEPA2Policy in the driver before validate_cfg.
import spike_ext.rlinf_ext as _rlinf_ext

_rlinf_ext.register()

import hydra
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

mp.set_start_method("spawn", force=True)


@hydra.main(
    version_base="1.1", config_path="../config", config_name="gym_ppo_cart_pole_vjepa2_pixel"
)
def main(cfg) -> None:
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(
        cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path
    )
    component_placement = HybridComponentPlacement(cfg, cluster)

    actor_placement = component_placement.get_strategy("actor")

    from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor

    actor_worker_cls = EmbodiedFSDPActor

    actor_group = actor_worker_cls.create_group(cfg).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
    )

    rollout_placement = component_placement.get_strategy("rollout")
    rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
        cluster, name=cfg.rollout.group_name, placement_strategy=rollout_placement
    )

    env_placement = component_placement.get_strategy("env")
    env_group = EnvWorker.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )

    runner = EmbodiedRunner(
        cfg=cfg,
        actor=actor_group,
        rollout=rollout_group,
        env=env_group,
        reward=None,
    )

    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
