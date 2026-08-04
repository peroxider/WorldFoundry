# W35.3 spike: gymnasium CartPole-v1 vector adapter for RLinf EnvWorker.
# Mirrors the interface of rlinf.envs.embodichain.embodichain_env.EmbodiChainEnv
# so that EnvWorker / rollout / actor machinery works unchanged.
#
# Action contract: policy outputs continuous 1-dim action per chunk step;
# env-side threshold: a > 0 -> move right (1), else left (0).
#
# S4 (mode B): `pixel_obs: true` adds a "main_images" key to every obs dict —
# [num_envs, K, 3, S, S] float32 in [0,1], K stacked RGB frames rendered via
# gymnasium render_mode="rgb_array" and resized to S x S (maniskill key
# convention, see maniskill_env.py:155). "states" is always kept.

from collections import deque
from typing import Any, Optional, Union

import os

# Headless rendering (WSL/CPU workers have no display; pygame would otherwise
# fail to init video). Must be set before pygame is imported anywhere.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import gymnasium as gym
import numpy as np
import torch


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class GymnasiumCartPoleEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: Any,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info: Any,
    ):
        super().__init__()
        self.cfg = cfg
        self.seed_offset = int(seed_offset)
        self.total_num_processes = int(total_num_processes)
        self.worker_info = worker_info

        self.seed = int(_cfg_get(cfg, "seed", 0)) + self.seed_offset
        self.num_envs = int(num_envs)
        self.group_size = int(_cfg_get(cfg, "group_size", 1))
        self.num_group = self.num_envs // self.group_size
        self.auto_reset = bool(_cfg_get(cfg, "auto_reset", True))
        self.ignore_terminations = bool(_cfg_get(cfg, "ignore_terminations", False))
        self.max_episode_steps = int(_cfg_get(cfg, "max_episode_steps", 500))
        self.video_cfg = _cfg_get(cfg, "video_cfg", None)
        self.gym_id = str(_cfg_get(cfg, "gym_id", "CartPole-v1"))

        # S4 mode B: pixel observations
        self.pixel_obs = bool(_cfg_get(cfg, "pixel_obs", False))
        self.pixel_size = int(_cfg_get(cfg, "pixel_size", 64))
        self.num_frames = int(_cfg_get(cfg, "num_frames", 2))

        self._device = torch.device("cpu")
        self._is_start = True

        if self.pixel_obs:
            self._envs = [
                gym.make(self.gym_id, render_mode="rgb_array")
                for _ in range(self.num_envs)
            ]
            self._frame_stacks: list[deque] = [
                deque(maxlen=self.num_frames) for _ in range(self.num_envs)
            ]
        else:
            self._envs = [gym.make(self.gym_id) for _ in range(self.num_envs)]
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self.observation_space = self._envs[0].observation_space

        self.prev_step_reward = torch.zeros(self.num_envs, dtype=torch.float32)
        self._elapsed_steps = torch.zeros(self.num_envs, dtype=torch.int32)
        self._init_metrics()

    # ---- properties expected by EnvWorker ----
    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def elapsed_steps(self) -> torch.Tensor:
        return self._elapsed_steps.to(self.device)

    @property
    def is_start(self) -> bool:
        return self._is_start

    @is_start.setter
    def is_start(self, value: bool) -> None:
        self._is_start = value

    @property
    def info_logging_keys(self) -> list[str]:
        return []

    # ---- metrics (mirrors EmbodiChainEnv) ----
    def _init_metrics(self) -> None:
        self.success_once = torch.zeros(self.num_envs, dtype=torch.bool)
        self.fail_once = torch.zeros(self.num_envs, dtype=torch.bool)
        self.returns = torch.zeros(self.num_envs, dtype=torch.float32)

    def _reset_metrics(self, env_idx: Optional[torch.Tensor] = None) -> None:
        if env_idx is None:
            self.prev_step_reward.zero_()
            self.success_once.zero_()
            self.fail_once.zero_()
            self.returns.zero_()
            self._elapsed_steps.zero_()
            return
        env_idx = torch.as_tensor(env_idx, dtype=torch.long)
        self.prev_step_reward[env_idx] = 0.0
        self.success_once[env_idx] = False
        self.fail_once[env_idx] = False
        self.returns[env_idx] = 0.0
        self._elapsed_steps[env_idx] = 0

    def _record_metrics(
        self, step_reward: torch.Tensor, infos: dict[str, Any]
    ) -> dict[str, Any]:
        episode_info: dict[str, Any] = {}
        self.returns += step_reward
        episode_info["return"] = self.returns.clone()
        episode_len = self.elapsed_steps
        episode_info["episode_len"] = episode_len.clone()
        denom = torch.clamp(episode_len.float(), min=1.0)
        episode_info["reward"] = episode_info["return"] / denom
        infos["episode"] = episode_info
        return infos

    # ---- pixel helpers (S4) ----
    def _render_frame(self, env_idx: int) -> np.ndarray:
        """Render env i -> float32 CHW frame resized to pixel_size, in [0,1]."""
        from PIL import Image

        frame = self._envs[env_idx].render()  # [H, W, 3] uint8
        frame = np.asarray(
            Image.fromarray(frame).resize(
                (self.pixel_size, self.pixel_size), Image.BILINEAR
            )
        )
        return frame.astype(np.float32).transpose(2, 0, 1) / 255.0

    def _push_frame(self, env_idx: int, fill: bool = False) -> None:
        frame = self._render_frame(env_idx)
        stack = self._frame_stacks[env_idx]
        if fill:
            stack.clear()
            for _ in range(self.num_frames):
                stack.append(frame)
        else:
            stack.append(frame)

    def _stacked_frames(self, env_idx: int) -> np.ndarray:
        # [K, 3, S, S]; left-pad with oldest frame if stack is short
        stack = list(self._frame_stacks[env_idx])
        while len(stack) < self.num_frames:
            stack.insert(0, stack[0])
        return np.stack(stack[-self.num_frames :], axis=0)

    # ---- obs / info wrapping ----
    def _wrap_obs(
        self,
        obs_batch: list[np.ndarray],
        frame_override: Optional[list[np.ndarray]] = None,
    ) -> dict[str, torch.Tensor]:
        states = torch.as_tensor(
            np.stack(obs_batch), dtype=torch.float32, device=self.device
        )
        obs = {"states": states.reshape(self.num_envs, -1)}
        if self.pixel_obs:
            if frame_override is not None:
                frames = np.stack(frame_override, axis=0)
            else:
                frames = np.stack(
                    [self._stacked_frames(i) for i in range(self.num_envs)], axis=0
                )
            obs["main_images"] = torch.as_tensor(
                frames, dtype=torch.float32, device=self.device
            )
        return obs

    # ---- gym API used by EnvWorker ----
    def reset(
        self,
        *,
        seed: Optional[Union[int, list[int]]] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        options = {} if options is None else dict(options)
        env_idx = options.pop("env_idx", None)
        obs_batch: list[np.ndarray] = [None] * self.num_envs
        if env_idx is None:
            for i, env in enumerate(self._envs):
                obs, _ = env.reset(
                    seed=self.seed + i if seed is None else int(seed) + i
                )
                obs_batch[i] = obs
                if self.pixel_obs:
                    self._push_frame(i, fill=True)
            self._reset_metrics()
        else:
            reset_ids = torch.as_tensor(env_idx, dtype=torch.long).tolist()
            for i in reset_ids:
                obs, _ = self._envs[int(i)].reset(seed=self.seed + int(i))
                obs_batch[int(i)] = obs
                if self.pixel_obs:
                    self._push_frame(int(i), fill=True)
            # fill non-reset envs by reusing current obs
            for i in range(self.num_envs):
                if obs_batch[i] is None:
                    obs_batch[i] = self._last_obs[i]
            self._reset_metrics(torch.as_tensor(reset_ids, dtype=torch.long))
        self._last_obs = obs_batch
        self._is_start = True
        return self._wrap_obs(obs_batch), {}

    def step(
        self, actions: Union[np.ndarray, torch.Tensor]
    ) -> tuple[
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        action_tensor = (
            actions.to(self.device, dtype=torch.float32)
            if isinstance(actions, torch.Tensor)
            else torch.as_tensor(actions, dtype=torch.float32)
        )
        if action_tensor.ndim == 1:
            action_tensor = action_tensor.unsqueeze(0).repeat(self.num_envs, 1)
        action_tensor = action_tensor.reshape(self.num_envs, -1)

        obs_batch: list[np.ndarray] = []
        rewards = torch.zeros(self.num_envs, dtype=torch.float32)
        terminations = torch.zeros(self.num_envs, dtype=torch.bool)
        truncations = torch.zeros(self.num_envs, dtype=torch.bool)
        final_obs: dict[int, np.ndarray] = {}
        final_frames: dict[int, np.ndarray] = {}

        for i, env in enumerate(self._envs):
            discrete_action = 1 if float(action_tensor[i, 0]) > 0.0 else 0
            obs, reward, term, trunc, _info = env.step(discrete_action)
            if self.pixel_obs:
                self._push_frame(i)
            if term or trunc:
                final_obs[i] = obs
                if self.pixel_obs:
                    final_frames[i] = self._stacked_frames(i)
                if self.auto_reset:
                    obs, _ = env.reset()
                    if self.pixel_obs:
                        self._push_frame(i, fill=True)
            obs_batch.append(obs)
            rewards[i] = float(reward)
            terminations[i] = bool(term)
            truncations[i] = bool(trunc)

        self._last_obs = obs_batch
        self._elapsed_steps += 1

        infos: dict[str, Any] = {}
        infos = self._record_metrics(rewards, infos)
        if self.ignore_terminations:
            terminations = torch.zeros_like(terminations)
        dones = torch.logical_or(terminations, truncations)
        if dones.any() and self.auto_reset:
            done_ids = torch.nonzero(dones, as_tuple=False).flatten()
            # final obs/info for bootstrap values
            frame_override = None
            if self.pixel_obs:
                frame_override = [
                    final_frames.get(int(i), self._stacked_frames(int(i)))
                    for i in range(self.num_envs)
                ]
            final_obs_wrapped = self._wrap_obs(
                [final_obs.get(int(i), self._last_obs[int(i)]) for i in range(self.num_envs)],
                frame_override=frame_override,
            )
            infos["final_observation"] = final_obs_wrapped
            # EnvWorker expects final_info["episode"][key] to be per-env
            # tensors indexable by the done mask (env_worker.py:438).
            infos["final_info"] = {
                "episode": dict(infos.get("episode", {}))
            }
            infos["_final_info"] = dones
            infos["_final_observation"] = dones
            infos["_elapsed_steps"] = dones
            self._reset_metrics(done_ids)
        self._is_start = False
        return (
            self._wrap_obs(obs_batch),
            rewards,
            terminations,
            truncations,
            infos,
        )

    def chunk_step(self, chunk_actions: Union[np.ndarray, torch.Tensor]):
        chunk_actions = (
            chunk_actions.to(self.device, dtype=torch.float32)
            if isinstance(chunk_actions, torch.Tensor)
            else torch.as_tensor(chunk_actions, dtype=torch.float32)
        )
        if chunk_actions.ndim != 3:
            raise ValueError(
                "chunk_actions must have shape [num_envs, chunk_steps, action_dim], "
                f"got {tuple(chunk_actions.shape)}."
            )

        obs_list, infos_list = [], []
        chunk_rewards, raw_terms, raw_truncs = [], [], []

        for step_idx in range(int(chunk_actions.shape[1])):
            obs, rewards, terminations, truncations, infos = self.step(
                chunk_actions[:, step_idx]
            )
            obs_list.append(obs)
            infos_list.append(infos)
            chunk_rewards.append(rewards)
            raw_terms.append(terminations)
            raw_truncs.append(truncations)

        chunk_rewards_t = torch.stack(chunk_rewards, dim=1)
        raw_terms_t = torch.stack(raw_terms, dim=1)
        raw_truncs_t = torch.stack(raw_truncs, dim=1)

        past_terminations = raw_terms_t.any(dim=1)
        past_truncations = raw_truncs_t.any(dim=1)

        chunk_terminations = torch.zeros_like(raw_terms_t)
        chunk_terminations[:, -1] = past_terminations
        chunk_truncations = torch.zeros_like(raw_truncs_t)
        chunk_truncations[:, -1] = past_truncations

        return (
            obs_list,
            chunk_rewards_t,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    def update_reset_state_ids(self):
        return None

    def sample_action_space(self) -> torch.Tensor:
        return torch.as_tensor(
            self.action_space.sample(), dtype=torch.float32, device=self.device
        )

    def close(self):
        for env in self._envs:
            try:
                env.close()
            except Exception:
                pass
