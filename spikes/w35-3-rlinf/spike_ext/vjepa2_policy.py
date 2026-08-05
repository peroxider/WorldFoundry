# W35.3 S2 spike: VJEPA2Policy — random-init small V-JEPA 2 encoder (mode A:
# state projection to pseudo token sequence) + linear action head + value head.
#
# Contract mirrors rlinf/models/embodiment/mlp_policy/mlp_policy.py exactly:
#   - default_forward(forward_inputs, compute_logprobs, compute_entropy,
#     compute_values) -> dict{logprobs, entropy, values}
#   - predict_action_batch(env_obs, calculate_logprobs, calculate_values,
#     return_obs, mode) -> (chunk_actions, result{prev_logprobs, prev_values,
#     forward_inputs})   # all kwargs carry defaults (rollout omits `mode`
#                        # for unlisted model types)
#   - value head MUST be named `value_head`: the FSDP optimizer splits param
#     groups by the "value_head" name substring (fsdp_model_manager.py:516).

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.modules.utils import layer_init


class VJEPA2Policy(nn.Module, BasePolicy):
    def __init__(
        self,
        obs_dim,
        action_dim,
        num_action_chunks,
        add_value_head,
        hidden_size=384,
        num_hidden_layers=4,
        num_attention_heads=6,
        num_pseudo_tokens=8,
        freeze_encoder=True,
        mlp_ratio=4.0,
        obs_mode="state",
        crop_size=64,
        patch_size=16,
        frames_per_clip=2,
        tubelet_size=2,
    ):
        super().__init__()
        from transformers.models.vjepa2.configuration_vjepa2 import VJEPA2Config
        from transformers.models.vjepa2.modeling_vjepa2 import VJEPA2Encoder

        self.obs_mode = obs_mode
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_action_chunks = num_action_chunks
        self.hidden_size = hidden_size
        self.num_pseudo_tokens = num_pseudo_tokens
        self.torch_compile_enabled = False
        # mirror MLPPolicy defaults (PPO path)
        self.independent_std = True
        self.final_tanh = False
        self.action_scale = None
        self.cuda_graph_manager = None

        # Random-init small V-JEPA 2 encoder built from VJEPA2Config (plan §3.3).
        # Mode A (state): crop_size/patch_size/frames_per_clip/tubelet_size only
        # shape the RoPE position decomposition; pseudo tokens are fed straight
        # into the transformer blocks (VJEPA2Embeddings 3D patch conv bypassed).
        # Mode B (pixel): full encoder forward over pixel_values_videos
        # [B, K, 3, S, S]; frames_per_clip/tubelet_size must match the frame
        # stack K, crop_size/patch_size the rendered frame size.
        if obs_mode == "pixel":
            config = VJEPA2Config(
                patch_size=patch_size,
                crop_size=crop_size,
                frames_per_clip=frames_per_clip,
                tubelet_size=tubelet_size,
                in_chans=3,
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_hidden_layers=num_hidden_layers,
                mlp_ratio=mlp_ratio,
            )
        else:
            config = VJEPA2Config(
                patch_size=16,
                crop_size=32,
                frames_per_clip=4,
                tubelet_size=2,
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_hidden_layers=num_hidden_layers,
                mlp_ratio=mlp_ratio,
            )
        # from_pretrained normally sets this; manual construction must set it
        # explicitly or VJEPA2RopeAttention hits KeyError: None.
        config._attn_implementation = "sdpa"
        self.vjepa2_config = config
        self.encoder = VJEPA2Encoder(config)
        self.freeze_encoder = bool(freeze_encoder)
        if self.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # Mode A observation adapter: state -> pseudo token sequence (plan §3.2).
        self.obs_projection = layer_init(
            nn.Linear(obs_dim, num_pseudo_tokens * hidden_size)
        )

        self.actor_mean = layer_init(
            nn.Linear(hidden_size, action_dim), std=0.01 * np.sqrt(2)
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, action_dim) * -0.5)

        if add_value_head:
            self.value_head = nn.Sequential(
                layer_init(nn.Linear(hidden_size, 256)),
                nn.Tanh(),
                layer_init(nn.Linear(256, 1)),
            )

    # ---- V-JEPA 2 encoder over pseudo tokens (mode A) ----
    def _encode(self, states: torch.Tensor) -> torch.Tensor:
        # states: [B, obs_dim] -> pooled feature [B, hidden_size]
        batch = states.shape[0]
        hidden = self.obs_projection(states).view(
            batch, self.num_pseudo_tokens, self.hidden_size
        )
        for layer_module in self.encoder.layer:
            hidden = layer_module(hidden, None, None, False)[0]
        hidden = self.encoder.layernorm(hidden)
        return hidden.mean(dim=1)

    # ---- V-JEPA 2 encoder over rendered frames (mode B) ----
    def _encode_pixel(self, images: torch.Tensor) -> torch.Tensor:
        # images: [B, K, 3, S, S] float in [0,1] -> pooled [B, hidden_size]
        outputs = self.encoder(pixel_values_videos=images)
        return outputs.last_hidden_state.mean(dim=1)

    def _encode_obs(self, obs: dict) -> torch.Tensor:
        if self.obs_mode == "pixel":
            return self._encode_pixel(obs["main_images"])
        return self._encode(obs["states"])

    # ---- RLinf policy contract (mirrors MLPPolicy) ----
    def preprocess_env_obs(self, env_obs):
        device = next(self.parameters()).device
        if self.obs_mode == "pixel":
            return {"main_images": env_obs["main_images"].to(device)}
        return {"states": env_obs["states"].to(device)}

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError

    def default_forward(
        self,
        forward_inputs,
        compute_logprobs=True,
        compute_entropy=True,
        compute_values=True,
        **kwargs,
    ):
        states = forward_inputs["states"] if self.obs_mode != "pixel" else None
        action = forward_inputs["action"]

        feat = self._encode_obs(forward_inputs)
        action_mean = self.actor_mean(feat)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        output_dict: dict[str, Any] = {}
        if compute_logprobs:
            logprobs = probs.log_prob(action)
            output_dict.update(logprobs=logprobs)
        if compute_entropy:
            entropy = probs.entropy()
            output_dict.update(entropy=entropy)
        if compute_values:
            if getattr(self, "value_head", None):
                values = self.value_head(feat)
                output_dict.update(values=values)
            else:
                raise NotImplementedError
        return output_dict

    def _sample_actions(self, obs: dict):
        feat = self._encode_obs(obs)
        action_mean = self.actor_mean(feat)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        return action_mean, action_logstd

    def _generate_actions(
        self,
        obs: dict,
        mode: str = "train",
        calculate_values: bool = True,
        use_rsample: bool = False,
    ):
        feat = self._encode_obs(obs)
        action_mean = self.actor_mean(feat)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if mode == "train":
            raw_action = probs.rsample() if use_rsample else probs.sample()
        elif mode == "eval":
            raw_action = action_mean.clone()
        else:
            raise NotImplementedError(f"{mode=}")

        chunk_logprobs = probs.log_prob(raw_action)
        action = raw_action
        chunk_actions = action.reshape(-1, self.num_action_chunks, self.action_dim)
        if hasattr(self, "value_head") and calculate_values:
            chunk_values = self.value_head(feat)
        else:
            chunk_values = torch.zeros_like(chunk_logprobs[..., :1])

        return action, chunk_actions, chunk_logprobs, chunk_values

    @torch.inference_mode()
    def predict_action_batch(
        self,
        env_obs,
        calculate_logprobs=True,
        calculate_values=True,
        return_obs=True,
        mode="train",
        **kwargs,
    ):
        env_obs = self.preprocess_env_obs(env_obs=env_obs)

        action, chunk_actions, chunk_logprobs, chunk_values = self._generate_actions(
            env_obs, mode=mode, calculate_values=calculate_values
        )

        forward_inputs = {"action": action, "model_action": action}
        if return_obs:
            forward_inputs.update(env_obs)

        result = {
            "prev_logprobs": chunk_logprobs,
            "prev_values": chunk_values,
            "forward_inputs": forward_inputs,
        }
        return chunk_actions, result


def get_model(cfg, torch_dtype=None):
    """ModelBuilder for rlinf.models.register_model (cfg = actor.model node)."""
    return VJEPA2Policy(
        obs_dim=cfg.obs_dim,
        action_dim=cfg.action_dim,
        num_action_chunks=cfg.num_action_chunks,
        add_value_head=cfg.get("add_value_head", True),
        hidden_size=cfg.get("vjepa2_hidden_size", 384),
        num_hidden_layers=cfg.get("vjepa2_num_hidden_layers", 4),
        num_attention_heads=cfg.get("vjepa2_num_attention_heads", 6),
        num_pseudo_tokens=cfg.get("vjepa2_num_pseudo_tokens", 8),
        freeze_encoder=cfg.get("vjepa2_freeze_encoder", True),
        obs_mode=cfg.get("obs_mode", "state"),
        crop_size=cfg.get("vjepa2_crop_size", 64),
        patch_size=cfg.get("vjepa2_patch_size", 16),
        frames_per_clip=cfg.get("vjepa2_frames_per_clip", 2),
        tubelet_size=cfg.get("vjepa2_tubelet_size", 2),
    )
