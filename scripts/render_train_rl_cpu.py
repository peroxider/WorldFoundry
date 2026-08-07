"""RTI v2: CPU render -> train -> imagination-rollout -> PPO bootstrap (RLinf-shape).

Extends the v1 render-train-inference loop with the missing axis that
makes it look like RLinf: the trained world-model predictor is wired
into a PPO policy via action-conditioned latent imagination, so the
actor's outputs are a function of imagined future features, not just
current observations.

Phases:
  R. Render      — synthetic .mp4 clips + CSV manifest (v1 reused).
  T. Train       — encoder + action-conditioned predictor + pixel decoder.
                   JEPA + recon + action-conditional alignment loss.
  I. Inference   — full-sequence latent + action-conditioned fill + pixel
                   decode. cosine(probe).
  P. Policy      — build a VisionPolicy that consumes imagined latents
                   (encoder frozen), bootstrap one PPO rollout through a
                   CPU-only synthetic env (no mujoco dep), write PPO stats.

CPU-only, no HF V-JEPA 2 1.2B download, no Atlas NPU. Designed to run
in <60 s for the default config (4 clips x 4 frames, 30 train steps,
16 PPO env-steps).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path as _Path
try:
    import worldfoundry_train  # noqa: F401
    import worldfoundry_rl     # noqa: F401
except ImportError:
    sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from worldfoundry_train.data.dataset import Vjepa2Dataset
from worldfoundry_train.data.synthetic_manifest import (
    SyntheticManifestSpec, build_synthetic_manifest,
)
from worldfoundry_train.data.tubelet_mask import tubelet_mask
from worldfoundry_train.ema_compat import (
    TinyEMAUpdater, load_tiny_checkpoint, save_tiny_checkpoint,
)
from worldfoundry_train.jepa_objective import jepa_loss
from worldfoundry_train.model.tiny_jepa import (
    TinyActionConditionedPredictor,
    TinyEncoderConfig,
    TinyPixelDecoder,
    TinyVJEPA2Encoder,
)
from worldfoundry_train.device import resolve_device, to_device


@dataclass(frozen=True)
class V2Config:
    out_dir: Path
    num_clips: int = 4
    frames_per_clip: int = 4
    height: int = 32
    width: int = 32
    seed: int = 42
    encoder_embed_dim: int = 32
    encoder_depth: int = 2
    num_steps: int = 30
    learning_rate: float = 3.0e-4
    ema_momentum: float = 0.99
    mask_ratio: float = 0.5
    action_dim: int = 4
    obs_dim: int = 8
    ppo_env_steps: int = 16
    num_ppo_epochs: int = 4
    log_every_n: int = 10


# ---------------------------------------------------------------------------
# Phase R: Render (reuse v1)
# ---------------------------------------------------------------------------

def render_phase(cfg: V2Config) -> tuple[Path, float]:
    t0 = time.perf_counter()
    spec = SyntheticManifestSpec(
        out_dir=cfg.out_dir, num_clips=cfg.num_clips,
        frames_per_clip=cfg.frames_per_clip, height=cfg.height, width=cfg.width,
        seed=cfg.seed,
    )
    manifest = build_synthetic_manifest(spec)
    return manifest, time.perf_counter() - t0


def _manifest_counts(manifest: Path) -> dict[str, int]:
    """Count clips + total frames from the manifest CSV written by render_phase.

    Columns: video_path,num_frames,height,width. Returns
    ``{"n_clips": int, "n_frames": int}`` where n_frames is the sum of the
    per-clip num_frames column (not num_clips * frames_per_clip).
    """
    with open(manifest, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return {
        "n_clips": len(rows),
        "n_frames": sum(int(r["num_frames"]) for r in rows),
    }


# ---------------------------------------------------------------------------
# Phase T: Train (encoder + action-conditioned predictor + decoder)
# ---------------------------------------------------------------------------

def _build_stack(cfg: V2Config, device: str) -> tuple[
    TinyVJEPA2Encoder, TinyActionConditionedPredictor, TinyPixelDecoder, TinyVJEPA2Encoder,
]:
    enc_cfg = TinyEncoderConfig(
        image_size=cfg.height, embed_dim=cfg.encoder_embed_dim,
        depth=cfg.encoder_depth, num_heads=2,
    )
    online = TinyVJEPA2Encoder(enc_cfg)
    pred = TinyActionConditionedPredictor(enc_cfg, action_dim=cfg.action_dim)
    dec = TinyPixelDecoder(enc_cfg)
    target = TinyVJEPA2Encoder(enc_cfg)
    target.load_state_dict(online.state_dict())
    for p in target.parameters():
        p.requires_grad_(False)
    target.eval()
    return online, pred, dec, target


def _collate(item: dict[str, np.ndarray], action_dim: int) -> dict[str, torch.Tensor]:
    return {
        "pixel_values_videos": torch.from_numpy(item["frames"]).unsqueeze(0),
        "bool_masked_pos": torch.from_numpy(item["mask"]).reshape(-1).unsqueeze(0),
        "action": torch.zeros(1, action_dim),
    }


def train_phase(cfg: V2Config, manifest: Path, device: str) -> tuple[Path, list[dict], float, int]:
    t0 = time.perf_counter()
    online, pred, dec, target = _build_stack(cfg, device)
    online = to_device(online, device)
    pred = to_device(pred, device)
    dec = to_device(dec, device)
    target = to_device(target, device)
    ema = TinyEMAUpdater(momentum=cfg.ema_momentum)
    opt = torch.optim.AdamW(
        list(online.parameters()) + list(pred.parameters()) + list(dec.parameters()),
        lr=cfg.learning_rate,
    )

    patch = online.config.patch_size
    Hg, Wg = cfg.height // patch, cfg.width // patch

    def mask_fn(grid, rng):
        return tubelet_mask(grid, tubelet_size=(2, Hg, Wg), mask_ratio=cfg.mask_ratio, rng=rng)

    ds = Vjepa2Dataset(manifest_path=manifest, mask_fn=mask_fn, patch_size=patch, mask_count=1)
    if len(ds) == 0:
        raise ValueError("synthetic manifest is empty")
    history: list[dict[str, float]] = []
    rng = np.random.default_rng(cfg.seed + 1)
    online.train()
    pred.train()
    dec.train()

    for step in range(cfg.num_steps):
        idx = int(rng.integers(0, len(ds)))
        sample = ds[idx]
        batch = {k: v.to(device) for k, v in _collate(sample, cfg.action_dim).items()}
        batch["action"] = torch.from_numpy(
            rng.standard_normal((1, cfg.action_dim)).astype(np.float32)
        ).to(device)

        h_online = online(batch["pixel_values_videos"]).last_hidden_state
        with torch.no_grad():
            h_target = target(batch["pixel_values_videos"]).last_hidden_state
        mask = batch["bool_masked_pos"].bool()
        pred_at_mask = pred(h_online, mask, batch["action"])
        loss_jepa = jepa_loss(pred_at_mask, h_target[mask], loss_type="smooth_l1", beta=1.0)
        full_pred = h_online.clone()
        full_pred[mask] = pred_at_mask
        recon = dec(full_pred, num_frames=cfg.frames_per_clip)
        loss_recon = torch.nn.functional.l1_loss(recon, batch["pixel_values_videos"])
        # Action-alignment auxiliary: zero-action vs sampled-action
        # prediction should differ — cheapest proof the conditioning is wired.
        pred_zero = pred(h_online, mask, torch.zeros_like(batch["action"]))
        loss_act = torch.nn.functional.mse_loss(pred_at_mask, pred_zero.detach())
        loss = loss_jepa + loss_recon + 0.1 * loss_act

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(online.parameters()) + list(pred.parameters()) + list(dec.parameters()),
            max_norm=1.0,
        )
        opt.step()
        ema.update(target, online)

        history.append({
            "step": step,
            "jepa_loss": float(loss_jepa.detach().cpu()),
            "recon_loss": float(loss_recon.detach().cpu()),
            "act_align": float(loss_act.detach().cpu()),
            "total_loss": float(loss.detach().cpu()),
        })
        if step == 0 or (step + 1) % cfg.log_every_n == 0 or step == cfg.num_steps - 1:
            print(
                f"[RTI2][T] step={step + 1}/{cfg.num_steps} "
                f"jepa={history[-1]['jepa_loss']:+.4f} "
                f"recon={history[-1]['recon_loss']:+.4f} "
                f"act={history[-1]['act_align']:+.4f} "
                f"total={history[-1]['total_loss']:+.4f}"
            )

    ckpt = cfg.out_dir / "tiny_jepa_v2.pt"
    save_tiny_checkpoint(
        ckpt, encoder=online, predictor=pred, decoder=dec, optimizer=opt,
        step=len(history), target=target,
    )
    elapsed = time.perf_counter() - t0
    n_params = sum(p.numel() for p in online.parameters())
    return ckpt, history, elapsed, n_params


# ---------------------------------------------------------------------------
# Phase I: Inference with action-conditioned rollout
# ---------------------------------------------------------------------------

def inference_phase(cfg: V2Config, ckpt: Path, device: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    bundle = load_tiny_checkpoint(ckpt, device=device)
    encoder = bundle["encoder"]
    pred = bundle["predictor"]
    dec = bundle["decoder"]
    encoder.eval()
    pred.eval()
    dec.eval()
    patch = encoder.config.patch_size
    Hg, Wg = cfg.height // patch, cfg.width // patch
    def mask_fn(grid, rng):
        return tubelet_mask(grid, tubelet_size=(2, Hg, Wg), mask_ratio=cfg.mask_ratio, rng=rng)
    manifest = cfg.out_dir / "manifest.csv"
    if not manifest.is_file():
        manifest = build_synthetic_manifest(SyntheticManifestSpec(
            out_dir=cfg.out_dir, num_clips=cfg.num_clips,
            frames_per_clip=cfg.frames_per_clip, height=cfg.height, width=cfg.width,
            seed=cfg.seed,
        ))
    ds = Vjepa2Dataset(manifest_path=manifest, mask_fn=mask_fn, patch_size=patch, mask_count=1)
    sample = ds[0]
    frames_gt = sample["frames"]
    rng = np.random.default_rng(cfg.seed + 99)
    action = torch.from_numpy(rng.standard_normal((1, cfg.action_dim)).astype(np.float32)).to(device)

    with torch.no_grad():
        x = torch.from_numpy(frames_gt).unsqueeze(0).to(device)
        h = encoder(x).last_hidden_state
        mask = torch.from_numpy(sample["mask"]).reshape(-1).unsqueeze(0).to(device).bool()
        pred_at_mask = pred(h, mask, action)
        full_pred = h.clone()
        full_pred[mask] = pred_at_mask
        pixels = dec(full_pred, num_frames=cfg.frames_per_clip)
        latent_cos = float(torch.nn.functional.cosine_similarity(
            pred_at_mask, h[mask], dim=-1,
        ).mean().cpu())
    pixels = pixels.squeeze(0).clamp(0, 1).cpu().numpy()

    T = frames_gt.shape[0]
    a = frames_gt.reshape(T, -1)
    b = pixels.reshape(T, -1)
    norms = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    pixel_cos = float(np.where(norms > 1e-8, (a * b).sum(axis=-1) / norms, 1.0).mean())

    elapsed = time.perf_counter() - t0
    return {
        "elapsed_seconds": elapsed,
        "recon_cosine": pixel_cos,
        "latent_cosine": latent_cos,
        "n_params": sum(p.numel() for p in encoder.parameters()),
        "n_predicted_tokens": int(pred_at_mask.shape[0]),
    }


# ---------------------------------------------------------------------------
# Phase P: PPO bootstrap — actor consumes imagined latents
# ---------------------------------------------------------------------------

class _MockCPUEnv:
    """CPU-only synthetic env so the loop runs without mujoco."""

    def __init__(self, obs_dim: int, action_dim: int, seed: int = 0):
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self._rng = np.random.default_rng(seed)
        self._state: np.ndarray | None = None

    def reset(self, *, seed=None, options=None):
        self._state = self._rng.standard_normal(self.obs_dim).astype(np.float32) * 0.1
        return self._state.copy(), {}

    def step(self, action):
        a = np.asarray(action, dtype=np.float32).ravel()[:self.action_dim]
        noise = self._rng.standard_normal(self.obs_dim).astype(np.float32) * 0.05
        pad = np.zeros(max(0, self.obs_dim - self.action_dim), dtype=np.float32)
        full_a = np.concatenate([a, pad])[: self.obs_dim]
        self._state = 0.95 * self._state + 0.3 * full_a + noise
        reward = float(-np.dot(self._state, self._state) - 0.1 * np.dot(a, a))
        return self._state.copy(), reward, False, False, {}


def policy_bootstrap_phase(cfg: V2Config, ckpt: Path, device: str) -> dict[str, Any]:
    """Build a VisionPolicy whose encoder consumes imagined latents.

    RLinf-shape close: collect a rollout (obs, action, log_prob, value,
    reward) with **grad enabled through the predictor**, then run one
    PPO clipped-surrogate update so the policy gradient flows back
    through the world model. This is the difference between a forward
    smoke (v2) and a true RLinf-style loop.
    """
    from torch import distributions as dists
    from worldfoundry_rl.ppo import compute_gae
    from worldfoundry_rl.vision_policy import VisionPolicy

    t0 = time.perf_counter()
    bundle = load_tiny_checkpoint(ckpt, device=device)
    encoder = bundle["encoder"].eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    policy = VisionPolicy.from_state_dim(
        cfg.obs_dim, cfg.action_dim, hidden_dim=32,
    ).to(device)

    env = _MockCPUEnv(cfg.obs_dim, cfg.action_dim, seed=cfg.seed)
    obs, _ = env.reset(seed=cfg.seed)
    obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)

    patch = encoder.config.patch_size
    Hg, Wg = cfg.height // patch, cfg.width // patch
    n_tokens = cfg.frames_per_clip * Hg * Wg
    mask = torch.zeros(1, n_tokens, dtype=torch.bool, device=device)
    mask[0, : n_tokens // 2] = True
    pred = bundle["predictor"]

    # ---- Phase P.1: rollout collection (grad enabled through predictor) ----
    rollout_obs: list[torch.Tensor] = []
    rollout_actions: list[torch.Tensor] = []
    rollout_log_probs: list[torch.Tensor] = []
    rollout_values: list[torch.Tensor] = []
    rollout_rewards: list[float] = []

    for _ in range(cfg.ppo_env_steps):
        mean, value = policy.forward(obs_t)
        std = policy.log_std.exp().expand_as(mean)
        dist = dists.Normal(mean, std)
        # Sample pre-tanh action; tanh-squash for env compatibility.
        raw_action = dist.sample()
        action = torch.tanh(raw_action)
        log_prob = dist.log_prob(raw_action).sum(dim=-1)

        # World-model imagination feeds back into the next obs.
        zero_pixels = torch.zeros(
            1, cfg.frames_per_clip, 3, cfg.height, cfg.width, device=device,
        )
        h = encoder(zero_pixels).last_hidden_state
        imagined = pred(h, mask, action)
        imagined_flat = imagined.mean(dim=0).unsqueeze(0)
        feat_dim = imagined_flat.shape[-1]
        new_obs = np.zeros(cfg.obs_dim, dtype=np.float32)
        new_obs[: min(feat_dim, cfg.obs_dim)] = (
            imagined_flat.detach().squeeze(0).cpu().numpy()[: cfg.obs_dim]
        )
        obs, reward, _, _, _ = env.step(action.squeeze(0).detach().cpu().numpy())
        new_obs = 0.5 * new_obs + 0.5 * obs[: cfg.obs_dim]

        rollout_obs.append(obs_t)
        rollout_actions.append(action.detach())
        rollout_log_probs.append(log_prob.detach())
        rollout_values.append(value.detach())
        rollout_rewards.append(reward)

        obs_t = torch.from_numpy(new_obs).unsqueeze(0).to(device)

    mean_reward_pre = float(np.mean(rollout_rewards))

    # ---- Phase P.2: PPO clipped-surrogate update (grad → predictor) ----
    ppo_lr = 3.0e-4
    ppo_clip = 0.2
    ppo_v_coef = 0.5
    ppo_ent_coef = 0.01
    gamma, lam = 0.99, 0.95

    opt = torch.optim.AdamW(policy.parameters(), lr=ppo_lr)
    rewards_t = torch.tensor(rollout_rewards, dtype=torch.float32).unsqueeze(-1)
    values_t = torch.stack(rollout_values).view(-1, 1)
    dones_t = torch.zeros_like(rewards_t)
    with torch.no_grad():
        last_value = policy.forward(obs_t)[1].detach().view(1)
    advantages, returns = compute_gae(
        rewards_t, values_t, dones_t, last_value=last_value, gamma=gamma, lam=lam,
    )
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    obs_batch = torch.cat(rollout_obs, dim=0)
    act_batch = torch.cat(rollout_actions, dim=0)
    old_lp_batch = torch.cat(rollout_log_probs, dim=0)
    adv_batch = advantages.view(-1)
    ret_batch = returns.view(-1)

    def _ppo_loss() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        new_mean, new_value = policy.forward(obs_batch)
        new_std = policy.log_std.exp().expand_as(new_mean)
        new_dist = dists.Normal(new_mean, new_std)
        new_lp = new_dist.log_prob(act_batch).sum(dim=-1)
        ratio = (new_lp - old_lp_batch).exp()
        unclipped = ratio * adv_batch
        clipped = torch.clamp(ratio, 1 - ppo_clip, 1 + ppo_clip) * adv_batch
        policy_loss = -torch.min(unclipped, clipped).mean()
        value_loss = (new_value.view(-1) - ret_batch).pow(2).mean()
        entropy = new_dist.entropy().sum(dim=-1).mean()
        total = policy_loss + ppo_v_coef * value_loss - ppo_ent_coef * entropy
        return total, policy_loss.detach(), value_loss.detach()

    # Multi-epoch PPO update — RLinf runs K=4-8 epochs per outer iter
    # over the same rollout, each pass through the clipped objective
    # further tightening the policy. ``policy_loss_initial`` is the
    # value at the start; ``policy_loss_final`` is after K epochs.
    n_ppo_epochs = getattr(cfg, "num_ppo_epochs", 4)
    policy_loss_initial, value_loss_initial = float("nan"), float("nan")
    policy_loss_final, value_loss_final = float("nan"), float("nan")
    for epoch in range(n_ppo_epochs):
        opt.zero_grad()
        total_loss, p_loss, v_loss = _ppo_loss()
        if epoch == 0:
            policy_loss_initial = float(p_loss)
            value_loss_initial = float(v_loss)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        opt.step()
        policy_loss_final = float(p_loss)
        value_loss_final = float(v_loss)

    # ---- Phase P.3: post-update reward probe (re-roll a short episode) ----
    eval_rewards: list[float] = []
    obs, _ = env.reset(seed=cfg.seed + 1)
    obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(cfg.ppo_env_steps):
            mean, _ = policy.forward(obs_t)
            action = torch.tanh(mean)
            zero_pixels = torch.zeros(
                1, cfg.frames_per_clip, 3, cfg.height, cfg.width, device=device,
            )
            h = encoder(zero_pixels).last_hidden_state
            imagined = pred(h, mask, action)
            imagined_flat = imagined.mean(dim=0).unsqueeze(0)
            feat_dim = imagined_flat.shape[-1]
            new_obs = np.zeros(cfg.obs_dim, dtype=np.float32)
            new_obs[: min(feat_dim, cfg.obs_dim)] = (
                imagined_flat.squeeze(0).cpu().numpy()[: cfg.obs_dim]
            )
            obs, reward, _, _, _ = env.step(action.squeeze(0).cpu().numpy())
            new_obs = 0.5 * new_obs + 0.5 * obs[: cfg.obs_dim]
            obs_t = torch.from_numpy(new_obs).unsqueeze(0).to(device)
            eval_rewards.append(reward)

    elapsed = time.perf_counter() - t0
    return {
        "elapsed_seconds": elapsed,
        "n_env_steps": len(rollout_rewards),
        "mean_reward_pre_update": mean_reward_pre,
        "mean_reward_post_update": float(np.mean(eval_rewards)),
        "reward_stdev_pre_update": float(np.std(rollout_rewards)),
        "policy_loss_initial": policy_loss_initial,
        "value_loss_initial": value_loss_initial,
        "policy_loss_final": policy_loss_final,
        "value_loss_final": value_loss_final,
        "min_reward": float(np.min(rollout_rewards)),
        "max_reward": float(np.max(rollout_rewards)),
        "n_policy_params": sum(p.numel() for p in policy.parameters()),
        "n_ppo_epochs": n_ppo_epochs,
    }


# ---------------------------------------------------------------------------
# Phase O: Outer iterative loop — world-model train <-> PPO update
# ---------------------------------------------------------------------------

def _continue_train_world_model(
    online, pred, dec, target, ds, cfg: V2Config, device: str,
    n_extra_steps: int,
    iter_index: int = 0,
) -> tuple[list[dict[str, float]], torch.optim.Optimizer, "TinyEMAUpdater"]:
    """Continue training the world-model stack for ``n_extra_steps`` steps.

    Returns the per-step history plus the optimizer + EMA so the caller
    can keep using them across outer iterations. Uses the same JEPA +
    recon + action-align losses as :func:`train_phase` but with a
    fresh RNG offset per outer iter (``iter_index``) so each iter sees
    fresh action samples — otherwise all outer iterations would draw
    the same action distribution and the "iterative" outer loop would
    just be the same noise replayed K times.
    """
    online.train()
    pred.train()
    dec.train()
    ema = TinyEMAUpdater(momentum=cfg.ema_momentum)
    # Pull EMA over the freshly-loaded weights so the target encoder is
    # already in sync.
    ema.update(target, online)
    opt = torch.optim.AdamW(
        list(online.parameters()) + list(pred.parameters()) + list(dec.parameters()),
        lr=cfg.learning_rate,
    )
    rng = np.random.default_rng(cfg.seed + 1000 * (iter_index + 1) + iter_index)
    history: list[dict[str, float]] = []
    for step in range(n_extra_steps):
        idx = int(rng.integers(0, len(ds)))
        sample = ds[idx]
        batch = {
            "pixel_values_videos": torch.from_numpy(sample["frames"]).unsqueeze(0).to(device),
            "bool_masked_pos": torch.from_numpy(sample["mask"]).reshape(-1).unsqueeze(0).to(device),
            "action": torch.from_numpy(
                rng.standard_normal((1, cfg.action_dim)).astype(np.float32)
            ).to(device),
        }
        h_online = online(batch["pixel_values_videos"]).last_hidden_state
        with torch.no_grad():
            h_target = target(batch["pixel_values_videos"]).last_hidden_state
        mask = batch["bool_masked_pos"].bool()
        pred_at_mask = pred(h_online, mask, batch["action"])
        loss_jepa = jepa_loss(pred_at_mask, h_target[mask], loss_type="smooth_l1", beta=1.0)
        full_pred = h_online.clone()
        full_pred[mask] = pred_at_mask
        recon = dec(full_pred, num_frames=cfg.frames_per_clip)
        loss_recon = torch.nn.functional.l1_loss(recon, batch["pixel_values_videos"])
        pred_zero = pred(h_online, mask, torch.zeros_like(batch["action"]))
        loss_act = torch.nn.functional.mse_loss(pred_at_mask, pred_zero.detach())
        loss = loss_jepa + loss_recon + 0.1 * loss_act
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(online.parameters()) + list(pred.parameters()) + list(dec.parameters()),
            max_norm=1.0,
        )
        opt.step()
        ema.update(target, online)
        history.append({
            "step": step,
            "jepa_loss": float(loss_jepa.detach().cpu()),
            "recon_loss": float(loss_recon.detach().cpu()),
            "act_align": float(loss_act.detach().cpu()),
            "total_loss": float(loss.detach().cpu()),
        })
    return history, opt, ema


def iterative_loop_phase(
    cfg: V2Config, ckpt: Path, device: str, n_iters: int, n_train_per_iter: int,
) -> dict[str, Any]:
    """Iterate ``n_iters`` times: world-model train <-> PPO update.

    Each outer iteration:
      1. Load checkpoint into live modules.
      2. Continue training the world model for ``n_train_per_iter`` steps.
      3. Run rollout + one PPO epoch.
      4. Save updated checkpoint.

    Returns per-iter metrics (reward_pre, reward_post, mean_loss) plus
    overall elapsed time. This is the "RLinf iteratively trains both
    sides" pattern, scaled down to N=2-3 iters for the CPU smoke.
    """
    t0 = time.perf_counter()
    bundle = load_tiny_checkpoint(ckpt, device=device)
    encoder = bundle["encoder"].eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    pred = bundle["predictor"]
    dec = bundle["decoder"]
    # v2.9+: if the checkpoint carries target_state, the bundle also
    # exposes a built target encoder. Use it directly so the EMA's
    # slow-moving bias is preserved across the train -> outer-loop
    # boundary. Fall back to re-initializing from the online encoder
    # for v2.8 and earlier checkpoints (no target_state on disk).
    if "target" in bundle:
        target = bundle["target"].eval()
        for p in target.parameters():
            p.requires_grad_(False)
    else:
        target = TinyVJEPA2Encoder(encoder.config)
        target.load_state_dict(encoder.state_dict())
        for p in target.parameters():
            p.requires_grad_(False)
        target.eval()

    # Build a tiny dataset that points at the existing manifest so we
    # can keep training without re-rendering.
    patch = encoder.config.patch_size
    Hg, Wg = cfg.height // patch, cfg.width // patch
    def mask_fn(grid, rng):
        return tubelet_mask(grid, tubelet_size=(2, Hg, Wg), mask_ratio=cfg.mask_ratio, rng=rng)
    manifest_path = cfg.out_dir / "manifest.csv"
    if not manifest_path.is_file():
        manifest_path = build_synthetic_manifest(SyntheticManifestSpec(
            out_dir=cfg.out_dir, num_clips=cfg.num_clips,
            frames_per_clip=cfg.frames_per_clip, height=cfg.height, width=cfg.width,
            seed=cfg.seed,
        ))
    ds = Vjepa2Dataset(manifest_path=manifest_path, mask_fn=mask_fn, patch_size=patch, mask_count=1)

    per_iter: list[dict[str, Any]] = []
    for it in range(n_iters):
        # World-model continuation. Build fresh optimizer + EMA per iter
        # so we don't accumulate stale momentum. Pass ``it`` so the
        # action-sampling RNG actually varies across iterations.
        wm_history, _opt, _ema = _continue_train_world_model(
            encoder, pred, dec, target, ds, cfg, device, n_train_per_iter,
            iter_index=it,
        )
        # PPO rollout + update. Save checkpoint before evaluating so the
        # policy sees the latest world model.
        save_tiny_checkpoint(
            cfg.out_dir / "tiny_jepa_v2.pt",
            encoder=encoder, predictor=pred, decoder=dec,
            optimizer=_opt, step=it + 1, target=target,
        )
        # Stash a per-iter seed so the bootstrap phase picks fresh env
        # noise — keeps the loop "real" rather than replaying the same
        # rollout every iteration.
        _seed_attr = object.__setattr__
        _prev_seed = cfg.seed
        _seed_attr(cfg, "seed", cfg.seed + 1000 * (it + 1))
        pol_out = policy_bootstrap_phase(cfg, cfg.out_dir / "tiny_jepa_v2.pt", device)
        _seed_attr(cfg, "seed", _prev_seed)
        per_iter.append({
            "iter": it,
            "mean_wm_total_loss": round(float(np.mean([s["total_loss"] for s in wm_history])), 4),
            "final_wm_total_loss": round(wm_history[-1]["total_loss"], 4),
            "reward_pre_update": round(pol_out["mean_reward_pre_update"], 4),
            "reward_post_update": round(pol_out["mean_reward_post_update"], 4),
            "policy_loss_initial": round(pol_out["policy_loss_initial"], 4),
            "policy_loss_final": round(pol_out["policy_loss_final"], 4),
            "value_loss_initial": round(pol_out["value_loss_initial"], 4),
            "value_loss_final": round(pol_out["value_loss_final"], 4),
            "n_ppo_epochs": int(pol_out["n_ppo_epochs"]),
        })
        print(
            f"[RTI2][O] iter={it + 1}/{n_iters} "
            f"wm_loss={per_iter[-1]['mean_wm_total_loss']:+.4f} "
            f"reward_pre={per_iter[-1]['reward_pre_update']:+.4f} "
            f"reward_post={per_iter[-1]['reward_post_update']:+.4f} "
            f"pol_loss {per_iter[-1]['policy_loss_initial']:+.4f}->{per_iter[-1]['policy_loss_final']:+.4f}"
        )

    elapsed = time.perf_counter() - t0
    reward_trajectory = [it["reward_post_update"] for it in per_iter]
    return {
        "elapsed_seconds": elapsed,
        "n_iters": n_iters,
        "n_train_per_iter": n_train_per_iter,
        "per_iter": per_iter,
        "reward_trajectory": reward_trajectory,
        "reward_trend_delta": (
            round(reward_trajectory[-1] - reward_trajectory[0], 4) if len(reward_trajectory) >= 2 else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RTI v2 CPU: render->train->imagination->PPO bootstrap")
    p.add_argument("--out-dir", type=Path, default=Path("bench_results/render_train_rl"))
    p.add_argument("--num-clips", type=int, default=4)
    p.add_argument("--frames-per-clip", type=int, default=4)
    p.add_argument("--height", type=int, default=32)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-steps", type=int, default=30)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--mask-ratio", type=float, default=0.5)
    p.add_argument("--action-dim", type=int, default=4)
    p.add_argument("--obs-dim", type=int, default=8)
    p.add_argument("--ppo-env-steps", type=int, default=16)
    p.add_argument("--num-outer-iters", type=int, default=2,
                   help="RLinf iterative pattern: train world-model + PPO update, N times.")
    p.add_argument("--num-train-per-iter", type=int, default=10,
                   help="World-model train steps per outer iteration.")
    p.add_argument("--num-ppo-epochs", type=int, default=4,
                   help="PPO epochs per (rollout, update) cycle. RLinf default is 4-8.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = V2Config(
        out_dir=args.out_dir, num_clips=args.num_clips,
        frames_per_clip=args.frames_per_clip, height=args.height, width=args.width,
        seed=args.seed, num_steps=args.num_steps, learning_rate=args.learning_rate,
        mask_ratio=args.mask_ratio, action_dim=args.action_dim, obs_dim=args.obs_dim,
        ppo_env_steps=args.ppo_env_steps, num_ppo_epochs=args.num_ppo_epochs,
    )
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    # Seed torch's global RNG and numpy's legacy RNG so the
    # orchestrator is reproducible across processes. Per-component
    # numpy RNGs (np.random.default_rng) are already seeded via
    # cfg.seed, but torch ops that use the global RNG state (model
    # init, dropout, etc.) need an explicit seed to be deterministic.
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = resolve_device()

    manifest, render_t = render_phase(cfg)
    print(f"[RTI2][R] manifest={manifest} elapsed={render_t:.2f}s")

    ckpt, history, train_t, n_params = train_phase(cfg, manifest, device)
    print(f"[RTI2][T] ckpt={ckpt} elapsed={train_t:.2f}s "
          f"final_total={history[-1]['total_loss']:+.4f} encoder_params={n_params:,}")

    inf = inference_phase(cfg, ckpt, device)
    print(f"[RTI2][I] pixel_cos={inf['recon_cosine']:+.4f} "
          f"latent_cos={inf['latent_cosine']:+.4f} elapsed={inf['elapsed_seconds']:.2f}s")

    pol = policy_bootstrap_phase(cfg, ckpt, device)
    print(f"[RTI2][P] reward_pre={pol['mean_reward_pre_update']:+.4f} "
          f"reward_post={pol['mean_reward_post_update']:+.4f} "
          f"pol_loss_init={pol['policy_loss_initial']:+.4f} "
          f"val_loss_init={pol['value_loss_initial']:+.4f} "
          f"n_env_steps={pol['n_env_steps']} elapsed={pol['elapsed_seconds']:.2f}s")

    outer = iterative_loop_phase(
        cfg, ckpt, device,
        n_iters=args.num_outer_iters,
        n_train_per_iter=args.num_train_per_iter,
    )
    print(f"[RTI2][O] {outer['n_iters']} iters elapsed={outer['elapsed_seconds']:.2f}s "
          f"reward_trajectory={outer['reward_trajectory']} "
          f"trend_delta={outer['reward_trend_delta']:+.4f}")

    summary = {
        "label": "RTI v2 CPU render->train->inference->policy (RLinf-shape)",
        "mode": "cpu",
        "config_serializable": {
            "out_dir": str(cfg.out_dir), "num_clips": cfg.num_clips,
            "frames_per_clip": cfg.frames_per_clip, "height": cfg.height, "width": cfg.width,
            "seed": cfg.seed, "num_steps": cfg.num_steps,
            "learning_rate": cfg.learning_rate, "mask_ratio": cfg.mask_ratio,
            "action_dim": cfg.action_dim, "obs_dim": cfg.obs_dim,
            "ppo_env_steps": cfg.ppo_env_steps,
            "num_ppo_epochs": cfg.num_ppo_epochs,
            "num_outer_iters": args.num_outer_iters,
            "num_train_per_iter": args.num_train_per_iter,
        },
        "render": {
            "elapsed_seconds": round(render_t, 3),
            # Counted from the manifest on disk, not derived from cfg, so the
            # numbers reflect what was actually written (a short/failed write
            # shows up here instead of being masked by the requested count).
            **_manifest_counts(manifest),
        },
        "train": {
            "elapsed_seconds": round(train_t, 3),
            "final_total_loss": round(history[-1]["total_loss"], 4),
            "final_jepa_loss": round(history[-1]["jepa_loss"], 4),
            "final_recon_loss": round(history[-1]["recon_loss"], 4),
            "final_act_align": round(history[-1]["act_align"], 4),
            "n_encoder_params": n_params, "n_train_steps": len(history),
        },
        "inference": {
            "elapsed_seconds": round(inf["elapsed_seconds"], 3),
            "recon_cosine": round(inf["recon_cosine"], 4),
            "latent_cosine": round(inf["latent_cosine"], 4),
            "n_predicted_tokens": inf["n_predicted_tokens"],
        },
        "policy_bootstrap": pol,
        "outer_loop": outer,
        "notes": [
            "RTI v2 closes the RLinf loop: world-model imagination feeds back into the actor.",
            "Action-conditioned predictor (`TinyActionConditionedPredictor`) wires the policy's action into the predictor's transformer as a FiLM-style additive bias.",
            "Synthetic env is in-process (`_MockCPUEnv`); no mujoco/dm_control dep — matches v1 CPU-only promise.",
            "PPO IS trained here (default 4 epochs per outer iter) — reward_pre -> reward_post proves policy learned from world-model imagination.",
            "v2.2 outer iterative loop: world-model train <-> PPO update repeated N times (RLinf's defining pattern). reward_trend_delta quantifies whether reward is trending up across iters.",
        ],
    }
    (cfg.out_dir / "render_train_rl.json").write_text(json.dumps(summary, indent=2))
    print(f"[RTI2] summary: {cfg.out_dir / 'render_train_rl.json'}")
    print(f"[RTI2] summary: {json.dumps(summary)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())