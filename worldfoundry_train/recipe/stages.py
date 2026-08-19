"""Stage interface + CPU stage implementations for the RTI recipe.

Decoupling model
----------------
* :class:`Stage` is the unit of execution — a named step with a stable
  ``run(ctx)`` contract. Pipelines are just ordered stage lists.
* :class:`RecipeContext` is the shared, backend-agnostic state passed
  between stages. It carries the config, the resolved device, and a
  mutable ``artifacts`` dict (manifest path, checkpoint path, raw
  frames/actions) so later stages can consume earlier stages' output
  without knowing *which* backend produced it.
* The CPU stages here wrap the already-proven phase functions in
  ``scripts/render_train_rl_cpu.py`` (loaded once through
  :func:`_load_phase_module`). This is the single import seam — swapping
  an ``npu`` backend means registering different stage classes, not
  touching this file's training math.

For NPU: a future ``npu`` backend registers stages with the same
``name`` + ``run(ctx)`` contract. The only CPU-specific assumptions
these stages make are (a) they ask the device via ``ctx.device`` and
(b) they route models through ``worldfoundry_train.device.to_device`` —
both already generic.
"""
from __future__ import annotations

import importlib.util
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from worldfoundry_train.recipe.config import RecipeConfig


# ---------------------------------------------------------------------------
# Single import seam to the proven phase implementations
# ---------------------------------------------------------------------------

_PHASE_REPO_ROOT = Path(__file__).resolve().parents[2]
_PHASE_MODULE_PATH = _PHASE_REPO_ROOT / "scripts" / "render_train_rl_cpu.py"


def _load_phase_module():
    """Load ``scripts/render_train_rl_cpu.py`` once via importlib.

    ``scripts`` is not a package (no ``__init__.py``), so it cannot be a
    normal import; the existing test suite uses the same
    ``spec_from_file_location`` pattern to reuse these functions.
    """
    if "_rti_phase_mod" in sys.modules:
        return sys.modules["_rti_phase_mod"]
    spec = importlib.util.spec_from_file_location(
        "_rti_phase_mod", str(_PHASE_MODULE_PATH),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_rti_phase_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Context + Stage contract
# ---------------------------------------------------------------------------

@dataclass
class RecipeContext:
    """Backend-agnostic state flowing through the pipeline.

    Attributes:
        cfg: the :class:`RecipeConfig` for this run.
        device: the resolved device string (``"cpu"`` / ``"npu"``).
        artifacts: mutable dict carrying cross-stage outputs. Keys are
            documented per-stage below; later stages read them by name.
        results: stage-name -> metrics-dict, for the final summary.
    """

    cfg: RecipeConfig
    device: str
    artifacts: dict[str, Any] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)


class Stage(ABC):
    """A named step in the RTI recipe pipeline.

    Subclasses implement :meth:`run`, read from ``ctx.artifacts``,
    and write both metrics (into ``ctx.results[self.name]``) and
    any artifacts later stages need.
    """

    name: str = ""

    @abstractmethod
    def run(self, ctx: RecipeContext) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# CPU stage implementations (wrap the proven phase functions)
# ---------------------------------------------------------------------------

class RenderStage(Stage):
    """Renders source video clips + manifest.

    Artifacts: ``manifest`` (Path), plus ``frames``/``actions`` when the
    data source is ``franka_kitchen``.
    """

    name = "render"

    def run(self, ctx: RecipeContext) -> None:
        cfg = ctx.cfg
        ph = _load_phase_module()
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        # W25-cleanup: --skip-render shortcut. Mirror v2's
        # ``render_phase`` short-circuit at the recipe layer so
        # callers can keep the same CLI surface (``-stages render``)
        # but reuse an existing manifest instead of regenerating
        # clips. The missing-manifest check is delegated to
        # ``ph.render_phase`` (which raises SystemExit rc=2 with the
        # same FATAL message v1 used).
        out = ph.render_phase(
            ph.V2Config(
                out_dir=cfg.out_dir, num_clips=cfg.num_clips,
                frames_per_clip=cfg.frames_per_clip,
                height=cfg.height, width=cfg.width, seed=cfg.seed,
                data_source=cfg.data_source, env_type=cfg.env_type,
                franka_task_name=cfg.franka_task_name, camera_id=cfg.camera_id,
                skip_render=cfg.skip_render,
            )
        )
        ctx.artifacts["manifest"] = out.manifest
        ctx.artifacts["frames"] = out.frames
        ctx.artifacts["actions"] = out.actions
        ctx.results[self.name] = {
            "elapsed_seconds": out.elapsed_seconds,
            "manifest": str(out.manifest),
            "data_source": cfg.data_source,
        }
        print(
            f"[recipe][render] manifest={out.manifest} "
            f"elapsed={out.elapsed_seconds:.2f}s data_source={cfg.data_source}"
        )


class TrainStage(Stage):
    """Trains the JEPA world-model (encoder + predictor + decoder).

    Reads: ``manifest`` (+ ``frames``/``actions`` for franka).
    Artifacts: ``ckpt`` (Path), ``train_history`` (list).
    """

    name = "train"

    def run(self, ctx: RecipeContext) -> None:
        cfg = ctx.cfg
        ph = _load_phase_module()
        manifest = ctx.artifacts["manifest"]
        ckpt, history, elapsed, n_params = ph.train_phase(
            ph.V2Config(
                out_dir=cfg.out_dir, num_clips=cfg.num_clips,
                frames_per_clip=cfg.frames_per_clip,
                height=cfg.height, width=cfg.width, seed=cfg.seed,
                encoder_embed_dim=cfg.encoder_embed_dim,
                encoder_depth=cfg.encoder_depth,
                # W22-quant: HF encoder dimensions + W8 flag passthrough.
                encoder_hidden_size=cfg.encoder_hidden_size,
                encoder_num_layers=cfg.encoder_num_layers,
                encoder_num_heads=cfg.encoder_num_heads,
                num_steps=cfg.num_steps, learning_rate=cfg.learning_rate,
                ema_momentum=cfg.ema_momentum, mask_ratio=cfg.mask_ratio,
                action_dim=cfg.action_dim, obs_dim=cfg.obs_dim,
                log_every_n=cfg.log_every_n, data_source=cfg.data_source,
                env_type=cfg.env_type, encoder_kind=cfg.encoder_kind,
                quant=cfg.quant,
                encoder_pretrained_repo_id=cfg.encoder_pretrained,
                hf_tubelet_size=cfg.hf_tubelet_size,
            ),
            manifest,
            ctx.device,
            frames=ctx.artifacts.get("frames"),
            actions=ctx.artifacts.get("actions"),
        )
        ctx.artifacts["ckpt"] = ckpt
        ctx.artifacts["train_history"] = history
        ctx.results[self.name] = {
            "elapsed_seconds": elapsed,
            "final_total_loss": history[-1]["total_loss"],
            "n_encoder_params": n_params,
            "n_train_steps": len(history),
            "ckpt": str(ckpt),
        }
        print(
            f"[recipe][train] ckpt={ckpt} elapsed={elapsed:.2f}s "
            f"final_total={history[-1]['total_loss']:+.4f} "
            f"encoder_params={n_params:,}"
        )


class InferenceStage(Stage):
    """Runs inference probes (pixel + latent cosine similarity).

    Reads: ``ckpt`` (+ ``frames``/``actions`` for franka). When the
    ``train`` stage was not run in this pipeline, the checkpoint is
    resolved from ``<out_dir>/tiny_jepa_v2.pt`` on disk so this stage
    can resume standalone.
    """

    name = "inference"

    def run(self, ctx: RecipeContext) -> None:
        cfg = ctx.cfg
        ph = _load_phase_module()
        ckpt = ctx.artifacts.get("ckpt") or (cfg.out_dir / "tiny_jepa_v2.pt")
        inf = ph.inference_phase(
            ph.V2Config(
                out_dir=cfg.out_dir, num_clips=cfg.num_clips,
                frames_per_clip=cfg.frames_per_clip,
                height=cfg.height, width=cfg.width, seed=cfg.seed,
                mask_ratio=cfg.mask_ratio, action_dim=cfg.action_dim,
                data_source=cfg.data_source, env_type=cfg.env_type,
                encoder_kind=cfg.encoder_kind,
                encoder_pretrained_repo_id=cfg.encoder_pretrained,
                hf_tubelet_size=cfg.hf_tubelet_size,
            ),
            ckpt,
            ctx.device,
            frames=ctx.artifacts.get("frames"),
            actions=ctx.artifacts.get("actions"),
        )
        ctx.results[self.name] = {
            "elapsed_seconds": inf["elapsed_seconds"],
            "recon_cosine": inf["recon_cosine"],
            "latent_cosine": inf["latent_cosine"],
            "n_predicted_tokens": inf["n_predicted_tokens"],
        }
        print(
            f"[recipe][inference] pixel_cos={inf['recon_cosine']:+.4f} "
            f"latent_cos={inf['latent_cosine']:+.4f} "
            f"elapsed={inf['elapsed_seconds']:.2f}s"
        )


class PolicyStage(Stage):
    """Bootstraps a PPO policy consuming imagined latents.

    Reads: ``ckpt``. Writes PPO bootstrap metrics.
    """

    name = "policy"

    def run(self, ctx: RecipeContext) -> None:
        cfg = ctx.cfg
        ph = _load_phase_module()
        pol = ph.policy_bootstrap_phase(
            ph.V2Config(
                out_dir=cfg.out_dir, height=cfg.height, width=cfg.width,
                seed=cfg.seed, action_dim=cfg.action_dim, obs_dim=cfg.obs_dim,
                ppo_env_steps=cfg.ppo_env_steps, num_ppo_epochs=cfg.num_ppo_epochs,
                env_type=cfg.env_type, franka_task_name=cfg.franka_task_name,
                policy_encoder=cfg.policy_encoder,
                rollout_horizon=cfg.rollout_horizon,
                encoder_kind=cfg.encoder_kind,
                encoder_pretrained_repo_id=cfg.encoder_pretrained,
                hf_tubelet_size=cfg.hf_tubelet_size,
            ),
            ctx.artifacts["ckpt"],
            ctx.device,
        )
        ctx.results[self.name] = pol
        print(
            f"[recipe][policy] reward_pre={pol['mean_reward_pre_update']:+.4f} "
            f"reward_post={pol['mean_reward_post_update']:+.4f} "
            f"n_env_steps={pol['n_env_steps']} "
            f"elapsed={pol['elapsed_seconds']:.2f}s"
        )


class OuterLoopStage(Stage):
    """RLinf-style outer loop: world-model train <-> PPO update, N times.

    Reads: ``ckpt``. This is the loop's defining iterative pattern.
    """

    name = "outer_loop"

    def run(self, ctx: RecipeContext) -> None:
        cfg = ctx.cfg
        ph = _load_phase_module()
        outer = ph.iterative_loop_phase(
            ph.V2Config(
                out_dir=cfg.out_dir, num_clips=cfg.num_clips,
                frames_per_clip=cfg.frames_per_clip,
                height=cfg.height, width=cfg.width, seed=cfg.seed,
                encoder_embed_dim=cfg.encoder_embed_dim,
                encoder_depth=cfg.encoder_depth,
                # W22-quant: HF encoder dimensions + W8 flag passthrough.
                encoder_hidden_size=cfg.encoder_hidden_size,
                encoder_num_layers=cfg.encoder_num_layers,
                encoder_num_heads=cfg.encoder_num_heads,
                num_steps=cfg.num_steps, learning_rate=cfg.learning_rate,
                ema_momentum=cfg.ema_momentum, mask_ratio=cfg.mask_ratio,
                action_dim=cfg.action_dim, obs_dim=cfg.obs_dim,
                ppo_env_steps=cfg.ppo_env_steps,
                num_ppo_epochs=cfg.num_ppo_epochs,
                data_source=cfg.data_source, env_type=cfg.env_type,
                franka_task_name=cfg.franka_task_name,
                policy_encoder=cfg.policy_encoder,
                rollout_horizon=cfg.rollout_horizon,
                encoder_kind=cfg.encoder_kind,
                quant=cfg.quant,
                encoder_pretrained_repo_id=cfg.encoder_pretrained,
                hf_tubelet_size=cfg.hf_tubelet_size,
            ),
            ctx.artifacts["ckpt"],
            ctx.device,
            n_iters=cfg.num_outer_iters,
            n_train_per_iter=cfg.num_train_per_iter,
        )
        ctx.results[self.name] = outer
        print(
            f"[recipe][outer_loop] {outer['n_iters']} iters "
            f"elapsed={outer['elapsed_seconds']:.2f}s "
            f"reward_trajectory={outer['reward_trajectory']} "
            f"trend_delta={outer['reward_trend_delta']:+.4f}"
        )


# ---------------------------------------------------------------------------
# Backend registry: cpu (default) registered below; npu pluggable later
# ---------------------------------------------------------------------------

CPU_STAGES: dict[str, type[Stage]] = {
    "render": RenderStage,
    "train": TrainStage,
    "inference": InferenceStage,
    "policy": PolicyStage,
    "outer_loop": OuterLoopStage,
}
