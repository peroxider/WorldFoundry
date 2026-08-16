"""Recipe configuration: every knob + the resolved execution device.

Keeping the device in the config (rather than scattered through stage
code) is the primary decoupling seam for NPU: ``device`` is filled by
:func:`worldfoundry_train.device.resolve_device`, which already honors
the ``WORLDFORGE_DEVICE`` env override and NPU auto-detection. On a CPU
box it resolves to ``"cpu"``; on an Atlas box to ``"npu"`` — with no
recipe code change.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from worldfoundry_train.device import resolve_device


@dataclass(frozen=True)
class RecipeConfig:
    """Declarative config for one RTI recipe run.

    Mirrors the proven ``V2Config`` knobs from
    ``scripts/render_train_rl_cpu.py`` plus the backend/device selection.
    """

    # --- backend / device -------------------------------------------------
    backend: str = "cpu"        # registry key; "npu" can be added later
    device: str | None = None   # resolved via resolve_device() if None

    # --- output -----------------------------------------------------------
    out_dir: Path = Path("bench_results/recipe")

    # --- render -----------------------------------------------------------
    num_clips: int = 4
    frames_per_clip: int = 4
    height: int = 32
    width: int = 32
    seed: int = 42

    # --- train ------------------------------------------------------------
    encoder_embed_dim: int = 32
    encoder_depth: int = 2
    # W22-quant: HF encoder dimensions (effective only when
    # ``encoder_kind == "hf_vjepa2"``; the tiny path uses
    # ``encoder_embed_dim`` / ``encoder_depth`` instead). Defaults
    # match the W2.18 smoke (hidden_size=64, num_hidden_layers=2).
    encoder_hidden_size: int = 64
    encoder_num_layers: int = 2
    encoder_num_heads: int = 2
    num_steps: int = 30
    learning_rate: float = 3.0e-4
    ema_momentum: float = 0.99
    mask_ratio: float = 0.5
    encoder_kind: str = "tiny"   # "tiny" | "hf_vjepa2"
    # W22-quant: weight-only int8 on the HF encoder base. Only takes
    # effect when ``encoder_kind == "hf_vjepa2"``; the tiny stack has
    # no torchao integration. ``"none"`` keeps the HF base in fp32.
    # ``"w8_only"`` applies ``Int8WeightOnlyConfig`` BEFORE LoRA wrap
    # (QLoRA order). Mirrors ``scripts/render_train_rl_cpu.py``.
    quant: str = "none"

    # --- env / data -------------------------------------------------------
    env_type: str = "mock"       # "mock" | "franka_kitchen"
    franka_task_name: str = "lift_brick_features"
    data_source: str = "synthetic"  # "synthetic" | "franka_kitchen"
    camera_id: int | str = 0

    # --- policy / PPO -----------------------------------------------------
    action_dim: int = 4
    obs_dim: int = 8
    policy_encoder: str = "state_mlp"
    rollout_horizon: int = 1
    ppo_env_steps: int = 16
    num_ppo_epochs: int = 4
    num_outer_iters: int = 2
    num_train_per_iter: int = 10
    log_every_n: int = 10

    # --- staging control --------------------------------------------------
    # Which stages to run. A stage not in this list is skipped, so callers
    # can e.g. resume from a checkpoint or run inference-only.
    stages: tuple[str, ...] = ("render", "train", "inference", "policy", "outer_loop")
    # W25-cleanup: mirror of v2's ``V2Config.skip_render``. When True,
    # ``RenderStage`` short-circuits and reuses an existing
    # ``<out_dir>/manifest.csv`` instead of regenerating clips
    # (SystemExit rc=2 if missing). Lets test harnesses skip the
    # cv2.VideoWriter cost while still exercising the train/inference/
    # policy/outer_loop chain.
    skip_render: bool = False

    def resolve_device(self) -> str:
        """Return the concrete device string for this run."""
        return resolve_device(override=self.device)
