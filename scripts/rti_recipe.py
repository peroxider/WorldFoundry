#!/usr/bin/env python3
"""Decoupled RTI recipe CLI: render -> train -> inference -> policy (RLinf-shape).

A stage-based, backend-swappable runner over the CPU-verified RTI loop.
The device is resolved via ``WORLDFORGE_DEVICE`` (env) / NPU auto-detect,
so the same recipe runs on ``cpu`` now and ``npu`` later without code change.

Examples:
    # Default (mock env, synthetic data) on CPU:
    python scripts/rti_recipe.py

    # Real MuJoCo 9-DoF manipulation env (dm_control) + real RGB data:
    python scripts/rti_recipe.py --env franka_kitchen --data-source franka_kitchen

    # REAL end-to-end recipe (1.2B V-JEPA2 + real Franka data + NPU):
    #   Atlas box (container) with the 1.2B weights snapshot local:
    #   export MUJOCO_GL=egl   # headless GL (Mesa EGL); missing -> MjrContext error
    #   python scripts/rti_recipe.py \\
    #       --device npu \\
    #       --encoder hf_vjepa2 \\
    #       --encoder-pretrained /data/hf_cache/models--facebook--vjepa2-vitl-fpc64-256/snapshots/<sha> \\
    #       --env franka_kitchen --data-source franka_kitchen \\
    #       --franka-task-name lift_brick_features \\
    #       --action-dim 9 --obs-dim 6 \\
    #       --num-clips 3 --frames-per-clip 4 --num-steps 6 \\
    #       --num-outer-iters 1 --num-train-per-iter 2 --num-ppo-epochs 1 \\
    #       --out-dir bench_results/recipe_atlas_real_e2e
    #   Verified on Atlas NPU card 7 (2026-08-19): final_total=950.3947
    #   (matches w25). See bench_results/w46_atlas_real_e2e/summary.md.

    # Pin the device explicitly (WORLDFORGE_DEVICE honored too):
    python scripts/rti_recipe.py --device npu

    # Run a subset of stages (e.g. resume inference from an existing ckpt):
    python scripts/rti_recipe.py --stages inference

Exit codes:
    0  all stages complete + summary JSON written
    2  usage / config error
    1  a stage failed
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import worldfoundry_train  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from worldfoundry_train.recipe.config import RecipeConfig
from worldfoundry_train.recipe.pipeline import run_recipe

_DEFAULT_STAGES = ("render", "train", "inference", "policy", "outer_loop")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # backend / device
    p.add_argument("--backend", default="cpu",
                   help="backend registry key (default cpu; npu pluggable later)")
    p.add_argument("--device", default=None,
                   help="device string (default: WORLDFORGE_DEVICE env / auto)")
    # output
    p.add_argument("--out-dir", type=Path, default=Path("bench_results/recipe"))
    # render
    p.add_argument("--num-clips", type=int, default=4)
    p.add_argument("--frames-per-clip", type=int, default=4)
    p.add_argument("--height", type=int, default=32)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    # train
    p.add_argument("--encoder-embed-dim", type=int, default=32)
    p.add_argument("--encoder-depth", type=int, default=2)
    p.add_argument("--encoder-hidden-size", type=int, default=64,
                   help="W22-quant: HF encoder hidden_size (effective only "
                        "when --encoder hf_vjepa2). Default 64 = W2.18 "
                        "tiny smoke. 1024 = V-JEPA 2 Large (~326M params).")
    p.add_argument("--encoder-num-layers", type=int, default=2,
                   help="W22-quant: HF encoder num_hidden_layers (effective "
                        "only when --encoder hf_vjepa2). Default 2. 24 = "
                        "V-JEPA 2 Large.")
    p.add_argument("--encoder-num-heads", type=int, default=2,
                   help="W22-quant: HF encoder num_attention_heads "
                        "(effective only when --encoder hf_vjepa2). "
                        "Must divide --encoder-hidden-size. Default 2. "
                        "16 = V-JEPA 2 Large.")
    p.add_argument("--quant", choices=("none", "w8_only"), default="none",
                   help="W22-quant: weight-only int8 on the HF encoder base. "
                        "``none`` (default) keeps fp32. ``w8_only`` applies "
                        "torchao.Int8WeightOnlyConfig BEFORE LoRA wrap "
                        "(QLoRA order). Requires --encoder hf_vjepa2; "
                        "raises SystemExit otherwise.")
    p.add_argument("--num-steps", type=int, default=30)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--mask-ratio", type=float, default=0.5)
    p.add_argument("--encoder", choices=("tiny", "hf_vjepa2"), default="tiny")
    p.add_argument(
        "--encoder-pretrained",
        default=None,
        help="Local HF snapshot / repo id for real weights when --encoder "
             "hf_vjepa2 (e.g. /data/hf_cache/models--facebook--vjepa2-"
             "vitl-fpc64-256/snapshots/<sha>). None -> HF-hub default.",
    )
    p.add_argument(
        "--hf-tubelet-size", type=int, default=None,
        help="HF encoder tubelet size (effective only when --encoder "
             "hf_vjepa2). Real 1.2B vjepa2-vitl-fpc64-256 uses 2; None -> "
             "tubelet_size=1 (W2.18 default).",
    )
    # env / data
    p.add_argument(
        "--env",
        choices=("mock", "mujoco", "franka_kitchen", "droid", "manipulator"),
        default="mock",
        help="Env for the policy phase. ``mock`` (default) is the in-process "
             "random agent; ``franka_kitchen`` / ``mujoco`` (alias) use a "
             "real MuJoCo 9-DoF manipulation env via dm_control.manipulation; "
             "``manipulator`` uses the dm_control 5-DoF bring_ball env "
             "(flat obs 27-dim, action 5-dim); ``droid`` is offline dataset "
             "replay only and cannot drive the live policy loop.",
    )
    p.add_argument("--franka-task-name", default="lift_brick_features")
    p.add_argument("--data-source", choices=("synthetic", "franka_kitchen"),
                   default="synthetic")
    p.add_argument("--camera-id", default=0)
    # policy / PPO
    p.add_argument("--action-dim", type=int, default=4)
    p.add_argument("--obs-dim", type=int, default=8)
    p.add_argument("--policy-encoder", choices=("state_mlp", "tiny_vjepa2"),
                   default="state_mlp")
    p.add_argument("--rollout-horizon", type=int, default=1)
    p.add_argument("--ppo-env-steps", type=int, default=16)
    p.add_argument("--num-ppo-epochs", type=int, default=4)
    p.add_argument("--num-outer-iters", type=int, default=2)
    p.add_argument("--num-train-per-iter", type=int, default=10)
    # staging control
    p.add_argument("--skip-render", action="store_true",
                   help="W25-cleanup: reuse existing <out-dir>/manifest.csv "
                        "instead of regenerating clips. Raises rc=2 "
                        "if missing. Mirrors v2 --skip-render.")
    p.add_argument("--stages", nargs="*", default=list(_DEFAULT_STAGES),
                   help=f"stages to run (default: {' '.join(_DEFAULT_STAGES)})")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    # franka env overrides obs/action dims (mirrors render_train_rl_cpu.py).
    if args.env in ("franka_kitchen", "mujoco"):
        obs_dim, action_dim = 6, 9
    elif args.env == "manipulator":
        # Flat 27-dim obs (5+5+5+3+3+3+3) / 5-DoF action — mirrors
        # render_train_rl_cpu.py's _ManipulatorAdapter.FLAT_OBS_DIM/ACTION_DIM.
        obs_dim, action_dim = 27, 5
    elif args.env == "droid":
        print(
            "ERROR: --env=droid is offline dataset replay only (no live "
            "reset/step env); it cannot drive the policy phase. Use "
            "--env=mock / franka_kitchen / mujoco / manipulator.",
            file=sys.stderr,
        )
        return 2
    else:
        obs_dim, action_dim = args.obs_dim, args.action_dim
    if args.data_source == "franka_kitchen" and args.env not in (
        "franka_kitchen", "mujoco",
    ):
        print(
            f"ERROR: --data-source=franka_kitchen requires --env=franka_kitchen "
            f"or --env=mujoco (env={args.env})",
            file=sys.stderr,
        )
        return 2
    # W22-quant cross-validation: ``--quant`` only takes effect on the
    # HF encoder path. The tiny stack has no torchao integration, so
    # combining them is a programmer error.
    if args.quant != "none" and args.encoder != "hf_vjepa2":
        print(
            f"ERROR: --quant={args.quant} requires --encoder=hf_vjepa2 "
            f"(got --encoder={args.encoder}); the tiny encoder has no "
            f"torchao integration.",
            file=sys.stderr,
        )
        return 2

    cfg = RecipeConfig(
        backend=args.backend, device=args.device, out_dir=args.out_dir,
        num_clips=args.num_clips, frames_per_clip=args.frames_per_clip,
        height=args.height, width=args.width, seed=args.seed,
        encoder_embed_dim=args.encoder_embed_dim, encoder_depth=args.encoder_depth,
        # W22-quant: HF encoder dimensions (effective only when
        # ``encoder == "hf_vjepa2"``).
        encoder_hidden_size=args.encoder_hidden_size,
        encoder_num_layers=args.encoder_num_layers,
        encoder_num_heads=args.encoder_num_heads,
        num_steps=args.num_steps, learning_rate=args.learning_rate,
        mask_ratio=args.mask_ratio, encoder_kind=args.encoder,
        encoder_pretrained=args.encoder_pretrained,
        hf_tubelet_size=args.hf_tubelet_size,
        # W22-quant: weight-only int8 on the HF encoder base.
        quant=args.quant,
        env_type=args.env, franka_task_name=args.franka_task_name,
        data_source=args.data_source, camera_id=args.camera_id,
        action_dim=action_dim, obs_dim=obs_dim,
        policy_encoder=args.policy_encoder, rollout_horizon=args.rollout_horizon,
        ppo_env_steps=args.ppo_env_steps, num_ppo_epochs=args.num_ppo_epochs,
        num_outer_iters=args.num_outer_iters, num_train_per_iter=args.num_train_per_iter,
        stages=tuple(args.stages),
        # W25-cleanup: --skip-render short-circuit flag (see RecipeConfig).
        skip_render=args.skip_render,
    )
    run_recipe(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
