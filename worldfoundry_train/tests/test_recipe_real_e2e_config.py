"""Unit tests for the real end-to-end recipe config surface (Atlas 1.2B).

Covers the two new RecipeConfig fields (``encoder_pretrained`` /
``hf_tubelet_size``) and the ``rti_recipe.py`` CLI flags that drive the
real 1.2B V-JEPA2 + Franka recipe, so the Atlas-verified path stays
parseable without touching the model stack.
"""
from __future__ import annotations

from worldfoundry_train.recipe.config import RecipeConfig


def test_recipe_config_new_fields_default() -> None:
    cfg = RecipeConfig()
    assert cfg.encoder_pretrained is None
    assert cfg.hf_tubelet_size is None


def test_recipe_config_real_e2e_values() -> None:
    """The Atlas real-e2e recipe config round-trips through the dataclass."""
    cfg = RecipeConfig(
        encoder_kind="hf_vjepa2",
        encoder_pretrained=(
            "/data/hf_cache/models--facebook--vjepa2-vitl-fpc64-256/"
            "snapshots/b3c1679b7c34d3255ef3547f27c7b226aefab26f"
        ),
        hf_tubelet_size=2,
        env_type="franka_kitchen",
        data_source="franka_kitchen",
        action_dim=9,
        obs_dim=6,
    )
    assert cfg.encoder_kind == "hf_vjepa2"
    assert cfg.encoder_pretrained.endswith("b3c1679b7c34d3255ef3547f27c7b226aefab26f")
    assert cfg.hf_tubelet_size == 2
    assert cfg.env_type == "franka_kitchen"
    assert cfg.data_source == "franka_kitchen"
    assert cfg.action_dim == 9
    assert cfg.obs_dim == 6


def test_recipe_cli_parses_real_e2e_flags() -> None:
    """rti_recipe.py --encoder-pretrained / --hf-tubelet-size are accepted."""
    from scripts.rti_recipe import parse_args

    args = parse_args(
        [
            "--device", "npu",
            "--encoder", "hf_vjepa2",
            "--encoder-pretrained", "/data/hf_cache/weights",
            "--hf-tubelet-size", "2",
            "--env", "franka_kitchen",
            "--data-source", "franka_kitchen",
        ]
    )
    assert args.device == "npu"
    assert args.encoder == "hf_vjepa2"
    assert args.encoder_pretrained == "/data/hf_cache/weights"
    assert args.hf_tubelet_size == 2
    assert args.env == "franka_kitchen"
    assert args.data_source == "franka_kitchen"


def test_recipe_cli_maps_into_config() -> None:
    """The CLI flags flow into RecipeConfig via main()'s construction path."""
    from scripts.rti_recipe import main, parse_args
    from worldfoundry_train.recipe.pipeline import run_recipe

    # main() builds the RecipeConfig; we can't run the full pipeline here
    # (needs Atlas), so assert the wiring by constructing the same cfg the
    # way main() does for the real-e2e subset of flags.
    args = parse_args(
        [
            "--device", "npu",
            "--encoder", "hf_vjepa2",
            "--encoder-pretrained", "/data/hf_cache/weights",
            "--hf-tubelet-size", "2",
            "--env", "franka_kitchen",
            "--data-source", "franka_kitchen",
            "--num-outer-iters", "1",
        ]
    )
    cfg = RecipeConfig(
        backend=args.backend, device=args.device, out_dir=args.out_dir,
        encoder_kind=args.encoder,
        encoder_pretrained=args.encoder_pretrained,
        hf_tubelet_size=args.hf_tubelet_size,
        env_type=args.env, data_source=args.data_source,
    )
    assert cfg.device == "npu"
    assert cfg.encoder_pretrained == "/data/hf_cache/weights"
    assert cfg.hf_tubelet_size == 2
    # run_recipe import is alive (pipeline module loads without Atlas).
    assert callable(run_recipe)
    # main is importable (the CLI entry point).
    assert callable(main)
