"""HF VJEPA2Model shim tests (W2.18).

Verifies the seam: the shim exposes the same
``(pixel_values_videos).last_hidden_state`` contract as
:class:`worldfoundry_train.model.tiny_jepa.TinyVJEPA2Encoder` so the
CPU orchestrator's ``_build_stack`` chokepoint can swap encoders without
touching any downstream phase.

Tests:

- ``test_shim_config_from_vjepa2_config_round_trip`` -- shim config
  mirrors ``VJEPA2Config`` field-for-field.
- ``test_shim_forward_matches_tiny_encoder_shape`` -- the shim's
  ``last_hidden_state`` shape is byte-shape compatible with
  ``TinyVJEPA2Encoder``'s output for the same input.
- ``test_shim_lora_trainable_count_positive_and_subset`` -- LoRA
  produces a positive-but-small trainable subset of the base params.
- ``test_shim_predictor_accepts_action_kwarg_as_noop`` -- the predictor
  shim ignores ``action`` so the orchestrator's uniform call site works.

Tests 1-3 are gated on ``pytest.importorskip("transformers")`` because
the ``from_random_init`` factory requires ``transformers>=5.1.0``. Test
4 exercises only the predictor shim (which only depends on the local
``JEPAPredictor``), so it can run without ``transformers``.
"""
from __future__ import annotations

import pytest
import torch

from worldfoundry_train.model.tiny_jepa import TinyEncoderConfig, TinyVJEPA2Encoder


# ---------------------------------------------------------------------------
# Test 1: shim config mirrors VJEPA2Config field-for-field
# ---------------------------------------------------------------------------

def test_shim_config_from_vjepa2_config_round_trip():
    """Every geometry field on VJEPA2Config survives the shim mapping."""
    transformers = pytest.importorskip("transformers")
    from worldfoundry_train.model.hf_vjepa_shim import HfEncoderShimConfig

    cfg = transformers.VJEPA2Config(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
        crop_size=32, patch_size=8, tubelet_size=2,
        frames_per_clip=4, in_chans=3, drop_path_rate=0.0,
    )
    shim_cfg = HfEncoderShimConfig.from_vjepa2(cfg)
    assert shim_cfg.image_size == 32
    assert shim_cfg.patch_size == 8
    assert shim_cfg.in_channels == 3
    assert shim_cfg.embed_dim == 64
    assert shim_cfg.num_heads == 2
    assert shim_cfg.depth == 2
    assert shim_cfg.tubelet_size == 2
    assert shim_cfg.frames_per_clip == 4
    # Grid geometry is derivable from image_size / patch_size.
    assert shim_cfg.grid_h == 4
    assert shim_cfg.grid_w == 4


# ---------------------------------------------------------------------------
# Test 2: shim forward produces the same output shape as the tiny encoder
# ---------------------------------------------------------------------------

def test_shim_forward_matches_tiny_encoder_shape():
    """Both encoders emit ``(B, T*Hg*Wg, embed_dim)`` for the same input.

    The orchestrator's downstream consumer is
    ``online(pixel_values_videos).last_hidden_state`` -- the only
    requirement is that the shim returns the same shape as the tiny
    encoder. Values differ (random init vs. tiny init); we pin shape,
    not values.
    """
    pytest.importorskip("transformers")
    from worldfoundry_train.model.hf_vjepa_shim import HfVJEPA2EncoderShim

    # Tiny encoder side: same embed_dim=64 so shapes are directly comparable.
    tiny_cfg = TinyEncoderConfig(
        image_size=32, patch_size=8, in_channels=3,
        embed_dim=64, depth=2, num_heads=2,
    )
    tiny = TinyVJEPA2Encoder(tiny_cfg).eval()

    shim = HfVJEPA2EncoderShim.from_random_init(
        image_size=32, patch_size=8, in_channels=3,
        hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
        tubelet_size=1, frames_per_clip=1,
    ).eval()

    # Both want (B, T, C, H, W). HF VJEPA2Model with frames_per_clip=1
    # wants a single-frame input; tiny encoder supports the same.
    x = torch.zeros(1, 1, 3, 32, 32)
    with torch.no_grad():
        tiny_out = tiny(pixel_values_videos=x).last_hidden_state
        shim_out = shim(pixel_values_videos=x).last_hidden_state

    # Tiny: (1, T*Hg*Wg, 64) = (1, 16, 64)
    # Shim: same shape (Hg*Wg tokens, hidden_size=64).
    assert tiny_out.shape == (1, 16, 64)
    assert shim_out.shape == tiny_out.shape


# ---------------------------------------------------------------------------
# Test 3: LoRA produces a positive-but-small trainable subset
# ---------------------------------------------------------------------------

def test_shim_lora_trainable_count_positive_and_subset():
    """LoRA wraps the base so trainable > 0 and << total.

    Regression guard for the G4 invariant (silently-zero training).
    """
    pytest.importorskip("transformers")
    from worldfoundry_train.model.hf_vjepa_shim import HfVJEPA2EncoderShim

    shim = HfVJEPA2EncoderShim.from_random_init(
        image_size=32, patch_size=8, in_channels=3,
        hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
        tubelet_size=2, frames_per_clip=4, lora_r=4, lora_alpha=8,
    )
    trainable = sum(p.numel() for p in shim.parameters() if p.requires_grad)
    total = sum(p.numel() for p in shim.parameters())
    stats = shim.lora_stats()
    assert trainable > 0, (
        f"LoRA produced 0 trainable params (G4 violation). stats={stats}"
    )
    assert total > trainable, (
        f"LoRA trainable={trainable} >= total={total}; the backbone "
        f"should be frozen, not full-FT"
    )
    # LoRA stats captured at construction match a manual recount.
    assert stats["trainable"] == trainable
    assert stats["total"] == total


# ---------------------------------------------------------------------------
# Test 4: predictor shim accepts/ignores ``action`` kwarg
# ---------------------------------------------------------------------------

def test_shim_predictor_accepts_action_kwarg_as_noop():
    """``HfJEPAPredictorShim.forward(h, mask, action=...)`` works for
    ``action=None`` and ``action=<random tensor>``; both return the same
    shape. Pins the orchestrator's uniform ``pred(h, mask, action)`` call.
    """
    # No transformers import -- the predictor shim only depends on
    # the local JEPAPredictor. Keeps this test runnable on boxes
    # where ``transformers`` is not installed.
    from worldfoundry_train.model.hf_vjepa_shim import HfJEPAPredictorShim

    B, T, Hg, Wg = 1, 2, 4, 4
    N = T * Hg * Wg
    encoder_dim = 32
    pred = HfJEPAPredictorShim(
        num_patches=N, grid_size=(T, Hg, Wg), encoder_dim=encoder_dim,
        predictor_dim=16, depth=1, num_heads=2,
    )
    h = torch.randn(B, N, encoder_dim)
    mask = torch.zeros(B, N, dtype=torch.bool)
    mask[0, : N // 2] = True  # half the positions masked

    out_none = pred(h, mask, action=None)
    out_rand = pred(h, mask, action=torch.randn(B, 4))

    # Both return (M_total, encoder_dim) where M_total = sum(mask).
    assert out_none.shape == (B * (N // 2), encoder_dim)
    assert out_rand.shape == out_none.shape
    # action is intentionally ignored -- different actions => same output.
    assert torch.equal(out_none, out_rand)


# ---------------------------------------------------------------------------
# Test 5: signature() round-trips HF-specific fields
# ---------------------------------------------------------------------------

def test_shim_signature_round_trips_hf_specific_fields():
    """``HfEncoderShimConfig.signature()`` carries the seven HF-specific
    fields needed to rebuild the LoRA-wrapped VJEPA2Model from a saved
    checkpoint (W2.18 ``load_tiny_checkpoint`` HF branch).

    Tiny-path consumers ignore the extra keys; HF-path consumers
    rebuild ``HfVJEPA2EncoderShim.from_random_init`` from the dict.
    """
    pytest.importorskip("transformers")
    from worldfoundry_train.model.hf_vjepa_shim import HfVJEPA2EncoderShim

    shim = HfVJEPA2EncoderShim.from_random_init(
        image_size=32, patch_size=8, in_channels=3,
        hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
        tubelet_size=1, frames_per_clip=4, lora_r=4, lora_alpha=8,
        lora_dropout=0.05,
    )
    sig = shim.config.signature()

    # Common fields (also present on the tiny path).
    assert sig["image_size"] == 32
    assert sig["patch_size"] == 8
    assert sig["in_channels"] == 3
    assert sig["embed_dim"] == 64
    assert sig["depth"] == 2
    assert sig["num_heads"] == 2
    assert sig["encoder_kind"] == "hf_vjepa2"

    # HF-specific fields needed for ``from_random_init`` rebuild.
    assert sig["hidden_size"] == 64
    assert sig["num_attention_heads"] == 2
    assert sig["tubelet_size"] == 1
    assert sig["frames_per_clip"] == 4
    assert sig["lora_r"] == 4
    assert sig["lora_alpha"] == 8
    assert sig["lora_dropout"] == 0.05

    # Round-trip via ``from_random_init(**filter)``: rebuild a fresh shim
    # using only the HF fields and confirm param count matches.
    # Note: ``from_random_init`` uses ``num_hidden_layers``; the shim
    # config exposes it as ``depth`` so we map.
    hf_kwargs = {
        "image_size": sig["image_size"],
        "patch_size": sig["patch_size"],
        "in_channels": sig["in_channels"],
        "hidden_size": sig["hidden_size"],
        "num_hidden_layers": sig["depth"],
        "num_attention_heads": sig["num_attention_heads"],
        "tubelet_size": sig["tubelet_size"],
        "frames_per_clip": sig["frames_per_clip"],
        "lora_r": sig["lora_r"],
        "lora_alpha": sig["lora_alpha"],
        "lora_dropout": sig["lora_dropout"],
    }
    rebuilt = HfVJEPA2EncoderShim.from_random_init(**hf_kwargs)
    n_orig = sum(p.numel() for p in shim.parameters())
    n_rebuilt = sum(p.numel() for p in rebuilt.parameters())
    assert n_orig == n_rebuilt, (
        f"signature round-trip drifted: orig={n_orig} rebuilt={n_rebuilt}"
    )


# ---------------------------------------------------------------------------
# Test 6: random-init shim's signature carries ``pretrained_repo_id=None``
# ---------------------------------------------------------------------------

def test_shim_signature_from_random_init_carries_none_pretrained():
    """``from_random_init()`` returns a shim whose signature reports
    ``pretrained_repo_id=None`` and ``pretrained_revision=None``.

    Regression guard for W2.22: W2.18-era random-init checkpoints stay
    backward compatible with the new dispatch in
    ``load_tiny_checkpoint`` (when the field is missing / ``None`` the
    random-init path is taken).
    """
    pytest.importorskip("transformers")
    from worldfoundry_train.model.hf_vjepa_shim import HfVJEPA2EncoderShim

    shim = HfVJEPA2EncoderShim.from_random_init(
        image_size=32, patch_size=8, in_channels=3,
        hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
        tubelet_size=1, frames_per_clip=4,
    )
    sig = shim.config.signature()
    assert sig["pretrained_repo_id"] is None
    assert sig["pretrained_revision"] is None


# ---------------------------------------------------------------------------
# Test 7: ``from_pretrained`` with local fixture loads real weights
# ---------------------------------------------------------------------------

def test_shim_from_pretrained_loads_real_weights_from_fixture():
    """``from_pretrained(<local fixture path>)`` loads a real saved
    VJEPA2Model and produces outputs that differ from
    ``from_random_init`` with the same geometry.

    The fixture is a tiny HF VJEPA2Model saved via
    ``VJEPA2Model.save_pretrained`` to
    ``bench_results/fixtures/tiny_vjepa2/`` (created once via a one-off
    script — NOT regenerated every CI run). When
    ``WORLDFORGE_HF_FIXTURE`` env var is unset (default), this test
    skips cleanly so local CI doesn't pay the 80 MB fixture cost.

    The discrimination check (``|out_pre - out_rand| > 0.1``) proves
    that real weights were loaded instead of fresh random init.
    """
    pytest.importorskip("transformers")
    import os
    from pathlib import Path

    from worldfoundry_train.model.hf_vjepa_shim import HfVJEPA2EncoderShim

    fix_root = os.environ.get("WORLDFORGE_HF_FIXTURE")
    if not fix_root:
        pytest.skip(
            "WORLDFORGE_HF_FIXTURE env var not set; from_pretrained "
            "test requires the tiny_vjepa2 fixture (~85 MB saved "
            "VJEPA2Model at bench_results/fixtures/tiny_vjepa2/)."
        )
    fix_path = Path(fix_root)
    if not fix_path.is_dir():
        pytest.skip(f"WORLDFORGE_HF_FIXTURE points at non-dir: {fix_path}")
    if not (fix_path / "config.json").is_file():
        pytest.skip(f"Fixture missing config.json: {fix_path}")

    common_kwargs = dict(
        image_size=32, patch_size=8, in_channels=3,
        hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
        tubelet_size=1, frames_per_clip=4,
        lora_r=4, lora_alpha=8, lora_dropout=0.0,
    )
    pretrained = HfVJEPA2EncoderShim.from_pretrained(
        pretrained_repo_id=str(fix_path),
        **common_kwargs,
    )
    random_init = HfVJEPA2EncoderShim.from_random_init(**common_kwargs)

    # Both shim expose the same forward surface and accept the same
    # input shape; the LoRA-wrapped ``VJEPA2Model`` is the differentiator.
    assert pretrained.config.pretrained_repo_id == str(fix_path)
    sig = pretrained.config.signature()
    assert sig["pretrained_repo_id"] == str(fix_path)
    assert sig["pretrained_revision"] is None  # no revision pin for the fixture

    # Use a non-zero input so the patch embed doesn't collapse to its
    # bias (zeros-input produces constant token sequences, which all
    # normalise to the same LayerNorm output regardless of base
    # weights). The test cares about the BASE weights, not the input.
    x = torch.randn(1, 1, 3, 32, 32)
    with torch.no_grad():
        out_pre = pretrained(pixel_values_videos=x).last_hidden_state
        out_rand = random_init(pixel_values_videos=x).last_hidden_state

    # Real weights != random init: difference is large in absolute terms.
    diff = (out_pre - out_rand).abs().mean().item()
    assert diff > 0.1, (
        f"from_pretrained output matches random-init too closely "
        f"(mean |out_pre - out_rand| = {diff:.4f} < 0.1) -- weights "
        f"may not have loaded from the fixture."
    )
    # Sanity: both outputs are the same shape (smoke stays apples-to-apples).
    assert out_pre.shape == out_rand.shape